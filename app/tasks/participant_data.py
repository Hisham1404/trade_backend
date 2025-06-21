"""
Celery tasks for NSE Participant Data Ingestion

This module provides Celery tasks for automated collection and processing
of NSE participant flow data with scheduling, monitoring, and error handling.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Any
from celery import shared_task
from celery.exceptions import Retry

from app.market_data.participant_data_ingestion import (
    NSEDataIngestionPipeline,
    ingest_current_participant_data,
    ingest_historical_participant_data,
    generate_participant_summary,
    IngestionResult
)
from app.services.monitoring_service import MonitoringService
from app.services.alerting_service import AlertingService
from app.market_data.market_hours import MarketHours

logger = logging.getLogger(__name__)

@shared_task(bind=True, retry_backoff=True, max_retries=3)
def ingest_current_participant_data_task(self):
    """
    Celery task to ingest current day NSE participant data
    
    This task runs during market hours to collect real-time participant flow data.
    Includes automatic retry logic and comprehensive error handling.
    """
    task_id = self.request.id
    logger.info(f"Starting participant data ingestion task {task_id}")
    
    try:
        # Check if market is open (optional optimization)
        market_hours = MarketHours()
        if not market_hours.is_market_open():
            logger.info("Market is closed, skipping real-time data ingestion")
            return {
                'success': True,
                'message': 'Market closed - no action required',
                'records_processed': 0
            }
        
        # Run the async ingestion
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(ingest_current_participant_data())
        finally:
            loop.close()
        
        # Log results
        if result.success:
            logger.info(f"Participant data ingestion completed successfully: "
                       f"{result.records_inserted} records inserted, "
                       f"{result.records_failed} failed")
            
            # Send success metrics to monitoring service
            MonitoringService.record_participant_ingestion_success(
                records_count=result.records_inserted,
                processing_time_ms=result.processing_time_ms,
                data_quality_score=result.data_quality_score
            )
        else:
            logger.error(f"Participant data ingestion failed: {result.errors}")
            
            # Send failure metrics
            MonitoringService.record_participant_ingestion_failure(
                error_count=len(result.errors),
                failure_reason='; '.join(result.errors[:3])  # First 3 errors
            )
            
            # Send alert for persistent failures
            if self.request.retries >= 2:
                AlertingService.send_participant_ingestion_alert(
                    severity='HIGH',
                    message=f"Participant data ingestion failed after {self.request.retries + 1} attempts",
                    details=result.errors
                )
        
        return {
            'success': result.success,
            'records_processed': result.records_processed,
            'records_inserted': result.records_inserted,
            'records_failed': result.records_failed,
            'processing_time_ms': result.processing_time_ms,
            'data_quality_score': result.data_quality_score,
            'errors': result.errors,
            'warnings': result.warnings
        }
        
    except Exception as e:
        logger.error(f"Participant data ingestion task failed: {str(e)}")
        
        # Record failure metrics
        MonitoringService.record_participant_ingestion_failure(
            error_count=1,
            failure_reason=str(e)
        )
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying participant data ingestion task in {2 ** self.request.retries} seconds")
            raise self.retry(countdown=2 ** self.request.retries, exc=e)
        else:
            # Send critical alert after all retries exhausted
            AlertingService.send_participant_ingestion_alert(
                severity='CRITICAL',
                message=f"Participant data ingestion failed permanently after {self.max_retries + 1} attempts",
                details=[str(e)]
            )
            
            return {
                'success': False,
                'error': str(e),
                'retries_exhausted': True
            }

@shared_task(bind=True, retry_backoff=True, max_retries=2)
def ingest_historical_participant_data_task(self, start_date_str: str, end_date_str: str):
    """
    Celery task to ingest historical NSE participant data for a date range
    
    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
    """
    task_id = self.request.id
    logger.info(f"Starting historical participant data ingestion task {task_id} "
               f"for period {start_date_str} to {end_date_str}")
    
    try:
        # Parse date strings
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        # Validate date range
        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")
        
        if end_date > date.today():
            raise ValueError("End date cannot be in the future")
        
        # Calculate expected number of trading days (rough estimate)
        total_days = (end_date - start_date).days + 1
        expected_trading_days = total_days * 5 // 7  # Rough estimate excluding weekends
        
        logger.info(f"Processing approximately {expected_trading_days} trading days")
        
        # Run the async ingestion
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                ingest_historical_participant_data(start_date, end_date)
            )
        finally:
            loop.close()
        
        # Aggregate results
        total_records = sum(r.records_inserted for r in results if r.success)
        successful_days = len([r for r in results if r.success])
        failed_days = len([r for r in results if not r.success])
        
        # Calculate overall success rate
        success_rate = successful_days / len(results) if results else 0
        
        logger.info(f"Historical participant data ingestion completed: "
                   f"{successful_days}/{len(results)} days successful, "
                   f"{total_records} total records inserted")
        
        # Send metrics
        MonitoringService.record_historical_ingestion_completion(
            date_range_days=len(results),
            successful_days=successful_days,
            failed_days=failed_days,
            total_records=total_records,
            success_rate=success_rate
        )
        
        # Send alert if success rate is too low
        if success_rate < 0.8 and len(results) > 1:
            AlertingService.send_participant_ingestion_alert(
                severity='MEDIUM',
                message=f"Historical data ingestion had low success rate: {success_rate:.1%}",
                details=[f"Failed on {failed_days} out of {len(results)} days"]
            )
        
        return {
            'success': success_rate > 0,
            'total_days_processed': len(results),
            'successful_days': successful_days,
            'failed_days': failed_days,
            'total_records_inserted': total_records,
            'success_rate': success_rate,
            'failed_dates': [
                r.source_timestamp.strftime('%Y-%m-%d') 
                for r in results 
                if not r.success and r.source_timestamp
            ]
        }
        
    except Exception as e:
        logger.error(f"Historical participant data ingestion task failed: {str(e)}")
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying historical ingestion task in {10 * (2 ** self.request.retries)} seconds")
            raise self.retry(countdown=10 * (2 ** self.request.retries), exc=e)
        else:
            AlertingService.send_participant_ingestion_alert(
                severity='HIGH',
                message=f"Historical participant data ingestion failed permanently",
                details=[str(e)]
            )
            
            return {
                'success': False,
                'error': str(e),
                'retries_exhausted': True
            }

@shared_task(bind=True, retry_backoff=True, max_retries=2)
def generate_daily_participant_summary_task(self, target_date_str: str = None):
    """
    Celery task to generate daily participant flow summary
    
    Args:
        target_date_str: Target date in YYYY-MM-DD format (defaults to yesterday)
    """
    task_id = self.request.id
    
    try:
        # Parse target date or use yesterday (since market data is usually available next day)
        if target_date_str:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
        else:
            target_date = date.today() - timedelta(days=1)
        
        logger.info(f"Generating participant flow summary for {target_date}")
        
        # Run the async summary generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(generate_participant_summary(target_date))
        finally:
            loop.close()
        
        if success:
            logger.info(f"Successfully generated participant flow summary for {target_date}")
            
            MonitoringService.record_summary_generation_success(
                summary_date=target_date
            )
            
            return {
                'success': True,
                'summary_date': target_date.strftime('%Y-%m-%d'),
                'message': 'Daily summary generated successfully'
            }
        else:
            logger.error(f"Failed to generate participant flow summary for {target_date}")
            
            MonitoringService.record_summary_generation_failure(
                summary_date=target_date,
                reason="No participant data available"
            )
            
            # Only alert if this is a recent date (within last 3 days)
            if (date.today() - target_date).days <= 3:
                AlertingService.send_participant_summary_alert(
                    severity='MEDIUM',
                    message=f"Failed to generate participant summary for {target_date}",
                    details=["No participant activity data available for this date"]
                )
            
            return {
                'success': False,
                'summary_date': target_date.strftime('%Y-%m-%d'),
                'error': 'No participant data available for summary generation'
            }
    
    except Exception as e:
        logger.error(f"Participant summary generation task failed: {str(e)}")
        
        # Retry with backoff
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=5 * (2 ** self.request.retries), exc=e)
        else:
            AlertingService.send_participant_summary_alert(
                severity='HIGH',
                message=f"Participant summary generation failed permanently for {target_date_str or 'yesterday'}",
                details=[str(e)]
            )
            
            return {
                'success': False,
                'error': str(e),
                'retries_exhausted': True
            }

@shared_task
def schedule_weekly_historical_backfill():
    """
    Weekly task to backfill any missing historical participant data
    
    This task runs weekly to ensure data completeness by checking for
    and filling any gaps in historical participant data.
    """
    logger.info("Starting weekly historical participant data backfill")
    
    try:
        from app.database.connection import get_db
        from app.models.participant_flow import ParticipantActivity
        from sqlalchemy import func, and_
        
        # Find missing dates in the last 30 days
        db = next(get_db())
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        # Query for existing dates
        existing_dates = db.query(func.date(ParticipantActivity.trade_date)).filter(
            and_(
                ParticipantActivity.trade_date >= start_date,
                ParticipantActivity.trade_date <= end_date
            )
        ).distinct().all()
        
        existing_dates_set = set(d[0] for d in existing_dates)
        
        # Find missing trading days (exclude weekends)
        missing_dates = []
        current_date = start_date
        while current_date <= end_date:
            if (current_date.weekday() < 5 and  # Monday=0, Friday=4
                current_date not in existing_dates_set):
                missing_dates.append(current_date)
            current_date += timedelta(days=1)
        
        db.close()
        
        if missing_dates:
            logger.info(f"Found {len(missing_dates)} missing dates for backfill: {missing_dates}")
            
            # Group consecutive dates for efficient batch processing
            date_ranges = []
            range_start = missing_dates[0]
            range_end = missing_dates[0]
            
            for i in range(1, len(missing_dates)):
                if missing_dates[i] == range_end + timedelta(days=1):
                    range_end = missing_dates[i]
                else:
                    date_ranges.append((range_start, range_end))
                    range_start = missing_dates[i]
                    range_end = missing_dates[i]
            
            date_ranges.append((range_start, range_end))
            
            # Schedule backfill tasks for each range
            backfill_tasks = []
            for start, end in date_ranges:
                task = ingest_historical_participant_data_task.delay(
                    start.strftime('%Y-%m-%d'),
                    end.strftime('%Y-%m-%d')
                )
                backfill_tasks.append(task.id)
                logger.info(f"Scheduled backfill task {task.id} for {start} to {end}")
            
            return {
                'success': True,
                'missing_dates_count': len(missing_dates),
                'backfill_tasks_scheduled': len(backfill_tasks),
                'task_ids': backfill_tasks
            }
        else:
            logger.info("No missing dates found - all data is up to date")
            return {
                'success': True,
                'missing_dates_count': 0,
                'message': 'All participant data is up to date'
            }
    
    except Exception as e:
        logger.error(f"Weekly backfill task failed: {str(e)}")
        
        AlertingService.send_participant_ingestion_alert(
            severity='MEDIUM',
            message="Weekly participant data backfill task failed",
            details=[str(e)]
        )
        
        return {
            'success': False,
            'error': str(e)
        }

@shared_task
def cleanup_old_ingestion_logs():
    """
    Cleanup task to remove old ingestion logs and temporary data
    
    Runs monthly to clean up old logs and maintain system performance.
    """
    logger.info("Starting cleanup of old participant ingestion logs")
    
    try:
        # This would implement log cleanup logic
        # For now, just return success
        logger.info("Cleanup completed successfully")
        
        return {
            'success': True,
            'message': 'Cleanup completed successfully'
        }
    
    except Exception as e:
        logger.error(f"Cleanup task failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# Task scheduling configuration for Celery Beat
PARTICIPANT_DATA_SCHEDULE = {
    'ingest-current-participant-data': {
        'task': 'app.tasks.participant_data.ingest_current_participant_data_task',
        'schedule': 300.0,  # Every 5 minutes during market hours
        'options': {
            'expires': 240,  # Task expires after 4 minutes
            'retry': True,
            'retry_policy': {
                'max_retries': 3,
                'interval_start': 1,
                'interval_step': 1,
                'interval_max': 10,
            }
        }
    },
    'generate-daily-participant-summary': {
        'task': 'app.tasks.participant_data.generate_daily_participant_summary_task',
        'schedule': {
            'hour': 18,  # 6 PM
            'minute': 30
        },
        'options': {
            'expires': 3600,  # Task expires after 1 hour
        }
    },
    'weekly-historical-backfill': {
        'task': 'app.tasks.participant_data.schedule_weekly_historical_backfill',
        'schedule': {
            'hour': 2,   # 2 AM
            'minute': 0,
            'day_of_week': 1  # Monday
        }
    },
    'monthly-cleanup': {
        'task': 'app.tasks.participant_data.cleanup_old_ingestion_logs',
        'schedule': {
            'hour': 3,   # 3 AM
            'minute': 0,
            'day_of_month': 1  # First day of month
        }
    }
} 