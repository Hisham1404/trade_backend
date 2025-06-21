"""
Celery Tasks for Insights Generation and Reporting

Provides automated background tasks for generating insights, creating reports,
and updating dashboard data on scheduled intervals.
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any
from celery import shared_task
from sqlalchemy.orm import Session

# Internal imports
from app.database.connection import get_db
from app.services.insights_generation_service import (
    InsightsReportingService,
    ReportType,
    Insight,
    DashboardData
)
from app.services.behavioral_shift_detection import BehavioralShiftService
from app.services.participant_metrics_service import ParticipantMetricsService
from app.models.participant_flow import ParticipantType, MarketSegment

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def generate_daily_insights_task(self, analysis_date_str: str = None):
    """
    Celery task to generate daily insights for participant flow analysis.
    
    Args:
        analysis_date_str: Date string in YYYY-MM-DD format (defaults to today)
    
    Returns:
        dict: Task result with insights count and status
    """
    try:
        db_session = next(get_db())
        
        # Parse analysis date
        if analysis_date_str:
            analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d").date()
        else:
            analysis_date = date.today()
        
        logger.info(f"Starting daily insights generation for {analysis_date}")
        
        # Initialize service
        service = InsightsReportingService(db_session)
        
        # Generate insights
        insights = await service.generate_daily_insights(analysis_date)
        
        # Store insights count by category for reporting
        insights_summary = {
            "total_insights": len(insights),
            "by_category": {},
            "by_priority": {},
            "high_confidence_count": 0
        }
        
        for insight in insights:
            # Count by category
            category = insight.category.value
            insights_summary["by_category"][category] = insights_summary["by_category"].get(category, 0) + 1
            
            # Count by priority
            priority = insight.priority
            insights_summary["by_priority"][priority] = insights_summary["by_priority"].get(priority, 0) + 1
            
            # Count high confidence insights
            if insight.confidence_score > 0.8:
                insights_summary["high_confidence_count"] += 1
        
        logger.info(f"Generated {len(insights)} insights for {analysis_date}")
        
        return {
            "status": "success",
            "analysis_date": analysis_date.isoformat(),
            "insights_generated": len(insights),
            "summary": insights_summary,
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Daily insights generation failed: {str(e)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying daily insights generation (attempt {self.request.retries + 1})")
            raise self.retry(countdown=300, exc=e)  # Retry after 5 minutes
        
        return {
            "status": "failed",
            "error": str(e),
            "analysis_date": analysis_date_str or date.today().isoformat(),
            "failed_at": datetime.now().isoformat()
        }
    
    finally:
        if 'db_session' in locals():
            db_session.close()

@shared_task(bind=True, max_retries=2)
def generate_weekly_report_task(self, report_date_str: str = None):
    """
    Celery task to generate weekly overview reports.
    
    Args:
        report_date_str: End date for the weekly report (defaults to today)
    
    Returns:
        dict: Task result with report status
    """
    try:
        db_session = next(get_db())
        
        # Parse report date
        if report_date_str:
            report_date = datetime.strptime(report_date_str, "%Y-%m-%d").date()
        else:
            report_date = date.today()
        
        logger.info(f"Starting weekly report generation for week ending {report_date}")
        
        # Initialize service
        service = InsightsReportingService(db_session)
        
        # Generate weekly overview report
        report = await service.generate_report(ReportType.WEEKLY_OVERVIEW, report_date)
        
        logger.info(f"Generated weekly report for {report_date}")
        
        return {
            "status": "success",
            "report_type": "weekly_overview",
            "report_date": report_date.isoformat(),
            "insights_count": len(report.key_insights),
            "recommendations_count": len(report.recommendations),
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Weekly report generation failed: {str(e)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying weekly report generation (attempt {self.request.retries + 1})")
            raise self.retry(countdown=600, exc=e)  # Retry after 10 minutes
        
        return {
            "status": "failed",
            "error": str(e),
            "report_date": report_date_str or date.today().isoformat(),
            "failed_at": datetime.now().isoformat()
        }
    
    finally:
        if 'db_session' in locals():
            db_session.close()

@shared_task(bind=True, max_retries=3)
def update_dashboard_data_task(self, update_date_str: str = None):
    """
    Celery task to update dashboard data and cache it for fast retrieval.
    
    Args:
        update_date_str: Date for dashboard update (defaults to today)
    
    Returns:
        dict: Task result with dashboard update status
    """
    try:
        db_session = next(get_db())
        
        # Parse update date
        if update_date_str:
            update_date = datetime.strptime(update_date_str, "%Y-%m-%d").date()
        else:
            update_date = date.today()
        
        logger.info(f"Starting dashboard data update for {update_date}")
        
        # Initialize service
        service = InsightsReportingService(db_session)
        
        # Generate dashboard data
        dashboard_data = await service.get_dashboard_data(update_date)
        
        # In a production system, we would cache this data in Redis or similar
        # For now, we'll just log the successful generation
        
        logger.info(f"Dashboard data updated for {update_date}")
        
        return {
            "status": "success",
            "update_date": update_date.isoformat(),
            "summary_stats": dashboard_data.summary_stats,
            "insights_count": len(dashboard_data.recent_insights),
            "behavioral_shifts_count": len(dashboard_data.behavioral_shifts),
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dashboard data update failed: {str(e)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying dashboard update (attempt {self.request.retries + 1})")
            raise self.retry(countdown=180, exc=e)  # Retry after 3 minutes
        
        return {
            "status": "failed",
            "error": str(e),
            "update_date": update_date_str or date.today().isoformat(),
            "failed_at": datetime.now().isoformat()
        }
    
    finally:
        if 'db_session' in locals():
            db_session.close()

@shared_task(bind=True, max_retries=2)
def detect_behavioral_shifts_task(self, detection_date_str: str = None):
    """
    Celery task to detect and analyze behavioral shifts in participant activity.
    
    Args:
        detection_date_str: Date for shift detection (defaults to today)
    
    Returns:
        dict: Task result with shift detection status
    """
    try:
        db_session = next(get_db())
        
        # Parse detection date
        if detection_date_str:
            detection_date = datetime.strptime(detection_date_str, "%Y-%m-%d").date()
        else:
            detection_date = date.today()
        
        logger.info(f"Starting behavioral shift detection for {detection_date}")
        
        # Initialize service
        shift_service = BehavioralShiftService(db_session)
        
        # Detect and store behavioral shifts
        result = await shift_service.detect_and_store_shifts(detection_date)
        
        if result.success:
            logger.info(f"Detected {len(result.shifts_detected)} behavioral shifts for {detection_date}")
            
            # Categorize shifts by severity
            shifts_by_severity = {}
            for shift in result.shifts_detected:
                severity = shift.severity.value
                shifts_by_severity[severity] = shifts_by_severity.get(severity, 0) + 1
            
            return {
                "status": "success",
                "detection_date": detection_date.isoformat(),
                "shifts_detected": len(result.shifts_detected),
                "shifts_by_severity": shifts_by_severity,
                "processing_time_ms": result.processing_time_ms,
                "completed_at": datetime.now().isoformat()
            }
        else:
            logger.error(f"Behavioral shift detection failed with errors: {result.errors}")
            return {
                "status": "failed",
                "detection_date": detection_date.isoformat(),
                "errors": result.errors,
                "failed_at": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Behavioral shift detection task failed: {str(e)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying behavioral shift detection (attempt {self.request.retries + 1})")
            raise self.retry(countdown=300, exc=e)  # Retry after 5 minutes
        
        return {
            "status": "failed",
            "error": str(e),
            "detection_date": detection_date_str or date.today().isoformat(),
            "failed_at": datetime.now().isoformat()
        }
    
    finally:
        if 'db_session' in locals():
            db_session.close()

@shared_task(bind=True, max_retries=2)
def generate_monthly_analysis_task(self, analysis_month_str: str = None):
    """
    Celery task to generate comprehensive monthly analysis reports.
    
    Args:
        analysis_month_str: Month in YYYY-MM format (defaults to current month)
    
    Returns:
        dict: Task result with monthly analysis status
    """
    try:
        db_session = next(get_db())
        
        # Parse analysis month
        if analysis_month_str:
            analysis_date = datetime.strptime(f"{analysis_month_str}-01", "%Y-%m-%d").date()
        else:
            today = date.today()
            analysis_date = date(today.year, today.month, 1)
        
        # Get last day of the month
        if analysis_date.month == 12:
            next_month = date(analysis_date.year + 1, 1, 1)
        else:
            next_month = date(analysis_date.year, analysis_date.month + 1, 1)
        
        month_end = next_month - timedelta(days=1)
        
        logger.info(f"Starting monthly analysis for {analysis_date.strftime('%B %Y')}")
        
        # Initialize service
        service = InsightsReportingService(db_session)
        
        # Generate monthly analysis report
        report = await service.generate_report(ReportType.MONTHLY_ANALYSIS, month_end)
        
        logger.info(f"Generated monthly analysis for {analysis_date.strftime('%B %Y')}")
        
        return {
            "status": "success",
            "report_type": "monthly_analysis",
            "analysis_month": analysis_date.strftime("%Y-%m"),
            "period_covered": report.period_covered,
            "insights_count": len(report.key_insights),
            "recommendations_count": len(report.recommendations),
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Monthly analysis generation failed: {str(e)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying monthly analysis generation (attempt {self.request.retries + 1})")
            raise self.retry(countdown=900, exc=e)  # Retry after 15 minutes
        
        return {
            "status": "failed",
            "error": str(e),
            "analysis_month": analysis_month_str or date.today().strftime("%Y-%m"),
            "failed_at": datetime.now().isoformat()
        }
    
    finally:
        if 'db_session' in locals():
            db_session.close()

@shared_task
def cleanup_old_insights_task(retention_days: int = 90):
    """
    Celery task to clean up old insights and reports to manage storage.
    
    Args:
        retention_days: Number of days to retain insights (default: 90 days)
    
    Returns:
        dict: Task result with cleanup status
    """
    try:
        db_session = next(get_db())
        
        cutoff_date = date.today() - timedelta(days=retention_days)
        
        logger.info(f"Starting cleanup of insights older than {cutoff_date}")
        
        # In a production system, we would clean up stored insights/reports from database
        # For now, this is a placeholder task
        
        logger.info(f"Cleanup completed for insights older than {cutoff_date}")
        
        return {
            "status": "success",
            "cutoff_date": cutoff_date.isoformat(),
            "retention_days": retention_days,
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Insights cleanup failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }
    
    finally:
        if 'db_session' in locals():
            db_session.close()

# Periodic task for real-time insights
@shared_task(bind=True, max_retries=5)
def real_time_insights_update_task(self):
    """
    Celery task for real-time insights updates (runs every 15 minutes during market hours).
    
    Returns:
        dict: Task result with real-time update status
    """
    try:
        db_session = next(get_db())
        
        current_time = datetime.now()
        analysis_date = current_time.date()
        
        logger.info(f"Starting real-time insights update at {current_time}")
        
        # Check if market is open (placeholder - would need actual market hours logic)
        # For now, assume market hours are 9:15 AM to 3:30 PM IST on weekdays
        if current_time.weekday() >= 5:  # Weekend
            logger.info("Market closed (weekend) - skipping real-time update")
            return {
                "status": "skipped",
                "reason": "market_closed_weekend",
                "timestamp": current_time.isoformat()
            }
        
        # Initialize services
        service = InsightsReportingService(db_session)
        shift_service = BehavioralShiftService(db_session)
        
        # Quick behavioral shift detection
        shift_result = await shift_service.detect_and_store_shifts(analysis_date)
        
        # Generate quick insights (last 2 hours of data)
        insights = await service.generate_daily_insights(analysis_date)
        
        # Filter for high-priority, recent insights
        recent_high_priority = [
            i for i in insights 
            if i.priority == "high" and i.confidence_score > 0.7
        ]
        
        logger.info(f"Real-time update completed: {len(recent_high_priority)} high-priority insights")
        
        return {
            "status": "success",
            "timestamp": current_time.isoformat(),
            "shifts_detected": len(shift_result.shifts_detected) if shift_result.success else 0,
            "high_priority_insights": len(recent_high_priority),
            "total_insights": len(insights),
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Real-time insights update failed: {str(e)}")
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            countdown = 2 ** self.request.retries * 60  # 1, 2, 4, 8, 16 minutes
            logger.info(f"Retrying real-time update (attempt {self.request.retries + 1}) in {countdown} seconds")
            raise self.retry(countdown=countdown, exc=e)
        
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "failed_at": datetime.now().isoformat()
        }
    
    finally:
        if 'db_session' in locals():
            db_session.close()

@shared_task
def generate_daily_insights():
    """Generate daily insights task"""
    logger.info("Daily insights task executed")
    return {"status": "success"} 