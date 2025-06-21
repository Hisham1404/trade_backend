"""
Celery Application Configuration
Production-ready setup with Redis broker, task routing, and monitoring
"""

import os
from celery import Celery
from celery.schedules import crontab
from kombu import Queue
from app.core.config import settings
from app.services.logging_service import get_logger

logger = get_logger(__name__)

# Create Celery application instance
celery_app = Celery(
    'trading_intelligence',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        'app.tasks.scraping',
        'app.tasks.analysis',
        'app.tasks.market_data',
        'app.tasks.alerts',
        'app.tasks.monitoring',
        'app.tasks.participant_data',
        'app.tasks.insights_tasks'
    ]
)

# Configure Celery with production-ready settings
celery_app.conf.update(
    # Task serialization
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_RESULT_SERIALIZER,
    accept_content=settings.CELERY_ACCEPT_CONTENT,
    
    # Timezone configuration
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=settings.CELERY_ENABLE_UTC,
    
    # Task execution settings
    task_track_started=settings.CELERY_TASK_TRACK_STARTED,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    
    # Worker configuration
    worker_prefetch_multiplier=settings.CELERY_WORKER_PREFETCH_MULTIPLIER,
    task_acks_late=settings.CELERY_TASK_ACKS_LATE,
    worker_send_task_events=settings.CELERY_WORKER_SEND_TASK_EVENTS,
    task_send_sent_event=settings.CELERY_TASK_SEND_SENT_EVENT,
    
    # Testing configuration
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
    
    # Result backend settings
    result_expires=settings.CELERY_TASK_RESULT_EXPIRES,
    
    # Redis connection settings with production optimizations
    broker_transport_options={
        'visibility_timeout': 3600,  # 1 hour for long-running tasks
        'fanout_prefix': True,
        'fanout_patterns': True,
        'socket_keepalive': True,
        'socket_keepalive_options': {
            'TCP_KEEPINTVL': 1,
            'TCP_KEEPCNT': 3,
            'TCP_KEEPIDLE': 1,
        },
        'retry_on_timeout': True,
        'health_check_interval': 30,
    },
    
    result_backend_transport_options={
        'retry_on_timeout': True,
        'socket_timeout': 300,
        'socket_connect_timeout': 30,
        'socket_keepalive': True,
        'socket_keepalive_options': {
            'TCP_KEEPINTVL': 1,
            'TCP_KEEPCNT': 3,
            'TCP_KEEPIDLE': 1,
        },
        'health_check_interval': 30,
    },
    
    # Connection pool configuration
    broker_pool_limit=settings.CELERY_BROKER_POOL_LIMIT,
    broker_connection_retry=settings.CELERY_BROKER_CONNECTION_RETRY,
    broker_connection_max_retries=settings.CELERY_BROKER_CONNECTION_MAX_RETRIES,
    result_backend_connection_retry=settings.CELERY_RESULT_BACKEND_CONNECTION_RETRY,
)

# Task routing configuration
celery_app.conf.task_routes = {
    # Scraping tasks - high frequency, moderate priority
    'app.tasks.scraping.scrape_high_priority': {'queue': 'high_priority'},
    'app.tasks.scraping.scrape_general_sources': {'queue': 'scraping'},
    'app.tasks.scraping.scrape_source': {'queue': 'scraping'},
    'app.tasks.scraping.validate_scraping_results': {'queue': 'scraping'},
    
    # Analysis tasks - CPU intensive, high priority
    'app.tasks.analysis.analyze_news_item': {'queue': 'analysis'},
    'app.tasks.analysis.process_unanalyzed_news': {'queue': 'analysis'},
    'app.tasks.analysis.run_sentiment_analysis': {'queue': 'analysis'},
    'app.tasks.analysis.update_market_correlation': {'queue': 'analysis'},
    
    # Market data tasks - time sensitive, highest priority
    'app.tasks.market_data.update_option_chains': {'queue': 'market_data'},
    'app.tasks.market_data.update_participant_flow': {'queue': 'market_data'},
    'app.tasks.market_data.fetch_real_time_prices': {'queue': 'high_priority'},
    'app.tasks.market_data.update_volatility_data': {'queue': 'market_data'},
    
    # Participant data tasks - time sensitive market data
    'app.tasks.participant_data.ingest_current_participant_data_task': {'queue': 'market_data'},
    'app.tasks.participant_data.ingest_historical_participant_data_task': {'queue': 'market_data'},
    'app.tasks.participant_data.generate_daily_participant_summary_task': {'queue': 'analysis'},
    'app.tasks.participant_data.schedule_weekly_historical_backfill': {'queue': 'maintenance'},
    'app.tasks.participant_data.cleanup_old_ingestion_logs': {'queue': 'maintenance'},
    
    # Insights tasks - analysis and reporting
    'app.tasks.insights_tasks.generate_daily_insights_task': {'queue': 'analysis'},
    'app.tasks.insights_tasks.update_dashboard_data_task': {'queue': 'analysis'},
    'app.tasks.insights_tasks.detect_behavioral_shifts_task': {'queue': 'analysis'},
    
    # Alert tasks - immediate priority
    'app.tasks.alerts.process_alert_queue': {'queue': 'alerts'},
    'app.tasks.alerts.send_immediate_alert': {'queue': 'high_priority'},
    'app.tasks.alerts.cleanup_expired_alerts': {'queue': 'maintenance'},
    
    # Monitoring and maintenance tasks - low priority
    'app.tasks.monitoring.health_check': {'queue': 'monitoring'},
    'app.tasks.monitoring.system_metrics': {'queue': 'monitoring'},
    'app.tasks.monitoring.cleanup_old_data': {'queue': 'maintenance'},
    'app.tasks.monitoring.backup_database': {'queue': 'maintenance'},
}

# Queue definitions with specific configurations
celery_app.conf.task_queues = (
    # High priority queue for time-sensitive tasks
    Queue('high_priority', 
          routing_key='high_priority',
          queue_arguments={'x-max-priority': 10}),
    
    # Market data queue for financial data updates
    Queue('market_data', 
          routing_key='market_data',
          queue_arguments={'x-max-priority': 8}),
    
    # Alerts queue for notification processing
    Queue('alerts', 
          routing_key='alerts',
          queue_arguments={'x-max-priority': 9}),
    
    # Analysis queue for data processing
    Queue('analysis', 
          routing_key='analysis',
          queue_arguments={'x-max-priority': 6}),
    
    # Scraping queue for data collection
    Queue('scraping', 
          routing_key='scraping',
          queue_arguments={'x-max-priority': 5}),
    
    # Monitoring queue for system health
    Queue('monitoring', 
          routing_key='monitoring',
          queue_arguments={'x-max-priority': 3}),
    
    # Maintenance queue for cleanup tasks
    Queue('maintenance', 
          routing_key='maintenance',
          queue_arguments={'x-max-priority': 1}),
    
    # Default queue
    Queue('celery', 
          routing_key='celery',
          queue_arguments={'x-max-priority': 4}),
)

# Celery Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # High-priority sources every minute during market hours
    'high-priority-scraping': {
        'task': 'app.tasks.scraping.scrape_high_priority',
        'schedule': float(settings.CELERY_BEAT_SCHEDULE_HIGH_PRIORITY_INTERVAL),
        'options': {'queue': 'high_priority'},
    },
    
    # General scraping every 30 minutes
    'general-scraping': {
        'task': 'app.tasks.scraping.scrape_general_sources',
        'schedule': float(settings.CELERY_BEAT_SCHEDULE_GENERAL_SCRAPING_INTERVAL),
        'options': {'queue': 'scraping'},
    },
    
    # Option chain updates every 5 minutes during market hours
    'option-chain-update': {
        'task': 'app.tasks.market_data.update_option_chains',
        'schedule': float(settings.CELERY_BEAT_SCHEDULE_OPTION_CHAIN_INTERVAL),
        'kwargs': {'market_hours_only': True},
        'options': {'queue': 'market_data'},
    },
    
    # Participant flow update daily after market close
    'participant-flow-update': {
        'task': 'app.tasks.market_data.update_participant_flow',
        'schedule': crontab(
            hour=settings.CELERY_BEAT_SCHEDULE_PARTICIPANT_FLOW_HOUR,
            minute=settings.CELERY_BEAT_SCHEDULE_PARTICIPANT_FLOW_MINUTE
        ),
        'options': {'queue': 'market_data'},
    },
    
    # NSE Participant data ingestion every 5 minutes during market hours
    'ingest-participant-data': {
        'task': 'app.tasks.participant_data.ingest_current_participant_data_task',
        'schedule': 300.0,  # Every 5 minutes
        'options': {'queue': 'market_data', 'expires': 240},
    },
    
    # Generate participant summary daily at 6:30 PM
    'generate-participant-summary': {
        'task': 'app.tasks.participant_data.generate_daily_participant_summary_task',
        'schedule': crontab(hour=18, minute=30),
        'options': {'queue': 'analysis'},
    },
    
    # Weekly historical data backfill on Monday at 2 AM
    'participant-data-backfill': {
        'task': 'app.tasks.participant_data.schedule_weekly_historical_backfill',
        'schedule': crontab(hour=2, minute=0, day_of_week=1),
        'options': {'queue': 'maintenance'},
    },
    
    # Monthly cleanup on first day at 3 AM
    'participant-data-cleanup': {
        'task': 'app.tasks.participant_data.cleanup_old_ingestion_logs',
        'schedule': crontab(hour=3, minute=0, day_of_month=1),
        'options': {'queue': 'maintenance'},
    },
    
    # Generate daily insights at 7 PM
    'generate-daily-insights': {
        'task': 'app.tasks.insights_tasks.generate_daily_insights_task',
        'schedule': crontab(hour=19, minute=0),
        'options': {'queue': 'analysis'},
    },
    
    # Update dashboard data every 30 minutes during market hours
    'update-dashboard-data': {
        'task': 'app.tasks.insights_tasks.update_dashboard_data_task',
        'schedule': 1800.0,  # 30 minutes
        'options': {'queue': 'analysis', 'expires': 1500},
    },
    
    # Detect behavioral shifts daily at 6 PM
    'detect-behavioral-shifts': {
        'task': 'app.tasks.insights_tasks.detect_behavioral_shifts_task',
        'schedule': crontab(hour=18, minute=0),
        'options': {'queue': 'analysis'},
    },
    
    # Process unanalyzed news every 10 minutes
    'process-unanalyzed-news': {
        'task': 'app.tasks.analysis.process_unanalyzed_news',
        'schedule': 600.0,  # 10 minutes
        'options': {'queue': 'analysis'},
    },
    
    # System health check every 5 minutes
    'system-health-check': {
        'task': 'app.tasks.monitoring.health_check',
        'schedule': 300.0,  # 5 minutes
        'options': {'queue': 'monitoring'},
    },
    
    # System metrics collection every minute
    'system-metrics': {
        'task': 'app.tasks.monitoring.system_metrics',
        'schedule': 60.0,  # 1 minute
        'options': {'queue': 'monitoring'},
    },
    
    # Cleanup expired alerts daily at 2 AM
    'cleanup-expired-alerts': {
        'task': 'app.tasks.alerts.cleanup_expired_alerts',
        'schedule': crontab(hour=2, minute=0),
        'options': {'queue': 'maintenance'},
    },
    
    # Cleanup old data weekly on Sunday at 3 AM
    'cleanup-old-data': {
        'task': 'app.tasks.monitoring.cleanup_old_data',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),
        'options': {'queue': 'maintenance'},
    },
    
    # Database backup daily at 4 AM
    'database-backup': {
        'task': 'app.tasks.monitoring.backup_database',
        'schedule': crontab(hour=4, minute=0),
        'options': {'queue': 'maintenance'},
    },
}

# Set the default queue
celery_app.conf.task_default_queue = 'celery'
celery_app.conf.task_default_exchange_type = 'direct'
celery_app.conf.task_default_routing_key = 'celery'

# Task execution configuration
celery_app.conf.task_compression = 'gzip'
celery_app.conf.result_compression = 'gzip'

# Error handling
celery_app.conf.task_reject_on_worker_lost = True
celery_app.conf.task_ignore_result = False

# Configure task retry defaults
celery_app.conf.task_annotations = {
    '*': {
        'rate_limit': '100/s',
        'retry_kwargs': {'max_retries': 3, 'countdown': 60},
    },
    'app.tasks.scraping.*': {
        'rate_limit': '10/s',
        'retry_kwargs': {'max_retries': 5, 'countdown': 30},
    },
    'app.tasks.market_data.*': {
        'rate_limit': '20/s',
        'retry_kwargs': {'max_retries': 3, 'countdown': 15},
    },
    'app.tasks.alerts.*': {
        'rate_limit': '50/s',
        'retry_kwargs': {'max_retries': 2, 'countdown': 5},
    },
}

# Logging configuration
celery_app.conf.worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
celery_app.conf.worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'

# Monitor task failures and retries
@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery configuration"""
    logger.info(f'Request: {self.request!r}')
    return {'message': 'Debug task executed successfully', 'worker': self.request.hostname}

# Initialize Celery application logging
logger.info("Celery application configured successfully")
logger.info(f"Broker URL: {settings.CELERY_BROKER_URL}")
logger.info(f"Result Backend: {settings.CELERY_RESULT_BACKEND}")
logger.info(f"Task Queues: {list(celery_app.conf.task_queues)}")

# Export the celery app
__all__ = ['celery_app'] 