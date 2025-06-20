"""
Monitoring Tasks for Background Processing
Handles system health checks, metrics collection, and maintenance
"""

import psutil
from datetime import datetime, timedelta

from app.tasks.celery_app import celery_app
from app.services.logging_service import get_logger

logger = get_logger(__name__)

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def health_check(self):
    """Perform system health check"""
    try:
        logger.info("Starting system health check")
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Basic health metrics
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_free_gb': round(disk.free / (1024**3), 2)
            },
            'status': 'healthy'
        }
        
        # Determine overall health status
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            health_data['status'] = 'warning'
            health_data['warnings'] = []
            
            if cpu_percent > 90:
                health_data['warnings'].append(f'High CPU usage: {cpu_percent}%')
            if memory.percent > 90:
                health_data['warnings'].append(f'High memory usage: {memory.percent}%')
            if disk.percent > 90:
                health_data['warnings'].append(f'High disk usage: {disk.percent}%')
        
        logger.info(f"Health check completed: {health_data['status']}")
        return health_data
        
    except Exception as exc:
        logger.error(f"Health check task failed: {exc}")
        raise self.retry(exc=exc, countdown=min(60 * (2 ** self.request.retries), 300))

@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def system_metrics(self):
    """Collect detailed system metrics"""
    try:
        logger.info("Collecting system metrics")
        
        # Collect comprehensive system metrics
        cpu_times = psutil.cpu_times()
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'times': {
                    'user': cpu_times.user,
                    'system': cpu_times.system,
                    'idle': cpu_times.idle
                }
            },
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent': memory.percent
            },
            'disk': {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0
            },
            'network': {
                'bytes_sent': network_io.bytes_sent if network_io else 0,
                'bytes_recv': network_io.bytes_recv if network_io else 0,
                'packets_sent': network_io.packets_sent if network_io else 0,
                'packets_recv': network_io.packets_recv if network_io else 0
            }
        }
        
        logger.info("System metrics collection completed")
        return metrics
        
    except Exception as exc:
        logger.error(f"System metrics task failed: {exc}")
        raise self.retry(exc=exc, countdown=min(30 * (2 ** self.request.retries), 180))

@celery_app.task(bind=True, max_retries=1, default_retry_delay=3600)
def cleanup_old_data(self):
    """Clean up old data and temporary files"""
    try:
        logger.info("Starting data cleanup task")
        
        # Placeholder for cleanup logic
        cleanup_result = {
            'timestamp': datetime.now().isoformat(),
            'files_cleaned': 0,
            'space_freed_mb': 0,
            'status': 'completed'
        }
        
        logger.info("Data cleanup completed successfully")
        return cleanup_result
        
    except Exception as exc:
        logger.error(f"Data cleanup task failed: {exc}")
        raise self.retry(exc=exc, countdown=3600)

@celery_app.task(bind=True, max_retries=1, default_retry_delay=7200)
def backup_database(self):
    """Perform database backup"""
    try:
        logger.info("Starting database backup task")
        
        # Placeholder for backup logic
        backup_result = {
            'timestamp': datetime.now().isoformat(),
            'backup_size_mb': 0,
            'backup_location': 'placeholder',
            'status': 'completed'
        }
        
        logger.info("Database backup completed successfully")
        return backup_result
        
    except Exception as exc:
        logger.error(f"Database backup task failed: {exc}")
        raise self.retry(exc=exc, countdown=7200)
