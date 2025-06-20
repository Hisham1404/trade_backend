"""
Celery Monitoring Service
Provides comprehensive monitoring for Celery workers, tasks, and queues
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from celery import Celery
from celery.events.state import State
from kombu import Connection

from app.tasks.celery_app import celery_app
from app.core.config import settings
from app.services.logging_service import get_logger

logger = get_logger(__name__)

class CeleryMonitoringService:
    """
    Comprehensive monitoring service for Celery infrastructure
    Provides health checks, metrics, and real-time monitoring
    """
    
    def __init__(self):
        self.celery_app = celery_app
        self.state = State()
        self.last_health_check = None
        self.task_metrics = {}
        self.worker_metrics = {}
        self.queue_metrics = {}
        
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status of the Celery infrastructure"""
        try:
            logger.info("Performing comprehensive Celery health check")
            
            timestamp = datetime.now()
            
            # Check broker connectivity
            broker_health = await self._check_broker_health()
            workers_health = await self._check_workers_health()
            queues_health = await self._check_queues_health()
            tasks_health = await self._check_tasks_health()
            
            # Calculate overall health status
            all_checks = [broker_health, workers_health, queues_health, tasks_health]
            overall_status = "healthy"
            
            warning_count = sum(1 for check in all_checks if check.get('status') == 'warning')
            error_count = sum(1 for check in all_checks if check.get('status') == 'error')
            
            if error_count > 0:
                overall_status = "error"
            elif warning_count > 0:
                overall_status = "warning"
            
            health_report = {
                'timestamp': timestamp.isoformat(),
                'overall_status': overall_status,
                'broker': broker_health,
                'workers': workers_health,
                'queues': queues_health,
                'tasks': tasks_health,
                'summary': {
                    'healthy_components': len([c for c in all_checks if c.get('status') == 'healthy']),
                    'warning_components': warning_count,
                    'error_components': error_count,
                    'total_components': len(all_checks)
                }
            }
            
            self.last_health_check = health_report
            logger.info(f"Health check completed: {overall_status}")
            return health_report
            
        except Exception as e:
            logger.error(f"Failed to perform health check: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e),
                'message': 'Health check failed'
            }
    
    async def _check_broker_health(self) -> Dict[str, Any]:
        """Check Redis broker connectivity and health"""
        try:
            with Connection(settings.CELERY_BROKER_URL) as conn:
                conn.connect()
                info = conn.info() if hasattr(conn, 'info') else 'available'
                
                return {
                    'status': 'healthy',
                    'connection': 'connected',
                    'broker_url': settings.CELERY_BROKER_URL,
                    'transport': conn.transport_cls.__name__,
                    'info': info
                }
                
        except Exception as e:
            logger.error(f"Broker health check failed: {e}")
            return {
                'status': 'error',
                'connection': 'failed',
                'error': str(e),
                'broker_url': settings.CELERY_BROKER_URL
            }

# Global monitoring service instance
_monitoring_service = None

def get_celery_monitoring_service() -> CeleryMonitoringService:
    """Get or create the monitoring service instance"""
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = CeleryMonitoringService()
    
    return _monitoring_service
