"""
Celery Monitoring API Endpoints
Provides REST API for monitoring Celery workers, tasks, and infrastructure
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from datetime import datetime

from app.services.celery_monitoring_service import get_celery_monitoring_service
from app.services.logging_service import get_logger
from app.auth import get_current_user
from app.models.user import User
from app.core.config import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/monitoring/celery", tags=["Celery Monitoring"])

@router.get("/health", summary="Get overall Celery health status")
async def get_celery_health(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Get comprehensive health status of Celery infrastructure"""
    try:
        monitoring_service = get_celery_monitoring_service()
        health_data = await monitoring_service.get_overall_health()
        logger.info(f"Celery health check requested by user {current_user.id}")
        return health_data
    except Exception as e:
        logger.error(f"Failed to get Celery health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/workers", summary="Get worker status and information")
async def get_worker_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Get detailed information about active Celery workers"""
    try:
        from app.tasks.celery_app import celery_app
        
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active_tasks = inspect.active()
        
        workers_info = []
        total_active_tasks = 0
        
        if stats:
            for worker_name, worker_stats in stats.items():
                worker_active_tasks = len(active_tasks.get(worker_name, []))
                total_active_tasks += worker_active_tasks
                
                workers_info.append({
                    'name': worker_name,
                    'status': 'active',
                    'active_tasks': worker_active_tasks,
                    'pool': worker_stats.get('pool', {}),
                    'clock': worker_stats.get('clock', 0)
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'worker_count': len(workers_info),
            'total_active_tasks': total_active_tasks,
            'workers': workers_info
        }
    except Exception as e:
        logger.error(f"Failed to get worker status: {e}")
        raise HTTPException(status_code=500, detail=f"Worker status check failed: {str(e)}")

@router.get("/queues", summary="Get queue status and metrics")
async def get_queue_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Get status of all Celery queues including backlogs and priorities"""
    try:
        import redis
        from app.tasks.celery_app import celery_app
        
        redis_client = redis.from_url(settings.CELERY_BROKER_URL)
        queue_info = []
        total_pending_tasks = 0
        
        for queue in celery_app.conf.task_queues:
            queue_name = queue.name
            queue_length = redis_client.llen(queue_name)
            total_pending_tasks += queue_length
            
            status = 'healthy'
            if queue_length > 1000:
                status = 'error'
            elif queue_length > 100:
                status = 'warning'
            
            queue_info.append({
                'name': queue_name,
                'length': queue_length,
                'status': status,
                'priority': queue.queue_arguments.get('x-max-priority', 0)
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_pending_tasks': total_pending_tasks,
            'queue_count': len(queue_info),
            'queues': queue_info
        }
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=f"Queue status check failed: {str(e)}")

@router.get("/metrics", summary="Get performance metrics")
async def get_performance_metrics(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Get detailed performance metrics for Celery infrastructure"""
    try:
        import psutil
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_free_gb': round(disk.free / (1024**3), 2)
            },
            'celery': {
                'broker_url': settings.CELERY_BROKER_URL,
                'result_backend': settings.CELERY_RESULT_BACKEND
            }
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

@router.post("/workers/ping", summary="Ping all workers")
async def ping_workers(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Send ping to all active workers to test responsiveness"""
    try:
        from app.tasks.celery_app import celery_app
        
        inspect = celery_app.control.inspect()
        ping_results = inspect.ping()
        
        if not ping_results:
            return {
                'status': 'error',
                'message': 'No workers responded to ping',
                'responsive_workers': 0,
                'workers': []
            }
        
        worker_heartbeats = []
        responsive_count = 0
        
        for worker_name, ping_result in ping_results.items():
            is_responsive = ping_result.get('ok') == 'pong'
            if is_responsive:
                responsive_count += 1
            
            worker_heartbeats.append({
                'name': worker_name,
                'responsive': is_responsive,
                'response': ping_result
            })
        
        return {
            'status': 'healthy' if responsive_count > 0 else 'error',
            'responsive_workers': responsive_count,
            'total_workers': len(worker_heartbeats),
            'workers': worker_heartbeats,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to ping workers: {e}")
        raise HTTPException(status_code=500, detail=f"Worker ping failed: {str(e)}")

@router.get("/dashboard", summary="Get monitoring dashboard summary")
async def get_dashboard_summary(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Get a comprehensive summary for monitoring dashboard"""
    try:
        from app.tasks.celery_app import celery_app
        import redis
        import psutil
        
        # Get worker info
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active_tasks = inspect.active()
        
        # Get queue info
        redis_client = redis.from_url(settings.CELERY_BROKER_URL)
        total_pending = 0
        for queue in celery_app.conf.task_queues:
            total_pending += redis_client.llen(queue.name)
        
        # Get system info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Calculate totals
        worker_count = len(stats) if stats else 0
        total_active = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
        
        # Determine overall status
        overall_status = 'healthy'
        alerts = []
        
        if worker_count == 0:
            overall_status = 'error'
            alerts.append({'level': 'critical', 'message': 'No active workers found'})
        
        if total_pending > 1000:
            overall_status = 'warning' if overall_status == 'healthy' else overall_status
            alerts.append({'level': 'warning', 'message': f'High queue backlog: {total_pending} tasks'})
        
        if cpu_percent > 90 or memory.percent > 90:
            overall_status = 'warning' if overall_status == 'healthy' else overall_status
            alerts.append({'level': 'warning', 'message': 'High system resource usage'})
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'worker_count': worker_count,
            'total_pending_tasks': total_pending,
            'total_active_tasks': total_active,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent
            },
            'alerts': alerts
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard summary failed: {str(e)}")
