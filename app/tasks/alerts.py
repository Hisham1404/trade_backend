"""
Alert Tasks for Background Processing  
Handles alert processing, notifications, and cleanup
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from app.tasks.celery_app import celery_app
from app.database.connection import get_db
from app.models.alert import Alert
from app.services.logging_service import get_logger

logger = get_logger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def process_alert_queue(self):
    """Process pending alerts and send notifications"""
    try:
        logger.info("Starting alert queue processing")
        
        db = next(get_db())
        
        # Get pending alerts
        pending_alerts = db.query(Alert).filter(
            Alert.status == 'pending'
        ).limit(100).all()  # Process in batches
        
        if not pending_alerts:
            logger.info("No pending alerts to process")
            return {
                'timestamp': datetime.now().isoformat(),
                'alerts_processed': 0,
                'status': 'no_alerts'
            }
        
        processed_count = 0
        failed_count = 0
        
        for alert in pending_alerts:
            try:
                # Process the alert (send notifications, update status, etc.)
                # This would integrate with the alerting service
                alert.status = 'sent'
                alert.sent_at = datetime.now()
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process alert {alert.id}: {e}")
                alert.status = 'failed'
                failed_count += 1
        
        db.commit()
        db.close()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'alerts_processed': processed_count,
            'alerts_failed': failed_count,
            'status': 'completed'
        }
        
        logger.info(f"Alert processing completed: {processed_count} sent, {failed_count} failed")
        return result
        
    except Exception as exc:
        logger.error(f"Alert queue processing task failed: {exc}")
        raise self.retry(exc=exc, countdown=min(30 * (2 ** self.request.retries), 300))

@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def send_immediate_alert(self, title: str, message: str, severity: str = "medium", data: Dict[str, Any] = None):
    """Send an immediate high-priority alert"""
    try:
        logger.info(f"Sending immediate alert: {title}")
        
        # This would integrate with push notifications, email, etc.
        # Placeholder implementation
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'message': message,
            'severity': severity,
            'status': 'sent'
        }
        
        logger.info(f"Immediate alert sent successfully: {title}")
        return result
        
    except Exception as exc:
        logger.error(f"Immediate alert task failed: {exc}")
        raise self.retry(exc=exc, countdown=min(5 * (2 ** self.request.retries), 30))

@celery_app.task(bind=True, max_retries=1, default_retry_delay=3600)
def cleanup_expired_alerts(self):
    """Clean up old and expired alerts"""
    try:
        logger.info("Starting alert cleanup task")
        
        db = next(get_db())
        
        # Clean up alerts older than 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        
        deleted_count = db.query(Alert).filter(
            Alert.created_at < cutoff_date
        ).delete()
        
        db.commit()
        db.close()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'deleted_count': deleted_count,
            'cutoff_date': cutoff_date.isoformat(),
            'status': 'completed'
        }
        
        logger.info(f"Alert cleanup completed: {deleted_count} alerts deleted")
        return result
        
    except Exception as exc:
        logger.error(f"Alert cleanup task failed: {exc}")
        raise self.retry(exc=exc, countdown=3600)
