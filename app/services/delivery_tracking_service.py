from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
import logging
from dataclasses import dataclass

from app.models.delivery import (
    AlertDelivery, DeliveryAttempt, DeliveryStats, DeadLetterQueue,
    DeliveryStatus, DeliveryChannel
)
from app.models.alert import Alert
from app.database.connection import get_db

logger = logging.getLogger(__name__)

@dataclass
class DeliveryResult:
    """Result of a delivery attempt"""
    success: bool
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    latency_ms: Optional[int] = None
    delivery_context: Optional[Dict[str, Any]] = None

@dataclass
class DeliveryMetrics:
    """Delivery metrics for monitoring"""
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    pending_deliveries: int
    avg_latency_ms: float
    success_rate: float

class DeliveryTrackingService:
    """Service for tracking alert deliveries with retry logic and analytics"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.max_retry_attempts = 5
        self.base_retry_delay = 60  # Base delay in seconds
        self.max_retry_delay = 3600  # Maximum delay (1 hour)
        self.delivery_timeout_hours = 24  # After this, mark as expired
        
    async def create_delivery_tracking(
        self,
        alert_id: int,
        user_id: str,
        channel: DeliveryChannel,
        device_token: Optional[str] = None,
        priority: int = 1,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AlertDelivery:
        """Create a new delivery tracking record"""
        
        if expires_at is None:
            expires_at = datetime.utcnow() + timedelta(hours=self.delivery_timeout_hours)
            
        delivery = AlertDelivery(
            alert_id=alert_id,
            user_id=user_id,
            channel=channel,
            device_token=device_token,
            priority=priority,
            expires_at=expires_at,
            delivery_metadata=metadata or {},
            max_attempts=self.max_retry_attempts
        )
        
        self.db.add(delivery)
        self.db.commit()
        self.db.refresh(delivery)
        
        logger.info(f"Created delivery tracking {delivery.id} for alert {alert_id} via {channel.value}")
        return delivery
    
    async def record_delivery_attempt(
        self,
        delivery_id: int,
        result: DeliveryResult,
        attempt_number: int
    ) -> DeliveryAttempt:
        """Record a delivery attempt with its result"""
        
        attempt = DeliveryAttempt(
            delivery_id=delivery_id,
            attempt_number=attempt_number,
            attempted_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            success=result.success,
            error_message=result.error_message,
            error_code=result.error_code,
            latency_ms=result.latency_ms,
            delivery_context=result.delivery_context
        )
        
        self.db.add(attempt)
        
        # Update the delivery record
        delivery = self.db.query(AlertDelivery).filter(
            AlertDelivery.id == delivery_id
        ).first()
        
        if delivery:
            delivery.attempts = attempt_number
            delivery.last_attempt_at = datetime.utcnow()
            
            if delivery.first_attempt_at is None:
                delivery.first_attempt_at = datetime.utcnow()
            
            if result.success:
                delivery.status = DeliveryStatus.DELIVERED
                delivery.delivered_at = datetime.utcnow()
                if delivery.first_attempt_at:
                    latency = datetime.utcnow() - delivery.first_attempt_at
                    delivery.delivery_latency_ms = int(latency.total_seconds() * 1000)
            else:
                delivery.error_message = result.error_message
                delivery.error_code = result.error_code
                
                # Determine next action based on attempts
                if attempt_number >= delivery.max_attempts:
                    delivery.status = DeliveryStatus.FAILED
                    # Move to dead letter queue
                    await self._move_to_dead_letter_queue(delivery)
                else:
                    delivery.status = DeliveryStatus.RETRYING
                    # Schedule next retry with exponential backoff
                    delay_seconds = min(
                        self.base_retry_delay * (2 ** (attempt_number - 1)),
                        self.max_retry_delay
                    )
                    delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
                    delivery.retry_backoff_seconds = delay_seconds
        
        self.db.commit()
        self.db.refresh(attempt)
        
        logger.info(f"Recorded delivery attempt {attempt.id} for delivery {delivery_id}: {result.success}")
        return attempt
    
    async def get_pending_retries(self, limit: int = 100) -> List[AlertDelivery]:
        """Get deliveries that are ready for retry"""
        
        now = datetime.utcnow()
        
        deliveries = self.db.query(AlertDelivery).filter(
            and_(
                AlertDelivery.status == DeliveryStatus.RETRYING,
                AlertDelivery.next_retry_at <= now,
                AlertDelivery.expires_at > now,
                AlertDelivery.attempts < AlertDelivery.max_attempts
            )
        ).order_by(
            AlertDelivery.priority.desc(),
            AlertDelivery.next_retry_at.asc()
        ).limit(limit).all()
        
        return deliveries
    
    async def get_expired_deliveries(self, limit: int = 100) -> List[AlertDelivery]:
        """Get deliveries that have expired"""
        
        now = datetime.utcnow()
        
        deliveries = self.db.query(AlertDelivery).filter(
            and_(
                AlertDelivery.expires_at <= now,
                AlertDelivery.status.in_([
                    DeliveryStatus.PENDING,
                    DeliveryStatus.RETRYING
                ])
            )
        ).limit(limit).all()
        
        return deliveries
    
    async def mark_expired_deliveries(self) -> int:
        """Mark expired deliveries and move them to dead letter queue"""
        
        expired_deliveries = await self.get_expired_deliveries()
        count = 0
        
        for delivery in expired_deliveries:
            delivery.status = DeliveryStatus.EXPIRED
            await self._move_to_dead_letter_queue(delivery)
            count += 1
        
        if count > 0:
            self.db.commit()
            logger.info(f"Marked {count} deliveries as expired")
        
        return count
    
    async def _move_to_dead_letter_queue(self, delivery: AlertDelivery) -> None:
        """Move a failed delivery to the dead letter queue"""
        
        # Get the alert data for the payload
        alert = self.db.query(Alert).filter(Alert.id == delivery.alert_id).first()
        if not alert:
            logger.error(f"Alert {delivery.alert_id} not found for dead letter queue")
            return
        
        # Create alert payload
        alert_payload = {
            "id": alert.id,
            "name": alert.name,
            "alert_type": alert.alert_type,
            "symbol": alert.symbol,
            "condition_type": alert.condition_type,
            "trigger_value": float(alert.trigger_value) if alert.trigger_value else None,
            "priority": alert.priority,
            "message_template": alert.message_template,
            "created_at": alert.created_at.isoformat() if alert.created_at else None
        }
        
        dlq_entry = DeadLetterQueue(
            original_delivery_id=delivery.id,
            alert_id=delivery.alert_id,
            user_id=delivery.user_id,
            channel=delivery.channel,
            final_error_message=delivery.error_message,
            final_error_code=delivery.error_code,
            total_attempts=delivery.attempts,
            alert_payload=alert_payload,
            last_attempt_at=delivery.last_attempt_at or datetime.utcnow()
        )
        
        self.db.add(dlq_entry)
        logger.info(f"Moved delivery {delivery.id} to dead letter queue")
    
    async def get_delivery_metrics(
        self,
        hours: int = 24,
        channel: Optional[DeliveryChannel] = None
    ) -> DeliveryMetrics:
        """Get delivery metrics for the specified time period"""
        
        since = datetime.utcnow() - timedelta(hours=hours)
        
        query = self.db.query(AlertDelivery).filter(
            AlertDelivery.created_at >= since
        )
        
        if channel:
            query = query.filter(AlertDelivery.channel == channel)
        
        deliveries = query.all()
        
        total = len(deliveries)
        successful = len([d for d in deliveries if d.status == DeliveryStatus.DELIVERED])
        failed = len([d for d in deliveries if d.status == DeliveryStatus.FAILED])
        pending = len([d for d in deliveries if d.status in [
            DeliveryStatus.PENDING, DeliveryStatus.RETRYING
        ]])
        
        # Calculate average latency for successful deliveries
        successful_deliveries = [d for d in deliveries if d.delivery_latency_ms is not None]
        avg_latency = (
            sum(d.delivery_latency_ms for d in successful_deliveries) / len(successful_deliveries)
            if successful_deliveries else 0.0
        )
        
        success_rate = (successful / total * 100) if total > 0 else 0.0
        
        return DeliveryMetrics(
            total_deliveries=total,
            successful_deliveries=successful,
            failed_deliveries=failed,
            pending_deliveries=pending,
            avg_latency_ms=avg_latency,
            success_rate=success_rate
        )
    
    async def get_delivery_status(self, delivery_id: int) -> Optional[AlertDelivery]:
        """Get the current status of a delivery"""
        
        return self.db.query(AlertDelivery).filter(
            AlertDelivery.id == delivery_id
        ).first()
    
    async def get_user_deliveries(
        self,
        user_id: str,
        status: Optional[DeliveryStatus] = None,
        limit: int = 50
    ) -> List[AlertDelivery]:
        """Get deliveries for a specific user"""
        
        query = self.db.query(AlertDelivery).filter(
            AlertDelivery.user_id == user_id
        )
        
        if status:
            query = query.filter(AlertDelivery.status == status)
        
        return query.order_by(
            AlertDelivery.created_at.desc()
        ).limit(limit).all()
    
    async def cancel_delivery(self, delivery_id: int) -> bool:
        """Cancel a pending or retrying delivery"""
        
        delivery = self.db.query(AlertDelivery).filter(
            AlertDelivery.id == delivery_id
        ).first()
        
        if not delivery:
            return False
        
        if delivery.status in [DeliveryStatus.PENDING, DeliveryStatus.RETRYING]:
            delivery.status = DeliveryStatus.CANCELLED
            self.db.commit()
            logger.info(f"Cancelled delivery {delivery_id}")
            return True
        
        return False
    
    async def update_delivery_statistics(self) -> None:
        """Update aggregate delivery statistics"""
        
        now = datetime.utcnow()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        current_date = now.date()
        
        # Update hourly stats for each channel
        for channel in DeliveryChannel:
            await self._update_stats_for_period(
                channel, current_date, current_hour.hour
            )
        
        # Update daily stats for each channel (hour=None for daily)
        for channel in DeliveryChannel:
            await self._update_stats_for_period(
                channel, current_date, None
            )
    
    async def _update_stats_for_period(
        self,
        channel: DeliveryChannel,
        date: datetime.date,
        hour: Optional[int]
    ) -> None:
        """Update statistics for a specific time period"""
        
        # Define time range
        if hour is not None:
            start_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
            end_time = start_time + timedelta(hours=1)
        else:
            start_time = datetime.combine(date, datetime.min.time())
            end_time = start_time + timedelta(days=1)
        
        # Query deliveries in this period
        deliveries = self.db.query(AlertDelivery).filter(
            and_(
                AlertDelivery.channel == channel,
                AlertDelivery.created_at >= start_time,
                AlertDelivery.created_at < end_time
            )
        ).all()
        
        # Calculate metrics
        total = len(deliveries)
        successful = len([d for d in deliveries if d.status == DeliveryStatus.DELIVERED])
        failed = len([d for d in deliveries if d.status == DeliveryStatus.FAILED])
        retried = len([d for d in deliveries if d.attempts > 1])
        expired = len([d for d in deliveries if d.status == DeliveryStatus.EXPIRED])
        
        # Latency metrics
        successful_with_latency = [d for d in deliveries if d.delivery_latency_ms is not None]
        avg_latency = (
            sum(d.delivery_latency_ms for d in successful_with_latency) / len(successful_with_latency)
            if successful_with_latency else None
        )
        max_latency = max((d.delivery_latency_ms for d in successful_with_latency), default=None)
        min_latency = min((d.delivery_latency_ms for d in successful_with_latency), default=None)
        
        success_rate = (successful / total * 100) if total > 0 else None
        
        # Check if stats record exists
        existing_stats = self.db.query(DeliveryStats).filter(
            and_(
                DeliveryStats.date == start_time,
                DeliveryStats.hour == hour,
                DeliveryStats.channel == channel
            )
        ).first()
        
        if existing_stats:
            # Update existing record
            existing_stats.total_deliveries = total
            existing_stats.successful_deliveries = successful
            existing_stats.failed_deliveries = failed
            existing_stats.retried_deliveries = retried
            existing_stats.expired_deliveries = expired
            existing_stats.avg_latency_ms = avg_latency
            existing_stats.max_latency_ms = max_latency
            existing_stats.min_latency_ms = min_latency
            existing_stats.success_rate = success_rate
            existing_stats.updated_at = datetime.utcnow()
        else:
            # Create new record
            stats = DeliveryStats(
                date=start_time,
                hour=hour,
                channel=channel,
                total_deliveries=total,
                successful_deliveries=successful,
                failed_deliveries=failed,
                retried_deliveries=retried,
                expired_deliveries=expired,
                avg_latency_ms=avg_latency,
                max_latency_ms=max_latency,
                min_latency_ms=min_latency,
                success_rate=success_rate
            )
            self.db.add(stats)
        
        self.db.commit()

# Singleton service instance
_delivery_tracking_service: Optional[DeliveryTrackingService] = None

def get_delivery_tracking_service() -> DeliveryTrackingService:
    """Get or create the delivery tracking service instance"""
    global _delivery_tracking_service
    
    if _delivery_tracking_service is None:
        db_session = next(get_db())
        _delivery_tracking_service = DeliveryTrackingService(db_session)
    
    return _delivery_tracking_service 