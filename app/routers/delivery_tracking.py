from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from app.database.connection import get_db
from app.services.delivery_tracking_service import get_delivery_tracking_service, DeliveryMetrics
from app.models.delivery import (
    AlertDelivery, DeliveryAttempt, DeadLetterQueue, DeliveryStats,
    DeliveryStatus, DeliveryChannel
)
from app.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/v1/delivery", tags=["delivery-tracking"])

# Pydantic schemas for API responses
class DeliveryStatusResponse(BaseModel):
    id: int
    alert_id: int
    user_id: str
    channel: str
    status: str
    attempts: int
    max_attempts: int
    created_at: datetime
    first_attempt_at: Optional[datetime] = None
    last_attempt_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    delivery_latency_ms: Optional[int] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    next_retry_at: Optional[datetime] = None

class DeliveryAttemptResponse(BaseModel):
    id: int
    delivery_id: int
    attempt_number: int
    attempted_at: datetime
    completed_at: Optional[datetime] = None
    success: bool
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    latency_ms: Optional[int] = None

class DeliveryMetricsResponse(BaseModel):
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    pending_deliveries: int
    avg_latency_ms: float
    success_rate: float

class DeadLetterQueueResponse(BaseModel):
    id: int
    alert_id: int
    user_id: str
    channel: str
    final_error_message: Optional[str] = None
    final_error_code: Optional[str] = None
    total_attempts: int
    created_at: datetime
    last_attempt_at: datetime
    reviewed: bool
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None

class DeliveryStatsResponse(BaseModel):
    date: datetime
    hour: Optional[int] = None
    channel: str
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    retried_deliveries: int
    expired_deliveries: int
    avg_latency_ms: Optional[float] = None
    success_rate: Optional[float] = None

@router.get("/status/{delivery_id}", response_model=DeliveryStatusResponse)
async def get_delivery_status(
    delivery_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the status of a specific delivery"""
    
    delivery_service = get_delivery_tracking_service()
    delivery = await delivery_service.get_delivery_status(delivery_id)
    
    if not delivery:
        raise HTTPException(status_code=404, detail="Delivery not found")
    
    # Check if user has access to this delivery
    if delivery.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return DeliveryStatusResponse(
        id=delivery.id,
        alert_id=delivery.alert_id,
        user_id=delivery.user_id,
        channel=delivery.channel.value,
        status=delivery.status.value,
        attempts=delivery.attempts,
        max_attempts=delivery.max_attempts,
        created_at=delivery.created_at,
        first_attempt_at=delivery.first_attempt_at,
        last_attempt_at=delivery.last_attempt_at,
        delivered_at=delivery.delivered_at,
        expires_at=delivery.expires_at,
        delivery_latency_ms=delivery.delivery_latency_ms,
        error_message=delivery.error_message,
        error_code=delivery.error_code,
        next_retry_at=delivery.next_retry_at
    )

@router.get("/user/{user_id}/deliveries", response_model=List[DeliveryStatusResponse])
async def get_user_deliveries(
    user_id: str,
    status: Optional[str] = Query(None, description="Filter by delivery status"),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get deliveries for a specific user"""
    
    # Check if user has access (admin or self)
    if str(current_user.id) != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    delivery_service = get_delivery_tracking_service()
    
    # Convert status string to enum if provided
    status_enum = None
    if status:
        try:
            status_enum = DeliveryStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    deliveries = await delivery_service.get_user_deliveries(
        user_id=user_id,
        status=status_enum,
        limit=limit
    )
    
    return [
        DeliveryStatusResponse(
            id=delivery.id,
            alert_id=delivery.alert_id,
            user_id=delivery.user_id,
            channel=delivery.channel.value,
            status=delivery.status.value,
            attempts=delivery.attempts,
            max_attempts=delivery.max_attempts,
            created_at=delivery.created_at,
            first_attempt_at=delivery.first_attempt_at,
            last_attempt_at=delivery.last_attempt_at,
            delivered_at=delivery.delivered_at,
            expires_at=delivery.expires_at,
            delivery_latency_ms=delivery.delivery_latency_ms,
            error_message=delivery.error_message,
            error_code=delivery.error_code,
            next_retry_at=delivery.next_retry_at
        )
        for delivery in deliveries
    ]

@router.get("/metrics", response_model=DeliveryMetricsResponse)
async def get_delivery_metrics(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    channel: Optional[str] = Query(None, description="Filter by delivery channel"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get delivery metrics for monitoring"""
    
    delivery_service = get_delivery_tracking_service()
    
    # Convert channel string to enum if provided
    channel_enum = None
    if channel:
        try:
            channel_enum = DeliveryChannel(channel)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid channel: {channel}")
    
    metrics = await delivery_service.get_delivery_metrics(
        hours=hours,
        channel=channel_enum
    )
    
    return DeliveryMetricsResponse(
        total_deliveries=metrics.total_deliveries,
        successful_deliveries=metrics.successful_deliveries,
        failed_deliveries=metrics.failed_deliveries,
        pending_deliveries=metrics.pending_deliveries,
        avg_latency_ms=metrics.avg_latency_ms,
        success_rate=metrics.success_rate
    )

@router.get("/attempts/{delivery_id}", response_model=List[DeliveryAttemptResponse])
async def get_delivery_attempts(
    delivery_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all delivery attempts for a specific delivery"""
    
    # First check if delivery exists and user has access
    delivery_service = get_delivery_tracking_service()
    delivery = await delivery_service.get_delivery_status(delivery_id)
    
    if not delivery:
        raise HTTPException(status_code=404, detail="Delivery not found")
    
    if delivery.user_id != str(current_user.id) and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get attempts from database
    attempts = db.query(DeliveryAttempt).filter(
        DeliveryAttempt.delivery_id == delivery_id
    ).order_by(DeliveryAttempt.attempted_at.desc()).all()
    
    return [
        DeliveryAttemptResponse(
            id=attempt.id,
            delivery_id=attempt.delivery_id,
            attempt_number=attempt.attempt_number,
            attempted_at=attempt.attempted_at,
            completed_at=attempt.completed_at,
            success=attempt.success,
            error_message=attempt.error_message,
            error_code=attempt.error_code,
            latency_ms=attempt.latency_ms
        )
        for attempt in attempts
    ]

@router.post("/cancel/{delivery_id}")
async def cancel_delivery(
    delivery_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Cancel a pending or retrying delivery"""
    
    delivery_service = get_delivery_tracking_service()
    delivery = await delivery_service.get_delivery_status(delivery_id)
    
    if not delivery:
        raise HTTPException(status_code=404, detail="Delivery not found")
    
    if delivery.user_id != str(current_user.id) and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = await delivery_service.cancel_delivery(delivery_id)
    
    if not success:
        raise HTTPException(
            status_code=400, 
            detail="Delivery cannot be cancelled (already delivered or failed)"
        )
    
    return {"message": "Delivery cancelled successfully"}

# Admin-only endpoints
@router.get("/admin/pending-retries", response_model=List[DeliveryStatusResponse])
async def get_pending_retries(
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get deliveries that are ready for retry (admin only)"""
    
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    delivery_service = get_delivery_tracking_service()
    deliveries = await delivery_service.get_pending_retries(limit=limit)
    
    return [
        DeliveryStatusResponse(
            id=delivery.id,
            alert_id=delivery.alert_id,
            user_id=delivery.user_id,
            channel=delivery.channel.value,
            status=delivery.status.value,
            attempts=delivery.attempts,
            max_attempts=delivery.max_attempts,
            created_at=delivery.created_at,
            first_attempt_at=delivery.first_attempt_at,
            last_attempt_at=delivery.last_attempt_at,
            delivered_at=delivery.delivered_at,
            expires_at=delivery.expires_at,
            delivery_latency_ms=delivery.delivery_latency_ms,
            error_message=delivery.error_message,
            error_code=delivery.error_code,
            next_retry_at=delivery.next_retry_at
        )
        for delivery in deliveries
    ]

@router.get("/admin/dead-letter-queue", response_model=List[DeadLetterQueueResponse])
async def get_dead_letter_queue(
    limit: int = Query(50, ge=1, le=200),
    reviewed: Optional[bool] = Query(None, description="Filter by review status"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get items from the dead letter queue (admin only)"""
    
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    query = db.query(DeadLetterQueue)
    
    if reviewed is not None:
        query = query.filter(DeadLetterQueue.reviewed == reviewed)
    
    dlq_items = query.order_by(
        DeadLetterQueue.created_at.desc()
    ).limit(limit).all()
    
    return [
        DeadLetterQueueResponse(
            id=item.id,
            alert_id=item.alert_id,
            user_id=item.user_id,
            channel=item.channel.value,
            final_error_message=item.final_error_message,
            final_error_code=item.final_error_code,
            total_attempts=item.total_attempts,
            created_at=item.created_at,
            last_attempt_at=item.last_attempt_at,
            reviewed=item.reviewed,
            reviewed_at=item.reviewed_at,
            reviewed_by=item.reviewed_by
        )
        for item in dlq_items
    ]

@router.post("/admin/dead-letter-queue/{dlq_id}/review")
async def review_dead_letter_item(
    dlq_id: int,
    resolution_notes: str = Query(..., description="Notes about the resolution"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Mark a dead letter queue item as reviewed (admin only)"""
    
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    dlq_item = db.query(DeadLetterQueue).filter(
        DeadLetterQueue.id == dlq_id
    ).first()
    
    if not dlq_item:
        raise HTTPException(status_code=404, detail="Dead letter queue item not found")
    
    dlq_item.reviewed = True
    dlq_item.reviewed_at = datetime.utcnow()
    dlq_item.reviewed_by = str(current_user.id)
    dlq_item.resolution_notes = resolution_notes
    
    db.commit()
    
    return {"message": "Dead letter queue item reviewed successfully"}

@router.get("/admin/stats", response_model=List[DeliveryStatsResponse])
async def get_delivery_statistics(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    channel: Optional[str] = Query(None, description="Filter by delivery channel"),
    hourly: bool = Query(False, description="Return hourly stats instead of daily"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get aggregate delivery statistics (admin only)"""
    
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    since = datetime.utcnow() - timedelta(hours=hours)
    
    query = db.query(DeliveryStats).filter(
        DeliveryStats.date >= since
    )
    
    if channel:
        try:
            channel_enum = DeliveryChannel(channel)
            query = query.filter(DeliveryStats.channel == channel_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid channel: {channel}")
    
    if hourly:
        query = query.filter(DeliveryStats.hour.isnot(None))
    else:
        query = query.filter(DeliveryStats.hour.is_(None))
    
    stats = query.order_by(DeliveryStats.date.desc()).all()
    
    return [
        DeliveryStatsResponse(
            date=stat.date,
            hour=stat.hour,
            channel=stat.channel.value,
            total_deliveries=stat.total_deliveries,
            successful_deliveries=stat.successful_deliveries,
            failed_deliveries=stat.failed_deliveries,
            retried_deliveries=stat.retried_deliveries,
            expired_deliveries=stat.expired_deliveries,
            avg_latency_ms=stat.avg_latency_ms,
            success_rate=stat.success_rate
        )
        for stat in stats
    ]

@router.post("/admin/update-stats")
async def update_delivery_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Manually trigger delivery statistics update (admin only)"""
    
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    delivery_service = get_delivery_tracking_service()
    await delivery_service.update_delivery_statistics()
    
    return {"message": "Delivery statistics updated successfully"}

@router.post("/admin/cleanup-expired")
async def cleanup_expired_deliveries(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Manually cleanup expired deliveries (admin only)"""
    
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    delivery_service = get_delivery_tracking_service()
    count = await delivery_service.mark_expired_deliveries()
    
    return {"message": f"Marked {count} deliveries as expired"} 