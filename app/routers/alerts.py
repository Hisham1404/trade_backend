from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from app.database.connection import get_db
from app.services.alerting_service import get_alerting_service
from app.auth.jwt_handler import get_current_user
from app.models.user import User
from app.models.alert import Alert
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class AlertRuleCreate(BaseModel):
    rule_id: str
    name: str
    description: str
    alert_type: str
    conditions: List[Dict[str, Any]]
    priority: str
    template: str
    cooldown_minutes: int = 30
    rate_limit_per_hour: int = 10


class UserAlertCreate(BaseModel):
    """Schema for creating user alerts"""
    asset_symbol: str = Field(..., description="Asset symbol to monitor")
    alert_type: str = Field(..., description="Type of alert (price, sentiment, technical)")
    threshold_value: Optional[float] = Field(None, description="Threshold value for triggering")
    condition: str = Field(..., description="Condition (above, below, equal)")
    is_active: bool = Field(True, description="Whether alert is active")
    notes: Optional[str] = Field(None, description="Additional notes")


class UserAlertUpdate(BaseModel):
    """Schema for updating user alerts"""
    threshold_value: Optional[float] = None
    condition: Optional[str] = None
    is_active: Optional[bool] = None
    notes: Optional[str] = None


class UserAlertResponse(BaseModel):
    """Schema for user alert responses"""
    id: int
    asset_symbol: str
    alert_type: str
    threshold_value: Optional[float]
    condition: str
    is_active: bool
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_triggered: Optional[datetime]
    trigger_count: int
    
    class Config:
        orm_mode = True


class AlertNotificationResponse(BaseModel):
    """Schema for alert notification responses"""
    id: int
    alert_id: int
    title: str
    message: str
    asset_symbol: str
    alert_type: str
    priority: str
    is_read: bool
    created_at: datetime
    
    class Config:
        orm_mode = True


class MarketImpactAlertRequest(BaseModel):
    asset_symbol: str
    impact_score: float
    confidence: float
    reason: str
    user_id: Optional[int] = None


class SentimentAlertRequest(BaseModel):
    asset_symbol: str
    sentiment_score: float
    sentiment_label: str
    news_title: str
    user_id: Optional[int] = None


class PriceMovementAlertRequest(BaseModel):
    asset_symbol: str
    current_price: float
    price_change_percent: float
    user_id: Optional[int] = None


class TestAlertRequest(BaseModel):
    user_id: int
    alert_type: str = "system_notification"
    title: str = "Test Alert"
    message: str = "This is a test alert from the system"


# User Alert Management Endpoints
@router.post("/", response_model=UserAlertResponse, status_code=201)
async def create_alert(
    alert_data: UserAlertCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new personalized alert for the current user"""
    try:
        # Create new alert record
        new_alert = Alert(
            user_id=current_user.id,
            asset_symbol=alert_data.asset_symbol.upper(),
            alert_type=alert_data.alert_type,
            threshold_value=alert_data.threshold_value,
            condition=alert_data.condition,
            is_active=alert_data.is_active,
            notes=alert_data.notes,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            trigger_count=0
        )
        
        db.add(new_alert)
        db.commit()
        db.refresh(new_alert)
        
        logger.info(f"Created alert {new_alert.id} for user {current_user.id}")
        return new_alert
        
    except Exception as e:
        logger.error(f"Failed to create alert: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create alert: {str(e)}"
        )


@router.get("/", response_model=List[UserAlertResponse])
async def get_alerts(
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    asset_symbol: Optional[str] = Query(None, description="Filter by asset symbol"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of alerts to return"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's personalized alerts with filtering and pagination"""
    try:
        query = db.query(Alert).filter(Alert.user_id == current_user.id)
        
        # Apply filters
        if is_active is not None:
            query = query.filter(Alert.is_active == is_active)
        if asset_symbol:
            query = query.filter(Alert.asset_symbol == asset_symbol.upper())
        if alert_type:
            query = query.filter(Alert.alert_type == alert_type)
        
        # Apply pagination and ordering
        alerts = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()
        
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get alerts for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )


@router.get("/{alert_id}", response_model=UserAlertResponse)
async def get_alert(
    alert_id: int = Path(..., description="Alert ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific alert by ID"""
    try:
        alert = db.query(Alert).filter(
            Alert.id == alert_id,
            Alert.user_id == current_user.id
        ).first()
        
        if not alert:
            raise HTTPException(
                status_code=404,
                detail="Alert not found"
            )
        
        return alert
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alert: {str(e)}"
        )


@router.put("/{alert_id}", response_model=UserAlertResponse)
async def update_alert(
    alert_id: int,
    alert_data: UserAlertUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an existing alert"""
    try:
        alert = db.query(Alert).filter(
            Alert.id == alert_id,
            Alert.user_id == current_user.id
        ).first()
        
        if not alert:
            raise HTTPException(
                status_code=404,
                detail="Alert not found"
            )
        
        # Update fields if provided
        if alert_data.threshold_value is not None:
            alert.threshold_value = alert_data.threshold_value
        if alert_data.condition is not None:
            alert.condition = alert_data.condition
        if alert_data.is_active is not None:
            alert.is_active = alert_data.is_active
        if alert_data.notes is not None:
            alert.notes = alert_data.notes
        
        alert.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(alert)
        
        logger.info(f"Updated alert {alert_id} for user {current_user.id}")
        return alert
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update alert {alert_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update alert: {str(e)}"
        )


@router.delete("/{alert_id}", status_code=204)
async def delete_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an alert"""
    try:
        alert = db.query(Alert).filter(
            Alert.id == alert_id,
            Alert.user_id == current_user.id
        ).first()
        
        if not alert:
            raise HTTPException(
                status_code=404,
                detail="Alert not found"
            )
        
        db.delete(alert)
        db.commit()
        
        logger.info(f"Deleted alert {alert_id} for user {current_user.id}")
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete alert {alert_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete alert: {str(e)}"
        )


@router.post("/{alert_id}/acknowledge", status_code=200)
async def acknowledge_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Acknowledge an alert notification"""
    try:
        alert = db.query(Alert).filter(
            Alert.id == alert_id,
            Alert.user_id == current_user.id
        ).first()
        
        if not alert:
            raise HTTPException(
                status_code=404,
                detail="Alert not found"
            )
        
        # Update acknowledgment timestamp
        alert.last_acknowledged = datetime.utcnow()
        db.commit()
        
        return {"status": "acknowledged", "timestamp": alert.last_acknowledged}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to acknowledge alert: {str(e)}"
        )


@router.get("/notifications/recent")
async def get_recent_notifications(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    is_read: Optional[bool] = Query(None, description="Filter by read status"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get recent alert notifications for the user"""
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # This would query actual alert notifications from a notifications table
        # For now, return triggered alerts as notifications
        query = db.query(Alert).filter(
            Alert.user_id == current_user.id,
            Alert.last_triggered.isnot(None),
            Alert.last_triggered >= since
        )
        
        notifications = query.order_by(Alert.last_triggered.desc()).all()
        
        return {
            "notifications": [
                {
                    "id": alert.id,
                    "title": f"{alert.alert_type.title()} Alert: {alert.asset_symbol}",
                    "message": f"Alert triggered for {alert.asset_symbol}",
                    "asset_symbol": alert.asset_symbol,
                    "alert_type": alert.alert_type,
                    "priority": "medium",  # Could be derived from alert type
                    "created_at": alert.last_triggered,
                    "is_read": alert.last_acknowledged is not None
                }
                for alert in notifications
            ],
            "total_count": len(notifications),
            "unread_count": sum(1 for alert in notifications if alert.last_acknowledged is None)
        }
        
    except Exception as e:
        logger.error(f"Failed to get notifications for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve notifications: {str(e)}"
        )


@router.get("/rules")
async def get_alert_rules():
    """Get all alert rules"""
    try:
        alerting_service = await get_alerting_service()
        rules = await alerting_service.get_alert_rules()
        return {
            "success": True,
            "data": rules,
            "message": "Alert rules retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alert rules: {str(e)}"
        )


@router.post("/rules")
async def create_alert_rule(rule: AlertRuleCreate):
    """Create a new alert rule"""
    try:
        alerting_service = await get_alerting_service()
        success = await alerting_service.create_custom_alert_rule(
            rule_id=rule.rule_id,
            name=rule.name,
            description=rule.description,
            alert_type=rule.alert_type,
            conditions=rule.conditions,
            priority=rule.priority,
            template=rule.template,
            cooldown_minutes=rule.cooldown_minutes,
            rate_limit_per_hour=rule.rate_limit_per_hour
        )
        
        if success:
            return {
                "success": True,
                "message": f"Alert rule '{rule.name}' created successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create alert rule"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alert rule: {str(e)}"
        )


@router.put("/rules/{rule_id}/status")
async def update_alert_rule_status(rule_id: str, is_active: bool):
    """Update alert rule active status"""
    try:
        alerting_service = await get_alerting_service()
        success = await alerting_service.update_alert_rule_status(rule_id, is_active)
        
        if success:
            return {
                "success": True,
                "message": f"Alert rule {rule_id} status updated to {'active' if is_active else 'inactive'}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert rule {rule_id} not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update alert rule status: {str(e)}"
        )


@router.delete("/rules/{rule_id}")
async def remove_alert_rule(rule_id: str):
    """Remove an alert rule"""
    try:
        alerting_service = await get_alerting_service()
        success = await alerting_service.remove_alert_rule(rule_id)
        
        if success:
            return {
                "success": True,
                "message": f"Alert rule {rule_id} removed successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert rule {rule_id} not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove alert rule: {str(e)}"
        )


@router.post("/process/market-impact")
async def process_market_impact_alert(request: MarketImpactAlertRequest):
    """Process market impact data and generate alerts"""
    try:
        alerting_service = await get_alerting_service()
        alerts = await alerting_service.process_market_impact_alert(
            asset_symbol=request.asset_symbol,
            impact_score=request.impact_score,
            confidence=request.confidence,
            reason=request.reason,
            user_id=request.user_id
        )
        
        return {
            "success": True,
            "data": {
                "alerts_triggered": len(alerts),
                "alerts": [
                    {
                        "rule_id": alert.rule_id,
                        "alert_type": alert.alert_type.value,
                        "priority": alert.priority.value,
                        "title": alert.title,
                        "message": alert.message,
                        "asset_symbol": alert.asset_symbol,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in alerts
                ]
            },
            "message": f"Processed market impact alert for {request.asset_symbol}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process market impact alert: {str(e)}"
        )


@router.post("/process/sentiment")
async def process_sentiment_alert(request: SentimentAlertRequest):
    """Process sentiment data and generate alerts"""
    try:
        alerting_service = await get_alerting_service()
        alerts = await alerting_service.process_sentiment_alert(
            asset_symbol=request.asset_symbol,
            sentiment_score=request.sentiment_score,
            sentiment_label=request.sentiment_label,
            news_title=request.news_title,
            user_id=request.user_id
        )
        
        return {
            "success": True,
            "data": {
                "alerts_triggered": len(alerts),
                "alerts": [
                    {
                        "rule_id": alert.rule_id,
                        "alert_type": alert.alert_type.value,
                        "priority": alert.priority.value,
                        "title": alert.title,
                        "message": alert.message,
                        "asset_symbol": alert.asset_symbol,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in alerts
                ]
            },
            "message": f"Processed sentiment alert for {request.asset_symbol}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process sentiment alert: {str(e)}"
        )


@router.post("/process/price-movement")
async def process_price_movement_alert(request: PriceMovementAlertRequest):
    """Process price movement data and generate alerts"""
    try:
        alerting_service = await get_alerting_service()
        alerts = await alerting_service.process_price_movement_alert(
            asset_symbol=request.asset_symbol,
            current_price=request.current_price,
            price_change_percent=request.price_change_percent,
            user_id=request.user_id
        )
        
        return {
            "success": True,
            "data": {
                "alerts_triggered": len(alerts),
                "alerts": [
                    {
                        "rule_id": alert.rule_id,
                        "alert_type": alert.alert_type.value,
                        "priority": alert.priority.value,
                        "title": alert.title,
                        "message": alert.message,
                        "asset_symbol": alert.asset_symbol,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in alerts
                ]
            },
            "message": f"Processed price movement alert for {request.asset_symbol}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process price movement alert: {str(e)}"
        )


@router.post("/test")
async def send_test_alert(request: TestAlertRequest):
    """Send a test alert to a specific user"""
    try:
        alerting_service = await get_alerting_service()
        success = await alerting_service.send_test_alert(
            user_id=request.user_id,
            alert_type=request.alert_type,
            title=request.title,
            message=request.message
        )
        
        if success:
            return {
                "success": True,
                "message": f"Test alert sent to user {request.user_id}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send test alert"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send test alert: {str(e)}"
        )


@router.get("/stats")
async def get_alert_stats():
    """Get alerting service statistics"""
    try:
        alerting_service = await get_alerting_service()
        stats = await alerting_service.get_alert_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Alert statistics retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alert statistics: {str(e)}"
        ) 