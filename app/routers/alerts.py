from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.database.connection import get_db
from app.services.alerting_service import get_alerting_service

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


@router.get("/")
async def get_alerts():
    """Get user alerts"""
    return {"message": "Alerts endpoint - to be implemented"}


@router.post("/")
async def create_alert():
    """Create new alert"""
    return {"message": "Create alert - to be implemented"}


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