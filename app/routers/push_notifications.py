"""
Push Notification Router for Trading Agent
Handles device token registration and notification sending
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from app.auth import get_current_user
from app.services.logging_service import get_logger

logger = get_logger(__name__)

# Import push notification service with fallback
try:
    from app.services.push_notification_service import (
        get_push_notification_service,
        PlatformType,
        NotificationType,
        NotificationPayload,
        PushNotificationService
    )
    PUSH_SERVICE_AVAILABLE = True
except ImportError:
    logger.warning("Push notification service not available")
    PUSH_SERVICE_AVAILABLE = False
    
    # Fallback classes for API documentation
    class PlatformType(Enum):
        IOS = "ios"
        ANDROID = "android"
        WEB = "web"
    
    class NotificationType(Enum):
        PRICE_ALERT = "price_alert"
        MARKET_IMPACT = "market_impact"
        SENTIMENT_ALERT = "sentiment_alert"
        PORTFOLIO_UPDATE = "portfolio_update"
        NEWS_ALERT = "news_alert"
        SYSTEM_NOTIFICATION = "system_notification"

router = APIRouter(prefix="/api/v1/push", tags=["Push Notifications"])
security = HTTPBearer()

# Pydantic models for API
class DeviceTokenRequest(BaseModel):
    """Request model for device token registration"""
    token: str = Field(..., description="Device push notification token")
    platform: PlatformType = Field(..., description="Device platform type")
    replace_existing: bool = Field(True, description="Replace existing tokens for this platform")

class DeviceTokenResponse(BaseModel):
    """Response model for device token operations"""
    success: bool
    message: str
    user_id: int

class NotificationRequest(BaseModel):
    """Request model for sending notifications"""
    title: str = Field(..., description="Notification title")
    body: str = Field(..., description="Notification body")
    notification_type: NotificationType = Field(..., description="Type of notification")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data payload")
    icon: Optional[str] = Field(None, description="Icon for notification")
    image: Optional[str] = Field(None, description="Image URL for notification")
    click_action: Optional[str] = Field(None, description="Action when notification is clicked")
    badge: Optional[int] = Field(None, description="Badge count for iOS")
    sound: str = Field("default", description="Sound to play")
    priority: str = Field("high", description="Notification priority")
    platforms: Optional[List[PlatformType]] = Field(None, description="Target platforms (all if not specified)")
    schedule_time: Optional[datetime] = Field(None, description="Schedule for future delivery")

class TradingAlertRequest(BaseModel):
    """Request model for trading-specific alerts"""
    alert_type: NotificationType
    asset_symbol: str
    message: str
    additional_data: Optional[Dict[str, Any]] = None
    target_users: Optional[List[int]] = Field(None, description="Specific user IDs to notify (all if not specified)")
    platforms: Optional[List[PlatformType]] = Field(None, description="Target platforms")

class DeliveryStatsResponse(BaseModel):
    """Response model for delivery statistics"""
    total_sent_24h: int
    successful_24h: int
    failed_24h: int
    success_rate: float
    active_devices: int
    active_users: int
    firebase_available: bool
    apns_available: bool

@router.post("/register-token", response_model=DeviceTokenResponse)
async def register_device_token(
    request: DeviceTokenRequest,
    current_user = Depends(get_current_user)
):
    """
    Register a device token for push notifications
    
    - **token**: The push notification token from the client
    - **platform**: Device platform (ios, android, web)
    - **replace_existing**: Whether to replace existing tokens for this platform
    """
    if not PUSH_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Push notification service not available")
    
    try:
        push_service = await get_push_notification_service()
        
        success = await push_service.register_device_token(
            user_id=current_user.id,
            token=request.token,
            platform=request.platform,
            replace_existing=request.replace_existing
        )
        
        if success:
            return DeviceTokenResponse(
                success=True,
                message=f"Device token registered successfully for {request.platform.value}",
                user_id=current_user.id
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to register device token"
            )
            
    except Exception as e:
        logger.error(f"Error registering device token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/unregister-token")
async def unregister_device_token(
    token: str,
    current_user = Depends(get_current_user)
):
    """
    Unregister a device token
    
    - **token**: The push notification token to remove
    """
    if not PUSH_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Push notification service not available")
    
    try:
        push_service = await get_push_notification_service()
        
        success = await push_service.unregister_device_token(
            user_id=current_user.id,
            token=token
        )
        
        if success:
            return {"success": True, "message": "Device token unregistered successfully"}
        else:
            return {"success": False, "message": "Token not found"}
            
    except Exception as e:
        logger.error(f"Error unregistering device token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send")
async def send_notification(
    request: NotificationRequest,
    current_user = Depends(get_current_user)
):
    """
    Send a push notification to current user's devices
    
    - **title**: Notification title
    - **body**: Notification message
    - **notification_type**: Type of notification
    - **data**: Additional data payload
    - **platforms**: Target platforms (optional)
    - **schedule_time**: Schedule for future delivery (optional)
    """
    if not PUSH_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Push notification service not available")
    
    try:
        push_service = await get_push_notification_service()
        
        # Create notification payload
        payload = NotificationPayload(
            title=request.title,
            body=request.body,
            notification_type=request.notification_type,
            data=request.data or {},
            icon=request.icon,
            image=request.image,
            click_action=request.click_action,
            badge=request.badge,
            sound=request.sound,
            priority=request.priority
        )
        
        # Send notification
        results = await push_service.send_notification(
            user_id=current_user.id,
            payload=payload,
            platforms=request.platforms,
            schedule_time=request.schedule_time
        )
        
        # Convert results to serializable format
        delivery_results = []
        for result in results:
            delivery_results.append({
                "token": result.token[:8] + "...",  # Partial token for privacy
                "status": result.status.value,
                "message_id": result.message_id,
                "error": result.error,
                "timestamp": result.timestamp.isoformat()
            })
        
        return {
            "success": True,
            "message": f"Notification sent to {len(results)} devices",
            "delivery_results": delivery_results
        }
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-trading-alert")
async def send_trading_alert(
    request: TradingAlertRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)  # Admin check could be added here
):
    """
    Send trading-specific alerts
    
    - **alert_type**: Type of trading alert
    - **asset_symbol**: Symbol of the asset
    - **message**: Alert message
    - **additional_data**: Extra data for the alert
    - **target_users**: Specific users to notify (all if not specified)
    - **platforms**: Target platforms
    """
    if not PUSH_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Push notification service not available")
    
    try:
        push_service = await get_push_notification_service()
        
        # Create trading alert payload
        payload = push_service.create_trading_alert_payload(
            alert_type=request.alert_type,
            asset_symbol=request.asset_symbol,
            message=request.message,
            additional_data=request.additional_data
        )
        
        # Determine target users
        target_users = request.target_users or [current_user.id]  # Default to current user
        
        # Send to each target user in background
        async def send_to_users():
            total_sent = 0
            for user_id in target_users:
                try:
                    results = await push_service.send_notification(
                        user_id=user_id,
                        payload=payload,
                        platforms=request.platforms
                    )
                    total_sent += len(results)
                except Exception as e:
                    logger.error(f"Failed to send alert to user {user_id}: {e}")
            
            logger.info(f"Trading alert sent to {total_sent} devices for {len(target_users)} users")
        
        background_tasks.add_task(send_to_users)
        
        return {
            "success": True,
            "message": f"Trading alert queued for {len(target_users)} users",
            "alert_type": request.alert_type.value,
            "asset_symbol": request.asset_symbol
        }
        
    except Exception as e:
        logger.error(f"Error sending trading alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test")
async def send_test_notification(
    current_user = Depends(get_current_user)
):
    """
    Send a test notification to verify push notification setup
    """
    if not PUSH_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Push notification service not available")
    
    try:
        push_service = await get_push_notification_service()
        
        # Create test payload
        payload = NotificationPayload(
            title="Test Notification",
            body="This is a test notification from Trading Agent",
            notification_type=NotificationType.SYSTEM_NOTIFICATION,
            data={"test": True, "timestamp": datetime.utcnow().isoformat()},
            icon="test_icon",
            sound="default",
            priority="high"
        )
        
        # Send test notification
        results = await push_service.send_notification(
            user_id=current_user.id,
            payload=payload
        )
        
        if not results:
            return {
                "success": False,
                "message": "No registered devices found for test notification"
            }
        
        successful = len([r for r in results if r.status.value == "sent"])
        
        return {
            "success": True,
            "message": f"Test notification sent to {len(results)} devices ({successful} successful)",
            "total_devices": len(results),
            "successful_deliveries": successful
        }
        
    except Exception as e:
        logger.error(f"Error sending test notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=DeliveryStatsResponse)
async def get_delivery_stats(
    current_user = Depends(get_current_user)
):
    """
    Get push notification delivery statistics
    """
    if not PUSH_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Push notification service not available")
    
    try:
        push_service = await get_push_notification_service()
        
        stats = await push_service.get_delivery_stats(user_id=current_user.id)
        
        return DeliveryStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting delivery stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/stats")
async def get_admin_delivery_stats(
    current_user = Depends(get_current_user)  # Add admin check here if needed
):
    """
    Get system-wide push notification delivery statistics (admin only)
    """
    if not PUSH_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Push notification service not available")
    
    try:
        push_service = await get_push_notification_service()
        
        stats = await push_service.get_delivery_stats()  # No user_id for system-wide stats
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting admin delivery stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_push_notification_status():
    """
    Get the status of push notification services
    """
    status = {
        "service_available": PUSH_SERVICE_AVAILABLE,
        "firebase_support": False,
        "apns_support": False
    }
    
    if PUSH_SERVICE_AVAILABLE:
        try:
            push_service = await get_push_notification_service()
            stats = await push_service.get_delivery_stats()
            status["firebase_support"] = stats.get("firebase_available", False)
            status["apns_support"] = stats.get("apns_available", False)
        except Exception as e:
            logger.error(f"Error checking push service status: {e}")
            status["error"] = str(e)
    
    return status 