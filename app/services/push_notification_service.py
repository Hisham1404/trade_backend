from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import os
from app.core.config import settings

# Firebase Admin SDK for FCM (Android, Web)
try:
    import firebase_admin
    from firebase_admin import credentials, messaging
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# APNs2 for APNs (iOS) - Temporarily disabled
APNS_AVAILABLE = False

logger = logging.getLogger(__name__)

class PlatformType(Enum):
    """Device platform types"""
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"

class NotificationType(Enum):
    """Trading-specific notification types"""
    PRICE_ALERT = "price_alert"
    MARKET_IMPACT = "market_impact"
    SENTIMENT_ALERT = "sentiment_alert"
    PORTFOLIO_UPDATE = "portfolio_update"
    NEWS_ALERT = "news_alert"
    SYSTEM_NOTIFICATION = "system_notification"

@dataclass
class DeviceToken:
    """Represents a user's device token for a specific platform"""
    token: str
    platform: PlatformType
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None

@dataclass
class NotificationPayload:
    """Standardized notification payload"""
    title: str
    body: str
    data: Dict[str, Any] = field(default_factory=dict)
    notification_type: NotificationType = NotificationType.SYSTEM_NOTIFICATION

@dataclass
class DeliveryResult:
    """Result of a notification delivery"""
    success: bool
    platform: PlatformType
    token: str
    error: Optional[str] = None
    message_id: Optional[str] = None
    latency_ms: Optional[int] = None

class PushNotificationService:
    """Service for sending cross-platform push notifications"""

    def __init__(self):
        self.firebase_app: Optional[firebase_admin.App] = None
        # self.apns_client: Optional[APNsClient] = None # Temporarily disabled
        
        # In-memory device token storage (replace with database in production)
        self.device_tokens: Dict[str, List[DeviceToken]] = {}
        
        # Rate limiting (example: 60 notifications per minute)
        self.rate_limit = 60
        self.rate_limit_period = timedelta(minutes=1)
        self.notification_timestamps: List[datetime] = []
        
        # Delivery stats
        self.delivery_stats = {
            "total_sent": 0,
            "successful": 0,
            "failed": 0
        }
        self.last_stats_reset = datetime.utcnow()

    async def initialize(self):
        """Initialize Firebase and APNs clients"""
        
        # Initialize Firebase
        if FIREBASE_AVAILABLE:
            try:
                if settings.FIREBASE_CREDENTIALS_PATH and os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
                    cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
                    if not firebase_admin._apps:
                        self.firebase_app = firebase_admin.initialize_app(cred)
                    else:
                        self.firebase_app = firebase_admin.get_app()
                    logger.info("Firebase Admin SDK initialized successfully.")
                else:
                    logger.warning("Firebase credentials not found. FCM (Android/Web) will be disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        
        # Initialize APNs - Temporarily disabled
        
    async def register_token(self, user_id: str, token: str, platform: PlatformType):
        """Register a device token for a user"""
        if not user_id or not token:
            raise ValueError("User ID and token are required")
        
        if user_id not in self.device_tokens:
            self.device_tokens[user_id] = []
            
        # Remove existing token if it exists to prevent duplicates
        self.device_tokens[user_id] = [t for t in self.device_tokens[user_id] if t.token != token]
        
        new_token = DeviceToken(token=token, platform=platform)
        self.device_tokens[user_id].append(new_token)
        logger.info(f"Registered token for user {user_id} on {platform.value}")

    async def unregister_token(self, user_id: str, token: str):
        """Unregister a device token"""
        if user_id in self.device_tokens:
            self.device_tokens[user_id] = [t for t in self.device_tokens[user_id] if t.token != token]
            logger.info(f"Unregistered token for user {user_id}")

    def create_trading_alert_payload(
        self,
        alert_type: NotificationType,
        asset_symbol: str,
        message: str,
        additional_data: Dict[str, Any] = {}
    ) -> NotificationPayload:
        """Create a standardized payload for trading alerts"""
        title = f"{asset_symbol} - {alert_type.value.replace('_', ' ').title()}"
        
        data = {
            "asset_symbol": asset_symbol,
            "alert_type": alert_type.value,
            **additional_data
        }
        
        return NotificationPayload(
            title=title,
            body=message,
            data=data,
            notification_type=alert_type
        )

    async def send_notification(
        self, 
        user_id: str, 
        payload: NotificationPayload
    ) -> List[DeliveryResult]:
        """Send a notification to all registered devices for a user"""
        if user_id not in self.device_tokens:
            logger.warning(f"No device tokens found for user {user_id}")
            return []
            
        user_tokens = self.device_tokens[user_id]
        results = []
        
        for device in user_tokens:
            result = await self.send_notification_to_device(device, payload)
            results.append(result)
            
        return results

    async def send_notification_to_device(self, device: DeviceToken, payload: NotificationPayload) -> DeliveryResult:
        """Send a notification to a single device."""
        # Rate limiting check
        await self._check_rate_limit()
        
        start_time = datetime.utcnow()
        
        # APNs sending is temporarily disabled
        if device.platform in [PlatformType.ANDROID, PlatformType.WEB] and self.firebase_app:
            result = await self._send_fcm(device.token, payload)
        else:
            result = DeliveryResult(
                success=False,
                platform=device.platform,
                token=device.token,
                error=f"{device.platform.value} client not initialized or supported"
            )
        
        end_time = datetime.utcnow()
        result.latency_ms = int((end_time - start_time).total_seconds() * 1000)
        
        self._update_stats(result)
        return result

    async def _send_fcm(self, token: str, payload: NotificationPayload) -> DeliveryResult:
        """Send notification via Firebase Cloud Messaging"""
        message = messaging.Message(
            notification=messaging.Notification(
                title=payload.title,
                body=payload.body,
            ),
            data=payload.data,
            token=token,
        )
        
        try:
            response = messaging.send(message)
            return DeliveryResult(
                success=True,
                platform=PlatformType.WEB, # Assuming web/android for FCM
                token=token,
                message_id=response
            )
        except Exception as e:
            logger.error(f"FCM send failed for token {token}: {e}")
            return DeliveryResult(
                success=False,
                platform=PlatformType.WEB,
                token=token,
                error=str(e)
            )

    # _send_apns is temporarily disabled

    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.utcnow()
        
        # Prune old timestamps
        self.notification_timestamps = [
            ts for ts in self.notification_timestamps 
            if now - ts < self.rate_limit_period
        ]
        
        if len(self.notification_timestamps) >= self.rate_limit:
            wait_time = (self.notification_timestamps[0] + self.rate_limit_period) - now
            logger.warning(f"Rate limit exceeded. Waiting for {wait_time.total_seconds()} seconds.")
            await asyncio.sleep(wait_time.total_seconds())
        
        self.notification_timestamps.append(now)

    def _update_stats(self, result: DeliveryResult):
        """Update delivery statistics"""
        self.delivery_stats["total_sent"] += 1
        if result.success:
            self.delivery_stats["successful"] += 1
        else:
            self.delivery_stats["failed"] += 1

    async def get_stats(self) -> Dict[str, Any]:
        """Get current delivery statistics"""
        return {
            "since": self.last_stats_reset.isoformat(),
            **self.delivery_stats,
            "tokens_by_user": {
                user_id: len(tokens) 
                for user_id, tokens in self.device_tokens.items()
            }
        }

    async def close(self):
        """Clean up resources"""
        if self.firebase_app:
            try:
                firebase_admin.delete_app(self.firebase_app)
                logger.info("Firebase app deleted.")
            except Exception as e:
                logger.error(f"Failed to delete Firebase app: {e}")
        
        # apns2 client doesn't have a close method
        pass


# Singleton instance
_push_notification_service: Optional[PushNotificationService] = None

async def get_push_notification_service() -> PushNotificationService:
    """Get or create the singleton push notification service instance"""
    global _push_notification_service
    if _push_notification_service is None:
        _push_notification_service = PushNotificationService()
        await _push_notification_service.initialize()
    return _push_notification_service

async def shutdown_push_notification_service():
    """Shutdown the push notification service"""
    global _push_notification_service
    if _push_notification_service:
        await _push_notification_service.close()
        _push_notification_service = None
        logger.info("Push notification service shut down.") 