"""
Alert Service
High-level service for managing alerts and integrating with the alert engine
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from app.database import get_db
from app.models import delivery as delivery_models
from app.services.alert_engine import (
    get_alert_engine, 
    AlertEngine, 
    AlertEvent, 
    AlertType, 
    AlertPriority,
    AlertRule,
    AlertCondition,
    ConditionOperator
)
from app.services.logging_service import get_logger

# Import delivery tracking service
try:
    from app.services.delivery_tracking_service import (
        get_delivery_tracking_service,
        DeliveryChannel,
        DeliveryResult
    )
    DELIVERY_TRACKING_AVAILABLE = True
except ImportError:
    DELIVERY_TRACKING_AVAILABLE = False

# Import push notification service with fallback
try:
    from app.services.push_notification_service import (
        get_push_notification_service,
        NotificationType,
        DeviceToken
    )
    from app.services.push_notification_service import DeliveryResult as PushDeliveryResult
    PUSH_NOTIFICATIONS_AVAILABLE = True
except ImportError:
    PUSH_NOTIFICATIONS_AVAILABLE = False

logger = get_logger(__name__)


class AlertingService:
    """
    High-level alerting service that coordinates between different data sources
    and the alert engine to generate and manage alerts.
    """
    
    def __init__(self):
        self.alert_engine: Optional[AlertEngine] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the alerting service"""
        if not self._initialized:
            self.alert_engine = await get_alert_engine()
            self._initialized = True
            logger.info("Alerting service initialized")
    
    async def process_market_impact_alert(
        self,
        asset_symbol: str,
        impact_score: float,
        impact_data: dict,
        user_id: Optional[int] = None
    ) -> List[AlertEvent]:
        """Process market impact data and generate alerts"""
        await self.initialize()
        assert self.alert_engine is not None
        
        impact_data = {
            'asset_symbol': asset_symbol,
            'impact_score': impact_score,
            'impact_data': impact_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        triggered_alerts = await self.alert_engine.evaluate_conditions(impact_data)
        
        # Send push notifications for triggered alerts
        if PUSH_NOTIFICATIONS_AVAILABLE and triggered_alerts:
            await self._send_push_notifications(triggered_alerts, user_id)
        
        return triggered_alerts
    
    async def process_sentiment_alert(
        self,
        asset_symbol: str,
        sentiment_score: float,
        sentiment_data: dict,
        user_id: Optional[int] = None
    ) -> List[AlertEvent]:
        """Process sentiment analysis data and generate alerts"""
        await self.initialize()
        assert self.alert_engine is not None
        
        sentiment_data = {
            'asset_symbol': asset_symbol,
            'sentiment_score': sentiment_score,
            'sentiment_data': sentiment_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        triggered_alerts = await self.alert_engine.evaluate_conditions(sentiment_data)
        
        # Send push notifications for triggered alerts
        if PUSH_NOTIFICATIONS_AVAILABLE and triggered_alerts:
            await self._send_push_notifications(triggered_alerts, user_id)
        
        return triggered_alerts
    
    async def process_price_movement_alert(
        self, 
        asset_symbol: str, 
        current_price: float, 
        price_change_percent: float,
        user_id: Optional[int] = None
    ) -> List[AlertEvent]:
        """Process price movement data and generate alerts"""
        await self.initialize()
        assert self.alert_engine is not None
        
        price_data = {
            'asset_symbol': asset_symbol,
            'current_price': current_price,
            'price_change_percent': price_change_percent,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        triggered_alerts = await self.alert_engine.process_price_data(price_data)
        
        # Send triggered alerts with delivery tracking
        for alert in triggered_alerts:
            await self._send_alert_with_tracking(alert)
        
        return triggered_alerts
    
    async def _send_alert_with_tracking(self, alert_event: AlertEvent) -> None:
        """Send an alert and create delivery tracking records"""
        assert self.alert_engine is not None
        
        # Create acknowledgment tracking first if available
        acknowledgment_id = None
        if alert_event.user_id:
            try:
                from app.services.acknowledgment_service import get_acknowledgment_service
                acknowledgment_service = get_acknowledgment_service()
                
                # Determine timeout based on alert priority
                timeout_minutes = 30 if alert_event.priority == AlertPriority.HIGH else 15
                
                acknowledgment = await acknowledgment_service.create_acknowledgment(
                    alert_id=hash(f"{alert_event.rule_id}_{alert_event.created_at}"),
                    user_id=str(alert_event.user_id),
                    timeout_minutes=timeout_minutes
                )
                
                if acknowledgment:
                    acknowledgment_id = str(acknowledgment.id)
                    logger.info(f"Created acknowledgment tracking for alert: {acknowledgment_id}")
                
            except ImportError:
                logger.debug("Acknowledgment service not available")
            except Exception as e:
                logger.error(f"Failed to create acknowledgment tracking: {e}")
        
        # Send the alert normally 
        await self.alert_engine.send_alert(alert_event)
        
        # Also send via WebSocket with acknowledgment info if available
        if acknowledgment_id and alert_event.user_id:
            try:
                from app.websockets.manager import connection_manager, create_alert_message
                
                # Create WebSocket alert message with acknowledgment info
                alert_message = create_alert_message(
                    alert_id=f"alert_{hash(f'{alert_event.rule_id}_{alert_event.created_at}')}",
                    user_id=str(alert_event.user_id),
                    alert_type=alert_event.alert_type.value,
                    title=alert_event.title,
                    message=alert_event.message,
                    severity=alert_event.priority.value,
                    data=alert_event.data,
                    acknowledgment_id=acknowledgment_id,
                    requires_acknowledgment=True,
                    timeout_minutes=timeout_minutes
                )
                
                # Send via WebSocket manager
                await connection_manager.send_alert(str(alert_event.user_id), alert_message)
                
            except ImportError:
                logger.debug("WebSocket manager not available")
            except Exception as e:
                logger.error(f"Failed to send WebSocket alert with acknowledgment: {e}")
        
        # Create delivery tracking records if available
        if DELIVERY_TRACKING_AVAILABLE and alert_event.user_id:
            await self._create_delivery_tracking_records(alert_event)
    
    async def _create_delivery_tracking_records(self, alert_event: AlertEvent) -> None:
        """Create delivery tracking records for an alert event"""
        try:
            delivery_service = get_delivery_tracking_service()
            
            # For now, simulate alert record creation
            # In practice, you would save the alert to database first
            alert_id = hash(f"{alert_event.rule_id}_{alert_event.created_at}")
            
            # Create tracking for WebSocket delivery
            await delivery_service.create_delivery_tracking(
                alert_id=alert_id,
                user_id=str(alert_event.user_id),
                channel=DeliveryChannel.WEBSOCKET,
                priority=2 if alert_event.priority == AlertPriority.HIGH else 1,
                metadata={
                    'rule_id': alert_event.rule_id,
                    'alert_type': alert_event.alert_type.value,
                    'title': alert_event.title,
                    'message': alert_event.message
                }
            )
            
            # Create tracking for push notification delivery
            if PUSH_NOTIFICATIONS_AVAILABLE and alert_event.user_id:
                try:
                    push_service = await get_push_notification_service()
                    user_tokens = push_service.device_tokens.get(str(alert_event.user_id), [])
                    
                    for token_info in user_tokens:
                        await delivery_service.create_delivery_tracking(
                            alert_id=alert_id,
                            user_id=str(alert_event.user_id),
                            channel=DeliveryChannel.PUSH_NOTIFICATION,
                            device_token=token_info.token,
                            priority=3 if alert_event.priority == AlertPriority.HIGH else 2,
                            metadata={
                                'rule_id': alert_event.rule_id,
                                'alert_type': alert_event.alert_type.value,
                                'platform': token_info.platform.value,
                                'title': alert_event.title,
                                'message': alert_event.message
                            }
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to create push notification delivery tracking: {e}")
            
            logger.info(f"Created delivery tracking records for alert {alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to create delivery tracking records: {e}")
    
    async def create_custom_alert_rule(
        self,
        rule_id: str,
        name: str,
        description: str,
        alert_type: str,
        conditions: List[Dict[str, Any]],
        priority: str,
        template: str,
        cooldown_minutes: int = 30,
        rate_limit_per_hour: int = 10
    ) -> bool:
        """Create a custom alert rule"""
        await self.initialize()
        assert self.alert_engine is not None
        
        try:
            # Convert string values to enums
            alert_type_enum = AlertType(alert_type)
            priority_enum = AlertPriority(priority)
            
            # Convert conditions
            alert_conditions = []
            for cond in conditions:
                operator = ConditionOperator(cond['operator'])
                condition = AlertCondition(
                    field=cond['field'],
                    operator=operator,
                    value=cond['value'],
                    weight=cond.get('weight', 1.0)
                )
                alert_conditions.append(condition)
            
            # Create rule
            rule = AlertRule(
                rule_id=rule_id,
                name=name,
                description=description,
                alert_type=alert_type_enum,
                conditions=alert_conditions,
                priority_base=priority_enum,
                template=template,
                cooldown_minutes=cooldown_minutes,
                rate_limit_per_hour=rate_limit_per_hour
            )
            
            self.alert_engine.add_rule(rule)
            logger.info(f"Created custom alert rule: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create custom alert rule: {e}")
            return False
    
    async def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get all alert rules"""
        await self.initialize()
        assert self.alert_engine is not None
        
        rules = self.alert_engine.get_rules()
        return [
            {
                'rule_id': rule.rule_id,
                'name': rule.name,
                'description': rule.description,
                'alert_type': rule.alert_type.value,
                'priority_base': rule.priority_base.value,
                'is_active': rule.is_active,
                'cooldown_minutes': rule.cooldown_minutes,
                'rate_limit_per_hour': rule.rate_limit_per_hour,
                'conditions': [
                    {
                        'field': cond.field,
                        'operator': cond.operator.value,
                        'value': cond.value,
                        'weight': cond.weight
                    }
                    for cond in rule.conditions
                ],
                'template': rule.template
            }
            for rule in rules
        ]
    
    async def update_alert_rule_status(self, rule_id: str, is_active: bool) -> bool:
        """Update alert rule active status"""
        await self.initialize()
        assert self.alert_engine is not None
        
        rules = self.alert_engine.get_rules()
        for rule in rules:
            if rule.rule_id == rule_id:
                rule.is_active = is_active
                logger.info(f"Updated alert rule {rule_id} status to: {is_active}")
                return True
        
        logger.warning(f"Alert rule not found: {rule_id}")
        return False
    
    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        await self.initialize()
        assert self.alert_engine is not None
        
        success = self.alert_engine.remove_rule(rule_id)
        if success:
            logger.info(f"Removed alert rule: {rule_id}")
        else:
            logger.warning(f"Failed to remove alert rule: {rule_id}")
        
        return success
    
    async def send_test_alert(
        self, 
        user_id: int, 
        alert_type: str = "system_notification",
        title: str = "Test Alert",
        message: str = "This is a test alert from the system"
    ) -> bool:
        """Send a test alert to a specific user"""
        await self.initialize()
        assert self.alert_engine is not None
        
        try:
            alert_event = AlertEvent(
                rule_id="test_alert",
                alert_type=AlertType(alert_type),
                priority=AlertPriority.LOW,
                title=title,
                message=message,
                data={'test': True},
                user_id=user_id
            )
            
            # Send with delivery tracking
            await self._send_alert_with_tracking(alert_event)
            
            logger.info(f"Test alert sent to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send test alert: {e}")
            return False
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting service statistics"""
        await self.initialize()
        assert self.alert_engine is not None
        
        engine_stats = self.alert_engine.get_stats()
        
        # Add delivery tracking stats if available
        delivery_stats = {}
        if DELIVERY_TRACKING_AVAILABLE:
            try:
                delivery_service = get_delivery_tracking_service()
                metrics = await delivery_service.get_delivery_metrics(hours=24)
                delivery_stats = {
                    'total_deliveries_24h': metrics.total_deliveries,
                    'successful_deliveries_24h': metrics.successful_deliveries,
                    'failed_deliveries_24h': metrics.failed_deliveries,
                    'success_rate_24h': metrics.success_rate,
                    'avg_latency_ms_24h': metrics.avg_latency_ms
                }
            except Exception as e:
                logger.error(f"Failed to get delivery stats: {e}")
                delivery_stats = {'error': 'Failed to retrieve delivery stats'}
        
        return {
            'service_initialized': self._initialized,
            'engine_stats': engine_stats,
            'delivery_stats': delivery_stats,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def process_news_impact_analysis(
        self,
        news_title: str,
        news_content: str,
        asset_symbol: str,
        impact_score: float,
        sentiment_score: float,
        sentiment_label: str,
        confidence: float,
        user_id: Optional[int] = None
    ) -> List[AlertEvent]:
        """Process comprehensive news impact analysis and generate appropriate alerts"""
        await self.initialize()
        assert self.alert_engine is not None
        
        all_alerts = []
        
        # Process market impact alert if significant
        if impact_score >= 6.0 and confidence >= 0.6:
            impact_alerts = await self.process_market_impact_alert(
                asset_symbol=asset_symbol,
                impact_score=impact_score,
                impact_data={},
                user_id=user_id
            )
            all_alerts.extend(impact_alerts)
        
        # Process sentiment alert if extreme
        if abs(sentiment_score) >= 0.7:
            sentiment_alerts = await self.process_sentiment_alert(
                asset_symbol=asset_symbol,
                sentiment_score=sentiment_score,
                sentiment_data={},
                user_id=user_id
            )
            all_alerts.extend(sentiment_alerts)
        
        return all_alerts
    
    async def monitor_portfolio_alerts(
        self,
        user_id: int,
        portfolio_assets: List[str],
        price_threshold: float = 5.0,
        sentiment_threshold: float = 0.8
    ) -> List[AlertEvent]:
        """Monitor a portfolio for alert conditions"""
        await self.initialize()
        assert self.alert_engine is not None
        
        all_alerts = []
        
        # This would integrate with real market data
        # For now, simulate some portfolio monitoring
        for asset_symbol in portfolio_assets:
            # Simulate checking various conditions
            # In practice, this would query real market data APIs
            
            # Example: Check for simulated price movements
            simulated_price_change = 0  # Would be real data
            if abs(simulated_price_change) >= price_threshold:
                price_alerts = await self.process_price_movement_alert(
                    asset_symbol=asset_symbol,
                    current_price=100.0,  # Would be real price
                    price_change_percent=simulated_price_change,
                    user_id=user_id
                )
                all_alerts.extend(price_alerts)
        
        return all_alerts
    
    async def check_alerts(self) -> List[AlertEvent]:
        """
        Periodically check for alert conditions based on system health and other metrics.
        This method is designed to be called by the monitoring service.
        """
        await self.initialize()
        assert self.alert_engine is not None
        all_triggered_alerts = []

        try:
            # Import locally to prevent circular dependency
            from app.services.health_service import health_checker
            
            # 1. Check system health
            health_status = await health_checker.check_health()
            
            for component in health_status.components:
                # Create a data packet for the alert engine
                health_data = {
                    "component_name": component.name,
                    "status": component.status,
                    "message": component.message,
                    "response_time_ms": component.response_time_ms,
                }

                # Evaluate conditions for this component data
                triggered = await self.alert_engine.evaluate_conditions(health_data)
                
                if triggered:
                    logger.info(f"Triggered {len(triggered)} alerts for component {component.name}")
                    all_triggered_alerts.extend(triggered)

            # 2. In a real application, you would also check other data sources:
            #    - Latest market data for all tracked assets
            #    - Recent news sentiment scores
            #    - Portfolio risk metrics

            # Send all triggered alerts with tracking
            for alert in all_triggered_alerts:
                await self._send_alert_with_tracking(alert)
                
            return all_triggered_alerts

        except Exception as e:
            logger.error(f"Error during periodic alert check: {e}")
            return []

    async def _send_push_notifications(
        self, 
        triggered_alerts: List[AlertEvent], 
        user_id: Optional[int] = None
    ):
        """Send push notifications for triggered alerts"""
        if not PUSH_NOTIFICATIONS_AVAILABLE:
            return
            
        try:
            push_service = await get_push_notification_service()
            delivery_service = None
            
            # Get delivery service if available
            if DELIVERY_TRACKING_AVAILABLE:
                delivery_service = get_delivery_tracking_service()
            
            for alert_event in triggered_alerts:
                # Map alert priority to notification type
                notification_type = self._map_alert_to_notification_type(alert_event)
                
                # Create notification payload
                payload = push_service.create_trading_alert_payload(
                    alert_type=notification_type,
                    asset_symbol=alert_event.asset_symbol or 'Unknown',
                    message=alert_event.message,
                    additional_data={
                        'rule_id': alert_event.rule_id,
                        'priority': alert_event.priority.value,
                        'timestamp': alert_event.created_at.isoformat(),
                        'data': alert_event.data
                    }
                )
                
                # Send to specific user or all registered users
                target_user_id = str(user_id or alert_event.user_id)
                if target_user_id:
                    user_devices = push_service.device_tokens.get(target_user_id, [])
                    for device in user_devices:
                        delivery_result = await push_service.send_notification_to_device(device, payload)
                        if delivery_service and delivery_result:
                            await self._record_push_delivery_attempt(
                                delivery_service, alert_event, target_user_id, delivery_result, device
                            )
                else:
                    # Send to all users with registered devices
                    for uid, devices in push_service.device_tokens.items():
                        for device in devices:
                            try:
                                delivery_result = await push_service.send_notification_to_device(device, payload)
                                if delivery_service and delivery_result:
                                    await self._record_push_delivery_attempt(
                                        delivery_service, alert_event, uid, delivery_result, device
                                    )
                            except Exception as e:
                                logger.error(f"Failed to send push notification to user {uid}: {e}")
                            
        except Exception as e:
            logger.error(f"Failed to send push notifications: {e}")
    
    async def _record_push_delivery_attempt(
        self, 
        delivery_service, 
        alert_event: AlertEvent, 
        user_id: str, 
        delivery_result: PushDeliveryResult,
        device: DeviceToken
    ):
        """Record a push notification delivery attempt"""
        try:
            db_session = next(get_db())
            # Find existing delivery tracking record for this token
            delivery_record = db_session.query(delivery_models.AlertDelivery).filter(
                delivery_models.AlertDelivery.user_id == user_id,
                delivery_models.AlertDelivery.device_token == device.token,
                delivery_models.AlertDelivery.channel == DeliveryChannel.PUSH_NOTIFICATION
            ).order_by(delivery_models.AlertDelivery.created_at.desc()).first()

            if delivery_record:
                # Convert PushDeliveryResult to DeliveryResult for the delivery tracking service
                tracking_result = DeliveryResult(
                    success=delivery_result.success,
                    error_message=delivery_result.error,
                    latency_ms=delivery_result.latency_ms,
                    delivery_context={
                        'platform': delivery_result.platform.value,
                        'token': delivery_result.token,
                        'message_id': delivery_result.message_id
                    }
                )
                
                await delivery_service.record_delivery_attempt(
                    delivery_id=delivery_record.id,
                    result=tracking_result,
                    attempt_number=delivery_record.attempts + 1
                )
            else:
                logger.warning(f"Could not find delivery tracking record for user {user_id} and token {device.token}")
            
        except Exception as e:
            logger.error(f"Failed to record push delivery attempt: {e}")
    
    def _map_alert_to_notification_type(self, alert_event: AlertEvent) -> NotificationType:
        """Map alert event to appropriate notification type"""
        rule_id = alert_event.rule_id.lower()
        
        if 'price' in rule_id:
            return NotificationType.PRICE_ALERT
        elif 'market_impact' in rule_id:
            return NotificationType.MARKET_IMPACT
        elif 'sentiment' in rule_id:
            return NotificationType.SENTIMENT_ALERT
        elif 'portfolio' in rule_id:
            return NotificationType.PORTFOLIO_UPDATE
        elif 'news' in rule_id:
            return NotificationType.NEWS_ALERT
        else:
            return NotificationType.SYSTEM_NOTIFICATION


# Global alerting service instance
alerting_service: Optional[AlertingService] = None


async def get_alerting_service() -> AlertingService:
    """Get or create the global alerting service instance"""
    global alerting_service
    if alerting_service is None:
        alerting_service = AlertingService()
    await alerting_service.initialize()
    return alerting_service


async def shutdown_alerting_service() -> None:
    """Shutdown the alerting service"""
    global alerting_service
    if alerting_service:
        logger.info("Alerting service shutdown complete")
        alerting_service = None 