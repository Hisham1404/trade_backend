"""
Alert Service
High-level service for managing alerts and integrating with the alert engine
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Alert, User, Asset, NewsItem
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

logger = logging.getLogger(__name__)


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
        confidence: float, 
        reason: str, 
        user_id: Optional[int] = None
    ) -> List[AlertEvent]:
        """Process market impact data and generate alerts"""
        await self.initialize()
        
        impact_data = {
            'asset_symbol': asset_symbol,
            'impact_score': impact_score,
            'confidence': confidence,
            'reason': reason,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        triggered_alerts = await self.alert_engine.process_market_impact_data(impact_data)
        
        # Send triggered alerts
        for alert in triggered_alerts:
            await self.alert_engine.send_alert(alert)
        
        return triggered_alerts
    
    async def process_sentiment_alert(
        self, 
        asset_symbol: str, 
        sentiment_score: float, 
        sentiment_label: str, 
        news_title: str,
        user_id: Optional[int] = None
    ) -> List[AlertEvent]:
        """Process sentiment analysis data and generate alerts"""
        await self.initialize()
        
        sentiment_data = {
            'asset_symbol': asset_symbol,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'news_title': news_title,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        triggered_alerts = await self.alert_engine.process_sentiment_data(sentiment_data)
        
        # Send triggered alerts
        for alert in triggered_alerts:
            await self.alert_engine.send_alert(alert)
        
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
        
        price_data = {
            'asset_symbol': asset_symbol,
            'current_price': current_price,
            'price_change_percent': price_change_percent,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        triggered_alerts = await self.alert_engine.process_price_data(price_data)
        
        # Send triggered alerts
        for alert in triggered_alerts:
            await self.alert_engine.send_alert(alert)
        
        return triggered_alerts
    
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
            
            success = await self.alert_engine.send_alert(alert_event)
            logger.info(f"Test alert sent to user {user_id}: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to send test alert: {e}")
            return False
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting service statistics"""
        await self.initialize()
        
        engine_stats = self.alert_engine.get_stats()
        
        return {
            'service_initialized': self._initialized,
            'engine_stats': engine_stats,
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
        
        all_alerts = []
        
        # Process market impact alert if significant
        if impact_score >= 6.0 and confidence >= 0.6:
            impact_alerts = await self.process_market_impact_alert(
                asset_symbol=asset_symbol,
                impact_score=impact_score,
                confidence=confidence,
                reason=f"News impact analysis: {news_title[:100]}...",
                user_id=user_id
            )
            all_alerts.extend(impact_alerts)
        
        # Process sentiment alert if extreme
        if abs(sentiment_score) >= 0.7:
            sentiment_alerts = await self.process_sentiment_alert(
                asset_symbol=asset_symbol,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                news_title=news_title,
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