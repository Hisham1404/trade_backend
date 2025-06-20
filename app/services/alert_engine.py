"""
Alert Generation Logic Engine
Configurable alert rules with priority classification, templating, deduplication, and rate limiting
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json

from app.websockets.manager import ConnectionManager

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(Enum):
    """Alert types"""
    MARKET_IMPACT = "market_impact"
    PRICE_MOVEMENT = "price_movement"
    NEWS_SENTIMENT = "news_sentiment"
    VOLUME_SPIKE = "volume_spike"
    TECHNICAL_INDICATOR = "technical_indicator"
    PORTFOLIO_RISK = "portfolio_risk"
    SYSTEM_NOTIFICATION = "system_notification"


class ConditionOperator(Enum):
    """Operators for alert conditions"""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUAL_TO = "eq"
    NOT_EQUAL_TO = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"


@dataclass
class AlertCondition:
    """Single alert condition"""
    field: str
    operator: ConditionOperator
    value: Any
    weight: float = 1.0  # Weight for priority calculation


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    conditions: List[AlertCondition]
    priority_base: AlertPriority
    template: str
    is_active: bool = True
    cooldown_minutes: int = 30  # Minimum time between identical alerts
    escalation_levels: Optional[List[Dict[str, Any]]] = None
    rate_limit_per_hour: int = 10
    user_filters: Optional[Dict[str, Any]] = None  # Filter which users get this alert


@dataclass
class AlertEvent:
    """Alert event data"""
    rule_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    data: Dict[str, Any]
    asset_symbol: Optional[str] = None
    user_id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_deduplication_key(self) -> str:
        """Generate key for deduplication"""
        key_data = {
            'rule_id': self.rule_id,
            'asset_symbol': self.asset_symbol,
            'user_id': self.user_id,
            'message_hash': hashlib.md5(self.message.encode()).hexdigest()[:8]
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


@dataclass
class AlertDeduplicationEntry:
    """Entry for alert deduplication tracking"""
    alert_key: str
    last_sent: datetime
    count: int = 1


class AlertTemplateEngine:
    """Template engine for alert messages"""
    
    @staticmethod
    def render_template(template: str, data: Dict[str, Any]) -> str:
        """Render alert template with data"""
        try:
            # Simple template substitution
            result = template
            for key, value in data.items():
                placeholder = f"{{{key}}}"
                if placeholder in result:
                    result = result.replace(placeholder, str(value))
            return result
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            return f"Alert: {data.get('title', 'Unknown Alert')}"


class AlertRateLimiter:
    """Rate limiter for alerts"""
    
    def __init__(self):
        self.counters: Dict[str, List[datetime]] = defaultdict(list)
    
    def is_allowed(self, key: str, limit: int, window_hours: int = 1) -> bool:
        """Check if alert is allowed based on rate limit"""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=window_hours)
        
        # Clean old entries
        self.counters[key] = [ts for ts in self.counters[key] if ts > cutoff]
        
        # Check limit
        if len(self.counters[key]) >= limit:
            return False
        
        # Add current timestamp
        self.counters[key].append(now)
        return True


class AlertPriorityCalculator:
    """Calculate alert priority based on conditions and data"""
    
    @staticmethod
    def calculate_priority(
        base_priority: AlertPriority,
        conditions: List[AlertCondition],
        data: Dict[str, Any]
    ) -> AlertPriority:
        """Calculate final alert priority"""
        try:
            # Start with base priority score
            priority_scores = {
                AlertPriority.INFO: 1,
                AlertPriority.LOW: 2,
                AlertPriority.MEDIUM: 3,
                AlertPriority.HIGH: 4,
                AlertPriority.CRITICAL: 5
            }
            
            base_score = priority_scores[base_priority]
            total_weight = sum(c.weight for c in conditions)
            
            # Calculate weighted boost based on condition matches
            boost = 0
            for condition in conditions:
                if AlertPriorityCalculator._condition_matches(condition, data):
                    boost += condition.weight
            
            # Apply boost (max 50% increase)
            if total_weight > 0:
                boost_factor = min(0.5, boost / total_weight)
                final_score = base_score + (base_score * boost_factor)
            else:
                final_score = base_score
            
            # Convert back to priority
            if final_score >= 4.5:
                return AlertPriority.CRITICAL
            elif final_score >= 3.5:
                return AlertPriority.HIGH
            elif final_score >= 2.5:
                return AlertPriority.MEDIUM
            elif final_score >= 1.5:
                return AlertPriority.LOW
            else:
                return AlertPriority.INFO
                
        except Exception as e:
            logger.error(f"Priority calculation failed: {e}")
            return base_priority
    
    @staticmethod
    def _condition_matches(condition: AlertCondition, data: Dict[str, Any]) -> bool:
        """Check if a condition matches the data"""
        try:
            field_value = data.get(condition.field)
            if field_value is None:
                return False
            
            if condition.operator == ConditionOperator.GREATER_THAN:
                return float(field_value) > float(condition.value)
            elif condition.operator == ConditionOperator.LESS_THAN:
                return float(field_value) < float(condition.value)
            elif condition.operator == ConditionOperator.EQUAL_TO:
                return field_value == condition.value
            elif condition.operator == ConditionOperator.NOT_EQUAL_TO:
                return field_value != condition.value
            elif condition.operator == ConditionOperator.CONTAINS:
                return str(condition.value).lower() in str(field_value).lower()
            elif condition.operator == ConditionOperator.NOT_CONTAINS:
                return str(condition.value).lower() not in str(field_value).lower()
            elif condition.operator == ConditionOperator.IN:
                return field_value in condition.value
            elif condition.operator == ConditionOperator.NOT_IN:
                return field_value not in condition.value
            elif condition.operator == ConditionOperator.BETWEEN:
                if isinstance(condition.value, (list, tuple)) and len(condition.value) == 2:
                    return condition.value[0] <= float(field_value) <= condition.value[1]
            
            return False
        except Exception:
            return False


class AlertEscalationManager:
    """Manage alert escalation workflows"""
    
    def __init__(self):
        self.escalations: Dict[str, Dict[str, Any]] = {}
    
    async def process_escalation(self, alert_event: AlertEvent, rule: AlertRule) -> None:
        """Process alert escalation if configured"""
        if not rule.escalation_levels:
            return
        
        alert_key = alert_event.get_deduplication_key()
        now = datetime.utcnow()
        
        # Track escalation state
        if alert_key not in self.escalations:
            self.escalations[alert_key] = {
                'level': 0,
                'last_escalated': now,
                'count': 1
            }
        else:
            self.escalations[alert_key]['count'] += 1
        
        escalation = self.escalations[alert_key]
        
        # Check if escalation is needed
        for level_config in rule.escalation_levels:
            level = level_config.get('level', 0)
            delay_minutes = level_config.get('delay_minutes', 60)
            threshold = level_config.get('threshold', 3)
            
            if (escalation['level'] < level and 
                escalation['count'] >= threshold and
                now - escalation['last_escalated'] >= timedelta(minutes=delay_minutes)):
                
                # Escalate
                await self._execute_escalation(alert_event, level_config)
                escalation['level'] = level
                escalation['last_escalated'] = now
                break


    async def _execute_escalation(self, alert_event: AlertEvent, level_config: Dict[str, Any]) -> None:
        """Execute escalation action"""
        action = level_config.get('action', 'notify')
        
        if action == 'notify':
            # Send escalated notification
            escalated_event = AlertEvent(
                rule_id=alert_event.rule_id,
                alert_type=alert_event.alert_type,
                priority=AlertPriority.CRITICAL,  # Escalated alerts are critical
                title=f"ESCALATED: {alert_event.title}",
                message=f"Alert escalated to level {level_config.get('level', 0)}: {alert_event.message}",
                data=alert_event.data,
                asset_symbol=alert_event.asset_symbol,
                user_id=alert_event.user_id
            )
            
            # Send escalated alert (this would integrate with notification service)
            logger.warning(f"Alert escalated: {escalated_event.title}")


class AlertEngine:
    """Main alert generation engine"""
    
    def __init__(self, websocket_manager: Optional[ConnectionManager] = None):
        self.rules: Dict[str, AlertRule] = {}
        self.deduplication_cache: Dict[str, AlertDeduplicationEntry] = {}
        self.rate_limiter = AlertRateLimiter()
        self.template_engine = AlertTemplateEngine()
        self.priority_calculator = AlertPriorityCalculator()
        self.escalation_manager = AlertEscalationManager()
        self.websocket_manager = websocket_manager
        
        # Load default rules
        self._load_default_rules()
    
    def _load_default_rules(self) -> None:
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="market_impact_high",
                name="High Market Impact Alert",
                description="Alert for high market impact events",
                alert_type=AlertType.MARKET_IMPACT,
                conditions=[
                    AlertCondition("impact_score", ConditionOperator.GREATER_THAN, 7.0, weight=2.0),
                    AlertCondition("confidence", ConditionOperator.GREATER_THAN, 0.8, weight=1.5)
                ],
                priority_base=AlertPriority.HIGH,
                template="🚨 High Impact Alert: {asset_symbol} - Impact Score: {impact_score}/10 (Confidence: {confidence:.1%}). Reason: {reason}",
                cooldown_minutes=15,
                rate_limit_per_hour=5
            ),
            AlertRule(
                rule_id="sentiment_extreme",
                name="Extreme Sentiment Alert",
                description="Alert for extreme sentiment changes",
                alert_type=AlertType.NEWS_SENTIMENT,
                conditions=[
                    AlertCondition("sentiment_score", ConditionOperator.GREATER_THAN, 0.8, weight=1.5),
                    AlertCondition("sentiment_score", ConditionOperator.LESS_THAN, -0.8, weight=1.5)
                ],
                priority_base=AlertPriority.MEDIUM,
                template="📈 Sentiment Alert: {asset_symbol} - {sentiment_label} sentiment detected (Score: {sentiment_score:.2f}). News: {news_title}",
                cooldown_minutes=30,
                rate_limit_per_hour=8
            ),
            AlertRule(
                rule_id="price_movement_significant",
                name="Significant Price Movement",
                description="Alert for significant price movements",
                alert_type=AlertType.PRICE_MOVEMENT,
                conditions=[
                    AlertCondition("price_change_percent", ConditionOperator.GREATER_THAN, 5.0, weight=1.0),
                    AlertCondition("price_change_percent", ConditionOperator.LESS_THAN, -5.0, weight=1.0)
                ],
                priority_base=AlertPriority.MEDIUM,
                template="📊 Price Alert: {asset_symbol} moved {price_change_percent:+.2f}% to ${current_price:.2f}",
                cooldown_minutes=10,
                rate_limit_per_hour=15
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add or update an alert rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules"""
        return list(self.rules.values())
    
    async def evaluate_conditions(self, data: Dict[str, Any]) -> List[AlertEvent]:
        """Evaluate all rules against provided data"""
        triggered_alerts = []
        
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            # Check if any condition matches
            matches = False
            for condition in rule.conditions:
                if self.priority_calculator._condition_matches(condition, data):
                    matches = True
                    break
            
            if matches:
                # Calculate priority
                priority = self.priority_calculator.calculate_priority(
                    rule.priority_base, rule.conditions, data
                )
                
                # Render message
                message = self.template_engine.render_template(rule.template, data)
                
                # Create alert event
                alert_event = AlertEvent(
                    rule_id=rule.rule_id,
                    alert_type=rule.alert_type,
                    priority=priority,
                    title=f"{rule.name}",
                    message=message,
                    data=data,
                    asset_symbol=data.get('asset_symbol'),
                    user_id=data.get('user_id')
                )
                
                # Check deduplication and rate limiting
                if await self._should_send_alert(alert_event, rule):
                    triggered_alerts.append(alert_event)
        
        return triggered_alerts
    
    async def _should_send_alert(self, alert_event: AlertEvent, rule: AlertRule) -> bool:
        """Check if alert should be sent based on deduplication and rate limiting"""
        alert_key = alert_event.get_deduplication_key()
        now = datetime.utcnow()
        
        # Check deduplication
        if alert_key in self.deduplication_cache:
            entry = self.deduplication_cache[alert_key]
            if now - entry.last_sent < timedelta(minutes=rule.cooldown_minutes):
                logger.debug(f"Alert deduplicated: {alert_key}")
                return False
            else:
                entry.last_sent = now
                entry.count += 1
        else:
            self.deduplication_cache[alert_key] = AlertDeduplicationEntry(
                alert_key=alert_key,
                last_sent=now
            )
        
        # Check rate limiting
        rate_key = f"{rule.rule_id}:{alert_event.user_id or 'global'}"
        if not self.rate_limiter.is_allowed(rate_key, rule.rate_limit_per_hour):
            logger.debug(f"Alert rate limited: {rate_key}")
            return False
        
        return True
    
    async def send_alert(self, alert_event: AlertEvent) -> bool:
        """Send alert via various channels"""
        try:
            # Send via WebSocket if manager is available
            if self.websocket_manager and alert_event.user_id:
                alert_data = {
                    "type": "alert",
                    "alert_type": alert_event.alert_type.value,
                    "priority": alert_event.priority.value,
                    "title": alert_event.title,
                    "message": alert_event.message,
                    "asset_symbol": alert_event.asset_symbol,
                    "timestamp": alert_event.created_at.isoformat(),
                    "data": alert_event.data
                }
                
                await self.websocket_manager.send_alert(
                    str(alert_event.user_id), 
                    alert_data
                )
            
            # Store in database
            await self._store_alert_in_database(alert_event)
            
            # Process escalation
            rule = self.rules.get(alert_event.rule_id)
            if rule:
                await self.escalation_manager.process_escalation(alert_event, rule)
            
            logger.info(f"Alert sent: {alert_event.title} ({alert_event.priority.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    async def _store_alert_in_database(self, alert_event: AlertEvent) -> None:
        """Store alert in database"""
        try:
            # This would integrate with your database layer
            # For now, just log the alert
            logger.info(f"Storing alert in database: {alert_event.title}")
        except Exception as e:
            logger.error(f"Failed to store alert in database: {e}")
    
    async def process_market_impact_data(self, impact_data: Dict[str, Any]) -> List[AlertEvent]:
        """Process market impact data and generate alerts"""
        return await self.evaluate_conditions(impact_data)
    
    async def process_sentiment_data(self, sentiment_data: Dict[str, Any]) -> List[AlertEvent]:
        """Process sentiment data and generate alerts"""
        return await self.evaluate_conditions(sentiment_data)
    
    async def process_price_data(self, price_data: Dict[str, Any]) -> List[AlertEvent]:
        """Process price data and generate alerts"""
        return await self.evaluate_conditions(price_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert engine statistics"""
        return {
            "total_rules": len(self.rules),
            "active_rules": sum(1 for r in self.rules.values() if r.is_active),
            "deduplication_cache_size": len(self.deduplication_cache),
            "rate_limiter_entries": len(self.rate_limiter.counters),
            "escalation_tracking": len(self.escalation_manager.escalations)
        }


# Global alert engine instance
alert_engine: Optional[AlertEngine] = None


async def get_alert_engine() -> AlertEngine:
    """Get or create the global alert engine instance"""
    global alert_engine
    if alert_engine is None:
        # Import here to avoid circular imports
        try:
            from app.websockets.manager import get_connection_manager
            websocket_manager = await get_connection_manager()
        except ImportError:
            websocket_manager = None
        alert_engine = AlertEngine(websocket_manager)
    return alert_engine


async def shutdown_alert_engine() -> None:
    """Shutdown the alert engine"""
    global alert_engine
    if alert_engine:
        logger.info("Alert engine shutdown complete")
        alert_engine = None 