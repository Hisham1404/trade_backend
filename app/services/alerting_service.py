"""
Alerting service for sending notifications when thresholds are exceeded.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from app.services.logging_service import get_logger
from app.services.metrics_service import metrics_registry

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Alert information."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    component: str
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    threshold: float
    operator: str  # >, <, >=, <=, ==
    severity: AlertSeverity
    component: str
    description: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes
    last_triggered: Optional[datetime] = None


class AlertingService:
    """Service for managing alerts and notifications."""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_channels: List[Callable] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Set up default alert rules."""
        default_rules = [
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage_percent",
                threshold=85.0,
                operator=">=",
                severity=AlertSeverity.HIGH,
                component="system",
                description="Memory usage is above 85%"
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                threshold=80.0,
                operator=">=",
                severity=AlertSeverity.HIGH,
                component="system",
                description="CPU usage is above 80%"
            ),
            AlertRule(
                name="high_disk_usage",
                metric_name="disk_usage_percent",
                threshold=90.0,
                operator=">=",
                severity=AlertSeverity.CRITICAL,
                component="system",
                description="Disk usage is above 90%"
            ),
            AlertRule(
                name="scraper_errors",
                metric_name="scraper_errors_total",
                threshold=10.0,
                operator=">=",
                severity=AlertSeverity.MEDIUM,
                component="scrapers",
                description="High number of scraper errors"
            ),
            AlertRule(
                name="database_slow_response",
                metric_name="database_response_time_ms",
                threshold=1000.0,
                operator=">=",
                severity=AlertSeverity.MEDIUM,
                component="database",
                description="Database response time is slow"
            ),
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
    
    def add_notification_channel(self, channel: Callable[[Alert], None]) -> None:
        """Add a notification channel (function that receives alerts)."""
        self.notification_channels.append(channel)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add or update an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info("Alert rule added/updated", rule_name=rule.name)
    
    def remove_alert_rule(self, rule_name: str) -> None:
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info("Alert rule removed", rule_name=rule_name)
    
    async def check_alerts(self) -> List[Alert]:
        """Check all alert rules and trigger alerts if needed."""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if (rule.last_triggered and 
                datetime.utcnow() - rule.last_triggered < 
                timedelta(seconds=rule.cooldown_seconds)):
                continue
            
            try:
                alert = await self._evaluate_rule(rule)
                if alert:
                    triggered_alerts.append(alert)
                    rule.last_triggered = datetime.utcnow()
                    
            except Exception as e:
                logger.error("Failed to evaluate alert rule", 
                           rule_name=rule_name, error=str(e))
        
        return triggered_alerts
    
    async def _evaluate_rule(self, rule: AlertRule) -> Optional[Alert]:
        """Evaluate a single alert rule."""
        # Get current metric value
        current_value = await self._get_metric_value(rule.metric_name, rule.component)
        
        if current_value is None:
            return None
        
        # Evaluate condition
        condition_met = self._evaluate_condition(
            current_value, rule.operator, rule.threshold
        )
        
        if condition_met:
            alert_id = f"{rule.component}_{rule.name}_{int(time.time())}"
            
            # Check if similar alert already exists
            existing_alert = self._find_existing_alert(rule.component, rule.metric_name)
            if existing_alert:
                # Update existing alert
                existing_alert.current_value = current_value
                existing_alert.updated_at = datetime.utcnow()
                return None  # Don't create duplicate
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                title=f"{rule.component.title()} Alert: {rule.name}",
                description=rule.description,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                component=rule.component,
                metric_name=rule.metric_name,
                threshold_value=rule.threshold,
                current_value=current_value,
                details={
                    "rule_name": rule.name,
                    "operator": rule.operator,
                }
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Update metrics
            metrics_registry.get_metric('alerts_triggered_total').labels(
                alert_type=rule.name, severity=rule.severity.value
            ).inc()
            
            logger.warning("Alert triggered", 
                         alert_id=alert_id, 
                         component=rule.component,
                         metric=rule.metric_name,
                         current_value=current_value,
                         threshold=rule.threshold)
            
            return alert
        
        return None
    
    async def _get_metric_value(self, metric_name: str, component: str) -> Optional[float]:
        """Get current value of a metric."""
        try:
            # This is a simplified implementation
            # In a real system, you'd query your metrics store
            
            if metric_name == "memory_usage_percent":
                import psutil
                return psutil.virtual_memory().percent
            elif metric_name == "cpu_usage_percent":
                import psutil
                return psutil.cpu_percent(interval=1)
            elif metric_name == "disk_usage_percent":
                import psutil
                disk = psutil.disk_usage('/')
                return (disk.used / disk.total) * 100
            elif metric_name == "scraper_errors_total":
                # This would need to query the actual metric
                return 0.0  # Placeholder
            elif metric_name == "database_response_time_ms":
                # This would need to query the actual metric
                return 0.0  # Placeholder
            
            return None
            
        except Exception as e:
            logger.error("Failed to get metric value", 
                       metric_name=metric_name, error=str(e))
            return None
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        else:
            logger.error("Unknown operator", operator=operator)
            return False
    
    def _find_existing_alert(self, component: str, metric_name: str) -> Optional[Alert]:
        """Find existing active alert for the same component and metric."""
        for alert in self.active_alerts.values():
            if (alert.component == component and 
                alert.metric_name == metric_name and 
                alert.status == AlertStatus.ACTIVE):
                return alert
        return None
    
    async def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications through all configured channels."""
        for channel in self.notification_channels:
            try:
                if asyncio.iscoroutinefunction(channel):
                    await channel(alert)
                else:
                    channel(alert)
            except Exception as e:
                logger.error("Failed to send notification", 
                           alert_id=alert.id, error=str(e))
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            logger.info("Alert resolved", alert_id=alert_id)
            return True
        
        return False
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.updated_at = datetime.utcnow()
            
            logger.info("Alert acknowledged", alert_id=alert_id)
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.active_alerts.values() 
                if alert.status == AlertStatus.ACTIVE]
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get a specific alert by ID."""
        return self.active_alerts.get(alert_id)
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history (most recent first)."""
        alerts = list(self.active_alerts.values())
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        return alerts[:limit]


# Default notification channels
async def log_notification_channel(alert: Alert) -> None:
    """Log alert as a notification."""
    logger.warning(
        "ALERT NOTIFICATION",
        alert_id=alert.id,
        title=alert.title,
        description=alert.description,
        severity=alert.severity.value,
        component=alert.component,
        current_value=alert.current_value,
        threshold_value=alert.threshold_value
    )


def console_notification_channel(alert: Alert) -> None:
    """Print alert to console (for development)."""
    print(f"\n🚨 ALERT: {alert.title}")
    print(f"   Severity: {alert.severity.value.upper()}")
    print(f"   Component: {alert.component}")
    print(f"   Description: {alert.description}")
    if alert.current_value and alert.threshold_value:
        print(f"   Current: {alert.current_value}, Threshold: {alert.threshold_value}")
    print(f"   Time: {alert.created_at}")
    print()


# Global alerting service instance
alerting_service = AlertingService()

# Add default notification channels
alerting_service.add_notification_channel(log_notification_channel)
alerting_service.add_notification_channel(console_notification_channel) 