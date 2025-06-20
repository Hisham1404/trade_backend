"""
Central monitoring service that coordinates health checks, metrics, and alerting.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from app.services.logging_service import get_logger
from app.services.health_service import health_checker, HealthStatus
from app.services.metrics_service import metrics_registry
from app.services.alerting_service import alerting_service

logger = get_logger(__name__)


class MonitoringService:
    """Central service for coordinating all monitoring activities."""
    
    def __init__(self):
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.check_interval = 60  # seconds
        self.alert_check_interval = 30  # seconds
        self.last_health_check: Optional[datetime] = None
        self.last_alert_check: Optional[datetime] = None
        
    async def start_monitoring(self, 
                             health_check_interval: int = 60,
                             alert_check_interval: int = 30) -> None:
        """
        Start the monitoring service.
        
        Args:
            health_check_interval: How often to run health checks (seconds)
            alert_check_interval: How often to check alert rules (seconds)
        """
        if self.is_running:
            logger.warning("Monitoring service is already running")
            return
        
        self.check_interval = health_check_interval
        self.alert_check_interval = alert_check_interval
        self.is_running = True
        
        # Update application status
        app_status_metric = metrics_registry.get_metric('app_status')
        if app_status_metric:
            app_status_metric.state('running')
        
        # Start monitoring loop
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Monitoring service started", 
                   health_interval=health_check_interval,
                   alert_interval=alert_check_interval)
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring service."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Update application status
        app_status_metric = metrics_registry.get_metric('app_status')
        if app_status_metric:
            app_status_metric.state('stopping')
        
        # Cancel monitoring task
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
        
        logger.info("Monitoring service stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # Run health checks
                await self._run_health_checks()
                
                # Run alert checks
                await self._run_alert_checks()
                
                # Update system metrics
                self._update_system_metrics()
                
                # Sleep until next check, accounting for time spent in checks
                elapsed = time.time() - loop_start
                sleep_time = max(0, min(self.check_interval, self.alert_check_interval) - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error("Monitoring loop failed", error=str(e))
            # Update application status to error
            app_status_metric = metrics_registry.get_metric('app_status')
            if app_status_metric:
                app_status_metric.state('error')
            raise
    
    async def _run_health_checks(self) -> None:
        """Run health checks if it's time."""
        now = datetime.utcnow()
        
        if (self.last_health_check is None or 
            now - self.last_health_check >= timedelta(seconds=self.check_interval)):
            
            try:
                start_time = time.time()
                health = await health_checker.check_health()
                check_duration = time.time() - start_time
                
                self.last_health_check = now
                
                # Log health status
                logger.info("Health check completed", 
                          overall_status=health.status.value,
                          duration_ms=check_duration * 1000,
                          components_checked=len(health.components))
                
                # Update metrics based on health status
                self._update_health_metrics(health)
                
                # Trigger manual alerts for unhealthy components
                await self._check_health_alerts(health)
                
            except Exception as e:
                logger.error("Health check failed", error=str(e))
                
                # Record error in metrics
                metrics_registry.get_metric('errors_total').labels(
                    error_type="health_check_failed",
                    severity="medium",
                    component="monitoring"
                ).inc()
    
    async def _run_alert_checks(self) -> None:
        """Run alert rule evaluations if it's time."""
        now = datetime.utcnow()
        
        if (self.last_alert_check is None or 
            now - self.last_alert_check >= timedelta(seconds=self.alert_check_interval)):
            
            try:
                start_time = time.time()
                triggered_alerts = await alerting_service.check_alerts()
                check_duration = time.time() - start_time
                
                self.last_alert_check = now
                
                if triggered_alerts:
                    logger.info("Alert check completed", 
                              alerts_triggered=len(triggered_alerts),
                              duration_ms=check_duration * 1000)
                
            except Exception as e:
                logger.error("Alert check failed", error=str(e))
                
                # Record error in metrics
                metrics_registry.get_metric('errors_total').labels(
                    error_type="alert_check_failed",
                    severity="medium",
                    component="monitoring"
                ).inc()
    
    def _update_system_metrics(self) -> None:
        """Update system-level metrics."""
        try:
            # Update system resource metrics
            metrics_registry.update_system_metrics()
            
            # Update application metrics
            active_connections_metric = metrics_registry.get_metric('active_connections')
            if active_connections_metric:
                # This would be updated by actual connection tracking
                # For now, just a placeholder
                pass
                
        except Exception as e:
            logger.error("Failed to update system metrics", error=str(e))
    
    def _update_health_metrics(self, health) -> None:
        """Update metrics based on health check results."""
        try:
            # Create a simple health score (1.0 = all healthy, 0.0 = all unhealthy)
            total_components = len(health.components)
            if total_components == 0:
                return
            
            healthy_count = sum(1 for comp in health.components 
                              if comp.status == HealthStatus.HEALTHY)
            degraded_count = sum(1 for comp in health.components 
                               if comp.status == HealthStatus.DEGRADED)
            
            # Weight degraded as 0.5
            health_score = (healthy_count + degraded_count * 0.5) / total_components
            
            # This could be a custom metric if we add it
            logger.debug("Health score calculated", 
                        score=health_score, 
                        healthy=healthy_count,
                        degraded=degraded_count,
                        total=total_components)
            
        except Exception as e:
            logger.error("Failed to update health metrics", error=str(e))
    
    async def _check_health_alerts(self, health) -> None:
        """Check for health-based alerts that need to be triggered."""
        try:
            for component in health.components:
                if component.status == HealthStatus.UNHEALTHY:
                    # This could trigger specific health alerts
                    logger.warning("Component unhealthy", 
                                 component=component.name,
                                 message=component.message)
                    
                    # Record error in metrics
                    metrics_registry.get_metric('errors_total').labels(
                        error_type="component_unhealthy",
                        severity="high",
                        component=component.name
                    ).inc()
                    
        except Exception as e:
            logger.error("Failed to check health alerts", error=str(e))
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring service status."""
        return {
            "is_running": self.is_running,
            "check_interval": self.check_interval,
            "alert_check_interval": self.alert_check_interval,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_alert_check": self.last_alert_check.isoformat() if self.last_alert_check else None,
            "active_alerts": len(alerting_service.get_active_alerts()),
            "alert_rules": len(alerting_service.alert_rules),
        }
    
    async def trigger_manual_health_check(self) -> Dict[str, Any]:
        """Trigger a manual health check and return results."""
        logger.info("Manual health check triggered")
        
        try:
            health = await health_checker.check_health()
            self.last_health_check = datetime.utcnow()
            
            # Update metrics
            self._update_health_metrics(health)
            
            return health_checker.to_dict(health)
            
        except Exception as e:
            logger.error("Manual health check failed", error=str(e))
            raise
    
    async def trigger_manual_alert_check(self) -> List[Dict[str, Any]]:
        """Trigger a manual alert check and return any new alerts."""
        logger.info("Manual alert check triggered")
        
        try:
            triggered_alerts = await alerting_service.check_alerts()
            self.last_alert_check = datetime.utcnow()
            
            return [
                {
                    "id": alert.id,
                    "title": alert.title,
                    "description": alert.description,
                    "severity": alert.severity.value,
                    "component": alert.component,
                    "created_at": alert.created_at.isoformat(),
                }
                for alert in triggered_alerts
            ]
            
        except Exception as e:
            logger.error("Manual alert check failed", error=str(e))
            raise


# Global monitoring service instance
monitoring_service = MonitoringService() 