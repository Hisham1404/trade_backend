"""
Monitoring and health check endpoints.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Response, Query
from prometheus_client import CONTENT_TYPE_LATEST

from app.services.health_service import health_checker
from app.services.metrics_service import metrics_registry
from app.services.alerting_service import alerting_service, Alert, AlertRule, AlertSeverity
from app.services.monitoring_service import monitoring_service
from app.services.logging_service import get_logger
from app.auth import get_current_user
from app.models.user import User

logger = get_logger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# Health Check Endpoints
@router.get("/health")
async def health_check(
    components: Optional[str] = Query(None, description="Comma-separated list of components to check")
) -> Dict[str, Any]:
    """
    Comprehensive health check for all or specified components.
    
    Returns detailed health information for system components.
    """
    try:
        component_list = None
        if components:
            component_list = [comp.strip() for comp in components.split(",")]
        
        health = await health_checker.check_health(component_list)
        return health_checker.to_dict(health)
        
    except Exception as e:
        logger.error("Health check endpoint failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    """
    Simple liveness check - is the application running?
    
    Used by container orchestrators to determine if the container should be restarted.
    """
    is_alive = await health_checker.check_liveness()
    if is_alive:
        return {"status": "alive"}
    else:
        raise HTTPException(status_code=503, detail="Application not alive")


@router.get("/health/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check - is the application ready to serve requests?
    
    Used by container orchestrators to determine if traffic should be routed to this instance.
    """
    is_ready = await health_checker.check_readiness()
    if is_ready:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Application not ready")


# Metrics Endpoints
@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping.
    """
    try:
        metrics_data = metrics_registry.generate_metrics()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error("Metrics endpoint failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


@router.get("/metrics/summary")
async def metrics_summary(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get a summary of key metrics in JSON format.
    
    Requires authentication for detailed metrics access.
    """
    try:
        # This would typically query the metrics registry for key values
        # For now, we'll return a basic summary
        summary = {
            "timestamp": monitoring_service.last_health_check.isoformat() if monitoring_service.last_health_check else None,
            "application": {
                "status": "running" if monitoring_service.is_running else "stopped",
                "uptime_seconds": health_checker.start_time,
            },
            "system": {
                # These would be fetched from actual metrics
                "cpu_usage_percent": 0.0,
                "memory_usage_percent": 0.0,
                "disk_usage_percent": 0.0,
            },
            "scrapers": {
                "active": 0,  # Would be queried from scraper manager
                "errors_last_hour": 0,  # Would be calculated from metrics
            },
            "alerts": {
                "active": len(alerting_service.get_active_alerts()),
                "rules": len(alerting_service.alert_rules),
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error("Metrics summary failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get metrics summary")


# Alert Management Endpoints
@router.get("/alerts")
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by alert status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get current alerts, optionally filtered by status or severity.
    """
    try:
        alerts = alerting_service.get_active_alerts()
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a.status.value == status]
        
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]
        
        return [
            {
                "id": alert.id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "component": alert.component,
                "metric_name": alert.metric_name,
                "threshold_value": alert.threshold_value,
                "current_value": alert.current_value,
                "created_at": alert.created_at.isoformat(),
                "updated_at": alert.updated_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "details": alert.details
            }
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error("Get alerts failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get alerts")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Acknowledge an active alert.
    """
    try:
        success = await alerting_service.acknowledge_alert(alert_id)
        if success:
            logger.info("Alert acknowledged", alert_id=alert_id, user_id=current_user.id)
            return {"status": "acknowledged", "alert_id": alert_id}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Acknowledge alert failed", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Resolve an active alert.
    """
    try:
        success = await alerting_service.resolve_alert(alert_id)
        if success:
            logger.info("Alert resolved", alert_id=alert_id, user_id=current_user.id)
            return {"status": "resolved", "alert_id": alert_id}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Resolve alert failed", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@router.get("/alerts/history")
async def get_alert_history(
    limit: int = Query(100, description="Maximum number of alerts to return"),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get alert history.
    """
    try:
        alerts = alerting_service.get_alert_history(limit)
        
        return [
            {
                "id": alert.id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "component": alert.component,
                "created_at": alert.created_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            }
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error("Get alert history failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get alert history")


# Monitoring Control Endpoints
@router.get("/status")
async def monitoring_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get monitoring service status and configuration.
    """
    try:
        return await monitoring_service.get_monitoring_status()
        
    except Exception as e:
        logger.error("Get monitoring status failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get monitoring status")


@router.post("/health/check")
async def trigger_health_check(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Trigger a manual health check.
    """
    try:
        return await monitoring_service.trigger_manual_health_check()
        
    except Exception as e:
        logger.error("Manual health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Manual health check failed")


@router.post("/alerts/check")
async def trigger_alert_check(
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Trigger a manual alert rule evaluation.
    """
    try:
        return await monitoring_service.trigger_manual_alert_check()
        
    except Exception as e:
        logger.error("Manual alert check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Manual alert check failed")


# Dashboard Data Endpoint
@router.get("/dashboard")
async def monitoring_dashboard(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive monitoring data for dashboard display.
    """
    try:
        # Get health status
        health = await health_checker.check_health()
        
        # Get active alerts
        active_alerts = alerting_service.get_active_alerts()
        
        # Get monitoring status
        monitoring_status = await monitoring_service.get_monitoring_status()
        
        # Combine into dashboard data
        dashboard_data = {
            "timestamp": health.timestamp.isoformat(),
            "overall_status": health.status.value,
            "uptime_seconds": health.uptime_seconds,
            "components": health_checker.to_dict(health)["components"],
            "alerts": {
                "active_count": len(active_alerts),
                "critical_count": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high_count": len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                "recent": [
                    {
                        "id": alert.id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "component": alert.component,
                        "created_at": alert.created_at.isoformat(),
                    }
                    for alert in sorted(active_alerts, key=lambda x: x.created_at, reverse=True)[:5]
                ]
            },
            "monitoring": monitoring_status,
            "system_summary": {
                # This would include key system metrics
                "services_healthy": len([c for c in health.components if c.status.value == "healthy"]),
                "services_total": len(health.components),
                "last_check": monitoring_status["last_health_check"],
            }
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error("Dashboard data failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get dashboard data") 