"""
Health monitoring service for checking application component health.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import aiohttp
import psutil
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db, engine
from app.services.logging_service import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a single component."""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Overall system health information."""
    status: HealthStatus
    timestamp: datetime
    components: List[ComponentHealth]
    uptime_seconds: float
    version: str = "0.1.0"


class HealthChecker:
    """Service for monitoring application health."""
    
    def __init__(self):
        self.start_time = time.time()
        self._component_checkers = {
            "database": self._check_database,
            "memory": self._check_memory,
            "disk": self._check_disk,
            "cpu": self._check_cpu,
            "scrapers": self._check_scrapers,
        }
    
    async def check_health(self, 
                          components: Optional[List[str]] = None) -> SystemHealth:
        """
        Check health of specified components or all components.
        
        Args:
            components: List of component names to check, or None for all
            
        Returns:
            SystemHealth object with status of all checked components
        """
        if components is None:
            components = list(self._component_checkers.keys())
        
        # Check each component
        component_results = []
        for component_name in components:
            if component_name in self._component_checkers:
                try:
                    result = await self._component_checkers[component_name]()
                    component_results.append(result)
                except Exception as e:
                    logger.error("Health check failed", 
                               component=component_name, error=str(e))
                    component_results.append(ComponentHealth(
                        name=component_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(e)}",
                        last_check=datetime.utcnow()
                    ))
            else:
                component_results.append(ComponentHealth(
                    name=component_name,
                    status=HealthStatus.UNKNOWN,
                    message="Unknown component",
                    last_check=datetime.utcnow()
                ))
        
        # Determine overall status
        overall_status = self._determine_overall_status(component_results)
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            components=component_results,
            uptime_seconds=time.time() - self.start_time
        )
    
    def _determine_overall_status(self, 
                                components: List[ComponentHealth]) -> HealthStatus:
        """Determine overall system status from component statuses."""
        if not components:
            return HealthStatus.UNKNOWN
        
        # Count status types
        statuses = [comp.status for comp in components]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    async def _check_database(self) -> ComponentHealth:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            # Test database connection
            db: Session = next(get_db())
            
            # Simple query to test connectivity
            result = db.execute(text("SELECT 1")).fetchone()
            
            response_time = (time.time() - start_time) * 1000
            
            if result and result[0] == 1:
                status = HealthStatus.HEALTHY
                message = "Database connection successful"
                
                # Check response time
                if response_time > 1000:  # 1 second
                    status = HealthStatus.DEGRADED
                    message = f"Database slow response: {response_time:.1f}ms"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Database query failed"
                
            db.close()
            
            return ComponentHealth(
                name="database",
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                details={
                    "engine": str(engine.url).split('@')[0] + '@***',  # Hide credentials
                    "pool_size": engine.pool.size(),
                    "checked_out_connections": engine.pool.checkedout(),
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.utcnow()
            )
    
    async def _check_memory(self) -> ComponentHealth:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            # Determine status based on memory usage
            if usage_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {usage_percent:.1f}%"
            elif usage_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {usage_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {usage_percent:.1f}%"
            
            return ComponentHealth(
                name="memory",
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                details={
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "usage_percent": usage_percent
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}",
                last_check=datetime.utcnow()
            )
    
    async def _check_disk(self) -> ComponentHealth:
        """Check disk space usage."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            # Determine status based on disk usage
            if usage_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1f}%"
            elif usage_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Disk usage high: {usage_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage critical: {usage_percent:.1f}%"
            
            return ComponentHealth(
                name="disk",
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                details={
                    "total_bytes": disk.total,
                    "free_bytes": disk.free,
                    "used_bytes": disk.used,
                    "usage_percent": usage_percent
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {str(e)}",
                last_check=datetime.utcnow()
            )
    
    async def _check_cpu(self) -> ComponentHealth:
        """Check CPU usage."""
        try:
            # Get CPU usage over a short interval
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Determine status based on CPU usage
            if cpu_percent < 70:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            elif cpu_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            
            return ComponentHealth(
                name="cpu",
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                details={
                    "usage_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count(),
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="cpu",
                status=HealthStatus.UNHEALTHY,
                message=f"CPU check failed: {str(e)}",
                last_check=datetime.utcnow()
            )
    
    async def _check_scrapers(self) -> ComponentHealth:
        """Check scraper service health."""
        try:
            # Import here to avoid circular imports
            from app.scrapers.manager import scraper_manager
            
            # Check if scrapers are running
            is_running = scraper_manager.is_background_running()
            
            # Get scraper status
            if is_running:
                status = HealthStatus.HEALTHY
                message = "Scrapers are running"
            else:
                status = HealthStatus.DEGRADED
                message = "Scrapers are not running"
            
            # Get additional details
            registered_scrapers = len(scraper_manager.registry.scrapers)
            
            return ComponentHealth(
                name="scrapers",
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                details={
                    "background_running": is_running,
                    "registered_scrapers": registered_scrapers,
                    "scraper_types": list(scraper_manager.registry.scrapers.keys())
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="scrapers",
                status=HealthStatus.UNHEALTHY,
                message=f"Scraper check failed: {str(e)}",
                last_check=datetime.utcnow()
            )
    
    async def check_liveness(self) -> bool:
        """Simple liveness check - is the application running?"""
        return True
    
    async def check_readiness(self) -> bool:
        """
        Readiness check - is the application ready to serve requests?
        
        Returns True if critical components are healthy.
        """
        try:
            health = await self.check_health(["database"])
            critical_components = ["database"]
            
            for component in health.components:
                if (component.name in critical_components and 
                    component.status == HealthStatus.UNHEALTHY):
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Readiness check failed", error=str(e))
            return False
    
    def to_dict(self, health: SystemHealth) -> Dict[str, Any]:
        """Convert SystemHealth to dictionary format."""
        return {
            "status": health.status.value,
            "timestamp": health.timestamp.isoformat(),
            "uptime_seconds": health.uptime_seconds,
            "version": health.version,
            "components": [
                {
                    "name": comp.name,
                    "status": comp.status.value,
                    "message": comp.message,
                    "response_time_ms": comp.response_time_ms,
                    "last_check": comp.last_check.isoformat() if comp.last_check else None,
                    "details": comp.details
                }
                for comp in health.components
            ]
        }


# Global health checker instance
health_checker = HealthChecker() 