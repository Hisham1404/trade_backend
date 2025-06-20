"""
Prometheus metrics service for monitoring application performance.
"""

from typing import Dict, Optional, Any
import time
import functools
from contextlib import contextmanager

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
import psutil

from app.services.logging_service import get_logger

logger = get_logger(__name__)


class MetricsRegistry:
    """Centralized metrics registry for the application."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._setup_application_metrics()
        self._setup_scraper_metrics()
        self._setup_system_metrics()
        self._setup_error_metrics()
    
    def _setup_application_metrics(self) -> None:
        """Set up general application metrics."""
        # Application info
        self._metrics['app_info'] = Info(
            'trading_agent_info',
            'Trading Agent application information',
            registry=self.registry
        )
        self._metrics['app_info'].info({
            'version': '0.1.0',
            'name': 'trading-intelligence-agent'
        })
        
        # Request metrics
        self._metrics['http_requests_total'] = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self._metrics['http_request_duration'] = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Application state
        self._metrics['app_status'] = Enum(
            'trading_agent_status',
            'Current status of the trading agent',
            states=['starting', 'running', 'stopping', 'error'],
            registry=self.registry
        )
        self._metrics['app_status'].state('starting')
        
        # Active connections
        self._metrics['active_connections'] = Gauge(
            'active_connections_total',
            'Number of active connections',
            registry=self.registry
        )
    
    def _setup_scraper_metrics(self) -> None:
        """Set up scraper-specific metrics."""
        # Scraping operations
        self._metrics['scraper_operations_total'] = Counter(
            'scraper_operations_total',
            'Total scraping operations',
            ['source_name', 'status'],
            registry=self.registry
        )
        
        self._metrics['scraper_duration'] = Histogram(
            'scraper_duration_seconds',
            'Time spent scraping each source',
            ['source_name'],
            registry=self.registry
        )
        
        self._metrics['items_scraped_total'] = Counter(
            'items_scraped_total',
            'Total items scraped',
            ['source_name', 'item_type'],
            registry=self.registry
        )
        
        self._metrics['validation_results_total'] = Counter(
            'validation_results_total',
            'Validation results for scraped items',
            ['source_name', 'validation_status'],
            registry=self.registry
        )
        
        # Data quality metrics
        self._metrics['data_quality_score'] = Histogram(
            'data_quality_score',
            'Data quality scores for scraped content',
            ['source_name'],
            registry=self.registry
        )
        
        # Scraper health
        self._metrics['scraper_last_success'] = Gauge(
            'scraper_last_success_timestamp',
            'Timestamp of last successful scrape',
            ['source_name'],
            registry=self.registry
        )
        
        self._metrics['scraper_errors_total'] = Counter(
            'scraper_errors_total',
            'Total scraper errors',
            ['source_name', 'error_type'],
            registry=self.registry
        )
    
    def _setup_system_metrics(self) -> None:
        """Set up system resource metrics."""
        # CPU usage
        self._metrics['cpu_usage_percent'] = Gauge(
            'cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        # Memory usage
        self._metrics['memory_usage_bytes'] = Gauge(
            'memory_usage_bytes',
            'Current memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        # Disk usage
        self._metrics['disk_usage_bytes'] = Gauge(
            'disk_usage_bytes',
            'Current disk usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        # Database metrics
        self._metrics['db_connections_active'] = Gauge(
            'database_connections_active',
            'Number of active database connections',
            registry=self.registry
        )
        
        self._metrics['db_operations_total'] = Counter(
            'database_operations_total',
            'Total database operations',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self._metrics['db_operation_duration'] = Histogram(
            'database_operation_duration_seconds',
            'Database operation duration in seconds',
            ['operation', 'table'],
            registry=self.registry
        )
    
    def _setup_error_metrics(self) -> None:
        """Set up error and alerting metrics."""
        self._metrics['errors_total'] = Counter(
            'errors_total',
            'Total errors by type and severity',
            ['error_type', 'severity', 'component'],
            registry=self.registry
        )
        
        self._metrics['alerts_triggered_total'] = Counter(
            'alerts_triggered_total',
            'Total alerts triggered',
            ['alert_type', 'severity'],
            registry=self.registry
        )
        
        self._metrics['recovery_operations_total'] = Counter(
            'recovery_operations_total',
            'Total recovery operations attempted',
            ['component', 'recovery_type', 'success'],
            registry=self.registry
        )
    
    def get_metric(self, name: str):
        """Get a metric by name."""
        return self._metrics.get(name)
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self._metrics['cpu_usage_percent'].set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self._metrics['memory_usage_bytes'].labels('used').set(memory.used)
            self._metrics['memory_usage_bytes'].labels('available').set(memory.available)
            self._metrics['memory_usage_bytes'].labels('total').set(memory.total)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self._metrics['disk_usage_bytes'].labels('used').set(disk.used)
            self._metrics['disk_usage_bytes'].labels('free').set(disk.free)
            self._metrics['disk_usage_bytes'].labels('total').set(disk.total)
            
        except Exception as e:
            logger.error("Failed to update system metrics", error=str(e))
    
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics output."""
        self.update_system_metrics()
        return generate_latest(self.registry).decode('utf-8')


# Global metrics registry instance
metrics_registry = MetricsRegistry()


# Decorator functions for common metric patterns
def track_requests(endpoint: str):
    """Decorator to track HTTP request metrics."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = "200"
            method = "GET"  # Default, can be enhanced
            
            try:
                # Extract method from request if available
                if args and hasattr(args[0], 'method'):
                    method = args[0].method
                
                result = await func(*args, **kwargs)
                
                # Extract status code from response if available
                if hasattr(result, 'status_code'):
                    status_code = str(result.status_code)
                
                return result
                
            except Exception as e:
                status_code = "500"
                logger.error("Request failed", endpoint=endpoint, error=str(e))
                raise
            finally:
                duration = time.time() - start_time
                
                # Update metrics
                metrics_registry.get_metric('http_requests_total').labels(
                    method=method, endpoint=endpoint, status_code=status_code
                ).inc()
                
                metrics_registry.get_metric('http_request_duration').labels(
                    method=method, endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator


def track_scraper_operation(source_name: str):
    """Decorator to track scraper operation metrics."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                
                # Update last success timestamp
                metrics_registry.get_metric('scraper_last_success').labels(
                    source_name=source_name
                ).set(time.time())
                
                return result
                
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                
                # Track scraper error
                metrics_registry.get_metric('scraper_errors_total').labels(
                    source_name=source_name, error_type=error_type
                ).inc()
                
                logger.error("Scraper operation failed", 
                           source_name=source_name, error=str(e))
                raise
            finally:
                duration = time.time() - start_time
                
                # Update operation metrics
                metrics_registry.get_metric('scraper_operations_total').labels(
                    source_name=source_name, status=status
                ).inc()
                
                metrics_registry.get_metric('scraper_duration').labels(
                    source_name=source_name
                ).observe(duration)
        
        return wrapper
    return decorator


@contextmanager
def track_db_operation(operation: str, table: str):
    """Context manager to track database operation metrics."""
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        
        metrics_registry.get_metric('db_operations_total').labels(
            operation=operation, table=table
        ).inc()
        
        metrics_registry.get_metric('db_operation_duration').labels(
            operation=operation, table=table
        ).observe(duration)


def create_metrics() -> MetricsRegistry:
    """Create and return the metrics registry."""
    return metrics_registry 