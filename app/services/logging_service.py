"""
Structured logging service using structlog.
Provides JSON logging for production and pretty printing for development.
"""

import sys
import logging
from typing import Any, Dict, Optional
from pathlib import Path

import structlog
import orjson
from rich.logging import RichHandler

from app.core.config import settings


class ORJSONRenderer:
    """Custom JSON renderer using orjson for better performance."""
    
    def __call__(self, logger: Any, method_name: str, event_dict: Dict[str, Any]) -> str:
        """Render log entry as JSON using orjson."""
        return orjson.dumps(event_dict).decode('utf-8')


def setup_logging() -> None:
    """
    Set up structured logging configuration.
    
    Uses pretty printing in development (when running in a terminal)
    and JSON output in production for log aggregation.
    """
    # Shared processors for all environments
    shared_processors = [
        # Add context variables from contextvars
        structlog.contextvars.merge_contextvars,
        # Filter by log level early
        structlog.stdlib.filter_by_level,
        # Add logger name and log level
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        # Format positional arguments
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Add ISO timestamp
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Add callsite info (filename, function, line number)
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        # Handle stack traces
        structlog.processors.StackInfoRenderer(),
    ]
    
    # Choose output format based on environment
    if sys.stderr.isatty() and settings.DEBUG:
        # Pretty printing for development
        processors = shared_processors + [
            # Pretty print exceptions with rich if available
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        
        # Set up rich handler for standard library logging
        logging.basicConfig(
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
            level=logging.INFO if not settings.DEBUG else logging.DEBUG,
        )
    else:
        # JSON output for production
        processors = shared_processors + [
            # Convert exceptions to structured format
            structlog.processors.dict_tracebacks,
            structlog.processors.UnicodeDecoder(),
            # Use orjson for fast JSON serialization
            ORJSONRenderer(),
        ]
        
        # Standard JSON logging for production
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.INFO,
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set log levels for various components
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    
    # Set our application to appropriate level
    app_level = logging.DEBUG if settings.DEBUG else logging.INFO
    logging.getLogger("app").setLevel(app_level)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name, defaults to the calling module's name
        
    Returns:
        A bound logger instance with structured logging capabilities
    """
    if name is None:
        # Get the calling module's name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            name = 'unknown'
    
    return structlog.get_logger(name)


def bind_request_context(request_id: str, endpoint: str, method: str, 
                        user_id: Optional[str] = None) -> None:
    """
    Bind request context variables that will be included in all log entries.
    
    Args:
        request_id: Unique request identifier
        endpoint: API endpoint being accessed
        method: HTTP method
        user_id: User ID if authenticated
    """
    context = {
        "request_id": request_id,
        "endpoint": endpoint,
        "method": method,
    }
    if user_id:
        context["user_id"] = user_id
        
    structlog.contextvars.bind_contextvars(**context)


def clear_request_context() -> None:
    """Clear all request context variables."""
    structlog.contextvars.clear_contextvars()


# Create a module-level logger for this service
logger = structlog.get_logger(__name__) 