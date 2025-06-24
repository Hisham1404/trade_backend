"""
Security middleware for the Trading Intelligence Agent API.

Implements comprehensive security features including CORS,
security headers, rate limiting, and request monitoring.
"""

import time
import uuid
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.auth.rate_limiter import check_rate_limit, get_client_identifier
from app.core.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to all responses.
    
    Implements comprehensive security headers following OWASP guidelines
    for web application security.
    """
    
    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-eval'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        
        if self.debug:
            # Relax CSP for development
            csp_directives = [
                "default-src 'self' 'unsafe-inline' 'unsafe-eval'",
                "connect-src 'self' http: https: ws: wss:"
            ]
        
        # Security headers
        security_headers = {
            # Content Security Policy
            "Content-Security-Policy": "; ".join(csp_directives),
            
            # XSS Protection
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            
            # HTTPS and Transport Security
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions Policy
            "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=()",
            
            # Server identification
            "Server": "Trading-Agent-API/1.0",
            
            # Cache control for sensitive endpoints
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        # Add headers to response
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging and monitoring API requests.
    
    Tracks request timing, user agents, IP addresses,
    and response codes for security monitoring.
    """
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Log request details and timing."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Get client information
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        # Log request (in production, use proper logging)
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "status_code": response.status_code,
            "process_time": process_time,
            "timestamp": time.time()
        }
        
        # In production, send to logging service
        print(f"API Request: {log_data}")
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for applying rate limiting to API requests.
    
    Implements intelligent rate limiting with different limits
    for different types of endpoints and users.
    """
    
    def __init__(self, app, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/api/v1/auth/health",
            "/health"
        ]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply rate limiting to requests."""
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Skip rate limiting for internal health checks
        if request.headers.get("x-internal-health-check"):
            return await call_next(request)
        
        try:
            # Check rate limit
            await check_rate_limit(request)
        except HTTPException as e:
            # Return rate limit error response
            return Response(
                content=str(e.detail),
                status_code=e.status_code,
                headers=getattr(e, 'headers', {})
            )
        
        return await call_next(request)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API key authentication on specific endpoints.
    
    Provides alternative authentication method using API keys
    for programmatic access to the API.
    """
    
    def __init__(self, app, api_key_endpoints: Optional[list] = None):
        super().__init__(app)
        self.api_key_endpoints = api_key_endpoints or [
            "/api/v1/trading",
            "/api/v1/position-sizing",
            "/api/v1/analytics"
        ]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Check API key for specific endpoints."""
        # Check if this endpoint requires API key auth
        path = request.url.path
        requires_api_key = any(path.startswith(endpoint) for endpoint in self.api_key_endpoints)
        
        if requires_api_key:
            # Check for API key in headers
            api_key = request.headers.get("x-api-key")
            
            if api_key:
                # Validate API key (implement actual validation)
                # For now, just set a flag that API key was provided
                request.state.api_key_provided = True
                request.state.api_key = api_key
            else:
                # No API key provided - will fall back to JWT auth
                request.state.api_key_provided = False
        
        return await call_next(request)


def setup_cors_middleware(app, environment: str = "development"):
    """
    Setup CORS middleware with environment-specific configuration.
    
    Args:
        app: FastAPI application instance
        environment: Application environment (development/production)
    """
    if environment == "development":
        # Permissive CORS for development
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        # Restrictive CORS for production
        allowed_origins = getattr(settings, 'ALLOWED_ORIGINS', [
            "https://yourdomain.com",
            "https://api.yourdomain.com"
        ])
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=[
                "Authorization",
                "Content-Type", 
                "X-API-Key",
                "X-Requested-With"
            ],
        )


def setup_security_middleware(app, debug: bool = False):
    """
    Setup all security middleware for the application.
    
    Args:
        app: FastAPI application instance
        debug: Debug mode flag
    """
    # Request logging (add first to capture all requests)
    app.add_middleware(RequestLoggingMiddleware)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware)
    
    # API key authentication
    app.add_middleware(APIKeyAuthMiddleware)
    
    # Security headers (add last to ensure they're on all responses)
    app.add_middleware(SecurityHeadersMiddleware, debug=debug)


# Security utility functions
def validate_request_size(request: Request, max_size: int = 10 * 1024 * 1024):  # 10MB
    """
    Validate request body size to prevent DoS attacks.
    
    Args:
        request: FastAPI request object
        max_size: Maximum allowed request size in bytes
        
    Raises:
        HTTPException: If request is too large
    """
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        if content_length > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request body too large. Maximum size: {max_size} bytes"
            )


def sanitize_user_input(input_string: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_string: Raw user input
        
    Returns:
        Sanitized input string
    """
    if not isinstance(input_string, str):
        return str(input_string)
    
    # Basic sanitization - remove potentially dangerous characters
    dangerous_chars = ["<", ">", "&", '"', "'", "/", "\\"]
    sanitized = input_string
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")
    
    # Limit length to prevent buffer overflows
    return sanitized[:1000]


def generate_csrf_token() -> str:
    """
    Generate CSRF token for forms.
    
    Returns:
        Secure random CSRF token
    """
    return str(uuid.uuid4())


def validate_csrf_token(token: str, expected: str) -> bool:
    """
    Validate CSRF token.
    
    Args:
        token: Token from request
        expected: Expected token value
        
    Returns:
        True if tokens match, False otherwise
    """
    return token == expected and token is not None 