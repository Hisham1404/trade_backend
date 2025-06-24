"""
Base broker abstraction layer for unified broker API integration.

This module provides the foundation for integrating with multiple broker APIs
with common functionality for authentication, rate limiting, error handling,
and credential management.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from urllib.parse import urljoin
import aiohttp
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported broker types."""
    ZERODHA = "zerodha"
    UPSTOX = "upstox"
    ANGEL_ONE = "angel_one"
    FYERS = "fyers"
    ALICE_BLUE = "alice_blue"


class AuthType(Enum):
    """Authentication types supported by brokers."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    TOKEN_BASED = "token_based"
    SESSION_BASED = "session_based"


class ProductType(Enum):
    """Product types for trading."""
    MIS = "MIS"  # Margin Intraday Square Off
    CNC = "CNC"  # Cash and Carry
    NRML = "NRML"  # Normal
    CO = "CO"  # Cover Order
    BO = "BO"  # Bracket Order


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    burst_limit: int = 10
    backoff_factor: float = 2.0
    max_retries: int = 3


@dataclass
class BrokerCredentials:
    """Broker authentication credentials."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    session_token: Optional[str] = None
    user_id: Optional[str] = None
    password: Optional[str] = None
    pin: Optional[str] = None
    totp_key: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if credentials are expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at


@dataclass
class MarginInfo:
    """Margin information for trading."""
    total: float = 0.0
    available: float = 0.0
    utilized: float = 0.0
    span: float = 0.0
    exposure: float = 0.0
    premium: float = 0.0
    var_margin: float = 0.0
    bo_margin: float = 0.0
    cash_margin: float = 0.0
    leverage: float = 1.0
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionInfo:
    """Position information."""
    symbol: str
    exchange: str
    product_type: str
    quantity: int
    average_price: float
    market_value: float
    pnl: float
    realized_pnl: float
    unrealized_pnl: float
    day_change: float
    day_change_percent: float
    additional_info: Dict[str, Any] = field(default_factory=dict)


class BrokerError(Exception):
    """Base exception for broker-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 response_data: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.response_data = response_data


class AuthenticationError(BrokerError):
    """Authentication-related errors."""
    pass


class RateLimitError(BrokerError):
    """Rate limiting errors."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class InsufficientMarginError(BrokerError):
    """Insufficient margin errors."""
    pass


class InvalidOrderError(BrokerError):
    """Invalid order errors."""
    pass


class RateLimiter:
    """Async rate limiter for broker API calls."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests_log: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()
            
            # Clean old entries
            self.requests_log = [t for t in self.requests_log if now - t < 3600]  # Keep 1 hour
            
            # Check rate limits
            recent_requests = [t for t in self.requests_log if now - t < 60]  # Last minute
            if len(recent_requests) >= self.config.requests_per_minute:
                sleep_time = 60 - (now - min(recent_requests))
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
            
            # Check burst limit
            recent_burst = [t for t in self.requests_log if now - t < 1]  # Last second
            if len(recent_burst) >= self.config.burst_limit:
                await asyncio.sleep(1.0 / self.config.requests_per_second)
            
            self.requests_log.append(now)


class BaseBroker(ABC):
    """
    Abstract base class for all broker integrations.
    
    Provides common functionality for authentication, rate limiting,
    error handling, and connection management.
    """
    
    def __init__(self, 
                 credentials: BrokerCredentials,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize the broker client.
        
        Args:
            credentials: Authentication credentials
            rate_limit_config: Rate limiting configuration
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.credentials = credentials
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self._authenticated = False
        self._auth_lock = asyncio.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def broker_type(self) -> BrokerType:
        """Return the broker type."""
        pass
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL for the broker API."""
        pass
    
    @property
    @abstractmethod
    def auth_type(self) -> AuthType:
        """Return the authentication type used by this broker."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self) -> None:
        """Initialize the broker client."""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool limit
                limit_per_host=30,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._get_default_headers()
            )
        
        await self.authenticate()
    
    async def close(self) -> None:
        """Close the broker client and cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
        self._authenticated = False
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": f"TradingAgent/1.0 ({self.broker_type.value})",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    @abstractmethod
    async def authenticate(self) -> None:
        """Authenticate with the broker API."""
        pass
    
    async def _make_request(self, 
                          method: str, 
                          endpoint: str, 
                          data: Optional[Dict] = None,
                          params: Optional[Dict] = None,
                          headers: Optional[Dict] = None,
                          retry_count: int = 0) -> Dict[str, Any]:
        """
        Make an authenticated API request with rate limiting and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            data: Request body data
            params: Query parameters
            headers: Additional headers
            retry_count: Current retry count
            
        Returns:
            Response data as dictionary
            
        Raises:
            BrokerError: For various broker-related errors
        """
        if not self._authenticated:
            async with self._auth_lock:
                if not self._authenticated:
                    await self.authenticate()
        
        # Check if credentials are expired
        if self.credentials.is_expired():
            self.logger.warning("Credentials expired, re-authenticating")
            async with self._auth_lock:
                await self.authenticate()
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        url = urljoin(self.base_url, endpoint)
        request_headers = self._get_default_headers()
        if headers:
            request_headers.update(headers)
        
        # Add authentication headers
        auth_headers = await self._get_auth_headers()
        request_headers.update(auth_headers)
        
        try:
            self.logger.debug(f"Making {method} request to {url}")
            
            if method.upper() == "GET":
                async with self.session.get(url, params=params, headers=request_headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == "POST":
                async with self.session.post(url, json=data, params=params, headers=request_headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == "PUT":
                async with self.session.put(url, json=data, params=params, headers=request_headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == "DELETE":
                async with self.session.delete(url, params=params, headers=request_headers) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if retry_count < self.max_retries:
                wait_time = (2 ** retry_count) * self.rate_limiter.config.backoff_factor
                self.logger.warning(f"Request failed, retrying in {wait_time}s (attempt {retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
                return await self._make_request(method, endpoint, data, params, headers, retry_count + 1)
            else:
                raise BrokerError(f"Request failed after {self.max_retries} retries: {str(e)}")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle API response and errors."""
        try:
            response_data = await response.json()
        except (json.JSONDecodeError, aiohttp.ContentTypeError):
            response_data = {"text": await response.text()}
        
        if response.status == 200:
            return response_data
        elif response.status == 401:
            self._authenticated = False
            raise AuthenticationError("Authentication failed", response_data=response_data)
        elif response.status == 429:
            retry_after = response.headers.get("Retry-After", 60)
            raise RateLimitError("Rate limit exceeded", retry_after=int(retry_after), response_data=response_data)
        elif response.status >= 400:
            error_message = response_data.get("message", f"HTTP {response.status} error")
            error_code = response_data.get("error_code", str(response.status))
            raise BrokerError(error_message, error_code=error_code, response_data=response_data)
        
        return response_data
    
    @abstractmethod
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests."""
        pass
    
    # Abstract methods for broker functionality
    
    @abstractmethod
    async def get_margin_info(self, segment: Optional[str] = None) -> MarginInfo:
        """Get margin information."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[PositionInfo]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings."""
        pass
    
    @abstractmethod
    async def calculate_margin_requirement(self, 
                                         symbol: str, 
                                         quantity: int, 
                                         product_type: ProductType,
                                         exchange: str = "NSE") -> Dict[str, Any]:
        """Calculate margin requirement for a trade."""
        pass
    
    @abstractmethod
    async def get_lot_size(self, symbol: str, exchange: str = "NSE") -> int:
        """Get lot size for a symbol."""
        pass
    
    @abstractmethod
    async def validate_order(self, 
                           symbol: str, 
                           quantity: int, 
                           product_type: ProductType,
                           order_type: str = "MARKET",
                           exchange: str = "NSE") -> Dict[str, Any]:
        """Validate an order without placing it."""
        pass
    
    # Utility methods
    
    def is_authenticated(self) -> bool:
        """Check if the client is authenticated."""
        return self._authenticated and not self.credentials.is_expired()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the broker connection."""
        try:
            # Try to fetch margin info as a health check
            margin = await self.get_margin_info()
            return {
                "status": "healthy",
                "broker": self.broker_type.value,
                "authenticated": self.is_authenticated(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "broker": self.broker_type.value,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Context manager for broker clients
@asynccontextmanager
async def broker_client(broker: BaseBroker):
    """Context manager for broker clients."""
    try:
        await broker.initialize()
        yield broker
    finally:
        await broker.close() 