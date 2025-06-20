"""
Base scraper framework for data collection.

Provides abstract base class and common functionality for all scrapers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Source, NewsItem


logger = logging.getLogger(__name__)


class ScrapingStatus(Enum):
    """Status of scraping operation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"


class ScrapingError(Exception):
    """Base exception for scraping operations."""
    
    def __init__(self, message: str, status: ScrapingStatus = ScrapingStatus.FAILED, details: Optional[Dict] = None):
        self.message = message
        self.status = status
        self.details = details or {}
        super().__init__(message)


@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    status: ScrapingStatus
    items_found: int = 0
    items_stored: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    

@dataclass 
class ScrapedItem:
    """A single scraped data item."""
    title: str
    content: str
    url: str
    published_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_specific_id: Optional[str] = None


class BaseScraper(ABC):
    """
    Abstract base class for all scrapers.
    
    Provides common functionality for:
    - HTTP session management
    - Rate limiting
    - Error handling and retry logic
    - Data validation and cleaning
    - Database operations
    """
    
    def __init__(self, source: Source, db_session: AsyncSession):
        self.source = source
        self.db_session = db_session
        self.last_checked = source.last_checked
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting settings
        self.rate_limit_delay = getattr(source, 'rate_limit_delay', 1.0)
        self.max_retries = 3
        self.retry_delay = 2.0
        
        # Session configuration
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.headers = {
            'User-Agent': 'Trading Agent Data Collector/1.0 (+https://example.com/bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._init_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()
        
    async def _init_session(self):
        """Initialize HTTP session with proper configuration."""
        connector = aiohttp.TCPConnector(
            limit=10,  # Connection pool limit
            limit_per_host=2,  # Limit per host to be respectful
            keepalive_timeout=300,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers=self.headers
        )
        
    async def _close_session(self):
        """Close HTTP session and cleanup."""
        if self.session:
            await self.session.close()
            # Allow some time for connections to close
            await asyncio.sleep(0.1)
            
    async def _make_request(
        self,
        url: str,
        method: str = "GET",
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make HTTP request with retry logic and error handling.
        
        Args:
            url: Target URL
            method: HTTP method
            **kwargs: Additional request parameters
            
        Returns:
            ClientResponse object
            
        Raises:
            ScrapingError: On persistent failures
        """
        if not self.session:
            await self._init_session()
            
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making {method} request to {url} (attempt {attempt + 1})")
                
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 429:  # Rate limited
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited by {self.source.url}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    elif response.status == 403:  # Blocked
                        raise ScrapingError(
                            f"Access blocked by {self.source.url}",
                            status=ScrapingStatus.BLOCKED,
                            details={'status_code': response.status, 'url': url}
                        )
                        
                    elif response.status >= 500:  # Server error
                        if attempt < self.max_retries - 1:
                            wait_time = self.retry_delay * (2 ** attempt)
                            logger.warning(f"Server error {response.status}, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise ScrapingError(
                                f"Server error: {response.status}",
                                status=ScrapingStatus.FAILED,
                                details={'status_code': response.status, 'url': url}
                            )
                            
                    elif not response.ok:
                        raise ScrapingError(
                            f"HTTP error: {response.status}",
                            status=ScrapingStatus.FAILED,
                            details={'status_code': response.status, 'url': url}
                        )
                        
                    # Success - return response
                    return response
                    
            except aiohttp.ClientError as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise ScrapingError(
                        f"Request failed after {self.max_retries} attempts: {e}",
                        status=ScrapingStatus.FAILED,
                        details={'error': str(e), 'url': url}
                    )
                    
            except Exception as e:
                logger.error(f"Unexpected error during request: {e}")
                raise ScrapingError(
                    f"Unexpected error: {e}",
                    status=ScrapingStatus.FAILED,
                    details={'error': str(e), 'url': url}
                )
                
        # Should not reach here
        raise ScrapingError("Max retries exceeded", status=ScrapingStatus.FAILED)
        
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        
        return text.strip()
        
    def _validate_item(self, item: ScrapedItem) -> bool:
        """
        Validate a scraped item.
        
        Args:
            item: Item to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation rules
        if not item.title or len(item.title.strip()) < 10:
            return False
            
        if not item.content or len(item.content.strip()) < 50:
            return False
            
        if not item.url or not item.url.startswith(('http://', 'https://')):
            return False
            
        return True
        
    async def _store_items(self, items: List[ScrapedItem]) -> int:
        """
        Store scraped items in the database.
        
        Args:
            items: List of items to store
            
        Returns:
            Number of items successfully stored
        """
        stored_count = 0
        
        for item in items:
            try:
                # Create new news item
                news_item = NewsItem(
                    source_id=self.source.id,
                    title=self._clean_text(item.title),
                    content=self._clean_text(item.content),
                    url=item.url,
                    published_at=item.published_at or datetime.utcnow(),
                    scraped_at=datetime.utcnow(),
                    metadata=item.metadata,
                    source_specific_id=item.source_specific_id
                )
                
                self.db_session.add(news_item)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to store item {item.url}: {e}")
                continue
                
        try:
            await self.db_session.commit()
            logger.info(f"Successfully stored {stored_count} items for source {self.source.name}")
        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"Failed to commit items: {e}")
            raise ScrapingError(f"Database commit failed: {e}")
            
        return stored_count
        
    async def _update_source_metadata(self, result: ScrapingResult):
        """Update source metadata after scraping."""
        try:
            self.source.last_checked = datetime.utcnow()
            if result.status == ScrapingStatus.SUCCESS:
                self.source.last_success = datetime.utcnow()
                
            await self.db_session.commit()
        except Exception as e:
            logger.error(f"Failed to update source metadata: {e}")
            await self.db_session.rollback()
            
    @abstractmethod
    async def fetch_data(self) -> Union[str, bytes]:
        """
        Fetch raw data from the source.
        
        Returns:
            Raw data (HTML, JSON, etc.)
        """
        pass
        
    @abstractmethod
    async def parse_data(self, raw_data: Union[str, bytes]) -> List[ScrapedItem]:
        """
        Parse raw data into structured items.
        
        Args:
            raw_data: Raw data to parse
            
        Returns:
            List of parsed items
        """
        pass
        
    async def process(self) -> ScrapingResult:
        """
        Main processing pipeline.
        
        Returns:
            ScrapingResult with operation details
        """
        start_time = datetime.utcnow()
        result = ScrapingResult(status=ScrapingStatus.FAILED)
        
        try:
            logger.info(f"Starting scraping for source: {self.source.name}")
            
            async with self:  # Use context manager for session management
                # Respect rate limiting
                if self.rate_limit_delay > 0:
                    await asyncio.sleep(self.rate_limit_delay)
                    
                # Fetch raw data
                raw_data = await self.fetch_data()
                
                # Parse data into structured items
                parsed_items = await self.parse_data(raw_data)
                result.items_found = len(parsed_items)
                
                # Validate items
                valid_items = [item for item in parsed_items if self._validate_item(item)]
                invalid_count = len(parsed_items) - len(valid_items)
                
                if invalid_count > 0:
                    result.warnings.append(f"Filtered out {invalid_count} invalid items")
                    
                # Store valid items
                if valid_items:
                    result.items_stored = await self._store_items(valid_items)
                    
                # Determine final status
                if result.items_stored == result.items_found and result.items_found > 0:
                    result.status = ScrapingStatus.SUCCESS
                elif result.items_stored > 0:
                    result.status = ScrapingStatus.PARTIAL
                else:
                    result.status = ScrapingStatus.FAILED
                    result.errors.append("No items were successfully stored")
                    
        except ScrapingError as e:
            result.status = e.status
            result.errors.append(e.message)
            result.metadata.update(e.details)
            logger.error(f"Scraping failed for {self.source.name}: {e.message}")
            
        except Exception as e:
            result.status = ScrapingStatus.FAILED
            result.errors.append(f"Unexpected error: {str(e)}")
            logger.error(f"Unexpected error scraping {self.source.name}: {e}", exc_info=True)
            
        finally:
            # Calculate duration
            result.duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Update source metadata
            await self._update_source_metadata(result)
            
            logger.info(
                f"Scraping completed for {self.source.name}: "
                f"Status={result.status.value}, "
                f"Found={result.items_found}, "
                f"Stored={result.items_stored}, "
                f"Duration={result.duration:.2f}s"
            )
            
        return result 