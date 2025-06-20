"""
RSS and feed scrapers for financial news and regulatory updates.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from urllib.parse import urljoin, urlparse

import feedparser
import aiohttp
from bs4 import BeautifulSoup

from .base import BaseScraper, ScrapedItem, ScrapingError, ScrapingStatus
from .manager import register_scraper

logger = logging.getLogger(__name__)


@register_scraper("rss_feed")
class RSSFeedScraper(BaseScraper):
    """Generic RSS feed scraper."""
    
    async def fetch_data(self) -> str:
        """Fetch RSS feed content."""
        async with self._make_request(self.source.url) as response:
            return await response.text()
    
    async def parse_data(self, raw_data: str) -> List[ScrapedItem]:
        """Parse RSS feed into structured items."""
        try:
            # Parse the RSS feed
            feed = feedparser.parse(raw_data)
            items = []
            
            for entry in feed.entries[:20]:  # Limit to 20 items
                try:
                    item = await self._parse_entry(entry)
                    if item:
                        items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to parse RSS entry: {e}")
                    continue
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to parse RSS feed: {e}")
            return []
    
    async def _parse_entry(self, entry) -> Optional[ScrapedItem]:
        """Parse individual RSS entry."""
        # Extract title
        title = getattr(entry, 'title', '').strip()
        if not title:
            return None
        
        # Extract URL
        url = getattr(entry, 'link', '').strip()
        if not url:
            return None
        
        # Extract content/summary
        content = ""
        if hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'description'):
            content = entry.description
        elif hasattr(entry, 'content'):
            if isinstance(entry.content, list) and entry.content:
                content = entry.content[0].get('value', '')
            else:
                content = str(entry.content)
        
        # Clean HTML from content
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            content = soup.get_text(strip=True)
        
        # Extract published date
        published_at = None
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                published_at = datetime(*entry.published_parsed[:6])
            except:
                pass
        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            try:
                published_at = datetime(*entry.updated_parsed[:6])
            except:
                pass
        
        # Extract categories/tags
        categories = []
        if hasattr(entry, 'tags'):
            categories = [tag.get('term', '') for tag in entry.tags if tag.get('term')]
        
        return ScrapedItem(
            title=title,
            content=content,
            url=url,
            published_at=published_at,
            metadata={
                'source_type': 'rss_feed',
                'categories': categories,
                'feed_title': getattr(entry, 'feed', {}).get('title', '')
            }
        )


@register_scraper("atom_feed")
class AtomFeedScraper(RSSFeedScraper):
    """Atom feed scraper (inherits from RSS scraper as feedparser handles both)."""
    
    async def _parse_entry(self, entry) -> Optional[ScrapedItem]:
        """Parse individual Atom entry."""
        item = await super()._parse_entry(entry)
        if item:
            # Update metadata for Atom feeds
            item.metadata['source_type'] = 'atom_feed'
        return item


@register_scraper("economic_times_rss")
class EconomicTimesRSScraper(RSSFeedScraper):
    """Specialized scraper for Economic Times RSS feeds."""
    
    async def _parse_entry(self, entry) -> Optional[ScrapedItem]:
        """Parse Economic Times RSS entry with specific handling."""
        item = await super()._parse_entry(entry)
        if item:
            # Extract section from URL or categories
            section = self._extract_section(item.url)
            item.metadata.update({
                'source_type': 'economic_times_rss',
                'section': section,
                'publisher': 'Economic Times'
            })
        return item
    
    def _extract_section(self, url: str) -> str:
        """Extract section from ET URL."""
        try:
            path = urlparse(url).path
            parts = [p for p in path.split('/') if p and p not in ['news', 'et']]
            return parts[0] if parts else 'general'
        except:
            return 'general'


@register_scraper("moneycontrol_rss")
class MoneycontrolRSScraper(RSSFeedScraper):
    """Specialized scraper for Moneycontrol RSS feeds."""
    
    async def _parse_entry(self, entry) -> Optional[ScrapedItem]:
        """Parse Moneycontrol RSS entry with specific handling."""
        item = await super()._parse_entry(entry)
        if item:
            # Extract section from URL or categories
            section = self._extract_section(item.url)
            item.metadata.update({
                'source_type': 'moneycontrol_rss',
                'section': section,
                'publisher': 'Moneycontrol'
            })
        return item
    
    def _extract_section(self, url: str) -> str:
        """Extract section from Moneycontrol URL."""
        try:
            path = urlparse(url).path
            parts = [p for p in path.split('/') if p and p != 'news']
            return parts[0] if parts else 'general'
        except:
            return 'general'


@register_scraper("rbi_rss")
class RBIRSScraper(RSSFeedScraper):
    """Specialized scraper for RBI (Reserve Bank of India) RSS feeds."""
    
    async def _parse_entry(self, entry) -> Optional[ScrapedItem]:
        """Parse RBI RSS entry with specific handling."""
        item = await super()._parse_entry(entry)
        if item:
            # RBI specific metadata
            item.metadata.update({
                'source_type': 'rbi_rss',
                'publisher': 'Reserve Bank of India',
                'category': 'regulatory',
                'importance': 'high'  # RBI announcements are typically high importance
            })
        return item


@register_scraper("sebi_rss")
class SEBIRSScraper(RSSFeedScraper):
    """Specialized scraper for SEBI (Securities and Exchange Board of India) RSS feeds."""
    
    async def _parse_entry(self, entry) -> Optional[ScrapedItem]:
        """Parse SEBI RSS entry with specific handling."""
        item = await super()._parse_entry(entry)
        if item:
            # SEBI specific metadata
            item.metadata.update({
                'source_type': 'sebi_rss',
                'publisher': 'Securities and Exchange Board of India',
                'category': 'regulatory',
                'importance': 'high'  # SEBI announcements are typically high importance
            })
        return item 