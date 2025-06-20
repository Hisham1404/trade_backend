"""
Specialized scrapers for financial news websites.
"""

import re
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import aiohttp

from .base import BaseScraper, ScrapedItem, ScrapingError, ScrapingStatus
from .manager import register_scraper

logger = logging.getLogger(__name__)


@register_scraper("economic_times")
class EconomicTimesScraper(BaseScraper):
    """Scraper for Economic Times financial news."""
    
    async def fetch_data(self) -> str:
        """Fetch HTML content from Economic Times."""
        async with self._make_request(self.source.url) as response:
            return await response.text()
    
    async def parse_data(self, raw_data: str) -> List[ScrapedItem]:
        """Parse Economic Times HTML into structured items."""
        soup = BeautifulSoup(raw_data, 'html.parser')
        items = []
        
        # ET uses various article containers
        article_selectors = [
            'article.story-card',
            '.story-list article',
            '.news-list article',
            '.story-container'
        ]
        
        articles = []
        for selector in article_selectors:
            found_articles = soup.select(selector)
            if found_articles:
                articles = found_articles
                break
        
        if not articles:
            # Fallback: look for links with specific patterns
            articles = soup.find_all('a', href=re.compile(r'/news/.*\.cms'))
        
        for article in articles[:20]:  # Limit to 20 articles per scrape
            try:
                item = await self._parse_article(article, soup)
                if item:
                    items.append(item)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue
                
        return items
    
    async def _parse_article(self, article_element, soup: BeautifulSoup) -> Optional[ScrapedItem]:
        """Parse individual article element."""
        # Extract title
        title_selectors = ['h2', 'h3', '.title', '.headline']
        title = None
        for selector in title_selectors:
            title_elem = article_element.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                break
        
        if not title:
            return None
            
        # Extract URL
        url = None
        if article_element.name == 'a':
            url = article_element.get('href')
        else:
            link_elem = article_element.find('a')
            if link_elem:
                url = link_elem.get('href')
        
        if not url:
            return None
            
        # Make URL absolute
        if url.startswith('/'):
            base_url = f"{urlparse(self.source.url).scheme}://{urlparse(self.source.url).netloc}"
            url = urljoin(base_url, url)
        
        # Extract summary/content
        content_selectors = ['.summary', '.excerpt', '.content', 'p']
        content = ""
        for selector in content_selectors:
            content_elem = article_element.select_one(selector)
            if content_elem:
                content = content_elem.get_text(strip=True)
                break
        
        # Extract published date
        published_at = None
        date_selectors = ['.time', '.date', '.published', 'time']
        for selector in date_selectors:
            date_elem = article_element.select_one(selector)
            if date_elem:
                date_text = date_elem.get('datetime') or date_elem.get_text(strip=True)
                published_at = self._parse_date(date_text)
                break
        
        return ScrapedItem(
            title=title,
            content=content,
            url=url,
            published_at=published_at,
            metadata={
                'source_type': 'economic_times',
                'section': self._extract_section(url)
            }
        )
    
    def _extract_section(self, url: str) -> str:
        """Extract section from URL."""
        try:
            path = urlparse(url).path
            parts = [p for p in path.split('/') if p and p != 'news']
            return parts[0] if parts else 'general'
        except:
            return 'general'
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from various formats."""
        if not date_text:
            return None
            
        # Common date patterns for ET
        patterns = [
            r'(\d{1,2})\s+(\w+)\s+(\d{4})',  # "12 Jan 2024"
            r'(\d{4})-(\d{2})-(\d{2})',      # "2024-01-12"
            r'(\d{1,2})/(\d{1,2})/(\d{4})'   # "12/01/2024"
        ]
        
        for pattern in patterns:
            try:
                match = re.search(pattern, date_text)
                if match:
                    # Try to parse based on pattern
                    return datetime.strptime(match.group(), pattern.replace(r'\s+', ' '))
            except:
                continue
                
        return None


@register_scraper("moneycontrol")
class MoneycontrolScraper(BaseScraper):
    """Scraper for Moneycontrol financial news."""
    
    async def fetch_data(self) -> str:
        """Fetch HTML content from Moneycontrol."""
        async with self._make_request(self.source.url) as response:
            return await response.text()
    
    async def parse_data(self, raw_data: str) -> List[ScrapedItem]:
        """Parse Moneycontrol HTML into structured items."""
        soup = BeautifulSoup(raw_data, 'html.parser')
        items = []
        
        # Moneycontrol article selectors
        article_selectors = [
            '.newslist li',
            '.news-listing li',
            '.list-item',
            'article'
        ]
        
        articles = []
        for selector in article_selectors:
            found_articles = soup.select(selector)
            if found_articles:
                articles = found_articles
                break
        
        for article in articles[:20]:  # Limit to 20 articles
            try:
                item = await self._parse_article(article, soup)
                if item:
                    items.append(item)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue
                
        return items
    
    async def _parse_article(self, article_element, soup: BeautifulSoup) -> Optional[ScrapedItem]:
        """Parse individual article element."""
        # Extract title
        title_selectors = ['h2', 'h3', '.title', '.headline', 'a']
        title = None
        for selector in title_selectors:
            title_elem = article_element.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title:  # Make sure we got actual text
                    break
        
        if not title:
            return None
            
        # Extract URL
        url = None
        if article_element.name == 'a':
            url = article_element.get('href')
        else:
            link_elem = article_element.find('a')
            if link_elem:
                url = link_elem.get('href')
        
        if not url:
            return None
            
        # Make URL absolute
        if url.startswith('/'):
            base_url = f"{urlparse(self.source.url).scheme}://{urlparse(self.source.url).netloc}"
            url = urljoin(base_url, url)
        
        # Extract summary/content
        content_selectors = ['.summary', '.excerpt', '.content', 'p', '.desc']
        content = ""
        for selector in content_selectors:
            content_elem = article_element.select_one(selector)
            if content_elem:
                content = content_elem.get_text(strip=True)
                if content:  # Make sure we got actual content
                    break
        
        # Extract published date
        published_at = None
        date_selectors = ['.time', '.date', '.published', 'time', '.news-date']
        for selector in date_selectors:
            date_elem = article_element.select_one(selector)
            if date_elem:
                date_text = date_elem.get('datetime') or date_elem.get_text(strip=True)
                published_at = self._parse_date(date_text)
                if published_at:
                    break
        
        return ScrapedItem(
            title=title,
            content=content,
            url=url,
            published_at=published_at,
            metadata={
                'source_type': 'moneycontrol',
                'section': self._extract_section(url)
            }
        )
    
    def _extract_section(self, url: str) -> str:
        """Extract section from URL."""
        try:
            path = urlparse(url).path
            parts = [p for p in path.split('/') if p and p != 'news']
            return parts[0] if parts else 'general'
        except:
            return 'general'
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from various formats."""
        if not date_text:
            return None
            
        # Common date patterns for Moneycontrol
        patterns = [
            r'(\d{1,2})\s+(\w+)\s+(\d{4})',  # "12 Jan 2024"
            r'(\d{4})-(\d{2})-(\d{2})',      # "2024-01-12"
            r'(\d{1,2})/(\d{1,2})/(\d{4})'   # "12/01/2024"
        ]
        
        for pattern in patterns:
            try:
                match = re.search(pattern, date_text)
                if match:
                    # Try to parse based on pattern
                    return datetime.strptime(match.group(), pattern.replace(r'\s+', ' '))
            except:
                continue
                
        return None 