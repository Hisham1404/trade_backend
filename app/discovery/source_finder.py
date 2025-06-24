"""
Intelligent Source Discovery System
"""

import aiohttp
import asyncio
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm import Session

from app.models import Source, Asset
from app.discovery.source_categorizer import SourceCategorizer


@dataclass
class DiscoveredSource:
    """Data class for discovered source information"""
    url: str
    title: str
    domain: str
    source_type: str
    reliability_score: float
    category: str
    content_quality: int
    financial_relevance: float
    has_rss: bool = False
    has_api: bool = False
    language: str = "en"


class ContentAnalyzer:
    """Analyzes web content for financial relevance and quality"""
    FINANCIAL_KEYWORDS = {
        'high_value': ['stock', 'market', 'nse', 'bse', 'sebi', 'rbi', 'trading', 'investment', 
                      'equity', 'mutual fund', 'ipo', 'earnings', 'financial results'],
        'medium_value': ['economy', 'banking', 'finance', 'business', 'shares', 'bonds',
                        'portfolio', 'dividend', 'analyst', 'forecast'],
        'low_value': ['money', 'price', 'growth', 'industry', 'company', 'revenue',
                     'profit', 'loss', 'news', 'update']
    }
    QUALITY_INDICATORS = {
        'positive': ['article', 'author', 'published', 'date', 'time', 'byline', 
                    'journalist', 'reporter', 'source', 'contact'],
        'negative': ['click here', 'buy now', 'limited time', 'urgent', 'spam',
                    'advertisement', 'popup', 'banner']
    }
    DOMAIN_PATTERNS = {
        'official': ['nseindia.com', 'bseindia.com', 'sebi.gov.in', 'rbi.org.in'],
        'verified_media': ['economictimes.indiatimes.com', 'moneycontrol.com', 'livemint.com'],
        'social': ['twitter.com', 'linkedin.com']
    }
    
    @classmethod
    def analyze_financial_relevance(cls, content: str, title: str = "") -> float:
        combined_text = f"{title} {content}".lower()
        score = sum(combined_text.count(kw) * w for w, kws in zip([3, 2, 1], cls.FINANCIAL_KEYWORDS.values()) for kw in kws)
        return min(score / 50.0, 1.0)
    
    @classmethod
    def analyze_content_quality(cls, soup: BeautifulSoup, content: str) -> int:
        score = 0
        text = content.lower()
        score += sum(1 for ind in cls.QUALITY_INDICATORS['positive'] if ind in text or soup.find(class_=ind))
        score -= sum(2 for ind in cls.QUALITY_INDICATORS['negative'] if ind in text)
        if soup.find('article'): score += 2
        if soup.find('time') or soup.find(class_='date'): score += 1
        if soup.find(class_='author'): score += 1
        if len(content) > 1000: score += 1
        return max(0, min(score, 10))
    
    @classmethod
    def determine_source_type(cls, domain: str, content: str) -> Tuple[str, float]:
        domain = domain.lower()
        for cat, doms in cls.DOMAIN_PATTERNS.items():
            if any(d in domain for d in doms):
                return cat, {'official': 9.0, 'verified_media': 7.5, 'social': 4.0}.get(cat, 5.0)
        if 'news' in content.lower(): return 'news', 6.0
        if 'blog' in content.lower(): return 'blog', 4.5
        return 'general', 5.0


class SourceDiscovery:
    """Main class for intelligent source discovery"""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.session_timeout = 10
        self.max_concurrent = 5
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.search_engines = {
            'google': 'https://www.google.com/search?q={query}&tbm=nws',
            'bing': 'https://www.bing.com/news/search?q={query}'
        }
        self.categorizer = SourceCategorizer()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if db_session is None:
            # Create a temporary in-memory SQLite session for testing convenience
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            engine = create_engine("sqlite:///:memory:")
            SessionLocal = sessionmaker(bind=engine)
            self.db_session = SessionLocal()
        else:
            self.db_session = db_session
    
    async def discover_sources_for_asset(self, asset_id: int, max_sources: int = 10) -> List[DiscoveredSource]:
        asset = self.db_session.query(Asset).get(asset_id)
        if not asset: return []
        queries = self._generate_search_queries(asset)
        urls = await self._discover_urls_from_searches(queries)
        existing = {s.url for s in self.db_session.query(Source.url).all()}
        new_urls = [u for u in urls if u not in existing]
        validated = await self._analyze_sources(new_urls, max_sources)
        await self._store_discovered_sources(validated)
        return validated
    
    def _generate_search_queries(self, asset: Asset) -> List[str]:
        return [f"{asset.symbol} stock news", f"{asset.name} company financial results"]
    
    async def _discover_urls_from_searches(self, queries: List[str]) -> Set[str]:
        urls = set()
        async with aiohttp.ClientSession() as s:
            tasks = [self._search(s, q) for q in queries]
            for res in await asyncio.gather(*tasks):
                urls.update(res)
        return urls
    
    async def _search(self, session, query):
        urls = set()
        for base in self.search_engines.values():
            try:
                async with session.get(base.format(query=quote_plus(query)), headers=self.headers) as resp:
                    if resp.status == 200:
                        soup = BeautifulSoup(await resp.text(), 'html.parser')
                        for a in soup.select('a[href]'):
                            href = a.get('href')
                            if href and href.startswith('http') and not any(d in href for d in ['google.com', 'bing.com']):
                                urls.add(href)
            except Exception as e:
                self.logger.warning(f"Search failed for {query}: {e}")
        return urls

    async def _analyze_sources(self, urls: List[str], max_sources: int) -> List[DiscoveredSource]:
        async with aiohttp.ClientSession() as s:
            tasks = [self._analyze_single(s, u) for u in urls[:max_sources*2]]
            results = [r for r in await asyncio.gather(*tasks) if r]
        results.sort(key=lambda x: x.reliability_score, reverse=True)
        return results[:max_sources]

    async def _analyze_single(self, session, url):
        try:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status != 200: return None
                html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                content = soup.get_text()
                if len(content) < 100: return None
                
                domain = urlparse(url).netloc
                title = soup.title.string if soup.title else "No Title"
                
                # Enhanced categorization using the new system
                category_result = await self.categorizer.categorize_source(url, title, content)
                
                # Fallback to old analysis for backward compatibility
                f_rev = ContentAnalyzer.analyze_financial_relevance(content, title)
                c_qual = ContentAnalyzer.analyze_content_quality(soup, content)
                
                # Use a more balanced scoring model
                score = (
                    (category_result.authority_score * 0.4) +  # 40% weight
                    (f_rev * 10 * 0.3) +                       # 30% weight (f_rev is 0-1)
                    (c_qual * 0.2) +                           # 20% weight (c_qual is 0-10)
                    (category_result.confidence * 10 * 0.1)    # 10% weight
                )

                # Adjusted filtering logic based on the new balanced score
                if score >= 4.5:
                    return DiscoveredSource(
                        url=url, 
                        title=title, 
                        domain=domain, 
                        source_type=category_result.domain_category,
                        reliability_score=score, 
                        category=category_result.primary_category,
                        content_quality=c_qual, 
                        financial_relevance=f_rev,
                        has_rss=self._check_rss_feed(soup),
                        has_api=self._check_api_availability(domain),
                        language='en'  # Default for now
                    )
        except Exception as e:
            self.logger.warning(f"Analysis failed for {url}: {e}")
        return None

    def _check_rss_feed(self, soup: BeautifulSoup) -> bool:
        """Check if the page has RSS/Atom feeds"""
        rss_links = soup.find_all('link', {'type': ['application/rss+xml', 'application/atom+xml']})
        return len(rss_links) > 0

    def _check_api_availability(self, domain: str) -> bool:
        """Check if the domain likely has API access"""
        api_indicators = ['api.', '/api/', 'developer.', 'dev.']
        return any(indicator in domain.lower() for indicator in api_indicators)

    async def _store_discovered_sources(self, sources: List[DiscoveredSource]):
        for s in sources:
            # Enhanced source creation with more metadata
            source_obj = Source(
                name=s.title[:100], 
                url=s.url, 
                category=s.category,
                type=s.source_type, 
                reliability_score=s.reliability_score,
                auto_discovered=True, 
                is_active=True, 
                check_frequency=30,
                language=s.language,
                last_checked=datetime.utcnow()
            )
            self.db_session.add(source_obj)
            
            # Log the discovery for monitoring
            self.logger.info(f"Discovered source: {s.url} - Category: {s.category} - "
                           f"Reliability: {s.reliability_score:.2f} - Type: {s.source_type}")
        
        try:
            self.db_session.commit()
            self.logger.info(f"Successfully stored {len(sources)} discovered sources")
        except Exception as e:
            self.logger.error(f"DB commit failed: {e}")
            self.db_session.rollback() 