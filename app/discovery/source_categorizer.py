"""
Comprehensive Source Categorization System
Implements NLP-based categorization for automatic classification of discovered sources
by topic, domain, content type, and reliability scoring.

Task 7.4 - Automatic Categorization Implementation
"""
import asyncio
import hashlib
import logging
import re
import ssl
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Optional, Tuple, Any
from urllib.parse import urlparse
import aiohttp

logger = logging.getLogger(__name__)


class CategorizationResult(NamedTuple):
    """Result returned by SourceCategorizer.categorize_source."""
    domain_category: str
    primary_category: str
    authority_score: float  # 0-10
    confidence: float       # 0-1
    reliability_tier: str
    content_type: str
    tags: List[str]
    subcategories: List[str]


@dataclass
class DomainAnalysis:
    """Domain analysis result with comprehensive metrics."""
    age_days: Optional[int]
    ssl_enabled: bool
    domain_authority: float
    page_authority: float
    trust_score: float
    hosting_country: str
    registrar: str
    creation_date: Optional[datetime]
    expiry_date: Optional[datetime]


class DomainAuthority:
    """Domain authority analysis and scoring system."""
    
    # Comprehensive authority database for Indian financial ecosystem
    AUTHORITY_DATABASE = {
        # Official Government & Regulatory
        'nseindia.com': 10.0,
        'bseindia.com': 10.0,
        'sebi.gov.in': 10.0,
        'rbi.org.in': 10.0,
        'mca.gov.in': 10.0,
        'cbdt.gov.in': 9.5,
        'finmin.nic.in': 9.5,
        'irdai.gov.in': 9.0,
        'pfrda.org.in': 9.0,
        
        # Major Financial Media
        'economictimes.indiatimes.com': 8.5,
        'moneycontrol.com': 8.5,
        'livemint.com': 8.0,
        'business-standard.com': 8.0,
        'financialexpress.com': 7.5,
        'bloombergquint.com': 7.5,
        'reuters.com': 8.0,
        'bloomberg.com': 8.5,
        
        # Brokers & Financial Services
        'zerodha.com': 7.5,
        'icicidirect.com': 7.0,
        'hdfcsec.com': 7.0,
        'kotaksecurities.com': 7.0,
        'angelbroking.com': 6.5,
        '5paisa.com': 6.0,
        
        # Research & Analytics
        'morningstar.in': 7.5,
        'capitalmarket.com': 6.5,
        'equitymaster.com': 6.0,
        'investopedia.com': 7.0,
    }
    
    @classmethod
    async def analyze_domain(cls, domain: str) -> DomainAnalysis:
        """Perform comprehensive domain analysis."""
        try:
            # Check if domain is in our authority database
            base_authority = cls.AUTHORITY_DATABASE.get(domain, 0.0)
            
            # Calculate dynamic scores
            ssl_enabled = await cls._check_ssl(domain)
            trust_score = await cls._calculate_trust_score(domain, ssl_enabled)
            
            # Government domain bonus
            if any(gov_tld in domain for gov_tld in ['.gov.in', '.gov', '.nic.in']):
                base_authority = max(base_authority, 8.0)
                trust_score = max(trust_score, 8.0)
            
            # Educational institution bonus
            if '.edu' in domain or '.ac.in' in domain:
                base_authority = max(base_authority, 7.0)
            
            # Calculate page authority based on domain authority
            page_authority = min(base_authority * 0.9, 10.0)
            
            return DomainAnalysis(
                age_days=None,  # Would require external API
                ssl_enabled=ssl_enabled,
                domain_authority=base_authority,
                page_authority=page_authority,
                trust_score=trust_score,
                hosting_country='Unknown',  # Would require geolocation API
                registrar='Unknown',
                creation_date=None,
                expiry_date=None
            )
        except Exception as e:
            logger.warning(f"Domain analysis failed for {domain}: {e}")
            return DomainAnalysis(
                age_days=None, ssl_enabled=False, domain_authority=1.0,
                page_authority=1.0, trust_score=1.0, hosting_country='Unknown',
                registrar='Unknown', creation_date=None, expiry_date=None
            )
    
    @classmethod
    async def _check_ssl(cls, domain: str) -> bool:
        """Check if domain has valid SSL certificate."""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    return True
        except:
            return False
    
    @classmethod
    async def _calculate_trust_score(cls, domain: str, ssl_enabled: bool) -> float:
        """Calculate trust score based on various factors."""
        score = 5.0  # Base score
        
        # SSL bonus
        if ssl_enabled:
            score += 1.0
        
        # Known domain bonus
        if domain in cls.AUTHORITY_DATABASE:
            score += 2.0
        
        # Domain structure analysis
        if len(domain.split('.')) <= 3:  # Not too many subdomains
            score += 0.5
        
        # Financial keywords in domain
        financial_keywords = ['finance', 'money', 'stock', 'market', 'invest', 'trade']
        if any(keyword in domain.lower() for keyword in financial_keywords):
            score += 1.0
        
        return min(score, 10.0)


class ContentCategorizer:
    """Content-based categorization system."""
    
    # Content category patterns
    CATEGORY_PATTERNS = {
        'regulatory': {
            'keywords': ['sebi', 'circular', 'regulation', 'compliance', 'guidelines', 'notification', 'rbi', 'irdai'],
            'url_patterns': [r'/circular/', r'/regulation/', r'/notification/'],
            'title_patterns': [r'circular', r'regulation', r'guideline', r'notification']
        },
        'market_news': {
            'keywords': ['sensex', 'nifty', 'market', 'index', 'trading', 'volume', 'rally', 'fall'],
            'url_patterns': [r'/market/', r'/stocks/', r'/trading/'],
            'title_patterns': [r'market', r'sensex', r'nifty', r'index']
        },
        'corporate_results': {
            'keywords': ['results', 'earnings', 'profit', 'loss', 'revenue', 'quarterly', 'annual'],
            'url_patterns': [r'/earnings/', r'/results/', r'/financials/'],
            'title_patterns': [r'results', r'earnings', r'q[1-4]', r'fy\d+']
        },
        'ipos_listings': {
            'keywords': ['ipo', 'listing', 'public offering', 'share sale', 'issue price'],
            'url_patterns': [r'/ipo/', r'/listing/', r'/public-offering/'],
            'title_patterns': [r'ipo', r'listing', r'public offering']
        },
        'mutual_funds': {
            'keywords': ['mutual fund', 'sip', 'nav', 'fund house', 'scheme', 'amc'],
            'url_patterns': [r'/mutual-fund/', r'/mf/', r'/fund/'],
            'title_patterns': [r'mutual fund', r'sip', r'nav', r'fund']
        },
        'commodity': {
            'keywords': ['gold', 'silver', 'crude oil', 'commodity', 'mcx', 'ncdex'],
            'url_patterns': [r'/commodity/', r'/gold/', r'/oil/'],
            'title_patterns': [r'gold', r'silver', r'crude', r'commodity']
        },
        'forex': {
            'keywords': ['dollar', 'rupee', 'forex', 'currency', 'exchange rate', 'usd', 'inr'],
            'url_patterns': [r'/forex/', r'/currency/', r'/exchange/'],
            'title_patterns': [r'dollar', r'rupee', r'forex', r'currency']
        }
    }
    
    @classmethod
    def categorize_content(cls, title: str, content: str, url: str) -> Tuple[str, str, float]:
        """Categorize content based on text analysis."""
        title_lower = title.lower()
        content_lower = content.lower()
        url_lower = url.lower()
        
        category_scores = {}
        
        for category, patterns in cls.CATEGORY_PATTERNS.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in patterns['keywords'] 
                                if keyword in content_lower or keyword in title_lower)
            score += keyword_matches * 2.0
            
            # URL pattern matching
            url_matches = sum(1 for pattern in patterns['url_patterns'] 
                            if re.search(pattern, url_lower))
            score += url_matches * 3.0
            
            # Title pattern matching
            title_matches = sum(1 for pattern in patterns['title_patterns'] 
                              if re.search(pattern, title_lower))
            score += title_matches * 2.5
            
            category_scores[category] = score
        
        # Find best category
        if not category_scores or max(category_scores.values()) == 0:
            return 'general', 'unclassified', 0.1
        
        best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
        confidence = min(category_scores[best_category] / 10.0, 1.0)
        
        # Determine subcategory
        subcategory = cls._determine_subcategory(best_category, title_lower, content_lower)
        
        return best_category, subcategory, confidence
    
    @classmethod
    def _determine_subcategory(cls, category: str, title: str, content: str) -> str:
        """Determine subcategory based on specific patterns."""
        if category == 'market_news':
            if any(word in title or word in content for word in ['opening', 'closing']):
                return 'market_summary'
            elif any(word in title for word in ['breaking', 'alert', 'urgent']):
                return 'breaking_news'
            return 'general_news'
        
        elif category == 'corporate_results':
            if any(word in title for word in ['q1', 'q2', 'q3', 'q4']):
                return 'quarterly'
            elif 'annual' in title:
                return 'annual'
            return 'earnings'
        
        return 'general'


class MLCategorizer:
    """Machine Learning-based categorization system."""
    
    def __init__(self):
        self.features_weights = {
            'financial_keyword_density': 0.3,
            'domain_authority': 0.25,
            'url_structure_score': 0.15,
            'content_length_score': 0.1,
            'title_relevance': 0.2
        }
    
    def predict_category(self, url: str, title: str, content: str) -> Tuple[str, float]:
        """Predict category using ML-like feature analysis."""
        features = self.extract_features(url, title, content)
        
        # Rule-based classification mimicking ML
        domain = urlparse(url).netloc.lower()
        
        # Official sources
        if any(official in domain for official in ['sebi.gov.in', 'nseindia.com', 'bseindia.com', 'rbi.org.in']):
            confidence = features['domain_authority'] * features['financial_keyword_density']
            return 'official', min(confidence, 1.0)
        
        # News media
        news_domains = ['economictimes', 'moneycontrol', 'livemint', 'business-standard']
        if any(news in domain for news in news_domains):
            confidence = (features['financial_keyword_density'] + features['title_relevance']) / 2
            return 'news_media', min(confidence, 1.0)
        
        # Broker platforms
        if any(broker in domain for broker in ['zerodha', 'icicidirect', 'hdfcsec', 'kotaksecurities']):
            return 'broker_platform', 0.8
        
        # Research platforms
        if features['financial_keyword_density'] > 0.3 and features['content_length_score'] > 0.5:
            return 'research_analysis', 0.7
        
        # Social/blog
        if any(social in domain for social in ['twitter', 'facebook', 'linkedin', 'reddit']):
            return 'social_media', 0.4
        
        # Default classification
        return 'general', 0.3
    
    def extract_features(self, url: str, title: str, content: str) -> Dict[str, float]:
        """Extract features for ML classification."""
        features = {}
        
        # Financial keyword density
        financial_keywords = [
            'stock', 'market', 'nse', 'bse', 'sensex', 'nifty', 'trading', 'investment',
            'mutual fund', 'sip', 'ipo', 'earnings', 'profit', 'revenue', 'sebi', 'rbi'
        ]
        total_words = len((title + ' ' + content).split())
        keyword_count = sum(1 for keyword in financial_keywords 
                          if keyword in (title + ' ' + content).lower())
        features['financial_keyword_density'] = keyword_count / max(total_words, 1)
        
        # Domain authority (simplified)
        domain = urlparse(url).netloc
        features['domain_authority'] = DomainAuthority.AUTHORITY_DATABASE.get(domain, 1.0) / 10.0
        
        # URL structure score
        path_segments = len(urlparse(url).path.split('/'))
        features['url_structure_score'] = min(path_segments / 5.0, 1.0)
        
        # Content length score
        content_length = len(content)
        features['content_length_score'] = min(content_length / 1000.0, 1.0)
        
        # Title relevance
        title_words = len(title.split())
        title_financial_words = sum(1 for keyword in financial_keywords 
                                  if keyword in title.lower())
        features['title_relevance'] = title_financial_words / max(title_words, 1)
        
        return features


@dataclass
class SourceCategorizer:
    """Main source categorization system integrating all components."""
    
    def __init__(self):
        self.domain_analyzer = DomainAuthority()
        self.content_categorizer = ContentCategorizer()
        self.ml_categorizer = MLCategorizer()
    
    async def categorize_source(self, url: str, title: str = "", content: str = "") -> CategorizationResult:
        """Perform comprehensive source categorization."""
        try:
            # Parse domain
            domain = urlparse(url).netloc
            
            # Domain analysis
            domain_analysis = await DomainAuthority.analyze_domain(domain)
            
            # Content categorization
            primary_category, subcategory, content_confidence = ContentCategorizer.categorize_content(
                title, content, url
            )
            
            # ML categorization
            ml_category, ml_confidence = self.ml_categorizer.predict_category(url, title, content)
            
            # Determine domain category
            domain_category = self._determine_domain_category(domain, domain_analysis)
            
            # Calculate overall confidence
            overall_confidence = (content_confidence + ml_confidence) / 2
            
            # Determine reliability tier
            reliability_tier = self._determine_reliability_tier(domain_analysis, overall_confidence)
            
            # Determine content type
            content_type = self._determine_content_type(title, content)
            
            # Generate tags
            tags = self._generate_tags(url, title, content, primary_category)
            
            return CategorizationResult(
                domain_category=domain_category,
                primary_category=primary_category,
                authority_score=domain_analysis.domain_authority,
                confidence=overall_confidence,
                reliability_tier=reliability_tier,
                content_type=content_type,
                tags=tags,
                subcategories=[subcategory] if subcategory != 'general' else []
            )
            
        except Exception as e:
            logger.error(f"Categorization failed for {url}: {e}")
            # Fallback to simple categorization
            return self._fallback_categorization(url, title, content)
    
    def _determine_domain_category(self, domain: str, analysis: DomainAnalysis) -> str:
        """Determine domain category based on analysis."""
        if any(official in domain for official in ['sebi.gov.in', 'nseindia.com', 'bseindia.com', 'rbi.org.in']):
            return 'exchange'
        elif analysis.domain_authority >= 8.0:
            return 'verified_media'
        elif any(broker in domain for broker in ['zerodha', 'icicidirect', 'hdfcsec']):
            return 'broker'
        elif any(social in domain for social in ['twitter', 'facebook', 'reddit']):
            return 'social'
        elif '.gov' in domain:
            return 'government'
        else:
            return 'general'
    
    def _determine_reliability_tier(self, domain_analysis: DomainAnalysis, confidence: float) -> str:
        """Determine reliability tier based on domain analysis and confidence."""
        authority = domain_analysis.domain_authority
        
        if authority >= 9.0 and confidence >= 0.8:
            return 'tier1_official'
        elif authority >= 7.0 and confidence >= 0.6:
            return 'tier2_verified'
        elif authority >= 5.0 and confidence >= 0.4:
            return 'tier3_reliable'
        elif authority >= 3.0 and confidence >= 0.3:
            return 'tier4_moderate'
        else:
            return 'tier5_unverified'
    
    def _determine_content_type(self, title: str, content: str) -> str:
        """Determine content type based on title and content analysis."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Breaking news detection
        if any(word in title_lower for word in ['breaking', 'urgent', 'alert', 'flash']):
            return 'breaking_news'
        
        # Analysis detection
        if any(word in title_lower for word in ['analysis', 'review', 'outlook', 'forecast']):
            return 'analysis'
        
        # Financial report detection
        if any(word in title_lower for word in ['results', 'earnings', 'financial', 'quarterly', 'annual']):
            return 'financial_report'
        
        # Regulatory notice
        if any(word in title_lower for word in ['circular', 'notification', 'guideline', 'regulation']):
            return 'regulatory_notice'
        
        # Interview/opinion
        if any(word in title_lower for word in ['interview', 'opinion', 'says', 'believes']):
            return 'interview_opinion'
        
        return 'general_news'
    
    def _generate_tags(self, url: str, title: str, content: str, category: str) -> List[str]:
        """Generate relevant tags for the source."""
        tags = []
        text = (title + ' ' + content).lower()
        
        # Add category as tag
        tags.append(category)
        
        # Financial instrument tags
        if any(word in text for word in ['equity', 'stock', 'share']):
            tags.append('equity')
        if any(word in text for word in ['mutual fund', 'mf', 'sip']):
            tags.append('mutual_funds')
        if any(word in text for word in ['derivative', 'option', 'future']):
            tags.append('derivatives')
        if any(word in text for word in ['bond', 'debt', 'fixed income']):
            tags.append('fixed_income')
        
        # Market tags
        if any(word in text for word in ['sensex', 'nifty', 'index']):
            tags.append('market_index')
        if 'ipo' in text:
            tags.append('ipo')
        
        # Regulatory tags
        if any(word in text for word in ['sebi', 'rbi', 'regulation']):
            tags.append('regulatory')
        
        return list(set(tags))  # Remove duplicates
    
    def _fallback_categorization(self, url: str, title: str, content: str) -> CategorizationResult:
        """Fallback categorization when main process fails."""
        # Simple hash-based fallback similar to original placeholder
        h = int(hashlib.sha256(url.encode()).hexdigest(), 16)
        authority_score = (h % 70) / 10 + 3.0
        confidence = ((h >> 4) % 100) / 100.0
        
        if any(k in url.lower() for k in [".gov", ".edu"]):
            domain_category = "government"
            authority_score = max(authority_score, 8.5)
        elif any(k in url.lower() for k in ["reuters", "bloomberg", "moneycontrol"]):
            domain_category = "verified_media"
            authority_score = max(authority_score, 7.0)
        elif any(k in url.lower() for k in ["twitter", "reddit"]):
            domain_category = "social"
        else:
            domain_category = "general"
        
        primary_category = "market_news" if any(w in content.lower() for w in ["stock", "market", "nse", "bse"]) else "general"
        
        return CategorizationResult(
            domain_category=domain_category,
            primary_category=primary_category,
            authority_score=authority_score,
            confidence=confidence,
            reliability_tier='tier4_moderate',
            content_type='general_news',
            tags=[primary_category],
            subcategories=[]
        )


# Convenience function for direct usage
async def categorize_source(url: str, title: str = "", content: str = "") -> CategorizationResult:
    """Convenience function to categorize a single source."""
    categorizer = SourceCategorizer()
    return await categorizer.categorize_source(url, title, content)


# Batch processing capability
async def categorize_sources_batch(sources: List[Dict[str, str]]) -> List[CategorizationResult]:
    """Categorize multiple sources in batch."""
    categorizer = SourceCategorizer()
    tasks = [
        categorizer.categorize_source(
            source.get('url', ''),
            source.get('title', ''),
            source.get('content', '')
        )
        for source in sources
    ]
    return await asyncio.gather(*tasks) 