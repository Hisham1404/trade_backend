"""
Asset Correlation Analysis System

This module implements a comprehensive asset correlation analysis engine that
links news events to specific assets using entity extraction, relationship
scoring, and advanced correlation algorithms.
"""

import logging
import re
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from enum import Enum
import asyncio
from collections import defaultdict
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities that can be extracted from news"""
    COMPANY = "company"
    TICKER_SYMBOL = "ticker_symbol"
    PERSON = "person"
    INDUSTRY = "industry"
    SECTOR = "sector"
    PRODUCT = "product"
    FINANCIAL_METRIC = "financial_metric"
    GEOGRAPHIC = "geographic"
    EVENT = "event"
    CURRENCY = "currency"


class CorrelationType(Enum):
    """Types of correlations between assets and news"""
    DIRECT = "direct"              # Direct mention of company/ticker
    INDIRECT = "indirect"          # Related through industry/sector
    COMPETITIVE = "competitive"    # Competitor relationship
    SUPPLY_CHAIN = "supply_chain"  # Supply chain relationship
    MARKET_WIDE = "market_wide"    # Broad market impact
    SECTOR_WIDE = "sector_wide"    # Sector-wide impact


@dataclass
class EntityMatch:
    """Represents an extracted entity and its confidence"""
    text: str
    entity_type: EntityType
    confidence: float
    start_position: int
    end_position: int
    normalized_value: Optional[str] = None


@dataclass
class AssetCorrelation:
    """Represents a correlation between news and an asset"""
    asset_symbol: str
    asset_name: str
    correlation_type: CorrelationType
    correlation_score: float  # 0-1 confidence of correlation
    reasoning: str
    entities_matched: List[EntityMatch] = field(default_factory=list)
    indirect_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class NewsContext:
    """Context information for news analysis"""
    title: str
    content: str
    source: str
    timestamp: datetime
    entities: List[EntityMatch] = field(default_factory=list)
    asset_correlations: List[AssetCorrelation] = field(default_factory=list)


class EntityExtractor:
    """
    Extracts financial entities from news text using pattern matching
    and NLP techniques
    """
    
    def __init__(self):
        # Common ticker symbol patterns
        self.ticker_patterns = [
            re.compile(r'\b([A-Z]{1,5})\b(?:\s+(?:stock|shares|ticker))?', re.IGNORECASE),
            re.compile(r'\(([A-Z]{2,5})\)', re.IGNORECASE),
            re.compile(r'NYSE:\s*([A-Z]{1,5})', re.IGNORECASE),
            re.compile(r'NASDAQ:\s*([A-Z]{1,5})', re.IGNORECASE),
        ]
        
        # Company name patterns
        self.company_patterns = [
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Co\.?)', re.IGNORECASE),
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Corporation|Company|Incorporated)', re.IGNORECASE),
        ]
        
        # Financial metrics patterns
        self.financial_patterns = [
            re.compile(r'\b(revenue|earnings|profit|loss|EPS|P/E ratio|market cap)\b', re.IGNORECASE),
            re.compile(r'\$\d+(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K)', re.IGNORECASE),
        ]
        
        # Industry/sector patterns
        self.industry_patterns = [
            re.compile(r'\b(technology|healthcare|finance|energy|utilities|telecommunications|automotive|aerospace|retail|real estate)\b', re.IGNORECASE),
        ]
        
        # Known company mappings (expandable database)
        self.company_mappings = {
            'apple': 'AAPL',
            'apple inc': 'AAPL',
            'microsoft': 'MSFT',
            'microsoft corp': 'MSFT',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'tesla': 'TSLA',
            'meta': 'META',
            'facebook': 'META',
            'nvidia': 'NVDA',
            'netflix': 'NFLX',
            'oracle': 'ORCL',
            'salesforce': 'CRM',
            'adobe': 'ADBE',
            'intel': 'INTC',
            'cisco': 'CSCO',
            'paypal': 'PYPL',
            'uber': 'UBER',
            'zoom': 'ZM',
            'slack': 'WORK',
            'twitter': 'TWTR',
            'snapchat': 'SNAP',
            'spotify': 'SPOT',
            'shopify': 'SHOP',
            'square': 'SQ',
            'robinhood': 'HOOD',
            'coinbase': 'COIN',
            'palantir': 'PLTR',
            'snowflake': 'SNOW',
            'airbnb': 'ABNB',
            'doordash': 'DASH',
            'peloton': 'PTON',
            'beyond meat': 'BYND',
            'zoom video': 'ZM',
            'jpmorgan': 'JPM',
            'bank of america': 'BAC',
            'wells fargo': 'WFC',
            'goldman sachs': 'GS',
            'morgan stanley': 'MS',
            'berkshire hathaway': 'BRK.A',
            'exxon mobil': 'XOM',
            'chevron': 'CVX',
            'johnson & johnson': 'JNJ',
            'pfizer': 'PFE',
            'coca cola': 'KO',
            'pepsi': 'PEP',
            'walmart': 'WMT',
            'target': 'TGT',
            'home depot': 'HD',
            'disney': 'DIS',
            'nike': 'NKE',
            'mcdonald\'s': 'MCD',
            'starbucks': 'SBUX',
            'general electric': 'GE',
            'ibm': 'IBM',
            'boeing': 'BA',
            'caterpillar': 'CAT',
            'ford': 'F',
            'general motors': 'GM',
            'at&t': 'T',
            'verizon': 'VZ',
            'comcast': 'CMCSA',
        }
        
        # Sector mappings
        self.sector_mappings = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ORCL', 'CRM', 'ADBE'],
            'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BRK.A'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'automotive': ['TSLA', 'F', 'GM', 'TM', 'HMC'],
            'retail': ['WMT', 'TGT', 'HD', 'COST', 'AMZN'],
            'telecommunications': ['T', 'VZ', 'CMCSA', 'TMUS', 'CHTR']
        }
    
    def extract_entities(self, text: str) -> List[EntityMatch]:
        """
        Extract entities from text using pattern matching
        
        Args:
            text: Text to analyze
            
        Returns:
            List of EntityMatch objects
        """
        entities = []
        text_lower = text.lower()
        
        # Extract ticker symbols
        for pattern in self.ticker_patterns:
            for match in pattern.finditer(text):
                ticker = match.group(1).upper()
                if len(ticker) >= 2 and ticker.isalpha():
                    entities.append(EntityMatch(
                        text=ticker,
                        entity_type=EntityType.TICKER_SYMBOL,
                        confidence=0.9,
                        start_position=match.start(),
                        end_position=match.end(),
                        normalized_value=ticker
                    ))
        
        # Extract company names from known mappings
        for company_name, ticker in self.company_mappings.items():
            if company_name in text_lower:
                start_pos = text_lower.index(company_name)
                entities.append(EntityMatch(
                    text=company_name.title(),
                    entity_type=EntityType.COMPANY,
                    confidence=0.95,
                    start_position=start_pos,
                    end_position=start_pos + len(company_name),
                    normalized_value=ticker
                ))
        
        # Extract company names using patterns
        for pattern in self.company_patterns:
            for match in pattern.finditer(text):
                company_name = match.group(1).strip()
                entities.append(EntityMatch(
                    text=company_name,
                    entity_type=EntityType.COMPANY,
                    confidence=0.7,
                    start_position=match.start(),
                    end_position=match.end(),
                    normalized_value=self._normalize_company_name(company_name)
                ))
        
        # Extract financial metrics
        for pattern in self.financial_patterns:
            for match in pattern.finditer(text):
                entities.append(EntityMatch(
                    text=match.group(0),
                    entity_type=EntityType.FINANCIAL_METRIC,
                    confidence=0.8,
                    start_position=match.start(),
                    end_position=match.end()
                ))
        
        # Extract industries/sectors
        for pattern in self.industry_patterns:
            for match in pattern.finditer(text):
                entities.append(EntityMatch(
                    text=match.group(0),
                    entity_type=EntityType.SECTOR,
                    confidence=0.85,
                    start_position=match.start(),
                    end_position=match.end(),
                    normalized_value=match.group(0).lower()
                ))
        
        # Remove duplicates and overlapping entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _normalize_company_name(self, company_name: str) -> Optional[str]:
        """Normalize company name to ticker symbol if possible"""
        normalized = company_name.lower().strip()
        return self.company_mappings.get(normalized)
    
    def _deduplicate_entities(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """Remove duplicate and overlapping entities"""
        # Sort by start position
        entities.sort(key=lambda x: x.start_position)
        
        deduplicated = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlap = False
            for existing in deduplicated:
                if (entity.start_position < existing.end_position and 
                    entity.end_position > existing.start_position):
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                    else:
                        overlap = True
                    break
            
            if not overlap:
                deduplicated.append(entity)
        
        return deduplicated


class AssetCorrelationEngine:
    """
    Core engine for analyzing correlations between news events and assets
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.entity_extractor = EntityExtractor()
        
        # Correlation scoring weights
        self.correlation_weights = {
            'direct_mention': 0.9,      # Direct company/ticker mention
            'company_mapping': 0.85,    # Known company name mapping
            'sector_correlation': 0.6,  # Sector-based correlation
            'competitive_relation': 0.5, # Competitive relationship
            'supply_chain': 0.4,        # Supply chain relationship
            'market_sentiment': 0.3     # General market sentiment
        }
        
        # Cache for asset information
        self._asset_cache = {}
        self._sector_cache = {}
    
    async def analyze_news_correlations(self, news_context: NewsContext,
                                      target_assets: Optional[List[str]] = None) -> List[AssetCorrelation]:
        """
        Analyze correlations between news content and assets
        
        Args:
            news_context: News context with title and content
            target_assets: Optional list of specific assets to analyze
            
        Returns:
            List of AssetCorrelation objects
        """
        try:
            # Extract entities from news content
            full_text = f"{news_context.title} {news_context.content}"
            entities = self.entity_extractor.extract_entities(full_text)
            news_context.entities = entities
            
            # Get assets to analyze
            if target_assets:
                assets = await self._get_assets_by_symbols(target_assets)
            else:
                assets = await self._get_all_assets()
            
            correlations = []
            
            for asset in assets:
                correlation = await self._calculate_asset_correlation(
                    news_context, asset, entities
                )
                if correlation and correlation.correlation_score > 0.1:  # Minimum threshold
                    correlations.append(correlation)
            
            # Sort by correlation score
            correlations.sort(key=lambda x: x.correlation_score, reverse=True)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error analyzing news correlations: {str(e)}")
            raise
    
    async def _calculate_asset_correlation(self, news_context: NewsContext, 
                                         asset: Dict, entities: List[EntityMatch]) -> Optional[AssetCorrelation]:
        """
        Calculate correlation score between news and a specific asset
        
        Args:
            news_context: News context
            asset: Asset information dictionary
            entities: Extracted entities from news
            
        Returns:
            AssetCorrelation object or None
        """
        asset_symbol = asset['symbol']
        asset_name = asset.get('name', asset_symbol)
        
        correlation_score = 0.0
        correlation_type = CorrelationType.MARKET_WIDE
        reasoning_parts = []
        matched_entities = []
        indirect_factors = {}
        
        # 1. Direct ticker symbol mention
        for entity in entities:
            if (entity.entity_type == EntityType.TICKER_SYMBOL and 
                entity.normalized_value == asset_symbol):
                correlation_score += self.correlation_weights['direct_mention']
                correlation_type = CorrelationType.DIRECT
                reasoning_parts.append(f"Direct ticker mention: {entity.text}")
                matched_entities.append(entity)
        
        # 2. Company name mapping
        for entity in entities:
            if (entity.entity_type == EntityType.COMPANY and 
                entity.normalized_value == asset_symbol):
                correlation_score += self.correlation_weights['company_mapping']
                if correlation_type != CorrelationType.DIRECT:
                    correlation_type = CorrelationType.DIRECT
                reasoning_parts.append(f"Company name match: {entity.text}")
                matched_entities.append(entity)
        
        # 3. Sector correlation
        asset_sector = asset.get('sector')
        if asset_sector and isinstance(asset_sector, str):
            sector_score = await self._calculate_sector_correlation(entities, asset_sector)
            if sector_score > 0:
                correlation_score += sector_score * self.correlation_weights['sector_correlation']
                if correlation_type == CorrelationType.MARKET_WIDE:
                    correlation_type = CorrelationType.SECTOR_WIDE
                reasoning_parts.append(f"Sector correlation: {asset_sector}")
                indirect_factors['sector_score'] = sector_score
        
        # 4. Industry keywords
        industry_score = await self._calculate_industry_correlation(news_context, asset)
        if industry_score > 0:
            correlation_score += industry_score * 0.4
            reasoning_parts.append(f"Industry relevance: {industry_score:.2f}")
            indirect_factors['industry_score'] = industry_score
        
        # 5. Competitive relationships
        competitive_score = await self._calculate_competitive_correlation(entities, asset)
        if competitive_score > 0:
            correlation_score += competitive_score * self.correlation_weights['competitive_relation']
            if correlation_type == CorrelationType.MARKET_WIDE:
                correlation_type = CorrelationType.COMPETITIVE
            reasoning_parts.append(f"Competitive relationship: {competitive_score:.2f}")
            indirect_factors['competitive_score'] = competitive_score
        
        # 6. Market-wide sentiment
        market_sentiment_score = await self._calculate_market_sentiment_correlation(news_context, asset)
        if market_sentiment_score > 0:
            correlation_score += market_sentiment_score * self.correlation_weights['market_sentiment']
            reasoning_parts.append(f"Market sentiment: {market_sentiment_score:.2f}")
            indirect_factors['market_sentiment'] = market_sentiment_score
        
        # Normalize correlation score to 0-1 range
        correlation_score = min(correlation_score, 1.0)
        
        if correlation_score > 0.1:  # Minimum threshold
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "General market correlation"
            
            return AssetCorrelation(
                asset_symbol=asset_symbol,
                asset_name=asset_name,
                correlation_type=correlation_type,
                correlation_score=correlation_score,
                reasoning=reasoning,
                entities_matched=matched_entities,
                indirect_factors=indirect_factors
            )
        
        return None
    
    async def _calculate_sector_correlation(self, entities: List[EntityMatch], 
                                          asset_sector: str) -> float:
        """Calculate correlation based on sector mentions"""
        sector_score = 0.0
        
        for entity in entities:
            if entity.entity_type == EntityType.SECTOR:
                if entity.normalized_value and entity.normalized_value == asset_sector.lower():
                    sector_score = 1.0
                elif (entity.normalized_value and 
                      self._sectors_related(entity.normalized_value, asset_sector.lower())):
                    sector_score = max(sector_score, 0.6)
        
        return sector_score
    
    async def _calculate_industry_correlation(self, news_context: NewsContext, 
                                            asset: Dict) -> float:
        """Calculate correlation based on industry keywords"""
        industry_keywords = {
            'technology': ['tech', 'software', 'hardware', 'digital', 'AI', 'cloud', 'cybersecurity'],
            'finance': ['bank', 'financial', 'credit', 'loan', 'investment', 'trading'],
            'healthcare': ['health', 'medical', 'pharma', 'biotech', 'drug', 'clinical'],
            'energy': ['oil', 'gas', 'renewable', 'solar', 'wind', 'battery'],
            'retail': ['retail', 'consumer', 'shopping', 'store', 'e-commerce']
        }
        
        asset_sector = asset.get('sector', '').lower()
        full_text = f"{news_context.title} {news_context.content}".lower()
        
        if asset_sector in industry_keywords:
            keyword_matches = sum(1 for keyword in industry_keywords[asset_sector] 
                                if keyword in full_text)
            return min(keyword_matches * 0.2, 1.0)
        
        return 0.0
    
    async def _calculate_competitive_correlation(self, entities: List[EntityMatch], 
                                               asset: Dict) -> float:
        """Calculate correlation based on competitive relationships"""
        # Simplified competitive mapping (would be enhanced with real data)
        competitive_groups = {
            'AAPL': ['GOOGL', 'MSFT', 'AMZN', 'META'],
            'GOOGL': ['AAPL', 'MSFT', 'META', 'AMZN'],
            'MSFT': ['AAPL', 'GOOGL', 'ORCL', 'CRM'],
            'AMZN': ['AAPL', 'GOOGL', 'MSFT', 'WMT'],
            'TSLA': ['F', 'GM', 'NIO', 'RIVN'],
            'JPM': ['BAC', 'WFC', 'GS', 'MS'],
            'KO': ['PEP'],
            'PEP': ['KO']
        }
        
        asset_symbol = asset['symbol']
        competitors = competitive_groups.get(asset_symbol, [])
        
        for entity in entities:
            if (entity.entity_type in [EntityType.TICKER_SYMBOL, EntityType.COMPANY] and
                entity.normalized_value in competitors):
                return 0.8
        
        return 0.0
    
    async def _calculate_market_sentiment_correlation(self, news_context: NewsContext, 
                                                    asset: Dict) -> float:
        """Calculate general market sentiment correlation"""
        market_keywords = ['market', 'stocks', 'trading', 'investment', 'economy', 'fed', 'inflation']
        full_text = f"{news_context.title} {news_context.content}".lower()
        
        keyword_matches = sum(1 for keyword in market_keywords if keyword in full_text)
        return min(keyword_matches * 0.15, 0.5)  # Max 0.5 for market-wide sentiment
    
    def _sectors_related(self, sector1: str, sector2: str) -> bool:
        """Check if two sectors are related"""
        related_sectors = {
            'technology': ['telecommunications', 'software'],
            'finance': ['insurance', 'real estate'],
            'healthcare': ['pharmaceuticals', 'biotech'],
            'energy': ['utilities'],
        }
        
        for base_sector, related in related_sectors.items():
            if ((sector1 == base_sector and sector2 in related) or
                (sector2 == base_sector and sector1 in related)):
                return True
        
        return False
    
    async def _get_assets_by_symbols(self, symbols: List[str]) -> List[Dict]:
        """Get asset information by symbols"""
        # TODO: Replace with actual database query
        # This is a placeholder implementation
        
        mock_assets = []
        for symbol in symbols:
            # Use cached data if available
            if symbol in self._asset_cache:
                mock_assets.append(self._asset_cache[symbol])
            else:
                # Create mock asset data (would come from database)
                asset_data = {
                    'symbol': symbol,
                    'name': self._get_company_name_for_symbol(symbol),
                    'sector': self._get_sector_for_symbol(symbol),
                    'industry': self._get_industry_for_symbol(symbol)
                }
                self._asset_cache[symbol] = asset_data
                mock_assets.append(asset_data)
        
        return mock_assets
    
    async def _get_all_assets(self) -> List[Dict]:
        """Get all available assets"""
        # TODO: Replace with actual database query
        # This returns a subset of major assets for testing
        
        major_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
                        'JPM', 'JNJ', 'XOM', 'WMT', 'PG', 'HD', 'CVX', 'BAC']
        
        return await self._get_assets_by_symbols(major_symbols)
    
    def _get_company_name_for_symbol(self, symbol: str) -> str:
        """Get company name for ticker symbol"""
        name_mappings = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'XOM': 'Exxon Mobil Corporation',
            'WMT': 'Walmart Inc.',
            'PG': 'Procter & Gamble Co.',
            'HD': 'The Home Depot Inc.',
            'CVX': 'Chevron Corporation',
            'BAC': 'Bank of America Corp'
        }
        return name_mappings.get(symbol, symbol)
    
    def _get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector for ticker symbol"""
        sector_mappings = {
            'AAPL': 'technology',
            'MSFT': 'technology',
            'GOOGL': 'technology',
            'AMZN': 'technology',
            'TSLA': 'automotive',
            'META': 'technology',
            'NVDA': 'technology',
            'JPM': 'finance',
            'JNJ': 'healthcare',
            'XOM': 'energy',
            'WMT': 'retail',
            'PG': 'consumer_goods',
            'HD': 'retail',
            'CVX': 'energy',
            'BAC': 'finance'
        }
        return sector_mappings.get(symbol, 'unknown')
    
    def _get_industry_for_symbol(self, symbol: str) -> str:
        """Get industry for ticker symbol"""
        industry_mappings = {
            'AAPL': 'Consumer Electronics',
            'MSFT': 'Software',
            'GOOGL': 'Internet Services',
            'AMZN': 'E-commerce',
            'TSLA': 'Electric Vehicles',
            'META': 'Social Media',
            'NVDA': 'Semiconductors',
            'JPM': 'Banking',
            'JNJ': 'Pharmaceuticals',
            'XOM': 'Oil & Gas',
            'WMT': 'Discount Stores',
            'PG': 'Household Products',
            'HD': 'Home Improvement',
            'CVX': 'Oil & Gas',
            'BAC': 'Banking'
        }
        return industry_mappings.get(symbol, 'Unknown')


class CorrelationAnalyzer:
    """
    High-level analyzer for asset correlation analysis
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.correlation_engine = AssetCorrelationEngine(db_session)
    
    async def analyze_news_to_assets(self, title: str, content: str, 
                                   source: str = "unknown",
                                   target_assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze correlations between news and assets
        
        Args:
            title: News title
            content: News content
            source: News source
            target_assets: Optional list of specific assets to analyze
            
        Returns:
            Dictionary with correlation analysis results
        """
        try:
            # Create news context
            news_context = NewsContext(
                title=title,
                content=content,
                source=source,
                timestamp=datetime.now()
            )
            
            # Analyze correlations
            correlations = await self.correlation_engine.analyze_news_correlations(
                news_context, target_assets
            )
            
            # Categorize correlations by type
            direct_correlations = [c for c in correlations if c.correlation_type == CorrelationType.DIRECT]
            indirect_correlations = [c for c in correlations if c.correlation_type != CorrelationType.DIRECT]
            
            # Calculate summary statistics
            total_assets_analyzed = len(correlations)
            high_confidence = [c for c in correlations if c.correlation_score > 0.7]
            medium_confidence = [c for c in correlations if 0.4 <= c.correlation_score <= 0.7]
            low_confidence = [c for c in correlations if c.correlation_score < 0.4]
            
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'news_summary': {
                    'title': title,
                    'source': source,
                    'entities_extracted': len(news_context.entities)
                },
                'correlations': {
                    'total_found': total_assets_analyzed,
                    'direct_correlations': [self._correlation_to_dict(c) for c in direct_correlations],
                    'indirect_correlations': [self._correlation_to_dict(c) for c in indirect_correlations]
                },
                'confidence_breakdown': {
                    'high_confidence': len(high_confidence),
                    'medium_confidence': len(medium_confidence),
                    'low_confidence': len(low_confidence)
                },
                'top_correlations': [
                    {
                        'asset_symbol': c.asset_symbol,
                        'asset_name': c.asset_name,
                        'correlation_score': round(c.correlation_score, 3),
                        'correlation_type': c.correlation_type.value,
                        'reasoning': c.reasoning
                    }
                    for c in correlations[:10]  # Top 10
                ],
                'extracted_entities': [
                    {
                        'text': e.text,
                        'type': e.entity_type.value,
                        'confidence': round(e.confidence, 3),
                        'normalized_value': e.normalized_value
                    }
                    for e in news_context.entities
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            raise
    
    async def bulk_analyze_correlations(self, news_items: List[Dict],
                                      target_assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze correlations for multiple news items
        
        Args:
            news_items: List of news items with title, content, source
            target_assets: Optional list of specific assets to analyze
            
        Returns:
            Dictionary with bulk analysis results
        """
        results = []
        asset_correlation_counts = defaultdict(int)
        total_correlations = 0
        
        for i, news_item in enumerate(news_items):
            try:
                analysis = await self.analyze_news_to_assets(
                    news_item.get('title', ''),
                    news_item.get('content', ''),
                    news_item.get('source', 'unknown'),
                    target_assets
                )
                
                results.append({
                    'news_index': i,
                    'analysis': analysis
                })
                
                # Count correlations per asset
                for correlation in analysis['correlations']['direct_correlations']:
                    asset_correlation_counts[correlation['asset_symbol']] += 1
                    total_correlations += 1
                
                for correlation in analysis['correlations']['indirect_correlations']:
                    asset_correlation_counts[correlation['asset_symbol']] += 1
                    total_correlations += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing news item {i}: {str(e)}")
                results.append({
                    'news_index': i,
                    'error': str(e)
                })
        
        # Find most mentioned assets
        top_assets = sorted(asset_correlation_counts.items(), 
                          key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'bulk_analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_news_items': len(news_items),
                'successful_analyses': len([r for r in results if 'analysis' in r]),
                'failed_analyses': len([r for r in results if 'error' in r]),
                'total_correlations_found': total_correlations,
                'unique_assets_mentioned': len(asset_correlation_counts)
            },
            'top_mentioned_assets': [
                {
                    'asset_symbol': symbol,
                    'mention_count': count,
                    'percentage': round(count / len(news_items) * 100, 1)
                }
                for symbol, count in top_assets
            ],
            'detailed_results': results
        }
    
    def _correlation_to_dict(self, correlation: AssetCorrelation) -> Dict[str, Any]:
        """Convert AssetCorrelation to dictionary"""
        return {
            'asset_symbol': correlation.asset_symbol,
            'asset_name': correlation.asset_name,
            'correlation_type': correlation.correlation_type.value,
            'correlation_score': round(correlation.correlation_score, 3),
            'reasoning': correlation.reasoning,
            'entities_matched_count': len(correlation.entities_matched),
            'indirect_factors': correlation.indirect_factors
        }


# Utility functions for integration
def create_correlation_analyzer(db_session: Session) -> CorrelationAnalyzer:
    """
    Factory function to create CorrelationAnalyzer instance
    
    Args:
        db_session: Database session
        
    Returns:
        CorrelationAnalyzer: Configured analyzer instance
    """
    return CorrelationAnalyzer(db_session)


async def quick_asset_correlation(title: str, content: str, 
                                 target_assets: Optional[List[str]] = None) -> List[Dict]:
    """
    Quick utility function for asset correlation analysis
    
    Args:
        title: News title
        content: News content
        target_assets: Optional list of assets to analyze
        
    Returns:
        List of correlation dictionaries
    """
    # Create temporary analyzer (in production, would use proper session)
    from app.database.connection import get_db
    db_session = next(get_db())
    
    analyzer = CorrelationAnalyzer(db_session)
    result = await analyzer.analyze_news_to_assets(title, content, target_assets=target_assets)
    
    return result['top_correlations']


def extract_asset_symbols_from_text(text: str) -> List[str]:
    """
    Quick utility to extract asset symbols from text
    
    Args:
        text: Text to analyze
        
    Returns:
        List of extracted asset symbols
    """
    extractor = EntityExtractor()
    entities = extractor.extract_entities(text)
    
    symbols = []
    for entity in entities:
        if entity.entity_type == EntityType.TICKER_SYMBOL:
            symbols.append(entity.text)
        elif entity.entity_type == EntityType.COMPANY and entity.normalized_value:
            symbols.append(entity.normalized_value)
    
    return list(set(symbols))  # Remove duplicates
 