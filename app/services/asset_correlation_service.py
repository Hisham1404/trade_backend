"""
Asset Correlation Service

Service layer for integrating the Asset Correlation Analysis system with the
existing trading application. Provides high-level methods for analyzing
correlations between news events and assets.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session

from app.analysis.asset_correlation import (
    create_correlation_analyzer,
    quick_asset_correlation,
    extract_asset_symbols_from_text
)
from app.database.connection import get_db

logger = logging.getLogger(__name__)


class AssetCorrelationService:
    """
    Service for analyzing correlations between news events and financial assets
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.correlation_analyzer = create_correlation_analyzer(db_session)
        
    async def analyze_news_correlation(self, news_title: str, news_content: str,
                                     source: str = "unknown",
                                     target_assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze correlations between a news article and financial assets
        
        Args:
            news_title: News article title
            news_content: News article content
            source: News source
            target_assets: Optional list of specific assets to analyze
            
        Returns:
            Dict containing correlation analysis results
        """
        try:
            analysis_result = await self.correlation_analyzer.analyze_news_to_assets(
                news_title, news_content, source, target_assets
            )
            
            # Enhance result with additional metadata
            enhanced_result = {
                **analysis_result,
                'service_metadata': {
                    'analysis_type': 'single_news_correlation',
                    'service_version': '1.0.0',
                    'processing_time': datetime.now().isoformat()
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in news correlation analysis: {str(e)}")
            raise

    async def get_asset_mentions(self, text: str) -> Dict[str, Any]:
        """
        Extract and analyze asset mentions from text
        
        Args:
            text: Text to analyze for asset mentions
            
        Returns:
            Dict containing extracted asset symbols and analysis
        """
        try:
            # Extract asset symbols
            asset_symbols = extract_asset_symbols_from_text(text)
            
            # Get detailed correlation analysis if symbols found
            if asset_symbols:
                correlation_analysis = await quick_asset_correlation(
                    text, "", target_assets=asset_symbols
                )
            else:
                correlation_analysis = []
            
            return {
                'extracted_symbols': asset_symbols,
                'symbol_count': len(asset_symbols),
                'correlation_analysis': correlation_analysis,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error extracting asset mentions: {str(e)}")
            raise
    
    async def find_related_assets(self, primary_asset: str, 
                                news_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Find assets related to a primary asset through various correlation types
        
        Args:
            primary_asset: Primary asset symbol
            news_context: Optional news context for enhanced correlation
            
        Returns:
            Dict containing related assets and correlation types
        """
        try:
            # Use static relationship mappings for basic correlation
            related_assets = self._get_static_related_assets(primary_asset)
            
            return {
                'primary_asset': primary_asset,
                'related_assets': related_assets,
                'relationship_types': self._categorize_relationships(related_assets),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error finding related assets: {str(e)}")
            raise
    
    async def analyze_competitive_landscape(self, asset_symbol: str) -> Dict[str, Any]:
        """
        Analyze the competitive landscape for an asset
        
        Args:
            asset_symbol: Asset symbol to analyze
            
        Returns:
            Dict containing competitive analysis
        """
        try:
            # Get competitive relationships
            competitive_mapping = {
                'AAPL': ['GOOGL', 'MSFT', 'AMZN', 'META'],
                'GOOGL': ['AAPL', 'MSFT', 'META', 'AMZN'],
                'MSFT': ['AAPL', 'GOOGL', 'ORCL', 'CRM'],
                'AMZN': ['AAPL', 'GOOGL', 'MSFT', 'WMT'],
                'TSLA': ['F', 'GM', 'NIO', 'RIVN'],
                'JPM': ['BAC', 'WFC', 'GS', 'MS']
            }
            
            competitors = competitive_mapping.get(asset_symbol, [])
            sector_peers = await self._get_sector_peers(asset_symbol)
            
            return {
                'primary_asset': asset_symbol,
                'direct_competitors': competitors,
                'sector_peers': sector_peers,
                'competitive_analysis': {
                    'direct_competitor_count': len(competitors),
                    'sector_peer_count': len(sector_peers),
                    'competitive_intensity': 'high' if len(competitors) > 3 else 'medium' if len(competitors) > 1 else 'low'
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing competitive landscape: {str(e)}")
            raise
    
    def _get_static_related_assets(self, asset_symbol: str) -> List[Dict[str, Any]]:
        """Get statically defined related assets"""
        relationships = {
            'AAPL': [
                {'symbol': 'MSFT', 'relationship_type': 'competitor', 'confidence': 0.8},
                {'symbol': 'GOOGL', 'relationship_type': 'competitor', 'confidence': 0.7},
                {'symbol': 'NVDA', 'relationship_type': 'supplier', 'confidence': 0.6}
            ],
            'TSLA': [
                {'symbol': 'F', 'relationship_type': 'competitor', 'confidence': 0.9},
                {'symbol': 'GM', 'relationship_type': 'competitor', 'confidence': 0.8},
                {'symbol': 'NVDA', 'relationship_type': 'supplier', 'confidence': 0.5}
            ]
        }
        
        return relationships.get(asset_symbol, [])
    
    def _categorize_relationships(self, related_assets: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize relationship types"""
        categories = {}
        for asset in related_assets:
            rel_type = asset.get('relationship_type', 'unknown')
            categories[rel_type] = categories.get(rel_type, 0) + 1
        
        return categories
    
    async def _get_sector_peers(self, asset_symbol: str) -> List[str]:
        """Get sector peers for an asset"""
        sector_peers = {
            'AAPL': ['MSFT', 'GOOGL', 'META', 'NVDA'],
            'TSLA': ['F', 'GM'],
            'JPM': ['BAC', 'WFC', 'GS', 'MS']
        }
        
        return sector_peers.get(asset_symbol, [])


# Factory function for creating service instances
def create_asset_correlation_service(db_session: Optional[Session] = None) -> AssetCorrelationService:
    """
    Factory function to create AssetCorrelationService instance
    
    Args:
        db_session: Database session (optional, will create if not provided)
        
    Returns:
        AssetCorrelationService: Configured service instance
    """
    if db_session is None:
        db_session = next(get_db())
    
    # Ensure we have a valid Session object
    if not isinstance(db_session, Session):
        raise ValueError("Invalid database session provided")
    
    return AssetCorrelationService(db_session)
