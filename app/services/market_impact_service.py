"""
Market Impact Service

Service layer for integrating the market impact scoring system with the
existing trading application. Provides high-level methods for analyzing
market impact of news events on assets and portfolios.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from sqlalchemy.orm import Session

from app.analysis.market_impact import (
    MarketImpactScorer, 
    PredictiveImpactAnalyzer,
    NewsEvent,
    ImpactScore,
    create_news_event_from_db,
    calculate_portfolio_impact
)
from app.analysis.sentiment_engine import SentimentEngine
from app.database.connection import get_db
from fastapi import Depends

logger = logging.getLogger(__name__)


class MarketImpactService:
    """
    Service for analyzing market impact of news events on assets and portfolios
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.impact_scorer = MarketImpactScorer()
        self.predictive_analyzer = PredictiveImpactAnalyzer(self.impact_scorer)
        self.sentiment_engine = SentimentEngine()
        
    async def analyze_news_impact(self, news_item, asset_symbol: str) -> Dict[str, Any]:
        """
        Analyze the market impact of a news item on a specific asset
        
        Args:
            news_item: Database news item object
            asset_symbol: Asset symbol to analyze impact for
            
        Returns:
            Dict containing impact analysis results
        """
        try:
            # Calculate sentiment score for the news
            sentiment_result = await self.sentiment_engine.analyze_text(
                news_item.content or news_item.title or ""
            )
            
            # Create NewsEvent object
            news_event = create_news_event_from_db(
                news_item, 
                sentiment_score=sentiment_result.get('sentiment_score', 0.0)
            )
            
            # Calculate market impact
            impact_score = await self.impact_scorer.calculate_market_impact(
                news_event, asset_symbol
            )
            
            # Generate enhanced analysis with predictions
            enhanced_analysis = await self.predictive_analyzer.enhanced_impact_analysis(
                news_event, asset_symbol
            )
            
            return {
                'news_id': getattr(news_item, 'id', None),
                'asset_symbol': asset_symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'impact_analysis': enhanced_analysis,
                'raw_impact_score': impact_score.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news impact: {str(e)}")
            raise
    
    async def analyze_portfolio_impact(self, portfolio_assets: List[str], 
                                     time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze market impact on an entire portfolio
        
        Args:
            portfolio_assets: List of asset symbols in portfolio
            time_window_hours: Time window for news analysis
            
        Returns:
            Dict containing portfolio impact analysis
        """
        try:
            # Get recent news items (this would typically query the database)
            # For now, we'll use a placeholder
            news_events = await self._get_recent_news_events(time_window_hours)
            
            # Calculate portfolio impact
            portfolio_analysis = await calculate_portfolio_impact(
                self.impact_scorer, news_events, portfolio_assets
            )
            
            return {
                'portfolio_assets': portfolio_assets,
                'analysis_timestamp': datetime.now().isoformat(),
                'time_window_hours': time_window_hours,
                'portfolio_analysis': portfolio_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio impact: {str(e)}")
            raise
    
    async def generate_impact_alerts(self, asset_symbol: str, 
                                   impact_threshold: float = 6.0,
                                   confidence_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Generate alerts based on high-impact market events
        
        Args:
            asset_symbol: Asset symbol to monitor
            impact_threshold: Minimum impact score to trigger alert
            confidence_threshold: Minimum confidence to trigger alert
            
        Returns:
            List of alert dictionaries
        """
        try:
            alerts = []
            
            # Get recent news events
            news_events = await self._get_recent_news_events(24)
            
            for news_event in news_events:
                # Calculate impact for this asset
                impact_score = await self.impact_scorer.calculate_market_impact(
                    news_event, asset_symbol
                )
                
                # Check if alert criteria are met
                if (impact_score.score >= impact_threshold and 
                    impact_score.confidence >= confidence_threshold):
                    
                    alert_data = {
                        'asset_symbol': asset_symbol,
                        'impact_score': impact_score.score,
                        'direction': impact_score.direction.value,
                        'confidence': impact_score.confidence,
                        'category': impact_score.category.value,
                        'reasoning': impact_score.reasoning,
                        'news_title': news_event.title,
                        'news_source': news_event.source,
                        'timestamp': impact_score.timestamp.isoformat(),
                        'alert_type': 'high_impact_event'
                    }
                    
                    alerts.append(alert_data)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating impact alerts: {str(e)}")
            raise
    
    async def get_impact_summary(self, asset_symbol: str, 
                               time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get a summary of market impact for an asset over a time window
        
        Args:
            asset_symbol: Asset symbol
            time_window_hours: Time window for analysis
            
        Returns:
            Dict containing impact summary
        """
        try:
            # Get recent news events
            news_events = await self._get_recent_news_events(time_window_hours)
            
            # Calculate impacts for all relevant events
            impact_scores = []
            for news_event in news_events:
                relevance = self.impact_scorer.calculate_asset_relevance_score(
                    news_event, asset_symbol
                )
                
                if relevance > 0.1:  # Only include relevant events
                    impact = await self.impact_scorer.calculate_market_impact(
                        news_event, asset_symbol
                    )
                    impact_scores.append(impact)
            
            if not impact_scores:
                return {
                    'asset_symbol': asset_symbol,
                    'time_window_hours': time_window_hours,
                    'summary': 'No significant impact events found',
                    'aggregate_score': 0,
                    'event_count': 0
                }
            
            # Aggregate the scores
            aggregated_impact = self.impact_scorer.aggregate_impact_scores(
                impact_scores, time_window_hours
            )
            
            # Calculate additional metrics
            high_impact_events = [s for s in impact_scores if s.score > 6]
            positive_events = [s for s in impact_scores if s.direction.value == 'positive']
            negative_events = [s for s in impact_scores if s.direction.value == 'negative']
            
            return {
                'asset_symbol': asset_symbol,
                'time_window_hours': time_window_hours,
                'aggregate_impact': aggregated_impact.to_dict(),
                'event_count': len(impact_scores),
                'high_impact_event_count': len(high_impact_events),
                'positive_event_count': len(positive_events),
                'negative_event_count': len(negative_events),
                'average_confidence': sum(s.confidence for s in impact_scores) / len(impact_scores),
                'summary': self._generate_impact_summary_text(aggregated_impact, impact_scores)
            }
            
        except Exception as e:
            logger.error(f"Error getting impact summary: {str(e)}")
            raise
    
    async def _get_recent_news_events(self, time_window_hours: int) -> List[NewsEvent]:
        """
        Get recent news events from database (placeholder implementation)
        
        Args:
            time_window_hours: Time window for news retrieval
            
        Returns:
            List of NewsEvent objects
        """
        # TODO: Implement actual database query for news items
        # This is a placeholder that would typically query the news table
        
        # For now, create some sample news events for testing
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        sample_events = [
            NewsEvent(
                title="Tech stocks rally on positive earnings outlook",
                content="Major technology companies show strong Q4 performance with increased revenue and optimistic guidance for next quarter.",
                source="Reuters",
                timestamp=datetime.now() - timedelta(hours=2),
                sentiment_score=0.7,
                entities=["technology", "earnings", "Q4"],
                asset_symbols=["AAPL", "GOOGL", "MSFT"]
            ),
            NewsEvent(
                title="Federal Reserve signals potential interest rate changes",
                content="Fed officials indicate possible monetary policy adjustments based on economic indicators and inflation data.",
                source="Bloomberg",
                timestamp=datetime.now() - timedelta(hours=6),
                sentiment_score=-0.3,
                entities=["federal reserve", "interest rates", "monetary policy"],
                asset_symbols=["SPY", "QQQ"]
            ),
            NewsEvent(
                title="Oil prices surge on geopolitical tensions",
                content="Crude oil futures climb amid ongoing geopolitical developments affecting global supply chains.",
                source="CNBC",
                timestamp=datetime.now() - timedelta(hours=8),
                sentiment_score=0.4,
                entities=["oil", "geopolitical", "supply chain"],
                asset_symbols=["USO", "XLE"]
            )
        ]
        
        # Filter by time window
        recent_events = [
            event for event in sample_events 
            if event.timestamp >= cutoff_time
        ]
        
        return recent_events
    
    def _generate_impact_summary_text(self, aggregated_impact: ImpactScore, 
                                    individual_scores: List[ImpactScore]) -> str:
        """
        Generate human-readable summary text for impact analysis
        
        Args:
            aggregated_impact: Aggregated impact score
            individual_scores: List of individual impact scores
            
        Returns:
            str: Summary text
        """
        direction_text = {
            'positive': 'bullish',
            'negative': 'bearish',
            'neutral': 'neutral'
        }.get(aggregated_impact.direction.value, 'mixed')
        
        confidence_text = {
            'very_high': 'very high',
            'high': 'high',
            'medium': 'moderate',
            'low': 'low',
            'very_low': 'very low'
        }.get(aggregated_impact.confidence_level.value, 'uncertain')
        
        score_text = 'significant' if aggregated_impact.score > 6 else 'moderate' if aggregated_impact.score > 3 else 'minor'
        
        return (f"Overall {direction_text} sentiment with {score_text} impact "
               f"(score: {aggregated_impact.score:.1f}) and {confidence_text} confidence "
               f"based on {len(individual_scores)} events.")
    
    async def train_impact_model(self, historical_data: List[Dict]) -> None:
        """
        Train the predictive impact model with historical data
        
        Args:
            historical_data: Historical impact scores and actual outcomes
        """
        try:
            self.predictive_analyzer.train_prediction_model(historical_data)
            logger.info("Market impact model training completed")
            
        except Exception as e:
            logger.error(f"Error training impact model: {str(e)}")
            raise
    
    async def get_real_time_impact_feed(self, asset_symbols: List[str]) -> Dict[str, Any]:
        """
        Get real-time impact feed for multiple assets
        
        Args:
            asset_symbols: List of asset symbols to monitor
            
        Returns:
            Dict containing real-time impact data
        """
        try:
            # Get very recent news (last 4 hours)
            recent_events = await self._get_recent_news_events(4)
            
            # Calculate impact for each asset
            asset_impacts = {}
            for symbol in asset_symbols:
                impacts = []
                for event in recent_events:
                    relevance = self.impact_scorer.calculate_asset_relevance_score(event, symbol)
                    if relevance > 0.2:  # Higher threshold for real-time feed
                        impact = await self.impact_scorer.calculate_market_impact(event, symbol)
                        impacts.append(impact)
                
                if impacts:
                    # Get the most recent high-confidence impact
                    high_confidence_impacts = [i for i in impacts if i.confidence > 0.5]
                    if high_confidence_impacts:
                        latest_impact = max(high_confidence_impacts, key=lambda x: x.timestamp)
                        asset_impacts[symbol] = latest_impact.to_dict()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'feed_type': 'real_time_impact',
                'asset_impacts': asset_impacts,
                'total_events_analyzed': len(recent_events),
                'assets_with_impact': len(asset_impacts)
            }
            
        except Exception as e:
            logger.error(f"Error generating real-time impact feed: {str(e)}")
            raise


# Factory function for creating service instances
def create_market_impact_service(
    db_session: Session = Depends(get_db),
) -> MarketImpactService:
    """Create and return a configured MarketImpactService instance."""
    return MarketImpactService(db_session) 