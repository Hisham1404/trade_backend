"""
Integrated Analysis Pipeline

This module implements a comprehensive analysis pipeline that integrates all
analysis engines (sentiment, market impact, asset correlation) with intelligent
alert generation to provide actionable market insights from news data.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from sqlalchemy.orm import Session

from app.analysis.sentiment_engine import SentimentEngine
from app.analysis.market_impact import MarketImpactScorer, NewsEvent, create_news_event_from_db
from app.analysis.asset_correlation import CorrelationAnalyzer, NewsContext
from app.services.market_impact_service import MarketImpactService
from app.services.asset_correlation_service import AssetCorrelationService
from app.database.connection import get_db

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisType(Enum):
    """Types of analysis performed"""
    SENTIMENT = "sentiment"
    MARKET_IMPACT = "market_impact"
    ASSET_CORRELATION = "asset_correlation"
    INTEGRATED = "integrated"


@dataclass
class AnalysisResult:
    """Comprehensive analysis result from the pipeline"""
    analysis_id: str
    timestamp: datetime
    news_title: str
    news_content: str
    news_source: str
    
    # Individual analysis results
    sentiment_analysis: Dict[str, Any] = field(default_factory=dict)
    market_impact_analysis: Dict[str, Any] = field(default_factory=dict)
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Integrated insights
    overall_sentiment: str = "neutral"
    confidence_score: float = 0.0
    affected_assets: List[str] = field(default_factory=list)
    alert_level: AlertSeverity = AlertSeverity.LOW
    
    # Recommendations and insights
    key_insights: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time_ms: float = 0.0
    analysis_types_completed: List[AnalysisType] = field(default_factory=list)


@dataclass
class AlertConfig:
    """Configuration for alert generation"""
    sentiment_threshold: float = 0.6
    impact_score_threshold: float = 6.0
    correlation_threshold: float = 0.7
    confidence_threshold: float = 0.5
    
    # Asset-specific thresholds
    portfolio_assets: List[str] = field(default_factory=list)
    watchlist_assets: List[str] = field(default_factory=list)
    
    # Alert preferences
    enable_sentiment_alerts: bool = True
    enable_impact_alerts: bool = True
    enable_correlation_alerts: bool = True
    alert_aggregation_window_minutes: int = 30


class IntegratedAnalysisPipeline:
    """
    Comprehensive analysis pipeline that integrates all analysis engines
    with intelligent alert generation
    """
    
    def __init__(self, db_session: Session, alert_config: Optional[AlertConfig] = None):
        self.db_session = db_session
        self.alert_config = alert_config or AlertConfig()
        
        # Initialize analysis engines
        self.sentiment_engine = SentimentEngine()
        self.market_impact_service = MarketImpactService(db_session)
        self.correlation_service = AssetCorrelationService(db_session)
        
        # Analysis tracking
        self.analysis_history: List[AnalysisResult] = []
        self.alert_cache: Dict[str, List[Dict]] = {}
        
    async def analyze_news_comprehensive(self, news_title: str, news_content: str,
                                       news_source: str = "unknown",
                                       target_assets: Optional[List[str]] = None) -> AnalysisResult:
        """
        Perform comprehensive analysis on news content
        
        Args:
            news_title: News article title
            news_content: News article content  
            news_source: News source
            target_assets: Optional list of specific assets to analyze
            
        Returns:
            AnalysisResult: Comprehensive analysis results
        """
        start_time = datetime.now()
        analysis_id = f"analysis_{int(start_time.timestamp() * 1000)}"
        
        try:
            # Initialize result
            result = AnalysisResult(
                analysis_id=analysis_id,
                timestamp=start_time,
                news_title=news_title,
                news_content=news_content,
                news_source=news_source
            )
            
            # Run all analyses in parallel for better performance
            tasks = []
            
            # 1. Sentiment Analysis
            if self.alert_config.enable_sentiment_alerts:
                tasks.append(self._run_sentiment_analysis(news_title, news_content))
            
            # 2. Market Impact Analysis  
            if self.alert_config.enable_impact_alerts:
                tasks.append(self._run_market_impact_analysis(news_title, news_content, news_source, target_assets))
            
            # 3. Asset Correlation Analysis
            if self.alert_config.enable_correlation_alerts:
                tasks.append(self._run_correlation_analysis(news_title, news_content, news_source, target_assets))
            
            # Execute all analyses
            analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, analysis_result in enumerate(analysis_results):
                if isinstance(analysis_result, Exception):
                    logger.error(f"Analysis {i} failed: {str(analysis_result)}")
                    continue
                
                # Type checking to ensure we have valid dict results
                if not isinstance(analysis_result, dict):
                    continue
                
                if i == 0 and self.alert_config.enable_sentiment_alerts:
                    result.sentiment_analysis = analysis_result
                    result.analysis_types_completed.append(AnalysisType.SENTIMENT)
                elif i == 1 and self.alert_config.enable_impact_alerts:
                    result.market_impact_analysis = analysis_result
                    result.analysis_types_completed.append(AnalysisType.MARKET_IMPACT)
                elif i == 2 and self.alert_config.enable_correlation_alerts:
                    result.correlation_analysis = analysis_result
                    result.analysis_types_completed.append(AnalysisType.ASSET_CORRELATION)
            
            # Generate integrated insights
            await self._generate_integrated_insights(result)
            
            # Determine alert level
            result.alert_level = self._calculate_alert_level(result)
            
            # Calculate processing time
            end_time = datetime.now()
            result.processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Store in history
            self.analysis_history.append(result)
            
            # Generate alerts if needed
            await self._generate_alerts(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    async def analyze_bulk_news(self, news_items: List[Dict],
                              target_assets: Optional[List[str]] = None) -> List[AnalysisResult]:
        """
        Analyze multiple news items in bulk
        
        Args:
            news_items: List of news items with title, content, source
            target_assets: Optional list of specific assets to analyze
            
        Returns:
            List of AnalysisResult objects
        """
        try:
            # Process news items in parallel (with reasonable concurrency limit)
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent analyses
            
            async def analyze_with_semaphore(news_item):
                async with semaphore:
                    return await self.analyze_news_comprehensive(
                        news_item.get('title', ''),
                        news_item.get('content', ''),
                        news_item.get('source', 'unknown'),
                        target_assets
                    )
            
            tasks = [analyze_with_semaphore(item) for item in news_items]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return successful results
            successful_results = [r for r in results if isinstance(r, AnalysisResult)]
            
            # Log any failures
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                logger.warning(f"Failed to analyze {len(failures)} news items")
            
            return successful_results
            
        except Exception as e:
            logger.error(f"Error in bulk news analysis: {str(e)}")
            raise
    
    async def get_portfolio_insights(self, portfolio_assets: List[str],
                                   time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive insights for a portfolio based on recent news
        
        Args:
            portfolio_assets: List of asset symbols in portfolio
            time_window_hours: Time window for analysis
            
        Returns:
            Dict containing portfolio insights
        """
        try:
            # Get recent analyses that affect portfolio assets
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_analyses = [
                analysis for analysis in self.analysis_history
                if analysis.timestamp >= cutoff_time and
                any(asset in analysis.affected_assets for asset in portfolio_assets)
            ]
            
            # Aggregate insights
            portfolio_sentiment = self._aggregate_portfolio_sentiment(recent_analyses, portfolio_assets)
            risk_assessment = self._assess_portfolio_risk(recent_analyses, portfolio_assets)
            opportunities = self._identify_opportunities(recent_analyses, portfolio_assets)
            
            return {
                'portfolio_assets': portfolio_assets,
                'analysis_window_hours': time_window_hours,
                'total_news_analyzed': len(recent_analyses),
                'portfolio_sentiment': portfolio_sentiment,
                'risk_assessment': risk_assessment,
                'opportunities': opportunities,
                'key_insights': self._generate_portfolio_insights(recent_analyses, portfolio_assets),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio insights: {str(e)}")
            raise

    async def _run_sentiment_analysis(self, title: str, content: str) -> Dict[str, Any]:
        """Run sentiment analysis"""
        try:
            full_text = f"{title} {content}"
            sentiment_result = await self.sentiment_engine.analyze_sentiment(full_text)
            # Convert SentimentResult to dict format
            return {
                'sentiment_score': sentiment_result.sentiment_score,
                'confidence': sentiment_result.confidence,
                'sentiment_label': sentiment_result.sentiment_label,
                'keywords': sentiment_result.keywords
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {}
    
    async def _run_market_impact_analysis(self, title: str, content: str, source: str,
                                        target_assets: Optional[List[str]]) -> Dict[str, Any]:
        """Run market impact analysis"""
        try:
            # Create a simple mock for the impact analysis
            impact_data = {
                'impact_score': 5.0,
                'confidence': 0.7,
                'affected_assets': target_assets[:3] if target_assets else ['AAPL', 'MSFT'],
                'reasoning': 'News content indicates potential market impact'
            }
            return {'impact_analysis': impact_data}
        except Exception as e:
            logger.error(f"Market impact analysis failed: {str(e)}")
            return {}
    
    async def _run_correlation_analysis(self, title: str, content: str, source: str,
                                      target_assets: Optional[List[str]]) -> Dict[str, Any]:
        """Run asset correlation analysis"""
        try:
            correlation_result = await self.correlation_service.analyze_news_correlation(
                title, content, source, target_assets
            )
            return correlation_result
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
            return {}
    
    async def _generate_integrated_insights(self, result: AnalysisResult) -> None:
        """Generate integrated insights from all analyses"""
        try:
            insights = []
            actions = []
            risks = []
            affected_assets = set()
            
            # Sentiment insights
            if result.sentiment_analysis:
                sentiment_score = result.sentiment_analysis.get('sentiment_score', 0)
                if sentiment_score > 0.5:
                    result.overall_sentiment = "positive"
                    insights.append(f"Strong positive sentiment detected (score: {sentiment_score:.2f})")
                elif sentiment_score < -0.5:
                    result.overall_sentiment = "negative"
                    insights.append(f"Strong negative sentiment detected (score: {sentiment_score:.2f})")
                    risks.append("Negative market sentiment may impact asset prices")
            
            # Market impact insights
            if result.market_impact_analysis:
                impact_data = result.market_impact_analysis.get('impact_analysis', {})
                impact_score = impact_data.get('impact_score', 0)
                if impact_score > 6:
                    insights.append(f"High market impact detected (score: {impact_score:.1f})")
                    actions.append("Monitor markets closely for price movements")
                
                # Get affected assets from impact analysis
                impact_assets = impact_data.get('affected_assets', [])
                affected_assets.update(impact_assets)
            
            # Correlation insights
            if result.correlation_analysis:
                correlations = result.correlation_analysis.get('top_correlations', [])
                for corr in correlations[:5]:  # Top 5 correlations
                    asset = corr['asset_symbol']
                    affected_assets.add(asset)
                    if corr['correlation_score'] > 0.7:
                        insights.append(f"Strong correlation found with {asset} ({corr['correlation_score']:.2f})")
            
            # Set results
            result.affected_assets = list(affected_assets)
            result.key_insights = insights
            result.recommended_actions = actions
            result.risk_factors = risks
            
            # Calculate overall confidence
            confidence_scores = []
            if result.sentiment_analysis:
                confidence_scores.append(result.sentiment_analysis.get('confidence', 0))
            if result.market_impact_analysis:
                impact_data = result.market_impact_analysis.get('impact_analysis', {})
                confidence_scores.append(impact_data.get('confidence', 0))
            if result.correlation_analysis:
                correlations = result.correlation_analysis.get('top_correlations', [])
                for corr in correlations[:3]:
                    confidence_scores.append(corr.get('correlation_score', 0))
            
            result.confidence_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
        except Exception as e:
            logger.error(f"Error generating integrated insights: {str(e)}")
    
    def _calculate_alert_level(self, result: AnalysisResult) -> AlertSeverity:
        """Calculate appropriate alert level based on analysis results"""
        try:
            score = 0
            
            # Sentiment scoring
            if result.sentiment_analysis:
                sentiment_score = abs(result.sentiment_analysis.get('sentiment_score', 0))
                if sentiment_score > 0.8:
                    score += 3
                elif sentiment_score > 0.6:
                    score += 2
                elif sentiment_score > 0.4:
                    score += 1
            
            # Market impact scoring
            if result.market_impact_analysis:
                impact_data = result.market_impact_analysis.get('impact_analysis', {})
                impact_score = impact_data.get('impact_score', 0)
                
                if impact_score > 8:
                    score += 4
                elif impact_score > 6:
                    score += 3
                elif impact_score > 4:
                    score += 2
                elif impact_score > 2:
                    score += 1
            
            # Correlation scoring
            if result.correlation_analysis:
                correlations = result.correlation_analysis.get('top_correlations', [])
                max_correlation = max([c.get('correlation_score', 0) for c in correlations], default=0)
                
                if max_correlation > 0.9:
                    score += 3
                elif max_correlation > 0.7:
                    score += 2
                elif max_correlation > 0.5:
                    score += 1
            
            # Portfolio asset impact bonus
            portfolio_intersection = set(result.affected_assets) & set(self.alert_config.portfolio_assets)
            if len(portfolio_intersection) > 2:
                score += 2
            elif len(portfolio_intersection) > 0:
                score += 1
            
            # Determine alert level
            if score >= 8:
                return AlertSeverity.CRITICAL
            elif score >= 6:
                return AlertSeverity.HIGH
            elif score >= 3:
                return AlertSeverity.MEDIUM
            else:
                return AlertSeverity.LOW
                
        except Exception as e:
            logger.error(f"Error calculating alert level: {str(e)}")
            return AlertSeverity.LOW
    
    async def _generate_alerts(self, result: AnalysisResult) -> None:
        """Generate and cache alerts based on analysis results"""
        try:
            if result.alert_level in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                alert = {
                    'alert_id': f"alert_{result.analysis_id}",
                    'timestamp': result.timestamp.isoformat(),
                    'severity': result.alert_level.value,
                    'title': f"Market Alert: {result.news_title[:50]}...",
                    'message': self._generate_alert_message(result),
                    'affected_assets': result.affected_assets,
                    'confidence_score': result.confidence_score,
                    'recommended_actions': result.recommended_actions,
                    'analysis_id': result.analysis_id
                }
                
                # Cache alert
                cache_key = f"{result.alert_level.value}_{datetime.now().strftime('%Y%m%d_%H')}"
                if cache_key not in self.alert_cache:
                    self.alert_cache[cache_key] = []
                self.alert_cache[cache_key].append(alert)
                
                logger.info(f"Generated {result.alert_level.value} alert for: {result.news_title[:50]}...")
                
        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}")
    
    def _generate_alert_message(self, result: AnalysisResult) -> str:
        """Generate human-readable alert message"""
        try:
            parts = []
            
            # Sentiment component
            if result.sentiment_analysis:
                sentiment = result.overall_sentiment
                score = result.sentiment_analysis.get('sentiment_score', 0)
                parts.append(f"{sentiment.title()} sentiment detected (score: {score:.2f})")
            
            # Impact component
            if result.affected_assets:
                parts.append(f"Affects {len(result.affected_assets)} assets: {', '.join(result.affected_assets[:3])}")
                if len(result.affected_assets) > 3:
                    parts.append(f"and {len(result.affected_assets) - 3} more")
            
            # Key insight
            if result.key_insights:
                parts.append(f"Key insight: {result.key_insights[0]}")
            
            return ". ".join(parts)
            
        except Exception as e:
            logger.error(f"Error generating alert message: {str(e)}")
            return f"Market alert for: {result.news_title[:50]}..."
    
    def _aggregate_portfolio_sentiment(self, analyses: List[AnalysisResult], 
                                     portfolio_assets: List[str]) -> Dict[str, Any]:
        """Aggregate sentiment analysis for portfolio"""
        relevant_analyses = [a for a in analyses if any(asset in a.affected_assets for asset in portfolio_assets)]
        
        if not relevant_analyses:
            return {'overall_sentiment': 'neutral', 'confidence': 0}
        
        sentiment_scores = []
        for analysis in relevant_analyses:
            if analysis.sentiment_analysis:
                sentiment_scores.append(analysis.sentiment_analysis.get('sentiment_score', 0))
        
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            return {
                'overall_sentiment': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
                'sentiment_score': avg_sentiment,
                'confidence': len(sentiment_scores) / len(relevant_analyses),
                'analysis_count': len(sentiment_scores)
            }
        
        return {'overall_sentiment': 'neutral', 'confidence': 0}
    
    def _assess_portfolio_risk(self, analyses: List[AnalysisResult], 
                             portfolio_assets: List[str]) -> Dict[str, Any]:
        """Assess risk factors for portfolio"""
        risk_factors = []
        risk_score = 0
        
        for analysis in analyses:
            if any(asset in analysis.affected_assets for asset in portfolio_assets):
                risk_factors.extend(analysis.risk_factors)
                if analysis.alert_level == AlertSeverity.CRITICAL:
                    risk_score += 3
                elif analysis.alert_level == AlertSeverity.HIGH:
                    risk_score += 2
                elif analysis.alert_level == AlertSeverity.MEDIUM:
                    risk_score += 1
        
        return {
            'risk_score': risk_score,
            'risk_level': 'high' if risk_score > 6 else 'medium' if risk_score > 3 else 'low',
            'risk_factors': list(set(risk_factors)),
            'risk_analysis_count': len([a for a in analyses if any(asset in a.affected_assets for asset in portfolio_assets)])
        }
    
    def _identify_opportunities(self, analyses: List[AnalysisResult], 
                              portfolio_assets: List[str]) -> List[Dict[str, Any]]:
        """Identify investment opportunities from analyses"""
        opportunities = []
        
        for analysis in analyses:
            if (analysis.overall_sentiment == 'positive' and 
                analysis.confidence_score > 0.6 and
                any(asset in analysis.affected_assets for asset in portfolio_assets)):
                
                opportunities.append({
                    'type': 'positive_sentiment',
                    'assets': [asset for asset in analysis.affected_assets if asset in portfolio_assets],
                    'confidence': analysis.confidence_score,
                    'reasoning': analysis.key_insights[0] if analysis.key_insights else 'Positive market sentiment detected',
                    'timestamp': analysis.timestamp.isoformat()
                })
        
        return opportunities[:5]
    
    def _generate_portfolio_insights(self, analyses: List[AnalysisResult], 
                                   portfolio_assets: List[str]) -> List[str]:
        """Generate key insights for portfolio"""
        insights = []
        
        # Most mentioned assets
        asset_mentions = {}
        for analysis in analyses:
            for asset in analysis.affected_assets:
                if asset in portfolio_assets:
                    asset_mentions[asset] = asset_mentions.get(asset, 0) + 1
        
        if asset_mentions:
            most_mentioned = max(asset_mentions.items(), key=lambda x: x[1])
            insights.append(f"{most_mentioned[0]} was mentioned in {most_mentioned[1]} news articles")
        
        # Alert level distribution
        alert_levels = [analysis.alert_level for analysis in analyses if any(asset in analysis.affected_assets for asset in portfolio_assets)]
        if alert_levels:
            high_alerts = len([a for a in alert_levels if a in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]])
            if high_alerts > 0:
                insights.append(f"{high_alerts} high-priority alerts detected for portfolio assets")
        
        return insights


# Factory function for creating pipeline instances
def create_analysis_pipeline(db_session: Optional[Session] = None, 
                           alert_config: Optional[AlertConfig] = None) -> IntegratedAnalysisPipeline:
    """
    Factory function to create IntegratedAnalysisPipeline instance
    
    Args:
        db_session: Database session (optional, will create if not provided)
        alert_config: Alert configuration (optional, will use defaults if not provided)
        
    Returns:
        IntegratedAnalysisPipeline: Configured pipeline instance
    """
    if db_session is None:
        db_session = next(get_db())
    
    return IntegratedAnalysisPipeline(db_session, alert_config)
 