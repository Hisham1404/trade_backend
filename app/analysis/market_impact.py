"""
Market Impact Scoring System

This module implements a comprehensive market impact scoring system that evaluates
the potential influence of news events, sentiment changes, and market conditions
on asset prices using advanced algorithms and machine learning techniques.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class EventCategory(Enum):
    """Event categories for market impact assessment"""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    GEOPOLITICAL = "geopolitical"
    ECONOMIC_INDICATOR = "economic_indicator"
    COMPANY_NEWS = "company_news"
    MARKET_VOLATILITY = "market_volatility"
    SECTOR_NEWS = "sector_news"
    EARNINGS_GUIDANCE = "earnings_guidance"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"


class ImpactDirection(Enum):
    """Direction of market impact"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ConfidenceLevel(Enum):
    """Confidence levels for impact predictions"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ImpactScore:
    """Market impact score with metadata"""
    score: float  # 1-10 scale
    direction: ImpactDirection
    confidence: float  # 0-1 scale
    confidence_level: ConfidenceLevel
    reasoning: str
    category: EventCategory
    timestamp: datetime
    factors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'score': self.score,
            'direction': self.direction.value,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'reasoning': self.reasoning,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'factors': self.factors
        }


@dataclass
class NewsEvent:
    """News event for impact analysis"""
    title: str
    content: str
    source: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    entities: List[str]
    asset_symbols: List[str]
    category: Optional[EventCategory] = None
    relevance_score: float = 0.0


class MarketImpactScorer:
    """
    Comprehensive market impact scoring system that evaluates the potential
    influence of news events, sentiment changes, and market conditions.
    """
    
    def __init__(self):
        self.category_weights = {
            EventCategory.EARNINGS: 0.9,
            EventCategory.MERGER_ACQUISITION: 0.85,
            EventCategory.REGULATORY: 0.8,
            EventCategory.GEOPOLITICAL: 0.7,
            EventCategory.ECONOMIC_INDICATOR: 0.75,
            EventCategory.COMPANY_NEWS: 0.6,
            EventCategory.MARKET_VOLATILITY: 0.65,
            EventCategory.SECTOR_NEWS: 0.55,
            EventCategory.EARNINGS_GUIDANCE: 0.8,
            EventCategory.ANALYST_UPGRADE: 0.7,
            EventCategory.ANALYST_DOWNGRADE: 0.75
        }
        
        self.sentiment_multipliers = {
            'very_positive': 1.2,
            'positive': 1.1,
            'neutral': 1.0,
            'negative': 1.15,
            'very_negative': 1.25
        }
        
        # Factor weights for impact calculation
        self.factor_weights = {
            'sentiment_strength': 0.25,
            'news_volume': 0.20,
            'asset_relevance': 0.15,
            'source_credibility': 0.15,
            'time_decay': 0.10,
            'market_conditions': 0.10,
            'historical_correlation': 0.05
        }
    
    def categorize_event(self, news_event: NewsEvent) -> EventCategory:
        """
        Categorize news event based on content analysis
        
        Args:
            news_event: News event to categorize
            
        Returns:
            EventCategory: Predicted category
        """
        content_lower = (news_event.title + " " + news_event.content).lower()
        
        # Keyword-based categorization (can be enhanced with ML)
        if any(word in content_lower for word in ['earnings', 'quarterly', 'revenue', 'profit']):
            return EventCategory.EARNINGS
        elif any(word in content_lower for word in ['merger', 'acquisition', 'takeover', 'buyout']):
            return EventCategory.MERGER_ACQUISITION
        elif any(word in content_lower for word in ['regulation', 'regulatory', 'compliance', 'sec', 'fda']):
            return EventCategory.REGULATORY
        elif any(word in content_lower for word in ['war', 'conflict', 'trade war', 'sanctions', 'geopolitical']):
            return EventCategory.GEOPOLITICAL
        elif any(word in content_lower for word in ['gdp', 'inflation', 'unemployment', 'interest rate', 'fed']):
            return EventCategory.ECONOMIC_INDICATOR
        elif any(word in content_lower for word in ['upgrade', 'outperform', 'overweight']):
            return EventCategory.ANALYST_UPGRADE
        elif any(word in content_lower for word in ['downgrade', 'underperform', 'underweight']):
            return EventCategory.ANALYST_DOWNGRADE
        elif any(word in content_lower for word in ['guidance', 'forecast', 'outlook']):
            return EventCategory.EARNINGS_GUIDANCE
        else:
            return EventCategory.COMPANY_NEWS
    
    def calculate_sentiment_strength(self, sentiment_score: float) -> Tuple[str, float]:
        """
        Calculate sentiment strength and category
        
        Args:
            sentiment_score: Sentiment score (-1 to 1)
            
        Returns:
            Tuple of (sentiment_category, strength_factor)
        """
        abs_sentiment = abs(sentiment_score)
        
        if abs_sentiment >= 0.8:
            category = 'very_positive' if sentiment_score > 0 else 'very_negative'
        elif abs_sentiment >= 0.5:
            category = 'positive' if sentiment_score > 0 else 'negative'
        else:
            category = 'neutral'
        
        # Strength factor based on absolute sentiment
        strength_factor = min(abs_sentiment * 2, 1.0)  # Scale to 0-1
        
        return category, strength_factor
    
    def calculate_time_decay_factor(self, event_timestamp: datetime, 
                                   current_time: Optional[datetime] = None) -> float:
        """
        Calculate time decay factor for event impact
        
        Args:
            event_timestamp: When the event occurred
            current_time: Current time (defaults to now)
            
        Returns:
            float: Decay factor (0-1)
        """
        if current_time is None:
            current_time = datetime.now()
        
        time_diff = current_time - event_timestamp
        hours_passed = time_diff.total_seconds() / 3600
        
        # Exponential decay with half-life of 24 hours
        decay_factor = np.exp(-0.693 * hours_passed / 24)
        return max(decay_factor, 0.1)  # Minimum 10% impact retention
    
    def calculate_news_volume_factor(self, asset_symbol: str, 
                                   time_window_hours: int = 24) -> float:
        """
        Calculate news volume factor for an asset
        
        Args:
            asset_symbol: Asset symbol
            time_window_hours: Time window for volume calculation
            
        Returns:
            float: Volume factor (0-2)
        """
        # This would typically query a database for recent news count
        # For now, return a baseline factor
        # TODO: Implement actual news volume calculation from database
        
        # Placeholder implementation
        baseline_volume = 5  # Average news items per day
        current_volume = 8   # This would be calculated from actual data
        
        volume_factor = min(current_volume / baseline_volume, 2.0)
        return volume_factor
    
    def calculate_asset_relevance_score(self, news_event: NewsEvent, 
                                      asset_symbol: str) -> float:
        """
        Calculate how relevant a news event is to a specific asset
        
        Args:
            news_event: News event
            asset_symbol: Asset symbol
            
        Returns:
            float: Relevance score (0-1)
        """
        relevance_score = 0.0
        content = (news_event.title + " " + news_event.content).lower()
        asset_lower = asset_symbol.lower()
        
        # Direct mention of asset symbol
        if asset_lower in content:
            relevance_score += 0.5
        
        # Mention in entities
        if asset_symbol in news_event.asset_symbols:
            relevance_score += 0.3
        
        # Sector relevance (simplified)
        if asset_symbol in news_event.entities:
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)
    
    def get_source_credibility_score(self, source: str) -> float:
        """
        Get credibility score for news source
        
        Args:
            source: News source name
            
        Returns:
            float: Credibility score (0-1)
        """
        # Credibility mapping (can be enhanced with external ratings)
        high_credibility = ['reuters', 'bloomberg', 'wsj', 'ft', 'cnbc', 'marketwatch']
        medium_credibility = ['yahoo', 'cnn', 'bbc', 'fortune', 'business insider']
        
        source_lower = source.lower()
        
        if any(cred in source_lower for cred in high_credibility):
            return 0.9
        elif any(cred in source_lower for cred in medium_credibility):
            return 0.7
        else:
            return 0.5  # Default credibility
    
    def get_market_conditions_factor(self) -> float:
        """
        Get current market conditions factor
        
        Returns:
            float: Market conditions factor (0.5-1.5)
        """
        # This would typically analyze market volatility, volume, etc.
        # For now, return a neutral factor
        # TODO: Implement actual market conditions analysis
        
        return 1.0  # Neutral market conditions
    
    def calculate_base_impact_score(self, news_event: NewsEvent, 
                                  asset_symbol: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate base impact score using weighted factors
        
        Args:
            news_event: News event
            asset_symbol: Asset symbol
            
        Returns:
            Tuple of (base_impact_score, factors_dict)
        """
        # Calculate individual factors
        sentiment_category, sentiment_strength = self.calculate_sentiment_strength(
            news_event.sentiment_score
        )
        
        factors = {
            'sentiment_strength': sentiment_strength,
            'news_volume': self.calculate_news_volume_factor(asset_symbol),
            'asset_relevance': self.calculate_asset_relevance_score(news_event, asset_symbol),
            'source_credibility': self.get_source_credibility_score(news_event.source),
            'time_decay': self.calculate_time_decay_factor(news_event.timestamp),
            'market_conditions': self.get_market_conditions_factor(),
            'historical_correlation': 0.7  # Placeholder for historical correlation
        }
        
        # Calculate weighted sum
        weighted_score = sum(
            factors[factor] * weight 
            for factor, weight in self.factor_weights.items()
        )
        
        # Apply category weight
        category_weight = self.category_weights.get(news_event.category, 0.5)
        
        # Apply sentiment multiplier
        sentiment_multiplier = self.sentiment_multipliers.get(sentiment_category, 1.0)
        
        # Final score calculation (1-10 scale)
        base_score = weighted_score * category_weight * sentiment_multiplier * 10
        final_score = max(1.0, min(base_score, 10.0))
        
        return final_score, factors
    
    def calculate_confidence_score(self, factors: Dict[str, float], 
                                 news_event: NewsEvent) -> Tuple[float, ConfidenceLevel]:
        """
        Calculate confidence score for the impact prediction
        
        Args:
            factors: Calculated factors
            news_event: News event
            
        Returns:
            Tuple of (confidence_score, confidence_level)
        """
        # Factors that contribute to confidence
        source_credibility = factors['source_credibility']
        asset_relevance = factors['asset_relevance']
        sentiment_strength = factors['sentiment_strength']
        
        # Calculate confidence based on these factors
        confidence = (
            source_credibility * 0.4 +
            asset_relevance * 0.35 +
            sentiment_strength * 0.25
        )
        
        # Determine confidence level
        if confidence >= 0.8:
            level = ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.65:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            level = ConfidenceLevel.MEDIUM
        elif confidence >= 0.35:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
        
        return confidence, level
    
    def determine_impact_direction(self, news_event: NewsEvent, 
                                 impact_score: float) -> ImpactDirection:
        """
        Determine the direction of market impact
        
        Args:
            news_event: News event
            impact_score: Calculated impact score
            
        Returns:
            ImpactDirection: Direction of impact
        """
        sentiment_score = news_event.sentiment_score
        
        # For low impact scores, consider neutral
        if impact_score < 3.0:
            return ImpactDirection.NEUTRAL
        
        # For high impact scores, use sentiment to determine direction
        if sentiment_score > 0.1:
            return ImpactDirection.POSITIVE
        elif sentiment_score < -0.1:
            return ImpactDirection.NEGATIVE
        else:
            return ImpactDirection.NEUTRAL
    
    def generate_reasoning(self, news_event: NewsEvent, impact_score: float,
                         factors: Dict[str, float], direction: ImpactDirection) -> str:
        """
        Generate human-readable reasoning for the impact score
        
        Args:
            news_event: News event
            impact_score: Calculated impact score
            factors: Calculated factors
            direction: Impact direction
            
        Returns:
            str: Reasoning explanation
        """
        category = news_event.category.value.replace('_', ' ').title()
        sentiment_desc = "positive" if news_event.sentiment_score > 0 else "negative"
        
        reasoning_parts = [
            f"Impact score of {impact_score:.1f} assigned to {category.lower()} event",
            f"with {sentiment_desc} sentiment ({news_event.sentiment_score:.2f})"
        ]
        
        # Add key contributing factors
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
        for factor_name, factor_value in top_factors:
            factor_desc = factor_name.replace('_', ' ').title()
            reasoning_parts.append(f"{factor_desc}: {factor_value:.2f}")
        
        # Add direction reasoning
        if direction == ImpactDirection.POSITIVE:
            reasoning_parts.append("Expected to drive prices higher")
        elif direction == ImpactDirection.NEGATIVE:
            reasoning_parts.append("Expected to pressure prices lower")
        else:
            reasoning_parts.append("Limited directional impact expected")
        
        return ". ".join(reasoning_parts) + "."
    
    async def calculate_market_impact(self, news_event: NewsEvent, 
                                    asset_symbol: str) -> ImpactScore:
        """
        Calculate comprehensive market impact score for a news event and asset
        
        Args:
            news_event: News event to analyze
            asset_symbol: Asset symbol to analyze impact for
            
        Returns:
            ImpactScore: Comprehensive impact assessment
        """
        try:
            # Categorize the event if not already done
            if news_event.category is None:
                news_event.category = self.categorize_event(news_event)
            
            # Calculate base impact score and factors
            impact_score, factors = self.calculate_base_impact_score(news_event, asset_symbol)
            
            # Calculate confidence
            confidence, confidence_level = self.calculate_confidence_score(factors, news_event)
            
            # Determine direction
            direction = self.determine_impact_direction(news_event, impact_score)
            
            # Generate reasoning
            reasoning = self.generate_reasoning(news_event, impact_score, factors, direction)
            
            return ImpactScore(
                score=impact_score,
                direction=direction,
                confidence=confidence,
                confidence_level=confidence_level,
                reasoning=reasoning,
                category=news_event.category,
                timestamp=datetime.now(),
                factors=factors
            )
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {str(e)}")
            raise
    
    async def batch_calculate_impact(self, news_events: List[NewsEvent], 
                                   asset_symbols: List[str]) -> Dict[str, List[ImpactScore]]:
        """
        Calculate market impact for multiple news events and assets
        
        Args:
            news_events: List of news events
            asset_symbols: List of asset symbols
            
        Returns:
            Dict mapping asset symbols to impact scores
        """
        results = {symbol: [] for symbol in asset_symbols}
        
        for news_event in news_events:
            for asset_symbol in asset_symbols:
                # Calculate relevance first to filter irrelevant combinations
                relevance = self.calculate_asset_relevance_score(news_event, asset_symbol)
                
                if relevance > 0.1:  # Only calculate for relevant combinations
                    impact_score = await self.calculate_market_impact(news_event, asset_symbol)
                    results[asset_symbol].append(impact_score)
        
        return results
    
    def aggregate_impact_scores(self, impact_scores: List[ImpactScore], 
                              time_window_hours: int = 24) -> ImpactScore:
        """
        Aggregate multiple impact scores into a single score
        
        Args:
            impact_scores: List of impact scores to aggregate
            time_window_hours: Time window for aggregation
            
        Returns:
            ImpactScore: Aggregated impact score
        """
        if not impact_scores:
            raise ValueError("No impact scores to aggregate")
        
        # Filter by time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_scores = [score for score in impact_scores if score.timestamp >= cutoff_time]
        
        if not recent_scores:
            recent_scores = impact_scores[-5:]  # Use last 5 if none in time window
        
        # Weighted average based on confidence and recency
        total_weight = 0
        weighted_score = 0
        weighted_confidence = 0
        
        for score in recent_scores:
            # Weight by confidence and time decay
            time_weight = self.calculate_time_decay_factor(score.timestamp)
            weight = score.confidence * time_weight
            
            weighted_score += score.score * weight
            weighted_confidence += score.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            total_weight = 1
        
        final_score = weighted_score / total_weight
        final_confidence = weighted_confidence / total_weight
        
        # Determine overall direction
        positive_weight = sum(s.confidence for s in recent_scores if s.direction == ImpactDirection.POSITIVE)
        negative_weight = sum(s.confidence for s in recent_scores if s.direction == ImpactDirection.NEGATIVE)
        
        if positive_weight > negative_weight:
            direction = ImpactDirection.POSITIVE
        elif negative_weight > positive_weight:
            direction = ImpactDirection.NEGATIVE
        else:
            direction = ImpactDirection.NEUTRAL
        
        # Determine confidence level
        if final_confidence >= 0.8:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif final_confidence >= 0.65:
            confidence_level = ConfidenceLevel.HIGH
        elif final_confidence >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        elif final_confidence >= 0.35:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW
        
        return ImpactScore(
            score=final_score,
            direction=direction,
            confidence=final_confidence,
            confidence_level=confidence_level,
            reasoning=f"Aggregated score from {len(recent_scores)} events in last {time_window_hours} hours",
            category=recent_scores[0].category,  # Use most recent category
            timestamp=datetime.now(),
            factors={'aggregated_from': len(recent_scores)}
        )


class PredictiveImpactAnalyzer:
    """
    Predictive impact analyzer using machine learning for enhanced accuracy
    """
    
    def __init__(self, scorer: MarketImpactScorer):
        self.scorer = scorer
        self.historical_data = []  # Store historical impact vs actual outcome
        
    def train_prediction_model(self, historical_data: List[Dict]) -> None:
        """
        Train predictive model with historical data
        
        Args:
            historical_data: Historical impact scores and actual outcomes
        """
        # TODO: Implement ML model training with historical data
        # This would use features like sentiment, volume, category, etc.
        # to predict actual price movements and improve impact scoring
        
        logger.info(f"Training model with {len(historical_data)} historical records")
        self.historical_data = historical_data
    
    async def predict_price_movement(self, impact_score: ImpactScore, 
                                   asset_symbol: str) -> Dict[str, float]:
        """
        Predict price movement based on impact score
        
        Args:
            impact_score: Calculated impact score
            asset_symbol: Asset symbol
            
        Returns:
            Dict with prediction metrics
        """
        # This would use the trained ML model to predict:
        # - Expected price change percentage
        # - Probability of direction being correct
        # - Time horizon for impact realization
        
        # Placeholder implementation
        base_change = impact_score.score / 100  # Simple conversion
        
        if impact_score.direction == ImpactDirection.POSITIVE:
            expected_change = base_change * impact_score.confidence
        elif impact_score.direction == ImpactDirection.NEGATIVE:
            expected_change = -base_change * impact_score.confidence
        else:
            expected_change = 0
        
        return {
            'expected_price_change_pct': expected_change,
            'direction_probability': impact_score.confidence,
            'impact_horizon_hours': 24,
            'volatility_increase_factor': 1 + (impact_score.score / 20)
        }
    
    async def enhanced_impact_analysis(self, news_event: NewsEvent, 
                                     asset_symbol: str) -> Dict[str, Any]:
        """
        Perform enhanced impact analysis combining scoring and prediction
        
        Args:
            news_event: News event to analyze
            asset_symbol: Asset symbol
            
        Returns:
            Dict with comprehensive analysis results
        """
        # Calculate base impact score
        impact_score = await self.scorer.calculate_market_impact(news_event, asset_symbol)
        
        # Generate price movement prediction
        price_prediction = await self.predict_price_movement(impact_score, asset_symbol)
        
        # Combine results
        return {
            'impact_score': impact_score.to_dict(),
            'price_prediction': price_prediction,
            'risk_assessment': {
                'volatility_risk': 'high' if impact_score.score > 7 else 'medium' if impact_score.score > 4 else 'low',
                'confidence_risk': 'low' if impact_score.confidence > 0.7 else 'medium' if impact_score.confidence > 0.4 else 'high',
                'time_sensitivity': 'immediate' if impact_score.category in [EventCategory.EARNINGS, EventCategory.MERGER_ACQUISITION] else 'short_term'
            },
            'recommended_actions': self._generate_action_recommendations(impact_score, price_prediction)
        }
    
    def _generate_action_recommendations(self, impact_score: ImpactScore, 
                                       price_prediction: Dict[str, float]) -> List[str]:
        """
        Generate actionable recommendations based on impact analysis
        
        Args:
            impact_score: Calculated impact score
            price_prediction: Price prediction results
            
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        # High impact recommendations
        if impact_score.score > 7 and impact_score.confidence > 0.6:
            if impact_score.direction == ImpactDirection.POSITIVE:
                recommendations.append("Consider increasing long positions")
                recommendations.append("Monitor for profit-taking opportunities")
            elif impact_score.direction == ImpactDirection.NEGATIVE:
                recommendations.append("Consider reducing exposure or hedging")
                recommendations.append("Evaluate stop-loss levels")
        
        # Medium impact recommendations
        elif impact_score.score > 4:
            recommendations.append("Monitor price action closely")
            recommendations.append("Consider volatility trading strategies")
        
        # General recommendations
        if impact_score.confidence < 0.5:
            recommendations.append("Exercise caution due to low confidence")
        
        if price_prediction['volatility_increase_factor'] > 1.3:
            recommendations.append("Expect increased volatility")
        
        return recommendations


# Utility functions for integration with existing systems
def create_news_event_from_db(news_item, sentiment_score: float = 0.0) -> NewsEvent:
    """
    Create NewsEvent from database news item
    
    Args:
        news_item: Database news item object
        sentiment_score: Calculated sentiment score
        
    Returns:
        NewsEvent: Formatted news event
    """
    return NewsEvent(
        title=news_item.title or "",
        content=news_item.content or "",
        source=news_item.source or "unknown",
        timestamp=news_item.created_at or datetime.now(),
        sentiment_score=sentiment_score,
        entities=news_item.entities or [],
        asset_symbols=news_item.related_assets or []
    )


async def calculate_portfolio_impact(scorer: MarketImpactScorer, 
                                   news_events: List[NewsEvent],
                                   portfolio_assets: List[str]) -> Dict[str, Any]:
    """
    Calculate overall portfolio impact from multiple news events
    
    Args:
        scorer: Market impact scorer instance
        news_events: List of news events
        portfolio_assets: List of assets in portfolio
        
    Returns:
        Dict with portfolio impact analysis
    """
    # Calculate impact for all assets
    impact_results = await scorer.batch_calculate_impact(news_events, portfolio_assets)
    
    # Aggregate portfolio-level metrics
    total_positive_impact = 0
    total_negative_impact = 0
    highest_risk_asset = None
    highest_impact_score = 0
    
    asset_summaries = {}
    
    for asset, impacts in impact_results.items():
        if not impacts:
            continue
            
        # Aggregate impacts for this asset
        aggregated = scorer.aggregate_impact_scores(impacts)
        asset_summaries[asset] = aggregated.to_dict()
        
        # Track highest impact
        if aggregated.score > highest_impact_score:
            highest_impact_score = aggregated.score
            highest_risk_asset = asset
        
        # Sum positive and negative impacts
        if aggregated.direction == ImpactDirection.POSITIVE:
            total_positive_impact += aggregated.score * aggregated.confidence
        elif aggregated.direction == ImpactDirection.NEGATIVE:
            total_negative_impact += aggregated.score * aggregated.confidence
    
    # Calculate net impact
    net_impact = total_positive_impact - total_negative_impact
    
    return {
        'portfolio_summary': {
            'net_impact_score': net_impact,
            'total_positive_impact': total_positive_impact,
            'total_negative_impact': total_negative_impact,
            'highest_risk_asset': highest_risk_asset,
            'highest_impact_score': highest_impact_score,
            'num_assets_analyzed': len([a for a in asset_summaries.values() if a])
        },
        'asset_impacts': asset_summaries,
        'overall_recommendation': (
            "BULLISH" if net_impact > 5 else
            "BEARISH" if net_impact < -5 else
            "NEUTRAL"
        )
    }
