"""
Market Impact API Routes

FastAPI routes for the market impact scoring system, providing endpoints
for analyzing market impact of news events on assets and portfolios.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.market_impact_service import MarketImpactService, create_market_impact_service
from app.database.connection import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market-impact", tags=["Market Impact"])


# Request/Response Models
class NewsImpactRequest(BaseModel):
    """Request model for news impact analysis"""
    news_title: str = Field(..., description="News article title")
    news_content: str = Field(..., description="News article content")
    news_source: str = Field(default="unknown", description="News source")
    asset_symbol: str = Field(..., description="Asset symbol to analyze impact for")


class PortfolioImpactRequest(BaseModel):
    """Request model for portfolio impact analysis"""
    asset_symbols: List[str] = Field(..., description="List of asset symbols in portfolio")
    time_window_hours: int = Field(default=24, description="Time window for analysis in hours")


class ImpactAlertRequest(BaseModel):
    """Request model for impact alert generation"""
    asset_symbol: str = Field(..., description="Asset symbol to monitor")
    impact_threshold: float = Field(default=6.0, ge=1.0, le=10.0, description="Minimum impact score (1-10)")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence (0-1)")


class ImpactResponse(BaseModel):
    """Response model for impact analysis"""
    success: bool
    data: Dict[str, Any]
    message: str
    timestamp: str


@router.post("/analyze-news", response_model=ImpactResponse)
async def analyze_news_impact(
    request: NewsImpactRequest,
    service: MarketImpactService = Depends(create_market_impact_service)
):
    """
    Analyze the market impact of a news article on a specific asset
    
    This endpoint evaluates how a news event might affect an asset's price
    using sentiment analysis, event categorization, and impact scoring algorithms.
    """
    try:
        # Create a mock news item object for the service
        class MockNewsItem:
            def __init__(self, title: str, content: str, source: str):
                self.title = title
                self.content = content
                self.source = source
                self.created_at = datetime.now()
                self.id = None
        
        mock_news = MockNewsItem(request.news_title, request.news_content, request.news_source)
        
        # Analyze impact
        impact_analysis = await service.analyze_news_impact(mock_news, request.asset_symbol)
        
        return ImpactResponse(
            success=True,
            data=impact_analysis,
            message="News impact analysis completed successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in news impact analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-portfolio", response_model=ImpactResponse)
async def analyze_portfolio_impact(
    request: PortfolioImpactRequest,
    service: MarketImpactService = Depends(create_market_impact_service)
):
    """
    Analyze market impact on an entire portfolio of assets
    
    This endpoint evaluates how recent news events might affect a portfolio
    of assets, providing aggregate impact scores and risk assessments.
    """
    try:
        portfolio_analysis = await service.analyze_portfolio_impact(
            request.asset_symbols, 
            request.time_window_hours
        )
        
        return ImpactResponse(
            success=True,
            data=portfolio_analysis,
            message="Portfolio impact analysis completed successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in portfolio impact analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")


@router.post("/generate-alerts", response_model=ImpactResponse)
async def generate_impact_alerts(
    request: ImpactAlertRequest,
    service: MarketImpactService = Depends(create_market_impact_service)
):
    """
    Generate alerts for high-impact market events
    
    This endpoint monitors recent news events and generates alerts when
    events exceed specified impact and confidence thresholds.
    """
    try:
        alerts = await service.generate_impact_alerts(
            request.asset_symbol,
            request.impact_threshold,
            request.confidence_threshold
        )
        
        return ImpactResponse(
            success=True,
            data={
                'alerts': alerts,
                'alert_count': len(alerts),
                'asset_symbol': request.asset_symbol,
                'thresholds': {
                    'impact_threshold': request.impact_threshold,
                    'confidence_threshold': request.confidence_threshold
                }
            },
            message=f"Generated {len(alerts)} impact alerts",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating impact alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Alert generation failed: {str(e)}")


@router.get("/summary/{asset_symbol}", response_model=ImpactResponse)
async def get_impact_summary(
    asset_symbol: str,
    time_window_hours: int = Query(default=24, ge=1, le=168, description="Time window in hours (1-168)"),
    service: MarketImpactService = Depends(create_market_impact_service)
):
    """
    Get a comprehensive impact summary for an asset
    
    This endpoint provides a summary of market impact events for a specific
    asset over a given time window, including aggregate scores and statistics.
    """
    try:
        impact_summary = await service.get_impact_summary(asset_symbol, time_window_hours)
        
        return ImpactResponse(
            success=True,
            data=impact_summary,
            message="Impact summary retrieved successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting impact summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summary retrieval failed: {str(e)}")


@router.get("/real-time-feed", response_model=ImpactResponse)
async def get_real_time_impact_feed(
    asset_symbols: List[str] = Query(..., description="Asset symbols to monitor"),
    service: MarketImpactService = Depends(create_market_impact_service)
):
    """
    Get real-time impact feed for multiple assets
    
    This endpoint provides real-time market impact data for a list of assets,
    focusing on the most recent high-confidence impact events.
    """
    try:
        if len(asset_symbols) > 20:  # Limit to prevent overload
            raise HTTPException(status_code=400, detail="Maximum 20 asset symbols allowed")
        
        impact_feed = await service.get_real_time_impact_feed(asset_symbols)
        
        return ImpactResponse(
            success=True,
            data=impact_feed,
            message="Real-time impact feed retrieved successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting real-time impact feed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feed retrieval failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the market impact service
    """
    return {
        "status": "healthy",
        "service": "market_impact",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@router.get("/event-categories")
async def get_event_categories():
    """
    Get available event categories for market impact analysis
    """
    from app.analysis.market_impact import EventCategory
    
    categories = [
        {
            "value": category.value,
            "name": category.value.replace('_', ' ').title(),
            "description": _get_category_description(category)
        }
        for category in EventCategory
    ]
    
    return {
        "categories": categories,
        "total_count": len(categories),
        "timestamp": datetime.now().isoformat()
    }


def _get_category_description(category) -> str:
    """Get description for event category"""
    descriptions = {
        "earnings": "Quarterly or annual earnings reports and financial results",
        "merger_acquisition": "Mergers, acquisitions, takeovers, and corporate restructuring",
        "regulatory": "Government regulations, compliance issues, and regulatory changes", 
        "geopolitical": "Political events, conflicts, trade wars, and sanctions",
        "economic_indicator": "Economic data releases like GDP, inflation, unemployment",
        "company_news": "General company announcements and corporate news",
        "market_volatility": "Market-wide volatility events and systematic risks",
        "sector_news": "Industry-specific news affecting multiple companies",
        "earnings_guidance": "Forward-looking guidance and outlook statements",
        "analyst_upgrade": "Analyst upgrades, positive ratings changes",
        "analyst_downgrade": "Analyst downgrades, negative ratings changes"
    }
    
    return descriptions.get(category.value, "General market event category")


# Impact scoring utilities endpoint
@router.get("/scoring-info")
async def get_scoring_info():
    """
    Get information about the impact scoring methodology
    """
    return {
        "scoring_methodology": {
            "impact_score_range": "1-10",
            "confidence_range": "0-1",
            "direction_options": ["positive", "negative", "neutral"],
            "confidence_levels": ["very_low", "low", "medium", "high", "very_high"]
        },
        "factor_weights": {
            "sentiment_strength": 0.25,
            "news_volume": 0.20,
            "asset_relevance": 0.15,
            "source_credibility": 0.15,
            "time_decay": 0.10,
            "market_conditions": 0.10,
            "historical_correlation": 0.05
        },
        "category_weights_sample": {
            "earnings": 0.9,
            "merger_acquisition": 0.85,
            "regulatory": 0.8,
            "geopolitical": 0.7
        },
        "description": "Market impact scores combine multiple factors to assess potential price movements",
        "timestamp": datetime.now().isoformat()
    } 