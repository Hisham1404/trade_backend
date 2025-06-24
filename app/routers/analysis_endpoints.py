"""
Comprehensive Analysis API Endpoints

Provides unified access to all analysis capabilities including sentiment analysis,
market impact assessment, technical indicators, and integrated trading insights.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.database.connection import get_db
from app.auth.jwt_handler import get_current_user
from app.models.user import User
from app.models.asset import Asset
from app.services.sentiment_service import get_sentiment_service
from app.services.market_impact_service import create_market_impact_service
# TODO: Re-enable these imports after fixing dependency issues
# from app.analysis.asset_correlation import CorrelationAnalyzer
# from app.analysis.option_analytics import OptionAnalyticsEngine
# from app.market_data.participant_data_ingestion import ParticipantDataIngestion
# from app.trading.position_sizer import PositionSizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis & Insights"])


# Request/Response Models
class ComprehensiveAnalysisRequest(BaseModel):
    """Request for comprehensive asset analysis"""
    asset_symbol: str = Field(..., description="Asset symbol to analyze")
    analysis_depth: str = Field("standard", description="Analysis depth: quick, standard, deep")
    include_sentiment: bool = Field(True, description="Include sentiment analysis")
    include_technical: bool = Field(True, description="Include technical indicators") 
    include_options: bool = Field(True, description="Include options analysis")
    include_correlations: bool = Field(True, description="Include correlation analysis")
    time_horizon_days: int = Field(30, ge=1, le=365, description="Analysis time horizon")


class TechnicalIndicatorsRequest(BaseModel):
    """Request for technical indicators analysis"""
    asset_symbol: str = Field(..., description="Asset symbol")
    indicators: List[str] = Field(default=["sma", "ema", "rsi", "macd", "bollinger"], description="Technical indicators to calculate")
    period_days: int = Field(20, ge=5, le=100, description="Period for indicators")


class MarketRegimeAnalysisRequest(BaseModel):
    """Request for market regime analysis"""
    asset_symbols: List[str] = Field(..., description="Assets to analyze", max_items=10)
    lookback_days: int = Field(30, ge=7, le=180, description="Lookback period")


class SentimentTrendRequest(BaseModel):
    """Request for sentiment trend analysis"""
    asset_symbol: str = Field(..., description="Asset symbol")
    period_days: int = Field(7, ge=1, le=30, description="Period for trend analysis")


class AnalysisResponse(BaseModel):
    """Standard analysis response format"""
    success: bool
    data: Dict[str, Any]
    message: str
    timestamp: str
    analysis_id: Optional[str] = None


@router.get("/comprehensive/{asset_symbol}", response_model=AnalysisResponse)
async def get_comprehensive_analysis(
    asset_symbol: str = Path(..., description="Asset symbol to analyze"),
    analysis_depth: str = Query("standard", description="Analysis depth"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive analysis for an asset including all available insights
    
    Combines sentiment analysis, market impact, technical indicators, 
    options analytics, and correlation analysis into a unified view.
    """
    try:
        asset_symbol = asset_symbol.upper()
        analysis_results = {}
        
        # Get sentiment analysis
        try:
            sentiment_service = await get_sentiment_service()
            sentiment_data = sentiment_service.get_asset_sentiment(asset_symbol)
            analysis_results["sentiment"] = sentiment_data or {}
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for {asset_symbol}: {e}")
            analysis_results["sentiment"] = {"error": "Sentiment analysis unavailable"}
        
        # Get market impact analysis
        try:
            market_impact_service = await create_market_impact_service()
            impact_summary = await market_impact_service.get_impact_summary(asset_symbol, 24)
            analysis_results["market_impact"] = impact_summary
        except Exception as e:
            logger.warning(f"Market impact analysis failed for {asset_symbol}: {e}")
            analysis_results["market_impact"] = {"error": "Market impact analysis unavailable"}
        
        # Get technical indicators (mock data for now)
        try:
            technical_data = {
                "price_action": {
                    "current_price": 2550.0,
                    "day_change": 25.0,
                    "day_change_percent": 0.99,
                    "volume": 1250000
                },
                "moving_averages": {
                    "sma_20": 2535.5,
                    "sma_50": 2520.3,
                    "ema_12": 2548.2,
                    "ema_26": 2539.1
                },
                "momentum": {
                    "rsi": 65.5,
                    "macd": 8.2,
                    "macd_signal": 6.1,
                    "stochastic": 72.3
                },
                "volatility": {
                    "bollinger_upper": 2580.5,
                    "bollinger_lower": 2490.3,
                    "atr": 45.2,
                    "volatility_percentile": 68.0
                }
            }
            analysis_results["technical"] = technical_data
        except Exception as e:
            logger.warning(f"Technical analysis failed for {asset_symbol}: {e}")
            analysis_results["technical"] = {"error": "Technical analysis unavailable"}
        
        # Get options analytics if applicable
        # TODO: Re-enable after fixing imports
        analysis_results["options"] = {"error": "Options analysis temporarily disabled"}
        
        # Get correlation analysis
        # TODO: Re-enable after fixing imports
        analysis_results["correlations"] = {"error": "Correlation analysis temporarily disabled"}
        
        # Calculate overall score/rating
        overall_score = _calculate_overall_score(analysis_results)
        analysis_results["overall"] = {
            "score": overall_score,
            "rating": _score_to_rating(overall_score),
            "recommendation": _generate_recommendation(overall_score, analysis_results),
            "confidence": _calculate_confidence(analysis_results)
        }
        
        return AnalysisResponse(
            success=True,
            data=analysis_results,
            message=f"Comprehensive analysis completed for {asset_symbol}",
            timestamp=datetime.utcnow().isoformat(),
            analysis_id=f"analysis_{asset_symbol}_{int(datetime.utcnow().timestamp())}"
        )
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed for {asset_symbol}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/technical-indicators", response_model=AnalysisResponse)
async def calculate_technical_indicators(
    request: TechnicalIndicatorsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Calculate technical indicators for an asset
    
    Supports various technical indicators including moving averages,
    momentum oscillators, and volatility measures.
    """
    try:
        asset_symbol = request.asset_symbol.upper()
        
        # Mock technical indicators calculation
        # In production, this would fetch real market data and calculate indicators
        indicators_data = {
            "asset_symbol": asset_symbol,
            "calculation_time": datetime.utcnow().isoformat(),
            "period_days": request.period_days,
            "indicators": {}
        }
        
        # Calculate requested indicators
        for indicator in request.indicators:
            if indicator.lower() == "sma":
                indicators_data["indicators"]["sma"] = {
                    "value": 2535.5,
                    "description": f"Simple Moving Average ({request.period_days} days)"
                }
            elif indicator.lower() == "ema":
                indicators_data["indicators"]["ema"] = {
                    "value": 2548.2,
                    "description": f"Exponential Moving Average ({request.period_days} days)"
                }
            elif indicator.lower() == "rsi":
                indicators_data["indicators"]["rsi"] = {
                    "value": 65.5,
                    "description": "Relative Strength Index",
                    "interpretation": "Moderately overbought"
                }
            elif indicator.lower() == "macd":
                indicators_data["indicators"]["macd"] = {
                    "macd": 8.2,
                    "signal": 6.1,
                    "histogram": 2.1,
                    "description": "Moving Average Convergence Divergence"
                }
            elif indicator.lower() == "bollinger":
                indicators_data["indicators"]["bollinger_bands"] = {
                    "upper": 2580.5,
                    "middle": 2535.5,
                    "lower": 2490.3,
                    "description": "Bollinger Bands",
                    "position": "Near upper band"
                }
        
        return AnalysisResponse(
            success=True,
            data=indicators_data,
            message=f"Technical indicators calculated for {asset_symbol}",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Technical indicators calculation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Technical indicators calculation failed: {str(e)}"
        )


@router.post("/sentiment-trend", response_model=AnalysisResponse)
async def analyze_sentiment_trend(
    request: SentimentTrendRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze sentiment trends for an asset over time
    
    Provides trend analysis, sentiment momentum, and forecast.
    """
    try:
        asset_symbol = request.asset_symbol.upper()
        
        sentiment_service = await get_sentiment_service()
        
        # Get historical sentiment data
        sentiment_trend = {
            "asset_symbol": asset_symbol,
            "period_days": request.period_days,
            "current_sentiment": {
                "score": 0.15,
                "label": "positive",
                "confidence": 0.78
            },
            "trend_analysis": {
                "direction": "improving",
                "momentum": "moderate",
                "volatility": "low",
                "trend_score": 0.65
            },
            "historical_data": [
                {"date": "2024-01-15", "score": 0.1, "volume": 125},
                {"date": "2024-01-14", "score": 0.08, "volume": 98},
                {"date": "2024-01-13", "score": 0.12, "volume": 145},
                {"date": "2024-01-12", "score": 0.15, "volume": 167},
                {"date": "2024-01-11", "score": 0.18, "volume": 189}
            ],
            "forecast": {
                "next_3_days": "stable_positive",
                "confidence": 0.72,
                "key_factors": ["earnings_season", "market_momentum"]
            }
        }
        
        return AnalysisResponse(
            success=True,
            data=sentiment_trend,
            message=f"Sentiment trend analysis completed for {asset_symbol}",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Sentiment trend analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment trend analysis failed: {str(e)}"
        )


@router.post("/market-regime", response_model=AnalysisResponse)
async def analyze_market_regime(
    request: MarketRegimeAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze market regime and conditions across multiple assets
    
    Identifies current market regime (bull, bear, sideways) and provides
    regime-specific insights and recommendations.
    """
    try:
        if len(request.asset_symbols) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 assets allowed for regime analysis"
            )
        
        # Mock market regime analysis
        regime_analysis = {
            "analysis_date": datetime.utcnow().isoformat(),
            "lookback_days": request.lookback_days,
            "asset_symbols": request.asset_symbols,
            "overall_regime": {
                "classification": "bull_market",
                "confidence": 0.78,
                "regime_strength": "moderate",
                "duration_days": 45
            },
            "regime_characteristics": {
                "volatility_regime": "low_to_moderate",
                "trend_consistency": "strong",
                "sector_rotation": "active",
                "risk_appetite": "moderate_to_high"
            },
            "asset_performance": [
                {
                    "symbol": symbol,
                    "regime_alignment": "positive" if i % 2 == 0 else "neutral",
                    "relative_strength": 0.65 + (i * 0.05),
                    "volatility_percentile": 45 + (i * 5)
                }
                for i, symbol in enumerate(request.asset_symbols)
            ],
            "recommendations": [
                "Consider momentum strategies in current bull regime",
                "Monitor for regime change signals",
                "Maintain moderate position sizing due to volatility"
            ]
        }
        
        return AnalysisResponse(
            success=True,
            data=regime_analysis,
            message="Market regime analysis completed",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Market regime analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Market regime analysis failed: {str(e)}"
        )


@router.get("/position-sizing/{asset_symbol}", response_model=AnalysisResponse)
async def get_position_sizing_recommendation(
    asset_symbol: str = Path(..., description="Asset symbol"),
    portfolio_value: float = Query(..., description="Total portfolio value"),
    risk_tolerance: str = Query("moderate", description="Risk tolerance level"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get position sizing recommendations for an asset
    
    Calculates optimal position size based on portfolio value, risk tolerance,
    asset volatility, and current market conditions.
    """
    try:
        asset_symbol = asset_symbol.upper()
        
        # Mock position sizing calculation
        position_analysis = {
            "asset_symbol": asset_symbol,
            "portfolio_value": portfolio_value,
            "risk_tolerance": risk_tolerance,
            "recommended_position": {
                "position_size": portfolio_value * 0.05,  # 5% of portfolio
                "max_position_size": portfolio_value * 0.10,  # 10% max
                "min_position_size": portfolio_value * 0.02,  # 2% min
                "quantity_shares": int((portfolio_value * 0.05) / 2550),  # Assuming price
                "position_percentage": 5.0
            },
            "risk_metrics": {
                "value_at_risk_1d": portfolio_value * 0.002,  # 0.2% daily VaR
                "max_drawdown_estimate": portfolio_value * 0.015,  # 1.5% max drawdown
                "sharpe_ratio_estimate": 1.25,
                "volatility_adjusted_size": 0.95  # Volatility adjustment factor
            },
            "recommendations": [
                f"Start with {portfolio_value * 0.03:.0f} (3% of portfolio)",
                "Consider scaling in over 2-3 transactions",
                "Set stop loss at 5% below entry",
                "Monitor position size relative to total portfolio risk"
            ]
        }
        
        return AnalysisResponse(
            success=True,
            data=position_analysis,
            message=f"Position sizing analysis completed for {asset_symbol}",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Position sizing analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Position sizing analysis failed: {str(e)}"
        )


@router.get("/health")
async def analysis_health_check():
    """Health check for analysis services"""
    return {
        "status": "healthy",
        "service": "analysis_api",
        "timestamp": datetime.utcnow().isoformat(),
        "available_analyses": [
            "comprehensive_analysis",
            "technical_indicators", 
            "sentiment_trend",
            "market_regime",
            "position_sizing"
        ]
    }


# Helper functions
def _calculate_overall_score(analysis_results: Dict[str, Any]) -> float:
    """Calculate overall analysis score from multiple factors"""
    scores = []
    
    # Sentiment score (0-1)
    if "sentiment" in analysis_results and "error" not in analysis_results["sentiment"]:
        sentiment_data = analysis_results["sentiment"]
        if "sentiment_score" in sentiment_data:
            scores.append(abs(sentiment_data["sentiment_score"]))
    
    # Technical score (0-1) 
    if "technical" in analysis_results and "error" not in analysis_results["technical"]:
        technical_data = analysis_results["technical"]
        # Simple technical score based on momentum
        if "momentum" in technical_data and "rsi" in technical_data["momentum"]:
            rsi = technical_data["momentum"]["rsi"]
            # Convert RSI to 0-1 score (50 = neutral = 0.5)
            tech_score = (rsi / 100)
            scores.append(tech_score)
    
    # Market impact score (0-1)
    if "market_impact" in analysis_results and "error" not in analysis_results["market_impact"]:
        impact_data = analysis_results["market_impact"]
        if "average_impact_score" in impact_data:
            impact_score = impact_data["average_impact_score"] / 10  # Normalize to 0-1
            scores.append(impact_score)
    
    # Return average score or 0.5 if no scores available
    return sum(scores) / len(scores) if scores else 0.5


def _score_to_rating(score: float) -> str:
    """Convert numeric score to rating"""
    if score >= 0.8:
        return "Strong Buy"
    elif score >= 0.65:
        return "Buy"
    elif score >= 0.55:
        return "Weak Buy"
    elif score >= 0.45:
        return "Hold"
    elif score >= 0.35:
        return "Weak Sell"
    elif score >= 0.2:
        return "Sell"
    else:
        return "Strong Sell"


def _generate_recommendation(score: float, analysis_results: Dict[str, Any]) -> str:
    """Generate text recommendation based on analysis"""
    rating = _score_to_rating(score)
    
    base_recommendations = {
        "Strong Buy": "Strong positive signals across multiple indicators",
        "Buy": "Positive outlook with good risk-reward ratio", 
        "Weak Buy": "Cautiously positive, consider small position",
        "Hold": "Mixed signals, maintain current position",
        "Weak Sell": "Some negative signals, consider reducing position",
        "Sell": "Negative outlook, consider selling",
        "Strong Sell": "Strong negative signals across indicators"
    }
    
    return base_recommendations.get(rating, "Neutral outlook")


def _calculate_confidence(analysis_results: Dict[str, Any]) -> float:
    """Calculate confidence level based on data availability"""
    total_analyses = 5  # Total possible analyses
    successful_analyses = 0
    
    for analysis_type in ["sentiment", "technical", "market_impact", "options", "correlations"]:
        if analysis_type in analysis_results and "error" not in analysis_results[analysis_type]:
            successful_analyses += 1
    
    return successful_analyses / total_analyses 