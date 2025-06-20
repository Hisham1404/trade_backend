"""
Google ADK Router
REST API endpoints for Google Agent Development Kit functionality
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.adk_service import get_adk_service, ADKService
from ..core.adk_config import validate_adk_setup

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/adk", tags=["adk"])


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Trading query to process")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")


class MarketAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of asset symbols to analyze")


class RiskAssessmentRequest(BaseModel):
    trade_proposal: Dict[str, Any] = Field(..., description="Trade proposal to assess")


class InsightsRequest(BaseModel):
    portfolio_data: Dict[str, Any] = Field(..., description="Portfolio data for insights generation")


class NewsAnalysisRequest(BaseModel):
    keywords: List[str] = Field(..., description="Keywords for news analysis")
    limit: int = Field(10, description="Maximum number of articles to analyze")


class StrategyRequest(BaseModel):
    parameters: Dict[str, Any] = Field(..., description="Parameters for strategy generation")


class ADKResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Health and Status Endpoints
@router.get("/status", response_model=ADKResponse)
async def get_adk_status(service: ADKService = Depends(get_adk_service)):
    """Get current ADK service status"""
    try:
        status = await service.get_status()
        return ADKResponse(
            success=True,
            data=status,
            message="ADK status retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Error getting ADK status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate", response_model=ADKResponse)
async def validate_configuration():
    """Validate ADK configuration"""
    try:
        validation = validate_adk_setup()
        return ADKResponse(
            success=validation["valid"],
            data=validation,
            message="Configuration validated successfully" if validation["valid"] else "Configuration validation failed"
        )
    except Exception as e:
        logger.error(f"Error validating ADK configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize", response_model=ADKResponse)
async def initialize_adk(service: ADKService = Depends(get_adk_service)):
    """Initialize or reinitialize the ADK service"""
    try:
        result = await service.initialize()
        return ADKResponse(
            success=result["success"],
            data=result,
            message=result["message"]
        )
    except Exception as e:
        logger.error(f"Error initializing ADK: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Core Agent Functionality
@router.post("/query", response_model=ADKResponse)
async def process_query(
    request: QueryRequest,
    service: ADKService = Depends(get_adk_service)
):
    """Process a general trading query using the ADK agent"""
    try:
        result = await service.process_query(request.query, request.context)
        return ADKResponse(
            success=result["success"],
            data=result,
            message="Query processed successfully" if result["success"] else result.get("message", "Query processing failed")
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Market Analysis Endpoints
@router.post("/analyze/market", response_model=ADKResponse)
async def analyze_market(
    request: MarketAnalysisRequest,
    service: ADKService = Depends(get_adk_service)
):
    """Analyze market conditions for specified symbols"""
    try:
        result = await service.analyze_market(request.symbols)
        return ADKResponse(
            success=result["success"],
            data=result,
            message="Market analysis completed successfully" if result["success"] else result.get("message", "Market analysis failed")
        )
    except Exception as e:
        logger.error(f"Error analyzing market: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/news", response_model=ADKResponse)
async def analyze_news_impact(
    request: NewsAnalysisRequest,
    service: ADKService = Depends(get_adk_service)
):
    """Analyze news impact on market"""
    try:
        result = await service.analyze_news_impact(request.keywords, request.limit)
        return ADKResponse(
            success=result["success"],
            data=result,
            message="News analysis completed successfully" if result["success"] else result.get("message", "News analysis failed")
        )
    except Exception as e:
        logger.error(f"Error analyzing news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Trading Intelligence Endpoints
@router.post("/insights", response_model=ADKResponse)
async def generate_insights(
    request: InsightsRequest,
    service: ADKService = Depends(get_adk_service)
):
    """Generate trading insights based on portfolio data"""
    try:
        result = await service.generate_insights(request.portfolio_data)
        return ADKResponse(
            success=result["success"],
            data=result,
            message="Insights generated successfully" if result["success"] else result.get("message", "Insight generation failed")
        )
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess/risk", response_model=ADKResponse)
async def assess_risk(
    request: RiskAssessmentRequest,
    service: ADKService = Depends(get_adk_service)
):
    """Assess risk for a trade proposal"""
    try:
        result = await service.assess_risk(request.trade_proposal)
        return ADKResponse(
            success=result["success"],
            data=result,
            message="Risk assessment completed successfully" if result["success"] else result.get("message", "Risk assessment failed")
        )
    except Exception as e:
        logger.error(f"Error assessing risk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategy", response_model=ADKResponse)
async def generate_strategy(
    request: StrategyRequest,
    service: ADKService = Depends(get_adk_service)
):
    """Generate trading strategy based on parameters"""
    try:
        result = await service.generate_trading_strategy(request.parameters)
        return ADKResponse(
            success=result["success"],
            data=result,
            message="Strategy generated successfully" if result["success"] else result.get("message", "Strategy generation failed")
        )
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Management Endpoints
@router.post("/shutdown", response_model=ADKResponse)
async def shutdown_adk(
    background_tasks: BackgroundTasks,
    service: ADKService = Depends(get_adk_service)
):
    """Shutdown the ADK service gracefully"""
    try:
        # Schedule shutdown in background to allow response to be sent
        background_tasks.add_task(service.shutdown)
        
        return ADKResponse(
            success=True,
            message="ADK service shutdown initiated successfully"
        )
    except Exception as e:
        logger.error(f"Error shutting down ADK: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Quick Analysis Endpoints
@router.get("/quick/market/{symbol}", response_model=ADKResponse)
async def quick_market_analysis(
    symbol: str,
    service: ADKService = Depends(get_adk_service)
):
    """Quick market analysis for a single symbol"""
    try:
        result = await service.analyze_market([symbol])
        return ADKResponse(
            success=result["success"],
            data=result,
            message=f"Quick analysis for {symbol} completed"
        )
    except Exception as e:
        logger.error(f"Error in quick market analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick/insights", response_model=ADKResponse)
async def quick_insights(
    service: ADKService = Depends(get_adk_service)
):
    """Quick general market insights"""
    try:
        query = "Provide quick insights on current market conditions and trading opportunities"
        result = await service.process_query(query, {"analysis_type": "quick_insights"})
        return ADKResponse(
            success=result["success"],
            data=result,
            message="Quick insights generated successfully"
        )
    except Exception as e:
        logger.error(f"Error generating quick insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Integration Endpoints
@router.get("/tools", response_model=ADKResponse)
async def get_available_tools(service: ADKService = Depends(get_adk_service)):
    """Get list of available ADK tools"""
    try:
        status = await service.get_status()
        
        tools_info = {
            "available_tools": [
                "get_market_data",
                "technical_analysis", 
                "analyze_news",
                "portfolio_analysis"
            ],
            "capabilities": {
                "market_analysis": True,
                "risk_assessment": True,
                "news_analysis": True,
                "strategy_generation": True,
                "portfolio_insights": True
            },
            "status": status
        }
        
        return ADKResponse(
            success=True,
            data=tools_info,
            message="Available tools retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Error getting available tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 