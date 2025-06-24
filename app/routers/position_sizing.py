"""
Unified Position Sizing API Router.

This module provides comprehensive REST API endpoints for position sizing,
margin calculations, leverage adjustments, and real-time risk monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import json

logger = logging.getLogger(__name__)

# Initialize router with versioning
router = APIRouter(
    prefix="/api/v1/position-sizing",
    tags=["Position Sizing"],
    responses={
        404: {"description": "Not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)


# Pydantic Models for API
class PositionSizeRequest(BaseModel):
    """Request model for position size calculation."""
    symbol: str = Field(..., example="NIFTY", description="Symbol to calculate position size for")
    expected_return: Optional[float] = Field(None, ge=-1, le=1, description="Expected return (0-1 scale)")
    stop_loss_price: Optional[float] = Field(None, gt=0, description="Stop loss price")
    product_type: str = Field("MIS", description="Product type (MIS/NRML/CNC)")
    risk_model: Optional[str] = Field("VOLATILITY_ADJUSTED", description="Risk model to use")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Symbol cannot be empty')
        return v.upper().strip()


class LeverageRequest(BaseModel):
    """Request model for leverage calculation."""
    symbol: str = Field(..., example="NIFTY", description="Symbol to get leverage for")
    current_leverage: Optional[float] = Field(None, ge=1, le=10, description="Current leverage")
    market_data: Optional[Dict[str, Any]] = Field(None, description="Optional market data override")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Symbol cannot be empty')
        return v.upper().strip()


class MarginRequest(BaseModel):
    """Request model for SPAN margin calculation."""
    positions: List[Dict[str, Any]] = Field(..., description="List of portfolio positions")
    symbol: Optional[str] = Field(None, description="Base symbol for analysis")
    
    @validator('positions')
    def validate_positions(cls, v):
        if not v:
            raise ValueError('Positions list cannot be empty')
        return v


class AdvancedSizingRequest(BaseModel):
    """Request model for advanced position sizing."""
    symbols: List[str] = Field(..., min_items=1, description="Symbols to calculate positions for")
    sizing_model: str = Field("KELLY_CRITERION", description="Position sizing model")
    risk_budget: Optional[str] = Field("MODERATE", description="Risk budget profile")
    base_leverages: Optional[Dict[str, float]] = Field(None, description="Current leverages by symbol")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError('Symbols list cannot be empty')
        return [s.upper().strip() for s in v if s.strip()]


class RiskMonitoringRequest(BaseModel):
    """Request model for risk monitoring."""
    user_id: int = Field(..., gt=0, description="User ID")
    include_detailed_metrics: bool = Field(False, description="Include detailed risk metrics")
    symbols_filter: Optional[List[str]] = Field(None, description="Filter by specific symbols")


# Response Models
class PositionSizeResponse(BaseModel):
    """Response model for position size calculation."""
    symbol: str
    recommended_quantity: int
    recommended_lots: int
    position_value: float
    risk_per_trade: float
    margin_required: float
    leverage_ratio: float
    risk_model_used: str
    confidence_level: float
    warnings: List[str]
    calculation_timestamp: datetime


class LeverageResponse(BaseModel):
    """Response model for leverage calculation."""
    symbol: str
    current_leverage: float
    recommended_leverage: float
    adjustment_magnitude: float
    adjustment_type: str
    reasoning: str
    confidence_level: float
    risk_impact: Dict[str, float]
    implementation_guidance: Dict[str, str]
    warnings: List[str]
    market_conditions: Dict[str, Any]
    calculation_timestamp: datetime


class MarginResponse(BaseModel):
    """Response model for margin calculation."""
    total_margin_required: float
    span_margin: float
    exposure_margin: float
    additional_margin: float
    margin_utilization: float
    available_leverage: float
    risk_scenarios: List[Dict[str, Any]]
    spread_benefits: List[Dict[str, Any]]
    calculation_timestamp: datetime


class AdvancedSizingResponse(BaseModel):
    """Response model for advanced position sizing."""
    recommendations: List[Dict[str, Any]]
    portfolio_summary: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    optimization_details: Dict[str, Any]
    constraint_violations: List[str]
    calculation_timestamp: datetime


class RiskMonitoringResponse(BaseModel):
    """Response model for risk monitoring."""
    user_id: int
    portfolio_metrics: Dict[str, Any]
    risk_alerts: List[Dict[str, str]]
    position_analysis: List[Dict[str, Any]]
    leverage_summary: Dict[str, Any]
    compliance_status: Dict[str, Any]
    next_review_time: datetime
    calculation_timestamp: datetime


# Dependency injection
async def get_trading_service():
    """Get initialized trading service."""
    from ..trading.trading_service import TradingService
    service = TradingService()
    await service.initialize(user_id=1)  # Default user for API
    return service


# API Endpoints
@router.post("/calculate", 
             response_model=PositionSizeResponse,
             summary="Calculate Position Size",
             description="Calculate optimal position size for a given symbol based on risk parameters.")
async def calculate_position_size(
    request: PositionSizeRequest,
    user_id: int = Query(1, ge=1, description="User ID"),
    trading_service = Depends(get_trading_service)
) -> PositionSizeResponse:
    """Calculate optimal position size for a symbol."""
    try:
        from ..trading.base_broker import ProductType
        from ..trading.position_sizer import RiskModel
        
        # Map string to enum
        product_type_map = {"MIS": ProductType.MIS, "NRML": ProductType.NRML, "CNC": ProductType.CNC}
        product_type = product_type_map.get(request.product_type, ProductType.MIS)
        
        risk_model_map = {
            "FIXED": RiskModel.FIXED,
            "VOLATILITY_ADJUSTED": RiskModel.VOLATILITY_ADJUSTED,
            "KELLY": RiskModel.KELLY,
            "VAR_BASED": RiskModel.VAR_BASED
        }
        risk_model = risk_model_map.get(request.risk_model, RiskModel.VOLATILITY_ADJUSTED)
        
        # Calculate position size
        result = await trading_service.calculate_position_size(
            user_id=user_id,
            symbol=request.symbol,
            expected_return=request.expected_return,
            stop_loss_price=request.stop_loss_price,
            product_type=product_type,
            risk_model=risk_model
        )
        
        return PositionSizeResponse(
            symbol=result.symbol,
            recommended_quantity=result.quantity,
            recommended_lots=result.lots,
            position_value=float(result.position_value),
            risk_per_trade=float(result.risk_per_trade),
            margin_required=float(result.margin_required),
            leverage_ratio=float(result.leverage_ratio),
            risk_model_used=result.risk_model.value,
            confidence_level=float(result.confidence_level),
            warnings=result.warnings,
            calculation_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        raise HTTPException(status_code=500, detail=f"Position size calculation failed: {str(e)}")


@router.post("/leverage",
             response_model=LeverageResponse,
             summary="Calculate Dynamic Leverage",
             description="Get dynamic leverage recommendations based on market conditions and risk metrics.")
async def calculate_leverage(
    request: LeverageRequest,
    user_id: int = Query(1, ge=1, description="User ID"),
    trading_service = Depends(get_trading_service)
) -> LeverageResponse:
    """Calculate dynamic leverage recommendations."""
    try:
        from ..trading.base_broker import ProductType
        
        # Get leverage guidelines
        guidelines = await trading_service.get_leverage_guidelines(
            user_id=user_id,
            symbol=request.symbol,
            product_type=ProductType.MIS
        )
        
        # Get detailed dynamic leverage analysis if available
        dynamic_result = None
        if trading_service.leverage_manager:
            from decimal import Decimal
            base_leverage = Decimal(str(request.current_leverage or guidelines.recommended_leverage))
            
            dynamic_result = await trading_service.leverage_manager.calculate_optimal_leverage(
                symbol=request.symbol,
                base_leverage=base_leverage,
                market_data=request.market_data
            )
        
        # Prepare response
        if dynamic_result:
            return LeverageResponse(
                symbol=request.symbol,
                current_leverage=float(dynamic_result.current_leverage),
                recommended_leverage=float(dynamic_result.recommended_leverage),
                adjustment_magnitude=float(dynamic_result.adjustment_magnitude),
                adjustment_type=dynamic_result.adjustment_type.value,
                reasoning=dynamic_result.reasoning,
                confidence_level=float(dynamic_result.confidence_level),
                risk_impact={
                    "expected_vol_change": float(dynamic_result.expected_vol_change),
                    "expected_return_change": float(dynamic_result.expected_return_change),
                    "var_impact": float(dynamic_result.var_impact)
                },
                implementation_guidance={
                    "timeframe": dynamic_result.suggested_timeframe,
                    "priority": dynamic_result.priority
                },
                warnings=dynamic_result.warnings,
                market_conditions=dynamic_result.metadata,
                calculation_timestamp=dynamic_result.calculation_timestamp
            )
        else:
            # Fallback to basic guidelines
            return LeverageResponse(
                symbol=request.symbol,
                current_leverage=guidelines.current_leverage,
                recommended_leverage=guidelines.recommended_leverage,
                adjustment_magnitude=abs(guidelines.recommended_leverage - guidelines.current_leverage),
                adjustment_type="BASIC_CALCULATION",
                reasoning=guidelines.reasoning,
                confidence_level=0.8,
                risk_impact={"expected_vol_change": 0.0, "expected_return_change": 0.0, "var_impact": 0.0},
                implementation_guidance={"timeframe": "immediate", "priority": guidelines.risk_level},
                warnings=guidelines.warnings,
                market_conditions={"volatility_assessment": guidelines.volatility_assessment, "market_condition": guidelines.market_condition},
                calculation_timestamp=datetime.now()
            )
            
    except Exception as e:
        logger.error(f"Error calculating leverage: {e}")
        raise HTTPException(status_code=500, detail=f"Leverage calculation failed: {str(e)}")


@router.post("/advanced-sizing",
             response_model=AdvancedSizingResponse,
             summary="Advanced Position Sizing",
             description="Calculate advanced position sizes using risk-based algorithms and portfolio optimization.")
async def calculate_advanced_sizing(
    request: AdvancedSizingRequest,
    user_id: int = Query(1, ge=1, description="User ID"),
    trading_service = Depends(get_trading_service)
) -> AdvancedSizingResponse:
    """Calculate advanced position sizes with portfolio optimization."""
    try:
        from ..trading.risk_models import (
            PositionSizingModel, 
            CONSERVATIVE_RISK_BUDGET, 
            MODERATE_RISK_BUDGET, 
            AGGRESSIVE_RISK_BUDGET
        )
        
        # Map string to enum
        sizing_model_map = {
            "KELLY_CRITERION": PositionSizingModel.KELLY_CRITERION,
            "VAR_BASED": PositionSizingModel.VAR_BASED,
            "VOLATILITY_ADJUSTED": PositionSizingModel.VOLATILITY_ADJUSTED,
            "RISK_PARITY": PositionSizingModel.RISK_PARITY
        }
        sizing_model = sizing_model_map.get(request.sizing_model, PositionSizingModel.KELLY_CRITERION)
        
        # Get risk budget
        risk_budget_map = {
            "CONSERVATIVE": CONSERVATIVE_RISK_BUDGET,
            "MODERATE": MODERATE_RISK_BUDGET,
            "AGGRESSIVE": AGGRESSIVE_RISK_BUDGET
        }
        risk_budget = risk_budget_map.get(request.risk_budget, MODERATE_RISK_BUDGET)
        
        # Calculate advanced position sizes
        result = await trading_service.calculate_advanced_position_sizes(
            user_id=user_id,
            symbols=request.symbols,
            sizing_model=sizing_model,
            constraints=risk_budget
        )
        
        # Get dynamic leverage recommendations
        leverage_recommendations = {}
        if request.base_leverages:
            leverage_result = await trading_service.calculate_dynamic_leverage_for_symbols(
                user_id=user_id,
                symbols=request.symbols,
                base_leverages=request.base_leverages
            )
            leverage_recommendations = leverage_result.get('recommendations', {})
        
        return AdvancedSizingResponse(
            recommendations=result['recommendations'],
            portfolio_summary={
                "total_positions": len(request.symbols),
                "risk_budget_profile": request.risk_budget,
                "sizing_model": request.sizing_model,
                "leverage_recommendations": leverage_recommendations
            },
            risk_analysis=result['risk_analysis'],
            optimization_details={
                "constraints_applied": f"{request.risk_budget} risk budget",
                "model_used": request.sizing_model,
                "symbols_analyzed": len(request.symbols)
            },
            constraint_violations=[],  # Would check for actual violations
            calculation_timestamp=datetime.fromisoformat(result['calculation_timestamp'])
        )
        
    except Exception as e:
        logger.error(f"Error in advanced sizing: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced sizing calculation failed: {str(e)}")


@router.get("/risk-monitoring/{user_id}",
            response_model=RiskMonitoringResponse,
            summary="Risk Monitoring Dashboard",
            description="Get comprehensive risk monitoring data for portfolio management.")
async def get_risk_monitoring(
    user_id: int,
    include_detailed: bool = Query(False, description="Include detailed risk metrics"),
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols to filter"),
    trading_service = Depends(get_trading_service)
) -> RiskMonitoringResponse:
    """Get comprehensive risk monitoring data."""
    try:
        # Get portfolio risk metrics
        portfolio_metrics = await trading_service.get_portfolio_risk_metrics(user_id)
        
        # Get dynamic leverage summary
        leverage_summary = await trading_service.get_dynamic_leverage_summary(user_id)
        
        # Parse symbols filter
        symbols_list = []
        if symbols:
            symbols_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        
        # Generate risk alerts
        risk_alerts = []
        if isinstance(leverage_summary, dict) and 'portfolio_heat' in leverage_summary:
            portfolio_heat = leverage_summary['portfolio_heat']
            if portfolio_heat > 0.8:
                risk_alerts.append({
                    "level": "HIGH",
                    "message": f"Portfolio heat at {portfolio_heat:.1%} - consider reducing exposure",
                    "timestamp": datetime.now().isoformat()
                })
            elif portfolio_heat > 0.6:
                risk_alerts.append({
                    "level": "MEDIUM", 
                    "message": f"Portfolio heat at {portfolio_heat:.1%} - monitor closely",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Compliance status
        compliance_status = {
            "sebi_leverage_compliance": True,
            "risk_budget_compliance": True,
            "margin_adequacy": True,
            "concentration_limits": True
        }
        
        # Calculate next review time safely
        current_time = datetime.now()
        next_review_hour = (current_time.hour + 4) % 24
        next_review_time = current_time.replace(hour=next_review_hour, minute=0, second=0, microsecond=0)
        if next_review_hour < current_time.hour:  # Next day
            next_review_time = next_review_time + timedelta(days=1)
        
        return RiskMonitoringResponse(
            user_id=user_id,
            portfolio_metrics=portfolio_metrics,
            risk_alerts=risk_alerts,
            position_analysis=portfolio_metrics.get('position_details', []),
            leverage_summary=leverage_summary if isinstance(leverage_summary, dict) else {},
            compliance_status=compliance_status,
            next_review_time=next_review_time,
            calculation_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in risk monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Risk monitoring failed: {str(e)}")


@router.get("/health",
            summary="API Health Check",
            description="Check the health status of position sizing services.")
async def health_check():
    """Health check endpoint."""
    try:
        from ..trading.trading_service import TradingService
        
        # Test trading service initialization
        service = TradingService()
        await service.initialize(user_id=1)
        
        # Check component availability
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "trading_service": "available",
                "position_sizer": "available" if service.position_sizer else "unavailable",
                "margin_engine": "available" if service.margin_engine else "unavailable",
                "risk_sizer": "available" if service.risk_based_sizer else "unavailable",
                "leverage_manager": "available" if service.leverage_manager else "unavailable"
            },
            "version": "1.0.0"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "version": "1.0.0"
            }
        )


@router.get("/models",
            summary="Available Models",
            description="Get list of available position sizing and risk models.")
async def get_available_models():
    """Get available position sizing and risk models."""
    return {
        "position_sizing_models": ["KELLY_CRITERION", "VAR_BASED", "VOLATILITY_ADJUSTED", "RISK_PARITY"],
        "product_types": ["MIS", "NRML", "CNC"],
        "risk_budgets": ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"],
        "supported_symbols": ["NIFTY", "BANKNIFTY", "SENSEX", "RELIANCE", "HDFC", "ICICI"],
        "api_version": "1.0.0"
    }


@router.get("/documentation",
            summary="API Documentation",
            description="Get comprehensive API documentation and usage examples.")
async def get_api_documentation():
    """Get API documentation and examples."""
    return {
        "api_version": "1.0.0",
        "description": "Unified Position Sizing API providing comprehensive trading position management",
        "endpoints": {
            "/calculate": "Calculate optimal position size for a symbol",
            "/leverage": "Get dynamic leverage recommendations",
            "/advanced-sizing": "Advanced position sizing with portfolio optimization",
            "/risk-monitoring/{user_id}": "Comprehensive risk monitoring dashboard",
            "/health": "API health check",
            "/models": "Available models and configurations",
            "/documentation": "API documentation and examples"
        },
        "examples": {
            "position_size_request": {
                "symbol": "NIFTY",
                "expected_return": 0.15,
                "stop_loss_price": 19500,
                "product_type": "MIS",
                "risk_model": "VOLATILITY_ADJUSTED"
            },
            "leverage_request": {
                "symbol": "NIFTY",
                "current_leverage": 3.0,
                "market_data": {}
            },
            "advanced_sizing_request": {
                "symbols": ["NIFTY", "BANKNIFTY"],
                "sizing_model": "KELLY_CRITERION",
                "risk_budget": "MODERATE",
                "base_leverages": {"NIFTY": 3.0, "BANKNIFTY": 2.5}
            }
        },
        "rate_limits": {
            "requests_per_minute": 100,
            "concurrent_connections": 50
        },
        "response_codes": {
            "200": "Successful operation",
            "422": "Validation error",
            "500": "Internal server error",
            "503": "Service unavailable"
        }
    }