"""
Portfolio Management Schemas for request validation and response formatting.

Defines Pydantic models for portfolio CRUD operations, holdings management,
transaction tracking, and performance analytics.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from enum import Enum


class TransactionType(str, Enum):
    """Transaction types for portfolio operations."""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    BONUS = "bonus"
    SPLIT = "split"
    MERGER = "merger"


class AssetSegment(str, Enum):
    """Asset segments for categorization."""
    EQUITY = "equity"
    DERIVATIVE = "derivative"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    MUTUAL_FUND = "mutual_fund"
    BOND = "bond"


class RiskTolerance(str, Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate" 
    AGGRESSIVE = "aggressive"


# Portfolio Schemas
class PortfolioCreate(BaseModel):
    """Schema for creating a new portfolio."""
    name: str = Field(..., min_length=1, max_length=100, description="Portfolio name")
    description: Optional[str] = Field(None, max_length=500, description="Portfolio description")
    initial_capital: float = Field(..., gt=0, description="Initial capital amount")
    risk_tolerance: RiskTolerance = Field(default=RiskTolerance.MODERATE, description="Risk tolerance level")
    benchmark: Optional[str] = Field(None, description="Benchmark symbol for comparison")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Growth Portfolio",
                "description": "Long-term growth focused portfolio",
                "initial_capital": 100000.0,
                "risk_tolerance": "moderate",
                "benchmark": "NIFTY50"
            }
        }


class PortfolioUpdate(BaseModel):
    """Schema for updating portfolio information."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    risk_tolerance: Optional[RiskTolerance] = None
    benchmark: Optional[str] = None
    is_active: Optional[bool] = None


class PortfolioResponse(BaseModel):
    """Schema for portfolio response."""
    id: int = Field(..., description="Portfolio unique identifier")
    name: str = Field(..., description="Portfolio name")
    description: Optional[str] = Field(None, description="Portfolio description")
    initial_capital: float = Field(..., description="Initial capital")
    current_value: float = Field(..., description="Current portfolio value")
    cash_balance: float = Field(..., description="Available cash balance")
    invested_amount: float = Field(..., description="Total invested amount")
    total_pnl: float = Field(..., description="Total profit/loss")
    total_pnl_percentage: float = Field(..., description="Total PnL percentage")
    day_pnl: float = Field(..., description="Today's profit/loss")
    day_pnl_percentage: float = Field(..., description="Today's PnL percentage")
    risk_tolerance: str = Field(..., description="Risk tolerance level")
    benchmark: Optional[str] = Field(None, description="Benchmark symbol")
    is_active: bool = Field(default=True, description="Portfolio active status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Growth Portfolio",
                "description": "Long-term growth focused portfolio",
                "initial_capital": 100000.0,
                "current_value": 105500.0,
                "cash_balance": 15000.0,
                "invested_amount": 90500.0,
                "total_pnl": 5500.0,
                "total_pnl_percentage": 5.5,
                "day_pnl": 750.0,
                "day_pnl_percentage": 0.71,
                "risk_tolerance": "moderate",
                "benchmark": "NIFTY50",
                "is_active": True,
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T16:30:00Z"
            }
        }


# Holdings Schemas
class HoldingCreate(BaseModel):
    """Schema for creating a new holding."""
    symbol: str = Field(..., min_length=1, max_length=20, description="Stock symbol")
    quantity: int = Field(..., gt=0, description="Number of shares")
    average_price: float = Field(..., gt=0, description="Average purchase price")
    segment: AssetSegment = Field(default=AssetSegment.EQUITY, description="Asset segment")
    exchange: str = Field(..., description="Exchange name")
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "RELIANCE",
                "quantity": 100,
                "average_price": 2500.0,
                "segment": "equity",
                "exchange": "NSE"
            }
        }


class HoldingUpdate(BaseModel):
    """Schema for updating holding information."""
    quantity: Optional[int] = Field(None, gt=0)
    average_price: Optional[float] = Field(None, gt=0)
    current_price: Optional[float] = Field(None, gt=0)


class HoldingResponse(BaseModel):
    """Schema for holding response."""
    id: int = Field(..., description="Holding unique identifier")
    portfolio_id: int = Field(..., description="Portfolio ID")
    symbol: str = Field(..., description="Stock symbol")
    quantity: int = Field(..., description="Number of shares")
    average_price: float = Field(..., description="Average purchase price")
    current_price: float = Field(..., description="Current market price")
    total_value: float = Field(..., description="Total holding value")
    invested_amount: float = Field(..., description="Total invested amount")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")
    unrealized_pnl_percentage: float = Field(..., description="Unrealized PnL percentage")
    day_pnl: float = Field(..., description="Today's profit/loss")
    day_pnl_percentage: float = Field(..., description="Today's PnL percentage")
    segment: str = Field(..., description="Asset segment")
    exchange: str = Field(..., description="Exchange name")
    last_updated: datetime = Field(..., description="Last price update timestamp")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "portfolio_id": 1,
                "symbol": "RELIANCE",
                "quantity": 100,
                "average_price": 2500.0,
                "current_price": 2550.0,
                "total_value": 255000.0,
                "invested_amount": 250000.0,
                "unrealized_pnl": 5000.0,
                "unrealized_pnl_percentage": 2.0,
                "day_pnl": 1000.0,
                "day_pnl_percentage": 0.39,
                "segment": "equity",
                "exchange": "NSE",
                "last_updated": "2024-01-15T16:30:00Z"
            }
        }


# Transaction Schemas
class TransactionCreate(BaseModel):
    """Schema for creating a new transaction."""
    symbol: str = Field(..., min_length=1, max_length=20, description="Stock symbol")
    transaction_type: TransactionType = Field(..., description="Transaction type")
    quantity: int = Field(..., gt=0, description="Number of shares")
    price: float = Field(..., gt=0, description="Transaction price per share")
    fees: float = Field(default=0.0, ge=0, description="Transaction fees")
    exchange: str = Field(..., description="Exchange name")
    order_id: Optional[str] = Field(None, description="Broker order ID")
    notes: Optional[str] = Field(None, max_length=500, description="Transaction notes")
    
    @validator('fees')
    def validate_fees(cls, v, values):
        if 'quantity' in values and 'price' in values:
            max_fees = values['quantity'] * values['price'] * 0.1  # 10% max fees
            if v > max_fees:
                raise ValueError('Fees cannot exceed 10% of transaction value')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "RELIANCE",
                "transaction_type": "buy",
                "quantity": 50,
                "price": 2500.0,
                "fees": 125.0,
                "exchange": "NSE",
                "order_id": "ORD123456",
                "notes": "Regular purchase"
            }
        }


class TransactionResponse(BaseModel):
    """Schema for transaction response."""
    id: int = Field(..., description="Transaction unique identifier")
    portfolio_id: int = Field(..., description="Portfolio ID")
    symbol: str = Field(..., description="Stock symbol")
    transaction_type: str = Field(..., description="Transaction type")
    quantity: int = Field(..., description="Number of shares")
    price: float = Field(..., description="Transaction price per share")
    total_amount: float = Field(..., description="Gross transaction amount")
    fees: float = Field(..., description="Transaction fees")
    net_amount: float = Field(..., description="Net transaction amount")
    exchange: str = Field(..., description="Exchange name")
    order_id: Optional[str] = Field(None, description="Broker order ID")
    notes: Optional[str] = Field(None, description="Transaction notes")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "portfolio_id": 1,
                "symbol": "RELIANCE",
                "transaction_type": "buy",
                "quantity": 50,
                "price": 2500.0,
                "total_amount": 125000.0,
                "fees": 125.0,
                "net_amount": 125125.0,
                "exchange": "NSE",
                "order_id": "ORD123456",
                "notes": "Regular purchase",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# Performance Schemas
class PortfolioPerformance(BaseModel):
    """Schema for portfolio performance metrics."""
    portfolio_id: int = Field(..., description="Portfolio ID")
    total_value: float = Field(..., description="Total portfolio value")
    total_pnl: float = Field(..., description="Total profit/loss")
    total_pnl_percentage: float = Field(..., description="Total PnL percentage")
    realized_pnl: float = Field(..., description="Realized profit/loss")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")
    day_pnl: float = Field(..., description="Today's profit/loss")
    day_pnl_percentage: float = Field(..., description="Today's PnL percentage")
    
    # Risk metrics
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    volatility: float = Field(..., description="Portfolio volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    beta: Optional[float] = Field(None, description="Portfolio beta vs benchmark")
    alpha: Optional[float] = Field(None, description="Portfolio alpha vs benchmark")
    
    class Config:
        schema_extra = {
            "example": {
                "portfolio_id": 1,
                "total_value": 105500.0,
                "total_pnl": 5500.0,
                "total_pnl_percentage": 5.5,
                "realized_pnl": 2000.0,
                "unrealized_pnl": 3500.0,
                "day_pnl": 750.0,
                "day_pnl_percentage": 0.71,
                "max_drawdown": -2.5,
                "volatility": 1.8,
                "sharpe_ratio": 1.15,
                "beta": 0.85,
                "alpha": 0.5
            }
        }


class PortfolioAnalytics(BaseModel):
    """Schema for detailed portfolio analytics."""
    portfolio_id: int = Field(..., description="Portfolio ID")
    
    # Diversification metrics
    sector_allocation: Dict[str, float] = Field(..., description="Sector-wise allocation percentages")
    market_cap_allocation: Dict[str, float] = Field(..., description="Market cap allocation")
    concentration_risk: str = Field(..., description="Concentration risk level")
    
    # Performance attribution
    top_contributors: List[Dict[str, Any]] = Field(..., description="Top performing holdings")
    top_detractors: List[Dict[str, Any]] = Field(..., description="Worst performing holdings")
    
    # Risk analysis
    portfolio_beta: float = Field(..., description="Portfolio beta")
    correlation_with_benchmark: float = Field(..., description="Correlation with benchmark")
    risk_score: float = Field(..., ge=0, le=10, description="Risk score (0-10)")
    
    class Config:
        schema_extra = {
            "example": {
                "portfolio_id": 1,
                "sector_allocation": {
                    "Technology": 30.0,
                    "Banking": 25.0,
                    "Energy": 20.0,
                    "Healthcare": 15.0,
                    "Others": 10.0
                },
                "market_cap_allocation": {
                    "Large Cap": 60.0,
                    "Mid Cap": 25.0,
                    "Small Cap": 15.0
                },
                "concentration_risk": "Low",
                "top_contributors": [
                    {"symbol": "RELIANCE", "contribution": 1250.0},
                    {"symbol": "TCS", "contribution": 980.0}
                ],
                "top_detractors": [
                    {"symbol": "HDFC", "contribution": -320.0}
                ],
                "portfolio_beta": 0.85,
                "correlation_with_benchmark": 0.78,
                "risk_score": 6.5
            }
        }


# List response schemas
class PortfolioListResponse(BaseModel):
    """Schema for portfolio list response."""
    portfolios: List[PortfolioResponse] = Field(..., description="List of portfolios")
    total_count: int = Field(..., description="Total number of portfolios")
    page: int = Field(default=1, description="Current page number")
    per_page: int = Field(default=20, description="Items per page")


class HoldingListResponse(BaseModel):
    """Schema for holdings list response."""
    holdings: List[HoldingResponse] = Field(..., description="List of holdings")
    portfolio_summary: Dict[str, Any] = Field(..., description="Portfolio summary")
    total_count: int = Field(..., description="Total number of holdings")


class TransactionListResponse(BaseModel):
    """Schema for transactions list response."""
    transactions: List[TransactionResponse] = Field(..., description="List of transactions")
    total_count: int = Field(..., description="Total number of transactions")
    page: int = Field(default=1, description="Current page number")
    per_page: int = Field(default=50, description="Items per page")


# Error response schemas
class PortfolioErrorResponse(BaseModel):
    """Schema for portfolio-related error responses."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid portfolio data provided",
                "details": {
                    "field": "initial_capital",
                    "issue": "must be greater than 0"
                }
            }
        } 