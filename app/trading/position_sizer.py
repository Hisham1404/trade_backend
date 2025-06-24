"""
Position Sizing Calculator for risk-based position management.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class RiskModel(Enum):
    """Risk models for position sizing."""
    FIXED = "fixed"
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY = "kelly"
    VAR_BASED = "var_based"


class VolatilityRegime(Enum):
    """Market volatility regimes."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_portfolio_risk: float = 2.0
    max_position_size: float = 10.0
    max_leverage: float = 5.0
    stop_loss_percentage: float = 3.0


@dataclass
class AssetMetrics:
    """Metrics for a specific asset."""
    symbol: str
    current_price: float = 0.0
    volatility: float = 20.0
    beta: float = 1.0
    sector: str = "Unknown"
    liquidity_score: float = 1.0
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.volatility_regime = self._determine_volatility_regime()
    
    def _determine_volatility_regime(self) -> VolatilityRegime:
        """Determine volatility regime."""
        if self.volatility < 15:
            return VolatilityRegime.LOW
        elif self.volatility < 25:
            return VolatilityRegime.NORMAL
        elif self.volatility < 40:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    symbol: str
    quantity: int  # Renamed from recommended_quantity
    lots: int      # Renamed from recommended_lots
    lot_size: int
    position_value: float
    margin_required: float
    risk_per_trade: float  # Renamed from risk_amount
    leverage_ratio: float  # Renamed from leverage_used
    portfolio_allocation: float
    confidence_level: float  # Renamed from confidence_score
    max_loss_amount: float
    max_loss_percentage: float
    expected_return: float
    risk_reward_ratio: float
    warnings: List[str]
    constraints_applied: List[str]
    risk_model: RiskModel  # Renamed from model_used
    volatility_regime: VolatilityRegime
    calculation_timestamp: datetime


class PositionSizer:
    """Position sizing calculator."""
    
    def __init__(self, broker_manager=None, risk_params: Optional[RiskParameters] = None, *, capital: float = None, risk_percentage: float = None, max_positions: int = None):
        """Initialize position sizer.

        For backward compatibility the constructor can be called with the
        legacy signature `PositionSizer(capital=..., risk_percentage=...,
        max_positions=...)` without providing a broker_manager.  In that case
        we create a default BrokerManager instance internally.
        """
        from .broker_manager import BrokerManager  # local import to avoid cycles

        # If the first positional arg was omitted the user may provide only
        # the legacy kwargs – ensure we still have a broker manager.
        self.broker_manager = broker_manager or BrokerManager()

        self.risk_params = risk_params or RiskParameters()

        # Legacy kwargs mapping ------------------------------------------------
        if capital is not None:
            self.portfolio_value = float(capital)
        else:
            self.portfolio_value = 1000000.0

        if risk_percentage is not None:
            self.risk_params.max_portfolio_risk = float(risk_percentage)
        if max_positions is not None:
            self.max_positions = int(max_positions)

        self.current_positions = {}
        self.sector_allocations = {}
    
    def set_portfolio_value(self, value: float) -> None:
        """Set portfolio value."""
        self.portfolio_value = value
    
    def update_current_positions(self, positions: Dict[str, float]) -> None:
        """Update current positions."""
        self.current_positions = positions.copy()
    
    async def calculate_position_size(self, 
                                    symbol: str,
                                    asset_metrics: AssetMetrics,
                                    product_type,
                                    target_return: Optional[float] = None,
                                    stop_loss_price: Optional[float] = None,
                                    model: RiskModel = RiskModel.VOLATILITY_ADJUSTED,
                                    exchange: str = "NSE") -> PositionSizeResult:
        """Calculate optimal position size."""
        # Simplified implementation for testing
        lot_size = await self.broker_manager.get_lot_size(symbol, exchange)
        
        # Basic fixed fractional sizing
        risk_per_share = (self.risk_params.stop_loss_percentage / 100) * asset_metrics.current_price
        max_risk_amount = (self.risk_params.max_portfolio_risk / 100) * self.portfolio_value
        
        if risk_per_share <= 0:
            base_quantity = 0
        else:
            base_quantity = max_risk_amount / risk_per_share
        
        # Convert to lots
        lots = max(1, round(base_quantity / lot_size))
        final_quantity = lots * lot_size
        
        # Calculate metrics
        position_value = final_quantity * asset_metrics.current_price
        margin_required = position_value * 0.2  # Simplified margin calculation
        risk_amount = (self.risk_params.stop_loss_percentage / 100) * position_value
        
        return PositionSizeResult(
            symbol=symbol,
            quantity=final_quantity,
            lots=lots,
            lot_size=lot_size,
            position_value=position_value,
            margin_required=margin_required,
            risk_per_trade=risk_amount,
            leverage_ratio=5.0,
            portfolio_allocation=(position_value / self.portfolio_value) * 100,
            confidence_level=0.8,
            max_loss_amount=risk_amount,
            max_loss_percentage=(risk_amount / self.portfolio_value) * 100,
            expected_return=0.0,
            risk_reward_ratio=0.0,
            warnings=[],
            constraints_applied=[],
            risk_model=model,
            volatility_regime=asset_metrics.volatility_regime,
            calculation_timestamp=datetime.now()
        )