"""
Risk Models for Position Sizing.

This module implements advanced risk models for intelligent position sizing,
including Kelly Criterion, VaR, Expected Shortfall, and correlation analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class PositionSizingModel(Enum):
    """Position sizing models available."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    VAR_BASED = "var_based"
    OPTIMAL_F = "optimal_f"
    RISK_PARITY = "risk_parity"


class RiskMetric(Enum):
    """Risk metrics for evaluation."""
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    SHARPE_RATIO = "sharpe_ratio"


@dataclass
class AssetRiskMetrics:
    """Risk metrics for a specific asset."""
    symbol: str
    expected_return: Decimal  # Annual expected return
    volatility: Decimal  # Annual volatility
    skewness: Decimal = Decimal('0')
    kurtosis: Decimal = Decimal('3')
    max_drawdown: Decimal = Decimal('0')
    sharpe_ratio: Decimal = Decimal('0')
    beta: Decimal = Decimal('1')
    
    # VaR metrics
    var_95: Decimal = Decimal('0')
    var_99: Decimal = Decimal('0')
    expected_shortfall_95: Decimal = Decimal('0')
    expected_shortfall_99: Decimal = Decimal('0')
    
    # Historical data for calculations
    historical_returns: List[Decimal] = field(default_factory=list)
    calculation_date: datetime = field(default_factory=datetime.now)
    
    def calculate_risk_metrics(self):
        """Calculate VaR and other risk metrics from historical returns."""
        if not self.historical_returns:
            return
        
        returns = [float(r) for r in self.historical_returns]
        returns_array = np.array(returns)
        
        # Calculate VaR
        self.var_95 = Decimal(str(np.percentile(returns, 5)))
        self.var_99 = Decimal(str(np.percentile(returns, 1)))
        
        # Calculate Expected Shortfall
        var_95_value = float(self.var_95)
        var_99_value = float(self.var_99)
        
        tail_95 = returns_array[returns_array <= var_95_value]
        tail_99 = returns_array[returns_array <= var_99_value]
        
        if len(tail_95) > 0:
            self.expected_shortfall_95 = Decimal(str(np.mean(tail_95)))
        if len(tail_99) > 0:
            self.expected_shortfall_99 = Decimal(str(np.mean(tail_99)))
        
        # Calculate basic statistics
        if len(returns) > 1:
            self.expected_return = Decimal(str(np.mean(returns) * 252))  # Annualized
            self.volatility = Decimal(str(np.std(returns) * np.sqrt(252)))  # Annualized
        
        # Calculate maximum drawdown
        if len(returns) > 0:
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            self.max_drawdown = Decimal(str(np.min(drawdown)))


@dataclass
class RiskBudget:
    """Risk budget constraints for position sizing."""
    max_portfolio_var: Decimal  # Maximum portfolio VaR
    max_position_weight: Decimal  # Maximum single position weight
    max_sector_weight: Decimal  # Maximum sector concentration
    max_correlation_exposure: Decimal  # Maximum exposure to correlated assets
    max_leverage: Decimal  # Maximum portfolio leverage
    target_volatility: Decimal  # Target portfolio volatility
    
    # Risk limits
    stop_loss_threshold: Decimal = Decimal('0.05')  # 5% stop loss
    max_drawdown_limit: Decimal = Decimal('0.15')  # 15% max drawdown


@dataclass
class PositionSizeRecommendation:
    """Position size recommendation with risk analysis."""
    symbol: str
    recommended_weight: Decimal  # Portfolio weight
    recommended_quantity: int
    recommended_lots: int
    position_value: Decimal
    risk_contribution: Decimal  # Contribution to portfolio risk
    expected_return: Decimal
    
    # Risk metrics
    position_var: Decimal
    
    # Sizing model used
    sizing_model: PositionSizingModel
    confidence_level: Decimal
    
    # Constraints and warnings
    constraints_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Optimization details
    kelly_fraction: Optional[Decimal] = None
    optimal_f: Optional[Decimal] = None
    volatility_scalar: Optional[Decimal] = None
    
    calculation_timestamp: datetime = field(default_factory=datetime.now)


class KellyCriterion:
    """Kelly Criterion implementation for optimal position sizing."""
    
    @staticmethod
    def calculate_kelly_fraction(win_probability: float, 
                                avg_win: float, 
                                avg_loss: float) -> float:
        """Calculate Kelly fraction for position sizing."""
        if avg_loss <= 0 or win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        b = avg_win / avg_loss
        p = win_probability
        q = 1 - p
        
        kelly_f = (b * p - q) / b
        
        # Cap Kelly fraction at reasonable levels (max 25%)
        return max(0.0, min(kelly_f, 0.25))
    
    @staticmethod
    def calculate_kelly_from_returns(returns: List[float]) -> float:
        """Calculate Kelly fraction from historical returns."""
        if not returns or len(returns) < 10:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Split into wins and losses
        wins = returns_array[returns_array > 0]
        losses = returns_array[returns_array < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        win_probability = len(wins) / len(returns_array)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        return KellyCriterion.calculate_kelly_fraction(win_probability, avg_win, avg_loss)


class VaRCalculator:
    """Value at Risk calculation methods."""
    
    @staticmethod
    def parametric_var(returns: List[float], 
                      confidence_level: float = 0.95) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-score for confidence level (approximation)
        z_score = -1.645 if confidence_level == 0.95 else -2.326  # 95% or 99%
        
        # VaR calculation
        var = mean_return + z_score * std_return
        
        return var
    
    @staticmethod
    def historical_var(returns: List[float], 
                      confidence_level: float = 0.95) -> float:
        """Calculate historical VaR using empirical distribution."""
        if not returns:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def expected_shortfall(returns: List[float], 
                          confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if not returns:
            return 0.0
        
        var_threshold = VaRCalculator.historical_var(returns, confidence_level)
        tail_returns = [r for r in returns if r <= var_threshold]
        
        if not tail_returns:
            return var_threshold
        
        return np.mean(tail_returns)


class CorrelationAnalyzer:
    """Portfolio correlation analysis for risk management."""
    
    def __init__(self):
        self.correlation_matrix: Optional[pd.DataFrame] = None
    
    def calculate_correlation_matrix(self, 
                                   returns_data: Dict[str, List[float]]) -> pd.DataFrame:
        """Calculate correlation matrix from returns data."""
        if not returns_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        self.correlation_matrix = df.corr()
        
        return self.correlation_matrix
    
    def get_highly_correlated_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find highly correlated asset pairs."""
        if self.correlation_matrix is None:
            return []
        
        pairs = []
        symbols = self.correlation_matrix.columns
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                correlation = self.correlation_matrix.loc[symbol1, symbol2]
                if abs(correlation) >= threshold:
                    pairs.append((symbol1, symbol2, correlation))
        
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio diversification ratio."""
        if self.correlation_matrix is None or not weights:
            return 1.0
        
        # Weighted average correlation
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 1.0
        
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        weighted_corr = 0.0
        for symbol1, weight1 in normalized_weights.items():
            for symbol2, weight2 in normalized_weights.items():
                if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[symbol1, symbol2]
                    weighted_corr += weight1 * weight2 * corr
        
        # Diversification ratio = 1 / sqrt(weighted_correlation)
        return 1.0 / math.sqrt(max(weighted_corr, 0.01))


# Default risk budgets for different risk profiles
CONSERVATIVE_RISK_BUDGET = RiskBudget(
    max_portfolio_var=Decimal('0.02'),  # 2% daily VaR
    max_position_weight=Decimal('0.05'),  # 5% max position
    max_sector_weight=Decimal('0.20'),  # 20% max sector
    max_correlation_exposure=Decimal('0.30'),  # 30% max correlated exposure
    max_leverage=Decimal('2.0'),  # 2x max leverage
    target_volatility=Decimal('0.12'),  # 12% annual volatility
    max_drawdown_limit=Decimal('0.10')  # 10% max drawdown
)

MODERATE_RISK_BUDGET = RiskBudget(
    max_portfolio_var=Decimal('0.03'),  # 3% daily VaR
    max_position_weight=Decimal('0.10'),  # 10% max position
    max_sector_weight=Decimal('0.30'),  # 30% max sector
    max_correlation_exposure=Decimal('0.50'),  # 50% max correlated exposure
    max_leverage=Decimal('3.0'),  # 3x max leverage
    target_volatility=Decimal('0.18'),  # 18% annual volatility
    max_drawdown_limit=Decimal('0.15')  # 15% max drawdown
)

AGGRESSIVE_RISK_BUDGET = RiskBudget(
    max_portfolio_var=Decimal('0.05'),  # 5% daily VaR
    max_position_weight=Decimal('0.15'),  # 15% max position
    max_sector_weight=Decimal('0.40'),  # 40% max sector
    max_correlation_exposure=Decimal('0.70'),  # 70% max correlated exposure
    max_leverage=Decimal('5.0'),  # 5x max leverage
    target_volatility=Decimal('0.25'),  # 25% annual volatility
    max_drawdown_limit=Decimal('0.20')  # 20% max drawdown
)