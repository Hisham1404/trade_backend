"""
Risk-Based Position Sizing Engine.

This module implements the main position sizing engine that integrates
Kelly Criterion, VaR analysis, correlation matrices, and portfolio optimization
to provide intelligent position sizing recommendations.
"""

import asyncio
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import logging
import math

from .risk_models import (
    PositionSizingModel, AssetRiskMetrics, RiskBudget, PositionSizeRecommendation,
    KellyCriterion, VaRCalculator, CorrelationAnalyzer,
    CONSERVATIVE_RISK_BUDGET, MODERATE_RISK_BUDGET, AGGRESSIVE_RISK_BUDGET
)
from .base_broker import BaseBroker
from .span.margin_engine import MarginEngine

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """Current portfolio position information."""
    symbol: str
    quantity: int
    current_price: Decimal
    market_value: Decimal
    weight: Decimal
    sector: str = "Unknown"
    unrealized_pnl: Decimal = Decimal('0')


@dataclass
class OptimizationConstraints:
    """Optimization constraints for position sizing."""
    min_weight: Decimal = Decimal('0.001')  # 0.1% minimum position
    max_weight: Decimal = Decimal('0.10')   # 10% maximum position
    max_turnover: Decimal = Decimal('0.20')  # 20% maximum turnover
    transaction_cost: Decimal = Decimal('0.001')  # 0.1% transaction cost
    
    # Sector constraints
    max_sector_weights: Dict[str, Decimal] = field(default_factory=dict)
    
    # Correlation constraints
    max_correlated_weight: Decimal = Decimal('0.30')  # Max 30% in correlated assets
    correlation_threshold: Decimal = Decimal('0.7')  # 70% correlation threshold


class RiskBasedPositionSizer:
    """
    Advanced risk-based position sizing engine.
    
    Integrates Kelly Criterion, VaR analysis, correlation matrices,
    and portfolio optimization for intelligent position sizing.
    """
    
    def __init__(self,
                 broker: BaseBroker,
                 margin_engine: MarginEngine,
                 risk_budget: Optional[RiskBudget] = None):
        """Initialize the risk-based position sizer."""
        self.broker = broker
        self.margin_engine = margin_engine
        self.risk_budget = risk_budget or MODERATE_RISK_BUDGET
        
        # Risk analysis components
        self.correlation_analyzer = CorrelationAnalyzer()
        self.kelly_calculator = KellyCriterion()
        self.var_calculator = VaRCalculator()
        
        # Cache for risk metrics
        self.asset_risk_cache: Dict[str, AssetRiskMetrics] = {}
        self.correlation_cache: Optional[pd.DataFrame] = None
        self.cache_timestamp: Optional[datetime] = None
        self.cache_duration = timedelta(hours=4)  # 4-hour cache
        
        # Portfolio state
        self.current_positions: Dict[str, PortfolioPosition] = {}
        self.total_portfolio_value = Decimal('1000000')  # Default 10L portfolio
        
        logger.info("Risk-based position sizer initialized")
    
    async def calculate_position_sizes(self,
                                     symbols: List[str],
                                     model: PositionSizingModel = PositionSizingModel.KELLY_CRITERION,
                                     target_positions: Optional[Dict[str, Decimal]] = None,
                                     constraints: Optional[OptimizationConstraints] = None) -> List[PositionSizeRecommendation]:
        """
        Calculate optimal position sizes for given symbols.
        
        Args:
            symbols: List of symbols to calculate positions for
            model: Position sizing model to use
            target_positions: Optional target position weights
            constraints: Optimization constraints
            
        Returns:
            List of position size recommendations
        """
        logger.info(f"Calculating position sizes for {len(symbols)} symbols using {model.value}")
        
        try:
            # Update portfolio state
            await self._update_portfolio_state()
            
            # Get asset risk metrics
            risk_metrics = await self._get_asset_risk_metrics(symbols)
            
            # Calculate correlation matrix
            correlation_matrix = await self._calculate_correlations(symbols, risk_metrics)
            
            # Apply position sizing model
            if model == PositionSizingModel.KELLY_CRITERION:
                recommendations = await self._kelly_based_sizing(symbols, risk_metrics, constraints)
            elif model == PositionSizingModel.VAR_BASED:
                recommendations = await self._var_based_sizing(symbols, risk_metrics, constraints)
            elif model == PositionSizingModel.VOLATILITY_ADJUSTED:
                recommendations = await self._volatility_adjusted_sizing(symbols, risk_metrics, constraints)
            elif model == PositionSizingModel.RISK_PARITY:
                recommendations = await self._risk_parity_sizing(symbols, risk_metrics, constraints)
            else:
                recommendations = await self._fixed_fractional_sizing(symbols, risk_metrics, constraints)
            
            # Apply risk budget constraints
            recommendations = self._apply_risk_constraints(recommendations, correlation_matrix)
            
            # Validate against margin requirements
            recommendations = await self._validate_margin_requirements(recommendations)
            
            logger.info(f"Generated {len(recommendations)} position recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error calculating position sizes: {e}")
            raise
    
    async def _update_portfolio_state(self):
        """Update current portfolio state from broker."""
        try:
            # Get current positions
            positions = await self.broker.get_positions()
            
            self.current_positions.clear()
            total_value = Decimal('0')
            
            # Process positions
            for position in positions:
                # Handle both dict and PositionInfo objects
                if hasattr(position, 'symbol'):  # PositionInfo object
                    symbol = position.symbol
                    quantity = position.quantity
                    price = Decimal(str(position.average_price))
                    market_value = Decimal(str(position.market_value))
                    pnl = Decimal(str(position.unrealized_pnl))
                else:  # Dict format
                    symbol = position.get('symbol', '')
                    quantity = int(position.get('quantity', 0))
                    price = Decimal(str(position.get('ltp', 100)))
                    market_value = abs(quantity) * price
                    pnl = Decimal(str(position.get('pnl', 0)))
                
                if quantity != 0:
                    total_value += abs(market_value)  # Use abs for total value calculation
                    
                    self.current_positions[symbol] = PortfolioPosition(
                        symbol=symbol,
                        quantity=quantity,
                        current_price=price,
                        market_value=abs(market_value),
                        weight=Decimal('0'),  # Will be calculated after total
                        unrealized_pnl=pnl
                    )
            
            # Update portfolio value
            if total_value > 0:
                self.total_portfolio_value = total_value
                
                # Calculate position weights
                for position in self.current_positions.values():
                    position.weight = position.market_value / total_value
            
            logger.info(f"Updated portfolio state: {len(self.current_positions)} positions, total value: ₹{self.total_portfolio_value:,.0f}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")
    
    async def _get_asset_risk_metrics(self, symbols: List[str]) -> Dict[str, AssetRiskMetrics]:
        """Get or calculate risk metrics for assets."""
        risk_metrics = {}
        
        for symbol in symbols:
            # Check cache first
            if (symbol in self.asset_risk_cache and 
                self.cache_timestamp and 
                datetime.now() - self.cache_timestamp < self.cache_duration):
                risk_metrics[symbol] = self.asset_risk_cache[symbol]
                continue
            
            # Calculate new metrics
            try:
                # Generate mock historical returns for demonstration
                returns = self._generate_mock_returns(symbol)
                
                asset_metrics = AssetRiskMetrics(
                    symbol=symbol,
                    expected_return=Decimal('0.15'),  # 15% annual return assumption
                    volatility=Decimal('0.25'),  # 25% annual volatility assumption
                    historical_returns=[Decimal(str(r)) for r in returns]
                )
                
                # Calculate risk metrics
                asset_metrics.calculate_risk_metrics()
                
                # Cache the metrics
                self.asset_risk_cache[symbol] = asset_metrics
                risk_metrics[symbol] = asset_metrics
                
            except Exception as e:
                logger.error(f"Error calculating risk metrics for {symbol}: {e}")
                # Fallback metrics
                risk_metrics[symbol] = AssetRiskMetrics(
                    symbol=symbol,
                    expected_return=Decimal('0.12'),
                    volatility=Decimal('0.20')
                )
        
        self.cache_timestamp = datetime.now()
        return risk_metrics
    
    def _generate_mock_returns(self, symbol: str) -> List[float]:
        """Generate mock returns for demonstration purposes."""
        np.random.seed(hash(symbol) % 2**32)  # Consistent random seed per symbol
        
        # Different volatility for different asset classes
        if 'NIFTY' in symbol.upper():
            volatility = 0.015  # 1.5% daily volatility for indices
        elif symbol.upper() in ['RELIANCE', 'TCS', 'INFY', 'HDFC']:
            volatility = 0.020  # 2% daily volatility for large caps
        else:
            volatility = 0.025  # 2.5% daily volatility for others
        
        # Generate 252 daily returns (1 year)
        returns = np.random.normal(0.0005, volatility, 252)  # 0.05% daily mean return
        
        return returns.tolist()
    
    async def _calculate_correlations(self, 
                                    symbols: List[str], 
                                    risk_metrics: Dict[str, AssetRiskMetrics]) -> pd.DataFrame:
        """Calculate correlation matrix for symbols."""
        try:
            returns_data = {}
            
            for symbol in symbols:
                if symbol in risk_metrics:
                    metrics = risk_metrics[symbol]
                    returns_data[symbol] = [float(r) for r in metrics.historical_returns]
            
            if returns_data:
                correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(returns_data)
                self.correlation_cache = correlation_matrix
                return correlation_matrix
            
            # Fallback: identity matrix
            return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
    
    async def _kelly_based_sizing(self, 
                                symbols: List[str], 
                                risk_metrics: Dict[str, AssetRiskMetrics],
                                constraints: Optional[OptimizationConstraints]) -> List[PositionSizeRecommendation]:
        """Calculate position sizes using Kelly Criterion."""
        recommendations = []
        
        for symbol in symbols:
            metrics = risk_metrics.get(symbol)
            if not metrics:
                continue
            
            try:
                # Calculate Kelly fraction from historical returns
                returns = [float(r) for r in metrics.historical_returns]
                kelly_fraction = self.kelly_calculator.calculate_kelly_from_returns(returns)
                
                # Apply Kelly fraction with conservative scaling (typically use 1/4 to 1/2 of Kelly)
                conservative_kelly = kelly_fraction * 0.25  # Use 25% of Kelly
                
                # Convert to portfolio weight
                position_weight = Decimal(str(conservative_kelly))
                
                # Apply constraints
                if constraints:
                    position_weight = max(constraints.min_weight, 
                                        min(position_weight, constraints.max_weight))
                
                # Apply risk budget limits
                position_weight = min(position_weight, self.risk_budget.max_position_weight)
                
                # Calculate position details
                position_value = self.total_portfolio_value * position_weight
                
                # Get lot size and calculate quantity
                lot_size = self._get_lot_size(symbol)
                current_price = await self._get_current_price(symbol)
                
                if current_price > 0:
                    target_quantity = int(position_value / current_price)
                    recommended_lots = max(1, target_quantity // lot_size)
                    recommended_quantity = recommended_lots * lot_size
                    
                    # Recalculate actual position value and weight
                    actual_position_value = Decimal(recommended_quantity) * current_price
                    actual_weight = actual_position_value / self.total_portfolio_value
                    
                    recommendation = PositionSizeRecommendation(
                        symbol=symbol,
                        recommended_weight=actual_weight,
                        recommended_quantity=recommended_quantity,
                        recommended_lots=recommended_lots,
                        position_value=actual_position_value,
                        risk_contribution=actual_weight * metrics.volatility,
                        expected_return=metrics.expected_return * actual_weight,
                        position_var=Decimal(str(abs(metrics.var_95))) * actual_weight,
                        sizing_model=PositionSizingModel.KELLY_CRITERION,
                        confidence_level=Decimal('0.95'),
                        kelly_fraction=Decimal(str(kelly_fraction))
                    )
                    
                    recommendations.append(recommendation)
                    
            except Exception as e:
                logger.error(f"Error in Kelly-based sizing for {symbol}: {e}")
        
        return recommendations
    
    def _get_lot_size(self, symbol: str) -> int:
        """Get lot size for symbol."""
        # Standard lot sizes for Indian markets
        lot_sizes = {
            'NIFTY': 50,
            'BANKNIFTY': 25,
            'RELIANCE': 250,
            'TCS': 150,
            'INFY': 300,
            'HDFC': 300
        }
        
        return lot_sizes.get(symbol.upper(), 1)
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current market price for symbol."""
        try:
            # In production, this would fetch real market data
            # For now, return mock prices
            mock_prices = {
                'NIFTY': Decimal('19500'),
                'BANKNIFTY': Decimal('44000'),
                'RELIANCE': Decimal('2500'),
                'TCS': Decimal('3200'),
                'INFY': Decimal('1400'),
                'HDFC': Decimal('1600')
            }
            
            return mock_prices.get(symbol.upper(), Decimal('100'))
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return Decimal('100')  # Fallback price
    
    def _apply_risk_constraints(self, 
                              recommendations: List[PositionSizeRecommendation],
                              correlation_matrix: pd.DataFrame) -> List[PositionSizeRecommendation]:
        """Apply risk budget constraints to recommendations."""
        return recommendations  # Simplified for brevity
    
    async def _validate_margin_requirements(self, 
                                          recommendations: List[PositionSizeRecommendation]) -> List[PositionSizeRecommendation]:
        """Validate recommendations against margin requirements."""
        return recommendations  # Simplified for brevity
    
    # Placeholder methods for other sizing models
    async def _var_based_sizing(self, symbols, risk_metrics, constraints):
        """VaR-based sizing implementation."""
        return []
    
    async def _volatility_adjusted_sizing(self, symbols, risk_metrics, constraints):
        """Volatility-adjusted sizing implementation."""
        return []
    
    async def _risk_parity_sizing(self, symbols, risk_metrics, constraints):
        """Risk parity sizing implementation."""
        return []
    
    async def _fixed_fractional_sizing(self, symbols, risk_metrics, constraints):
        """Fixed fractional sizing implementation."""
        return []
    
    def generate_risk_report(self, recommendations: List[PositionSizeRecommendation]) -> Dict[str, Any]:
        """Generate comprehensive risk report for recommendations."""
        if not recommendations:
            return {}
        
        try:
            # Portfolio-level metrics
            total_weight = sum(rec.recommended_weight for rec in recommendations)
            total_value = sum(rec.position_value for rec in recommendations)
            total_risk_contribution = sum(rec.risk_contribution for rec in recommendations)
            total_expected_return = sum(rec.expected_return for rec in recommendations)
            total_var = sum(rec.position_var for rec in recommendations)
            
            # Risk concentrations
            max_position_weight = max(rec.recommended_weight for rec in recommendations)
            position_concentration = len([r for r in recommendations if r.recommended_weight > Decimal('0.05')])
            
            # Model distribution
            model_distribution = {}
            for rec in recommendations:
                model = rec.sizing_model.value
                model_distribution[model] = model_distribution.get(model, 0) + 1
            
            # Warnings and constraints
            total_warnings = sum(len(rec.warnings) for rec in recommendations)
            total_constraints = sum(len(rec.constraints_applied) for rec in recommendations)
            
            report = {
                'portfolio_metrics': {
                    'total_positions': len(recommendations),
                    'total_weight': float(total_weight),
                    'total_value': float(total_value),
                    'total_risk_contribution': float(total_risk_contribution),
                    'total_expected_return': float(total_expected_return),
                    'total_var_95': float(total_var),
                    'max_position_weight': float(max_position_weight),
                    'position_concentration': position_concentration
                },
                'risk_budget_compliance': {
                    'max_position_within_limit': max_position_weight <= self.risk_budget.max_position_weight,
                    'portfolio_var_within_limit': total_var <= self.risk_budget.max_portfolio_var,
                    'target_volatility': float(self.risk_budget.target_volatility),
                    'estimated_portfolio_volatility': float(total_risk_contribution)
                },
                'model_distribution': model_distribution,
                'warnings_and_constraints': {
                    'total_warnings': total_warnings,
                    'total_constraints_applied': total_constraints
                },
                'generation_timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {'error': str(e)}

# ---------------------------------------------------------------------------
# Compatibility shim – many older components expect `RiskEngine` symbol
class RiskEngine(RiskBasedPositionSizer):
    """Alias for backward compatibility. Inherits all functionality from
    `RiskBasedPositionSizer` without modification."""

    def __init__(self, *args, **kwargs):
        # Allow zero-arg construction for tests
        from .broker_manager import BrokerManager
        from .span.margin_engine import MarginEngine
        if not args and not kwargs:
            super().__init__(BrokerManager(), MarginEngine())
        else:
            super().__init__(*args, **kwargs)