"""
Dynamic Leverage Management System.

This module implements intelligent leverage adjustment algorithms that respond to:
- Market volatility regimes
- Portfolio drawdown levels  
- Regulatory constraints
- Asset class characteristics
- Real-time market conditions
"""

import asyncio
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Market volatility regimes."""
    ULTRA_LOW = "ultra_low"      # VIX < 12, very calm markets
    LOW = "low"                  # VIX 12-16, normal conditions
    MODERATE = "moderate"        # VIX 16-20, slightly elevated
    HIGH = "high"                # VIX 20-30, stressed conditions
    EXTREME = "extreme"          # VIX > 30, crisis conditions


class MarketCondition(Enum):
    """Overall market conditions."""
    BULL_STRONG = "bull_strong"      # Strong uptrend, high momentum
    BULL_MODERATE = "bull_moderate"  # Moderate uptrend
    NEUTRAL = "neutral"              # Sideways/range-bound
    BEAR_MODERATE = "bear_moderate"  # Moderate downtrend
    BEAR_STRONG = "bear_strong"      # Strong downtrend, high fear


class LeverageAdjustmentType(Enum):
    """Types of leverage adjustments."""
    VOLATILITY_BASED = "volatility_based"
    DRAWDOWN_BASED = "drawdown_based"
    MOMENTUM_BASED = "momentum_based"
    CORRELATION_BASED = "correlation_based"
    REGULATORY_BASED = "regulatory_based"
    PORTFOLIO_HEAT = "portfolio_heat"


@dataclass
class VolatilityMetrics:
    """Market volatility metrics."""
    current_vol: Decimal
    vol_percentile: Decimal  # Historical percentile
    vol_regime: VolatilityRegime
    regime_persistence: int  # Days in current regime
    vol_trend: str  # "rising", "falling", "stable"
    vol_acceleration: Decimal
    
    # Term structure
    short_term_vol: Decimal  # 1-week
    medium_term_vol: Decimal  # 1-month
    long_term_vol: Decimal   # 3-month
    
    calculation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketConditionMetrics:
    """Market condition assessment."""
    condition: MarketCondition
    trend_strength: Decimal  # 0-1 scale
    momentum_score: Decimal  # -1 to 1
    sentiment_score: Decimal  # -1 to 1
    
    # Technical indicators
    rsi: Decimal
    moving_avg_slope: Decimal
    bollinger_position: Decimal  # Position within Bollinger bands
    
    # Cross-asset signals
    bond_equity_ratio: Decimal
    safe_haven_flows: Decimal
    
    calculation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DrawdownMetrics:
    """Portfolio drawdown analysis."""
    current_drawdown: Decimal
    max_drawdown_1m: Decimal
    max_drawdown_3m: Decimal
    max_drawdown_6m: Decimal
    max_drawdown_1y: Decimal
    
    # Recovery metrics
    drawdown_duration_days: int
    recovery_factor: Decimal  # Speed of recovery
    pain_index: Decimal  # Sustained drawdown measure
    
    # Risk-adjusted metrics
    calmar_ratio: Decimal
    sterling_ratio: Decimal
    
    calculation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LeverageAdjustment:
    """Leverage adjustment recommendation."""
    symbol: str
    current_leverage: Decimal
    recommended_leverage: Decimal
    adjustment_magnitude: Decimal
    adjustment_type: LeverageAdjustmentType
    reasoning: str
    confidence_level: Decimal
    
    # Risk impact
    expected_vol_change: Decimal
    expected_return_change: Decimal
    var_impact: Decimal
    
    # Implementation details
    suggested_timeframe: str  # "immediate", "gradual", "on_next_trade"
    priority: str  # "high", "medium", "low"
    
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    calculation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LeverageConstraints:
    """Leverage constraints by asset class and conditions."""
    
    # Asset class maximums
    equity_max: Decimal = Decimal('5.0')      # SEBI limit
    options_max: Decimal = Decimal('10.0')    # Higher for options
    futures_max: Decimal = Decimal('8.0')     # Moderate for futures
    commodity_max: Decimal = Decimal('3.0')   # Conservative for commodities
    
    # Volatility-based limits
    ultra_low_vol_max: Decimal = Decimal('8.0')
    low_vol_max: Decimal = Decimal('6.0')
    moderate_vol_max: Decimal = Decimal('4.0')
    high_vol_max: Decimal = Decimal('2.5')
    extreme_vol_max: Decimal = Decimal('1.5')
    
    # Drawdown-based limits
    drawdown_0_5_max: Decimal = Decimal('5.0')   # 0-5% drawdown
    drawdown_5_10_max: Decimal = Decimal('3.0')  # 5-10% drawdown
    drawdown_10_15_max: Decimal = Decimal('2.0') # 10-15% drawdown
    drawdown_15_plus_max: Decimal = Decimal('1.0') # >15% drawdown
    
    # Position concentration limits
    max_single_position_leverage: Decimal = Decimal('2.0')
    max_sector_leverage: Decimal = Decimal('4.0')
    
    # Intraday vs overnight
    intraday_multiplier: Decimal = Decimal('1.5')
    overnight_multiplier: Decimal = Decimal('0.8')


class DynamicLeverageManager:
    """
    Dynamic leverage management system.
    
    Intelligently adjusts leverage based on:
    - Market volatility regimes
    - Portfolio drawdown levels
    - Market conditions and momentum
    - Asset class characteristics
    - Regulatory constraints
    - Portfolio heat and concentration
    """
    
    def __init__(self,
                 constraints: Optional[LeverageConstraints] = None):
        """Initialize the dynamic leverage manager."""
        self.constraints = constraints or LeverageConstraints()
        
        # Historical data for analysis
        self.volatility_history: List[Decimal] = []
        self.return_history: List[Decimal] = []
        self.drawdown_history: List[Decimal] = []
        
        # Current state
        self.current_vol_regime = VolatilityRegime.MODERATE
        self.current_market_condition = MarketCondition.NEUTRAL
        self.current_portfolio_drawdown = Decimal('0')
        
        # Cache for expensive calculations
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_timestamp: Optional[datetime] = None
        self.cache_duration = timedelta(minutes=15)  # 15-minute cache
        
        logger.info("Dynamic leverage manager initialized")
    
    async def calculate_optimal_leverage(self,
                                       symbol: str,
                                       base_leverage: Decimal,
                                       market_data: Optional[Dict] = None) -> LeverageAdjustment:
        """
        Calculate optimal leverage for a symbol based on current conditions.
        
        Args:
            symbol: Symbol to calculate leverage for
            base_leverage: Starting/current leverage
            market_data: Optional market data override
            
        Returns:
            LeverageAdjustment recommendation
        """
        logger.info(f"Calculating optimal leverage for {symbol}, base: {base_leverage}")
        
        try:
            # Refresh market analysis
            await self._update_market_analysis(market_data)
            
            # Get current metrics
            vol_metrics = await self._calculate_volatility_metrics(symbol)
            market_metrics = await self._calculate_market_condition_metrics()
            drawdown_metrics = await self._calculate_drawdown_metrics()
            
            # Apply leverage adjustments in sequence
            adjustments = []
            
            # 1. Volatility-based adjustment
            vol_adjustment = self._apply_volatility_adjustment(base_leverage, vol_metrics)
            adjustments.append(vol_adjustment)
            
            # 2. Drawdown-based adjustment
            dd_adjustment = self._apply_drawdown_adjustment(vol_adjustment, drawdown_metrics)
            adjustments.append(dd_adjustment)
            
            # 3. Market condition adjustment
            market_adjustment = self._apply_market_condition_adjustment(dd_adjustment, market_metrics)
            adjustments.append(market_adjustment)
            
            # 4. Asset class constraints
            constrained_leverage = self._apply_asset_class_constraints(symbol, market_adjustment)
            
            # 5. Regulatory constraints
            final_leverage = self._apply_regulatory_constraints(symbol, constrained_leverage)
            
            # Calculate adjustment magnitude and reasoning
            adjustment_magnitude = abs(final_leverage - base_leverage)
            reasoning = self._generate_adjustment_reasoning(adjustments, vol_metrics, market_metrics, drawdown_metrics)
            
            # Assess confidence level
            confidence = self._calculate_confidence_level(vol_metrics, market_metrics, drawdown_metrics)
            
            # Determine adjustment type and priority
            adjustment_type = self._determine_primary_adjustment_type(adjustments)
            priority = "high" if adjustment_magnitude > Decimal('1.0') else "medium" if adjustment_magnitude > Decimal('0.5') else "low"
            
            # Calculate risk impact
            expected_vol_change = (final_leverage / base_leverage - 1) * vol_metrics.current_vol
            expected_return_change = (final_leverage / base_leverage - 1) * Decimal('0.15')  # Assume 15% base return
            var_impact = expected_vol_change * Decimal('1.645')  # 95% VaR approximation
            
            # Generate warnings
            warnings = self._generate_warnings(final_leverage, vol_metrics, drawdown_metrics)
            
            recommendation = LeverageAdjustment(
                symbol=symbol,
                current_leverage=base_leverage,
                recommended_leverage=final_leverage,
                adjustment_magnitude=adjustment_magnitude,
                adjustment_type=adjustment_type,
                reasoning=reasoning,
                confidence_level=confidence,
                expected_vol_change=expected_vol_change,
                expected_return_change=expected_return_change,
                var_impact=var_impact,
                suggested_timeframe="gradual" if adjustment_magnitude > Decimal('1.0') else "immediate",
                priority=priority,
                warnings=warnings,
                metadata={
                    'vol_regime': vol_metrics.vol_regime.value,
                    'market_condition': market_metrics.condition.value,
                    'current_drawdown': float(drawdown_metrics.current_drawdown),
                    'adjustment_chain': [adj for adj in adjustments]
                }
            )
            
            logger.info(f"Leverage recommendation for {symbol}: {base_leverage} → {final_leverage}")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error calculating optimal leverage for {symbol}: {e}")
            # Return conservative fallback
            return LeverageAdjustment(
                symbol=symbol,
                current_leverage=base_leverage,
                recommended_leverage=min(base_leverage, Decimal('2.0')),
                adjustment_magnitude=Decimal('0'),
                adjustment_type=LeverageAdjustmentType.REGULATORY_BASED,
                reasoning="Error in calculation, using conservative fallback",
                confidence_level=Decimal('0.5'),
                expected_vol_change=Decimal('0'),
                expected_return_change=Decimal('0'),
                var_impact=Decimal('0'),
                suggested_timeframe="immediate",
                priority="high",
                warnings=["Calculation error occurred"]
            )
    
    async def _update_market_analysis(self, market_data: Optional[Dict] = None):
        """Update market analysis and regime detection."""
        try:
            # Check cache first
            if (self.cache_timestamp and 
                datetime.now() - self.cache_timestamp < self.cache_duration):
                return
            
            # Update volatility history (mock data for demonstration)
            if not self.volatility_history or len(self.volatility_history) < 252:
                # Generate mock volatility data
                np.random.seed(int(datetime.now().timestamp()) % 1000)
                mock_vols = np.random.lognormal(np.log(0.20), 0.3, 252)  # 20% mean vol
                self.volatility_history = [Decimal(str(vol)) for vol in mock_vols[-100:]]
            
            # Update return history (mock data)
            if not self.return_history or len(self.return_history) < 252:
                np.random.seed(int(datetime.now().timestamp()) % 1000 + 1)
                mock_returns = np.random.normal(0.0005, 0.015, 252)  # 0.05% daily mean, 1.5% daily vol
                self.return_history = [Decimal(str(ret)) for ret in mock_returns[-100:]]
            
            # Calculate portfolio value for drawdown analysis
            portfolio_values = []
            cumulative_return = Decimal('1')
            for ret in self.return_history:
                cumulative_return *= (1 + ret)
                portfolio_values.append(cumulative_return)
            
            # Calculate drawdowns
            running_max = Decimal('1')
            drawdowns = []
            for value in portfolio_values:
                if value > running_max:
                    running_max = value
                drawdown = (value - running_max) / running_max
                drawdowns.append(drawdown)
            
            self.drawdown_history = drawdowns
            self.current_portfolio_drawdown = drawdowns[-1] if drawdowns else Decimal('0')
            
            # Update cache timestamp
            self.cache_timestamp = datetime.now()
            
            logger.debug("Market analysis updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating market analysis: {e}")
    
    async def _calculate_volatility_metrics(self, symbol: str) -> VolatilityMetrics:
        """Calculate comprehensive volatility metrics."""
        try:
            current_vol = self.volatility_history[-1] if self.volatility_history else Decimal('0.20')
            
            # Calculate percentile
            vol_percentile = Decimal('50')  # Default to 50th percentile
            if len(self.volatility_history) >= 20:
                sorted_vols = sorted(self.volatility_history)
                rank = sum(1 for v in sorted_vols if v <= current_vol)
                vol_percentile = Decimal(str(rank / len(sorted_vols) * 100))
            
            # Determine volatility regime
            vol_regime = VolatilityRegime.MODERATE
            if current_vol < Decimal('0.12'):
                vol_regime = VolatilityRegime.ULTRA_LOW
            elif current_vol < Decimal('0.16'):
                vol_regime = VolatilityRegime.LOW
            elif current_vol < Decimal('0.25'):
                vol_regime = VolatilityRegime.MODERATE
            elif current_vol < Decimal('0.35'):
                vol_regime = VolatilityRegime.HIGH
            else:
                vol_regime = VolatilityRegime.EXTREME
            
            # Calculate volatility trend
            vol_trend = "stable"
            if len(self.volatility_history) >= 5:
                recent_avg = sum(self.volatility_history[-5:]) / 5
                older_avg = sum(self.volatility_history[-10:-5]) / 5 if len(self.volatility_history) >= 10 else recent_avg
                
                if recent_avg > older_avg * Decimal('1.1'):
                    vol_trend = "rising"
                elif recent_avg < older_avg * Decimal('0.9'):
                    vol_trend = "falling"
            
            # Term structure (mock)
            short_term_vol = current_vol * Decimal('0.9')
            medium_term_vol = current_vol
            long_term_vol = current_vol * Decimal('1.1')
            
            return VolatilityMetrics(
                current_vol=current_vol,
                vol_percentile=vol_percentile,
                vol_regime=vol_regime,
                regime_persistence=5,  # Mock value
                vol_trend=vol_trend,
                vol_acceleration=Decimal('0.01'),
                short_term_vol=short_term_vol,
                medium_term_vol=medium_term_vol,
                long_term_vol=long_term_vol
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return VolatilityMetrics(
                current_vol=Decimal('0.20'),
                vol_percentile=Decimal('50'),
                vol_regime=VolatilityRegime.MODERATE,
                regime_persistence=1,
                vol_trend="stable",
                vol_acceleration=Decimal('0'),
                short_term_vol=Decimal('0.18'),
                medium_term_vol=Decimal('0.20'),
                long_term_vol=Decimal('0.22')
            )
    
    async def _calculate_market_condition_metrics(self) -> MarketConditionMetrics:
        """Calculate market condition metrics."""
        try:
            # Simple market condition assessment based on recent returns
            if len(self.return_history) >= 20:
                recent_returns = self.return_history[-20:]
                avg_return = sum(recent_returns) / len(recent_returns)
                volatility = Decimal(str(np.std([float(r) for r in recent_returns])))
                
                # Determine market condition
                if avg_return > Decimal('0.002'):
                    condition = MarketCondition.BULL_STRONG
                elif avg_return > Decimal('0.001'):
                    condition = MarketCondition.BULL_MODERATE
                elif avg_return > Decimal('-0.001'):
                    condition = MarketCondition.NEUTRAL
                elif avg_return > Decimal('-0.002'):
                    condition = MarketCondition.BEAR_MODERATE
                else:
                    condition = MarketCondition.BEAR_STRONG
                
                # Calculate momentum and sentiment scores
                momentum_score = max(Decimal('-1'), min(Decimal('1'), avg_return * 1000))
                sentiment_score = momentum_score * Decimal('0.8')  # Slightly dampened
                
                # Mock technical indicators
                rsi = Decimal('50')  # Neutral RSI
                if avg_return > 0:
                    rsi = Decimal('60') + momentum_score * 20
                else:
                    rsi = Decimal('40') + momentum_score * 20
                
                rsi = max(Decimal('10'), min(Decimal('90'), rsi))
                
            else:
                condition = MarketCondition.NEUTRAL
                momentum_score = Decimal('0')
                sentiment_score = Decimal('0')
                rsi = Decimal('50')
            
            return MarketConditionMetrics(
                condition=condition,
                trend_strength=abs(momentum_score),
                momentum_score=momentum_score,
                sentiment_score=sentiment_score,
                rsi=rsi,
                moving_avg_slope=momentum_score,
                bollinger_position=Decimal('0.5'),  # Middle of bands
                bond_equity_ratio=Decimal('0.3'),
                safe_haven_flows=Decimal('0')
            )
            
        except Exception as e:
            logger.error(f"Error calculating market condition metrics: {e}")
            return MarketConditionMetrics(
                condition=MarketCondition.NEUTRAL,
                trend_strength=Decimal('0.5'),
                momentum_score=Decimal('0'),
                sentiment_score=Decimal('0'),
                rsi=Decimal('50'),
                moving_avg_slope=Decimal('0'),
                bollinger_position=Decimal('0.5'),
                bond_equity_ratio=Decimal('0.3'),
                safe_haven_flows=Decimal('0')
            )
    
    async def _calculate_drawdown_metrics(self) -> DrawdownMetrics:
        """Calculate portfolio drawdown metrics."""
        try:
            current_drawdown = self.current_portfolio_drawdown
            
            # Calculate maximum drawdowns over different periods
            max_dd_1m = min(self.drawdown_history[-21:]) if len(self.drawdown_history) >= 21 else current_drawdown
            max_dd_3m = min(self.drawdown_history[-63:]) if len(self.drawdown_history) >= 63 else current_drawdown
            max_dd_6m = min(self.drawdown_history) if self.drawdown_history else current_drawdown
            max_dd_1y = max_dd_6m  # Same as 6m for mock data
            
            # Calculate recovery metrics
            drawdown_duration = 0
            if current_drawdown < 0:
                # Count consecutive negative periods
                for i in range(len(self.drawdown_history) - 1, -1, -1):
                    if self.drawdown_history[i] < 0:
                        drawdown_duration += 1
                    else:
                        break
            
            # Calculate ratios
            avg_return = sum(self.return_history) / len(self.return_history) if self.return_history else Decimal('0')
            annual_return = avg_return * 252
            
            calmar_ratio = Decimal('0')
            if max_dd_6m < 0:
                calmar_ratio = annual_return / abs(max_dd_6m)
            
            return DrawdownMetrics(
                current_drawdown=current_drawdown,
                max_drawdown_1m=max_dd_1m,
                max_drawdown_3m=max_dd_3m,
                max_drawdown_6m=max_dd_6m,
                max_drawdown_1y=max_dd_1y,
                drawdown_duration_days=drawdown_duration,
                recovery_factor=Decimal('0.5'),
                pain_index=abs(max_dd_6m) * drawdown_duration / 30,
                calmar_ratio=calmar_ratio,
                sterling_ratio=calmar_ratio * Decimal('0.9')
            )
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return DrawdownMetrics(
                current_drawdown=Decimal('0'),
                max_drawdown_1m=Decimal('0'),
                max_drawdown_3m=Decimal('0'),
                max_drawdown_6m=Decimal('0'),
                max_drawdown_1y=Decimal('0'),
                drawdown_duration_days=0,
                recovery_factor=Decimal('1'),
                pain_index=Decimal('0'),
                calmar_ratio=Decimal('1'),
                sterling_ratio=Decimal('1')
            )
    
    def _apply_volatility_adjustment(self, base_leverage: Decimal, vol_metrics: VolatilityMetrics) -> Decimal:
        """Apply volatility-based leverage adjustment."""
        # Base adjustment based on volatility regime
        regime_multipliers = {
            VolatilityRegime.ULTRA_LOW: Decimal('1.3'),
            VolatilityRegime.LOW: Decimal('1.1'),
            VolatilityRegime.MODERATE: Decimal('1.0'),
            VolatilityRegime.HIGH: Decimal('0.7'),
            VolatilityRegime.EXTREME: Decimal('0.4')
        }
        
        multiplier = regime_multipliers.get(vol_metrics.vol_regime, Decimal('1.0'))
        
        # Adjust for volatility trend
        if vol_metrics.vol_trend == "rising":
            multiplier *= Decimal('0.9')
        elif vol_metrics.vol_trend == "falling":
            multiplier *= Decimal('1.1')
        
        # Apply regime constraints
        max_leverage = getattr(self.constraints, f"{vol_metrics.vol_regime.value}_vol_max", Decimal('3.0'))
        
        adjusted_leverage = base_leverage * multiplier
        return min(adjusted_leverage, max_leverage)
    
    def _apply_drawdown_adjustment(self, current_leverage: Decimal, dd_metrics: DrawdownMetrics) -> Decimal:
        """Apply drawdown-based leverage adjustment."""
        drawdown_pct = abs(dd_metrics.current_drawdown) * 100
        
        # Progressive leverage reduction based on drawdown
        if drawdown_pct <= 5:
            multiplier = Decimal('1.0')
        elif drawdown_pct <= 10:
            multiplier = Decimal('0.8')
        elif drawdown_pct <= 15:
            multiplier = Decimal('0.6')
        else:
            multiplier = Decimal('0.4')
        
        # Additional reduction for extended drawdown periods
        if dd_metrics.drawdown_duration_days > 30:
            multiplier *= Decimal('0.9')
        elif dd_metrics.drawdown_duration_days > 60:
            multiplier *= Decimal('0.8')
        
        return current_leverage * multiplier
    
    def _apply_market_condition_adjustment(self, current_leverage: Decimal, market_metrics: MarketConditionMetrics) -> Decimal:
        """Apply market condition-based leverage adjustment."""
        condition_multipliers = {
            MarketCondition.BULL_STRONG: Decimal('1.2'),
            MarketCondition.BULL_MODERATE: Decimal('1.1'),
            MarketCondition.NEUTRAL: Decimal('1.0'),
            MarketCondition.BEAR_MODERATE: Decimal('0.8'),
            MarketCondition.BEAR_STRONG: Decimal('0.6')
        }
        
        multiplier = condition_multipliers.get(market_metrics.condition, Decimal('1.0'))
        
        # Adjust based on momentum strength
        momentum_adjustment = 1 + (market_metrics.momentum_score * Decimal('0.1'))
        multiplier *= momentum_adjustment
        
        return current_leverage * multiplier
    
    def _apply_asset_class_constraints(self, symbol: str, current_leverage: Decimal) -> Decimal:
        """Apply asset class-specific constraints."""
        # Determine asset class (simplified)
        if symbol.upper() in ['NIFTY', 'BANKNIFTY', 'SENSEX']:
            max_leverage = self.constraints.equity_max
        elif 'FUT' in symbol.upper():
            max_leverage = self.constraints.futures_max
        elif 'OPT' in symbol.upper() or 'CE' in symbol.upper() or 'PE' in symbol.upper():
            max_leverage = self.constraints.options_max
        else:
            max_leverage = self.constraints.equity_max
        
        return min(current_leverage, max_leverage)
    
    def _apply_regulatory_constraints(self, symbol: str, current_leverage: Decimal) -> Decimal:
        """Apply regulatory constraints (SEBI limits)."""
        # SEBI maximum leverage limit
        sebi_max = Decimal('5.0')
        
        # Additional constraints for specific conditions
        final_leverage = min(current_leverage, sebi_max)
        
        # Ensure minimum leverage of 1.0
        return max(final_leverage, Decimal('1.0'))
    
    def _generate_adjustment_reasoning(self, 
                                     adjustments: List[Decimal],
                                     vol_metrics: VolatilityMetrics,
                                     market_metrics: MarketConditionMetrics,
                                     dd_metrics: DrawdownMetrics) -> str:
        """Generate human-readable reasoning for leverage adjustments."""
        reasons = []
        
        # Volatility reasoning
        if vol_metrics.vol_regime == VolatilityRegime.EXTREME:
            reasons.append("Extreme volatility detected - significant leverage reduction")
        elif vol_metrics.vol_regime == VolatilityRegime.HIGH:
            reasons.append("High volatility environment - moderate leverage reduction")
        elif vol_metrics.vol_regime == VolatilityRegime.ULTRA_LOW:
            reasons.append("Ultra-low volatility - leverage increase opportunity")
        
        # Drawdown reasoning
        if abs(dd_metrics.current_drawdown) > Decimal('0.10'):
            reasons.append("Portfolio drawdown >10% - defensive leverage reduction")
        elif dd_metrics.drawdown_duration_days > 30:
            reasons.append("Extended drawdown period - additional risk reduction")
        
        # Market condition reasoning
        if market_metrics.condition == MarketCondition.BULL_STRONG:
            reasons.append("Strong bullish conditions - moderate leverage increase")
        elif market_metrics.condition in [MarketCondition.BEAR_MODERATE, MarketCondition.BEAR_STRONG]:
            reasons.append("Bearish market conditions - defensive positioning")
        
        return "; ".join(reasons) if reasons else "No significant adjustments required"
    
    def _calculate_confidence_level(self,
                                  vol_metrics: VolatilityMetrics,
                                  market_metrics: MarketConditionMetrics,
                                  dd_metrics: DrawdownMetrics) -> Decimal:
        """Calculate confidence level in the leverage recommendation."""
        confidence = Decimal('0.8')  # Base confidence
        
        # Reduce confidence in extreme conditions
        if vol_metrics.vol_regime == VolatilityRegime.EXTREME:
            confidence *= Decimal('0.7')
        elif vol_metrics.vol_regime == VolatilityRegime.HIGH:
            confidence *= Decimal('0.85')
        
        # Reduce confidence during large drawdowns
        if abs(dd_metrics.current_drawdown) > Decimal('0.15'):
            confidence *= Decimal('0.6')
        
        # Adjust for market condition clarity
        if market_metrics.trend_strength > Decimal('0.8'):
            confidence *= Decimal('1.1')
        elif market_metrics.trend_strength < Decimal('0.3'):
            confidence *= Decimal('0.9')
        
        return min(confidence, Decimal('0.95'))
    
    def _determine_primary_adjustment_type(self, adjustments: List[Decimal]) -> LeverageAdjustmentType:
        """Determine the primary reason for leverage adjustment."""
        # Simplified logic - in practice would track which adjustment had most impact
        return LeverageAdjustmentType.VOLATILITY_BASED
    
    def _generate_warnings(self,
                          final_leverage: Decimal,
                          vol_metrics: VolatilityMetrics,
                          dd_metrics: DrawdownMetrics) -> List[str]:
        """Generate warnings for the leverage recommendation."""
        warnings = []
        
        if final_leverage > Decimal('4.0'):
            warnings.append("High leverage recommendation - monitor closely")
        
        if vol_metrics.vol_regime == VolatilityRegime.EXTREME:
            warnings.append("Extreme volatility conditions - consider position reduction")
        
        if abs(dd_metrics.current_drawdown) > Decimal('0.15'):
            warnings.append("Significant portfolio drawdown - reassess risk tolerance")
        
        if vol_metrics.vol_trend == "rising":
            warnings.append("Rising volatility trend - be prepared for further adjustments")
        
        return warnings
    
    async def get_portfolio_leverage_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio leverage summary."""
        try:
            await self._update_market_analysis()
            
            vol_metrics = await self._calculate_volatility_metrics("PORTFOLIO")
            market_metrics = await self._calculate_market_condition_metrics()
            dd_metrics = await self._calculate_drawdown_metrics()
            
            return {
                'current_volatility_regime': vol_metrics.vol_regime.value,
                'market_condition': market_metrics.condition.value,
                'current_drawdown': float(dd_metrics.current_drawdown),
                'recommended_portfolio_leverage': float(self._calculate_portfolio_target_leverage(vol_metrics, dd_metrics)),
                'risk_budget_utilization': float(abs(dd_metrics.current_drawdown) / Decimal('0.15')),  # Against 15% max
                'leverage_constraints': {
                    'volatility_max': float(getattr(self.constraints, f"{vol_metrics.vol_regime.value}_vol_max", Decimal('3.0'))),
                    'drawdown_max': self._get_drawdown_leverage_limit(dd_metrics),
                    'regulatory_max': 5.0
                },
                'market_regime_persistence': vol_metrics.regime_persistence,
                'portfolio_heat': self._calculate_portfolio_heat(),
                'next_review_recommended': (datetime.now() + timedelta(hours=4)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio leverage summary: {e}")
            return {'error': str(e)}
    
    def _calculate_portfolio_target_leverage(self,
                                           vol_metrics: VolatilityMetrics,
                                           dd_metrics: DrawdownMetrics) -> Decimal:
        """Calculate target portfolio-level leverage."""
        # Start with base leverage
        base_leverage = Decimal('2.0')
        
        # Apply volatility adjustment
        vol_adjusted = self._apply_volatility_adjustment(base_leverage, vol_metrics)
        
        # Apply drawdown adjustment
        dd_adjusted = self._apply_drawdown_adjustment(vol_adjusted, dd_metrics)
        
        # Apply regulatory limits
        return min(dd_adjusted, Decimal('5.0'))
    
    def _get_drawdown_leverage_limit(self, dd_metrics: DrawdownMetrics) -> float:
        """Get leverage limit based on current drawdown."""
        drawdown_pct = abs(dd_metrics.current_drawdown) * 100
        
        if drawdown_pct <= 5:
            return 5.0
        elif drawdown_pct <= 10:
            return 3.0
        elif drawdown_pct <= 15:
            return 2.0
        else:
            return 1.0
    
    def _calculate_portfolio_heat(self) -> float:
        """Calculate portfolio heat measure (0-1 scale)."""
        # Simplified portfolio heat calculation
        base_heat = 0.3  # Base level
        
        # Add heat based on current drawdown
        if self.current_portfolio_drawdown < 0:
            drawdown_heat = abs(float(self.current_portfolio_drawdown)) * 2
            base_heat += drawdown_heat
        
        # Add heat based on volatility
        if self.volatility_history:
            current_vol = float(self.volatility_history[-1])
            if current_vol > 0.25:  # High volatility
                vol_heat = (current_vol - 0.25) * 2
                base_heat += vol_heat
        
        return min(base_heat, 1.0)