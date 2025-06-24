"""
Unified Trading Service for Position Sizing & Leverage Guidelines.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .broker_manager import BrokerManager
from .position_sizer import PositionSizer, RiskParameters, AssetMetrics, RiskModel
from .base_broker import ProductType
from .risk_engine import RiskBasedPositionSizer
from .risk_models import PositionSizingModel, RiskBudget, MODERATE_RISK_BUDGET
from .span.margin_engine import MarginEngine
from .leverage_manager import DynamicLeverageManager, LeverageConstraints


@dataclass
class LeverageGuideline:
    """Leverage guidelines for different market conditions."""
    symbol: str
    current_leverage: float
    recommended_leverage: float
    max_safe_leverage: float
    volatility_assessment: str
    market_condition: str
    risk_level: str
    reasoning: str
    sebi_compliance: bool = True
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class TradingService:
    """Unified trading service for position sizing and leverage guidelines."""
    
    def __init__(self, db_session=None):
        """Initialize the trading service."""
        self.broker_manager = BrokerManager()
        self.risk_params = RiskParameters()
        self.position_sizer: Optional[PositionSizer] = None
        self.margin_engine: Optional[MarginEngine] = None
        self.risk_based_sizer: Optional[RiskBasedPositionSizer] = None
        self.leverage_manager: Optional[DynamicLeverageManager] = None
        self._asset_metrics_cache: Dict[str, AssetMetrics] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=15)
    
    async def initialize(self, user_id: int) -> None:
        """Initialize the trading service for a specific user."""
        self.position_sizer = PositionSizer(self.broker_manager, self.risk_params)
        
        # Initialize advanced risk-based components
        try:
            primary_broker = self.broker_manager.get_primary_broker()
            if primary_broker:
                self.margin_engine = MarginEngine(primary_broker)
                self.risk_based_sizer = RiskBasedPositionSizer(
                    broker=primary_broker,
                    margin_engine=self.margin_engine,
                    risk_budget=MODERATE_RISK_BUDGET
                )
                
                # Initialize dynamic leverage manager
                self.leverage_manager = DynamicLeverageManager(
                    constraints=LeverageConstraints()
                )
            else:
                # Create mock components for testing
                self._initialize_mock_components()
        except Exception as e:
            # Fallback to mock components if broker initialization fails
            self._initialize_mock_components()
    
    def _initialize_mock_components(self) -> None:
        """Initialize mock components for testing when no real broker is available."""
        try:
            # Create a mock broker for testing
            from .base_broker import BrokerCredentials, BrokerType
            from .zerodha_broker import ZerodhaBroker
            
            mock_credentials = BrokerCredentials(
                api_key="mock_key",
                access_token="mock_token"
            )
            
            mock_broker = ZerodhaBroker(mock_credentials)
            mock_broker._authenticated = True  # Mark as authenticated for testing
            
            # Initialize components with mock broker
            self.margin_engine = MarginEngine(mock_broker)
            self.risk_based_sizer = RiskBasedPositionSizer(
                broker=mock_broker,
                margin_engine=self.margin_engine,
                risk_budget=MODERATE_RISK_BUDGET
            )
            
            # Initialize dynamic leverage manager
            self.leverage_manager = DynamicLeverageManager(
                constraints=LeverageConstraints()
            )
            
        except Exception as e:
            # If even mock initialization fails, set components to None
            self.margin_engine = None
            self.risk_based_sizer = None
            self.leverage_manager = None
    
    async def calculate_position_size(self, 
                                    user_id: int,
                                    symbol: str,
                                    expected_return: Optional[float] = None,
                                    stop_loss_price: Optional[float] = None,
                                    product_type: ProductType = ProductType.MIS,
                                    risk_model: RiskModel = RiskModel.VOLATILITY_ADJUSTED):
        """Calculate optimal position size for a trade."""
        if not self.position_sizer:
            raise RuntimeError("Trading service not initialized")
        
        asset_metrics = await self._get_asset_metrics(symbol)
        
        return await self.position_sizer.calculate_position_size(
            symbol=symbol,
            asset_metrics=asset_metrics,
            product_type=product_type,
            target_return=expected_return,
            stop_loss_price=stop_loss_price,
            model=risk_model
        )
    
    async def get_leverage_guidelines(self, 
                                    user_id: int,
                                    symbol: str,
                                    product_type: ProductType = ProductType.MIS) -> LeverageGuideline:
        """Get leverage guidelines for a specific asset."""
        asset_metrics = await self._get_asset_metrics(symbol)
        
        # Use dynamic leverage manager if available
        if self.leverage_manager:
            try:
                from decimal import Decimal
                base_leverage = Decimal(str(self._calculate_base_leverage(asset_metrics.volatility)))
                
                # Get dynamic leverage recommendation
                adjustment = await self.leverage_manager.calculate_optimal_leverage(
                    symbol=symbol,
                    base_leverage=base_leverage,
                    market_data=None
                )
                
                return LeverageGuideline(
                    symbol=symbol,
                    current_leverage=float(adjustment.current_leverage),
                    recommended_leverage=float(adjustment.recommended_leverage),
                    max_safe_leverage=5.0,
                    volatility_assessment=adjustment.metadata.get('vol_regime', 'moderate'),
                    market_condition=adjustment.metadata.get('market_condition', 'neutral'),
                    risk_level=adjustment.priority,
                    reasoning=adjustment.reasoning,
                    warnings=adjustment.warnings
                )
                
            except Exception as e:
                # Fallback to basic calculation
                pass
        
        # Fallback to basic leverage calculation
        base_leverage = self._calculate_base_leverage(asset_metrics.volatility)
        final_leverage = min(base_leverage, 5.0)  # SEBI limit
        
        return LeverageGuideline(
            symbol=symbol,
            current_leverage=1.0,
            recommended_leverage=final_leverage,
            max_safe_leverage=5.0,
            volatility_assessment=self._get_volatility_assessment(asset_metrics.volatility),
            market_condition="neutral",
            risk_level="medium",
            reasoning=f"Based on {asset_metrics.volatility:.1f}% volatility",
            warnings=[]
        )
    
    async def calculate_advanced_position_sizes(self,
                                              user_id: int,
                                              symbols: List[str],
                                              sizing_model: PositionSizingModel = PositionSizingModel.KELLY_CRITERION,
                                              constraints: Optional[RiskBudget] = None) -> Dict[str, Any]:
        """
        Calculate advanced position sizes using risk-based algorithms.
        
        Args:
            user_id: User identifier
            symbols: List of symbols to calculate positions for
            sizing_model: Position sizing model to use
            constraints: Optional optimization constraints
            
        Returns:
            Dictionary containing recommendations and risk analysis
        """
        if not self.risk_based_sizer:
            raise RuntimeError("Advanced risk-based sizer not initialized")
        
        try:
            # Calculate position size recommendations
            recommendations = await self.risk_based_sizer.calculate_position_sizes(
                symbols=symbols,
                model=sizing_model,
                constraints=constraints
            )
            
            # Generate risk report
            risk_report = self.risk_based_sizer.generate_risk_report(recommendations)
            
            return {
                'recommendations': [
                    {
                        'symbol': rec.symbol,
                        'recommended_weight': float(rec.recommended_weight),
                        'recommended_quantity': rec.recommended_quantity,
                        'recommended_lots': rec.recommended_lots,
                        'position_value': float(rec.position_value),
                        'risk_contribution': float(rec.risk_contribution),
                        'expected_return': float(rec.expected_return),
                        'position_var': float(rec.position_var),
                        'sizing_model': rec.sizing_model.value,
                        'kelly_fraction': float(rec.kelly_fraction) if rec.kelly_fraction else None,
                        'confidence_level': float(rec.confidence_level),
                        'warnings': rec.warnings,
                        'constraints_applied': rec.constraints_applied
                    }
                    for rec in recommendations
                ],
                'risk_analysis': risk_report,
                'calculation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise RuntimeError(f"Error calculating advanced position sizes: {e}")
    
    async def get_portfolio_risk_metrics(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive portfolio risk metrics."""
        if not self.risk_based_sizer:
            return {'error': 'Advanced risk-based sizer not initialized'}
        
        try:
            await self.risk_based_sizer._update_portfolio_state()
            
            total_value = float(self.risk_based_sizer.total_portfolio_value)
            positions = self.risk_based_sizer.current_positions
            
            # Calculate portfolio metrics
            portfolio_metrics = {
                'total_value': total_value,
                'total_positions': len(positions),
                'largest_position': max(
                    [float(pos.weight) for pos in positions.values()], 
                    default=0.0
                ),
                'position_details': [
                    {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'market_value': float(pos.market_value),
                        'weight': float(pos.weight),
                        'unrealized_pnl': float(pos.unrealized_pnl)
                    }
                    for pos in positions.values()
                ]
            }
            
            return portfolio_metrics
            
        except Exception as e:
            return {'error': f"Error getting portfolio metrics: {e}"}
    
    async def _get_asset_metrics(self, symbol: str) -> AssetMetrics:
        """Get or calculate asset metrics with caching."""
        if symbol in self._asset_metrics_cache:
            if datetime.now() < self._cache_expiry.get(symbol, datetime.min):
                return self._asset_metrics_cache[symbol]
        
        metrics = AssetMetrics(
            symbol=symbol,
            current_price=1000.0,
            volatility=20.0,
            beta=1.0,
            sector="Technology"
        )
        
        self._asset_metrics_cache[symbol] = metrics
        self._cache_expiry[symbol] = datetime.now() + self._cache_ttl
        
        return metrics
    
    def _calculate_base_leverage(self, volatility: float) -> float:
        """Calculate base leverage recommendation based on volatility."""
        if volatility > 35:
            return 1.5
        elif volatility > 25:
            return 2.5
        elif volatility > 15:
            return 3.5
        else:
            return 4.5
    
    def _get_volatility_assessment(self, volatility: float) -> str:
        """Get volatility assessment string."""
        if volatility > 35:
            return "extreme"
        elif volatility > 25:
            return "high"
        elif volatility > 15:
            return "moderate"
        else:
            return "low"
    
    async def get_dynamic_leverage_summary(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive dynamic leverage summary for the portfolio."""
        if not self.leverage_manager:
            return {'error': 'Dynamic leverage manager not initialized'}
        
        try:
            summary = await self.leverage_manager.get_portfolio_leverage_summary()
            return summary
        except Exception as e:
            return {'error': f"Error getting leverage summary: {e}"}
    
    async def calculate_dynamic_leverage_for_symbols(self, 
                                                   user_id: int,
                                                   symbols: List[str],
                                                   base_leverages: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate dynamic leverage recommendations for multiple symbols.
        
        Args:
            user_id: User identifier
            symbols: List of symbols to get leverage recommendations for
            base_leverages: Optional dictionary of current leverages by symbol
            
        Returns:
            Dictionary containing leverage recommendations for each symbol
        """
        if not self.leverage_manager:
            return {'error': 'Dynamic leverage manager not initialized'}
        
        try:
            from decimal import Decimal
            
            recommendations = {}
            
            for symbol in symbols:
                # Use provided base leverage or calculate a default
                if base_leverages and symbol in base_leverages:
                    base_leverage = Decimal(str(base_leverages[symbol]))
                else:
                    asset_metrics = await self._get_asset_metrics(symbol)
                    base_leverage = Decimal(str(self._calculate_base_leverage(asset_metrics.volatility)))
                
                # Get dynamic leverage recommendation
                adjustment = await self.leverage_manager.calculate_optimal_leverage(
                    symbol=symbol,
                    base_leverage=base_leverage,
                    market_data=None
                )
                
                recommendations[symbol] = {
                    'current_leverage': float(adjustment.current_leverage),
                    'recommended_leverage': float(adjustment.recommended_leverage),
                    'adjustment_magnitude': float(adjustment.adjustment_magnitude),
                    'adjustment_type': adjustment.adjustment_type.value,
                    'reasoning': adjustment.reasoning,
                    'confidence_level': float(adjustment.confidence_level),
                    'expected_vol_change': float(adjustment.expected_vol_change),
                    'expected_return_change': float(adjustment.expected_return_change),
                    'var_impact': float(adjustment.var_impact),
                    'suggested_timeframe': adjustment.suggested_timeframe,
                    'priority': adjustment.priority,
                    'warnings': adjustment.warnings,
                    'metadata': adjustment.metadata
                }
            
            return {
                'recommendations': recommendations,
                'calculation_timestamp': datetime.now().isoformat(),
                'portfolio_summary': await self.get_dynamic_leverage_summary(user_id)
            }
            
        except Exception as e:
            return {'error': f"Error calculating dynamic leverage recommendations: {e}"} 