"""
Margin Engine for SPAN Integration.

This module provides the main interface for SPAN margin calculations,
integrating with broker APIs and position data.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union

from .span_calculator import SPANCalculator
from .span_models import (
    Portfolio, PortfolioPosition, SPANParameters, MarginResult,
    ContractType, OptionType, MarginType, DEFAULT_SPAN_PARAMETERS
)
from ..base_broker import ProductType
from ..broker_manager import BrokerManager

logger = logging.getLogger(__name__)


class MarginEngine:
    """
    Main margin calculation engine integrating SPAN with broker APIs.
    
    Provides high-level interface for calculating margins, managing
    SPAN parameters, and integrating with broker position data.
    """
    
    def __init__(self, 
                 broker_manager: Optional[BrokerManager] = None,
                 span_parameters: Optional[Dict[str, SPANParameters]] = None):
        """
        Initialize margin engine.
        
        Args:
            broker_manager: Broker manager for fetching positions and market data
            span_parameters: Custom SPAN parameters (uses defaults if None)
        """
        self.broker_manager = broker_manager
        self.span_calculator = SPANCalculator(span_parameters)
        self.logger = logging.getLogger(__name__)
        
        # Cache for market data and calculations
        self._market_data_cache: Dict[str, Dict[str, Any]] = {}
        self._margin_cache: Dict[str, MarginResult] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update: Dict[str, datetime] = {}
    
    async def calculate_portfolio_margin(self, 
                                       portfolio_id: str,
                                       positions: Optional[List[Dict[str, Any]]] = None,
                                       broker_id: Optional[str] = None) -> MarginResult:
        """
        Calculate SPAN margin for a portfolio.
        
        Args:
            portfolio_id: Unique portfolio identifier
            positions: List of position dictionaries (if None, fetches from broker)
            broker_id: Specific broker to use (uses primary if None)
            
        Returns:
            MarginResult with detailed margin breakdown
        """
        try:
            self.logger.info(f"Calculating portfolio margin for {portfolio_id}")
            
            # Check cache first
            cache_key = f"{portfolio_id}_{broker_id}"
            if self._is_cache_valid(cache_key):
                self.logger.debug("Returning cached margin result")
                return self._margin_cache[cache_key]
            
            # Create portfolio from positions
            portfolio = await self._create_portfolio(portfolio_id, positions, broker_id)
            
            # Validate portfolio
            warnings = self.span_calculator.validate_portfolio(portfolio)
            if warnings:
                self.logger.warning(f"Portfolio validation warnings: {warnings}")
            
            # Calculate SPAN margin
            margin_result = self.span_calculator.calculate_portfolio_margin(portfolio)
            
            # Cache the result
            self._margin_cache[cache_key] = margin_result
            self._last_cache_update[cache_key] = datetime.now()
            
            self.logger.info(f"Portfolio margin calculated: {margin_result.total_margin}")
            return margin_result
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio margin: {str(e)}")
            raise
    
    async def calculate_position_margin(self, 
                                      symbol: str,
                                      quantity: int,
                                      product_type: ProductType = ProductType.MIS,
                                      underlying: str = "",
                                      strike_price: Optional[float] = None,
                                      option_type: Optional[str] = None,
                                      expiry_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Calculate margin for a single position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            product_type: Product type (MIS, CNC, NRML)
            underlying: Underlying symbol (for derivatives)
            strike_price: Strike price (for options)
            option_type: Option type (call/put)
            expiry_date: Expiry date
            
        Returns:
            Dictionary with margin details
        """
        try:
            # Create a temporary portfolio with single position
            portfolio = Portfolio(
                portfolio_id=f"temp_{symbol}",
                portfolio_date=date.today()
            )
            
            # Determine contract type
            contract_type = self._determine_contract_type(symbol, strike_price, option_type)
            
            # Get market data
            market_data = await self._get_market_data(symbol)
            
            # Create position
            position = PortfolioPosition(
                symbol=symbol,
                contract_type=contract_type,
                quantity=quantity,
                underlying_symbol=underlying or symbol,
                expiry_date=expiry_date or self._get_next_expiry(symbol),
                strike_price=Decimal(str(strike_price)) if strike_price else None,
                option_type=OptionType(option_type.lower()) if option_type else None,
                current_price=Decimal(str(market_data.get('price', 1000))),
                delta=Decimal(str(market_data.get('delta', 0.5))) if contract_type in [ContractType.OPTION_CALL, ContractType.OPTION_PUT] else None,
                lot_size=market_data.get('lot_size', 1),
                commodity_code=underlying or symbol
            )
            
            portfolio.add_position(position)
            
            # Calculate margin
            margin_result = self.span_calculator.calculate_portfolio_margin(portfolio)
            
            return {
                'symbol': symbol,
                'quantity': quantity,
                'total_margin': float(margin_result.total_margin),
                'scan_risk': float(margin_result.scan_risk),
                'inter_month_spread_charge': float(margin_result.inter_month_spread_charge),
                'spot_month_charge': float(margin_result.spot_month_charge),
                'delivery_charge': float(margin_result.delivery_charge),
                'product_type': product_type.value,
                'contract_type': contract_type.value,
                'position_value': float(position.get_position_value()),
                'margin_percentage': float((margin_result.total_margin / position.get_position_value()) * 100) if position.get_position_value() > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate position margin for {symbol}: {str(e)}")
            raise
    
    async def get_margin_requirements(self, 
                                    trades: List[Dict[str, Any]],
                                    portfolio_id: str = "temp") -> Dict[str, Any]:
        """
        Calculate margin requirements for multiple trades.
        
        Args:
            trades: List of trade dictionaries
            portfolio_id: Portfolio identifier
            
        Returns:
            Aggregate margin requirements
        """
        try:
            # Create portfolio from trades
            portfolio = Portfolio(portfolio_id=portfolio_id)
            
            total_position_value = Decimal('0')
            
            for trade in trades:
                # Create position from trade data
                position = await self._create_position_from_trade(trade)
                portfolio.add_position(position)
                total_position_value += position.get_position_value()
            
            # Calculate SPAN margin
            margin_result = self.span_calculator.calculate_portfolio_margin(portfolio)
            
            # Calculate broker margins for comparison
            broker_margins = []
            if self.broker_manager:
                for trade in trades:
                    try:
                        broker_margin = await self.broker_manager.calculate_margin_requirement(
                            trade['symbol'],
                            trade['quantity'], 
                            ProductType(trade.get('product_type', 'MIS'))
                        )
                        broker_margins.append(broker_margin)
                    except Exception as e:
                        self.logger.warning(f"Failed to get broker margin for {trade['symbol']}: {str(e)}")
            
            total_broker_margin = sum(margin.get('total', 0) for margin in broker_margins)
            
            return {
                'total_positions': len(trades),
                'total_position_value': float(total_position_value),
                'span_margin': {
                    'total': float(margin_result.total_margin),
                    'scan_risk': float(margin_result.scan_risk),
                    'inter_month_spread_charge': float(margin_result.inter_month_spread_charge),
                    'inter_commodity_spread_credit': float(margin_result.inter_commodity_spread_credit),
                    'spot_month_charge': float(margin_result.spot_month_charge),
                    'delivery_charge': float(margin_result.delivery_charge)
                },
                'broker_margin': {
                    'total': total_broker_margin,
                    'individual_margins': broker_margins
                },
                'margin_comparison': {
                    'span_vs_broker_ratio': float(margin_result.total_margin / total_broker_margin) if total_broker_margin > 0 else 1.0,
                    'difference': float(margin_result.total_margin - total_broker_margin),
                    'recommended_margin': float(max(margin_result.total_margin, total_broker_margin))
                },
                'portfolio_id': portfolio_id,
                'calculation_time': margin_result.calculation_timestamp.isoformat(),
                'currency': margin_result.currency
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate margin requirements: {str(e)}")
            raise
    
    async def _create_portfolio(self, 
                              portfolio_id: str,
                              positions: Optional[List[Dict[str, Any]]],
                              broker_id: Optional[str]) -> Portfolio:
        """Create portfolio from position data."""
        portfolio = Portfolio(portfolio_id=portfolio_id, portfolio_date=date.today())
        
        if positions:
            # Use provided positions
            for pos_data in positions:
                position = await self._create_position_from_data(pos_data)
                portfolio.add_position(position)
        elif self.broker_manager:
            # Fetch positions from broker
            broker_positions = await self.broker_manager.get_positions(broker_id)
            for broker_pos in broker_positions:
                position = await self._create_position_from_broker_data(broker_pos)
                portfolio.add_position(position)
        else:
            raise ValueError("No positions provided and no broker manager available")
        
        return portfolio
    
    async def _create_position_from_data(self, pos_data: Dict[str, Any]) -> PortfolioPosition:
        """Create position from dictionary data."""
        symbol = pos_data['symbol']
        contract_type = self._determine_contract_type(
            symbol, 
            pos_data.get('strike_price'),
            pos_data.get('option_type')
        )
        
        # Get market data
        market_data = await self._get_market_data(symbol)
        
        return PortfolioPosition(
            symbol=symbol,
            contract_type=contract_type,
            quantity=int(pos_data['quantity']),
            underlying_symbol=pos_data.get('underlying', symbol),
            expiry_date=pos_data.get('expiry_date', self._get_next_expiry(symbol)),
            strike_price=Decimal(str(pos_data['strike_price'])) if pos_data.get('strike_price') else None,
            option_type=OptionType(pos_data['option_type'].lower()) if pos_data.get('option_type') else None,
            current_price=Decimal(str(pos_data.get('current_price', market_data.get('price', 1000)))),
            delta=Decimal(str(pos_data.get('delta', market_data.get('delta', 0.5)))) if contract_type in [ContractType.OPTION_CALL, ContractType.OPTION_PUT] else None,
            lot_size=pos_data.get('lot_size', market_data.get('lot_size', 1)),
            commodity_code=pos_data.get('underlying', symbol)
        )
    
    async def _create_position_from_broker_data(self, broker_pos) -> PortfolioPosition:
        """Create position from broker position data."""
        symbol = broker_pos.symbol
        contract_type = self._determine_contract_type(symbol)
        
        # Get market data
        market_data = await self._get_market_data(symbol)
        
        return PortfolioPosition(
            symbol=symbol,
            contract_type=contract_type,
            quantity=broker_pos.quantity,
            underlying_symbol=symbol,  # Simplified - would need better logic
            expiry_date=self._get_next_expiry(symbol),
            current_price=Decimal(str(broker_pos.average_price)),
            lot_size=market_data.get('lot_size', 1),
            commodity_code=symbol
        )
    
    async def _create_position_from_trade(self, trade: Dict[str, Any]) -> PortfolioPosition:
        """Create position from trade dictionary."""
        return await self._create_position_from_data(trade)
    
    def _determine_contract_type(self, 
                                symbol: str, 
                                strike_price: Optional[float] = None,
                                option_type: Optional[str] = None) -> ContractType:
        """Determine contract type from symbol and parameters."""
        if strike_price is not None:
            if option_type and option_type.lower() == 'put':
                return ContractType.OPTION_PUT
            else:
                return ContractType.OPTION_CALL
        elif 'FUT' in symbol.upper():
            return ContractType.FUTURE
        else:
            return ContractType.PHYSICAL
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data for a symbol."""
        # Check cache
        if symbol in self._market_data_cache:
            cache_time = self._last_cache_update.get(symbol, datetime.min)
            if (datetime.now() - cache_time).seconds < self._cache_ttl:
                return self._market_data_cache[symbol]
        
        try:
            # Try to get from broker if available
            if self.broker_manager:
                # In practice, would fetch real market data
                pass
        except Exception as e:
            self.logger.warning(f"Failed to fetch market data for {symbol}: {str(e)}")
        
        # Default market data
        market_data = {
            'price': 1000.0,
            'delta': 0.5,
            'lot_size': self._get_default_lot_size(symbol)
        }
        
        # Cache the data
        self._market_data_cache[symbol] = market_data
        self._last_cache_update[symbol] = datetime.now()
        
        return market_data
    
    def _get_default_lot_size(self, symbol: str) -> int:
        """Get default lot size for a symbol."""
        lot_sizes = {
            'NIFTY': 50,
            'BANKNIFTY': 25,
            'RELIANCE': 250,
            'TCS': 150,
            'INFY': 300
        }
        
        # Check if symbol contains any of the known symbols
        for key, size in lot_sizes.items():
            if key in symbol.upper():
                return size
        
        return 1  # Default
    
    def _get_next_expiry(self, symbol: str) -> date:
        """Get next expiry date for a symbol."""
        # Simplified implementation - would use actual expiry calendar
        today = date.today()
        
        # For options/futures, use next Thursday
        days_ahead = 3 - today.weekday()  # Thursday is weekday 3
        if days_ahead <= 0:
            days_ahead += 7
        
        return today + timedelta(days=days_ahead)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._margin_cache:
            return False
        
        last_update = self._last_cache_update.get(cache_key, datetime.min)
        return (datetime.now() - last_update).seconds < self._cache_ttl
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._market_data_cache.clear()
        self._margin_cache.clear()
        self._last_cache_update.clear()
        self.logger.info("Cleared margin engine cache")
    
    def add_span_parameters(self, commodity_code: str, parameters: SPANParameters) -> None:
        """Add custom SPAN parameters."""
        self.span_calculator.add_span_parameters(commodity_code, parameters)
        self.clear_cache()  # Clear cache when parameters change
    
    def get_supported_commodities(self) -> List[str]:
        """Get list of supported commodity codes."""
        return self.span_calculator.get_supported_commodities()
    
    async def get_margin_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """Get margin calculation summary for a portfolio."""
        try:
            margin_result = await self.calculate_portfolio_margin(portfolio_id)
            
            return {
                'portfolio_id': portfolio_id,
                'total_margin': float(margin_result.total_margin),
                'margin_breakdown': {
                    'scan_risk': float(margin_result.scan_risk),
                    'inter_month_spread_charge': float(margin_result.inter_month_spread_charge),
                    'inter_commodity_spread_credit': float(margin_result.inter_commodity_spread_credit),
                    'spot_month_charge': float(margin_result.spot_month_charge),
                    'delivery_charge': float(margin_result.delivery_charge)
                },
                'position_count': len(margin_result.position_margins),
                'spread_credits': len(margin_result.spread_credits),
                'margin_type': margin_result.margin_type.value,
                'currency': margin_result.currency,
                'calculation_time': margin_result.calculation_timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get margin summary: {str(e)}")
            raise
