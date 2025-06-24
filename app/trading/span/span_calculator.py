"""
SPAN Margin Calculator Implementation.

This module implements the core SPAN (Standard Portfolio Analysis of Risk)
margin calculation algorithm as specified by various exchanges.
"""

import logging
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, timedelta

from .span_models import (
    Portfolio, PortfolioPosition, SPANParameters, MarginResult,
    RiskScenario, SpreadCredit, ContractType, OptionType, MarginType,
    DEFAULT_SPAN_PARAMETERS
)

logger = logging.getLogger(__name__)


class SPANCalculator:
    """
    Core SPAN margin calculation engine.
    
    Implements the Standard Portfolio Analysis of Risk methodology
    for calculating portfolio margin requirements.
    """
    
    def __init__(self, span_parameters: Optional[Dict[str, SPANParameters]] = None):
        """
        Initialize SPAN calculator.
        
        Args:
            span_parameters: Dictionary mapping commodity codes to SPAN parameters
        """
        self.span_parameters = span_parameters or DEFAULT_SPAN_PARAMETERS
        self.logger = logging.getLogger(__name__)
    
    def calculate_portfolio_margin(self, 
                                 portfolio: Portfolio,
                                 margin_type: MarginType = MarginType.INITIAL) -> MarginResult:
        """
        Calculate SPAN margin for an entire portfolio.
        
        Args:
            portfolio: Portfolio containing positions
            margin_type: Type of margin calculation (initial/maintenance)
            
        Returns:
            MarginResult with detailed margin breakdown
        """
        try:
            self.logger.info(f"Starting SPAN calculation for portfolio {portfolio.portfolio_id}")
            
            # Group positions by commodity/underlying
            commodity_groups = self._group_positions_by_commodity(portfolio)
            
            total_scan_risk = Decimal('0')
            total_inter_month_charge = Decimal('0')
            total_inter_commodity_credit = Decimal('0')
            total_spot_month_charge = Decimal('0')
            total_delivery_charge = Decimal('0')
            
            position_margins = {}
            all_spread_credits = []
            
            # Calculate margin for each commodity group
            for commodity_code, positions in commodity_groups.items():
                self.logger.debug(f"Calculating margin for commodity {commodity_code}")
                
                # Get SPAN parameters for this commodity
                span_params = self._get_span_parameters(commodity_code)
                
                # Calculate scan risk for this commodity group
                commodity_scan_risk = self._calculate_scan_risk(positions, span_params)
                total_scan_risk += commodity_scan_risk
                
                # Calculate inter-month spread charges
                inter_month_charge = self._calculate_inter_month_spread_charge(
                    positions, span_params
                )
                total_inter_month_charge += inter_month_charge
                
                # Calculate spot month charges
                spot_month_charge = self._calculate_spot_month_charge(
                    positions, span_params
                )
                total_spot_month_charge += spot_month_charge
                
                # Calculate delivery charges
                delivery_charge = self._calculate_delivery_charge(
                    positions, span_params
                )
                total_delivery_charge += delivery_charge
                
                # Store individual position margins
                for position in positions:
                    position_margins[position.symbol] = commodity_scan_risk / len(positions)
            
            # Calculate inter-commodity spread credits
            total_inter_commodity_credit, spread_credits = self._calculate_inter_commodity_spreads(
                commodity_groups
            )
            all_spread_credits.extend(spread_credits)
            
            # Calculate total margin
            total_margin = (
                total_scan_risk +
                total_inter_month_charge +
                total_spot_month_charge +
                total_delivery_charge -
                total_inter_commodity_credit
            )
            
            # Ensure minimum margin
            total_margin = max(total_margin, Decimal('0'))
            
            result = MarginResult(
                total_margin=total_margin,
                scan_risk=total_scan_risk,
                inter_month_spread_charge=total_inter_month_charge,
                inter_commodity_spread_credit=total_inter_commodity_credit,
                spot_month_charge=total_spot_month_charge,
                delivery_charge=total_delivery_charge,
                spread_credits=all_spread_credits,
                position_margins=position_margins,
                margin_type=margin_type,
                currency=portfolio.currency
            )
            
            self.logger.info(f"SPAN calculation completed. Total margin: {total_margin}")
            return result
            
        except Exception as e:
            self.logger.error(f"SPAN calculation failed: {str(e)}")
            raise
    
    def _group_positions_by_commodity(self, portfolio: Portfolio) -> Dict[str, List[PortfolioPosition]]:
        """Group positions by commodity code."""
        groups = {}
        
        for position in portfolio.positions:
            commodity_code = position.commodity_code or position.underlying_symbol
            if commodity_code not in groups:
                groups[commodity_code] = []
            groups[commodity_code].append(position)
        
        return groups
    
    def _get_span_parameters(self, commodity_code: str) -> SPANParameters:
        """Get SPAN parameters for a commodity."""
        if commodity_code in self.span_parameters:
            return self.span_parameters[commodity_code]
        
        # Use default parameters if specific ones not found
        self.logger.warning(f"No SPAN parameters found for {commodity_code}, using default")
        return SPANParameters(
            commodity_code=commodity_code,
            product_group="DEFAULT",
            price_scan_range=Decimal('5.0'),
            volatility_scan_range=Decimal('10.0'),
            inter_month_spread_rate=Decimal('50.0'),
            spot_month_charge_rate=Decimal('3.0'),
            delivery_charge_rate=Decimal('0.0')
        )
    
    def _calculate_scan_risk(self, 
                           positions: List[PortfolioPosition], 
                           span_params: SPANParameters) -> Decimal:
        """
        Calculate scan risk using SPAN methodology.
        
        This is the core SPAN calculation that evaluates portfolio
        risk under various market scenarios.
        """
        if not positions:
            return Decimal('0')
        
        # Generate risk scenarios
        risk_scenarios = span_params.get_risk_scenarios()
        
        scenario_losses = []
        
        # Calculate loss/gain for each scenario
        for scenario in risk_scenarios:
            scenario_pnl = Decimal('0')
            
            for position in positions:
                # Calculate position PnL under this scenario
                position_pnl = self._calculate_position_scenario_pnl(
                    position, scenario, span_params
                )
                scenario_pnl += position_pnl
            
            # SPAN takes the worst loss (maximum loss)
            scenario_losses.append(-scenario_pnl if scenario_pnl < 0 else Decimal('0'))
        
        # Scan risk is the maximum loss across all scenarios
        scan_risk = max(scenario_losses) if scenario_losses else Decimal('0')
        
        self.logger.debug(f"Calculated scan risk: {scan_risk}")
        return scan_risk
    
    def _calculate_position_scenario_pnl(self, 
                                       position: PortfolioPosition,
                                       scenario: RiskScenario,
                                       span_params: SPANParameters) -> Decimal:
        """Calculate PnL for a position under a specific scenario."""
        
        # Apply price scenario
        new_price = scenario.apply_to_price(position.current_price)
        price_change = new_price - position.current_price
        
        if position.contract_type == ContractType.FUTURE:
            # For futures, PnL is linear with price change
            pnl = price_change * Decimal(str(position.quantity)) * Decimal(str(position.lot_size))
            
        elif position.is_option():
            # For options, use Black-Scholes approximation with Greeks
            pnl = self._calculate_option_scenario_pnl(
                position, price_change, scenario, span_params
            )
            
        else:
            # For physical/cash positions
            pnl = price_change * Decimal(str(position.quantity)) * Decimal(str(position.lot_size))
        
        return pnl
    
    def _calculate_option_scenario_pnl(self, 
                                     position: PortfolioPosition,
                                     price_change: Decimal,
                                     scenario: RiskScenario,
                                     span_params: SPANParameters) -> Decimal:
        """Calculate option PnL using simplified Greeks approximation."""
        
        # Use delta for linear price sensitivity
        delta = position.delta or self._estimate_option_delta(position)
        
        # Linear approximation: PnL ≈ Delta * Price_Change * Quantity * Lot_Size
        linear_pnl = (
            delta * price_change * 
            Decimal(str(position.quantity)) * 
            Decimal(str(position.lot_size))
        )
        
        # Apply time decay effect (theta)
        time_decay_effect = Decimal('0')
        if scenario.time_decay > 0:
            # Simplified time decay: assume 1% value loss per day for ATM options
            if abs(position.current_price - (position.strike_price or Decimal('0'))) < position.current_price * Decimal('0.05'):
                time_decay_effect = (
                    position.current_price * 
                    Decimal(str(scenario.time_decay)) * 
                    Decimal('0.01') *
                    Decimal(str(position.quantity)) * 
                    Decimal(str(position.lot_size))
                )
        
        total_pnl = linear_pnl - time_decay_effect
        
        return total_pnl
    
    def _estimate_option_delta(self, position: PortfolioPosition) -> Decimal:
        """Estimate option delta using simplified approximation."""
        if not position.strike_price:
            return Decimal('0.5')  # Default neutral delta
        
        # Moneyness: S/K for calls, K/S for puts
        if position.option_type == OptionType.CALL:
            moneyness = position.current_price / position.strike_price
            # Simplified delta estimation for calls
            if moneyness > Decimal('1.1'):
                return Decimal('0.8')  # Deep ITM
            elif moneyness > Decimal('0.9'):
                return Decimal('0.5')  # ATM
            else:
                return Decimal('0.2')  # OTM
        else:  # PUT
            moneyness = position.strike_price / position.current_price
            # Simplified delta estimation for puts (negative)
            if moneyness > Decimal('1.1'):
                return Decimal('-0.8')  # Deep ITM
            elif moneyness > Decimal('0.9'):
                return Decimal('-0.5')  # ATM
            else:
                return Decimal('-0.2')  # OTM
    
    def _calculate_inter_month_spread_charge(self, 
                                           positions: List[PortfolioPosition],
                                           span_params: SPANParameters) -> Decimal:
        """Calculate inter-month spread charges."""
        
        # Group positions by expiry month
        expiry_groups = {}
        for position in positions:
            expiry_key = position.expiry_date.strftime('%Y-%m')
            if expiry_key not in expiry_groups:
                expiry_groups[expiry_key] = []
            expiry_groups[expiry_key].append(position)
        
        # If only one expiry month, no inter-month spread charge
        if len(expiry_groups) <= 1:
            return Decimal('0')
        
        # Calculate spread charge based on net positions across months
        total_charge = Decimal('0')
        expiry_months = sorted(expiry_groups.keys())
        
        for i in range(len(expiry_months) - 1):
            near_month = expiry_groups[expiry_months[i]]
            far_month = expiry_groups[expiry_months[i + 1]]
            
            # Calculate net positions for each month
            near_net = sum(Decimal(str(pos.quantity)) for pos in near_month)
            far_net = sum(Decimal(str(pos.quantity)) for pos in far_month)
            
            # Spread charge applies to the smaller of the two positions
            spread_quantity = min(abs(near_net), abs(far_net))
            
            if spread_quantity > 0:
                # Use average price for charge calculation
                avg_price = sum(pos.current_price for pos in near_month + far_month) / len(near_month + far_month)
                spread_charge = (
                    spread_quantity * avg_price * 
                    span_params.inter_month_spread_rate / Decimal('100')
                )
                total_charge += spread_charge
        
        return total_charge
    
    def _calculate_spot_month_charge(self, 
                                   positions: List[PortfolioPosition],
                                   span_params: SPANParameters) -> Decimal:
        """Calculate spot month charge for positions nearing expiry."""
        
        today = date.today()
        spot_month_charge = Decimal('0')
        
        for position in positions:
            days_to_expiry = (position.expiry_date - today).days
            
            # Apply spot month charge if position expires within 30 days
            if days_to_expiry <= 30:
                position_value = position.get_position_value()
                charge = position_value * span_params.spot_month_charge_rate / Decimal('100')
                spot_month_charge += charge
        
        return spot_month_charge
    
    def _calculate_delivery_charge(self, 
                                 positions: List[PortfolioPosition],
                                 span_params: SPANParameters) -> Decimal:
        """Calculate delivery charge for positions entering delivery period."""
        
        today = date.today()
        delivery_charge = Decimal('0')
        
        for position in positions:
            days_to_expiry = (position.expiry_date - today).days
            
            # Apply delivery charge if position expires within 5 days
            if days_to_expiry <= 5:
                position_value = position.get_position_value()
                charge = position_value * span_params.delivery_charge_rate / Decimal('100')
                delivery_charge += charge
        
        return delivery_charge
    
    def _calculate_inter_commodity_spreads(self, 
                                         commodity_groups: Dict[str, List[PortfolioPosition]]
                                         ) -> Tuple[Decimal, List[SpreadCredit]]:
        """Calculate inter-commodity spread credits."""
        
        # Simplified implementation - in practice would use complex spread matrices
        total_credit = Decimal('0')
        spread_credits = []
        
        commodity_codes = list(commodity_groups.keys())
        
        # Check for common spread relationships
        for i in range(len(commodity_codes)):
            for j in range(i + 1, len(commodity_codes)):
                commodity1 = commodity_codes[i]
                commodity2 = commodity_codes[j]
                
                # Example: NIFTY and BANKNIFTY spread credit
                if {commodity1, commodity2} == {"NIFTY", "BANKNIFTY"}:
                    spread_credit = SpreadCredit(
                        spread_type="inter_commodity",
                        credit_rate=Decimal('25.0'),  # 25% credit
                        max_credit_amount=Decimal('100000'),  # Max 1 lakh credit
                        leg1_position=commodity1,
                        leg2_position=commodity2
                    )
                    
                    # Calculate credit based on position values
                    leg1_value = sum(pos.get_position_value() for pos in commodity_groups[commodity1])
                    leg2_value = sum(pos.get_position_value() for pos in commodity_groups[commodity2])
                    
                    credit_amount = spread_credit.calculate_credit(leg1_value, leg2_value)
                    total_credit += credit_amount
                    spread_credits.append(spread_credit)
        
        return total_credit, spread_credits
    
    def add_span_parameters(self, commodity_code: str, parameters: SPANParameters) -> None:
        """Add or update SPAN parameters for a commodity."""
        self.span_parameters[commodity_code] = parameters
        self.logger.info(f"Added SPAN parameters for {commodity_code}")
    
    def get_supported_commodities(self) -> List[str]:
        """Get list of supported commodity codes."""
        return list(self.span_parameters.keys())
    
    def validate_portfolio(self, portfolio: Portfolio) -> List[str]:
        """Validate portfolio for SPAN calculation."""
        warnings = []
        
        if not portfolio.positions:
            warnings.append("Portfolio is empty")
            return warnings
        
        for position in portfolio.positions:
            # Check for required fields
            if not position.symbol:
                warnings.append(f"Position missing symbol")
            
            if position.current_price <= 0:
                warnings.append(f"Invalid current price for {position.symbol}")
            
            if position.quantity == 0:
                warnings.append(f"Zero quantity for {position.symbol}")
            
            # Check for option-specific validations
            if position.is_option():
                if not position.strike_price or position.strike_price <= 0:
                    warnings.append(f"Invalid strike price for option {position.symbol}")
            
            # Check for commodity parameters
            commodity_code = position.commodity_code or position.underlying_symbol
            if commodity_code not in self.span_parameters:
                warnings.append(f"No SPAN parameters found for {commodity_code}")
        
        return warnings
