"""
SPAN Margin Calculation Models and Data Structures.

This module contains all the data models and structures used in the
Standard Portfolio Analysis of Risk (SPAN) margin calculation system.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any


class ContractType(Enum):
    """Contract types supported by SPAN."""
    FUTURE = "future"
    OPTION_CALL = "option_call"
    OPTION_PUT = "option_put"
    PHYSICAL = "physical"
    CASH = "cash"


class OptionType(Enum):
    """Option types."""
    CALL = "call"
    PUT = "put"


class MarginType(Enum):
    """Types of margin calculations."""
    INITIAL = "initial"
    MAINTENANCE = "maintenance"


@dataclass
class PortfolioPosition:
    """Represents a position in a portfolio for SPAN calculation."""
    symbol: str
    contract_type: ContractType
    quantity: int
    underlying_symbol: str
    expiry_date: date
    strike_price: Optional[Decimal] = None
    option_type: Optional[OptionType] = None
    
    # Market data
    current_price: Decimal = Decimal('0')
    delta: Optional[Decimal] = None
    
    # Contract specifications
    lot_size: int = 1
    tick_size: Decimal = Decimal('0.01')
    
    # Additional attributes
    commodity_code: str = ""
    product_group: str = ""
    
    def get_position_value(self) -> Decimal:
        """Calculate the market value of the position."""
        return self.current_price * Decimal(str(self.quantity)) * Decimal(str(self.lot_size))
    
    def is_option(self) -> bool:
        """Check if this position is an option."""
        return self.contract_type in [ContractType.OPTION_CALL, ContractType.OPTION_PUT]


@dataclass
class RiskScenario:
    """Represents a risk scenario for SPAN calculation."""
    scenario_id: int
    price_scan_range: Decimal  # Price change as percentage
    volatility_scan_range: Decimal  # Volatility change as percentage
    time_decay: int  # Days of time decay
    
    def apply_to_price(self, base_price: Decimal) -> Decimal:
        """Apply price scenario to a base price."""
        price_change = base_price * (self.price_scan_range / Decimal('100'))
        return base_price + price_change


@dataclass
class SpreadCredit:
    """Represents a spread credit in SPAN calculation."""
    spread_type: str
    credit_rate: Decimal
    max_credit_amount: Decimal
    leg1_position: str
    leg2_position: str
    credit_amount: Decimal = Decimal('0')
    
    def calculate_credit(self, leg1_margin: Decimal, leg2_margin: Decimal) -> Decimal:
        """Calculate the actual spread credit amount."""
        base_credit = min(leg1_margin, leg2_margin) * (self.credit_rate / Decimal('100'))
        self.credit_amount = min(base_credit, self.max_credit_amount)
        return self.credit_amount


@dataclass
class MarginResult:
    """Result of SPAN margin calculation."""
    total_margin: Decimal
    scan_risk: Decimal
    inter_month_spread_charge: Decimal
    inter_commodity_spread_credit: Decimal
    spot_month_charge: Decimal
    delivery_charge: Decimal
    
    # Detailed breakdown
    scenario_risks: List[Decimal] = field(default_factory=list)
    spread_credits: List[SpreadCredit] = field(default_factory=list)
    position_margins: Dict[str, Decimal] = field(default_factory=dict)
    
    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    margin_type: MarginType = MarginType.INITIAL
    currency: str = "INR"


@dataclass
class SPANParameters:
    """SPAN parameters for a commodity or product group."""
    
    # Basic identifiers
    commodity_code: str
    product_group: str
    currency: str = "INR"
    
    # Scanning parameters
    price_scan_range: Decimal = Decimal('0')  # Percentage
    volatility_scan_range: Decimal = Decimal('0')  # Percentage
    price_scan_scenarios: int = 16  # Standard SPAN uses 16 scenarios
    
    # Time parameters
    time_decay_scenarios: List[int] = field(default_factory=lambda: [0, 1])  # Days
    
    # Margin rates
    inter_month_spread_rate: Decimal = Decimal('0')
    spot_month_charge_rate: Decimal = Decimal('0')
    delivery_charge_rate: Decimal = Decimal('0')
    
    # Risk-free rate
    risk_free_rate: Decimal = Decimal('6.0')  # Annual percentage
    
    # Contract specifications
    contract_size: int = 1
    tick_size: Decimal = Decimal('0.01')
    tick_value: Decimal = Decimal('1')
    
    # Exchange-specific parameters
    exchange_code: str = "NSE"
    
    # File metadata
    file_date: Optional[date] = None
    file_version: str = "1.0"
    
    def get_risk_scenarios(self) -> List[RiskScenario]:
        """Generate standard SPAN risk scenarios."""
        scenarios = []
        scenario_id = 1
        
        # Price scenarios (up and down moves)
        price_moves = [
            self.price_scan_range,
            -self.price_scan_range,
            self.price_scan_range / 3,
            -self.price_scan_range / 3
        ]
        
        # Volatility scenarios
        vol_moves = [
            self.volatility_scan_range,
            -self.volatility_scan_range,
            Decimal('0')
        ]
        
        # Time decay scenarios
        for time_decay in self.time_decay_scenarios:
            for price_move in price_moves:
                for vol_move in vol_moves:
                    scenarios.append(RiskScenario(
                        scenario_id=scenario_id,
                        price_scan_range=price_move,
                        volatility_scan_range=vol_move,
                        time_decay=time_decay
                    ))
                    scenario_id += 1
        
        return scenarios


@dataclass
class Portfolio:
    """Represents a portfolio for SPAN calculation."""
    portfolio_id: str
    positions: List[PortfolioPosition] = field(default_factory=list)
    portfolio_date: date = field(default_factory=date.today)
    currency: str = "INR"
    account_id: str = ""
    
    def add_position(self, position: PortfolioPosition) -> None:
        """Add a position to the portfolio."""
        self.positions.append(position)
    
    def get_positions_by_underlying(self, underlying: str) -> List[PortfolioPosition]:
        """Get all positions for a specific underlying."""
        return [pos for pos in self.positions if pos.underlying_symbol == underlying]
    
    def get_net_position_value(self) -> Decimal:
        """Calculate the total net position value."""
        return sum(pos.get_position_value() for pos in self.positions)
    
    def get_position_count(self) -> int:
        """Get the total number of positions."""
        return len(self.positions)


# Default SPAN parameters for common Indian market instruments
DEFAULT_SPAN_PARAMETERS = {
    "NIFTY": SPANParameters(
        commodity_code="NIFTY",
        product_group="INDEX_OPTIONS",
        price_scan_range=Decimal('3.5'),
        volatility_scan_range=Decimal('6.0'),
        inter_month_spread_rate=Decimal('50.0'),
        spot_month_charge_rate=Decimal('3.0'),
        delivery_charge_rate=Decimal('0.0'),
        exchange_code="NSE"
    ),
    "BANKNIFTY": SPANParameters(
        commodity_code="BANKNIFTY",
        product_group="INDEX_OPTIONS", 
        price_scan_range=Decimal('3.5'),
        volatility_scan_range=Decimal('6.0'),
        inter_month_spread_rate=Decimal('50.0'),
        spot_month_charge_rate=Decimal('3.0'),
        delivery_charge_rate=Decimal('0.0'),
        exchange_code="NSE"
    ),
    "RELIANCE": SPANParameters(
        commodity_code="RELIANCE",
        product_group="EQUITY_OPTIONS",
        price_scan_range=Decimal('7.5'),
        volatility_scan_range=Decimal('12.0'),
        inter_month_spread_rate=Decimal('75.0'),
        spot_month_charge_rate=Decimal('5.0'),
        delivery_charge_rate=Decimal('20.0'),
        exchange_code="NSE"
    )
}