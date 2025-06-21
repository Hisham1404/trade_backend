"""
Option Chain Database Models

This module contains database models for storing option chain data,
analytics, and market configuration for derivatives analysis.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, ForeignKey, 
    Numeric, JSON, Enum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, time
from decimal import Decimal
import enum

from app.database.connection import Base

# Enums for option types and statuses
class OptionType(enum.Enum):
    CALL = "CE"  # Call Option (NSE uses CE)
    PUT = "PE"   # Put Option (NSE uses PE)

class ExpiryType(enum.Enum):
    WEEKLY = "weekly"
    MONTHLY = "monthly" 
    QUARTERLY = "quarterly"

class MarketStatus(enum.Enum):
    PRE_MARKET = "pre_market"
    OPEN = "open"
    CLOSED = "closed"
    POST_MARKET = "post_market"
    HOLIDAY = "holiday"

class AlertSeverity(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Core Option Contract Model
class OptionContract(Base):
    """Individual option contract data"""
    __tablename__ = "option_contracts"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Asset relationship
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    
    # Contract identification
    symbol = Column(String, nullable=False, index=True)  # e.g., NIFTY
    strike_price = Column(Numeric(10, 2), nullable=False)
    expiry_date = Column(DateTime(timezone=True), nullable=False)
    option_type = Column(Enum(OptionType), nullable=False)
    expiry_type = Column(Enum(ExpiryType), nullable=False)
    
    # Contract specifications
    lot_size = Column(Integer, nullable=False, default=50)  # NSE lot size
    tick_size = Column(Numeric(8, 2), default=0.05)  # Minimum price movement
    
    # Market data (current snapshot)
    last_price = Column(Numeric(10, 2), nullable=True)
    bid_price = Column(Numeric(10, 2), nullable=True)
    ask_price = Column(Numeric(10, 2), nullable=True)
    volume = Column(Integer, default=0)
    open_interest = Column(Integer, default=0)
    
    # Greeks
    delta = Column(Numeric(8, 4), nullable=True)
    gamma = Column(Numeric(8, 4), nullable=True)
    theta = Column(Numeric(8, 4), nullable=True)
    vega = Column(Numeric(8, 4), nullable=True)
    rho = Column(Numeric(8, 4), nullable=True)
    implied_volatility = Column(Numeric(8, 4), nullable=True)
    
    # Price change metrics
    change = Column(Numeric(10, 2), default=0.00)
    pct_change = Column(Numeric(8, 4), default=0.0000)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_updated = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    asset = relationship("Asset")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('symbol', 'strike_price', 'expiry_date', 'option_type', 
                        name='_option_contract_unique'),
        Index('idx_option_contracts_symbol_expiry', 'symbol', 'expiry_date'),
        Index('idx_option_contracts_strike_type', 'strike_price', 'option_type'),
    )
    
    def __repr__(self):
        return f"<OptionContract(symbol={self.symbol}, strike={self.strike_price}, " \
               f"expiry={self.expiry_date.date()}, type={self.option_type.value})>"

# Option Chain Snapshot Model
class OptionChainSnapshot(Base):
    """Complete option chain snapshot at a specific time"""
    __tablename__ = "option_chain_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Asset relationship
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    
    # Snapshot metadata
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    market_status = Column(Enum(MarketStatus), nullable=False)
    
    # Underlying asset data
    underlying_price = Column(Numeric(10, 2), nullable=True)
    underlying_change = Column(Numeric(10, 2), nullable=True)
    underlying_pct_change = Column(Numeric(8, 4), nullable=True)
    
    # Aggregate metrics
    total_call_oi = Column(Integer, default=0)
    total_put_oi = Column(Integer, default=0)
    total_call_volume = Column(Integer, default=0)
    total_put_volume = Column(Integer, default=0)
    
    # Key ratios and levels
    put_call_ratio = Column(Numeric(8, 4), nullable=True)  # PCR (OI based)
    pcr_volume = Column(Numeric(8, 4), nullable=True)      # PCR (Volume based)
    max_pain_level = Column(Numeric(10, 2), nullable=True)
    
    # Support and resistance levels (top 3 each)
    resistance_levels = Column(JSON, nullable=True)  # [strike1, strike2, strike3]
    support_levels = Column(JSON, nullable=True)     # [strike1, strike2, strike3]
    
    # Volatility metrics
    avg_iv_calls = Column(Numeric(8, 4), nullable=True)
    avg_iv_puts = Column(Numeric(8, 4), nullable=True)
    iv_rank = Column(Numeric(5, 2), nullable=True)  # 0-100 percentile
    
    # Raw data storage
    raw_data = Column(JSON, nullable=True)  # Complete option chain JSON
    
    # Data quality flags
    is_complete = Column(Boolean, default=True)
    data_source = Column(String, default="NSE")
    fetch_duration_ms = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    asset = relationship("Asset")
    # Note: Option contracts exist independently; snapshot stores complete chain data in JSON
    alerts = relationship("OptionAlert", back_populates="snapshot")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_snapshots_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_snapshots_asset_timestamp', 'asset_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<OptionChainSnapshot(symbol={self.symbol}, " \
               f"timestamp={self.timestamp}, pcr={self.put_call_ratio})>"

# Option Analytics Model
class OptionAnalytics(Base):
    """Calculated analytics and derived metrics"""
    __tablename__ = "option_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Reference to snapshot
    snapshot_id = Column(Integer, ForeignKey("option_chain_snapshots.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    
    # Time period analysis
    analysis_period = Column(String, nullable=False)  # "5min", "1hour", "1day"
    calculated_at = Column(DateTime(timezone=True), nullable=False)
    
    # Trend analysis
    pcr_trend = Column(String, nullable=True)  # "bullish", "bearish", "neutral"
    pcr_change_pct = Column(Numeric(8, 4), nullable=True)
    oi_trend = Column(String, nullable=True)
    volume_trend = Column(String, nullable=True)
    
    # Unusual activity detection
    unusual_call_activity = Column(JSON, nullable=True)  # [{strike, reason, score}]
    unusual_put_activity = Column(JSON, nullable=True)   # [{strike, reason, score}]
    
    # Breakout analysis
    potential_breakout_up = Column(Numeric(10, 2), nullable=True)    # Resistance level
    potential_breakout_down = Column(Numeric(10, 2), nullable=True)  # Support level
    breakout_probability = Column(Numeric(5, 2), nullable=True)      # 0-100%
    
    # Volatility analysis
    iv_percentile = Column(Numeric(5, 2), nullable=True)  # IV rank
    iv_trend = Column(String, nullable=True)  # "expanding", "contracting", "stable"
    
    # Institutional activity indicators
    smart_money_flow = Column(String, nullable=True)  # "call_buying", "put_buying", "mixed"
    large_trade_count = Column(Integer, default=0)
    
    # Confidence scores
    analysis_confidence = Column(Numeric(5, 2), default=0.50)  # 0.0-1.0
    data_quality_score = Column(Numeric(5, 2), default=1.00)   # 0.0-1.0
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    snapshot = relationship("OptionChainSnapshot")
    asset = relationship("Asset")
    
    # Indexes
    __table_args__ = (
        Index('idx_analytics_asset_period', 'asset_id', 'analysis_period'),
        Index('idx_analytics_calculated_at', 'calculated_at'),
    )
    
    def __repr__(self):
        return f"<OptionAnalytics(asset_id={self.asset_id}, " \
               f"period={self.analysis_period}, confidence={self.analysis_confidence})>"

# Option Alerts Model
class OptionAlert(Base):
    """Alerts generated from option chain analysis"""
    __tablename__ = "option_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Reference relationships
    snapshot_id = Column(Integer, ForeignKey("option_chain_snapshots.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Can be system-wide
    
    # Alert details
    alert_type = Column(String, nullable=False)  # "unusual_activity", "pcr_spike", "breakout"
    severity = Column(Enum(AlertSeverity), nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    
    # Alert specifics
    trigger_condition = Column(JSON, nullable=False)  # What caused the alert
    current_values = Column(JSON, nullable=False)     # Current metrics
    historical_context = Column(JSON, nullable=True)  # Comparison with history
    
    # Action recommendations
    suggested_action = Column(String, nullable=True)  # "monitor", "investigate", "trade"
    risk_level = Column(String, nullable=True)        # "low", "medium", "high"
    
    # Alert metadata
    confidence_score = Column(Numeric(5, 2), nullable=False)  # 0.0-1.0
    priority_score = Column(Integer, default=5)               # 1-10
    is_automated = Column(Boolean, default=True)
    
    # Status tracking
    is_active = Column(Boolean, default=True)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    snapshot = relationship("OptionChainSnapshot", back_populates="alerts")
    asset = relationship("Asset")
    user = relationship("User", foreign_keys=[user_id])
    acknowledged_by_user = relationship("User", foreign_keys=[acknowledged_by])
    
    # Indexes
    __table_args__ = (
        Index('idx_option_alerts_asset_type', 'asset_id', 'alert_type'),
        Index('idx_option_alerts_severity_active', 'severity', 'is_active'),
        Index('idx_option_alerts_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<OptionAlert(type={self.alert_type}, severity={self.severity.value}, " \
               f"asset_id={self.asset_id})>"

# Market Hours Configuration Model
class MarketHoursConfig(Base):
    """Configuration for market hours and scheduling"""
    __tablename__ = "market_hours_config"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Market identification
    exchange = Column(String, nullable=False, unique=True)  # "NSE", "BSE"
    market_type = Column(String, nullable=False)            # "equity", "derivatives"
    
    # Regular market hours
    market_open_time = Column(String, nullable=False)       # "09:15:00"
    market_close_time = Column(String, nullable=False)      # "15:30:00"
    
    # Pre/Post market hours
    pre_market_start = Column(String, nullable=True)        # "09:00:00"
    pre_market_end = Column(String, nullable=True)          # "09:15:00"
    post_market_start = Column(String, nullable=True)       # "15:30:00"
    post_market_end = Column(String, nullable=True)         # "16:00:00"
    
    # Weekly schedule
    trading_days = Column(JSON, nullable=False)             # [1,2,3,4,5] (Mon-Fri)
    
    # Holidays and special days
    holiday_calendar = Column(JSON, nullable=True)          # ["2024-01-26", "2024-08-15"]
    early_close_days = Column(JSON, nullable=True)          # {"2024-12-31": "13:00:00"}
    
    # Data fetching schedule
    fetch_interval_minutes = Column(Integer, default=5)     # How often to fetch data
    fetch_during_pre_market = Column(Boolean, default=False)
    fetch_during_post_market = Column(Boolean, default=False)
    
    # Timezone configuration
    timezone = Column(String, default="Asia/Kolkata")
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<MarketHoursConfig(exchange={self.exchange}, " \
               f"open={self.market_open_time}, close={self.market_close_time})>"

# Option Trading Statistics Model
class OptionTradingStats(Base):
    """Daily/hourly trading statistics for performance tracking"""
    __tablename__ = "option_trading_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Time period
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    hour = Column(Integer, nullable=True)  # For hourly stats, null for daily
    
    # Asset reference
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    symbol = Column(String, nullable=False)
    
    # Volume and OI statistics
    total_call_volume = Column(Integer, default=0)
    total_put_volume = Column(Integer, default=0)
    total_call_oi = Column(Integer, default=0)
    total_put_oi = Column(Integer, default=0)
    
    # PCR statistics
    avg_pcr_oi = Column(Numeric(8, 4), nullable=True)
    avg_pcr_volume = Column(Numeric(8, 4), nullable=True)
    min_pcr = Column(Numeric(8, 4), nullable=True)
    max_pcr = Column(Numeric(8, 4), nullable=True)
    
    # Volatility statistics
    avg_iv = Column(Numeric(8, 4), nullable=True)
    max_iv = Column(Numeric(8, 4), nullable=True)
    min_iv = Column(Numeric(8, 4), nullable=True)
    
    # Price movement
    underlying_high = Column(Numeric(10, 2), nullable=True)
    underlying_low = Column(Numeric(10, 2), nullable=True)
    underlying_close = Column(Numeric(10, 2), nullable=True)
    
    # Activity metrics
    alerts_generated = Column(Integer, default=0)
    unusual_activity_count = Column(Integer, default=0)
    data_fetch_count = Column(Integer, default=0)
    successful_fetches = Column(Integer, default=0)
    
    # Performance metrics
    avg_fetch_time_ms = Column(Integer, nullable=True)
    data_quality_score = Column(Numeric(5, 2), default=1.00)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    asset = relationship("Asset")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('date', 'hour', 'asset_id', name='_option_stats_unique'),
        Index('idx_option_stats_date_symbol', 'date', 'symbol'),
        Index('idx_option_stats_asset_date', 'asset_id', 'date'),
    )
    
    def __repr__(self):
        period = f"hour {self.hour}" if self.hour else "daily"
        return f"<OptionTradingStats(symbol={self.symbol}, date={self.date.date()}, {period})>" 