"""
Participant Flow Tracking Database Models

This module contains database models for tracking market participant activity,
including FII/DII flows, retail behavior, and institutional trading patterns.
Designed for efficient querying of large time-series datasets with proper indexing.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, ForeignKey, 
    Numeric, JSON, Enum, Index, UniqueConstraint, BigInteger,
    CheckConstraint, event
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, date, timedelta
from decimal import Decimal
import enum

from app.database.connection import Base

# Enums for participant flow tracking
class ParticipantType(enum.Enum):
    """Types of market participants"""
    FII = "FII"           # Foreign Institutional Investors
    DII = "DII"           # Domestic Institutional Investors
    PROPRIETARY = "Pro"   # Proprietary Trading Desks
    RETAIL = "Retail"     # Retail Investors
    MUTUAL_FUND = "MF"    # Mutual Funds
    INSURANCE = "Insurance" # Insurance Companies
    CORPORATE = "Corporate" # Corporate Entities
    HNI = "HNI"           # High Net Worth Individuals

class MarketSegment(enum.Enum):
    """Market segments for tracking"""
    CASH = "cash"         # Cash/Equity segment
    FUTURES = "futures"   # Futures & Forwards
    OPTIONS = "options"   # Options
    CURRENCY = "currency" # Currency derivatives
    COMMODITY = "commodity" # Commodity derivatives

class FlowDirection(enum.Enum):
    """Direction of participant flow"""
    INFLOW = "inflow"     # Net buying
    OUTFLOW = "outflow"   # Net selling
    NEUTRAL = "neutral"   # Balanced

class ActivityType(enum.Enum):
    """Type of participant activity"""
    BUY = "buy"
    SELL = "sell"
    NET_POSITION = "net_position"
    TURNOVER = "turnover"

class DataSource(enum.Enum):
    """Source of participant data"""
    NSE = "NSE"
    BSE = "BSE"
    SEBI = "SEBI"
    RBI = "RBI"
    CDSL = "CDSL"
    NSDL = "NSDL"

class DataQuality(enum.Enum):
    """Quality level of data"""
    HIGH = "high"         # Complete, validated data
    MEDIUM = "medium"     # Minor gaps or estimates
    LOW = "low"           # Significant gaps or poor quality
    ESTIMATED = "estimated" # Calculated/estimated values

# Core Participant Flow Models

class ParticipantProfile(Base):
    """Master data for market participants"""
    __tablename__ = "participant_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Participant identification
    participant_type = Column(Enum(ParticipantType), nullable=False, index=True)
    participant_code = Column(String(50), nullable=True)  # Official code if available
    participant_name = Column(String(200), nullable=True) # Official name
    
    # Classification details
    category = Column(String(100), nullable=True)  # Sub-category within type
    registration_number = Column(String(100), nullable=True)
    regulatory_body = Column(String(50), nullable=True)  # SEBI, RBI, etc.
    
    # Geographic and operational info
    country_of_origin = Column(String(50), nullable=True)
    primary_exchange = Column(String(20), default="NSE")
    active_segments = Column(JSON, nullable=True)  # [cash, futures, options]
    
    # Profile metadata
    is_active = Column(Boolean, default=True)
    risk_category = Column(String(20), nullable=True)  # low, medium, high
    trading_style = Column(String(50), nullable=True)  # momentum, value, etc.
    
    # Data tracking
    first_activity_date = Column(DateTime(timezone=True), nullable=True)
    last_activity_date = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    activities = relationship("ParticipantActivity", back_populates="participant")
    flow_metrics = relationship("ParticipantFlowMetrics", back_populates="participant")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('participant_type', 'participant_code', 
                        name='_participant_unique'),
        Index('idx_participant_type_active', 'participant_type', 'is_active'),
    )
    
    def __repr__(self):
        return f"<ParticipantProfile(type={self.participant_type.value}, " \
               f"name={self.participant_name})>"

class ParticipantActivity(Base):
    """Daily participant trading activity data"""
    __tablename__ = "participant_activities"
    
    id = Column(BigInteger, primary_key=True, index=True)
    
    # Time and participant reference
    trade_date = Column(DateTime(timezone=True), nullable=False, index=True)
    participant_id = Column(Integer, ForeignKey("participant_profiles.id"), nullable=False)
    
    # Asset and market context
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=True)
    symbol = Column(String(50), nullable=True, index=True)
    market_segment = Column(Enum(MarketSegment), nullable=False, index=True)
    exchange = Column(String(20), default="NSE")
    
    # Trading activity metrics
    buy_value = Column(Numeric(15, 2), default=0.00)    # Total buy value (in crores)
    sell_value = Column(Numeric(15, 2), default=0.00)   # Total sell value (in crores)
    buy_quantity = Column(BigInteger, default=0)         # Total buy quantity
    sell_quantity = Column(BigInteger, default=0)        # Total sell quantity
    
    # Net positions and flows
    net_value = Column(Numeric(15, 2), default=0.00)     # buy_value - sell_value
    net_quantity = Column(BigInteger, default=0)          # buy_quantity - sell_quantity
    gross_turnover = Column(Numeric(15, 2), default=0.00) # buy_value + sell_value
    
    # Market share and participation
    market_share_pct = Column(Numeric(8, 4), nullable=True)  # % of total market
    participation_rate = Column(Numeric(8, 4), nullable=True) # Activity vs historical avg
    
    # Intraday statistics
    avg_trade_size = Column(Numeric(12, 2), nullable=True)
    total_trades = Column(Integer, default=0)
    peak_hour_activity = Column(String(10), nullable=True)  # "10:30", "14:15"
    
    # Options-specific data (when market_segment = OPTIONS)
    call_buy_value = Column(Numeric(15, 2), default=0.00)
    call_sell_value = Column(Numeric(15, 2), default=0.00)
    put_buy_value = Column(Numeric(15, 2), default=0.00)
    put_sell_value = Column(Numeric(15, 2), default=0.00)
    
    # Futures-specific data (when market_segment = FUTURES)
    long_positions = Column(Numeric(15, 2), default=0.00)
    short_positions = Column(Numeric(15, 2), default=0.00)
    open_interest_change = Column(Numeric(15, 2), default=0.00)
    
    # Data quality and source
    data_source = Column(Enum(DataSource), default=DataSource.NSE)
    data_quality = Column(Enum(DataQuality), default=DataQuality.HIGH)
    confidence_score = Column(Numeric(5, 3), default=1.000)  # 0.000-1.000
    
    # Raw data storage
    raw_data = Column(JSON, nullable=True)
    processing_notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    participant = relationship("ParticipantProfile", back_populates="activities")
    asset = relationship("Asset")
    flow_events = relationship("ParticipantFlowEvent", back_populates="activity")
    
    # Constraints and indexes for performance
    __table_args__ = (
        UniqueConstraint('trade_date', 'participant_id', 'market_segment', 'symbol',
                        name='_participant_activity_unique'),
        Index('idx_activities_date_participant', 'trade_date', 'participant_id'),
        Index('idx_activities_segment_date', 'market_segment', 'trade_date'),
        Index('idx_activities_symbol_date', 'symbol', 'trade_date'),
        Index('idx_activities_net_value', 'net_value'),  # For flow analysis
        CheckConstraint('buy_value >= 0', name='check_buy_value_positive'),
        CheckConstraint('sell_value >= 0', name='check_sell_value_positive'),
        CheckConstraint('gross_turnover >= 0', name='check_turnover_positive'),
    )
    
    def __repr__(self):
        return f"<ParticipantActivity(date={self.trade_date.date()}, " \
               f"participant_id={self.participant_id}, segment={self.market_segment.value}, " \
               f"net_value={self.net_value})>"

class ParticipantFlowMetrics(Base):
    """Aggregated metrics and trends for participant flows"""
    __tablename__ = "participant_flow_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Time period and participant
    calculation_date = Column(DateTime(timezone=True), nullable=False, index=True)
    participant_id = Column(Integer, ForeignKey("participant_profiles.id"), nullable=False)
    
    # Analysis period configuration
    period_type = Column(String(20), nullable=False)  # "daily", "weekly", "monthly"
    period_days = Column(Integer, nullable=False)     # Number of days analyzed
    market_segment = Column(Enum(MarketSegment), nullable=True) # Segment-specific or overall
    
    # Flow direction and magnitude
    overall_flow = Column(Enum(FlowDirection), nullable=False)
    flow_strength = Column(Numeric(8, 4), nullable=False)  # 0.0000-1.0000
    flow_consistency = Column(Numeric(8, 4), nullable=False) # How consistent the flow is
    
    # Rolling averages and trends
    avg_daily_net_value = Column(Numeric(15, 2), nullable=False)
    avg_daily_turnover = Column(Numeric(15, 2), nullable=False)
    trend_direction = Column(String(20), nullable=True)  # "increasing", "decreasing", "stable"
    trend_strength = Column(Numeric(8, 4), nullable=True) # 0.0000-1.0000
    
    # Volatility and consistency metrics
    flow_volatility = Column(Numeric(8, 4), nullable=True)    # Standard deviation of flows
    max_single_day_inflow = Column(Numeric(15, 2), nullable=True)
    max_single_day_outflow = Column(Numeric(15, 2), nullable=True)
    
    # Market impact assessment
    market_correlation = Column(Numeric(6, 4), nullable=True)  # Correlation with market movement
    influence_score = Column(Numeric(8, 4), nullable=True)     # Impact on market direction
    
    # Behavioral indicators
    activity_pattern = Column(String(50), nullable=True)   # "momentum", "contrarian", "random"
    concentration_ratio = Column(Numeric(8, 4), nullable=True) # How concentrated the activity is
    
    # Comparative metrics
    peer_group_rank = Column(Integer, nullable=True)       # Rank within participant type
    historical_percentile = Column(Numeric(5, 2), nullable=True) # Historical position (0-100)
    
    # Risk and anomaly indicators
    risk_score = Column(Numeric(8, 4), default=0.0000)     # Risk contribution to market
    anomaly_flags = Column(JSON, nullable=True)            # List of detected anomalies
    
    # Statistical metadata
    sample_size = Column(Integer, nullable=False)          # Number of data points used
    confidence_interval = Column(Numeric(8, 4), default=0.95) # Statistical confidence
    calculation_method = Column(String(100), nullable=True) # Method used for calculation
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    participant = relationship("ParticipantProfile", back_populates="flow_metrics")
    
    # Indexes for analytics queries
    __table_args__ = (
        UniqueConstraint('calculation_date', 'participant_id', 'period_type', 
                        'period_days', 'market_segment', name='_flow_metrics_unique'),
        Index('idx_flow_metrics_date_period', 'calculation_date', 'period_type'),
        Index('idx_flow_metrics_participant_segment', 'participant_id', 'market_segment'),
        Index('idx_flow_metrics_flow_strength', 'overall_flow', 'flow_strength'),
    )
    
    def __repr__(self):
        return f"<ParticipantFlowMetrics(participant_id={self.participant_id}, " \
               f"period={self.period_type}, flow={self.overall_flow.value}, " \
               f"strength={self.flow_strength})>"

class ParticipantFlowEvent(Base):
    """Significant flow events and behavioral shifts"""
    __tablename__ = "participant_flow_events"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Event identification
    event_date = Column(DateTime(timezone=True), nullable=False, index=True)
    activity_id = Column(BigInteger, ForeignKey("participant_activities.id"), nullable=True)
    participant_id = Column(Integer, ForeignKey("participant_profiles.id"), nullable=False)
    
    # Event classification
    event_type = Column(String(50), nullable=False, index=True)  # "trend_reversal", "volume_spike"
    event_severity = Column(String(20), nullable=False)        # "low", "medium", "high", "critical"
    event_category = Column(String(50), nullable=True)         # "behavioral", "volume", "directional"
    
    # Event details
    event_title = Column(String(200), nullable=False)
    event_description = Column(Text, nullable=False)
    
    # Quantitative measures
    magnitude = Column(Numeric(10, 4), nullable=False)         # Size of the event
    significance_score = Column(Numeric(8, 4), nullable=False) # Statistical significance
    duration_estimate = Column(String(50), nullable=True)      # "1-day", "3-days", "1-week"
    
    # Context and triggers
    trigger_conditions = Column(JSON, nullable=False)          # What caused the event
    market_context = Column(JSON, nullable=True)               # Market conditions during event
    related_events = Column(JSON, nullable=True)               # Related events in other participants
    
    # Impact assessment
    market_impact = Column(String(20), nullable=True)          # "bullish", "bearish", "neutral"
    confidence_level = Column(Numeric(5, 2), default=0.80)     # 0.00-1.00
    
    # Follow-up tracking
    is_active = Column(Boolean, default=True)
    resolution_date = Column(DateTime(timezone=True), nullable=True)
    actual_duration = Column(String(50), nullable=True)
    outcome_notes = Column(Text, nullable=True)
    
    # Alert and notification
    alert_generated = Column(Boolean, default=False)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)
    notification_sent = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    activity = relationship("ParticipantActivity", back_populates="flow_events")
    participant = relationship("ParticipantProfile")
    alert = relationship("Alert")
    
    # Indexes for event analysis
    __table_args__ = (
        Index('idx_flow_events_date_type', 'event_date', 'event_type'),
        Index('idx_flow_events_participant_severity', 'participant_id', 'event_severity'),
        Index('idx_flow_events_magnitude', 'magnitude'),
        Index('idx_flow_events_active', 'is_active', 'event_date'),
    )
    
    def __repr__(self):
        return f"<ParticipantFlowEvent(type={self.event_type}, " \
               f"participant_id={self.participant_id}, severity={self.event_severity})>"

class ParticipantBehaviorPattern(Base):
    """Long-term behavioral patterns and characteristics"""
    __tablename__ = "participant_behavior_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Pattern identification
    participant_id = Column(Integer, ForeignKey("participant_profiles.id"), nullable=False)
    pattern_name = Column(String(100), nullable=False)
    pattern_type = Column(String(50), nullable=False)  # "seasonal", "momentum", "contrarian"
    
    # Time frame and context
    discovery_date = Column(DateTime(timezone=True), nullable=False)
    pattern_start_date = Column(DateTime(timezone=True), nullable=False)
    pattern_end_date = Column(DateTime(timezone=True), nullable=True)
    analysis_period_days = Column(Integer, nullable=False)
    
    # Pattern characteristics
    pattern_description = Column(Text, nullable=False)
    pattern_strength = Column(Numeric(8, 4), nullable=False)    # 0.0000-1.0000
    statistical_significance = Column(Numeric(8, 4), nullable=False) # p-value or confidence
    
    # Pattern parameters
    trigger_conditions = Column(JSON, nullable=False)           # What triggers the pattern
    typical_duration = Column(String(50), nullable=True)        # How long it usually lasts
    success_rate = Column(Numeric(5, 2), nullable=True)         # Historical success rate
    
    # Market conditions
    market_phase_correlation = Column(JSON, nullable=True)      # Bull/bear market correlation
    volatility_correlation = Column(Numeric(6, 4), nullable=True) # VIX correlation
    seasonal_factors = Column(JSON, nullable=True)              # Monthly/quarterly patterns
    
    # Prediction and modeling
    predictive_power = Column(Numeric(8, 4), nullable=True)     # How well it predicts future behavior
    model_parameters = Column(JSON, nullable=True)              # ML model parameters if applicable
    last_occurrence = Column(DateTime(timezone=True), nullable=True)
    next_expected = Column(DateTime(timezone=True), nullable=True)
    
    # Pattern evolution
    is_active = Column(Boolean, default=True)
    evolution_notes = Column(Text, nullable=True)
    related_patterns = Column(JSON, nullable=True)              # IDs of related patterns
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    participant = relationship("ParticipantProfile")
    
    # Indexes
    __table_args__ = (
        Index('idx_behavior_patterns_participant_type', 'participant_id', 'pattern_type'),
        Index('idx_behavior_patterns_strength', 'pattern_strength'),
        Index('idx_behavior_patterns_active', 'is_active', 'discovery_date'),
    )
    
    def __repr__(self):
        return f"<ParticipantBehaviorPattern(participant_id={self.participant_id}, " \
               f"type={self.pattern_type}, strength={self.pattern_strength})>"

class ParticipantFlowSummary(Base):
    """Daily aggregated summary across all participants"""
    __tablename__ = "participant_flow_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Summary date and scope
    summary_date = Column(DateTime(timezone=True), nullable=False, index=True)
    market_segment = Column(Enum(MarketSegment), nullable=True, index=True)
    exchange = Column(String(20), default="NSE")
    
    # Overall market flows
    total_fii_net_flow = Column(Numeric(15, 2), default=0.00)
    total_dii_net_flow = Column(Numeric(15, 2), default=0.00)
    total_retail_net_flow = Column(Numeric(15, 2), default=0.00)
    total_proprietary_net_flow = Column(Numeric(15, 2), default=0.00)
    
    # Market totals
    total_market_turnover = Column(Numeric(18, 2), default=0.00)
    total_market_net_flow = Column(Numeric(15, 2), default=0.00)
    
    # Participation rates
    fii_market_share = Column(Numeric(8, 4), nullable=True)     # % of total turnover
    dii_market_share = Column(Numeric(8, 4), nullable=True)
    retail_market_share = Column(Numeric(8, 4), nullable=True)
    institutional_share = Column(Numeric(8, 4), nullable=True)  # FII + DII combined
    
    # Flow dynamics
    net_institutional_flow = Column(Numeric(15, 2), default=0.00) # FII + DII net
    retail_vs_institutional = Column(Numeric(15, 2), default=0.00) # Retail - Institutional
    
    # Market sentiment indicators
    flow_sentiment = Column(String(20), nullable=True)          # "bullish", "bearish", "mixed"
    flow_concentration = Column(Numeric(8, 4), nullable=True)   # How concentrated flows are
    flow_divergence = Column(Numeric(8, 4), nullable=True)      # Divergence between participant types
    
    # Volatility and stability
    flow_volatility_index = Column(Numeric(8, 4), nullable=True) # Measure of flow stability
    unusual_activity_count = Column(Integer, default=0)         # Number of unusual events
    
    # Historical context
    flow_vs_5d_avg = Column(Numeric(8, 4), nullable=True)       # % vs 5-day average
    flow_vs_20d_avg = Column(Numeric(8, 4), nullable=True)      # % vs 20-day average
    flow_percentile_rank = Column(Numeric(5, 2), nullable=True) # Historical percentile (0-100)
    
    # Data quality and completeness
    data_completeness = Column(Numeric(5, 2), default=100.00)   # % of expected data received
    participant_count = Column(Integer, nullable=False)         # Number of active participants
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('summary_date', 'market_segment', 'exchange',
                        name='_flow_summary_unique'),
        Index('idx_flow_summary_date_segment', 'summary_date', 'market_segment'),
        Index('idx_flow_summary_sentiment', 'flow_sentiment', 'summary_date'),
    )
    
    def __repr__(self):
        return f"<ParticipantFlowSummary(date={self.summary_date.date()}, " \
               f"segment={self.market_segment.value if self.market_segment else 'all'}, " \
               f"sentiment={self.flow_sentiment})>"

# Data retention and archival configuration
class ParticipantDataRetentionPolicy(Base):
    """Configuration for data retention and archival policies"""
    __tablename__ = "participant_data_retention_policies"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Policy identification
    policy_name = Column(String(100), nullable=False, unique=True)
    table_name = Column(String(100), nullable=False)
    
    # Retention periods
    hot_storage_days = Column(Integer, default=90)              # Fast access storage
    warm_storage_days = Column(Integer, default=365)            # Medium access storage
    cold_storage_days = Column(Integer, default=2555)           # Long-term storage (7 years)
    
    # Archival rules
    archive_after_days = Column(Integer, default=2555)          # When to archive
    delete_after_days = Column(Integer, nullable=True)          # When to delete (if ever)
    
    # Compression and optimization
    enable_compression = Column(Boolean, default=True)
    compression_threshold_days = Column(Integer, default=30)
    partition_strategy = Column(String(50), default="monthly")  # "daily", "weekly", "monthly"
    
    # Policy status
    is_active = Column(Boolean, default=True)
    last_executed = Column(DateTime(timezone=True), nullable=True)
    next_execution = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<ParticipantDataRetentionPolicy(name={self.policy_name}, " \
               f"table={self.table_name}, hot_days={self.hot_storage_days})>"

# Database event listeners for automatic maintenance
@event.listens_for(ParticipantActivity, 'after_insert')
def update_participant_last_activity(mapper, connection, target):
    """Update participant's last activity date when new activity is recorded"""
    connection.execute(
        f"UPDATE participant_profiles SET last_activity_date = '{target.trade_date}' "
        f"WHERE id = {target.participant_id}"
    )

@event.listens_for(ParticipantActivity, 'after_insert')
def update_first_activity_if_null(mapper, connection, target):
    """Set first activity date if it's null"""
    connection.execute(
        f"UPDATE participant_profiles SET first_activity_date = '{target.trade_date}' "
        f"WHERE id = {target.participant_id} AND first_activity_date IS NULL"
    ) 