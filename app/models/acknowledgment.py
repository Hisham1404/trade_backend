from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, Enum, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from enum import Enum as PyEnum
from typing import Optional, Dict, Any

from app.database import Base

class AcknowledgmentStatus(PyEnum):
    """Acknowledgment status enumeration"""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    TIMEOUT = "timeout"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"

class ResponseType(PyEnum):
    """User response type enumeration"""
    ACKNOWLEDGMENT = "acknowledgment"
    DISMISS = "dismiss"
    ESCALATE = "escalate"
    SNOOZE = "snooze"
    CUSTOM_ACTION = "custom_action"

class PreferenceType(PyEnum):
    """Alert preference types"""
    NOTIFICATION_CHANNEL = "notification_channel"
    QUIET_HOURS = "quiet_hours"
    SEVERITY_THRESHOLD = "severity_threshold"
    AUTO_ACKNOWLEDGE = "auto_acknowledge"
    RESPONSE_TIMEOUT = "response_timeout"

class AlertAcknowledgment(Base):
    """Track user acknowledgments for alerts"""
    __tablename__ = "alert_acknowledgments"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    
    # Acknowledgment details
    status = Column(Enum(AcknowledgmentStatus), default=AcknowledgmentStatus.PENDING, index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    response_time_ms = Column(Integer, nullable=True)  # Time from alert to acknowledgment
    
    # Device and channel information
    acknowledged_via = Column(String, nullable=True)  # websocket, push, email, sms
    device_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    
    # Timeout handling
    timeout_at = Column(DateTime, nullable=True)  # When acknowledgment times out
    timeout_duration_minutes = Column(Integer, default=15)  # Default 15 minutes
    
    # User response details
    response_message = Column(Text, nullable=True)  # Optional message from user
    response_data = Column(JSON, nullable=True)  # Additional response metadata
    
    # Synchronization across devices
    sync_token = Column(String, nullable=True, index=True)  # For cross-device sync
    is_synced = Column(Boolean, default=False)
    last_sync_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    alert = relationship("Alert", back_populates="acknowledgments")

class UserResponse(Base):
    """Track detailed user responses to alerts"""
    __tablename__ = "user_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    acknowledgment_id = Column(Integer, ForeignKey("alert_acknowledgments.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    
    # Response details
    response_type = Column(Enum(ResponseType), nullable=False)
    response_value = Column(Text, nullable=True)  # Response text or action taken
    confidence_score = Column(Float, nullable=True)  # User's confidence in their response
    
    # Action taken
    action_taken = Column(String, nullable=True)  # Specific action performed
    action_parameters = Column(JSON, nullable=True)  # Parameters for the action
    action_result = Column(Text, nullable=True)  # Result of the action
    
    # Context
    response_context = Column(JSON, nullable=True)  # Context when response was made
    location_data = Column(JSON, nullable=True)  # Optional location information
    
    # Performance metrics
    decision_time_ms = Column(Integer, nullable=True)  # Time taken to decide
    execution_time_ms = Column(Integer, nullable=True)  # Time taken to execute
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    acknowledgment = relationship("AlertAcknowledgment", backref="responses")

class AlertPreference(Base):
    """User alert preferences and settings"""
    __tablename__ = "alert_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    
    # Preference details
    preference_type = Column(Enum(PreferenceType), nullable=False)
    preference_key = Column(String, nullable=False)  # Specific preference identifier
    preference_value = Column(JSON, nullable=False)  # Preference settings
    
    # Scope and conditions
    asset_symbols = Column(JSON, nullable=True)  # Specific assets this applies to
    alert_types = Column(JSON, nullable=True)  # Specific alert types
    severity_levels = Column(JSON, nullable=True)  # Severity levels this applies to
    
    # Schedule and conditions
    active_hours = Column(JSON, nullable=True)  # When this preference is active
    days_of_week = Column(JSON, nullable=True)  # Days when preference applies
    timezone = Column(String, default="UTC")
    
    # Status
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=1)  # Priority when multiple preferences conflict
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # Optional expiration

class AcknowledgmentAnalytics(Base):
    """Analytics and metrics for acknowledgment patterns"""
    __tablename__ = "acknowledgment_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Time window
    date = Column(DateTime, nullable=False, index=True)
    hour = Column(Integer, nullable=True)  # Hour (0-23) for hourly stats
    
    # User and asset scope
    user_id = Column(String, nullable=True, index=True)  # Null for global stats
    asset_symbol = Column(String, nullable=True, index=True)  # Null for all assets
    alert_type = Column(String, nullable=True)
    
    # Acknowledgment metrics
    total_alerts = Column(Integer, default=0)
    acknowledged_alerts = Column(Integer, default=0)
    timeout_alerts = Column(Integer, default=0)
    dismissed_alerts = Column(Integer, default=0)
    escalated_alerts = Column(Integer, default=0)
    
    # Response time metrics
    avg_response_time_ms = Column(Float, nullable=True)
    median_response_time_ms = Column(Float, nullable=True)
    p95_response_time_ms = Column(Float, nullable=True)
    min_response_time_ms = Column(Integer, nullable=True)
    max_response_time_ms = Column(Integer, nullable=True)
    
    # Performance metrics
    acknowledgment_rate = Column(Float, nullable=True)  # Percentage acknowledged
    timeout_rate = Column(Float, nullable=True)  # Percentage timed out
    escalation_rate = Column(Float, nullable=True)  # Percentage escalated
    
    # Channel effectiveness
    channel_breakdown = Column(JSON, nullable=True)  # Stats by channel (websocket, push, etc.)
    device_breakdown = Column(JSON, nullable=True)  # Stats by device type
    
    # User behavior patterns
    peak_response_hours = Column(JSON, nullable=True)  # Hours with fastest responses
    response_consistency_score = Column(Float, nullable=True)  # How consistent user responses are
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AcknowledgmentTimeout(Base):
    """Track acknowledgment timeouts and automatic escalations"""
    __tablename__ = "acknowledgment_timeouts"
    
    id = Column(Integer, primary_key=True, index=True)
    acknowledgment_id = Column(Integer, ForeignKey("alert_acknowledgments.id"), nullable=False, index=True)
    
    # Timeout details
    timeout_trigger_at = Column(DateTime, nullable=False)  # When timeout should trigger
    timeout_duration_minutes = Column(Integer, nullable=False)
    
    # Status
    is_processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)
    timeout_action = Column(String, nullable=True)  # escalate, notify_manager, etc.
    timeout_result = Column(Text, nullable=True)
    
    # Retry logic
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    next_retry_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    acknowledgment = relationship("AlertAcknowledgment", backref="timeout_record") 