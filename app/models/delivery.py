from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, Enum, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
from enum import Enum as PyEnum
from typing import Optional, Dict, Any

from app.database import Base

class DeliveryStatus(PyEnum):
    """Delivery status enumeration"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class DeliveryChannel(PyEnum):
    """Delivery channel enumeration"""
    WEBSOCKET = "websocket"
    PUSH_NOTIFICATION = "push_notification"
    EMAIL = "email"
    SMS = "sms"

class AlertDelivery(Base):
    """Track delivery status for each alert to each user's device/channel"""
    __tablename__ = "alert_deliveries"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)  # User receiving the alert
    channel = Column(Enum(DeliveryChannel), nullable=False)
    device_token = Column(String, nullable=True)  # For push notifications
    
    # Delivery status tracking
    status = Column(Enum(DeliveryStatus), default=DeliveryStatus.PENDING, index=True)
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    first_attempt_at = Column(DateTime, nullable=True)
    last_attempt_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)  # Expiry time for delivery attempts
    
    # Delivery details
    delivery_latency_ms = Column(Integer, nullable=True)  # Time from creation to delivery
    error_message = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)
    
    # Retry scheduling
    next_retry_at = Column(DateTime, nullable=True)
    retry_backoff_seconds = Column(Integer, default=60)  # Exponential backoff base
    
    # Metadata and context
    delivery_metadata = Column(JSON, nullable=True)  # Store additional context
    priority = Column(Integer, default=1)  # Higher numbers = higher priority
    
    # Relationships
    alert = relationship("Alert", back_populates="deliveries")

class DeliveryAttempt(Base):
    """Track individual delivery attempts with detailed logging"""
    __tablename__ = "delivery_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    delivery_id = Column(Integer, ForeignKey("alert_deliveries.id"), nullable=False, index=True)
    
    # Attempt details
    attempt_number = Column(Integer, nullable=False)
    attempted_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Result
    success = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)
    http_status_code = Column(Integer, nullable=True)  # For HTTP-based deliveries
    
    # Performance metrics
    latency_ms = Column(Integer, nullable=True)
    payload_size_bytes = Column(Integer, nullable=True)
    
    # Context
    delivery_context = Column(JSON, nullable=True)  # Request/response details
    
    # Relationships
    delivery = relationship("AlertDelivery", backref="delivery_attempts")

class DeliveryStats(Base):
    """Aggregate delivery statistics for monitoring and analytics"""
    __tablename__ = "delivery_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Time window
    date = Column(DateTime, nullable=False, index=True)  # Date for this stats window
    hour = Column(Integer, nullable=True)  # Hour (0-23) for hourly stats, null for daily
    
    # Channel breakdown
    channel = Column(Enum(DeliveryChannel), nullable=False, index=True)
    
    # Delivery metrics
    total_deliveries = Column(Integer, default=0)
    successful_deliveries = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    retried_deliveries = Column(Integer, default=0)
    expired_deliveries = Column(Integer, default=0)
    
    # Performance metrics
    avg_latency_ms = Column(Float, nullable=True)
    max_latency_ms = Column(Integer, nullable=True)
    min_latency_ms = Column(Integer, nullable=True)
    
    # Success rate
    success_rate = Column(Float, nullable=True)  # Percentage (0-100)
    
    # Volume metrics
    peak_deliveries_per_minute = Column(Integer, default=0)
    
    # Error tracking
    common_error_codes = Column(JSON, nullable=True)  # Top error codes and counts
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class DeadLetterQueue(Base):
    """Store alerts that couldn't be delivered after all retry attempts"""
    __tablename__ = "dead_letter_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    original_delivery_id = Column(Integer, ForeignKey("alert_deliveries.id"), nullable=False)
    
    # Original alert data
    alert_id = Column(Integer, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    channel = Column(Enum(DeliveryChannel), nullable=False)
    
    # Failure details
    final_error_message = Column(Text, nullable=True)
    final_error_code = Column(String, nullable=True)
    total_attempts = Column(Integer, nullable=False)
    
    # Alert payload for potential manual retry
    alert_payload = Column(JSON, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_attempt_at = Column(DateTime, nullable=False)
    
    # Manual handling
    reviewed = Column(Boolean, default=False)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String, nullable=True)  # Admin user who reviewed
    resolution_notes = Column(Text, nullable=True)
    
    # Relationships
    original_delivery = relationship("AlertDelivery") 