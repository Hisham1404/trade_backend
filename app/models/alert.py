from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Numeric
from sqlalchemy.orm import relationship, synonym
from sqlalchemy.sql import func

from app.database.connection import Base

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Alert identification
    # NOTE: original field was `name`; tests expect `title`.
    # We store both columns referencing the same data for backward compatibility.
    name = Column(String, nullable=False)
    title = Column(String, nullable=False, default="")

    # Keep the two fields in sync via property helpers
    @property
    def display_title(self):
        return self.title or self.name

    @display_title.setter
    def display_title(self, value: str):
        self.title = value
        self.name = value
    alert_type = Column(String, nullable=False)  # price, news, portfolio, technical
    
    # Alert conditions
    symbol = Column(String, nullable=True, index=True)  # Asset symbol to monitor
    condition_type = Column(String, nullable=False)  # above, below, change_percent, etc.
    trigger_value = Column(Numeric(15, 2), nullable=True)  # Price or percentage value
    current_value = Column(Numeric(15, 2), nullable=True)  # Current price/value
    
    # Alert configuration
    is_active = Column(Boolean, default=True)
    is_recurring = Column(Boolean, default=False)  # Should alert repeat after triggering
    frequency = Column(String, default="once")  # once, daily, weekly
    
    # Notification settings
    notification_methods = Column(Text, nullable=True)  # JSON: ["email", "push", "sms"]
    message_template = Column(Text, nullable=True)  # Custom message template
    
    # Alert metadata
    description = Column(Text, nullable=True)
    priority = Column(String, default="medium")  # low, medium, high, critical
    category = Column(String, nullable=True)  # price_alert, news_alert, portfolio_alert
    
    # Alert status
    is_triggered = Column(Boolean, default=False)
    trigger_count = Column(Integer, default=0)
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Advanced conditions (JSON)
    conditions = Column(Text, nullable=True)  # Complex conditions as JSON
    
    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    deliveries = relationship("AlertDelivery", back_populates="alert")
    acknowledgments = relationship("AlertAcknowledgment", back_populates="alert")
    
    def __repr__(self):
        return (
            f"<Alert(id={self.id}, title='{self.title}', user_id={self.user_id}, "
            f"active={self.is_active})>"
        ) 