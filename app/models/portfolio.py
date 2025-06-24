from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from decimal import Decimal

from app.database.connection import Base

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Portfolio details
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Portfolio value tracking
    capital = Column(Numeric(15, 2), default=0.00)
    total_value = Column(Numeric(15, 2), default=0.00)
    cash_balance = Column(Numeric(15, 2), default=0.00)
    invested_amount = Column(Numeric(15, 2), default=0.00)
    
    # Performance metrics
    total_return = Column(Numeric(15, 4), default=0.0000)  # Percentage
    daily_return = Column(Numeric(15, 4), default=0.0000)  # Percentage
    unrealized_gain_loss = Column(Numeric(15, 2), default=0.00)
    realized_gain_loss = Column(Numeric(15, 2), default=0.00)
    
    # Portfolio settings
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)
    auto_rebalance = Column(Boolean, default=False)
    currency = Column(String, default="USD")
    
    # Risk metrics
    beta = Column(Numeric(10, 4), nullable=True)
    sharpe_ratio = Column(Numeric(10, 4), nullable=True)
    volatility = Column(Numeric(10, 4), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_rebalanced_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    assets = relationship("Asset", back_populates="portfolio", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Portfolio(id={self.id}, name='{self.name}', user_id={self.user_id}, value={self.total_value})>" 