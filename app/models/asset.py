from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from decimal import Decimal

from app.database.connection import Base

class Asset(Base):
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    
    # Asset identification
    symbol = Column(String, nullable=False, index=True)  # e.g., AAPL, BTC, etc.
    name = Column(String, nullable=False)  # e.g., Apple Inc., Bitcoin
    asset_type = Column(String, nullable=False)  # stock, crypto, bond, etf, etc.
    exchange = Column(String, nullable=True)  # NASDAQ, NYSE, BINANCE, etc.
    
    # Holdings information
    quantity = Column(Numeric(20, 8), nullable=False)  # Number of shares/units
    average_cost = Column(Numeric(15, 2), nullable=False)  # Average cost per unit
    current_price = Column(Numeric(15, 2), nullable=True)  # Current market price
    
    # Calculated values
    total_cost = Column(Numeric(15, 2), nullable=False)  # quantity * average_cost
    current_value = Column(Numeric(15, 2), nullable=True)  # quantity * current_price
    unrealized_gain_loss = Column(Numeric(15, 2), default=0.00)  # current_value - total_cost
    unrealized_gain_loss_pct = Column(Numeric(10, 4), default=0.0000)  # Percentage
    
    # Trading information
    first_purchase_date = Column(DateTime(timezone=True), nullable=False)
    last_updated_price_at = Column(DateTime(timezone=True), nullable=True)
    
    # Asset metadata
    sector = Column(String, nullable=True)  # Technology, Finance, etc.
    industry = Column(String, nullable=True)  # Software, Banking, etc.
    market_cap = Column(Numeric(20, 2), nullable=True)
    
    # Asset status
    is_active = Column(Boolean, default=True)
    is_watchlist = Column(Boolean, default=False)  # For tracking without owning
    
    # Risk metrics
    beta = Column(Numeric(10, 4), nullable=True)
    volatility = Column(Numeric(10, 4), nullable=True)
    
    # Additional info
    notes = Column(Text, nullable=True)
    target_allocation = Column(Numeric(5, 2), nullable=True)  # Target percentage in portfolio
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="assets")
    
    def __repr__(self):
        return f"<Asset(id={self.id}, symbol='{self.symbol}', quantity={self.quantity}, portfolio_id={self.portfolio_id})>" 