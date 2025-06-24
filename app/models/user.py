from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from app.database.connection import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    
    # API Key for authentication
    api_key = Column(String, unique=True, index=True, nullable=False)
    
    # Subscription and status
    subscription_tier = Column(String, default="free")  # free, premium, enterprise
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_active = Column(DateTime(timezone=True), nullable=True)
    
    # Profile information
    bio = Column(Text, nullable=True)
    timezone = Column(String, default="UTC")
    language = Column(String, default="en")
    
    # Preferences
    notification_preferences = Column(Text, nullable=True)  # JSON string
    risk_tolerance = Column(String, default="moderate")  # conservative, moderate, aggressive
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', username='{self.username}')>" 