from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database.connection import Base

class Source(Base):
    __tablename__ = "sources"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Source identification
    name = Column(String, nullable=False, unique=True)  # e.g., "Reuters", "Bloomberg"
    url = Column(String, nullable=True)  # Base URL of the source
    api_key = Column(String, nullable=True)  # API key if required
    
    # Source metadata
    category = Column(String, nullable=True)  # financial, crypto, general
    reliability_score = Column(Numeric(3, 2), default=5.0)  # 1-10 scale
    language = Column(String, default="en")
    
    # Source settings
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    rate_limit_per_hour = Column(Integer, default=100)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_fetched_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    news_items = relationship("NewsItem", back_populates="source")
    
    def __repr__(self):
        return f"<Source(id={self.id}, name='{self.name}')>"

class NewsItem(Base):
    __tablename__ = "news_items"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False)
    
    # Article information
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    url = Column(String, nullable=False)
    
    # Article metadata
    author = Column(String, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=False)
    category = Column(String, nullable=True)  # markets, stocks, crypto, economy
    tags = Column(Text, nullable=True)  # JSON array of tags
    
    # Article analysis
    sentiment_score = Column(Numeric(3, 2), nullable=True)  # -1 to 1 scale
    sentiment_label = Column(String, nullable=True)  # positive, negative, neutral
    relevance_score = Column(Numeric(3, 2), nullable=True)  # 0-1 scale
    
    # Asset relations
    mentioned_symbols = Column(Text, nullable=True)  # JSON array of symbols
    primary_symbol = Column(String, nullable=True)  # Main symbol the article is about
    
    # Article processing
    is_processed = Column(Boolean, default=False)
    is_relevant = Column(Boolean, default=True)
    is_duplicate = Column(Boolean, default=False)
    
    # Social metrics
    social_shares = Column(Integer, default=0)
    social_engagement = Column(Integer, default=0)
    
    # System fields
    external_id = Column(String, nullable=True, unique=True)  # ID from source API
    language = Column(String, default="en")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    source = relationship("Source", back_populates="news_items")
    
    def __repr__(self):
        return f"<NewsItem(id={self.id}, title='{self.title[:50]}...', source_id={self.source_id})>" 