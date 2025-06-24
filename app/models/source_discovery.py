"""
Source Discovery Persistent Storage Models

This module contains SQLAlchemy models for storing intelligent source discovery data,
including validation scores, metadata, historical tracking, and analytics.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, ForeignKey, 
    Numeric, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional

from app.database.connection import Base


class SourceScore(Base):
    """
    Historical validation scores for sources using the ValidationScoringSystem.
    Tracks score changes over time for analytics and trend analysis.
    """
    __tablename__ = "source_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    
    # Validation scores (from ValidationScoringSystem)
    credibility_score = Column(Numeric(4, 3), nullable=False)      # 0.000-1.000
    quality_score = Column(Numeric(4, 3), nullable=False)          # 0.000-1.000  
    relevance_score = Column(Numeric(4, 3), nullable=False)        # 0.000-1.000
    bias_score = Column(Numeric(4, 3), nullable=False)             # 0.000-1.000
    fact_check_score = Column(Numeric(4, 3), nullable=False)       # 0.000-1.000
    overall_score = Column(Numeric(4, 3), nullable=False)          # 0.000-1.000
    confidence = Column(Numeric(4, 3), nullable=False)             # 0.000-1.000
    
    # Score context
    content_sample = Column(Text, nullable=True)                   # Sample content used for scoring
    scoring_method = Column(String, default="comprehensive")       # comprehensive, rule_based, ml_based
    model_version = Column(String, nullable=True)                  # Version of ML model used
    
    # Performance tracking
    response_time_ms = Column(Integer, nullable=True)              # Time taken to compute scores
    content_length = Column(Integer, nullable=True)               # Length of analyzed content
    
    # Timestamps
    scored_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    source = relationship("Source", back_populates="scores")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_source_scores_time', 'source_id', 'scored_at'),
        Index('idx_source_scores_overall', 'overall_score', 'scored_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format matching ValidationScore"""
        return {
            'credibility': float(self.credibility_score),
            'quality': float(self.quality_score),
            'relevance': float(self.relevance_score),
            'bias': float(self.bias_score),
            'fact_check': float(self.fact_check_score),
            'overall': float(self.overall_score),
            'confidence': float(self.confidence),
            'scored_at': self.scored_at.isoformat() if self.scored_at else None,
            'scoring_method': self.scoring_method
        }


class SourceMetadata(Base):
    """
    Extended metadata for sources discovered through intelligent discovery.
    Stores domain analysis, content characteristics, and discovery context.
    """
    __tablename__ = "source_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, unique=True, index=True)
    
    # Domain analysis
    domain_category = Column(String, nullable=True)               # official, verified_media, social, general
    domain_authority = Column(Numeric(4, 1), nullable=True)       # Domain authority score (0-100)
    ssl_certificate = Column(Boolean, default=False)             # HTTPS enabled
    domain_age_days = Column(Integer, nullable=True)             # Age of domain in days
    
    # Content characteristics
    content_patterns = Column(JSON, nullable=True)               # JSON object with content analysis
    language_detection = Column(JSON, nullable=True)             # Primary and detected languages
    update_frequency = Column(String, nullable=True)             # daily, weekly, monthly, irregular
    content_quality_trend = Column(String, nullable=True)        # improving, stable, declining
    
    # Discovery context
    discovery_method = Column(String, nullable=True)             # search_engine, referral, manual, api
    discovery_keywords = Column(JSON, nullable=True)             # Keywords that led to discovery
    discovery_source = Column(String, nullable=True)             # What led to discovery
    discovery_confidence = Column(Numeric(4, 3), nullable=True)  # Confidence in discovery relevance
    
    # Technical characteristics
    response_times = Column(JSON, nullable=True)                 # Historical response time data
    availability_score = Column(Numeric(4, 3), default=1.0)      # Uptime/availability score
    rate_limit_detected = Column(Boolean, default=False)         # Whether rate limiting detected
    requires_authentication = Column(Boolean, default=False)     # Requires auth/API key
    
    # Content analysis summary
    typical_article_length = Column(Integer, nullable=True)      # Average article length
    media_richness = Column(String, nullable=True)              # text_only, images, videos, interactive
    citation_frequency = Column(Numeric(4, 3), nullable=True)    # How often sources cite others
    data_richness = Column(Numeric(4, 3), nullable=True)         # Presence of data/statistics
    
    # Behavioral patterns
    publishing_schedule = Column(JSON, nullable=True)            # When content is typically published
    peak_activity_hours = Column(JSON, nullable=True)            # Most active publishing hours
    content_lifecycle = Column(String, nullable=True)           # how_long_content_stays_relevant
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_analyzed = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    source = relationship("Source", back_populates="source_metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'domain_category': self.domain_category,
            'domain_authority': float(self.domain_authority) if self.domain_authority else None,
            'ssl_certificate': self.ssl_certificate,
            'discovery_method': self.discovery_method,
            'discovery_confidence': float(self.discovery_confidence) if self.discovery_confidence else None,
            'content_patterns': self.content_patterns,
            'update_frequency': self.update_frequency,
            'availability_score': float(self.availability_score) if self.availability_score else None,
            'last_analyzed': self.last_analyzed.isoformat() if self.last_analyzed else None
        }


class SourceHistory(Base):
    """
    Historical tracking of source changes, performance, and lifecycle events.
    Enables trend analysis and source evolution tracking.
    """
    __tablename__ = "source_history"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    
    # Event information
    event_type = Column(String, nullable=False, index=True)       # discovered, updated, validated, deactivated
    event_description = Column(Text, nullable=True)              # Detailed description of what changed
    event_category = Column(String, nullable=True)               # performance, content, technical, discovery
    
    # Change tracking
    field_changed = Column(String, nullable=True)                # Specific field that changed
    old_value = Column(Text, nullable=True)                      # Previous value (JSON if complex)
    new_value = Column(Text, nullable=True)                      # New value (JSON if complex)
    
    # Performance metrics at time of event
    reliability_score_snapshot = Column(Numeric(3, 2), nullable=True)
    overall_score_snapshot = Column(Numeric(4, 3), nullable=True)
    content_count_snapshot = Column(Integer, nullable=True)      # Number of articles at this time
    
    # Context data
    triggered_by = Column(String, nullable=True)                 # system, user, scheduler, discovery
    user_id = Column(Integer, nullable=True)                     # If triggered by user action
    system_version = Column(String, nullable=True)               # Version of discovery system
    
    # Impact assessment
    impact_level = Column(String, nullable=True)                 # low, medium, high, critical
    requires_review = Column(Boolean, default=False)             # Needs human review
    
    # Timestamps
    event_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    source = relationship("Source", back_populates="history")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_source_history_type_time', 'source_id', 'event_type', 'event_timestamp'),
        Index('idx_source_history_impact', 'impact_level', 'requires_review'),
    )


class DiscoveredContent(Base):
    """
    Stores analysis results of content discovered from sources.
    Links to NewsItem but provides additional discovery-specific analytics.
    """
    __tablename__ = "discovered_content"
    
    id = Column(Integer, primary_key=True, index=True)
    news_item_id = Column(Integer, ForeignKey("news_items.id"), nullable=False, unique=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    
    # Content analysis results
    content_quality_score = Column(Numeric(4, 3), nullable=True)     # Quality score from QualityScoring
    bias_detection_score = Column(Numeric(4, 3), nullable=True)      # Bias score from BiasDetection
    fact_check_score = Column(Numeric(4, 3), nullable=True)          # Fact-check score
    
    # Content characteristics
    structure_score = Column(Numeric(4, 3), nullable=True)           # Article structure quality
    language_quality_score = Column(Numeric(4, 3), nullable=True)    # Language quality
    information_density = Column(Numeric(4, 3), nullable=True)       # Information richness
    
    # Discovery context
    discovery_pathway = Column(String, nullable=True)                # How content was discovered
    discovery_ranking = Column(Integer, nullable=True)               # Rank in discovery results
    discovery_confidence = Column(Numeric(4, 3), nullable=True)      # Confidence in relevance
    
    # Content fingerprinting (for duplicate detection)
    content_hash = Column(String, nullable=True, index=True)         # Hash of content for deduplication
    title_similarity_hash = Column(String, nullable=True)            # Title similarity hash
    content_similarity_scores = Column(JSON, nullable=True)          # Similarity to other articles
    
    # Processing results
    extracted_entities = Column(JSON, nullable=True)                 # Named entities found
    extracted_keywords = Column(JSON, nullable=True)                 # Key terms extracted
    topic_classification = Column(JSON, nullable=True)               # Topic classifications
    financial_instruments = Column(JSON, nullable=True)              # Financial assets mentioned
    
    # Verification status
    verification_status = Column(String, default="pending")          # pending, verified, flagged, rejected
    verification_notes = Column(Text, nullable=True)                 # Notes from verification process
    verification_source = Column(String, nullable=True)              # Who/what verified it
    
    # Performance tracking
    processing_time_ms = Column(Integer, nullable=True)              # Time taken to analyze
    model_versions = Column(JSON, nullable=True)                     # Versions of models used
    
    # Timestamps
    discovered_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    news_item = relationship("NewsItem", back_populates="discovery_data")
    source = relationship("Source", back_populates="discovered_content")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_discovered_content_quality', 'content_quality_score', 'discovered_at'),
        Index('idx_discovered_content_verification', 'verification_status', 'discovered_at'),
        Index('idx_content_hash_dedup', 'content_hash'),
    )


class SourceRelationship(Base):
    """
    Tracks relationships between sources for network analysis and duplicate detection.
    Helps identify source networks, content sharing patterns, and redundancies.
    """
    __tablename__ = "source_relationships"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    related_source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    
    # Relationship characteristics
    relationship_type = Column(String, nullable=False)           # duplicate, similar, syndicated, competitor
    relationship_strength = Column(Numeric(4, 3), nullable=False) # 0.000-1.000 strength score
    relationship_direction = Column(String, nullable=True)       # bidirectional, source_to_related, related_to_source
    
    # Evidence for relationship
    content_similarity = Column(Numeric(4, 3), nullable=True)    # How similar their content is
    publishing_overlap = Column(Numeric(4, 3), nullable=True)    # How often they cover same topics
    timing_correlation = Column(Numeric(4, 3), nullable=True)    # How synchronized their publishing is
    domain_similarity = Column(Numeric(4, 3), nullable=True)     # Domain/URL similarity
    
    # Relationship metadata
    first_detected = Column(DateTime(timezone=True), server_default=func.now())
    last_confirmed = Column(DateTime(timezone=True), server_default=func.now())
    confidence_level = Column(String, default="medium")          # low, medium, high
    
    # Analysis details
    sample_articles_compared = Column(Integer, nullable=True)    # Number of articles analyzed
    analysis_method = Column(String, nullable=True)              # content_hash, similarity_ml, manual
    evidence_data = Column(JSON, nullable=True)                  # Supporting evidence
    
    # Status tracking
    is_active = Column(Boolean, default=True)                    # Whether relationship is still valid
    requires_review = Column(Boolean, default=False)             # Needs human verification
    reviewed_by = Column(String, nullable=True)                  # Who reviewed this relationship
    review_notes = Column(Text, nullable=True)                   # Review comments
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    source = relationship("Source", foreign_keys=[source_id], back_populates="source_relationships")
    related_source = relationship("Source", foreign_keys=[related_source_id])
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('source_id', 'related_source_id', name='uq_source_relationship'),
        Index('idx_relationship_strength', 'relationship_strength', 'relationship_type'),
        Index('idx_relationship_active', 'is_active', 'confidence_level'),
    )


class SourceAnalytics(Base):
    """
    Aggregated analytics and statistics for sources.
    Pre-computed metrics for dashboard and reporting purposes.
    """
    __tablename__ = "source_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, unique=True, index=True)
    
    # Performance metrics (last 30 days)
    avg_credibility_score = Column(Numeric(4, 3), nullable=True)
    avg_quality_score = Column(Numeric(4, 3), nullable=True)
    avg_relevance_score = Column(Numeric(4, 3), nullable=True)
    score_trend = Column(String, nullable=True)                  # improving, stable, declining
    
    # Content metrics
    total_articles = Column(Integer, default=0)
    articles_last_30_days = Column(Integer, default=0)
    avg_articles_per_day = Column(Numeric(6, 2), nullable=True)
    duplicate_rate = Column(Numeric(4, 3), nullable=True)        # Rate of duplicate content
    
    # Quality metrics
    high_quality_rate = Column(Numeric(4, 3), nullable=True)     # % of high-quality articles
    factual_accuracy_rate = Column(Numeric(4, 3), nullable=True) # % passing fact checks
    bias_neutrality_score = Column(Numeric(4, 3), nullable=True) # Average bias neutrality
    
    # Discovery metrics
    discovery_success_rate = Column(Numeric(4, 3), nullable=True) # % of successful discoveries
    false_positive_rate = Column(Numeric(4, 3), nullable=True)    # % of irrelevant content
    processing_efficiency = Column(Numeric(8, 2), nullable=True)  # Articles per processing hour
    
    # Reliability metrics
    uptime_percentage = Column(Numeric(5, 2), nullable=True)      # Source availability
    response_time_avg = Column(Integer, nullable=True)            # Average response time in ms
    error_rate = Column(Numeric(4, 3), nullable=True)            # Rate of fetch errors
    
    # Temporal patterns
    peak_publishing_hour = Column(Integer, nullable=True)         # Hour of day with most content
    avg_content_freshness = Column(Integer, nullable=True)        # Average content age when discovered
    publishing_consistency = Column(Numeric(4, 3), nullable=True) # How consistent publishing schedule is
    
    # Impact metrics
    unique_stories_contributed = Column(Integer, nullable=True)   # Stories only this source covered
    influence_score = Column(Numeric(4, 3), nullable=True)       # How often others reference this source
    market_impact_correlation = Column(Numeric(4, 3), nullable=True) # Correlation with market movements
    
    # Computation metadata
    last_computed = Column(DateTime(timezone=True), nullable=True)
    computation_duration_ms = Column(Integer, nullable=True)
    data_points_analyzed = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    source = relationship("Source", back_populates="analytics")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for APIs"""
        return {
            'performance': {
                'avg_credibility': float(self.avg_credibility_score) if self.avg_credibility_score else None,
                'avg_quality': float(self.avg_quality_score) if self.avg_quality_score else None,
                'avg_relevance': float(self.avg_relevance_score) if self.avg_relevance_score else None,
                'score_trend': self.score_trend
            },
            'content': {
                'total_articles': self.total_articles,
                'articles_last_30_days': self.articles_last_30_days,
                'avg_per_day': float(self.avg_articles_per_day) if self.avg_articles_per_day else None,
                'duplicate_rate': float(self.duplicate_rate) if self.duplicate_rate else None
            },
            'reliability': {
                'uptime_percentage': float(self.uptime_percentage) if self.uptime_percentage else None,
                'response_time_avg': self.response_time_avg,
                'error_rate': float(self.error_rate) if self.error_rate else None
            },
            'last_computed': self.last_computed.isoformat() if self.last_computed else None
        } 