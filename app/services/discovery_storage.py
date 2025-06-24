"""
Source Discovery Storage Service

Provides database operations for intelligent source discovery system.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from sqlalchemy.exc import SQLAlchemyError
import hashlib
import json

from app.models.news import Source, NewsItem
from app.models.source_discovery import (
    SourceScore, SourceMetadata, SourceHistory, DiscoveredContent, SourceAnalytics
)
from app.discovery.validation_scoring import ValidationScore


class SourceDiscoveryStorage:
    """Storage service for intelligent source discovery system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def store_validation_score(self, db: Session, source_id: int, 
                              validation_score: ValidationScore, 
                              content_sample: str = None) -> SourceScore:
        """Store validation score for a source."""
        try:
            source_score = SourceScore(
                source_id=source_id,
                credibility_score=validation_score.credibility_score,
                quality_score=validation_score.quality_score,
                relevance_score=validation_score.relevance_score,
                bias_score=validation_score.bias_score,
                fact_check_score=validation_score.fact_check_score,
                overall_score=validation_score.overall_score,
                confidence=validation_score.confidence,
                content_sample=content_sample[:1000] if content_sample else None,
                content_length=len(content_sample) if content_sample else None
            )
            
            db.add(source_score)
            db.commit()
            db.refresh(source_score)
            
            # Update source reliability score
            source = db.query(Source).filter(Source.id == source_id).first()
            if source:
                new_reliability = (source.reliability_score * 0.7 + validation_score.overall_score * 10 * 0.3)
                source.reliability_score = round(new_reliability, 2)
                db.commit()
            
            self.logger.info(f"Stored validation score for source {source_id}")
            return source_score
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error storing validation score: {e}")
            raise
    
    def store_source_metadata(self, db: Session, source_id: int, 
                             metadata_dict: Dict[str, Any]) -> SourceMetadata:
        """Store or update source metadata."""
        try:
            existing = db.query(SourceMetadata).filter(SourceMetadata.source_id == source_id).first()
            
            if existing:
                for key, value in metadata_dict.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.last_analyzed = datetime.utcnow()
                metadata = existing
            else:
                metadata = SourceMetadata(
                    source_id=source_id,
                    **metadata_dict,
                    last_analyzed=datetime.utcnow()
                )
                db.add(metadata)
            
            db.commit()
            db.refresh(metadata)
            return metadata
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error storing metadata: {e}")
            raise
    
    def log_source_event(self, db: Session, source_id: int, event_type: str,
                        event_description: str = None) -> SourceHistory:
        """Log source lifecycle event."""
        try:
            source = db.query(Source).filter(Source.id == source_id).first()
            reliability_snapshot = source.reliability_score if source else None
            
            latest_score = db.query(SourceScore.overall_score)\
                           .filter(SourceScore.source_id == source_id)\
                           .order_by(desc(SourceScore.scored_at))\
                           .first()
            score_snapshot = latest_score[0] if latest_score else None
            
            history_entry = SourceHistory(
                source_id=source_id,
                event_type=event_type,
                event_description=event_description,
                reliability_score_snapshot=reliability_snapshot,
                overall_score_snapshot=score_snapshot
            )
            
            db.add(history_entry)
            db.commit()
            db.refresh(history_entry)
            return history_entry
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error logging event: {e}")
            raise
    
    def store_discovered_content(self, db: Session, news_item_id: int, 
                                source_id: int, analysis_results: Dict[str, Any]) -> DiscoveredContent:
        """Store analysis results for discovered content."""
        try:
            news_item = db.query(NewsItem).filter(NewsItem.id == news_item_id).first()
            content_hash = None
            if news_item and news_item.content:
                content_hash = hashlib.sha256(news_item.content.encode()).hexdigest()[:32]
            
            discovery_data = DiscoveredContent(
                news_item_id=news_item_id,
                source_id=source_id,
                content_quality_score=analysis_results.get('quality_score'),
                bias_detection_score=analysis_results.get('bias_score'),
                fact_check_score=analysis_results.get('fact_check_score'),
                content_hash=content_hash,
                extracted_entities=analysis_results.get('entities'),
                extracted_keywords=analysis_results.get('keywords'),
                processing_time_ms=analysis_results.get('processing_time_ms')
            )
            
            db.add(discovery_data)
            db.commit()
            db.refresh(discovery_data)
            return discovery_data
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error storing discovery data: {e}")
            raise
    
    def compute_source_analytics(self, db: Session, source_id: int) -> SourceAnalytics:
        """Compute comprehensive analytics for a source."""
        try:
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            analytics = db.query(SourceAnalytics).filter(SourceAnalytics.source_id == source_id).first()
            if not analytics:
                analytics = SourceAnalytics(source_id=source_id)
                db.add(analytics)
            
            # Performance metrics
            recent_scores = db.query(SourceScore)\
                             .filter(
                                 SourceScore.source_id == source_id,
                                 SourceScore.scored_at >= thirty_days_ago
                             )\
                             .all()
            
            if recent_scores:
                analytics.avg_credibility_score = sum(s.credibility_score for s in recent_scores) / len(recent_scores)
                analytics.avg_quality_score = sum(s.quality_score for s in recent_scores) / len(recent_scores)
                analytics.avg_relevance_score = sum(s.relevance_score for s in recent_scores) / len(recent_scores)
            
            # Content metrics
            total_articles = db.query(func.count(NewsItem.id))\
                              .filter(NewsItem.source_id == source_id)\
                              .scalar()
            
            recent_articles = db.query(func.count(NewsItem.id))\
                               .filter(
                                   NewsItem.source_id == source_id,
                                   NewsItem.created_at >= thirty_days_ago
                               )\
                               .scalar()
            
            analytics.total_articles = total_articles
            analytics.articles_last_30_days = recent_articles
            analytics.avg_articles_per_day = recent_articles / 30.0 if recent_articles > 0 else 0
            analytics.last_computed = datetime.utcnow()
            
            db.commit()
            db.refresh(analytics)
            return analytics
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error computing analytics: {e}")
            raise
    
    def get_source_performance_summary(self, db: Session, source_id: int) -> Dict[str, Any]:
        """Get comprehensive performance summary for a source."""
        try:
            source = db.query(Source).filter(Source.id == source_id).first()
            if not source:
                return None
            
            latest_scores = db.query(SourceScore)\
                             .filter(SourceScore.source_id == source_id)\
                             .order_by(desc(SourceScore.scored_at))\
                             .limit(5)\
                             .all()
            
            analytics = db.query(SourceAnalytics).filter(SourceAnalytics.source_id == source_id).first()
            source_metadata = db.query(SourceMetadata).filter(SourceMetadata.source_id == source_id).first()
            
            return {
                'source': {
                    'id': source.id,
                    'name': source.name,
                    'url': source.url,
                    'type': source.type,
                    'reliability_score': float(source.reliability_score),
                    'auto_discovered': source.auto_discovered,
                    'is_active': source.is_active
                },
                'latest_validation': {
                    'scores': [score.to_dict() for score in latest_scores]
                },
                'analytics': analytics.to_summary_dict() if analytics else {},
                'metadata': source_metadata.to_dict() if source_metadata else {}
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting performance summary: {e}")
            raise
    
    def find_duplicate_content(self, db: Session, content_hash: str) -> List[DiscoveredContent]:
        """Find potentially duplicate content based on hash."""
        try:
            return db.query(DiscoveredContent)\
                     .filter(DiscoveredContent.content_hash == content_hash)\
                     .all()
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error finding duplicate content: {e}")
            raise


def create_source_discovery_tables():
    """Create all source discovery tables in the database."""
    try:
        from app.database.connection import Base, engine
        from app.models.source_discovery import (
            SourceScore, SourceMetadata, SourceHistory, 
            DiscoveredContent, SourceRelationship, SourceAnalytics
        )
        
        Base.metadata.create_all(bind=engine)
        logging.info("Created source discovery tables successfully")
        
    except Exception as e:
        logging.error(f"Error creating source discovery tables: {e}")
        raise 