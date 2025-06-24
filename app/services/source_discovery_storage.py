"""
Source Discovery Storage Service

This service provides comprehensive database operations for the intelligent source discovery system,
including scoring storage, analytics computation, and efficient querying capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import and_, or_, desc, asc, func, case, distinct
from sqlalchemy.exc import SQLAlchemyError
import hashlib
import json

from app.database.connection import get_db
from app.models.news import Source, NewsItem
from app.models.source_discovery import (
    SourceScore, SourceMetadata, SourceHistory, DiscoveredContent, 
    SourceRelationship, SourceAnalytics
)
from app.discovery.validation_scoring import ValidationScore, ValidationScoringSystem


class SourceDiscoveryStorage:
    """
    Comprehensive storage service for intelligent source discovery system.
    Provides high-level operations for storing, retrieving, and analyzing source data.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_system = ValidationScoringSystem()
    
    def store_validation_score(self, db: Session, source_id: int, 
                              validation_score: ValidationScore, 
                              content_sample: str = None,
                              scoring_method: str = "comprehensive",
                              response_time_ms: int = None) -> SourceScore:
        """
        Store a validation score for a source with full context.
        """
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
                content_sample=content_sample[:1000] if content_sample else None,  # Truncate for storage
                scoring_method=scoring_method,
                response_time_ms=response_time_ms,
                content_length=len(content_sample) if content_sample else None
            )
            
            db.add(source_score)
            db.commit()
            db.refresh(source_score)
            
            # Update source reliability score with latest overall score
            source = db.query(Source).filter(Source.id == source_id).first()
            if source:
                # Weighted average: 70% current score, 30% new score
                new_reliability = (source.reliability_score * 0.7 + validation_score.overall_score * 10 * 0.3)
                source.reliability_score = round(new_reliability, 2)
                db.commit()
            
            self.logger.info(f"Stored validation score for source {source_id}: {validation_score.overall_score:.3f}")
            return source_score
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error storing validation score for source {source_id}: {e}")
            raise
    
    def store_source_metadata(self, db: Session, source_id: int, 
                             metadata_dict: Dict[str, Any]) -> SourceMetadata:
        """
        Store or update comprehensive metadata for a source.
        """
        try:
            # Check if metadata already exists
            existing = db.query(SourceMetadata).filter(SourceMetadata.source_id == source_id).first()
            
            if existing:
                # Update existing metadata
                for key, value in metadata_dict.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.last_analyzed = datetime.utcnow()
                metadata = existing
            else:
                # Create new metadata
                metadata = SourceMetadata(
                    source_id=source_id,
                    **metadata_dict,
                    last_analyzed=datetime.utcnow()
                )
                db.add(metadata)
            
            db.commit()
            db.refresh(metadata)
            
            self.logger.info(f"Stored metadata for source {source_id}")
            return metadata
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error storing metadata for source {source_id}: {e}")
            raise
    
    def log_source_event(self, db: Session, source_id: int, event_type: str,
                        event_description: str = None, event_category: str = None,
                        field_changed: str = None, old_value: Any = None, 
                        new_value: Any = None, triggered_by: str = "system",
                        impact_level: str = "low") -> SourceHistory:
        """
        Log a source lifecycle event for historical tracking.
        """
        try:
            # Get current source state for snapshot
            source = db.query(Source).filter(Source.id == source_id).first()
            reliability_snapshot = source.reliability_score if source else None
            
            # Get latest overall score
            latest_score = db.query(SourceScore.overall_score)\
                           .filter(SourceScore.source_id == source_id)\
                           .order_by(desc(SourceScore.scored_at))\
                           .first()
            score_snapshot = latest_score[0] if latest_score else None
            
            # Get content count
            content_count = db.query(func.count(NewsItem.id))\
                             .filter(NewsItem.source_id == source_id)\
                             .scalar()
            
            history_entry = SourceHistory(
                source_id=source_id,
                event_type=event_type,
                event_description=event_description,
                event_category=event_category,
                field_changed=field_changed,
                old_value=json.dumps(old_value) if old_value is not None else None,
                new_value=json.dumps(new_value) if new_value is not None else None,
                reliability_score_snapshot=reliability_snapshot,
                overall_score_snapshot=score_snapshot,
                content_count_snapshot=content_count,
                triggered_by=triggered_by,
                impact_level=impact_level
            )
            
            db.add(history_entry)
            db.commit()
            db.refresh(history_entry)
            
            self.logger.info(f"Logged event '{event_type}' for source {source_id}")
            return history_entry
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error logging event for source {source_id}: {e}")
            raise
    
    def store_discovered_content(self, db: Session, news_item_id: int, 
                                source_id: int, analysis_results: Dict[str, Any],
                                discovery_context: Dict[str, Any] = None) -> DiscoveredContent:
        """
        Store analysis results for discovered content.
        """
        try:
            # Generate content hash for duplicate detection
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
                structure_score=analysis_results.get('structure_score'),
                language_quality_score=analysis_results.get('language_quality'),
                information_density=analysis_results.get('information_density'),
                content_hash=content_hash,
                extracted_entities=analysis_results.get('entities'),
                extracted_keywords=analysis_results.get('keywords'),
                topic_classification=analysis_results.get('topics'),
                financial_instruments=analysis_results.get('financial_instruments'),
                processing_time_ms=analysis_results.get('processing_time_ms'),
                model_versions=analysis_results.get('model_versions'),
                **discovery_context if discovery_context else {}
            )
            
            db.add(discovery_data)
            db.commit()
            db.refresh(discovery_data)
            
            self.logger.info(f"Stored discovery data for content {news_item_id}")
            return discovery_data
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error storing discovery data for content {news_item_id}: {e}")
            raise
    
    def detect_source_relationships(self, db: Session, source_id: int, 
                                   similarity_threshold: float = 0.8) -> List[SourceRelationship]:
        """
        Detect and store relationships between sources based on content similarity.
        """
        try:
            relationships = []
            
            # Get recent content from the source
            source_content = db.query(DiscoveredContent, NewsItem)\
                              .join(NewsItem, DiscoveredContent.news_item_id == NewsItem.id)\
                              .filter(DiscoveredContent.source_id == source_id)\
                              .order_by(desc(DiscoveredContent.discovered_at))\
                              .limit(10)\
                              .all()
            
            if not source_content:
                return relationships
            
            # Find potential similar sources based on content hashes
            content_hashes = [content.DiscoveredContent.content_hash for content in source_content 
                            if content.DiscoveredContent.content_hash]
            
            if content_hashes:
                similar_content = db.query(DiscoveredContent)\
                                   .filter(
                                       DiscoveredContent.content_hash.in_(content_hashes),
                                       DiscoveredContent.source_id != source_id
                                   )\
                                   .all()
                
                # Group by source and calculate similarity
                source_similarities = {}
                for content in similar_content:
                    other_source_id = content.source_id
                    if other_source_id not in source_similarities:
                        source_similarities[other_source_id] = {'matches': 0, 'total': len(content_hashes)}
                    source_similarities[other_source_id]['matches'] += 1
                
                # Create relationships for sources above threshold
                for other_source_id, similarity_data in source_similarities.items():
                    similarity_score = similarity_data['matches'] / similarity_data['total']
                    
                    if similarity_score >= similarity_threshold:
                        # Check if relationship already exists
                        existing = db.query(SourceRelationship)\
                                    .filter(
                                        or_(
                                            and_(SourceRelationship.source_id == source_id,
                                                 SourceRelationship.related_source_id == other_source_id),
                                            and_(SourceRelationship.source_id == other_source_id,
                                                 SourceRelationship.related_source_id == source_id)
                                        )
                                    ).first()
                        
                        if not existing:
                            relationship = SourceRelationship(
                                source_id=source_id,
                                related_source_id=other_source_id,
                                relationship_type="similar" if similarity_score < 0.95 else "duplicate",
                                relationship_strength=similarity_score,
                                content_similarity=similarity_score,
                                sample_articles_compared=similarity_data['total'],
                                analysis_method="content_hash",
                                confidence_level="high" if similarity_score > 0.9 else "medium"
                            )
                            
                            db.add(relationship)
                            relationships.append(relationship)
            
            if relationships:
                db.commit()
                self.logger.info(f"Detected {len(relationships)} relationships for source {source_id}")
            
            return relationships
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error detecting relationships for source {source_id}: {e}")
            raise
    
    def compute_source_analytics(self, db: Session, source_id: int) -> SourceAnalytics:
        """
        Compute comprehensive analytics for a source.
        """
        try:
            start_time = datetime.utcnow()
            thirty_days_ago = start_time - timedelta(days=30)
            
            # Get or create analytics record
            analytics = db.query(SourceAnalytics).filter(SourceAnalytics.source_id == source_id).first()
            if not analytics:
                analytics = SourceAnalytics(source_id=source_id)
                db.add(analytics)
            
            # Performance metrics (last 30 days)
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
                
                # Determine trend
                if len(recent_scores) >= 2:
                    first_half = recent_scores[:len(recent_scores)//2]
                    second_half = recent_scores[len(recent_scores)//2:]
                    
                    first_avg = sum(s.overall_score for s in first_half) / len(first_half)
                    second_avg = sum(s.overall_score for s in second_half) / len(second_half)
                    
                    if second_avg > first_avg + 0.05:
                        analytics.score_trend = "improving"
                    elif second_avg < first_avg - 0.05:
                        analytics.score_trend = "declining"
                    else:
                        analytics.score_trend = "stable"
            
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
            
            # Duplicate rate
            total_discovered = db.query(func.count(DiscoveredContent.id))\
                                .filter(DiscoveredContent.source_id == source_id)\
                                .scalar()
            
            duplicate_discovered = db.query(func.count(DiscoveredContent.id))\
                                    .filter(
                                        DiscoveredContent.source_id == source_id,
                                        DiscoveredContent.content_hash.in_(
                                            db.query(DiscoveredContent.content_hash)
                                            .filter(DiscoveredContent.source_id != source_id)
                                            .distinct()
                                        )
                                    )\
                                    .scalar()
            
            analytics.duplicate_rate = (duplicate_discovered / total_discovered) if total_discovered > 0 else 0
            
            # Quality metrics from discovered content
            quality_stats = db.query(
                func.avg(DiscoveredContent.content_quality_score),
                func.avg(DiscoveredContent.bias_detection_score),
                func.avg(DiscoveredContent.fact_check_score)
            ).filter(DiscoveredContent.source_id == source_id).first()
            
            if quality_stats[0]:  # If we have quality data
                analytics.high_quality_rate = db.query(func.count(DiscoveredContent.id))\
                                                .filter(
                                                    DiscoveredContent.source_id == source_id,
                                                    DiscoveredContent.content_quality_score >= 0.7
                                                )\
                                                .scalar() / total_discovered if total_discovered > 0 else 0
                
                analytics.bias_neutrality_score = quality_stats[1]
                analytics.factual_accuracy_rate = db.query(func.count(DiscoveredContent.id))\
                                                    .filter(
                                                        DiscoveredContent.source_id == source_id,
                                                        DiscoveredContent.fact_check_score >= 0.6
                                                    )\
                                                    .scalar() / total_discovered if total_discovered > 0 else 0
            
            # Computation metadata
            end_time = datetime.utcnow()
            analytics.last_computed = end_time
            analytics.computation_duration_ms = int((end_time - start_time).total_seconds() * 1000)
            analytics.data_points_analyzed = total_articles + len(recent_scores)
            
            db.commit()
            db.refresh(analytics)
            
            self.logger.info(f"Computed analytics for source {source_id}")
            return analytics
            
        except SQLAlchemyError as e:
            db.rollback()
            self.logger.error(f"Error computing analytics for source {source_id}: {e}")
            raise
    
    def get_source_performance_summary(self, db: Session, source_id: int) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for a source.
        """
        try:
            # Get basic source info
            source = db.query(Source)\
                       .options(
                           selectinload(Source.metadata),
                           selectinload(Source.analytics)
                       )\
                       .filter(Source.id == source_id)\
                       .first()
            
            if not source:
                return None
            
            # Get latest scores
            latest_scores = db.query(SourceScore)\
                             .filter(SourceScore.source_id == source_id)\
                             .order_by(desc(SourceScore.scored_at))\
                             .limit(5)\
                             .all()
            
            # Get recent content quality
            recent_content_quality = db.query(
                func.avg(DiscoveredContent.content_quality_score),
                func.count(DiscoveredContent.id)
            ).filter(
                DiscoveredContent.source_id == source_id,
                DiscoveredContent.discovered_at >= datetime.utcnow() - timedelta(days=7)
            ).first()
            
            # Get relationship count
            relationship_count = db.query(func.count(SourceRelationship.id))\
                                  .filter(SourceRelationship.source_id == source_id)\
                                  .scalar()
            
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
                    'scores': [score.to_dict() for score in latest_scores],
                    'trend': source.analytics.score_trend if source.analytics else None
                },
                'content_metrics': {
                    'total_articles': source.analytics.total_articles if source.analytics else 0,
                    'recent_quality_avg': float(recent_content_quality[0]) if recent_content_quality[0] else None,
                    'recent_content_count': recent_content_quality[1] if recent_content_quality else 0,
                    'duplicate_rate': float(source.analytics.duplicate_rate) if source.analytics and source.analytics.duplicate_rate else 0
                },
                'metadata': source.metadata.to_dict() if source.metadata else {},
                'analytics': source.analytics.to_summary_dict() if source.analytics else {},
                'relationships': relationship_count
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting performance summary for source {source_id}: {e}")
            raise
    
    def find_duplicate_content(self, db: Session, content_hash: str, 
                              exclude_source_id: int = None) -> List[DiscoveredContent]:
        """
        Find potentially duplicate content based on hash.
        """
        try:
            query = db.query(DiscoveredContent)\
                     .filter(DiscoveredContent.content_hash == content_hash)
            
            if exclude_source_id:
                query = query.filter(DiscoveredContent.source_id != exclude_source_id)
            
            return query.all()
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error finding duplicate content: {e}")
            raise
    
    def get_source_discovery_dashboard(self, db: Session, limit: int = 20) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for source discovery system.
        """
        try:
            # Top performing sources
            top_sources = db.query(Source, SourceAnalytics)\
                           .join(SourceAnalytics, Source.id == SourceAnalytics.source_id)\
                           .filter(Source.is_active == True)\
                           .order_by(desc(SourceAnalytics.avg_credibility_score))\
                           .limit(limit)\
                           .all()
            
            # Recent discoveries
            recent_discoveries = db.query(Source)\
                                  .filter(
                                      Source.auto_discovered == True,
                                      Source.created_at >= datetime.utcnow() - timedelta(days=7)
                                  )\
                                  .order_by(desc(Source.created_at))\
                                  .limit(10)\
                                  .all()
            
            # Content quality trends
            quality_trends = db.query(
                func.date(DiscoveredContent.discovered_at).label('date'),
                func.avg(DiscoveredContent.content_quality_score).label('avg_quality'),
                func.count(DiscoveredContent.id).label('content_count')
            ).filter(
                DiscoveredContent.discovered_at >= datetime.utcnow() - timedelta(days=30)
            ).group_by(
                func.date(DiscoveredContent.discovered_at)
            ).order_by('date').all()
            
            # System statistics
            total_sources = db.query(func.count(Source.id)).scalar()
            active_sources = db.query(func.count(Source.id)).filter(Source.is_active == True).scalar()
            auto_discovered = db.query(func.count(Source.id)).filter(Source.auto_discovered == True).scalar()
            
            return {
                'summary': {
                    'total_sources': total_sources,
                    'active_sources': active_sources,
                    'auto_discovered_sources': auto_discovered,
                    'discovery_rate': (auto_discovered / total_sources) if total_sources > 0 else 0
                },
                'top_sources': [
                    {
                        'source': {
                            'id': source.id,
                            'name': source.name,
                            'type': source.type,
                            'reliability_score': float(source.reliability_score)
                        },
                        'analytics': analytics.to_summary_dict()
                    }
                    for source, analytics in top_sources
                ],
                'recent_discoveries': [
                    {
                        'id': source.id,
                        'name': source.name,
                        'url': source.url,
                        'discovered_at': source.created_at.isoformat(),
                        'type': source.type
                    }
                    for source in recent_discoveries
                ],
                'quality_trends': [
                    {
                        'date': trend.date.isoformat(),
                        'avg_quality': float(trend.avg_quality) if trend.avg_quality else 0,
                        'content_count': trend.content_count
                    }
                    for trend in quality_trends
                ]
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            raise


# Utility functions for database operations
def create_source_discovery_tables(db: Session):
    """
    Create all source discovery tables in the database.
    This should be called during database initialization.
    """
    try:
        from app.database.connection import Base, engine
        
        # Import all models to ensure they're registered
        from app.models.source_discovery import (
            SourceScore, SourceMetadata, SourceHistory, 
            DiscoveredContent, SourceRelationship, SourceAnalytics
        )
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logging.info("Created source discovery tables successfully")
        
    except Exception as e:
        logging.error(f"Error creating source discovery tables: {e}")
        raise


def initialize_source_discovery_indexes(db: Session):
    """
    Create additional indexes for performance optimization.
    """
    try:
        # This would contain additional custom indexes
        # that aren't defined in the model classes
        logging.info("Initialized source discovery indexes")
        
    except Exception as e:
        logging.error(f"Error initializing indexes: {e}")
        raise 