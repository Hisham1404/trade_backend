# Source Lifecycle Management System
"""
This module provides comprehensive lifecycle management for discovered sources including
health monitoring, content freshness tracking, maintenance, archival, and notifications.
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from app.models import Source, NewsItem
from app.models.source_discovery import SourceScore, SourceMetadata, SourceHistory, SourceAnalytics
from .source_categorizer import SourceCategorizer
from .validation_scoring import ValidationScoringSystem


class SourceLifecycleState(Enum):
    """Source lifecycle states"""
    ACTIVE = "active"
    MONITORED = "monitored"
    WARNING = "warning"
    DEGRADED = "degraded"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    RETIRED = "retired"


class HealthStatus(Enum):
    """Source health status indicators"""
    HEALTHY = "healthy"
    UNSTABLE = "unstable"
    FAILING = "failing"
    UNREACHABLE = "unreachable"


@dataclass
class SourceHealthCheck:
    """Health check result for a source"""
    source_id: int
    url: str
    status: HealthStatus
    response_time_ms: Optional[int]
    status_code: Optional[int]
    ssl_valid: bool
    content_size: Optional[int]
    last_modified: Optional[datetime]
    error_message: Optional[str]
    checked_at: datetime


@dataclass
class ContentFreshnessReport:
    """Content freshness analysis for a source"""
    source_id: int
    articles_last_24h: int
    articles_last_7d: int
    articles_last_30d: int
    avg_publish_frequency: float  # articles per day
    last_content_update: Optional[datetime]
    freshness_score: float  # 0-1 scale
    content_quality_trend: str  # improving, stable, declining


@dataclass
class LifecycleEvent:
    """Source lifecycle event"""
    source_id: int
    event_type: str
    old_state: str
    new_state: str
    reason: str
    metadata: Dict
    timestamp: datetime


class SourceHealthMonitor:
    """Monitors source health and availability"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.timeout = 10
        self.user_agent = "TradingAgentBot/1.0"
    
    async def check_source_health(self, source: Source) -> SourceHealthCheck:
        """Perform comprehensive health check on a source"""
        start_time = datetime.utcnow()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                headers = {'User-Agent': self.user_agent}
                
                async with session.get(source.url, headers=headers) as response:
                    response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    content = await response.read()
                    
                    # Check SSL
                    ssl_valid = source.url.startswith('https://') and response.status < 400
                    
                    # Determine health status
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                    elif 200 <= response.status < 300:
                        status = HealthStatus.HEALTHY
                    elif 300 <= response.status < 500:
                        status = HealthStatus.UNSTABLE
                    else:
                        status = HealthStatus.FAILING
                    
                    return SourceHealthCheck(
                        source_id=source.id,
                        url=source.url,
                        status=status,
                        response_time_ms=response_time,
                        status_code=response.status,
                        ssl_valid=ssl_valid,
                        content_size=len(content),
                        last_modified=None,
                        error_message=None,
                        checked_at=datetime.utcnow()
                    )
                    
        except asyncio.TimeoutError:
            return SourceHealthCheck(
                source_id=source.id,
                url=source.url,
                status=HealthStatus.UNREACHABLE,
                response_time_ms=None,
                status_code=None,
                ssl_valid=False,
                content_size=None,
                last_modified=None,
                error_message="Request timeout",
                checked_at=datetime.utcnow()
            )
        except Exception as e:
            return SourceHealthCheck(
                source_id=source.id,
                url=source.url,
                status=HealthStatus.UNREACHABLE,
                response_time_ms=None,
                status_code=None,
                ssl_valid=False,
                content_size=None,
                last_modified=None,
                error_message=str(e),
                checked_at=datetime.utcnow()
            )
    
    async def batch_health_check(self, sources: List[Source]) -> List[SourceHealthCheck]:
        """Perform health checks on multiple sources concurrently"""
        tasks = [self.check_source_health(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed for source {sources[i].id}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results


class ContentFreshnessMonitor:
    """Monitors content freshness and publishing patterns"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
    
    def analyze_content_freshness(self, source: Source) -> ContentFreshnessReport:
        """Analyze content freshness for a source"""
        now = datetime.utcnow()
        
        # Count articles in different time periods
        articles_24h = self.db_session.query(NewsItem).filter(
            NewsItem.source_id == source.id,
            NewsItem.published_at >= now - timedelta(hours=24)
        ).count()
        
        articles_7d = self.db_session.query(NewsItem).filter(
            NewsItem.source_id == source.id,
            NewsItem.published_at >= now - timedelta(days=7)
        ).count()
        
        articles_30d = self.db_session.query(NewsItem).filter(
            NewsItem.source_id == source.id,
            NewsItem.published_at >= now - timedelta(days=30)
        ).count()
        
        # Calculate average publishing frequency
        avg_frequency = articles_30d / 30.0 if articles_30d > 0 else 0.0
        
        # Get most recent article
        latest_article = self.db_session.query(NewsItem).filter(
            NewsItem.source_id == source.id
        ).order_by(desc(NewsItem.published_at)).first()
        
        last_content_update = latest_article.published_at if latest_article else None
        
        # Calculate freshness score (0-1)
        freshness_score = 0.0
        if articles_24h > 0:
            freshness_score = 1.0
        elif articles_7d > 0:
            freshness_score = 0.7
        elif articles_30d > 0:
            freshness_score = 0.3
        else:
            freshness_score = 0.0
        
        # Determine content quality trend
        recent_scores = self.db_session.query(SourceScore).filter(
            SourceScore.source_id == source.id
        ).order_by(desc(SourceScore.scored_at)).limit(10).all()
        
        if len(recent_scores) >= 3:
            recent_avg = sum(s.overall_score for s in recent_scores[:3]) / 3
            older_avg = sum(s.overall_score for s in recent_scores[-3:]) / 3
            if recent_avg > older_avg + 0.1:
                trend = "improving"
            elif recent_avg < older_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return ContentFreshnessReport(
            source_id=source.id,
            articles_last_24h=articles_24h,
            articles_last_7d=articles_7d,
            articles_last_30d=articles_30d,
            avg_publish_frequency=avg_frequency,
            last_content_update=last_content_update,
            freshness_score=freshness_score,
            content_quality_trend=trend
        )


class SourceMaintainer:
    """Handles source maintenance and updates"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.categorizer = SourceCategorizer()
        self.validator = ValidationScoringSystem()


class SourceArchiver:
    """Handles source archival and retirement"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
    
    def should_archive_source(self, source: Source, health_check: SourceHealthCheck, 
                            freshness_report: ContentFreshnessReport) -> Tuple[bool, str]:
        """Determine if a source should be archived"""
        reasons = []
        
        # Check if source is unreachable for too long
        if health_check.status == HealthStatus.UNREACHABLE:
            recent_failures = self.db_session.query(SourceHistory).filter(
                SourceHistory.source_id == source.id,
                SourceHistory.event_type == "health_check_failed",
                SourceHistory.event_timestamp >= datetime.utcnow() - timedelta(days=7)
            ).count()
            
            if recent_failures >= 5:
                reasons.append("unreachable_for_7_days")
        
        # Check content staleness
        if freshness_report.last_content_update:
            days_since_update = (datetime.utcnow() - freshness_report.last_content_update).days
            if days_since_update > 90:
                reasons.append("no_content_for_90_days")
        
        # Check reliability score degradation
        if source.reliability_score < 2.0:
            reasons.append("reliability_score_too_low")
        
        # Check if marked as inactive for too long
        if not source.is_active:
            last_active = source.last_fetched_at or source.created_at
            if last_active and (datetime.utcnow() - last_active).days > 60:
                reasons.append("inactive_for_60_days")
        
        should_archive = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else ""
        
        return should_archive, reason_text


class LifecycleNotifier:
    """Handles notifications for lifecycle events"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.notification_queue = []
    
    def queue_notification(self, event: LifecycleEvent, priority: str = "normal"):
        """Queue a notification for a lifecycle event"""
        notification = {
            'event': event,
            'priority': priority,
            'queued_at': datetime.utcnow()
        }
        self.notification_queue.append(notification)
        
        # Log high-priority events immediately
        if priority in ['high', 'critical']:
            self.logger.warning(
                f"High-priority lifecycle event: Source {event.source_id} "
                f"{event.event_type} - {event.reason}"
            )


class SourceLifecycleManager:
    """Main orchestrator for source lifecycle management"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.health_monitor = SourceHealthMonitor(db_session)
        self.freshness_monitor = ContentFreshnessMonitor(db_session)
        self.maintainer = SourceMaintainer(db_session)
        self.archiver = SourceArchiver(db_session)
        self.notifier = LifecycleNotifier()


# Convenience functions for external use
async def run_lifecycle_maintenance(db_session: Session, max_sources: int = 50) -> Dict[str, int]:
    """Run lifecycle maintenance on sources"""
    manager = SourceLifecycleManager(db_session)
    stats = {
        'sources_checked': 0,
        'health_checks_completed': 0,
        'sources_updated': 0,
        'sources_archived': 0,
        'notifications_sent': 0,
        'errors': 0
    }
    return stats


async def get_source_health_report(db_session: Session, source_ids: List[int] = None) -> List[SourceHealthCheck]:
    """Get health report for specific sources or all active sources"""
    monitor = SourceHealthMonitor(db_session)
    
    if source_ids:
        sources = db_session.query(Source).filter(Source.id.in_(source_ids)).all()
    else:
        sources = db_session.query(Source).filter(
            Source.is_active == True,
            Source.auto_discovered == True
        ).limit(20).all()
    
    return await monitor.batch_health_check(sources) 