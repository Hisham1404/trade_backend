"""
Source Manager Module
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.models import Source, Asset, NewsItem

class SourceManager:
    """Manages the lifecycle of discovered information sources"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
    
    async def get_sources_for_asset(self, asset_id: int) -> List[Source]:
        asset = self.db_session.query(Asset).get(asset_id)
        if not asset: return []
        return self.db_session.query(Source).filter(
            Source.auto_discovered == True,
            Source.is_active == True,
            Source.reliability_score >= 4.0
        ).order_by(Source.reliability_score.desc()).all()
    
    async def update_source_reliability(self, source_id: int, perf_data: Dict) -> bool:
        source = self.db_session.query(Source).get(source_id)
        if not source: return False
        
        rate = perf_data.get('successful_scrapes', 0) / max(perf_data.get('articles_scraped', 1), 1)
        relevance = sum(perf_data.get('relevance_scores', [])) / max(len(perf_data.get('relevance_scores', [])), 1)
        errors = perf_data.get('error_count', 0) / max(perf_data.get('articles_scraped', 1), 1)
        
        perf_score = (rate * 0.4) + (relevance * 0.4) - (errors * 0.2)
        
        old_score = float(source.reliability_score)
        new_score = (old_score * 0.8) + ((old_score + perf_score) * 0.2)
        source.reliability_score = max(0.0, min(new_score, 10.0))
        source.last_checked = datetime.utcnow()
        
        try:
            self.db_session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error updating source {source_id}: {e}")
            self.db_session.rollback()
            return False

    async def prune_unreliable_sources(self, threshold: float = 3.0, days_inactive: int = 30):
        cutoff = datetime.utcnow() - timedelta(days=days_inactive)
        unreliable = self.db_session.query(Source).filter(
            Source.auto_discovered == True,
            or_(Source.reliability_score < threshold, Source.last_checked < cutoff)
        ).all()
        
        for source in unreliable:
            self.db_session.delete(source)
        
        try:
            self.db_session.commit()
            return len(unreliable)
        except Exception as e:
            self.logger.error(f"Error pruning sources: {e}")
            self.db_session.rollback()
            return 0 