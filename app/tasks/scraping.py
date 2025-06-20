"""
Scraping Tasks for Background Processing
Handles web scraping with retry logic and source management
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
from sqlalchemy import or_

from app.tasks.celery_app import celery_app
from app.database.connection import get_db
from app.models.news import Source
from app.scrapers.manager import ScraperManager
from app.services.logging_service import get_logger

logger = get_logger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def scrape_high_priority(self):
    """Scrape high-priority sources (exchange feeds, SEBI/RBI)"""
    try:
        db = next(get_db())
        logger.info("Starting high-priority scraping task")
        
        # Get high-priority sources that need updating
        sources = db.query(Source).filter(
            Source.type.in_(['official', 'exchange', 'regulatory']),
            or_(
                Source.last_checked.is_(None),
                Source.last_checked < datetime.now() - timedelta(minutes=1)
            ),
            Source.is_active == True
        ).all()
        
        if not sources:
            return {'timestamp': datetime.now().isoformat(), 'sources_count': 0, 'status': 'no_sources'}
        
        scraper_manager = ScraperManager(db)
        results = []
        successful_scrapes = 0
        failed_scrapes = 0
        
        for source in sources:
            try:
                result = scraper_manager.scrape_source(source.id)
                if result and result.get('success'):
                    results.append({
                        'source_id': source.id,
                        'source_name': source.name,
                        'url': source.url,
                        'status': 'success',
                        'items': result.get('items_count', 0)
                    })
                    successful_scrapes += 1
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                    results.append({
                        'source_id': source.id,
                        'source_name': source.name,
                        'url': source.url,
                        'status': 'error',
                        'error': error_msg
                    })
                    failed_scrapes += 1
            except Exception as e:
                results.append({
                    'source_id': source.id,
                    'source_name': source.name,
                    'url': source.url,
                    'status': 'error',
                    'error': str(e)
                })
                failed_scrapes += 1
        
        db.close()
        return {
            'timestamp': datetime.now().isoformat(),
            'sources_count': len(sources),
            'successful_scrapes': successful_scrapes,
            'failed_scrapes': failed_scrapes,
            'results': results,
            'status': 'completed'
        }
        
    except Exception as exc:
        logger.error(f"High-priority scraping task failed: {exc}")
        raise self.retry(exc=exc, countdown=min(60 * (2 ** self.request.retries), 300))

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def scrape_general_sources(self):
    """Scrape general sources (news sites, social media)"""
    try:
        db = next(get_db())
        logger.info("Starting general scraping task")
        
        sources = db.query(Source).filter(
            Source.type.in_(['news', 'social', 'general']),
            or_(
                Source.last_checked.is_(None),
                Source.last_checked < datetime.now() - timedelta(minutes=30)
            ),
            Source.is_active == True
        ).all()
        
        if not sources:
            return {'timestamp': datetime.now().isoformat(), 'sources_count': 0, 'status': 'no_sources'}
        
        scraper_manager = ScraperManager(db)
        results = []
        successful_scrapes = 0
        failed_scrapes = 0
        
        for source in sources:
            try:
                result = scraper_manager.scrape_source(source.id)
                if result and result.get('success'):
                    results.append({
                        'source_id': source.id,
                        'source_name': source.name,
                        'url': source.url,
                        'status': 'success',
                        'items': result.get('items_count', 0)
                    })
                    successful_scrapes += 1
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                    results.append({
                        'source_id': source.id,
                        'source_name': source.name,
                        'url': source.url,
                        'status': 'error',
                        'error': error_msg
                    })
                    failed_scrapes += 1
            except Exception as e:
                results.append({
                    'source_id': source.id,
                    'source_name': source.name,
                    'url': source.url,
                    'status': 'error',
                    'error': str(e)
                })
                failed_scrapes += 1
        
        db.close()
        return {
            'timestamp': datetime.now().isoformat(),
            'sources_count': len(sources),
            'successful_scrapes': successful_scrapes,
            'failed_scrapes': failed_scrapes,
            'results': results,
            'status': 'completed'
        }
        
    except Exception as exc:
        logger.error(f"General scraping task failed: {exc}")
        raise self.retry(exc=exc, countdown=min(300 * (2 ** self.request.retries), 900))