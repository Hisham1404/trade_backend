"""
Scraper manager for orchestrating data collection operations.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type, Any, Union
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Source, NewsItem
from .base import BaseScraper, ScrapingResult, ScrapingStatus, ScrapingError
from .validation import ContentValidator, BatchValidator


logger = logging.getLogger(__name__)


class ScraperRegistry:
    """Registry for managing available scrapers."""
    
    def __init__(self):
        self.scrapers: Dict[str, Type[BaseScraper]] = {}
        
    def register(self, source_type: str, scraper_class: Type[BaseScraper]):
        """Register a scraper for a specific source type."""
        self.scrapers[source_type] = scraper_class
        logger.info(f"Registered scraper: {source_type} -> {scraper_class.__name__}")
        
    def get_scraper(self, source_type: str) -> Optional[Type[BaseScraper]]:
        """Get scraper class for a source type."""
        return self.scrapers.get(source_type)
        
    def list_types(self) -> List[str]:
        """List all registered source types."""
        return list(self.scrapers.keys())


# Global registry instance
scraper_registry = ScraperRegistry()


class ScraperManager:
    """
    Manager for coordinating scraping operations across multiple sources.
    Handles database connections, validation, and parallel execution.
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.registry = scraper_registry
        self.max_concurrent = max_concurrent
        self.background_task: Optional[asyncio.Task] = None
        self._background_running = False
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
    @contextmanager
    def get_db_session(self):
        """Get database session with proper cleanup."""
        db = next(get_db())
        try:
            yield db
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
                
    def get_sources_due_for_check(
        self, 
        session: Session,
        min_interval_minutes: int = 30
    ) -> List[Source]:
        """
        Get sources that are due for checking.
        
        Args:
            session: Database session
            min_interval_minutes: Minimum interval between checks
            
        Returns:
            List of sources to scrape
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=min_interval_minutes)
        
        sources = session.query(Source).filter(
            and_(
                Source.is_active == True,
                Source.last_checked < cutoff_time
            )
        ).all()
        
        logger.info(f"Found {len(sources)} sources due for checking")
        return sources
        
    async def scrape_source(self, source: Source, session: Session, validate_data: bool = True) -> ScrapingResult:
        """
        Scrape a single source.
        
        Args:
            source: Source to scrape
            session: Database session
            validate_data: Whether to validate scraped data
            
        Returns:
            ScrapingResult
        """
        scraper_class = self.registry.get_scraper(source.source_type)
        
        if not scraper_class:
            logger.error(f"No scraper registered for source type: {source.source_type}")
            return ScrapingResult(
                status=ScrapingStatus.FAILED,
                errors=[f"No scraper available for type: {source.source_type}"]
            )
            
        try:
            # TODO: Implement proper scraper with sync session
            # For now, return a placeholder result
            logger.info(f"Simulating scrape of source {source.name}")
            
            # Update source last_checked timestamp
            source.last_checked = datetime.utcnow()
            session.commit()
            
            return ScrapingResult(
                status=ScrapingStatus.SUCCESS,
                items_found=0,
                items_stored=0,
                metadata={"simulated": True}
            )
            
        except Exception as e:
            logger.error(f"Failed to scrape source {source.name}: {e}", exc_info=True)
            return ScrapingResult(
                status=ScrapingStatus.FAILED,
                errors=[f"Scraper execution failed: {str(e)}"]
            )
            
    async def scrape_sources_parallel(
        self, 
        sources: List[Source],
        max_concurrent: Optional[int] = None
    ) -> List[ScrapingResult]:
        """
        Scrape multiple sources in parallel.
        
        Args:
            sources: List of sources to scrape
            max_concurrent: Override max concurrent scrapers
            
        Returns:
            List of scraping results
        """
        if not sources:
            return []
            
        max_concurrent = max_concurrent or self.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(source: Source) -> ScrapingResult:
            async with semaphore:
                with self.get_db_session() as session:
                    return await self.scrape_source(source, session, validate_data=True)
                    
        tasks = [scrape_with_semaphore(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Scraping task failed for {sources[i].name}: {result}")
                processed_results.append(ScrapingResult(
                    status=ScrapingStatus.FAILED,
                    errors=[f"Task execution failed: {str(result)}"]
                ))
            else:
                processed_results.append(result)
                
        return processed_results
        
    async def run_scheduled_scraping(
        self,
        min_interval_minutes: int = 30,
        max_concurrent: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run scraping for all sources due for checking.
        
        Args:
            min_interval_minutes: Minimum interval between checks
            max_concurrent: Override max concurrent scrapers
            
        Returns:
            Summary of scraping operation
        """
        start_time = datetime.utcnow()
        
        try:
            with self.get_db_session() as session:
                sources = self.get_sources_due_for_check(session, min_interval_minutes)
                
            if not sources:
                logger.info("No sources due for checking")
                return {
                    "status": "completed",
                    "sources_checked": 0,
                    "total_items": 0,
                    "duration": 0.0,
                    "errors": []
                }
                
            logger.info(f"Starting scheduled scraping for {len(sources)} sources")
            
            results = await self.scrape_sources_parallel(sources, max_concurrent)
            
            # Compile summary
            total_items = sum(result.items_stored for result in results)
            total_errors = sum(len(result.errors) for result in results)
            successful_sources = sum(1 for result in results if result.status == ScrapingStatus.SUCCESS)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            summary = {
                "status": "completed",
                "sources_checked": len(sources),
                "successful_sources": successful_sources,
                "total_items": total_items,
                "total_errors": total_errors,
                "duration": duration,
                "results": [
                    {
                        "source": sources[i].name,
                        "status": result.status.value,
                        "items_found": result.items_found,
                        "items_stored": result.items_stored,
                        "errors": result.errors,
                        "warnings": result.warnings,
                        "duration": result.duration
                    }
                    for i, result in enumerate(results)
                ]
            }
            
            logger.info(
                f"Scheduled scraping completed: "
                f"{successful_sources}/{len(sources)} sources successful, "
                f"{total_items} items collected in {duration:.2f}s"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Scheduled scraping failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "duration": (datetime.utcnow() - start_time).total_seconds()
            }
            
    async def scrape_source_by_id(self, source_id: int) -> ScrapingResult:
        """
        Scrape a specific source by ID.
        
        Args:
            source_id: Source ID to scrape
            
        Returns:
            ScrapingResult
        """
        with self.get_db_session() as session:
            source = session.query(Source).filter(Source.id == source_id).first()
            
            if not source:
                return ScrapingResult(
                    status=ScrapingStatus.FAILED,
                    errors=[f"Source with ID {source_id} not found"]
                )
                
            return await self.scrape_source(source, session)
            
    async def scrape_source_by_name(self, source_name: str) -> ScrapingResult:
        """
        Scrape a specific source by name.
        
        Args:
            source_name: Source name to scrape
            
        Returns:
            ScrapingResult
        """
        with self.get_db_session() as session:
            source = session.query(Source).filter(Source.name == source_name).first()
            
            if not source:
                return ScrapingResult(
                    status=ScrapingStatus.FAILED,
                    errors=[f"Source '{source_name}' not found"]
                )
                
            return await self.scrape_source(source, session)
            
    async def get_scraping_stats(self) -> Dict[str, Any]:
        """
        Get statistics about scraping operations.
        
        Returns:
            Statistics dictionary
        """
        with self.get_db_session() as session:
            # Get source counts by type and status
            sources = session.query(Source).all()
            
            stats = {
                "total_sources": len(sources),
                "active_sources": sum(1 for s in sources if s.is_active),
                "inactive_sources": sum(1 for s in sources if not s.is_active),
                "sources_by_type": {},
                "registered_scrapers": self.registry.list_types(),
                "last_check_distribution": {
                    "last_hour": 0,
                    "last_day": 0,
                    "last_week": 0,
                    "older": 0
                }
            }
            
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            week_ago = now - timedelta(weeks=1)
            
            for source in sources:
                # Count by type
                source_type = source.source_type
                if source_type not in stats["sources_by_type"]:
                    stats["sources_by_type"][source_type] = 0
                stats["sources_by_type"][source_type] += 1
                
                # Count by last check time
                if source.last_checked and source.last_checked > hour_ago:
                    stats["last_check_distribution"]["last_hour"] += 1
                elif source.last_checked and source.last_checked > day_ago:
                    stats["last_check_distribution"]["last_day"] += 1
                elif source.last_checked and source.last_checked > week_ago:
                    stats["last_check_distribution"]["last_week"] += 1
                else:
                    stats["last_check_distribution"]["older"] += 1
                    
            return stats
    
    async def _validate_scraped_items(self, items: List[Dict[str, Any]], source: Source, session: Session) -> Dict[str, Any]:
        """
        Validate a list of scraped items and return summary.
        
        Args:
            items: List of scraped items
            source: Source they came from
            session: Database session
            
        Returns:
            Validation summary
        """
        try:
            # Create validator with session
            validator = ContentValidator(session)
            batch_validator = BatchValidator(session)
            
            # Validate all items
            reports = await batch_validator.validate_batch(items, source)
            summary = batch_validator.get_validation_summary(reports)
            
            logger.info(f"Validated {len(items)} items from {source.name}: {summary['validation_rate']:.2%} valid")
            
            return summary
            
        except Exception as e:
            logger.error(f"Validation failed for source {source.name}: {e}")
            return {
                "validation_error": str(e),
                "items_validated": 0,
                "validation_rate": 0.0
            }
            
    def start_background_scraping(
        self,
        interval_minutes: int = 30,
        min_check_interval: int = 30
    ):
        """
        Start background scraping task.
        
        Args:
            interval_minutes: How often to run scraping
            min_check_interval: Minimum interval between source checks
        """
        if self._background_running:
            logger.warning("Background scraping is already running")
            return
            
        self._background_running = True
        
        async def background_task():
            while self._background_running:
                try:
                    await self.run_scheduled_scraping(min_check_interval)
                except Exception as e:
                    logger.error(f"Background scraping error: {e}", exc_info=True)
                    
                # Wait for next iteration
                await asyncio.sleep(interval_minutes * 60)
                
        self.background_task = asyncio.create_task(background_task())
        logger.info(f"Started background scraping with {interval_minutes} minute intervals")
        
    async def stop_background_scraping(self):
        """Stop background scraping task."""
        self._background_running = False
        
        if self.background_task:
            self.background_task.cancel()
            
        if self.background_task:
            await self.background_task
            
        logger.info("Stopped background scraping")


# Global manager instance
scraper_manager = ScraperManager()


def register_scraper(source_type: str):
    """Decorator to register a scraper class."""
    def decorator(scraper_class: Type[BaseScraper]):
        scraper_registry.register(source_type, scraper_class)
        return scraper_class
    return decorator 