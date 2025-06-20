"""
API endpoints for managing data scrapers.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
import logging
from app.scrapers.manager import ScraperManager, scraper_registry
from app.models import User, Source, NewsItem
from app.scrapers.validation import ContentValidator, BatchValidator
from app.scrapers.base import ScrapingStatus
from app.database import get_db
from app.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scrapers", tags=["scrapers"])


@router.get("/status", response_model=Dict[str, Any])
async def get_scraper_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get overall scraper system status"""
    try:
        manager = ScraperManager()
        status = await manager.get_scraping_stats()
        
        return {
            "status": "operational",
            "registered_scrapers": len(scraper_registry.list_types()),
            "scraper_types": scraper_registry.list_types(),
            "system_metrics": status
        }
    except Exception as e:
        logger.error(f"Error getting scraper status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scraper status")


@router.get("/types", response_model=List[str])
async def get_scraper_types(current_user: User = Depends(get_current_user)):
    """Get list of available scraper types"""
    return scraper_registry.list_types()


@router.post("/run/scheduled", response_model=Dict[str, Any])
async def run_scheduled_scraping(
    background_tasks: BackgroundTasks,
    validate: bool = Query(True, description="Enable data validation"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run scheduled scraping for all active sources"""
    try:
        manager = ScraperManager()
        
        # Run scraping in background
        background_tasks.add_task(
            manager.run_scheduled_scraping
        )
        
        return {
            "message": "Scheduled scraping started",
            "validation_enabled": validate,
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Error starting scheduled scraping: {e}")
        raise HTTPException(status_code=500, detail="Failed to start scheduled scraping")


@router.post("/run/source/{source_id}", response_model=Dict[str, Any])
async def run_source_scraping(
    source_id: int,
    validate: bool = Query(True, description="Enable data validation"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run scraping for a specific source"""
    try:
        # Get source
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        manager = ScraperManager()
        result = await manager.scrape_source_by_id(source_id)
        
        if result.status == ScrapingStatus.SUCCESS:
            return {
                "message": f"Scraping completed for source {source_id}",
                "source_url": source.url,
                "items_collected": result.items_stored,
                "validation_enabled": validate,
                "validation_summary": result.metadata.get("validation_summary") if result.metadata else None,
                "status": "completed",
                "errors": result.errors,
                "warnings": result.warnings
            }
        else:
            raise HTTPException(status_code=500, detail=f"Scraping failed: {', '.join(result.errors)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running source scraping: {e}")
        raise HTTPException(status_code=500, detail="Failed to run source scraping")


@router.post("/test", response_model=Dict[str, Any])
async def test_scraper(
    source_id: int,
    limit: int = Query(5, description="Number of items to test"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Test a scraper without saving data to database"""
    try:
        # Get source
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        manager = ScraperManager(db)
        scraper_class = scraper_registry.get_scraper(source.type)
        
        if not scraper_class:
            raise HTTPException(
                status_code=400, 
                detail=f"No scraper available for source type: {source.type}"
            )
        
        # Create scraper and test
        scraper = scraper_class(source, db)
        test_result = await scraper.test_extraction(limit=limit)
        
        return {
            "message": f"Test completed for source {source_id}",
            "source_url": source.url,
            "source_type": source.type,
            "test_items": test_result.get("items", []),
            "items_found": len(test_result.get("items", [])),
            "extraction_success": test_result.get("success", False),
            "errors": test_result.get("errors", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing scraper: {e}")
        raise HTTPException(status_code=500, detail="Failed to test scraper")

# NEW VALIDATION ENDPOINTS

@router.post("/validate/test-item", response_model=Dict[str, Any])
async def validate_test_item(
    item_data: Dict[str, Any],
    source_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Validate a test scraped item"""
    try:
        # Get source
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        # Validate the item
        validator = ContentValidator(db)
        report = await validator.validate_scraped_item(item_data, source)
        
        return {
            "is_valid": report.is_valid,
            "validation_score": report.score,
            "errors": [
                {
                    "level": result.level.value,
                    "field": result.field,
                    "message": result.message,
                    "value": result.value,
                    "suggestion": result.suggestion
                }
                for result in report.errors
            ],
            "warnings": [
                {
                    "level": result.level.value,
                    "field": result.field,
                    "message": result.message,
                    "value": result.value,
                    "suggestion": result.suggestion
                }
                for result in report.warnings
            ],
            "info": [
                {
                    "level": result.level.value,
                    "field": result.field,
                    "message": result.message,
                    "value": result.value,
                    "suggestion": result.suggestion
                }
                for result in report.info
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating test item: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate test item")

@router.get("/validate/recent-items/{source_id}")
async def get_recent_validation_results(
    source_id: int,
    limit: int = Query(10, description="Number of recent items to check"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get validation results for recent items from a source"""
    try:
        # Get source
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        # Get recent items
        recent_items = db.query(NewsItem)\
            .filter(NewsItem.source_id == source_id)\
            .order_by(NewsItem.scraped_at.desc())\
            .limit(limit)\
            .all()
        
        if not recent_items:
            return {
                "message": "No recent items found for this source",
                "source_id": source_id,
                "items_checked": 0,
                "validation_results": []
            }
        
        # Validate each item
        validator = ContentValidator(db)
        validation_results = []
        
        for item in recent_items:
            item_data = {
                "title": item.title,
                "content": item.content,
                "url": item.url,
                "published_at": item.published_at
            }
            
            report = await validator.validate_scraped_item(item_data, source)
            
            validation_results.append({
                "item_id": item.id,
                "title": item.title[:100] + "..." if len(item.title) > 100 else item.title,
                "scraped_at": item.scraped_at.isoformat(),
                "is_valid": report.is_valid,
                "validation_score": report.score,
                "error_count": len(report.errors),
                "warning_count": len(report.warnings),
                "issues": [
                    {
                        "level": result.level.value,
                        "field": result.field,
                        "message": result.message
                    }
                    for result in (report.errors + report.warnings)
                ]
            })
        
        return {
            "message": f"Validation completed for {len(recent_items)} recent items",
            "source_id": source_id,
            "source_url": source.url,
            "items_checked": len(recent_items),
            "validation_results": validation_results,
            "summary": {
                "valid_items": sum(1 for r in validation_results if r["is_valid"]),
                "invalid_items": sum(1 for r in validation_results if not r["is_valid"]),
                "average_score": sum(r["validation_score"] for r in validation_results) / len(validation_results),
                "total_errors": sum(r["error_count"] for r in validation_results),
                "total_warnings": sum(r["warning_count"] for r in validation_results)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting validation results: {e}")
        raise HTTPException(status_code=500, detail="Failed to get validation results")

@router.post("/validate/batch")
async def validate_batch_items(
    source_id: int,
    items: List[Dict[str, Any]],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Validate a batch of scraped items"""
    try:
        # Get source
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        if not items:
            raise HTTPException(status_code=400, detail="No items provided for validation")
        
        if len(items) > 100:
            raise HTTPException(status_code=400, detail="Batch size too large (max 100 items)")
        
        # Validate batch
        batch_validator = BatchValidator(db)
        reports = await batch_validator.validate_batch(items, source)
        summary = batch_validator.get_validation_summary(reports)
        
        return {
            "message": f"Batch validation completed for {len(items)} items",
            "source_id": source_id,
            "items_processed": len(items),
            "validation_summary": summary,
            "detailed_results": [
                {
                    "item_index": i,
                    "is_valid": report.is_valid,
                    "validation_score": report.score,
                    "error_count": len(report.errors),
                    "warning_count": len(report.warnings),
                    "key_issues": [
                        f"{result.field}: {result.message}"
                        for result in (report.errors + report.warnings)[:3]  # Show top 3 issues
                    ]
                }
                for i, report in enumerate(reports)
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating batch items: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate batch items")

@router.get("/validate/stats")
async def get_validation_stats(
    days: int = Query(7, description="Number of days to analyze"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get validation statistics across all sources"""
    try:
        from datetime import timedelta
        
        # Get recent items for analysis
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_items = db.query(NewsItem)\
            .filter(NewsItem.scraped_at >= cutoff_date)\
            .all()
        
        if not recent_items:
            return {
                "message": f"No items found in the last {days} days",
                "days_analyzed": days,
                "total_items": 0,
                "validation_stats": {}
            }
        
        # Group by source and analyze
        source_stats = {}
        validator = ContentValidator(db)
        
        for item in recent_items:
            source = item.source
            if source.id not in source_stats:
                source_stats[source.id] = {
                    "source_url": source.url,
                    "source_type": source.type,
                    "items": [],
                    "total_items": 0,
                    "valid_items": 0,
                    "total_score": 0,
                    "error_count": 0,
                    "warning_count": 0
                }
            
            # Quick validation for stats
            item_data = {
                "title": item.title,
                "content": item.content,
                "url": item.url,
                "published_at": item.published_at
            }
            
            report = await validator.validate_scraped_item(item_data, source)
            
            stats = source_stats[source.id]
            stats["total_items"] += 1
            stats["total_score"] += report.score
            stats["error_count"] += len(report.errors)
            stats["warning_count"] += len(report.warnings)
            
            if report.is_valid:
                stats["valid_items"] += 1
        
        # Calculate aggregated stats
        total_items = sum(stats["total_items"] for stats in source_stats.values())
        total_valid = sum(stats["valid_items"] for stats in source_stats.values())
        
        return {
            "message": f"Validation stats for last {days} days",
            "days_analyzed": days,
            "total_items": total_items,
            "total_valid_items": total_valid,
            "overall_validation_rate": total_valid / max(total_items, 1),
            "sources_analyzed": len(source_stats),
            "source_breakdown": {
                str(source_id): {
                    **stats,
                    "validation_rate": stats["valid_items"] / max(stats["total_items"], 1),
                    "average_score": stats["total_score"] / max(stats["total_items"], 1),
                    "errors_per_item": stats["error_count"] / max(stats["total_items"], 1),
                    "warnings_per_item": stats["warning_count"] / max(stats["total_items"], 1)
                }
                for source_id, stats in source_stats.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting validation stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get validation statistics") 