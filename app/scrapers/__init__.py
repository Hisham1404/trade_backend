"""
Data Collection and Scraping Layer

This module provides comprehensive web scraping capabilities with:
- Base scraper framework with rate limiting and error handling
- Specialized scrapers for different source types (news sites, RSS feeds, regulatory sites)
- Validation pipeline for data quality assurance
- Centralized management system with scheduling and monitoring
"""

from .base import BaseScraper, ScrapingResult, ScrapingStatus, ScrapedItem, ScrapingError
from .manager import ScraperManager, ScraperRegistry, register_scraper
from .validation import ContentValidator, BatchValidator, ValidationLevel, ValidationResult, ContentValidationReport
from .news_sites import EconomicTimesScraper, MoneycontrolScraper
from .feeds import RSSFeedScraper, AtomFeedScraper, EconomicTimesRSScraper, MoneycontrolRSScraper, RBIRSScraper, SEBIRSScraper

# Import scrapers to trigger registration
from . import news_sites, feeds

# Export key components for external use
__all__ = [
    # Base framework
    "BaseScraper",
    "ScrapingResult", 
    "ScrapingStatus",
    "ScrapedItem",
    "ScrapingError",
    
    # Management
    "ScraperManager",
    "ScraperRegistry",
    "register_scraper",
    
    # Validation
    "ContentValidator",
    "BatchValidator", 
    "ValidationLevel",
    "ValidationResult",
    "ContentValidationReport",
    
    # Scrapers
    "EconomicTimesScraper",
    "MoneycontrolScraper",
    "RSSFeedScraper",
    "AtomFeedScraper",
    "EconomicTimesRSScraper",
    "MoneycontrolRSScraper", 
    "RBIRSScraper",
    "SEBIRSScraper"
] 