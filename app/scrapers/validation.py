from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
import logging
import hashlib
import urllib.parse
from difflib import SequenceMatcher
import asyncio

from app.database.connection import get_db
from app.models import NewsItem, Source
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    level: ValidationLevel
    field: str
    message: str
    value: Any = None
    suggestion: Optional[str] = None

@dataclass
class ContentValidationReport:
    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[ValidationResult] = field(default_factory=list)
    warnings: List[ValidationResult] = field(default_factory=list)
    info: List[ValidationResult] = field(default_factory=list)
    
    def add_result(self, result: ValidationResult):
        if result.level == ValidationLevel.ERROR:
            self.errors.append(result)
        elif result.level == ValidationLevel.WARNING:
            self.warnings.append(result)
        else:
            self.info.append(result)
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

class ContentValidator:
    """Comprehensive validation system for scraped content"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        
        # Financial terms for relevance checking
        self.financial_terms = {
            'high_relevance': ['stock', 'market', 'trading', 'nse', 'bse', 'sebi', 'rbi', 
                              'investment', 'portfolio', 'equity', 'share', 'index', 'sensex', 
                              'nifty', 'mutual fund', 'ipo', 'earnings', 'dividend', 'revenue'],
            'medium_relevance': ['economy', 'finance', 'banking', 'company', 'business', 
                                'corporate', 'profit', 'loss', 'growth', 'quarterly', 'annual'],
            'low_relevance': ['rupee', 'inflation', 'gdp', 'policy', 'government', 'regulation']
        }
        
        # Common spam indicators
        self.spam_patterns = [
            r'(?i)click here for.*',
            r'(?i)limited time offer',
            r'(?i)guaranteed returns',
            r'(?i)risk-free.*investment',
            r'(?i)make money fast',
            r'(?i)forex.*signals',
            r'(?i)crypto.*pump',
            r'(?i)sure.*profit'
        ]
        
        # URL validation patterns
        self.valid_url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    async def validate_scraped_item(self, item: Dict[str, Any], source: Source) -> ContentValidationReport:
        """Validate a single scraped item comprehensively"""
        report = ContentValidationReport(is_valid=True, score=1.0)
        
        # Basic field validation
        await self._validate_required_fields(item, report)
        await self._validate_title(item.get('title'), report)
        await self._validate_content(item.get('content'), report)
        await self._validate_url(item.get('url'), report)
        await self._validate_dates(item, report)
        
        # Content quality analysis
        await self._analyze_content_quality(item, report)
        
        # Spam detection
        await self._detect_spam_content(item, report)
        
        # Relevance checking
        await self._check_financial_relevance(item, report)
        
        # Duplicate detection
        await self._check_for_duplicates(item, report)
        
        # Source-specific validation
        await self._validate_against_source(item, source, report)
        
        # Calculate final validation score
        report.score = self._calculate_validation_score(report)
        report.is_valid = report.score >= 0.6 and not report.has_errors
        
        return report
    
    async def _validate_required_fields(self, item: Dict[str, Any], report: ContentValidationReport):
        """Validate that all required fields are present and non-empty"""
        required_fields = ['title', 'content', 'url', 'published_at']
        
        for field in required_fields:
            if field not in item:
                report.add_result(ValidationResult(
                    level=ValidationLevel.ERROR,
                    field=field,
                    message=f"Required field '{field}' is missing",
                    suggestion=f"Ensure scraper extracts {field} from source"
                ))
            elif not item[field] or (isinstance(item[field], str) and not item[field].strip()):
                report.add_result(ValidationResult(
                    level=ValidationLevel.ERROR,
                    field=field,
                    message=f"Required field '{field}' is empty",
                    value=item[field]
                ))
    
    async def _validate_title(self, title: str, report: ContentValidationReport):
        """Validate title field"""
        if not title:
            return
        
        title = title.strip()
        
        # Length validation
        if len(title) < 10:
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="title",
                message="Title is too short (less than 10 characters)",
                value=title,
                suggestion="Check if title extraction is capturing complete text"
            ))
        elif len(title) > 200:
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="title",
                message="Title is unusually long (over 200 characters)",
                value=f"{title[:50]}...",
                suggestion="Verify title extraction isn't including extra content"
            ))
        
        # Check for placeholder or generic titles
        generic_patterns = [
            r'(?i)^(news|article|post|update|breaking)\s*$',
            r'(?i)^(untitled|no title|title).*',
            r'(?i)^(click here|read more).*'
        ]
        
        for pattern in generic_patterns:
            if re.match(pattern, title):
                report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    field="title",
                    message="Title appears to be a placeholder or generic text",
                    value=title
                ))
        
        # Check for excessive HTML tags (incomplete parsing)
        if '<' in title and '>' in title:
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="title",
                message="Title contains HTML tags - parsing may be incomplete",
                value=title,
                suggestion="Improve HTML parsing to extract clean text"
            ))
    
    async def _validate_content(self, content: str, report: ContentValidationReport):
        """Validate content field"""
        if not content:
            return
        
        content = content.strip()
        
        # Length validation
        if len(content) < 50:
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="content",
                message="Content is too short (less than 50 characters)",
                value=content,
                suggestion="Check if content extraction is capturing the full article"
            ))
        elif len(content) > 50000:
            report.add_result(ValidationResult(
                level=ValidationLevel.INFO,
                field="content",
                message="Content is very long (over 50,000 characters)",
                value=f"{content[:100]}...",
                suggestion="Consider if this includes navigation or other non-article content"
            ))
        
        # Check for excessive HTML tags
        html_tag_count = len(re.findall(r'<[^>]+>', content))
        if html_tag_count > 10:
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="content",
                message=f"Content contains {html_tag_count} HTML tags - parsing may be incomplete",
                suggestion="Improve HTML parsing to extract clean text"
            ))
        
        # Check for repeated phrases (poor extraction)
        words = content.split()
        if len(words) > 10:
            # Check for repeated sequences
            for i in range(len(words) - 5):
                phrase = ' '.join(words[i:i+5])
                remaining_text = ' '.join(words[i+5:])
                if phrase in remaining_text:
                    report.add_result(ValidationResult(
                        level=ValidationLevel.WARNING,
                        field="content",
                        message="Content contains repeated phrases - may indicate extraction issues",
                        value=phrase
                    ))
                    break
    
    async def _validate_url(self, url: str, report: ContentValidationReport):
        """Validate URL field"""
        if not url:
            return
        
        url = url.strip()
        
        # Basic URL format validation
        if not self.valid_url_pattern.match(url):
            report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                field="url",
                message="URL format is invalid",
                value=url,
                suggestion="Ensure URL includes proper protocol (http/https)"
            ))
            return
        
        # Parse URL for additional validation
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Check for suspicious domains
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
            if any(parsed.netloc.endswith(tld) for tld in suspicious_tlds):
                report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    field="url",
                    message="URL uses suspicious top-level domain",
                    value=parsed.netloc
                ))
            
            # Check for very long URLs (potential spam)
            if len(url) > 500:
                report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    field="url",
                    message="URL is unusually long",
                    value=f"{url[:100]}...",
                    suggestion="Long URLs may indicate tracking parameters or spam"
                ))
        
        except Exception as e:
            report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                field="url",
                message=f"Error parsing URL: {str(e)}",
                value=url
            ))
    
    async def _validate_dates(self, item: Dict[str, Any], report: ContentValidationReport):
        """Validate date fields"""
        published_at = item.get('published_at')
        
        if not published_at:
            return
        
        # Ensure published_at is a datetime object
        if isinstance(published_at, str):
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="published_at",
                message="Published date is string, should be datetime object",
                value=published_at,
                suggestion="Parse date string to datetime object during extraction"
            ))
            return
        
        if not isinstance(published_at, datetime):
            report.add_result(ValidationResult(
                level=ValidationLevel.ERROR,
                field="published_at",
                message="Published date is not a valid datetime object",
                value=str(published_at)
            ))
            return
        
        # Check if date is reasonable (not too far in future or past)
        now = datetime.now()
        if published_at > now + timedelta(hours=1):
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="published_at",
                message="Published date is in the future",
                value=published_at.isoformat(),
                suggestion="Check date parsing logic"
            ))
        elif published_at < now - timedelta(days=365*2):
            report.add_result(ValidationResult(
                level=ValidationLevel.INFO,
                field="published_at",
                message="Published date is more than 2 years old",
                value=published_at.isoformat(),
                suggestion="Consider if old articles are relevant for current analysis"
            ))
    
    async def _analyze_content_quality(self, item: Dict[str, Any], report: ContentValidationReport):
        """Analyze overall content quality"""
        title = item.get('title', '')
        content = item.get('content', '')
        
        if not title or not content:
            return
        
        # Calculate readability metrics
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(content.split())
        
        if sentences > 0:
            avg_sentence_length = words / sentences
            if avg_sentence_length > 50:
                report.add_result(ValidationResult(
                    level=ValidationLevel.INFO,
                    field="content",
                    message="Content has very long sentences (may be hard to process)",
                    value=f"Average: {avg_sentence_length:.1f} words per sentence"
                ))
        
        # Check for proper capitalization
        if title and title.islower():
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="title",
                message="Title is all lowercase",
                value=title,
                suggestion="Check if title extraction preserves proper capitalization"
            ))
        
        # Check content-to-title ratio
        if words > 0:
            title_words = len(title.split())
            content_title_ratio = words / max(title_words, 1)
            
            if content_title_ratio < 5:
                report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    field="content",
                    message="Content is very short compared to title",
                    value=f"Ratio: {content_title_ratio:.1f}",
                    suggestion="May be extracting summary instead of full article"
                ))
    
    async def _detect_spam_content(self, item: Dict[str, Any], report: ContentValidationReport):
        """Detect spam or promotional content"""
        title = item.get('title', '')
        content = item.get('content', '')
        combined_text = f"{title} {content}".lower()
        
        # Check against spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, combined_text):
                report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    field="content",
                    message="Content matches spam pattern",
                    value=pattern,
                    suggestion="Review content quality and source reliability"
                ))
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in title if c.isupper()) / max(len(title), 1)
        if caps_ratio > 0.5:
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="title",
                message="Title has excessive capitalization",
                value=f"Caps ratio: {caps_ratio:.2f}",
                suggestion="May indicate promotional or spam content"
            ))
        
        # Check for excessive exclamation marks
        exclamation_count = combined_text.count('!')
        if exclamation_count > 5:
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="content",
                message=f"Content has excessive exclamation marks ({exclamation_count})",
                suggestion="May indicate promotional content"
            ))
    
    async def _check_financial_relevance(self, item: Dict[str, Any], report: ContentValidationReport):
        """Check if content is relevant to financial topics"""
        title = item.get('title', '')
        content = item.get('content', '')
        combined_text = f"{title} {content}".lower()
        
        relevance_score = 0
        matched_terms = []
        
        # Check for financial terms
        for category, terms in self.financial_terms.items():
            weight = {'high_relevance': 3, 'medium_relevance': 2, 'low_relevance': 1}[category]
            for term in terms:
                if term in combined_text:
                    relevance_score += weight
                    matched_terms.append(term)
        
        # Normalize score (rough heuristic)
        max_possible_score = len(combined_text.split()) * 0.1  # Assume 10% of words could be financial
        normalized_score = min(relevance_score / max(max_possible_score, 1), 1.0)
        
        if normalized_score < 0.1:
            report.add_result(ValidationResult(
                level=ValidationLevel.WARNING,
                field="content",
                message="Content has low financial relevance",
                value=f"Score: {normalized_score:.2f}",
                suggestion="Verify this content is useful for financial analysis"
            ))
        elif normalized_score > 0.3:
            report.add_result(ValidationResult(
                level=ValidationLevel.INFO,
                field="content",
                message="Content has high financial relevance",
                value=f"Score: {normalized_score:.2f}, Terms: {', '.join(matched_terms[:5])}"
            ))
    
    async def _check_for_duplicates(self, item: Dict[str, Any], report: ContentValidationReport):
        """Check for duplicate content in database"""
        title = item.get('title', '')
        content = item.get('content', '')
        url = item.get('url', '')
        
        if not title and not content:
            return
        
        # Check for exact URL match
        if url:
            existing_url = self.db_session.query(NewsItem).filter(NewsItem.url == url).first()
            if existing_url:
                report.add_result(ValidationResult(
                    level=ValidationLevel.ERROR,
                    field="url",
                    message="URL already exists in database",
                    value=url,
                    suggestion="This is a duplicate article"
                ))
                return
        
        # Check for similar titles
        if title:
            # Create a simple hash for quick comparison
            title_hash = hashlib.md5(title.encode()).hexdigest()
            
            # Get recent articles for similarity comparison (last 7 days)
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_articles = self.db_session.query(NewsItem).filter(
                NewsItem.scraped_at >= recent_cutoff
            ).all()
            
            for article in recent_articles:
                if article.title:
                    similarity = SequenceMatcher(None, title.lower(), article.title.lower()).ratio()
                    if similarity > 0.85:
                        report.add_result(ValidationResult(
                            level=ValidationLevel.WARNING,
                            field="title",
                            message=f"Very similar title found (similarity: {similarity:.2f})",
                            value=f"Similar to: {article.title[:50]}...",
                            suggestion="May be duplicate or similar content"
                        ))
                        break
    
    async def _validate_against_source(self, item: Dict[str, Any], source: Source, report: ContentValidationReport):
        """Validate item against source-specific rules"""
        url = item.get('url', '')
        
        # Check if URL domain matches source domain
        if url and source.url:
            try:
                item_domain = urllib.parse.urlparse(url).netloc
                source_domain = urllib.parse.urlparse(source.url).netloc
                
                if item_domain != source_domain:
                    report.add_result(ValidationResult(
                        level=ValidationLevel.WARNING,
                        field="url",
                        message="URL domain doesn't match source domain",
                        value=f"Item: {item_domain}, Source: {source_domain}",
                        suggestion="May be external link or incorrect extraction"
                    ))
            except Exception as e:
                logger.warning(f"Error comparing domains: {e}")
        
        # Source-specific validation rules
        if source.type == 'rss_feed':
            # RSS feeds should have consistent structure
            if not item.get('published_at'):
                report.add_result(ValidationResult(
                    level=ValidationLevel.WARNING,
                    field="published_at",
                    message="RSS feed item missing publication date",
                    suggestion="Check RSS feed structure and parsing"
                ))
    
    def _calculate_validation_score(self, report: ContentValidationReport) -> float:
        """Calculate overall validation score based on issues found"""
        base_score = 1.0
        
        # Deduct points for each issue
        for error in report.errors:
            base_score -= 0.3  # Major deduction for errors
        
        for warning in report.warnings:
            base_score -= 0.1  # Moderate deduction for warnings
        
        # Info items don't reduce score but can provide insights
        
        return max(base_score, 0.0)


class BatchValidator:
    """Efficiently validate batches of scraped items"""
    
    def __init__(self, db_session: Session, max_concurrent: int = 10):
        self.db_session = db_session
        self.max_concurrent = max_concurrent
        self.validator = ContentValidator(db_session)
    
    async def validate_batch(self, items: List[Dict[str, Any]], source: Source) -> List[ContentValidationReport]:
        """Validate a batch of items concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def validate_with_semaphore(item):
            async with semaphore:
                return await self.validator.validate_scraped_item(item, source)
        
        tasks = [validate_with_semaphore(item) for item in items]
        reports = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        valid_reports = []
        for i, report in enumerate(reports):
            if isinstance(report, Exception):
                logger.error(f"Error validating item {i}: {report}")
                # Create a failed validation report
                failed_report = ContentValidationReport(is_valid=False, score=0.0)
                failed_report.add_result(ValidationResult(
                    level=ValidationLevel.ERROR,
                    field="validation",
                    message=f"Validation failed with exception: {str(report)}"
                ))
                valid_reports.append(failed_report)
            else:
                valid_reports.append(report)
        
        return valid_reports
    
    def get_validation_summary(self, reports: List[ContentValidationReport]) -> Dict[str, Any]:
        """Generate summary statistics for a batch of validation reports"""
        total_items = len(reports)
        valid_items = sum(1 for r in reports if r.is_valid)
        
        total_errors = sum(len(r.errors) for r in reports)
        total_warnings = sum(len(r.warnings) for r in reports)
        
        avg_score = sum(r.score for r in reports) / max(total_items, 1)
        
        # Most common issues
        error_counts = {}
        warning_counts = {}
        
        for report in reports:
            for error in report.errors:
                key = f"{error.field}: {error.message}"
                error_counts[key] = error_counts.get(key, 0) + 1
            
            for warning in report.warnings:
                key = f"{warning.field}: {warning.message}"
                warning_counts[key] = warning_counts.get(key, 0) + 1
        
        return {
            "total_items": total_items,
            "valid_items": valid_items,
            "invalid_items": total_items - valid_items,
            "validation_rate": valid_items / max(total_items, 1),
            "average_score": avg_score,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "most_common_errors": sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "most_common_warnings": sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        } 