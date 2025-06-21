"""
Option Chain Data Processing Pipeline

This module provides a comprehensive ETL (Extract, Transform, Load) pipeline for option chain data
processing, including data ingestion, transformation, loading, quality checks, duplicate detection,
error handling, batch and streaming processing capabilities with proper logging and monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import traceback
import uuid
from collections import defaultdict, deque
import time

# Database and SQLAlchemy imports
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

# Internal imports
from app.database.connection import get_db
from app.models.option_chain import (
    OptionContract, OptionChainSnapshot, OptionAnalytics, 
    OptionAlert, OptionTradingStats
)
from app.market_data.option_chain_fetcher import OptionChainService
from app.market_data.market_hours import MarketHoursManager, should_fetch_option_data
from app.market_data.data_validator import validate_option_data
from app.analysis.option_analytics import OptionAnalyticsEngine, AnalyticsResult

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Pipeline processing status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class ProcessingMode(Enum):
    """Pipeline processing modes"""
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"

@dataclass
class PipelineMetrics:
    """Pipeline performance and monitoring metrics"""
    total_processed: int = 0
    successful_records: int = 0
    failed_records: int = 0
    duplicate_records: int = 0
    quality_failures: int = 0
    processing_time_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class DataLineage:
    """Data lineage tracking for audit purposes"""
    pipeline_id: str
    execution_id: str
    source_type: str
    source_identifier: str
    transformation_steps: List[str] = field(default_factory=list)
    quality_checks: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingResult:
    """Result of pipeline processing"""
    status: ProcessingStatus
    records_processed: int
    metrics: PipelineMetrics
    lineage: DataLineage
    analytics_result: Optional[AnalyticsResult] = None
    errors: List[str] = field(default_factory=list)

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    def __init__(self):
        self.quality_rules = {
            'completeness': self._check_completeness,
            'validity': self._check_validity,
            'consistency': self._check_consistency,
            'accuracy': self._check_accuracy,
            'timeliness': self._check_timeliness
        }
    
    def validate_option_data(self, data: Dict[str, Any], context: Dict = None) -> Tuple[bool, List[str], float]:
        """
        Comprehensive data quality validation
        
        Returns:
            Tuple of (is_valid, error_messages, quality_score)
        """
        errors = []
        quality_scores = []
        
        try:
            # Run all quality checks
            for rule_name, rule_func in self.quality_rules.items():
                try:
                    is_valid, rule_errors, score = rule_func(data, context or {})
                    quality_scores.append(score)
                    
                    if not is_valid:
                        errors.extend([f"{rule_name}: {err}" for err in rule_errors])
                        
                except Exception as e:
                    errors.append(f"{rule_name} check failed: {str(e)}")
                    quality_scores.append(0.0)
            
            # Calculate overall quality score
            overall_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            is_valid = len(errors) == 0 and overall_score >= 0.7
            
            return is_valid, errors, overall_score
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {str(e)}")
            return False, [f"Validation error: {str(e)}"], 0.0
    
    def _check_completeness(self, data: Dict, context: Dict) -> Tuple[bool, List[str], float]:
        """Check data completeness"""
        errors = []
        required_fields = ['records', 'timestamp']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Check records completeness
        if 'records' in data and 'data' in data['records']:
            option_data = data['records']['data']
            if not option_data:
                errors.append("Empty option data")
            else:
                # Check individual option records
                incomplete_count = 0
                for record in option_data:
                    if not self._is_option_record_complete(record):
                        incomplete_count += 1
                
                completeness_ratio = 1 - (incomplete_count / len(option_data))
                if completeness_ratio < 0.8:
                    errors.append(f"Low data completeness: {completeness_ratio:.2%}")
        
        score = 1.0 if not errors else max(0.0, 1.0 - len(errors) * 0.2)
        return len(errors) == 0, errors, score
    
    def _check_validity(self, data: Dict, context: Dict) -> Tuple[bool, List[str], float]:
        """Check data validity and format"""
        errors = []
        
        # Validate timestamp
        if 'timestamp' in data:
            try:
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                if timestamp > datetime.now():
                    errors.append("Future timestamp detected")
            except ValueError:
                errors.append("Invalid timestamp format")
        
        # Validate option data structure
        if 'records' in data and 'data' in data['records']:
            option_data = data['records']['data']
            invalid_count = 0
            
            for record in option_data:
                if not self._is_option_record_valid(record):
                    invalid_count += 1
            
            if invalid_count > 0:
                validity_ratio = 1 - (invalid_count / len(option_data))
                if validity_ratio < 0.9:
                    errors.append(f"Low data validity: {validity_ratio:.2%}")
        
        score = 1.0 if not errors else max(0.0, 1.0 - len(errors) * 0.15)
        return len(errors) == 0, errors, score
    
    def _check_consistency(self, data: Dict, context: Dict) -> Tuple[bool, List[str], float]:
        """Check data consistency"""
        errors = []
        
        if 'records' in data and 'data' in data['records']:
            option_data = data['records']['data']
            
            # Check for consistent strike progression
            strikes = []
            for record in option_data:
                if 'strikePrice' in record:
                    strikes.append(float(record['strikePrice']))
            
            if strikes:
                strikes.sort()
                # Check for reasonable strike intervals
                intervals = [strikes[i+1] - strikes[i] for i in range(len(strikes)-1)]
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    inconsistent_intervals = sum(1 for interval in intervals 
                                               if abs(interval - avg_interval) > avg_interval * 0.5)
                    
                    if inconsistent_intervals > len(intervals) * 0.2:
                        errors.append("Inconsistent strike price intervals")
        
        score = 1.0 if not errors else max(0.0, 1.0 - len(errors) * 0.1)
        return len(errors) == 0, errors, score
    
    def _check_accuracy(self, data: Dict, context: Dict) -> Tuple[bool, List[str], float]:
        """Check data accuracy against expected ranges"""
        errors = []
        
        if 'records' in data and 'data' in data['records']:
            option_data = data['records']['data']
            
            for record in option_data:
                # Check option prices
                for option_type in ['CE', 'PE']:
                    if option_type in record and record[option_type]:
                        option_info = record[option_type]
                        
                        # Check price ranges
                        last_price = option_info.get('lastPrice', 0)
                        if last_price < 0:
                            errors.append(f"Negative option price detected: {last_price}")
                        
                        # Check implied volatility ranges
                        iv = option_info.get('impliedVolatility', 0)
                        if iv < 0 or iv > 5.0:  # 0% to 500%
                            errors.append(f"Unrealistic implied volatility: {iv}")
                        
                        # Check volume and OI
                        volume = option_info.get('totalTradedVolume', 0)
                        oi = option_info.get('openInterest', 0)
                        if volume < 0 or oi < 0:
                            errors.append("Negative volume or open interest")
        
        score = 1.0 if not errors else max(0.0, 1.0 - len(errors) * 0.1)
        return len(errors) == 0, errors, score
    
    def _check_timeliness(self, data: Dict, context: Dict) -> Tuple[bool, List[str], float]:
        """Check data timeliness"""
        errors = []
        
        if 'timestamp' in data:
            try:
                data_timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                age_minutes = (datetime.now() - data_timestamp).total_seconds() / 60
                
                # Data shouldn't be older than 30 minutes during market hours
                if should_fetch_option_data() and age_minutes > 30:
                    errors.append(f"Stale data: {age_minutes:.1f} minutes old")
                
            except ValueError:
                errors.append("Cannot validate timeliness due to invalid timestamp")
        
        score = 1.0 if not errors else 0.5
        return len(errors) == 0, errors, score
    
    def _is_option_record_complete(self, record: Dict) -> bool:
        """Check if an individual option record is complete"""
        required_fields = ['strikePrice']
        
        # Check required fields
        for field in required_fields:
            if field not in record:
                return False
        
        # Check if at least one option type exists
        has_option = False
        for option_type in ['CE', 'PE']:
            if option_type in record and record[option_type]:
                option_data = record[option_type]
                if 'lastPrice' in option_data or 'openInterest' in option_data:
                    has_option = True
                    break
        
        return has_option
    
    def _is_option_record_valid(self, record: Dict) -> bool:
        """Check if an individual option record is valid"""
        try:
            # Validate strike price
            if 'strikePrice' in record:
                strike = float(record['strikePrice'])
                if strike <= 0:
                    return False
            
            # Validate option data
            for option_type in ['CE', 'PE']:
                if option_type in record and record[option_type]:
                    option_data = record[option_type]
                    
                    # Check numeric fields
                    numeric_fields = ['lastPrice', 'openInterest', 'totalTradedVolume', 'impliedVolatility']
                    for field in numeric_fields:
                        if field in option_data:
                            value = option_data[field]
                            if not isinstance(value, (int, float)) or value < 0:
                                return False
            
            return True
            
        except (ValueError, TypeError):
            return False

class DuplicateDetector:
    """Advanced duplicate detection and deduplication"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.recent_hashes = deque(maxlen=window_size)
        self.hash_timestamps = {}
    
    def generate_data_hash(self, data: Dict[str, Any]) -> str:
        """Generate unique hash for option chain data"""
        try:
            # Extract key identifying information
            identifier = {
                'symbol': data.get('symbol', ''),
                'timestamp': data.get('timestamp', ''),
                'records_count': len(data.get('records', {}).get('data', [])),
            }
            
            # Add option chain fingerprint
            if 'records' in data and 'data' in data['records']:
                option_fingerprint = []
                for record in data['records']['data']:
                    if 'strikePrice' in record:
                        strike_info = {
                            'strike': record['strikePrice'],
                            'ce_oi': record.get('CE', {}).get('openInterest', 0) if record.get('CE') else 0,
                            'pe_oi': record.get('PE', {}).get('openInterest', 0) if record.get('PE') else 0,
                        }
                        option_fingerprint.append(strike_info)
                
                identifier['option_fingerprint'] = str(sorted(option_fingerprint, key=lambda x: x['strike']))
            
            # Generate hash
            data_string = json.dumps(identifier, sort_keys=True)
            return hashlib.md5(data_string.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to generate data hash: {str(e)}")
            return str(uuid.uuid4())  # Fallback to unique ID
    
    def is_duplicate(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if data is a duplicate
        
        Returns:
            Tuple of (is_duplicate, data_hash)
        """
        data_hash = self.generate_data_hash(data)
        
        # Check against recent hashes
        is_duplicate = data_hash in self.recent_hashes
        
        if not is_duplicate:
            self.recent_hashes.append(data_hash)
            self.hash_timestamps[data_hash] = datetime.now()
        
        # Clean old hashes periodically
        if len(self.hash_timestamps) > self.window_size * 1.5:
            self._cleanup_old_hashes()
        
        return is_duplicate, data_hash
    
    def _cleanup_old_hashes(self):
        """Clean up old hash entries"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        old_hashes = [h for h, ts in self.hash_timestamps.items() if ts < cutoff_time]
        
        for old_hash in old_hashes:
            del self.hash_timestamps[old_hash]

class ErrorHandler:
    """Comprehensive error handling and recovery"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_counts = defaultdict(int)
        self.last_errors = {}
    
    async def handle_with_retry(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Execute function with retry logic
        
        Returns:
            Tuple of (result, success)
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                return result, True
                
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                self.error_counts[error_type] += 1
                self.last_errors[error_type] = str(e)
                
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
        
        return None, False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of encountered errors"""
        return {
            'error_counts': dict(self.error_counts),
            'last_errors': dict(self.last_errors),
            'total_errors': sum(self.error_counts.values())
        }

class DataProcessingPipeline:
    """
    Comprehensive data processing pipeline for option chain data
    """
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.HYBRID,
        batch_size: int = 100,
        quality_threshold: float = 0.7,
        enable_duplicate_detection: bool = True,
        enable_analytics: bool = True
    ):
        self.pipeline_id = str(uuid.uuid4())
        self.mode = mode
        self.batch_size = batch_size
        self.quality_threshold = quality_threshold
        self.enable_duplicate_detection = enable_duplicate_detection
        self.enable_analytics = enable_analytics
        
        # Initialize components
        self.data_service = OptionChainService()
        self.market_hours = MarketHoursManager()
        self.quality_checker = DataQualityChecker()
        self.duplicate_detector = DuplicateDetector() if enable_duplicate_detection else None
        self.error_handler = ErrorHandler()
        self.analytics_engine = OptionAnalyticsEngine() if enable_analytics else None
        
        # Processing state
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized data processing pipeline {self.pipeline_id} in {mode.value} mode")
    
    async def process_symbol(
        self, 
        symbol: str, 
        is_equity: bool = False,
        historical_data: Dict = None
    ) -> ProcessingResult:
        """
        Process option chain data for a specific symbol
        
        Args:
            symbol: Symbol to process (e.g., 'NIFTY', 'BANKNIFTY')
            is_equity: Whether the symbol is an equity
            historical_data: Historical data for analytics context
        
        Returns:
            ProcessingResult with processing details
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        metrics = PipelineMetrics(start_time=datetime.now())
        
        lineage = DataLineage(
            pipeline_id=self.pipeline_id,
            execution_id=execution_id,
            source_type="NSE_API",
            source_identifier=symbol,
            metadata={'is_equity': is_equity}
        )
        
        try:
            logger.info(f"Starting processing for symbol: {symbol}")
            
            # Step 1: Extract - Fetch raw data
            raw_data, extract_success = await self.error_handler.handle_with_retry(
                self._extract_data, symbol, is_equity
            )
            
            if not extract_success or not raw_data:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    records_processed=0,
                    metrics=metrics,
                    lineage=lineage,
                    errors=["Data extraction failed"]
                )
            
            lineage.transformation_steps.append("data_extraction")
            metrics.total_processed = 1
            
            # Step 2: Validate - Quality checks
            quality_result = await self._validate_data(raw_data, lineage)
            if not quality_result['passed']:
                metrics.quality_failures += 1
                if quality_result['score'] < self.quality_threshold:
                    return ProcessingResult(
                        status=ProcessingStatus.FAILED,
                        records_processed=1,
                        metrics=metrics,
                        lineage=lineage,
                        errors=quality_result['errors']
                    )
            
            # Step 3: Duplicate Detection
            if self.duplicate_detector:
                is_duplicate, data_hash = self.duplicate_detector.is_duplicate(raw_data)
                if is_duplicate:
                    metrics.duplicate_records += 1
                    logger.info(f"Duplicate data detected for {symbol}, skipping")
                    return ProcessingResult(
                        status=ProcessingStatus.SKIPPED,
                        records_processed=1,
                        metrics=metrics,
                        lineage=lineage,
                        errors=["Duplicate data"]
                    )
                lineage.metadata['data_hash'] = data_hash
            
            # Step 4: Transform - Process and analyze data
            transformed_data, analytics_result = await self._transform_data(
                raw_data, symbol, historical_data, lineage
            )
            
            if not transformed_data:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    records_processed=1,
                    metrics=metrics,
                    lineage=lineage,
                    errors=["Data transformation failed"]
                )
            
            # Step 5: Load - Store to database
            load_success = await self._load_data(transformed_data, lineage)
            
            if load_success:
                metrics.successful_records += 1
                processing_time = (time.time() - start_time) * 1000
                metrics.processing_time_ms = processing_time
                metrics.end_time = datetime.now()
                
                logger.info(f"Successfully processed {symbol} in {processing_time:.2f}ms")
                
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    records_processed=1,
                    metrics=metrics,
                    lineage=lineage,
                    analytics_result=analytics_result
                )
            else:
                metrics.failed_records += 1
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    records_processed=1,
                    metrics=metrics,
                    lineage=lineage,
                    errors=["Data loading failed"]
                )
                
        except Exception as e:
            logger.error(f"Pipeline processing failed for {symbol}: {str(e)}")
            metrics.failed_records += 1
            metrics.errors.append(str(e))
            
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                records_processed=1,
                metrics=metrics,
                lineage=lineage,
                errors=[str(e)]
            )
    
    async def _extract_data(self, symbol: str, is_equity: bool) -> Dict[str, Any]:
        """Extract raw option chain data"""
        logger.debug(f"Extracting data for {symbol}")
        
        # Check market hours
        market_status = self.market_hours.get_current_status()
        if not market_status.is_trading_day:
            logger.info(f"Market is closed: {market_status.message}")
            return None
        
        # Fetch option chain data
        async with self.data_service.fetcher as fetcher:
            raw_data = await fetcher.fetch_option_chain(symbol, is_equity)
        
        if raw_data:
            # Add metadata
            raw_data['timestamp'] = datetime.now().isoformat()
            raw_data['symbol'] = symbol
            raw_data['market_status'] = market_status.status.value
        
        return raw_data
    
    async def _validate_data(self, data: Dict[str, Any], lineage: DataLineage) -> Dict[str, Any]:
        """Validate data quality"""
        logger.debug("Validating data quality")
        
        # Use existing data validator
        validation_result = validate_option_data(data)
        
        # Enhanced quality checks
        is_valid, errors, score = self.quality_checker.validate_option_data(data)
        
        quality_check = {
            'timestamp': datetime.now().isoformat(),
            'validator_score': validation_result.quality_score,
            'enhanced_score': score,
            'is_valid': is_valid and validation_result.is_valid,
            'errors': errors + validation_result.errors,
            'warnings': validation_result.warnings
        }
        
        lineage.quality_checks.append(quality_check)
        lineage.transformation_steps.append("data_validation")
        
        return {
            'passed': quality_check['is_valid'],
            'score': min(validation_result.quality_score, score),
            'errors': quality_check['errors'],
            'warnings': quality_check['warnings']
        }
    
    async def _transform_data(
        self, 
        raw_data: Dict[str, Any], 
        symbol: str,
        historical_data: Dict = None,
        lineage: DataLineage = None
    ) -> Tuple[Dict[str, Any], Optional[AnalyticsResult]]:
        """Transform and analyze data"""
        logger.debug(f"Transforming data for {symbol}")
        
        try:
            analytics_result = None
            
            # Calculate analytics if enabled
            if self.analytics_engine and 'records' in raw_data:
                option_data = raw_data['records']['data']
                spot_price = raw_data.get('underlyingValue', 0)
                
                # Calculate days to expiry (approximate)
                days_to_expiry = 7  # Default weekly expiry
                
                analytics_result = await self.analytics_engine.calculate_comprehensive_analytics(
                    option_data=option_data,
                    spot_price=spot_price,
                    days_to_expiry=days_to_expiry,
                    historical_data=historical_data
                )
                
                lineage.transformation_steps.append("analytics_calculation")
            
            # Prepare transformed data structure
            transformed_data = {
                'symbol': symbol,
                'raw_data': raw_data,
                'analytics': analytics_result,
                'processed_at': datetime.now(),
                'pipeline_id': self.pipeline_id,
                'lineage_id': lineage.execution_id if lineage else None
            }
            
            lineage.transformation_steps.append("data_transformation")
            
            return transformed_data, analytics_result
            
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            return None, None
    
    async def _load_data(self, data: Dict[str, Any], lineage: DataLineage) -> bool:
        """Load data to database"""
        logger.debug(f"Loading data for {data['symbol']}")
        
        try:
            # This is a placeholder for actual database loading
            # In a real implementation, you would:
            # 1. Create OptionChainSnapshot records
            # 2. Store analytics results
            # 3. Update statistics
            # 4. Store lineage information
            
            # For now, we'll just log the operation
            logger.info(f"Data loading simulation for {data['symbol']}")
            
            # Simulate database operations
            await asyncio.sleep(0.1)  # Simulate DB latency
            
            lineage.transformation_steps.append("data_loading")
            
            return True
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return False
    
    async def process_batch(self, symbols: List[str]) -> List[ProcessingResult]:
        """Process multiple symbols in batch mode"""
        logger.info(f"Starting batch processing for {len(symbols)} symbols")
        
        results = []
        
        # Process symbols concurrently
        tasks = [self.process_symbol(symbol) for symbol in symbols]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Symbol {symbols[i]} processing failed: {str(result)}")
                    results[i] = ProcessingResult(
                        status=ProcessingStatus.FAILED,
                        records_processed=0,
                        metrics=PipelineMetrics(),
                        lineage=DataLineage(
                            pipeline_id=self.pipeline_id,
                            execution_id=str(uuid.uuid4()),
                            source_type="NSE_API",
                            source_identifier=symbols[i]
                        ),
                        errors=[str(result)]
                    )
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            results = [ProcessingResult(
                status=ProcessingStatus.FAILED,
                records_processed=0,
                metrics=PipelineMetrics(),
                lineage=DataLineage(
                    pipeline_id=self.pipeline_id,
                    execution_id=str(uuid.uuid4()),
                    source_type="NSE_API",
                    source_identifier=symbol
                ),
                errors=[str(e)]
            ) for symbol in symbols]
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    async def start_streaming_mode(self, symbols: List[str], interval_seconds: int = 300):
        """Start streaming mode processing"""
        logger.info(f"Starting streaming mode for symbols: {symbols}")
        self.is_running = True
        
        while self.is_running:
            try:
                # Check if market is open
                if should_fetch_option_data():
                    # Process all symbols
                    results = await self.process_batch(symbols)
                    
                    # Log summary
                    successful = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
                    logger.info(f"Streaming cycle: {successful}/{len(results)} successful")
                else:
                    logger.debug("Market closed, skipping streaming cycle")
                
                # Wait for next cycle
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Streaming mode error: {str(e)}")
                await asyncio.sleep(interval_seconds)
    
    def stop_streaming_mode(self):
        """Stop streaming mode processing"""
        logger.info("Stopping streaming mode")
        self.is_running = False
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics"""
        return {
            'pipeline_id': self.pipeline_id,
            'mode': self.mode.value,
            'is_running': self.is_running,
            'quality_threshold': self.quality_threshold,
            'duplicate_detection_enabled': self.enable_duplicate_detection,
            'analytics_enabled': self.enable_analytics,
            'error_summary': self.error_handler.get_error_summary()
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Convenience functions for quick pipeline operations
async def process_single_symbol(symbol: str, is_equity: bool = False) -> ProcessingResult:
    """Quick processing of a single symbol"""
    pipeline = DataProcessingPipeline(mode=ProcessingMode.BATCH)
    return await pipeline.process_symbol(symbol, is_equity)

async def process_symbols_batch(symbols: List[str]) -> List[ProcessingResult]:
    """Quick batch processing of multiple symbols"""
    pipeline = DataProcessingPipeline(mode=ProcessingMode.BATCH)
    return await pipeline.process_batch(symbols)

def create_streaming_pipeline(symbols: List[str], interval_seconds: int = 300) -> DataProcessingPipeline:
    """Create a streaming pipeline for continuous processing"""
    pipeline = DataProcessingPipeline(mode=ProcessingMode.STREAMING)
    return pipeline 