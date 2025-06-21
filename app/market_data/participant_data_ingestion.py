"""
NSE Participant Data Ingestion Pipeline

This module provides a comprehensive ETL system for collecting and processing
National Stock Exchange (NSE) participant flow data, including real-time feeds,
historical data imports, data validation, and automated scheduling.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from decimal import Decimal, InvalidOperation as DecimalInvalidOperation
import json
import csv
import io
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

# Internal imports
from app.database.connection import get_db
from app.models.participant_flow import (
    ParticipantProfile, ParticipantActivity, ParticipantFlowSummary,
    ParticipantType, MarketSegment, DataSource, DataQuality,
    FlowDirection
)
from app.models.asset import Asset
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class IngestionResult:
    """Result of data ingestion operation"""
    success: bool
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_failed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    data_quality_score: float = 1.0
    source_timestamp: Optional[datetime] = None

@dataclass
class DataValidationResult:
    """Result of data validation"""
    is_valid: bool
    quality_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class NSEDataValidator:
    """Validates NSE participant data for quality and consistency"""
    
    def __init__(self):
        self.validation_rules = {
            'completeness': self._check_completeness,
            'format': self._check_format,
            'ranges': self._check_value_ranges,
            'consistency': self._check_consistency,
            'duplicates': self._check_duplicates
        }
    
    def validate_participant_data(self, data: Dict[str, Any]) -> DataValidationResult:
        """Comprehensive validation of participant data"""
        issues = []
        warnings = []
        quality_scores = []
        
        try:
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    is_valid, rule_issues, rule_warnings, score = rule_func(data)
                    quality_scores.append(score)
                    
                    if not is_valid:
                        issues.extend([f"{rule_name}: {issue}" for issue in rule_issues])
                    
                    if rule_warnings:
                        warnings.extend([f"{rule_name}: {warning}" for warning in rule_warnings])
                        
                except Exception as e:
                    issues.append(f"{rule_name} validation failed: {str(e)}")
                    quality_scores.append(0.0)
            
            overall_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            is_valid = len(issues) == 0 and overall_score >= 0.7
            
            return DataValidationResult(
                is_valid=is_valid,
                quality_score=overall_score,
                issues=issues,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return DataValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=[f"Validation error: {str(e)}"]
            )
    
    def _check_completeness(self, data: Dict) -> Tuple[bool, List[str], List[str], float]:
        """Check data completeness"""
        issues = []
        warnings = []
        
        # Required fields for participant data
        required_fields = ['date', 'participant_type']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
        
        # Check for participant activity data
        value_fields = ['buy_value', 'sell_value', 'net_value', 'gross_turnover']
        missing_values = sum(1 for field in value_fields if data.get(field) is None)
        
        if missing_values > len(value_fields) * 0.5:
            warnings.append(f"Missing {missing_values}/{len(value_fields)} value fields")
        
        completeness_score = 1.0 - (len(missing_fields) * 0.3) - (missing_values * 0.1)
        is_valid = len(issues) == 0 and completeness_score >= 0.7
        
        return is_valid, issues, warnings, max(0.0, completeness_score)
    
    def _check_format(self, data: Dict) -> Tuple[bool, List[str], List[str], float]:
        """Check data format and types"""
        issues = []
        warnings = []
        
        # Check date format
        if 'date' in data:
            try:
                if isinstance(data['date'], str):
                    datetime.fromisoformat(data['date'])
                elif not isinstance(data['date'], (datetime, date)):
                    issues.append("Invalid date format")
            except ValueError:
                issues.append("Invalid date format")
        
        # Check numeric fields
        numeric_fields = ['buy_value', 'sell_value', 'net_value', 'gross_turnover']
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    float(data[field])
                except (ValueError, TypeError):
                    issues.append(f"Invalid numeric format for {field}: {data[field]}")
        
        # Check participant type
        if 'participant_type' in data:
            valid_types = [pt.value for pt in ParticipantType]
            if data['participant_type'] not in valid_types:
                issues.append(f"Invalid participant type: {data['participant_type']}")
        
        format_score = 1.0 - (len(issues) * 0.2)
        is_valid = len(issues) == 0
        
        return is_valid, issues, warnings, max(0.0, format_score)
    
    def _check_value_ranges(self, data: Dict) -> Tuple[bool, List[str], List[str], float]:
        """Check value ranges and reasonableness"""
        issues = []
        warnings = []
        
        # Check for negative values where not expected
        non_negative_fields = ['buy_value', 'sell_value', 'gross_turnover']
        for field in non_negative_fields:
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    if value < 0:
                        issues.append(f"Negative value for {field}: {value}")
                except (ValueError, TypeError):
                    continue
        
        # Check for extremely large values (potential data errors)
        value_fields = ['buy_value', 'sell_value', 'gross_turnover']
        for field in value_fields:
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    if value > 1000000:  # > 10 lakh crores (unlikely for single day)
                        warnings.append(f"Unusually large value for {field}: {value}")
                except (ValueError, TypeError):
                    continue
        
        range_score = 1.0 - (len(issues) * 0.3) - (len(warnings) * 0.1)
        is_valid = len(issues) == 0
        
        return is_valid, issues, warnings, max(0.0, range_score)
    
    def _check_consistency(self, data: Dict) -> Tuple[bool, List[str], List[str], float]:
        """Check internal consistency of data"""
        issues = []
        warnings = []
        
        # Check net value calculation
        if all(field in data and data[field] is not None 
               for field in ['buy_value', 'sell_value', 'net_value']):
            try:
                buy_val = float(data['buy_value'])
                sell_val = float(data['sell_value'])
                net_val = float(data['net_value'])
                expected_net = buy_val - sell_val
                
                if abs(net_val - expected_net) > 0.01:  # Allow small rounding differences
                    issues.append(f"Net value inconsistency: {net_val} vs expected {expected_net}")
            except (ValueError, TypeError):
                pass
        
        # Check gross turnover calculation
        if all(field in data and data[field] is not None 
               for field in ['buy_value', 'sell_value', 'gross_turnover']):
            try:
                buy_val = float(data['buy_value'])
                sell_val = float(data['sell_value'])
                gross_val = float(data['gross_turnover'])
                expected_gross = buy_val + sell_val
                
                if abs(gross_val - expected_gross) > 0.01:
                    warnings.append(f"Gross turnover inconsistency: {gross_val} vs expected {expected_gross}")
            except (ValueError, TypeError):
                pass
        
        consistency_score = 1.0 - (len(issues) * 0.4) - (len(warnings) * 0.1)
        is_valid = len(issues) == 0
        
        return is_valid, issues, warnings, max(0.0, consistency_score)
    
    def _check_duplicates(self, data: Dict) -> Tuple[bool, List[str], List[str], float]:
        """Check for potential duplicate indicators"""
        issues = []
        warnings = []
        
        # For now, just return valid since duplicate checking requires database context
        # This would be enhanced with actual duplicate detection logic
        
        return True, issues, warnings, 1.0

class NSEAPIClient:
    """Client for fetching data from NSE APIs"""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com/api"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        self.session = None
        self.rate_limit_delay = 1.0  # Seconds between requests
        self.last_request_time = 0
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    async def fetch_participant_wise_trading_data(self, date_str: str = None) -> Optional[Dict]:
        """Fetch participant-wise trading data from NSE"""
        await self._rate_limit()
        
        try:
            # NSE participant data endpoint (this is a conceptual URL - actual endpoint may vary)
            url = f"{self.base_url}/participant-wise-trading-data"
            
            params = {}
            if date_str:
                params['date'] = date_str
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successfully fetched participant data for {date_str or 'latest'}")
                    return data
                else:
                    logger.warning(f"NSE API returned status {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("Timeout while fetching NSE participant data")
            return None
        except Exception as e:
            logger.error(f"Error fetching NSE participant data: {str(e)}")
            return None
    
    async def fetch_derivatives_statistics(self, symbol: str = "NIFTY") -> Optional[Dict]:
        """Fetch derivatives statistics for participant analysis"""
        await self._rate_limit()
        
        try:
            url = f"{self.base_url}/derivatives-statistics"
            params = {'symbol': symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successfully fetched derivatives stats for {symbol}")
                    return data
                else:
                    logger.warning(f"NSE API returned status {response.status} for derivatives stats")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching derivatives statistics: {str(e)}")
            return None

class ParticipantDataProcessor:
    """Processes and transforms NSE participant data"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.validator = NSEDataValidator()
    
    def process_nse_participant_data(self, raw_data: Dict[str, Any], source_date: date = None) -> IngestionResult:
        """Process NSE participant data and convert to our model format"""
        start_time = time.time()
        result = IngestionResult(success=False)
        
        try:
            if not raw_data:
                result.errors.append("No data provided")
                return result
            
            # Extract participant data from NSE format
            participant_records = self._extract_participant_records(raw_data, source_date)
            
            if not participant_records:
                result.errors.append("No participant records found in data")
                return result
            
            # Process each participant record
            processed_count = 0
            inserted_count = 0
            failed_count = 0
            
            for record in participant_records:
                try:
                    # Validate the record
                    validation_result = self.validator.validate_participant_data(record)
                    
                    if not validation_result.is_valid:
                        result.warnings.extend(validation_result.issues)
                        failed_count += 1
                        continue
                    
                    # Transform and store the record
                    if self._store_participant_record(record, validation_result.quality_score):
                        inserted_count += 1
                    else:
                        failed_count += 1
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing participant record: {str(e)}")
                    result.errors.append(f"Record processing error: {str(e)}")
                    failed_count += 1
            
            # Calculate results
            processing_time = (time.time() - start_time) * 1000
            
            result.success = processed_count > 0
            result.records_processed = processed_count
            result.records_inserted = inserted_count
            result.records_failed = failed_count
            result.processing_time_ms = processing_time
            result.data_quality_score = 1.0 - (failed_count / max(1, processed_count))
            
            logger.info(f"Processed {processed_count} participant records in {processing_time:.2f}ms")
            
        except Exception as e:
            result.errors.append(f"Processing failed: {str(e)}")
            logger.error(f"Participant data processing failed: {str(e)}")
        
        return result
    
    def _extract_participant_records(self, raw_data: Dict, source_date: date = None) -> List[Dict]:
        """Extract participant records from NSE data format"""
        records = []
        
        try:
            trade_date = source_date or date.today()
            
            # Example structure - adapt based on actual NSE API response
            if 'participantWiseData' in raw_data:
                participant_data = raw_data['participantWiseData']
                
                for participant_type, type_data in participant_data.items():
                    # Map NSE participant names to our enum values
                    mapped_type = self._map_participant_type(participant_type)
                    
                    if mapped_type:
                        # Process different market segments
                        for segment, segment_data in type_data.items():
                            mapped_segment = self._map_market_segment(segment)
                            
                            if mapped_segment and isinstance(segment_data, dict):
                                record = {
                                    'date': trade_date.isoformat(),  # For validation
                                    'trade_date': trade_date,  # For database storage
                                    'participant_type': mapped_type.value,  # String value for validation
                                    'participant_type_enum': mapped_type,  # Enum for database storage  
                                    'market_segment': mapped_segment,
                                    'buy_value': self._safe_decimal(segment_data.get('buyValue', 0)),
                                    'sell_value': self._safe_decimal(segment_data.get('sellValue', 0)),
                                    'net_value': self._safe_decimal(segment_data.get('netValue', 0)),
                                    'gross_turnover': self._safe_decimal(segment_data.get('grossTurnover', 0)),
                                    'buy_quantity': self._safe_int(segment_data.get('buyQuantity', 0)),
                                    'sell_quantity': self._safe_int(segment_data.get('sellQuantity', 0)),
                                    'data_source': DataSource.NSE,
                                    'raw_data': segment_data
                                }
                                
                                # Calculate derived fields
                                record['net_quantity'] = record['buy_quantity'] - record['sell_quantity']
                                
                                # Handle options-specific data
                                if mapped_segment == MarketSegment.OPTIONS:
                                    record.update({
                                        'call_buy_value': self._safe_decimal(segment_data.get('callBuyValue', 0)),
                                        'call_sell_value': self._safe_decimal(segment_data.get('callSellValue', 0)),
                                        'put_buy_value': self._safe_decimal(segment_data.get('putBuyValue', 0)),
                                        'put_sell_value': self._safe_decimal(segment_data.get('putSellValue', 0))
                                    })
                                
                                # Handle futures-specific data
                                elif mapped_segment == MarketSegment.FUTURES:
                                    record.update({
                                        'long_positions': self._safe_decimal(segment_data.get('longPositions', 0)),
                                        'short_positions': self._safe_decimal(segment_data.get('shortPositions', 0)),
                                        'open_interest_change': self._safe_decimal(segment_data.get('oiChange', 0))
                                    })
                                
                                records.append(record)
            
            logger.info(f"Extracted {len(records)} participant records")
            
        except Exception as e:
            logger.error(f"Error extracting participant records: {str(e)}")
        
        return records
    
    def _map_participant_type(self, nse_type: str) -> Optional[ParticipantType]:
        """Map NSE participant type to our enum"""
        mapping = {
            'FII': ParticipantType.FII,
            'ForeignInstitutionalInvestors': ParticipantType.FII,
            'DII': ParticipantType.DII,
            'DomesticInstitutionalInvestors': ParticipantType.DII,
            'Pro': ParticipantType.PROPRIETARY,
            'Proprietary': ParticipantType.PROPRIETARY,
            'Retail': ParticipantType.RETAIL,
            'RetailInvestors': ParticipantType.RETAIL,
            'MutualFunds': ParticipantType.MUTUAL_FUND,
            'Insurance': ParticipantType.INSURANCE,
            'Corporate': ParticipantType.CORPORATE,
            'HNI': ParticipantType.HNI
        }
        
        return mapping.get(nse_type)
    
    def _map_market_segment(self, nse_segment: str) -> Optional[MarketSegment]:
        """Map NSE market segment to our enum"""
        mapping = {
            'Cash': MarketSegment.CASH,
            'Equity': MarketSegment.CASH,
            'Futures': MarketSegment.FUTURES,
            'Options': MarketSegment.OPTIONS,
            'Currency': MarketSegment.CURRENCY,
            'Commodity': MarketSegment.COMMODITY
        }
        
        return mapping.get(nse_segment)
    
    def _safe_decimal(self, value: Any) -> Decimal:
        """Safely convert value to Decimal"""
        try:
            if value is None or value == '':
                return Decimal('0.00')
            return Decimal(str(value))
        except (ValueError, TypeError, DecimalInvalidOperation, Exception):
            return Decimal('0.00')
    
    def _safe_int(self, value: Any) -> int:
        """Safely convert value to int"""
        try:
            if value is None or value == '':
                return 0
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def _store_participant_record(self, record: Dict, quality_score: float) -> bool:
        """Store participant record in database"""
        try:
            # Get or create participant profile
            participant = self._get_or_create_participant(
                record['participant_type_enum']
            )
            
            # Create participant activity record
            activity = ParticipantActivity(
                trade_date=record['trade_date'],
                participant_id=participant.id,
                market_segment=record['market_segment'],
                buy_value=record['buy_value'],
                sell_value=record['sell_value'],
                net_value=record['net_value'],
                gross_turnover=record['gross_turnover'],
                buy_quantity=record['buy_quantity'],
                sell_quantity=record['sell_quantity'],
                net_quantity=record['net_quantity'],
                data_source=record['data_source'],
                data_quality=self._determine_data_quality(quality_score),
                confidence_score=Decimal(str(quality_score)),
                raw_data=record.get('raw_data'),
                # Options-specific fields
                call_buy_value=record.get('call_buy_value', Decimal('0.00')),
                call_sell_value=record.get('call_sell_value', Decimal('0.00')),
                put_buy_value=record.get('put_buy_value', Decimal('0.00')),
                put_sell_value=record.get('put_sell_value', Decimal('0.00')),
                # Futures-specific fields
                long_positions=record.get('long_positions', Decimal('0.00')),
                short_positions=record.get('short_positions', Decimal('0.00')),
                open_interest_change=record.get('open_interest_change', Decimal('0.00'))
            )
            
            self.db_session.add(activity)
            self.db_session.commit()
            
            return True
            
        except IntegrityError:
            # Handle duplicate entries
            self.db_session.rollback()
            logger.warning(f"Duplicate participant activity record: {record['participant_type']} - {record['trade_date']}")
            return False
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error storing participant record: {str(e)}")
            return False
    
    def _get_or_create_participant(self, participant_type: ParticipantType) -> ParticipantProfile:
        """Get or create participant profile"""
        participant = self.db_session.query(ParticipantProfile).filter(
            ParticipantProfile.participant_type == participant_type,
            ParticipantProfile.participant_code.is_(None)  # Generic participant
        ).first()
        
        if not participant:
            participant = ParticipantProfile(
                participant_type=participant_type,
                participant_name=f"{participant_type.value} Participants",
                is_active=True
            )
            self.db_session.add(participant)
            self.db_session.flush()  # Get the ID
        
        return participant
    
    def _determine_data_quality(self, score: float) -> DataQuality:
        """Determine data quality level based on score"""
        if score >= 0.9:
            return DataQuality.HIGH
        elif score >= 0.7:
            return DataQuality.MEDIUM
        elif score >= 0.5:
            return DataQuality.LOW
        else:
            return DataQuality.ESTIMATED

class NSEDataIngestionPipeline:
    """Main class for NSE data ingestion pipeline"""
    
    def __init__(self):
        self.api_client = None
        self.processor = None
        self.db_session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize database session
        self.db_session = next(get_db())
        
        # Initialize components
        self.api_client = NSEAPIClient()
        await self.api_client.__aenter__()
        
        self.processor = ParticipantDataProcessor(self.db_session)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.api_client:
            await self.api_client.__aexit__(exc_type, exc_val, exc_tb)
        
        if self.db_session:
            self.db_session.close()
    
    async def ingest_current_day_data(self) -> IngestionResult:
        """Ingest current day participant data"""
        try:
            logger.info("Starting current day data ingestion")
            
            # Fetch current participant data
            raw_data = await self.api_client.fetch_participant_wise_trading_data()
            
            if not raw_data:
                return IngestionResult(
                    success=False,
                    errors=["Failed to fetch data from NSE API"]
                )
            
            # Process the data
            result = self.processor.process_nse_participant_data(raw_data)
            
            if result.success:
                logger.info(f"Successfully ingested current day data: {result.records_inserted} records")
            else:
                logger.error(f"Data ingestion failed: {result.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"Current day data ingestion failed: {str(e)}")
            return IngestionResult(
                success=False,
                errors=[f"Ingestion error: {str(e)}"]
            )
    
    async def ingest_historical_data(self, start_date: date, end_date: date) -> List[IngestionResult]:
        """Ingest historical participant data for date range"""
        results = []
        current_date = start_date
        
        logger.info(f"Starting historical data ingestion from {start_date} to {end_date}")
        
        while current_date <= end_date:
            try:
                # Skip weekends (NSE is closed)
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue
                
                date_str = current_date.strftime('%Y-%m-%d')
                logger.info(f"Ingesting data for {date_str}")
                
                # Fetch data for specific date
                raw_data = await self.api_client.fetch_participant_wise_trading_data(date_str)
                
                if raw_data:
                    # Process the data
                    result = self.processor.process_nse_participant_data(raw_data, current_date)
                    result.source_timestamp = datetime.combine(current_date, datetime.min.time())
                    results.append(result)
                else:
                    results.append(IngestionResult(
                        success=False,
                        errors=[f"No data available for {date_str}"],
                        source_timestamp=datetime.combine(current_date, datetime.min.time())
                    ))
                
                # Add delay between requests to avoid overwhelming the API
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error ingesting data for {current_date}: {str(e)}")
                results.append(IngestionResult(
                    success=False,
                    errors=[f"Ingestion error for {current_date}: {str(e)}"],
                    source_timestamp=datetime.combine(current_date, datetime.min.time())
                ))
            
            current_date += timedelta(days=1)
        
        successful_results = [r for r in results if r.success]
        logger.info(f"Historical data ingestion completed: {len(successful_results)}/{len(results)} successful")
        
        return results
    
    async def generate_daily_summary(self, target_date: date = None) -> bool:
        """Generate daily participant flow summary"""
        if not target_date:
            target_date = date.today()
        
        try:
            logger.info(f"Generating daily summary for {target_date}")
            
            # Query all participant activities for the date
            activities = self.db_session.query(ParticipantActivity).filter(
                ParticipantActivity.trade_date == target_date
            ).all()
            
            if not activities:
                logger.warning(f"No participant activities found for {target_date}")
                return False
            
            # Calculate summary by market segment
            segments = set(activity.market_segment for activity in activities)
            
            for segment in segments:
                segment_activities = [a for a in activities if a.market_segment == segment]
                
                # Calculate totals by participant type
                fii_net = sum(a.net_value for a in segment_activities 
                             if a.participant.participant_type == ParticipantType.FII)
                dii_net = sum(a.net_value for a in segment_activities 
                             if a.participant.participant_type == ParticipantType.DII)
                retail_net = sum(a.net_value for a in segment_activities 
                                if a.participant.participant_type == ParticipantType.RETAIL)
                prop_net = sum(a.net_value for a in segment_activities 
                              if a.participant.participant_type == ParticipantType.PROPRIETARY)
                
                total_turnover = sum(a.gross_turnover for a in segment_activities)
                total_net_flow = fii_net + dii_net + retail_net + prop_net
                
                # Determine flow sentiment
                if total_net_flow > 0:
                    sentiment = "bullish"
                elif total_net_flow < 0:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"
                
                # Create or update summary
                summary = ParticipantFlowSummary(
                    summary_date=target_date,
                    market_segment=segment,
                    total_fii_net_flow=fii_net,
                    total_dii_net_flow=dii_net,
                    total_retail_net_flow=retail_net,
                    total_proprietary_net_flow=prop_net,
                    total_market_turnover=total_turnover,
                    total_market_net_flow=total_net_flow,
                    net_institutional_flow=fii_net + dii_net,
                    retail_vs_institutional=(retail_net + prop_net) - (fii_net + dii_net),
                    flow_sentiment=sentiment,
                    participant_count=len(set(a.participant_id for a in segment_activities)),
                    data_completeness=Decimal('100.00')
                )
                
                self.db_session.add(summary)
            
            self.db_session.commit()
            logger.info(f"Successfully generated daily summary for {target_date}")
            return True
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error generating daily summary: {str(e)}")
            return False

# Convenience functions for easy usage
async def ingest_current_participant_data() -> IngestionResult:
    """Convenience function to ingest current day participant data"""
    async with NSEDataIngestionPipeline() as pipeline:
        return await pipeline.ingest_current_day_data()

async def ingest_historical_participant_data(start_date: date, end_date: date) -> List[IngestionResult]:
    """Convenience function to ingest historical participant data"""
    async with NSEDataIngestionPipeline() as pipeline:
        return await pipeline.ingest_historical_data(start_date, end_date)

async def generate_participant_summary(target_date: date = None) -> bool:
    """Convenience function to generate daily participant summary"""
    async with NSEDataIngestionPipeline() as pipeline:
        return await pipeline.generate_daily_summary(target_date) 