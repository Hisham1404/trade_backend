"""
Behavioral Shift Detection Service

This service provides advanced detection of significant changes in participant behavior
patterns, including trend reversals, volume shifts, and behavioral anomalies with
real-time monitoring and alerting capabilities.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

# Internal imports
from app.database.connection import get_db
from app.models.participant_flow import (
    ParticipantActivity, ParticipantFlowMetrics, ParticipantBehaviorPattern,
    ParticipantFlowEvent, ParticipantProfile,
    ParticipantType, MarketSegment, FlowDirection
)
from app.services.participant_metrics_service import ParticipantMetricsService

logger = logging.getLogger(__name__)

class ShiftType(Enum):
    """Types of behavioral shifts that can be detected"""
    TREND_REVERSAL = "trend_reversal"           # Bullish to bearish or vice versa
    VOLUME_SPIKE = "volume_spike"               # Sudden increase in activity
    FLOW_ACCELERATION = "flow_acceleration"     # Increasing momentum
    FLOW_DECELERATION = "flow_deceleration"     # Decreasing momentum
    VOLATILITY_CHANGE = "volatility_change"     # Change in flow volatility
    SECTOR_ROTATION = "sector_rotation"         # Shift between market segments
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"   # Unusual trading patterns
    CONCENTRATION_SHIFT = "concentration_shift" # Change in participant concentration

class ShiftSeverity(Enum):
    """Severity levels for detected shifts"""
    LOW = "low"           # Minor shift, informational
    MEDIUM = "medium"     # Moderate shift, monitor
    HIGH = "high"         # Significant shift, attention required
    CRITICAL = "critical" # Major shift, immediate action needed

@dataclass
class BehavioralShift:
    """Container for detected behavioral shift"""
    shift_type: ShiftType
    severity: ShiftSeverity
    participant_type: ParticipantType
    market_segment: MarketSegment
    detection_date: date
    confidence_score: float
    
    # Shift details
    description: str
    previous_state: Dict[str, Any]
    current_state: Dict[str, Any]
    
    # Statistical significance
    statistical_significance: float
    magnitude: float
    duration_estimate: Optional[str] = None
    
    # Impact assessment
    market_impact: Optional[str] = None  # "bullish", "bearish", "neutral"
    affected_instruments: List[str] = None
    
    # Recommendations
    action_required: bool = False
    recommended_actions: List[str] = None
    
    def __post_init__(self):
        if self.affected_instruments is None:
            self.affected_instruments = []
        if self.recommended_actions is None:
            self.recommended_actions = []

@dataclass
class ShiftDetectionResult:
    """Result of shift detection operation"""
    success: bool
    detection_date: date
    shifts_detected: List[BehavioralShift]
    processing_time_ms: float
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class StatisticalAnalyzer:
    """Statistical analysis methods for shift detection"""
    
    @staticmethod
    def calculate_z_score(current_value: float, historical_values: List[float]) -> float:
        """Calculate Z-score for current value vs historical distribution"""
        if len(historical_values) < 2:
            return 0.0
            
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        if std == 0:
            return 0.0
            
        return abs(current_value - mean) / std
    
    @staticmethod
    def detect_trend_change(values: List[float], window_size: int = 5) -> Tuple[bool, float]:
        """Detect trend change using moving averages"""
        if len(values) < window_size * 2:
            return False, 0.0
            
        recent_avg = np.mean(values[-window_size:])
        previous_avg = np.mean(values[-window_size*2:-window_size])
        
        if previous_avg == 0:
            return False, 0.0
            
        change_magnitude = abs(recent_avg - previous_avg) / abs(previous_avg)
        trend_changed = np.sign(recent_avg) != np.sign(previous_avg)
        
        return trend_changed, change_magnitude
    
    @staticmethod
    def calculate_volatility_change(recent_values: List[float], historical_values: List[float]) -> float:
        """Calculate change in volatility between periods"""
        if len(recent_values) < 2 or len(historical_values) < 2:
            return 0.0
            
        recent_vol = np.std(recent_values)
        historical_vol = np.std(historical_values)
        
        if historical_vol == 0:
            return 0.0
            
        return recent_vol / historical_vol
    
    @staticmethod
    def detect_outliers(values: List[float], threshold: float = 2.0) -> List[int]:
        """Detect outliers using Z-score method"""
        if len(values) < 3:
            return []
            
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
            
        outliers = []
        for i, value in enumerate(values):
            z_score = abs(value - mean) / std
            if z_score > threshold:
                outliers.append(i)
                
        return outliers

class BehavioralShiftDetector:
    """Core behavioral shift detection engine"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.analyzer = StatisticalAnalyzer()
        self.metrics_service = ParticipantMetricsService(db_session)
    
    async def detect_shifts(
        self,
        detection_date: date = None,
        participant_types: List[ParticipantType] = None,
        market_segments: List[MarketSegment] = None,
        lookback_days: int = 30
    ) -> ShiftDetectionResult:
        """Detect behavioral shifts for specified participants and segments"""
        
        start_time = datetime.now()
        result = ShiftDetectionResult(
            success=False,
            detection_date=detection_date or date.today(),
            shifts_detected=[],
            processing_time_ms=0.0
        )
        
        try:
            if participant_types is None:
                participant_types = list(ParticipantType)
            if market_segments is None:
                market_segments = list(MarketSegment)
            
            all_shifts = []
            
            # Analyze each participant-segment combination
            for participant_type in participant_types:
                for market_segment in market_segments:
                    try:
                        shifts = await self._analyze_participant_segment(
                            participant_type, market_segment, result.detection_date, lookback_days
                        )
                        all_shifts.extend(shifts)
                        
                    except Exception as e:
                        error_msg = f"Error analyzing {participant_type.value}-{market_segment.value}: {str(e)}"
                        result.errors.append(error_msg)
                        logger.error(error_msg)
            
            # Sort shifts by severity and confidence
            all_shifts.sort(key=lambda s: (s.severity.value, s.confidence_score), reverse=True)
            
            result.shifts_detected = all_shifts
            result.success = len(all_shifts) > 0 or len(result.errors) == 0
            result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Detected {len(all_shifts)} behavioral shifts")
            
        except Exception as e:
            result.errors.append(f"Shift detection failed: {str(e)}")
            logger.error(f"Behavioral shift detection failed: {str(e)}")
        
        return result
    
    async def _analyze_participant_segment(
        self,
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        detection_date: date,
        lookback_days: int
    ) -> List[BehavioralShift]:
        """Analyze a specific participant-segment combination for shifts"""
        
        shifts = []
        
        try:
            # Get historical activity data
            activities = await self._get_historical_activities(
                participant_type, market_segment, detection_date, lookback_days
            )
            
            if len(activities) < 10:  # Need sufficient data
                return shifts
            
            # Convert to DataFrame for analysis
            df = self._activities_to_dataframe(activities)
            
            # Run various shift detection algorithms
            shifts.extend(await self._detect_trend_reversals(df, participant_type, market_segment, detection_date))
            shifts.extend(await self._detect_volume_spikes(df, participant_type, market_segment, detection_date))
            
        except Exception as e:
            logger.error(f"Error in participant analysis: {str(e)}")
        
        return shifts
    
    async def _get_historical_activities(
        self,
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        end_date: date,
        days: int
    ) -> List[ParticipantActivity]:
        """Get historical activities for analysis"""
        
        start_date = end_date - timedelta(days=days)
        
        activities = self.db_session.query(ParticipantActivity).join(
            ParticipantProfile
        ).filter(
            and_(
                ParticipantProfile.participant_type == participant_type,
                ParticipantActivity.market_segment == market_segment,
                ParticipantActivity.trade_date >= start_date,
                ParticipantActivity.trade_date <= end_date
            )
        ).order_by(ParticipantActivity.trade_date).all()
        
        return activities
    
    def _activities_to_dataframe(self, activities: List[ParticipantActivity]) -> pd.DataFrame:
        """Convert activities to DataFrame for analysis"""
        data = []
        for activity in activities:
            data.append({
                'date': activity.trade_date,
                'net_value': float(activity.net_value),
                'gross_turnover': float(activity.gross_turnover),
                'buy_value': float(activity.buy_value),
                'sell_value': float(activity.sell_value),
                'buy_quantity': activity.buy_quantity,
                'sell_quantity': activity.sell_quantity
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    async def _detect_trend_reversals(
        self,
        df: pd.DataFrame,
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        detection_date: date
    ) -> List[BehavioralShift]:
        """Detect trend reversals in participant behavior"""
        
        shifts = []
        
        try:
            if len(df) < 10:
                return shifts
            
            net_values = df['net_value'].tolist()
            trend_changed, magnitude = self.analyzer.detect_trend_change(net_values)
            
            if trend_changed and magnitude > 0.3:  # Significant trend change
                
                # Determine severity based on magnitude
                if magnitude > 0.8:
                    severity = ShiftSeverity.CRITICAL
                elif magnitude > 0.6:
                    severity = ShiftSeverity.HIGH
                elif magnitude > 0.4:
                    severity = ShiftSeverity.MEDIUM
                else:
                    severity = ShiftSeverity.LOW
                
                # Calculate confidence score
                confidence = min(0.95, magnitude)
                
                # Determine market impact
                recent_avg = np.mean(net_values[-5:])
                market_impact = "bullish" if recent_avg > 0 else "bearish"
                
                shift = BehavioralShift(
                    shift_type=ShiftType.TREND_REVERSAL,
                    severity=severity,
                    participant_type=participant_type,
                    market_segment=market_segment,
                    detection_date=detection_date,
                    confidence_score=confidence,
                    description=f"Trend reversal detected: {magnitude:.1%} change in flow direction",
                    previous_state={
                        'avg_flow': float(np.mean(net_values[-10:-5])),
                        'trend': "bullish" if np.mean(net_values[-10:-5]) > 0 else "bearish"
                    },
                    current_state={
                        'avg_flow': float(recent_avg),
                        'trend': market_impact
                    },
                    statistical_significance=confidence,
                    magnitude=magnitude,
                    market_impact=market_impact,
                    action_required=severity in [ShiftSeverity.HIGH, ShiftSeverity.CRITICAL]
                )
                
                # Add recommended actions for high severity shifts
                if shift.action_required:
                    shift.recommended_actions = [
                        "Monitor position exposure",
                        "Review risk management parameters",
                        "Assess market sentiment indicators"
                    ]
                
                shifts.append(shift)
                
        except Exception as e:
            logger.error(f"Error in trend reversal detection: {str(e)}")
        
        return shifts
    
    async def _detect_volume_spikes(
        self,
        df: pd.DataFrame,
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        detection_date: date
    ) -> List[BehavioralShift]:
        """Detect unusual volume spikes"""
        
        shifts = []
        
        try:
            if len(df) < 5:
                return shifts
            
            volumes = df['gross_turnover'].tolist()
            recent_volume = volumes[-1]
            historical_volumes = volumes[:-1]
            
            z_score = self.analyzer.calculate_z_score(recent_volume, historical_volumes)
            
            if z_score > 2.5:  # Significant volume spike
                
                # Determine severity based on Z-score
                if z_score > 4.0:
                    severity = ShiftSeverity.CRITICAL
                elif z_score > 3.5:
                    severity = ShiftSeverity.HIGH
                elif z_score > 3.0:
                    severity = ShiftSeverity.MEDIUM
                else:
                    severity = ShiftSeverity.LOW
                
                confidence = min(0.95, z_score / 5.0)
                
                shift = BehavioralShift(
                    shift_type=ShiftType.VOLUME_SPIKE,
                    severity=severity,
                    participant_type=participant_type,
                    market_segment=market_segment,
                    detection_date=detection_date,
                    confidence_score=confidence,
                    description=f"Volume spike detected: {z_score:.1f}σ above historical average",
                    previous_state={
                        'avg_volume': float(np.mean(historical_volumes)),
                        'volume_range': [float(np.min(historical_volumes)), float(np.max(historical_volumes))]
                    },
                    current_state={
                        'current_volume': float(recent_volume),
                        'vs_average_multiple': float(recent_volume / np.mean(historical_volumes))
                    },
                    statistical_significance=confidence,
                    magnitude=z_score,
                    action_required=severity in [ShiftSeverity.HIGH, ShiftSeverity.CRITICAL]
                )
                
                if shift.action_required:
                    shift.recommended_actions = [
                        "Investigate volume drivers",
                        "Check for corporate announcements",
                        "Monitor liquidity conditions"
                    ]
                
                shifts.append(shift)
                
        except Exception as e:
            logger.error(f"Error in volume spike detection: {str(e)}")
        
        return shifts

class BehavioralShiftService:
    """Main service for behavioral shift detection and management"""
    
    def __init__(self, db_session: Session = None):
        self.db_session = db_session or next(get_db())
        self.detector = BehavioralShiftDetector(self.db_session)
    
    async def detect_and_store_shifts(
        self,
        detection_date: date = None,
        participant_types: List[ParticipantType] = None,
        market_segments: List[MarketSegment] = None
    ) -> ShiftDetectionResult:
        """Detect behavioral shifts and store significant ones in database"""
        
        result = await self.detector.detect_shifts(
            detection_date, participant_types, market_segments
        )
        
        if result.success and result.shifts_detected:
            await self._store_significant_shifts(result.shifts_detected)
        
        return result
    
    async def _store_significant_shifts(self, shifts: List[BehavioralShift]):
        """Store significant behavioral shifts as flow events"""
        
        try:
            for shift in shifts:
                # Only store medium severity and above
                if shift.severity not in [ShiftSeverity.MEDIUM, ShiftSeverity.HIGH, ShiftSeverity.CRITICAL]:
                    continue
                
                # Get participant profile
                participant = self.db_session.query(ParticipantProfile).filter(
                    ParticipantProfile.participant_type == shift.participant_type,
                    ParticipantProfile.participant_code.is_(None)
                ).first()
                
                if not participant:
                    continue
                
                # Create flow event record
                event = ParticipantFlowEvent(
                    event_date=datetime.combine(shift.detection_date, datetime.min.time()),
                    participant_id=participant.id,
                    event_type=shift.shift_type.value,
                    event_severity=shift.severity.value,
                    event_category="behavioral",
                    event_title=f"{shift.shift_type.value.replace('_', ' ').title()} Detected",
                    event_description=shift.description,
                    magnitude=Decimal(str(shift.magnitude)),
                    significance_score=Decimal(str(shift.statistical_significance)),
                    trigger_conditions={
                        'detection_algorithm': shift.shift_type.value,
                        'previous_state': shift.previous_state,
                        'current_state': shift.current_state
                    },
                    market_impact=shift.market_impact,
                    confidence_level=Decimal(str(shift.confidence_score * 100)),
                    is_active=True
                )
                
                self.db_session.add(event)
            
            self.db_session.commit()
            logger.info(f"Stored {len([s for s in shifts if s.severity != ShiftSeverity.LOW])} significant shifts")
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error storing behavioral shifts: {str(e)}")
            raise

# Convenience functions
async def detect_behavioral_shifts(detection_date: date = None) -> ShiftDetectionResult:
    """Convenience function to detect behavioral shifts"""
    service = BehavioralShiftService()
    return await service.detect_and_store_shifts(detection_date) 