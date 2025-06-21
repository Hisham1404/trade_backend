"""
Participant Flow Metrics Calculation Service

This service provides comprehensive calculation of financial and behavioral metrics
from participant flow data, including statistical analysis, pattern detection,
and real-time metric updates.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

# Internal imports
from app.database.connection import get_db
from app.models.participant_flow import (
    ParticipantActivity, ParticipantFlowMetrics, ParticipantBehaviorPattern,
    ParticipantFlowSummary, ParticipantProfile,
    ParticipantType, MarketSegment, DataSource, DataQuality
)
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class MetricsCalculationResult:
    """Result of metrics calculation operation"""
    success: bool
    metrics_calculated: int = 0
    patterns_detected: int = 0
    processing_time_ms: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

@dataclass
class FlowMetrics:
    """Container for calculated flow metrics"""
    # Volume-based metrics
    total_volume: Decimal
    net_flow: Decimal
    gross_turnover: Decimal
    volume_weighted_avg_price: Optional[Decimal] = None
    
    # Momentum indicators
    flow_momentum_1d: Decimal = Decimal('0.00')
    flow_momentum_5d: Decimal = Decimal('0.00')
    flow_momentum_20d: Decimal = Decimal('0.00')
    
    # Volatility measures
    flow_volatility_5d: Decimal = Decimal('0.00')
    flow_volatility_20d: Decimal = Decimal('0.00')
    
    # Trend indicators
    flow_direction_score: Decimal = Decimal('0.00')  # -1 to 1
    trend_strength: Decimal = Decimal('0.00')  # 0 to 1
    
    # Relative strength
    relative_flow_strength: Decimal = Decimal('0.00')  # vs market average
    participant_concentration: Decimal = Decimal('0.00')  # Herfindahl index
    
    # Statistical measures
    flow_zscore: Decimal = Decimal('0.00')
    flow_percentile: Decimal = Decimal('0.00')
    
    # Behavioral indicators
    consistency_score: Decimal = Decimal('0.00')  # Pattern consistency
    aggressiveness_score: Decimal = Decimal('0.00')  # Trading aggressiveness

@dataclass
class BehaviorPattern:
    """Container for detected behavior patterns"""
    pattern_type: str
    confidence_score: float
    start_date: date
    end_date: date
    description: str
    supporting_metrics: Dict[str, Any]

class ParticipantMetricsCalculator:
    """Core metrics calculation engine for participant flow data"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.lookback_periods = {
            'short': 5,    # 5 trading days
            'medium': 20,  # 20 trading days  
            'long': 60     # 60 trading days
        }
    
    def calculate_flow_metrics(
        self, 
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        calculation_date: date,
        lookback_days: int = 20
    ) -> Optional[FlowMetrics]:
        """Calculate comprehensive flow metrics for a participant type"""
        
        try:
            # Get historical data for the participant
            end_date = calculation_date
            start_date = end_date - timedelta(days=lookback_days * 2)  # Extra buffer for weekends
            
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
            
            if not activities:
                logger.warning(f"No activities found for {participant_type.value} in {market_segment.value}")
                return None
            
            # Convert to pandas DataFrame for easier analysis
            df = self._activities_to_dataframe(activities)
            
            if len(df) < 5:  # Need minimum data points
                logger.warning(f"Insufficient data points: {len(df)}")
                return None
            
            # Calculate various metrics
            metrics = FlowMetrics(
                total_volume=self._calculate_total_volume(df),
                net_flow=self._calculate_net_flow(df),
                gross_turnover=self._calculate_gross_turnover(df)
            )
            
            # Momentum indicators
            metrics.flow_momentum_1d = self._calculate_momentum(df, 1)
            metrics.flow_momentum_5d = self._calculate_momentum(df, 5)
            metrics.flow_momentum_20d = self._calculate_momentum(df, 20)
            
            # Volatility measures
            metrics.flow_volatility_5d = self._calculate_volatility(df, 5)
            metrics.flow_volatility_20d = self._calculate_volatility(df, 20)
            
            # Trend analysis
            metrics.flow_direction_score = self._calculate_direction_score(df)
            metrics.trend_strength = self._calculate_trend_strength(df)
            
            # Relative strength (requires market data)
            metrics.relative_flow_strength = self._calculate_relative_strength(
                df, participant_type, market_segment, calculation_date
            )
            
            # Statistical measures
            metrics.flow_zscore = self._calculate_zscore(df)
            metrics.flow_percentile = self._calculate_percentile(df)
            
            # Behavioral indicators
            metrics.consistency_score = self._calculate_consistency_score(df)
            metrics.aggressiveness_score = self._calculate_aggressiveness_score(df)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating flow metrics: {str(e)}")
            return None
    
    def _activities_to_dataframe(self, activities: List[ParticipantActivity]) -> pd.DataFrame:
        """Convert activity records to pandas DataFrame"""
        data = []
        for activity in activities:
            data.append({
                'date': activity.trade_date,
                'net_value': float(activity.net_value),
                'gross_turnover': float(activity.gross_turnover),
                'buy_value': float(activity.buy_value),
                'sell_value': float(activity.sell_value),
                'buy_quantity': activity.buy_quantity,
                'sell_quantity': activity.sell_quantity,
                'net_quantity': activity.net_quantity
            })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def _calculate_total_volume(self, df: pd.DataFrame) -> Decimal:
        """Calculate total volume over the period"""
        return Decimal(str(df['gross_turnover'].sum()))
    
    def _calculate_net_flow(self, df: pd.DataFrame) -> Decimal:
        """Calculate net flow over the period"""
        return Decimal(str(df['net_value'].sum()))
    
    def _calculate_gross_turnover(self, df: pd.DataFrame) -> Decimal:
        """Calculate gross turnover over the period"""
        return Decimal(str(df['gross_turnover'].sum()))
    
    def _calculate_momentum(self, df: pd.DataFrame, days: int) -> Decimal:
        """Calculate flow momentum over specified days"""
        if len(df) < days + 1:
            return Decimal('0.00')
        
        try:
            recent_avg = df['net_value'].tail(days).mean()
            previous_avg = df['net_value'].tail(days * 2).head(days).mean()
            
            if previous_avg == 0:
                return Decimal('0.00')
            
            momentum = (recent_avg - previous_avg) / abs(previous_avg)
            return Decimal(str(momentum)).quantize(Decimal('0.0001'))
            
        except Exception:
            return Decimal('0.00')
    
    def _calculate_volatility(self, df: pd.DataFrame, days: int) -> Decimal:
        """Calculate flow volatility over specified days"""
        if len(df) < days:
            return Decimal('0.00')
        
        try:
            recent_data = df['net_value'].tail(days)
            volatility = recent_data.std()
            return Decimal(str(volatility)).quantize(Decimal('0.0001'))
            
        except Exception:
            return Decimal('0.00')
    
    def _calculate_direction_score(self, df: pd.DataFrame) -> Decimal:
        """Calculate flow direction score (-1 to 1)"""
        try:
            net_flows = df['net_value']
            positive_days = (net_flows > 0).sum()
            total_days = len(net_flows)
            
            if total_days == 0:
                return Decimal('0.00')
            
            # Score from -1 (all negative) to 1 (all positive)
            direction_score = (2 * positive_days / total_days) - 1
            return Decimal(str(direction_score)).quantize(Decimal('0.0001'))
            
        except Exception:
            return Decimal('0.00')
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> Decimal:
        """Calculate trend strength (0 to 1)"""
        try:
            if len(df) < 5:
                return Decimal('0.00')
            
            # Use linear regression to measure trend strength
            x = np.arange(len(df))
            y = df['net_value'].values
            
            # Calculate correlation coefficient as trend strength
            correlation = np.corrcoef(x, y)[0, 1]
            
            if np.isnan(correlation):
                return Decimal('0.00')
            
            # Convert to absolute value (0 to 1 scale)
            trend_strength = abs(correlation)
            return Decimal(str(trend_strength)).quantize(Decimal('0.0001'))
            
        except Exception:
            return Decimal('0.00')
    
    def _calculate_relative_strength(
        self, 
        df: pd.DataFrame, 
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        calculation_date: date
    ) -> Decimal:
        """Calculate relative flow strength vs market average"""
        try:
            # Get market average for the same period
            participant_avg = df['net_value'].mean()
            
            # Get all participants' data for comparison
            market_avg = self._get_market_average(market_segment, calculation_date)
            
            if market_avg == 0:
                return Decimal('0.00')
            
            relative_strength = participant_avg / market_avg
            return Decimal(str(relative_strength)).quantize(Decimal('0.0001'))
            
        except Exception:
            return Decimal('0.00')
    
    def _get_market_average(self, market_segment: MarketSegment, calculation_date: date) -> float:
        """Get market average net flow for comparison"""
        try:
            # Get average net flow across all participants for the segment
            end_date = calculation_date
            start_date = end_date - timedelta(days=20)
            
            avg_flow = self.db_session.query(
                func.avg(ParticipantActivity.net_value)
            ).filter(
                and_(
                    ParticipantActivity.market_segment == market_segment,
                    ParticipantActivity.trade_date >= start_date,
                    ParticipantActivity.trade_date <= end_date
                )
            ).scalar()
            
            return float(avg_flow) if avg_flow else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_zscore(self, df: pd.DataFrame) -> Decimal:
        """Calculate Z-score for recent net flow"""
        try:
            if len(df) < 5:
                return Decimal('0.00')
            
            recent_flow = df['net_value'].iloc[-1]
            mean_flow = df['net_value'].mean()
            std_flow = df['net_value'].std()
            
            if std_flow == 0:
                return Decimal('0.00')
            
            zscore = (recent_flow - mean_flow) / std_flow
            return Decimal(str(zscore)).quantize(Decimal('0.0001'))
            
        except Exception:
            return Decimal('0.00')
    
    def _calculate_percentile(self, df: pd.DataFrame) -> Decimal:
        """Calculate percentile of recent flow vs historical"""
        try:
            if len(df) < 5:
                return Decimal('50.00')
            
            recent_flow = df['net_value'].iloc[-1]
            percentile = (df['net_value'] <= recent_flow).mean() * 100
            
            return Decimal(str(percentile)).quantize(Decimal('0.01'))
            
        except Exception:
            return Decimal('50.00')
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> Decimal:
        """Calculate consistency score based on flow patterns"""
        try:
            if len(df) < 5:
                return Decimal('0.00')
            
            # Measure consistency as inverse of coefficient of variation
            mean_flow = abs(df['net_value'].mean())
            std_flow = df['net_value'].std()
            
            if mean_flow == 0:
                return Decimal('0.00')
            
            cv = std_flow / mean_flow
            consistency = 1 / (1 + cv)  # Scale from 0 to 1
            
            return Decimal(str(consistency)).quantize(Decimal('0.0001'))
            
        except Exception:
            return Decimal('0.00')
    
    def _calculate_aggressiveness_score(self, df: pd.DataFrame) -> Decimal:
        """Calculate trading aggressiveness score"""
        try:
            if len(df) < 2:
                return Decimal('0.00')
            
            # Measure as frequency and magnitude of large moves
            daily_changes = df['net_value'].diff().abs()
            threshold = daily_changes.quantile(0.75)  # Top quartile
            
            aggressive_days = (daily_changes > threshold).sum()
            total_days = len(daily_changes) - 1  # Exclude first NaN
            
            if total_days == 0:
                return Decimal('0.00')
            
            aggressiveness = aggressive_days / total_days
            return Decimal(str(aggressiveness)).quantize(Decimal('0.0001'))
            
        except Exception:
            return Decimal('0.00')

class BehaviorPatternDetector:
    """Detects behavioral patterns in participant flow data"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def detect_patterns(
        self,
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        analysis_date: date,
        lookback_days: int = 60
    ) -> List[BehaviorPattern]:
        """Detect various behavioral patterns"""
        
        patterns = []
        
        try:
            # Get historical data
            activities = self._get_participant_activities(
                participant_type, market_segment, analysis_date, lookback_days
            )
            
            if len(activities) < 10:  # Need sufficient data
                return patterns
            
            df = self._activities_to_dataframe(activities)
            
            # Detect different pattern types
            patterns.extend(self._detect_trend_patterns(df))
            patterns.extend(self._detect_seasonal_patterns(df))
            patterns.extend(self._detect_volatility_patterns(df))
            patterns.extend(self._detect_reversal_patterns(df))
            patterns.extend(self._detect_momentum_patterns(df))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return patterns
    
    def _get_participant_activities(
        self,
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        analysis_date: date,
        lookback_days: int
    ) -> List[ParticipantActivity]:
        """Get participant activities for pattern analysis"""
        
        end_date = analysis_date
        start_date = end_date - timedelta(days=lookback_days * 2)
        
        return self.db_session.query(ParticipantActivity).join(
            ParticipantProfile
        ).filter(
            and_(
                ParticipantProfile.participant_type == participant_type,
                ParticipantActivity.market_segment == market_segment,
                ParticipantActivity.trade_date >= start_date,
                ParticipantActivity.trade_date <= end_date
            )
        ).order_by(ParticipantActivity.trade_date).all()
    
    def _activities_to_dataframe(self, activities: List[ParticipantActivity]) -> pd.DataFrame:
        """Convert activities to DataFrame for analysis"""
        data = []
        for activity in activities:
            data.append({
                'date': activity.trade_date,
                'net_value': float(activity.net_value),
                'gross_turnover': float(activity.gross_turnover),
                'buy_value': float(activity.buy_value),
                'sell_value': float(activity.sell_value)
            })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def _detect_trend_patterns(self, df: pd.DataFrame) -> List[BehaviorPattern]:
        """Detect trend-based patterns"""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
            
            # Detect persistent buying/selling trends
            net_flows = df['net_value']
            
            # Look for consecutive periods of buying or selling
            consecutive_buys = 0
            consecutive_sells = 0
            max_consecutive_buys = 0
            max_consecutive_sells = 0
            
            for flow in net_flows:
                if flow > 0:
                    consecutive_buys += 1
                    consecutive_sells = 0
                    max_consecutive_buys = max(max_consecutive_buys, consecutive_buys)
                elif flow < 0:
                    consecutive_sells += 1
                    consecutive_buys = 0
                    max_consecutive_sells = max(max_consecutive_sells, consecutive_sells)
                else:
                    consecutive_buys = 0
                    consecutive_sells = 0
            
            # Pattern: Persistent Buying
            if max_consecutive_buys >= 5:
                patterns.append(BehaviorPattern(
                    pattern_type="PERSISTENT_BUYING",
                    confidence_score=min(0.9, max_consecutive_buys / 10),
                    start_date=df['date'].iloc[0].date(),
                    end_date=df['date'].iloc[-1].date(),
                    description=f"Detected {max_consecutive_buys} consecutive days of net buying",
                    supporting_metrics={
                        'consecutive_days': max_consecutive_buys,
                        'avg_daily_flow': net_flows[net_flows > 0].mean()
                    }
                ))
            
            # Pattern: Persistent Selling
            if max_consecutive_sells >= 5:
                patterns.append(BehaviorPattern(
                    pattern_type="PERSISTENT_SELLING",
                    confidence_score=min(0.9, max_consecutive_sells / 10),
                    start_date=df['date'].iloc[0].date(),
                    end_date=df['date'].iloc[-1].date(),
                    description=f"Detected {max_consecutive_sells} consecutive days of net selling",
                    supporting_metrics={
                        'consecutive_days': max_consecutive_sells,
                        'avg_daily_flow': net_flows[net_flows < 0].mean()
                    }
                ))
            
        except Exception as e:
            logger.error(f"Error detecting trend patterns: {str(e)}")
        
        return patterns
    
    def _detect_seasonal_patterns(self, df: pd.DataFrame) -> List[BehaviorPattern]:
        """Detect seasonal/cyclical patterns"""
        patterns = []
        
        try:
            if len(df) < 30:
                return patterns
            
            # Add day of week analysis
            df['day_of_week'] = df['date'].dt.dayofweek
            
            # Check for day-of-week patterns
            dow_stats = df.groupby('day_of_week')['net_value'].agg(['mean', 'count'])
            
            # Look for strong day-of-week biases
            max_day = dow_stats['mean'].idxmax()
            min_day = dow_stats['mean'].idxmin()
            
            if dow_stats['mean'].std() > 0 and dow_stats['count'].min() >= 3:
                range_ratio = abs(dow_stats['mean'].max() - dow_stats['mean'].min()) / dow_stats['mean'].std()
                
                if range_ratio > 2:  # Strong pattern
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    patterns.append(BehaviorPattern(
                        pattern_type="DAY_OF_WEEK_BIAS",
                        confidence_score=min(0.8, range_ratio / 5),
                        start_date=df['date'].iloc[0].date(),
                        end_date=df['date'].iloc[-1].date(),
                        description=f"Strong {day_names[max_day]} bias (avg: {dow_stats['mean'].iloc[max_day]:.2f})",
                        supporting_metrics={
                            'strongest_day': day_names[max_day],
                            'weakest_day': day_names[min_day],
                            'avg_difference': float(dow_stats['mean'].max() - dow_stats['mean'].min())
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error detecting seasonal patterns: {str(e)}")
        
        return patterns
    
    def _detect_volatility_patterns(self, df: pd.DataFrame) -> List[BehaviorPattern]:
        """Detect volatility-based patterns"""
        patterns = []
        
        try:
            if len(df) < 15:
                return patterns
            
            # Calculate rolling volatility
            df['rolling_vol'] = df['net_value'].rolling(window=5).std()
            
            # Detect volatility regime changes
            recent_vol = df['rolling_vol'].tail(5).mean()
            historical_vol = df['rolling_vol'].head(-5).mean()
            
            if pd.notna(recent_vol) and pd.notna(historical_vol) and historical_vol > 0:
                vol_ratio = recent_vol / historical_vol
                
                # High volatility regime
                if vol_ratio > 2.0:
                    patterns.append(BehaviorPattern(
                        pattern_type="HIGH_VOLATILITY_REGIME",
                        confidence_score=min(0.9, (vol_ratio - 1) / 3),
                        start_date=df['date'].iloc[-5].date(),
                        end_date=df['date'].iloc[-1].date(),
                        description=f"Entered high volatility regime (vol increased by {(vol_ratio-1)*100:.1f}%)",
                        supporting_metrics={
                            'volatility_ratio': float(vol_ratio),
                            'recent_volatility': float(recent_vol),
                            'historical_volatility': float(historical_vol)
                        }
                    ))
                
                # Low volatility regime
                elif vol_ratio < 0.5:
                    patterns.append(BehaviorPattern(
                        pattern_type="LOW_VOLATILITY_REGIME",
                        confidence_score=min(0.9, (1 - vol_ratio) / 0.5),
                        start_date=df['date'].iloc[-5].date(),
                        end_date=df['date'].iloc[-1].date(),
                        description=f"Entered low volatility regime (vol decreased by {(1-vol_ratio)*100:.1f}%)",
                        supporting_metrics={
                            'volatility_ratio': float(vol_ratio),
                            'recent_volatility': float(recent_vol),
                            'historical_volatility': float(historical_vol)
                        }
                    ))
            
        except Exception as e:
            logger.error(f"Error detecting volatility patterns: {str(e)}")
        
        return patterns
    
    def _detect_reversal_patterns(self, df: pd.DataFrame) -> List[BehaviorPattern]:
        """Detect flow reversal patterns"""
        patterns = []
        
        try:
            if len(df) < 10:
                return patterns
            
            net_flows = df['net_value']
            
            # Look for significant reversals
            for i in range(3, len(net_flows) - 2):
                # Check for reversal: consistent direction before, opposite after
                before_flows = net_flows.iloc[i-3:i]
                after_flows = net_flows.iloc[i+1:i+3]
                
                before_direction = (before_flows > 0).mean()
                after_direction = (after_flows > 0).mean()
                
                # Strong reversal conditions
                if (before_direction > 0.8 and after_direction < 0.2) or \
                   (before_direction < 0.2 and after_direction > 0.8):
                    
                    reversal_magnitude = abs(before_flows.mean() - after_flows.mean())
                    
                    if reversal_magnitude > net_flows.std():
                        pattern_type = "BUYING_TO_SELLING" if before_direction > 0.5 else "SELLING_TO_BUYING"
                        
                        patterns.append(BehaviorPattern(
                            pattern_type=pattern_type,
                            confidence_score=min(0.9, reversal_magnitude / (2 * net_flows.std())),
                            start_date=df['date'].iloc[i-2].date(),
                            end_date=df['date'].iloc[i+2].date(),
                            description=f"Flow reversal detected on {df['date'].iloc[i].date()}",
                            supporting_metrics={
                                'reversal_magnitude': float(reversal_magnitude),
                                'before_avg': float(before_flows.mean()),
                                'after_avg': float(after_flows.mean())
                            }
                        ))
                        break  # Only detect one major reversal per analysis
            
        except Exception as e:
            logger.error(f"Error detecting reversal patterns: {str(e)}")
        
        return patterns
    
    def _detect_momentum_patterns(self, df: pd.DataFrame) -> List[BehaviorPattern]:
        """Detect momentum-based patterns"""
        patterns = []
        
        try:
            if len(df) < 10:
                return patterns
            
            # Calculate momentum indicators
            df['momentum_3d'] = df['net_value'].rolling(3).mean()
            df['momentum_10d'] = df['net_value'].rolling(10).mean()
            
            # Recent momentum vs longer-term momentum
            recent_momentum = df['momentum_3d'].tail(3).mean()
            longer_momentum = df['momentum_10d'].tail(10).mean()
            
            if pd.notna(recent_momentum) and pd.notna(longer_momentum):
                momentum_diff = recent_momentum - longer_momentum
                momentum_std = df['net_value'].std()
                
                if momentum_std > 0:
                    momentum_zscore = abs(momentum_diff) / momentum_std
                    
                    # Accelerating momentum
                    if momentum_zscore > 1.5:
                        pattern_type = "ACCELERATING_BUYING" if momentum_diff > 0 else "ACCELERATING_SELLING"
                        
                        patterns.append(BehaviorPattern(
                            pattern_type=pattern_type,
                            confidence_score=min(0.9, momentum_zscore / 3),
                            start_date=df['date'].iloc[-10].date(),
                            end_date=df['date'].iloc[-1].date(),
                            description=f"Momentum acceleration detected",
                            supporting_metrics={
                                'momentum_zscore': float(momentum_zscore),
                                'recent_momentum': float(recent_momentum),
                                'longer_momentum': float(longer_momentum)
                            }
                        ))
            
        except Exception as e:
            logger.error(f"Error detecting momentum patterns: {str(e)}")
        
        return patterns

class ParticipantMetricsService:
    """Main service for participant flow metrics calculation and management"""
    
    def __init__(self, db_session: Session = None):
        self.db_session = db_session or next(get_db())
        self.calculator = ParticipantMetricsCalculator(self.db_session)
    
    async def calculate_and_store_metrics(
        self,
        calculation_date: date = None,
        participant_types: List[ParticipantType] = None,
        market_segments: List[MarketSegment] = None
    ) -> MetricsCalculationResult:
        """Calculate and store metrics for specified participants and segments"""
        
        start_time = datetime.now()
        result = MetricsCalculationResult(success=False)
        
        try:
            if calculation_date is None:
                calculation_date = date.today()
            
            if participant_types is None:
                participant_types = list(ParticipantType)
            
            if market_segments is None:
                market_segments = list(MarketSegment)
            
            metrics_count = 0
            
            # Calculate metrics for each combination
            for participant_type in participant_types:
                for market_segment in market_segments:
                    try:
                        # Calculate flow metrics
                        flow_metrics = self.calculator.calculate_flow_metrics(
                            participant_type, market_segment, calculation_date
                        )
                        
                        if flow_metrics:
                            # Store metrics in database
                            await self._store_flow_metrics(
                                participant_type, market_segment, calculation_date, flow_metrics
                            )
                            metrics_count += 1
                        
                    except Exception as e:
                        error_msg = f"Error processing {participant_type.value}-{market_segment.value}: {str(e)}"
                        result.errors.append(error_msg)
                        logger.error(error_msg)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result.success = metrics_count > 0
            result.metrics_calculated = metrics_count
            result.patterns_detected = 0
            result.processing_time_ms = processing_time
            
            if result.success:
                logger.info(f"Calculated {metrics_count} metrics")
            
        except Exception as e:
            result.errors.append(f"Metrics calculation failed: {str(e)}")
            logger.error(f"Metrics calculation service failed: {str(e)}")
        
        return result
    
    async def _store_flow_metrics(
        self,
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        calculation_date: date,
        metrics: FlowMetrics
    ):
        """Store calculated flow metrics in database"""
        
        try:
            # Get or create participant profile
            participant = self.db_session.query(ParticipantProfile).filter(
                ParticipantProfile.participant_type == participant_type,
                ParticipantProfile.participant_code.is_(None)
            ).first()
            
            if not participant:
                participant = ParticipantProfile(
                    participant_type=participant_type,
                    participant_name=f"{participant_type.value} Participants",
                    is_active=True
                )
                self.db_session.add(participant)
                self.db_session.flush()
            
            # Create metrics record using actual model fields
            metrics_record = ParticipantFlowMetrics(
                calculation_date=calculation_date,
                participant_id=participant.id,
                market_segment=market_segment,
                
                # Required fields from model
                period_type="daily",
                period_days=20,
                overall_flow=self._determine_flow_direction(metrics.net_flow),
                flow_strength=float(metrics.trend_strength),
                flow_consistency=float(metrics.consistency_score),
                
                # Volume and flow metrics
                avg_daily_net_value=float(metrics.net_flow) / 20,  # Average over period
                avg_daily_turnover=float(metrics.gross_turnover) / 20,
                
                # Trend indicators
                trend_direction=self._determine_trend_direction(metrics.flow_direction_score),
                trend_strength=float(metrics.trend_strength),
                
                # Volatility measures
                flow_volatility=float(metrics.flow_volatility_20d),
                
                # Behavioral indicators
                activity_pattern=self._determine_activity_pattern(metrics.aggressiveness_score),
                concentration_ratio=float(metrics.consistency_score),
                
                # Market impact
                market_correlation=float(metrics.relative_flow_strength) if metrics.relative_flow_strength else None,
                
                # Statistical metadata
                sample_size=20,
                confidence_interval=Decimal('0.95'),
                calculation_method="statistical_analysis"
            )
            
            self.db_session.merge(metrics_record)  # Use merge to handle duplicates
            self.db_session.commit()
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error storing flow metrics: {str(e)}")
            raise
    
    def _determine_flow_direction(self, net_flow: Decimal) -> str:
        """Determine flow direction from net flow value"""
        from app.models.participant_flow import FlowDirection
        
        if net_flow > Decimal('1000'):  # Significant inflow
            return FlowDirection.INFLOW
        elif net_flow < Decimal('-1000'):  # Significant outflow
            return FlowDirection.OUTFLOW
        else:
            return FlowDirection.NEUTRAL
    
    def _determine_trend_direction(self, direction_score: Decimal) -> str:
        """Determine trend direction from direction score"""
        if direction_score > Decimal('0.3'):
            return "increasing"
        elif direction_score < Decimal('-0.3'):
            return "decreasing"
        else:
            return "stable"
    
    def _determine_activity_pattern(self, aggressiveness_score: Decimal) -> str:
        """Determine activity pattern from aggressiveness score"""
        if aggressiveness_score > Decimal('0.7'):
            return "momentum"
        elif aggressiveness_score < Decimal('0.3'):
            return "contrarian"
        else:
            return "random"
    

    
    async def get_latest_metrics(
        self,
        participant_type: ParticipantType,
        market_segment: MarketSegment,
        days_back: int = 30
    ) -> List[ParticipantFlowMetrics]:
        """Get latest metrics for a participant type and market segment"""
        
        try:
            cutoff_date = date.today() - timedelta(days=days_back)
            
            metrics = self.db_session.query(ParticipantFlowMetrics).join(
                ParticipantProfile
            ).filter(
                and_(
                    ParticipantProfile.participant_type == participant_type,
                    ParticipantFlowMetrics.market_segment == market_segment,
                    ParticipantFlowMetrics.calculation_date >= cutoff_date
                )
            ).order_by(desc(ParticipantFlowMetrics.calculation_date)).all()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving metrics: {str(e)}")
            return []
    


# Convenience functions
def calculate_participant_metrics(calculation_date: date = None) -> MetricsCalculationResult:
    """Convenience function to calculate metrics for all participants"""
    import asyncio
    service = ParticipantMetricsService()
    return asyncio.run(service.calculate_and_store_metrics(calculation_date))

def get_participant_metrics(
    participant_type: ParticipantType,
    market_segment: MarketSegment,
    days_back: int = 30
) -> List[ParticipantFlowMetrics]:
    """Convenience function to get participant metrics"""
    import asyncio
    service = ParticipantMetricsService()
    return asyncio.run(service.get_latest_metrics(participant_type, market_segment, days_back)) 