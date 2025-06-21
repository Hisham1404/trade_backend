"""
Insights Generation and Reporting Service

This service provides comprehensive reporting and visualization capabilities to generate
actionable insights from participant flow analysis, including dashboards, automated 
reports, and interactive analytics tools.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import date, datetime, timedelta
from decimal import Decimal
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc, asc

# Internal imports
from app.database.connection import get_db
from app.models.participant_flow import (
    ParticipantActivity, ParticipantFlowMetrics, ParticipantBehaviorPattern,
    ParticipantFlowEvent, ParticipantProfile, ParticipantFlowSummary,
    ParticipantType, MarketSegment, FlowDirection, ActivityType
)
from app.services.participant_metrics_service import ParticipantMetricsService
from app.services.behavioral_shift_detection import BehavioralShiftService

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of reports that can be generated"""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_OVERVIEW = "weekly_overview"  
    MONTHLY_ANALYSIS = "monthly_analysis"
    PARTICIPANT_PROFILE = "participant_profile"
    BEHAVIORAL_SHIFTS = "behavioral_shifts"
    MARKET_SENTIMENT = "market_sentiment"
    FLOW_TRENDS = "flow_trends"
    RISK_ASSESSMENT = "risk_assessment"

class InsightCategory(Enum):
    """Categories of insights"""
    TREND_ANALYSIS = "trend_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_SIGNALS = "risk_signals"
    OPPORTUNITY_IDENTIFICATION = "opportunity_identification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    BEHAVIORAL_INSIGHTS = "behavioral_insights"

@dataclass
class Insight:
    """Container for generated insights"""
    category: InsightCategory
    title: str
    description: str
    confidence_score: float
    priority: str  # "high", "medium", "low"
    
    # Data supporting the insight
    supporting_data: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    
    # Metadata
    generated_at: datetime
    data_period: str
    participants_involved: List[ParticipantType]
    market_segments: List[MarketSegment]
    
    # Risk/opportunity assessment
    potential_impact: str  # "positive", "negative", "neutral"
    time_horizon: str  # "immediate", "short_term", "medium_term", "long_term"

@dataclass
class DashboardData:
    """Container for dashboard visualization data"""
    summary_stats: Dict[str, Any]
    participant_flows: Dict[str, Any]
    behavioral_shifts: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    recent_insights: List[Insight]
    generated_at: datetime

@dataclass
class ReportData:
    """Container for comprehensive report data"""
    report_type: ReportType
    title: str
    executive_summary: str
    key_insights: List[Insight]
    detailed_analysis: Dict[str, Any]
    charts_data: Dict[str, Any]
    recommendations: List[str]
    appendices: Dict[str, Any]
    generated_at: datetime
    period_covered: str

class InsightGenerator:
    """Core insight generation engine"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.metrics_service = ParticipantMetricsService(db_session)
        self.shift_service = BehavioralShiftService(db_session)
    
    async def generate_insights(
        self,
        analysis_date: date = None,
        lookback_days: int = 30,
        participant_types: List[ParticipantType] = None
    ) -> List[Insight]:
        """Generate comprehensive insights from participant flow data"""
        
        insights = []
        analysis_date = analysis_date or date.today()
        
        try:
            if participant_types is None:
                participant_types = list(ParticipantType)
            
            # Generate different types of insights
            insights.extend(await self._generate_trend_insights(analysis_date, lookback_days, participant_types))
            insights.extend(await self._generate_pattern_insights(analysis_date, lookback_days, participant_types))
            insights.extend(await self._generate_risk_insights(analysis_date, lookback_days, participant_types))
            insights.extend(await self._generate_sentiment_insights(analysis_date, lookback_days, participant_types))
            insights.extend(await self._generate_behavioral_insights(analysis_date, lookback_days, participant_types))
            
            # Sort by priority and confidence
            insights.sort(key=lambda x: (x.priority == "high", x.confidence_score), reverse=True)
            
            logger.info(f"Generated {len(insights)} insights for {analysis_date}")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
        
        return insights
    
    async def _generate_trend_insights(
        self, 
        analysis_date: date, 
        lookback_days: int,
        participant_types: List[ParticipantType]
    ) -> List[Insight]:
        """Generate trend analysis insights"""
        
        insights = []
        
        try:
            # Get recent metrics for trend analysis
            recent_metrics = await self._get_recent_metrics(analysis_date, 7, participant_types)
            historical_metrics = await self._get_recent_metrics(
                analysis_date - timedelta(days=7), 14, participant_types
            )
            
            for participant_type in participant_types:
                recent_data = recent_metrics.get(participant_type, {})
                historical_data = historical_metrics.get(participant_type, {})
                
                if recent_data and historical_data:
                    trend_insight = await self._analyze_participant_trend(
                        participant_type, recent_data, historical_data, analysis_date
                    )
                    if trend_insight:
                        insights.append(trend_insight)
            
        except Exception as e:
            logger.error(f"Error generating trend insights: {str(e)}")
        
        return insights
    
    async def _analyze_participant_trend(
        self,
        participant_type: ParticipantType,
        recent_data: Dict[str, Any],
        historical_data: Dict[str, Any],
        analysis_date: date
    ) -> Optional[Insight]:
        """Analyze trend for specific participant type"""
        
        try:
            recent_net_flow = recent_data.get('avg_net_flow', 0)
            historical_net_flow = historical_data.get('avg_net_flow', 0)
            
            if historical_net_flow == 0:
                return None
            
            flow_change = (recent_net_flow - historical_net_flow) / abs(historical_net_flow)
            
            if abs(flow_change) > 0.3:  # Significant change
                trend_direction = "increasing" if flow_change > 0 else "decreasing"
                intensity = "strong" if abs(flow_change) > 0.5 else "moderate"
                
                confidence = min(0.95, abs(flow_change))
                priority = "high" if abs(flow_change) > 0.5 else "medium"
                
                return Insight(
                    category=InsightCategory.TREND_ANALYSIS,
                    title=f"{participant_type.value} Flow Trend Change",
                    description=f"{participant_type.value} participants show {intensity} {trend_direction} "
                               f"flow trend with {flow_change:.1%} change vs historical average",
                    confidence_score=confidence,
                    priority=priority,
                    supporting_data={
                        'recent_net_flow': recent_net_flow,
                        'historical_net_flow': historical_net_flow,
                        'change_percentage': flow_change,
                        'trend_direction': trend_direction,
                        'intensity': intensity
                    },
                    recommendations=[
                        f"Monitor {participant_type.value} activity closely",
                        "Review position sizing in affected segments",
                        "Check correlation with market events"
                    ],
                    generated_at=datetime.now(),
                    data_period=f"7 days ending {analysis_date}",
                    participants_involved=[participant_type],
                    market_segments=list(MarketSegment),
                    potential_impact="positive" if flow_change > 0 else "negative",
                    time_horizon="short_term"
                )
                
        except Exception as e:
            logger.error(f"Error analyzing participant trend: {str(e)}")
        
        return None
    
    async def _generate_behavioral_insights(
        self,
        analysis_date: date,
        lookback_days: int,
        participant_types: List[ParticipantType]
    ) -> List[Insight]:
        """Generate behavioral shift insights"""
        
        insights = []
        
        try:
            # Get recent behavioral shifts
            recent_shifts = await self.shift_service.get_recent_shifts(days_back=7)
            
            for shift_event in recent_shifts:
                insight = await self._convert_shift_to_insight(shift_event, analysis_date)
                if insight:
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating behavioral insights: {str(e)}")
        
        return insights
    
    async def _convert_shift_to_insight(
        self,
        shift_event: ParticipantFlowEvent,
        analysis_date: date
    ) -> Optional[Insight]:
        """Convert behavioral shift event to insight"""
        
        try:
            # Get participant type from profile
            participant = self.db_session.query(ParticipantProfile).filter(
                ParticipantProfile.id == shift_event.participant_id
            ).first()
            
            if not participant:
                return None
            
            priority_map = {
                'critical': 'high',
                'high': 'high', 
                'medium': 'medium',
                'low': 'low'
            }
            
            return Insight(
                category=InsightCategory.BEHAVIORAL_INSIGHTS,
                title=f"Behavioral Shift: {shift_event.event_title}",
                description=shift_event.event_description,
                confidence_score=float(shift_event.confidence_level) / 100.0,
                priority=priority_map.get(shift_event.event_severity, 'medium'),
                supporting_data={
                    'event_type': shift_event.event_type,
                    'magnitude': float(shift_event.magnitude),
                    'significance_score': float(shift_event.significance_score),
                    'trigger_conditions': shift_event.trigger_conditions
                },
                recommendations=[
                    "Investigate underlying causes",
                    "Adjust risk management parameters",
                    "Monitor for continuation of trend"
                ],
                generated_at=datetime.now(),
                data_period=f"Event detected on {shift_event.event_date.date()}",
                participants_involved=[participant.participant_type],
                market_segments=list(MarketSegment),
                potential_impact=shift_event.market_impact or "neutral",
                time_horizon="immediate"
            )
            
        except Exception as e:
            logger.error(f"Error converting shift to insight: {str(e)}")
        
        return None
    
    async def _get_recent_metrics(
        self,
        end_date: date,
        days: int,
        participant_types: List[ParticipantType]
    ) -> Dict[ParticipantType, Dict[str, Any]]:
        """Get aggregated metrics for recent period"""
        
        try:
            start_date = end_date - timedelta(days=days)
            
            metrics = {}
            
            for participant_type in participant_types:
                # Get activities for this participant type
                activities = self.db_session.query(ParticipantActivity).join(
                    ParticipantProfile
                ).filter(
                    and_(
                        ParticipantProfile.participant_type == participant_type,
                        ParticipantActivity.trade_date >= start_date,
                        ParticipantActivity.trade_date <= end_date
                    )
                ).all()
                
                if activities:
                    net_flows = [float(a.net_value) for a in activities]
                    metrics[participant_type] = {
                        'avg_net_flow': np.mean(net_flows),
                        'total_volume': sum(float(a.gross_turnover) for a in activities),
                        'activity_count': len(activities),
                        'volatility': np.std(net_flows) if len(net_flows) > 1 else 0
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting recent metrics: {str(e)}")
            return {}

    # Placeholder methods for other insight types
    async def _generate_pattern_insights(self, analysis_date, lookback_days, participant_types):
        return []
    
    async def _generate_risk_insights(self, analysis_date, lookback_days, participant_types):
        return []
    
    async def _generate_sentiment_insights(self, analysis_date, lookback_days, participant_types):
        return []

class DashboardService:
    """Service for generating dashboard data"""
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.insight_generator = InsightGenerator(db_session)
        self.metrics_service = ParticipantMetricsService(db_session)
    
    async def generate_dashboard_data(self, analysis_date: date = None) -> DashboardData:
        """Generate comprehensive dashboard data"""
        
        analysis_date = analysis_date or date.today()
        
        try:
            # Generate summary statistics
            summary_stats = await self._generate_summary_stats(analysis_date)
            
            # Get participant flows
            participant_flows = await self._get_participant_flows(analysis_date)
            
            # Get behavioral shifts
            behavioral_shifts = await self._get_behavioral_shifts_data(analysis_date)
            
            # Generate trend analysis
            trend_analysis = await self._generate_trend_analysis(analysis_date)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(analysis_date)
            
            # Get recent insights
            recent_insights = await self.insight_generator.generate_insights(
                analysis_date, lookback_days=14
            )
            
            return DashboardData(
                summary_stats=summary_stats,
                participant_flows=participant_flows,
                behavioral_shifts=behavioral_shifts,
                trend_analysis=trend_analysis,
                risk_metrics=risk_metrics,
                recent_insights=recent_insights[:10],  # Top 10 insights
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {str(e)}")
            raise

    async def _generate_summary_stats(self, analysis_date: date) -> Dict[str, Any]:
        """Generate summary statistics for dashboard"""
        
        try:
            # Get last 7 days of data
            start_date = analysis_date - timedelta(days=7)
            
            activities = self.db_session.query(ParticipantActivity).filter(
                and_(
                    ParticipantActivity.trade_date >= start_date,
                    ParticipantActivity.trade_date <= analysis_date
                )
            ).all()
            
            if not activities:
                return {}
            
            total_volume = sum(float(a.gross_turnover) for a in activities)
            net_flows_by_type = {}
            
            # Calculate net flows by participant type
            for participant_type in ParticipantType:
                type_activities = [a for a in activities 
                                if a.participant_profile.participant_type == participant_type]
                if type_activities:
                    net_flows_by_type[participant_type.value] = sum(
                        float(a.net_value) for a in type_activities
                    )
            
            return {
                'total_volume_7d': total_volume,
                'net_flows_by_participant': net_flows_by_type,
                'total_participants_active': len(set(a.participant_id for a in activities)),
                'analysis_date': analysis_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating summary stats: {str(e)}")
            return {}

    # Placeholder methods for other dashboard components
    async def _get_participant_flows(self, analysis_date):
        return {}
    
    async def _get_behavioral_shifts_data(self, analysis_date):
        return []
    
    async def _generate_trend_analysis(self, analysis_date):
        return {}
    
    async def _calculate_risk_metrics(self, analysis_date):
        return {}

class InsightsReportingService:
    """Main service for insights generation and reporting"""
    
    def __init__(self, db_session: Session = None):
        self.db_session = db_session or next(get_db())
        self.insight_generator = InsightGenerator(self.db_session)
        self.dashboard_service = DashboardService(self.db_session)
    
    async def generate_daily_insights(self, analysis_date: date = None) -> List[Insight]:
        """Generate daily insights"""
        
        analysis_date = analysis_date or date.today()
        return await self.insight_generator.generate_insights(
            analysis_date, lookback_days=7
        )
    
    async def get_dashboard_data(self, analysis_date: date = None) -> DashboardData:
        """Get comprehensive dashboard data"""
        
        return await self.dashboard_service.generate_dashboard_data(analysis_date)
    
    async def generate_report(
        self,
        report_type: ReportType,
        analysis_date: date = None,
        **kwargs
    ) -> ReportData:
        """Generate comprehensive report"""
        
        analysis_date = analysis_date or date.today()
        
        try:
            if report_type == ReportType.DAILY_SUMMARY:
                return await self._generate_daily_summary_report(analysis_date)
            elif report_type == ReportType.WEEKLY_OVERVIEW:
                return await self._generate_weekly_overview_report(analysis_date)
            elif report_type == ReportType.BEHAVIORAL_SHIFTS:
                return await self._generate_behavioral_shifts_report(analysis_date)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
                
        except Exception as e:
            logger.error(f"Error generating {report_type.value} report: {str(e)}")
            raise

    async def _generate_daily_summary_report(self, analysis_date: date) -> ReportData:
        """Generate daily summary report"""
        
        try:
            insights = await self.insight_generator.generate_insights(
                analysis_date, lookback_days=7
            )
            
            dashboard_data = await self.dashboard_service.generate_dashboard_data(analysis_date)
            
            # Create executive summary
            high_priority_insights = [i for i in insights if i.priority == "high"]
            exec_summary = f"Daily analysis for {analysis_date} identified {len(insights)} insights, "
            exec_summary += f"including {len(high_priority_insights)} high-priority items requiring attention."
            
            return ReportData(
                report_type=ReportType.DAILY_SUMMARY,
                title=f"Daily Participant Flow Summary - {analysis_date}",
                executive_summary=exec_summary,
                key_insights=insights[:5],  # Top 5 insights
                detailed_analysis=asdict(dashboard_data),
                charts_data=dashboard_data.participant_flows,
                recommendations=[i.recommendations[0] for i in high_priority_insights if i.recommendations],
                appendices={"raw_data": dashboard_data.summary_stats},
                generated_at=datetime.now(),
                period_covered=f"{analysis_date}"
            )
            
        except Exception as e:
            logger.error(f"Error generating daily summary report: {str(e)}")
            raise

    # Placeholder methods for other report types
    async def _generate_weekly_overview_report(self, analysis_date):
        pass
    
    async def _generate_behavioral_shifts_report(self, analysis_date):
        pass

# Convenience functions
async def generate_daily_insights(analysis_date: date = None) -> List[Insight]:
    """Convenience function to generate daily insights"""
    service = InsightsReportingService()
    return await service.generate_daily_insights(analysis_date)

async def get_dashboard_data(analysis_date: date = None) -> DashboardData:
    """Convenience function to get dashboard data"""
    service = InsightsReportingService()
    return await service.get_dashboard_data(analysis_date) 