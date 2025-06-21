"""
API Endpoints for Insights Generation and Reporting Service

Provides REST API endpoints for accessing participant flow insights, dashboards,
and automated reports with proper authentication and error handling.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, List, Optional, Any
from datetime import date, datetime, timedelta
from pydantic import BaseModel, Field
import logging

# Internal imports
from app.database.connection import get_db
from app.services.insights_generation_service import (
    InsightsReportingService,
    ReportType,
    InsightCategory,
    Insight,
    DashboardData,
    ReportData
)
from app.models.participant_flow import ParticipantType, MarketSegment
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for API responses
class InsightResponse(BaseModel):
    """API response model for insights"""
    category: str
    title: str
    description: str
    confidence_score: float
    priority: str
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime
    data_period: str
    participants_involved: List[str]
    market_segments: List[str]
    potential_impact: str
    time_horizon: str

class DashboardResponse(BaseModel):
    """API response model for dashboard data"""
    summary_stats: Dict[str, Any]
    participant_flows: Dict[str, Any]
    behavioral_shifts: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    recent_insights: List[InsightResponse]
    generated_at: datetime

class ReportResponse(BaseModel):
    """API response model for reports"""
    report_type: str
    title: str
    executive_summary: str
    key_insights: List[InsightResponse]
    detailed_analysis: Dict[str, Any]
    charts_data: Dict[str, Any]
    recommendations: List[str]
    appendices: Dict[str, Any]
    generated_at: datetime
    period_covered: str

# Helper functions
def convert_insight_to_response(insight: Insight) -> InsightResponse:
    """Convert internal Insight object to API response format"""
    return InsightResponse(
        category=insight.category.value,
        title=insight.title,
        description=insight.description,
        confidence_score=insight.confidence_score,
        priority=insight.priority,
        supporting_data=insight.supporting_data,
        recommendations=insight.recommendations,
        generated_at=insight.generated_at,
        data_period=insight.data_period,
        participants_involved=[p.value for p in insight.participants_involved],
        market_segments=[s.value for s in insight.market_segments],
        potential_impact=insight.potential_impact,
        time_horizon=insight.time_horizon
    )

def convert_dashboard_to_response(dashboard: DashboardData) -> DashboardResponse:
    """Convert internal DashboardData to API response format"""
    return DashboardResponse(
        summary_stats=dashboard.summary_stats,
        participant_flows=dashboard.participant_flows,
        behavioral_shifts=dashboard.behavioral_shifts,
        trend_analysis=dashboard.trend_analysis,
        risk_metrics=dashboard.risk_metrics,
        recent_insights=[convert_insight_to_response(i) for i in dashboard.recent_insights],
        generated_at=dashboard.generated_at
    )

def convert_report_to_response(report: ReportData) -> ReportResponse:
    """Convert internal ReportData to API response format"""
    return ReportResponse(
        report_type=report.report_type.value,
        title=report.title,
        executive_summary=report.executive_summary,
        key_insights=[convert_insight_to_response(i) for i in report.key_insights],
        detailed_analysis=report.detailed_analysis,
        charts_data=report.charts_data,
        recommendations=report.recommendations,
        appendices=report.appendices,
        generated_at=report.generated_at,
        period_covered=report.period_covered
    )

# API Endpoints

@router.get("/insights/daily", response_model=List[InsightResponse])
async def get_daily_insights(
    analysis_date: Optional[date] = Query(None, description="Analysis date (defaults to today)"),
    category: Optional[str] = Query(None, description="Filter by insight category"),
    priority: Optional[str] = Query(None, description="Filter by priority (high/medium/low)"),
    participant_type: Optional[str] = Query(None, description="Filter by participant type"),
    limit: int = Query(20, description="Maximum number of insights to return"),
    db: Session = Depends(get_db)
):
    """
    Get daily insights for participant flow analysis.
    
    Returns actionable insights generated from recent participant activity,
    behavioral shifts, and market trends.
    """
    try:
        service = InsightsReportingService(db)
        insights = await service.generate_daily_insights(analysis_date)
        
        # Apply filters
        if category:
            insights = [i for i in insights if i.category.value == category]
        
        if priority:
            insights = [i for i in insights if i.priority == priority]
        
        if participant_type:
            insights = [i for i in insights 
                       if any(p.value == participant_type for p in i.participants_involved)]
        
        # Apply limit
        insights = insights[:limit]
        
        return [convert_insight_to_response(insight) for insight in insights]
        
    except Exception as e:
        logger.error(f"Error getting daily insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_data(
    analysis_date: Optional[date] = Query(None, description="Analysis date (defaults to today)"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive dashboard data for participant flow monitoring.
    
    Returns summary statistics, participant flows, behavioral shifts,
    trend analysis, risk metrics, and recent insights.
    """
    try:
        service = InsightsReportingService(db)
        dashboard_data = await service.get_dashboard_data(analysis_date)
        
        return convert_dashboard_to_response(dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")

@router.get("/reports/{report_type}", response_model=ReportResponse)
async def generate_report(
    report_type: str,
    analysis_date: Optional[date] = Query(None, description="Analysis date (defaults to today)"),
    format: str = Query("json", description="Report format (json/pdf/excel)"),
    db: Session = Depends(get_db)
):
    """
    Generate comprehensive reports for participant flow analysis.
    
    Available report types:
    - daily_summary: Daily participant flow summary
    - weekly_overview: Weekly trend analysis
    - monthly_analysis: Monthly performance review
    - behavioral_shifts: Behavioral shift analysis
    - market_sentiment: Market sentiment analysis
    """
    try:
        # Validate report type
        try:
            report_type_enum = ReportType(report_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid report type. Available types: {[rt.value for rt in ReportType]}"
            )
        
        service = InsightsReportingService(db)
        report_data = await service.generate_report(report_type_enum, analysis_date)
        
        if format == "json":
            return convert_report_to_response(report_data)
        else:
            # For PDF/Excel, we would generate file and return FileResponse
            # This is a placeholder for file generation
            raise HTTPException(status_code=501, detail=f"Format '{format}' not yet implemented")
        
    except Exception as e:
        logger.error(f"Error generating {report_type} report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@router.get("/insights/categories")
async def get_insight_categories():
    """Get available insight categories"""
    return {
        "categories": [
            {
                "value": category.value,
                "name": category.value.replace("_", " ").title(),
                "description": f"Insights related to {category.value.replace('_', ' ')}"
            }
            for category in InsightCategory
        ]
    }

@router.get("/reports/types")
async def get_report_types():
    """Get available report types"""
    return {
        "report_types": [
            {
                "value": report_type.value,
                "name": report_type.value.replace("_", " ").title(),
                "description": f"Report focusing on {report_type.value.replace('_', ' ')}"
            }
            for report_type in ReportType
        ]
    }

@router.get("/participants/types")
async def get_participant_types():
    """Get available participant types for filtering"""
    return {
        "participant_types": [
            {
                "value": pt.value,
                "name": pt.value,
                "description": f"{pt.value} participant category"
            }
            for pt in ParticipantType
        ]
    }

@router.get("/market-segments")
async def get_market_segments():
    """Get available market segments"""
    return {
        "market_segments": [
            {
                "value": ms.value,
                "name": ms.value.title(),
                "description": f"{ms.value} market segment"
            }
            for ms in MarketSegment
        ]
    }

@router.get("/insights/summary")
async def get_insights_summary(
    days_back: int = Query(7, description="Number of days to look back"),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics about generated insights.
    
    Returns counts by category, priority, participant type, etc.
    """
    try:
        service = InsightsReportingService(db)
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        # Get insights for the period
        insights = await service.generate_daily_insights(end_date)
        
        # Generate summary statistics
        summary = {
            "period": f"{start_date} to {end_date}",
            "total_insights": len(insights),
            "by_category": {},
            "by_priority": {},
            "by_participant": {},
            "high_confidence_count": len([i for i in insights if i.confidence_score > 0.8]),
            "actionable_count": len([i for i in insights if i.recommendations])
        }
        
        # Count by category
        for insight in insights:
            category = insight.category.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
        
        # Count by priority
        for insight in insights:
            priority = insight.priority
            summary["by_priority"][priority] = summary["by_priority"].get(priority, 0) + 1
        
        # Count by participant type
        for insight in insights:
            for participant in insight.participants_involved:
                pt = participant.value
                summary["by_participant"][pt] = summary["by_participant"].get(pt, 0) + 1
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting insights summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

# Background task endpoints
@router.post("/insights/generate-daily")
async def trigger_daily_insights_generation(
    background_tasks: BackgroundTasks,
    analysis_date: Optional[date] = Query(None, description="Analysis date (defaults to today)"),
    db: Session = Depends(get_db)
):
    """
    Trigger background generation of daily insights.
    
    This endpoint starts the insight generation process in the background
    and returns immediately with a task ID.
    """
    try:
        analysis_date = analysis_date or date.today()
        
        # Add background task
        background_tasks.add_task(
            generate_insights_background_task,
            db,
            analysis_date
        )
        
        return {
            "message": "Daily insights generation started",
            "analysis_date": analysis_date,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error triggering daily insights generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")

async def generate_insights_background_task(db: Session, analysis_date: date):
    """Background task for generating insights"""
    try:
        service = InsightsReportingService(db)
        insights = await service.generate_daily_insights(analysis_date)
        logger.info(f"Generated {len(insights)} insights for {analysis_date}")
    except Exception as e:
        logger.error(f"Background insight generation failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for insights service"""
    return {
        "status": "healthy",
        "service": "insights_generation",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    } 