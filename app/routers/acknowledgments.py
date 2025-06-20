"""
Acknowledgment API Router
REST endpoints for user acknowledgment and response system
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel, Field
from datetime import datetime

from app.services.acknowledgment_service import get_acknowledgment_service, AcknowledgmentService
from app.models.acknowledgment import AcknowledgmentStatus, ResponseType, PreferenceType
from app.auth import get_current_user

router = APIRouter(prefix="/api/v1/acknowledgments", tags=["acknowledgments"])

# Request/Response Models

class AcknowledgmentRequest(BaseModel):
    alert_id: int
    timeout_minutes: int = Field(default=15, ge=1, le=1440)  # 1 minute to 24 hours
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    via_channel: str = Field(default="websocket")

class AcknowledgeRequest(BaseModel):
    response_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    device_id: Optional[str] = None

class DismissRequest(BaseModel):
    reason: Optional[str] = None

class EscalateRequest(BaseModel):
    escalation_reason: Optional[str] = None

class SnoozeRequest(BaseModel):
    snooze_minutes: int = Field(default=30, ge=5, le=1440)  # 5 minutes to 24 hours

class CustomResponseRequest(BaseModel):
    action_taken: str
    action_parameters: Optional[Dict[str, Any]] = None
    action_result: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class PreferenceRequest(BaseModel):
    preference_type: PreferenceType
    preference_key: str
    preference_value: Dict[str, Any]
    asset_symbols: Optional[List[str]] = None
    alert_types: Optional[List[str]] = None
    severity_levels: Optional[List[str]] = None
    active_hours: Optional[Dict[str, Any]] = None
    timezone: str = Field(default="UTC")

class AcknowledgmentResponse(BaseModel):
    id: int
    alert_id: int
    user_id: str
    status: AcknowledgmentStatus
    acknowledged_at: Optional[datetime]
    response_time_ms: Optional[int]
    acknowledged_via: Optional[str]
    device_id: Optional[str]
    timeout_at: Optional[datetime]
    timeout_duration_minutes: int
    response_message: Optional[str]
    response_data: Optional[Dict[str, Any]]
    sync_token: Optional[str]
    is_synced: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class UserResponseResponse(BaseModel):
    id: int
    acknowledgment_id: int
    user_id: str
    response_type: ResponseType
    response_value: Optional[str]
    confidence_score: Optional[float]
    action_taken: Optional[str]
    action_parameters: Optional[Dict[str, Any]]
    action_result: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class PreferenceResponse(BaseModel):
    id: int
    user_id: str
    preference_type: PreferenceType
    preference_key: str
    preference_value: Dict[str, Any]
    asset_symbols: Optional[List[str]]
    alert_types: Optional[List[str]]
    severity_levels: Optional[List[str]]
    active_hours: Optional[Dict[str, Any]]
    timezone: str
    is_active: bool
    priority: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class AnalyticsResponse(BaseModel):
    total_alerts: int
    acknowledged_count: int
    timeout_count: int
    dismissed_count: int
    escalated_count: int
    acknowledgment_rate: float
    timeout_rate: float
    avg_response_time_ms: float
    time_period_hours: int

# Acknowledgment Management Endpoints

@router.post("/create", response_model=AcknowledgmentResponse)
async def create_acknowledgment(
    request: AcknowledgmentRequest,
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Create a new acknowledgment for an alert"""
    try:
        acknowledgment = await service.create_acknowledgment(
            alert_id=request.alert_id,
            user_id=str(current_user.id),
            timeout_minutes=request.timeout_minutes,
            device_id=request.device_id,
            session_id=request.session_id,
            via_channel=request.via_channel
        )
        return AcknowledgmentResponse.from_orm(acknowledgment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create acknowledgment: {str(e)}")

@router.post("/{acknowledgment_id}/acknowledge")
async def acknowledge_alert(
    acknowledgment_id: int = Path(..., description="Acknowledgment ID"),
    request: AcknowledgeRequest = AcknowledgeRequest(),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Acknowledge an alert"""
    success = await service.acknowledge_alert(
        acknowledgment_id=acknowledgment_id,
        user_id=str(current_user.id),
        response_message=request.response_message,
        response_data=request.response_data,
        device_id=request.device_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Acknowledgment not found or already processed")
    
    return {"message": "Alert acknowledged successfully", "acknowledgment_id": acknowledgment_id}

@router.post("/{acknowledgment_id}/dismiss")
async def dismiss_alert(
    acknowledgment_id: int = Path(..., description="Acknowledgment ID"),
    request: DismissRequest = DismissRequest(),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Dismiss an alert"""
    success = await service.dismiss_alert(
        acknowledgment_id=acknowledgment_id,
        user_id=str(current_user.id),
        reason=request.reason
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Acknowledgment not found")
    
    return {"message": "Alert dismissed successfully", "acknowledgment_id": acknowledgment_id}

@router.post("/{acknowledgment_id}/escalate")
async def escalate_alert(
    acknowledgment_id: int = Path(..., description="Acknowledgment ID"),
    request: EscalateRequest = EscalateRequest(),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Escalate an alert to the next level"""
    success = await service.escalate_alert(
        acknowledgment_id=acknowledgment_id,
        user_id=str(current_user.id),
        escalation_reason=request.escalation_reason
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Acknowledgment not found")
    
    return {"message": "Alert escalated successfully", "acknowledgment_id": acknowledgment_id}

@router.post("/{acknowledgment_id}/snooze")
async def snooze_alert(
    acknowledgment_id: int = Path(..., description="Acknowledgment ID"),
    request: SnoozeRequest = SnoozeRequest(),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Snooze an alert for a specified time"""
    success = await service.snooze_alert(
        acknowledgment_id=acknowledgment_id,
        user_id=str(current_user.id),
        snooze_minutes=request.snooze_minutes
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Acknowledgment not found")
    
    return {
        "message": f"Alert snoozed for {request.snooze_minutes} minutes",
        "acknowledgment_id": acknowledgment_id,
        "snooze_minutes": request.snooze_minutes
    }

@router.post("/{acknowledgment_id}/custom-response", response_model=UserResponseResponse)
async def record_custom_response(
    acknowledgment_id: int = Path(..., description="Acknowledgment ID"),
    request: CustomResponseRequest = ...,
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Record a custom user response to an alert"""
    try:
        response = await service.record_custom_response(
            acknowledgment_id=acknowledgment_id,
            user_id=str(current_user.id),
            action_taken=request.action_taken,
            action_parameters=request.action_parameters,
            action_result=request.action_result,
            confidence_score=request.confidence_score
        )
        return UserResponseResponse.from_orm(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record custom response: {str(e)}")

# Acknowledgment Retrieval Endpoints

@router.get("/user/{user_id}", response_model=List[AcknowledgmentResponse])
async def get_user_acknowledgments(
    user_id: str = Path(..., description="User ID"),
    status: Optional[AcknowledgmentStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Get acknowledgments for a specific user"""
    # Check if current user can access the requested user's data
    if str(current_user.id) != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    acknowledgments = await service.get_user_acknowledgments(
        user_id=user_id,
        status=status,
        limit=limit,
        offset=offset
    )
    
    return [AcknowledgmentResponse.from_orm(ack) for ack in acknowledgments]

@router.get("/my-acknowledgments", response_model=List[AcknowledgmentResponse])
async def get_my_acknowledgments(
    status: Optional[AcknowledgmentStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Get acknowledgments for the current user"""
    acknowledgments = await service.get_user_acknowledgments(
        user_id=str(current_user.id),
        status=status,
        limit=limit,
        offset=offset
    )
    
    return [AcknowledgmentResponse.from_orm(ack) for ack in acknowledgments]

@router.get("/pending", response_model=List[AcknowledgmentResponse])
async def get_pending_acknowledgments(
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Get pending acknowledgments for timeout processing (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    acknowledgments = await service.get_pending_acknowledgments(limit=limit)
    return [AcknowledgmentResponse.from_orm(ack) for ack in acknowledgments]

@router.post("/process-timeouts")
async def process_timeouts(
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Process acknowledgment timeouts (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    processed_count = await service.process_timeouts()
    return {
        "message": f"Processed {processed_count} timeouts",
        "processed_count": processed_count
    }

# User Preferences Endpoints

@router.post("/preferences", response_model=PreferenceResponse)
async def set_user_preference(
    request: PreferenceRequest,
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Set or update a user alert preference"""
    try:
        preference = await service.set_user_preference(
            user_id=str(current_user.id),
            preference_type=request.preference_type,
            preference_key=request.preference_key,
            preference_value=request.preference_value,
            asset_symbols=request.asset_symbols,
            alert_types=request.alert_types,
            severity_levels=request.severity_levels,
            active_hours=request.active_hours,
            timezone=request.timezone
        )
        return PreferenceResponse.from_orm(preference)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set preference: {str(e)}")

@router.get("/preferences", response_model=List[PreferenceResponse])
async def get_user_preferences(
    preference_type: Optional[PreferenceType] = Query(None, description="Filter by preference type"),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Get user alert preferences"""
    preferences = await service.get_user_preferences(
        user_id=str(current_user.id),
        preference_type=preference_type
    )
    
    return [PreferenceResponse.from_orm(pref) for pref in preferences]

# Analytics Endpoints

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_acknowledgment_analytics(
    user_id: Optional[str] = Query(None, description="User ID for user-specific analytics"),
    asset_symbol: Optional[str] = Query(None, description="Asset symbol filter"),
    hours: int = Query(24, ge=1, le=8760, description="Time period in hours (max 1 year)"),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Get acknowledgment analytics"""
    # Check permissions for user-specific analytics
    if user_id and str(current_user.id) != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    analytics = await service.get_acknowledgment_analytics(
        user_id=user_id,
        asset_symbol=asset_symbol,
        hours=hours
    )
    
    return AnalyticsResponse(**analytics)

@router.get("/analytics/my-stats", response_model=AnalyticsResponse)
async def get_my_analytics(
    asset_symbol: Optional[str] = Query(None, description="Asset symbol filter"),
    hours: int = Query(24, ge=1, le=8760, description="Time period in hours (max 1 year)"),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Get analytics for the current user"""
    analytics = await service.get_acknowledgment_analytics(
        user_id=str(current_user.id),
        asset_symbol=asset_symbol,
        hours=hours
    )
    
    return AnalyticsResponse(**analytics)

@router.post("/analytics/update")
async def update_analytics(
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Update acknowledgment analytics (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await service.update_acknowledgment_analytics()
    return {"message": "Analytics updated successfully"}

# Device Synchronization Endpoints

@router.post("/sync/{sync_token}")
async def sync_acknowledgment_across_devices(
    sync_token: str = Path(..., description="Sync token"),
    device_id: str = Query(..., description="Device ID"),
    current_user = Depends(get_current_user),
    service: AcknowledgmentService = Depends(get_acknowledgment_service)
):
    """Sync acknowledgment across user's devices"""
    success = await service.sync_acknowledgment_across_devices(
        sync_token=sync_token,
        device_id=device_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Sync token not found")
    
    return {
        "message": "Acknowledgment synced successfully",
        "sync_token": sync_token,
        "device_id": device_id
    } 