"""
Acknowledgment Service
Handles user acknowledgments, responses, preferences, and analytics for alerts
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.database.connection import get_db
from app.models.acknowledgment import (
    AlertAcknowledgment, UserResponse, AlertPreference, AcknowledgmentAnalytics,
    AcknowledgmentTimeout, AcknowledgmentStatus, ResponseType, PreferenceType
)
from app.models.alert import Alert
from app.services.logging_service import get_logger

logger = get_logger(__name__)

class AcknowledgmentService:
    """Service for managing alert acknowledgments and user responses"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.sync_tokens: Dict[str, str] = {}  # In-memory sync token storage
        
    async def create_acknowledgment(
        self,
        alert_id: int,
        user_id: str,
        timeout_minutes: int = 15,
        device_id: Optional[str] = None,
        session_id: Optional[str] = None,
        via_channel: str = "websocket"
    ) -> AlertAcknowledgment:
        """Create a new acknowledgment record for an alert"""
        
        # Generate sync token for cross-device synchronization
        sync_token = str(uuid.uuid4())
        
        acknowledgment = AlertAcknowledgment(
            alert_id=alert_id,
            user_id=user_id,
            timeout_duration_minutes=timeout_minutes,
            timeout_at=datetime.utcnow() + timedelta(minutes=timeout_minutes),
            device_id=device_id,
            session_id=session_id,
            acknowledged_via=via_channel,
            sync_token=sync_token
        )
        
        self.db.add(acknowledgment)
        self.db.commit()
        self.db.refresh(acknowledgment)
        
        # Schedule timeout processing
        await self._schedule_timeout(acknowledgment)
        
        logger.info(f"Created acknowledgment {acknowledgment.id} for alert {alert_id} by user {user_id}")
        return acknowledgment
    
    async def acknowledge_alert(
        self,
        acknowledgment_id: int,
        user_id: str,
        response_message: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        device_id: Optional[str] = None
    ) -> bool:
        """Mark an alert as acknowledged by a user"""
        
        acknowledgment = self.db.query(AlertAcknowledgment).filter(
            AlertAcknowledgment.id == acknowledgment_id,
            AlertAcknowledgment.user_id == user_id
        ).first()
        
        if not acknowledgment:
            logger.warning(f"Acknowledgment {acknowledgment_id} not found for user {user_id}")
            return False
        
        if acknowledgment.status != AcknowledgmentStatus.PENDING:
            logger.warning(f"Acknowledgment {acknowledgment_id} already processed: {acknowledgment.status}")
            return False
        
        # Calculate response time
        response_time_ms = int((datetime.utcnow() - acknowledgment.created_at).total_seconds() * 1000)
        
        # Update acknowledgment
        acknowledgment.status = AcknowledgmentStatus.ACKNOWLEDGED
        acknowledgment.acknowledged_at = datetime.utcnow()
        acknowledgment.response_time_ms = response_time_ms
        acknowledgment.response_message = response_message
        acknowledgment.response_data = response_data
        acknowledgment.device_id = device_id or acknowledgment.device_id
        
        # Sync across devices
        await self._sync_acknowledgment(acknowledgment)
        
        self.db.commit()
        
        logger.info(f"Alert {acknowledgment.alert_id} acknowledged by user {user_id} in {response_time_ms}ms")
        return True
    
    async def dismiss_alert(
        self,
        acknowledgment_id: int,
        user_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """Dismiss an alert without acknowledgment"""
        
        acknowledgment = self.db.query(AlertAcknowledgment).filter(
            AlertAcknowledgment.id == acknowledgment_id,
            AlertAcknowledgment.user_id == user_id
        ).first()
        
        if not acknowledgment:
            return False
        
        acknowledgment.status = AcknowledgmentStatus.DISMISSED
        acknowledgment.response_message = reason
        acknowledgment.acknowledged_at = datetime.utcnow()
        
        await self._sync_acknowledgment(acknowledgment)
        self.db.commit()
        
        logger.info(f"Alert {acknowledgment.alert_id} dismissed by user {user_id}")
        return True
    
    async def escalate_alert(
        self,
        acknowledgment_id: int,
        user_id: str,
        escalation_reason: Optional[str] = None
    ) -> bool:
        """Escalate an alert to the next level"""
        
        acknowledgment = self.db.query(AlertAcknowledgment).filter(
            AlertAcknowledgment.id == acknowledgment_id,
            AlertAcknowledgment.user_id == user_id
        ).first()
        
        if not acknowledgment:
            return False
        
        acknowledgment.status = AcknowledgmentStatus.ESCALATED
        acknowledgment.response_message = escalation_reason
        acknowledgment.acknowledged_at = datetime.utcnow()
        
        # Record escalation response
        await self._record_user_response(
            acknowledgment.id,
            user_id,
            ResponseType.ESCALATE,
            escalation_reason
        )
        
        await self._sync_acknowledgment(acknowledgment)
        self.db.commit()
        
        logger.info(f"Alert {acknowledgment.alert_id} escalated by user {user_id}")
        return True
    
    async def snooze_alert(
        self,
        acknowledgment_id: int,
        user_id: str,
        snooze_minutes: int = 30
    ) -> bool:
        """Snooze an alert for a specified time"""
        
        acknowledgment = self.db.query(AlertAcknowledgment).filter(
            AlertAcknowledgment.id == acknowledgment_id,
            AlertAcknowledgment.user_id == user_id
        ).first()
        
        if not acknowledgment:
            return False
        
        # Extend timeout
        new_timeout = datetime.utcnow() + timedelta(minutes=snooze_minutes)
        acknowledgment.timeout_at = new_timeout
        acknowledgment.response_message = f"Snoozed for {snooze_minutes} minutes"
        
        # Update timeout record
        timeout_record = self.db.query(AcknowledgmentTimeout).filter(
            AcknowledgmentTimeout.acknowledgment_id == acknowledgment_id
        ).first()
        
        if timeout_record:
            timeout_record.timeout_trigger_at = new_timeout
            timeout_record.is_processed = False
        
        await self._sync_acknowledgment(acknowledgment)
        self.db.commit()
        
        logger.info(f"Alert {acknowledgment.alert_id} snoozed by user {user_id} for {snooze_minutes} minutes")
        return True
    
    async def record_custom_response(
        self,
        acknowledgment_id: int,
        user_id: str,
        action_taken: str,
        action_parameters: Optional[Dict[str, Any]] = None,
        action_result: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> UserResponse:
        """Record a custom user response to an alert"""
        
        response = await self._record_user_response(
            acknowledgment_id,
            user_id,
            ResponseType.CUSTOM_ACTION,
            action_taken,
            action_parameters=action_parameters,
            action_result=action_result,
            confidence_score=confidence_score
        )
        
        logger.info(f"Custom response recorded for acknowledgment {acknowledgment_id}: {action_taken}")
        return response
    
    async def get_user_acknowledgments(
        self,
        user_id: str,
        status: Optional[AcknowledgmentStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[AlertAcknowledgment]:
        """Get acknowledgments for a user"""
        
        query = self.db.query(AlertAcknowledgment).filter(
            AlertAcknowledgment.user_id == user_id
        )
        
        if status:
            query = query.filter(AlertAcknowledgment.status == status)
        
        return query.order_by(
            AlertAcknowledgment.created_at.desc()
        ).offset(offset).limit(limit).all()
    
    async def get_pending_acknowledgments(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AlertAcknowledgment]:
        """Get pending acknowledgments for timeout processing"""
        
        query = self.db.query(AlertAcknowledgment).filter(
            AlertAcknowledgment.status == AcknowledgmentStatus.PENDING,
            AlertAcknowledgment.timeout_at <= datetime.utcnow()
        )
        
        if user_id:
            query = query.filter(AlertAcknowledgment.user_id == user_id)
        
        return query.order_by(AlertAcknowledgment.timeout_at.asc()).limit(limit).all()
    
    async def process_timeouts(self) -> int:
        """Process acknowledgment timeouts and mark them as timed out"""
        
        pending_timeouts = await self.get_pending_acknowledgments()
        processed_count = 0
        
        for acknowledgment in pending_timeouts:
            try:
                acknowledgment.status = AcknowledgmentStatus.TIMEOUT
                acknowledgment.acknowledged_at = datetime.utcnow()
                
                # Mark timeout record as processed
                timeout_record = self.db.query(AcknowledgmentTimeout).filter(
                    AcknowledgmentTimeout.acknowledgment_id == acknowledgment.id
                ).first()
                
                if timeout_record:
                    timeout_record.is_processed = True
                    timeout_record.processed_at = datetime.utcnow()
                    timeout_record.timeout_action = "timeout"
                    timeout_record.timeout_result = "Alert acknowledgment timed out"
                
                await self._sync_acknowledgment(acknowledgment)
                processed_count += 1
                
                logger.info(f"Processed timeout for acknowledgment {acknowledgment.id}")
                
            except Exception as e:
                logger.error(f"Failed to process timeout for acknowledgment {acknowledgment.id}: {e}")
        
        if processed_count > 0:
            self.db.commit()
            logger.info(f"Processed {processed_count} acknowledgment timeouts")
        
        return processed_count
    
    async def set_user_preference(
        self,
        user_id: str,
        preference_type: PreferenceType,
        preference_key: str,
        preference_value: Dict[str, Any],
        asset_symbols: Optional[List[str]] = None,
        alert_types: Optional[List[str]] = None,
        severity_levels: Optional[List[str]] = None,
        active_hours: Optional[Dict[str, Any]] = None,
        timezone: str = "UTC"
    ) -> AlertPreference:
        """Set or update a user alert preference"""
        
        # Check for existing preference
        existing = self.db.query(AlertPreference).filter(
            AlertPreference.user_id == user_id,
            AlertPreference.preference_type == preference_type,
            AlertPreference.preference_key == preference_key
        ).first()
        
        if existing:
            # Update existing preference
            existing.preference_value = preference_value
            existing.asset_symbols = asset_symbols
            existing.alert_types = alert_types
            existing.severity_levels = severity_levels
            existing.active_hours = active_hours
            existing.timezone = timezone
            existing.updated_at = datetime.utcnow()
            preference = existing
        else:
            # Create new preference
            preference = AlertPreference(
                user_id=user_id,
                preference_type=preference_type,
                preference_key=preference_key,
                preference_value=preference_value,
                asset_symbols=asset_symbols,
                alert_types=alert_types,
                severity_levels=severity_levels,
                active_hours=active_hours,
                timezone=timezone
            )
            self.db.add(preference)
        
        self.db.commit()
        self.db.refresh(preference)
        
        logger.info(f"Set preference {preference_key} for user {user_id}")
        return preference
    
    async def get_user_preferences(
        self,
        user_id: str,
        preference_type: Optional[PreferenceType] = None
    ) -> List[AlertPreference]:
        """Get user alert preferences"""
        
        query = self.db.query(AlertPreference).filter(
            AlertPreference.user_id == user_id,
            AlertPreference.is_active
        )
        
        if preference_type:
            query = query.filter(AlertPreference.preference_type == preference_type)
        
        return query.order_by(AlertPreference.priority.desc()).all()
    
    async def get_acknowledgment_analytics(
        self,
        user_id: Optional[str] = None,
        asset_symbol: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get acknowledgment analytics for a user or globally"""
        
        since = datetime.utcnow() - timedelta(hours=hours)
        
        query = self.db.query(AlertAcknowledgment).filter(
            AlertAcknowledgment.created_at >= since
        )
        
        if user_id:
            query = query.filter(AlertAcknowledgment.user_id == user_id)
        
        if asset_symbol:
            query = query.join(Alert).filter(Alert.symbol == asset_symbol)
        
        acknowledgments = query.all()
        
        if not acknowledgments:
            return {
                'total_alerts': 0,
                'acknowledged_count': 0,
                'timeout_count': 0,
                'dismissed_count': 0,
                'escalated_count': 0,
                'acknowledgment_rate': 0.0,
                'avg_response_time_ms': 0.0
            }
        
        total = len(acknowledgments)
        acknowledged = len([a for a in acknowledgments if a.status == AcknowledgmentStatus.ACKNOWLEDGED])
        timeout = len([a for a in acknowledgments if a.status == AcknowledgmentStatus.TIMEOUT])
        dismissed = len([a for a in acknowledgments if a.status == AcknowledgmentStatus.DISMISSED])
        escalated = len([a for a in acknowledgments if a.status == AcknowledgmentStatus.ESCALATED])
        
        # Calculate average response time for acknowledged alerts
        response_times = [a.response_time_ms for a in acknowledgments 
                         if a.status == AcknowledgmentStatus.ACKNOWLEDGED and a.response_time_ms]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        return {
            'total_alerts': total,
            'acknowledged_count': acknowledged,
            'timeout_count': timeout,
            'dismissed_count': dismissed,
            'escalated_count': escalated,
            'acknowledgment_rate': (acknowledged / total * 100) if total > 0 else 0.0,
            'timeout_rate': (timeout / total * 100) if total > 0 else 0.0,
            'avg_response_time_ms': avg_response_time,
            'time_period_hours': hours
        }
    
    async def update_acknowledgment_analytics(self) -> None:
        """Update aggregate acknowledgment analytics"""
        
        now = datetime.utcnow()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        current_date = now.date()
        
        # Update hourly stats
        await self._update_analytics_for_period(current_date, current_hour.hour)
        
        # Update daily stats
        await self._update_analytics_for_period(current_date, None)
    
    async def sync_acknowledgment_across_devices(
        self,
        sync_token: str,
        device_id: str
    ) -> bool:
        """Sync acknowledgment across user's devices"""
        
        acknowledgment = self.db.query(AlertAcknowledgment).filter(
            AlertAcknowledgment.sync_token == sync_token
        ).first()
        
        if not acknowledgment:
            return False
        
        acknowledgment.is_synced = True
        acknowledgment.last_sync_at = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"Synced acknowledgment {acknowledgment.id} to device {device_id}")
        return True
    
    # Private helper methods
    
    async def _schedule_timeout(self, acknowledgment: AlertAcknowledgment) -> None:
        """Schedule timeout processing for an acknowledgment"""
        
        timeout = AcknowledgmentTimeout(
            acknowledgment_id=acknowledgment.id,
            timeout_trigger_at=acknowledgment.timeout_at,
            timeout_duration_minutes=acknowledgment.timeout_duration_minutes
        )
        
        self.db.add(timeout)
        self.db.commit()
    
    async def _sync_acknowledgment(self, acknowledgment: AlertAcknowledgment) -> None:
        """Sync acknowledgment status across devices"""
        
        # Store sync token for real-time sync
        if acknowledgment.sync_token:
            self.sync_tokens[acknowledgment.sync_token] = acknowledgment.user_id
    
    async def _record_user_response(
        self,
        acknowledgment_id: int,
        user_id: str,
        response_type: ResponseType,
        response_value: Optional[str] = None,
        action_parameters: Optional[Dict[str, Any]] = None,
        action_result: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> UserResponse:
        """Record a detailed user response"""
        
        response = UserResponse(
            acknowledgment_id=acknowledgment_id,
            user_id=user_id,
            response_type=response_type,
            response_value=response_value,
            action_taken=response_value,
            action_parameters=action_parameters,
            action_result=action_result,
            confidence_score=confidence_score
        )
        
        self.db.add(response)
        self.db.commit()
        self.db.refresh(response)
        
        return response
    
    async def _update_analytics_for_period(
        self,
        date: datetime.date,
        hour: Optional[int]
    ) -> None:
        """Update analytics for a specific time period"""
        
        # Define time range
        if hour is not None:
            start_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
            end_time = start_time + timedelta(hours=1)
        else:
            start_time = datetime.combine(date, datetime.min.time())
            end_time = start_time + timedelta(days=1)
        
        # Query acknowledgments in this period
        acknowledgments = self.db.query(AlertAcknowledgment).filter(
            and_(
                AlertAcknowledgment.created_at >= start_time,
                AlertAcknowledgment.created_at < end_time
            )
        ).all()
        
        if not acknowledgments:
            return
        
        # Calculate metrics
        total = len(acknowledgments)
        acknowledged = len([a for a in acknowledgments if a.status == AcknowledgmentStatus.ACKNOWLEDGED])
        timeout = len([a for a in acknowledgments if a.status == AcknowledgmentStatus.TIMEOUT])
        dismissed = len([a for a in acknowledgments if a.status == AcknowledgmentStatus.DISMISSED])
        escalated = len([a for a in acknowledgments if a.status == AcknowledgmentStatus.ESCALATED])
        
        # Response time metrics
        response_times = [a.response_time_ms for a in acknowledgments 
                         if a.status == AcknowledgmentStatus.ACKNOWLEDGED and a.response_time_ms]
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        median_response_time = sorted(response_times)[len(response_times) // 2] if response_times else None
        p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)] if response_times else None
        min_response_time = min(response_times) if response_times else None
        max_response_time = max(response_times) if response_times else None
        
        # Performance metrics
        acknowledgment_rate = (acknowledged / total * 100) if total > 0 else None
        timeout_rate = (timeout / total * 100) if total > 0 else None
        escalation_rate = (escalated / total * 100) if total > 0 else None
        
        # Check if analytics record exists
        existing_analytics = self.db.query(AcknowledgmentAnalytics).filter(
            and_(
                AcknowledgmentAnalytics.date == start_time,
                AcknowledgmentAnalytics.hour == hour,
                AcknowledgmentAnalytics.user_id.is_(None),
                AcknowledgmentAnalytics.asset_symbol.is_(None)
            )
        ).first()
        
        if existing_analytics:
            # Update existing record
            existing_analytics.total_alerts = total
            existing_analytics.acknowledged_alerts = acknowledged
            existing_analytics.timeout_alerts = timeout
            existing_analytics.dismissed_alerts = dismissed
            existing_analytics.escalated_alerts = escalated
            existing_analytics.avg_response_time_ms = avg_response_time
            existing_analytics.median_response_time_ms = median_response_time
            existing_analytics.p95_response_time_ms = p95_response_time
            existing_analytics.min_response_time_ms = min_response_time
            existing_analytics.max_response_time_ms = max_response_time
            existing_analytics.acknowledgment_rate = acknowledgment_rate
            existing_analytics.timeout_rate = timeout_rate
            existing_analytics.escalation_rate = escalation_rate
            existing_analytics.updated_at = datetime.utcnow()
        else:
            # Create new record
            analytics = AcknowledgmentAnalytics(
                date=start_time,
                hour=hour,
                total_alerts=total,
                acknowledged_alerts=acknowledged,
                timeout_alerts=timeout,
                dismissed_alerts=dismissed,
                escalated_alerts=escalated,
                avg_response_time_ms=avg_response_time,
                median_response_time_ms=median_response_time,
                p95_response_time_ms=p95_response_time,
                min_response_time_ms=min_response_time,
                max_response_time_ms=max_response_time,
                acknowledgment_rate=acknowledgment_rate,
                timeout_rate=timeout_rate,
                escalation_rate=escalation_rate
            )
            self.db.add(analytics)
        
        self.db.commit()

# Singleton service instance
_acknowledgment_service: Optional[AcknowledgmentService] = None

def get_acknowledgment_service() -> AcknowledgmentService:
    """Get or create the acknowledgment service instance"""
    global _acknowledgment_service
    
    if _acknowledgment_service is None:
        db_session = next(get_db())
        _acknowledgment_service = AcknowledgmentService(db_session)
    
    return _acknowledgment_service 