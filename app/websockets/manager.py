"""
WebSocket Connection Manager

This module handles WebSocket connections for real-time alert delivery,
providing connection management, message broadcasting, and user-specific
alert delivery capabilities.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection"""
    websocket: WebSocket
    user_id: str
    connected_at: datetime
    last_activity: datetime
    connection_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertMessage:
    """Alert message structure for WebSocket delivery"""
    alert_id: str
    user_id: str
    alert_type: str
    title: str
    message: str
    severity: str
    data: Dict[str, Any]
    timestamp: str
    acknowledgment_id: Optional[str] = None
    requires_acknowledgment: bool = True
    timeout_minutes: int = 15


class ConnectionManager:
    """
    Manages WebSocket connections for real-time alert delivery
    """
    
    def __init__(self):
        # Active connections: user_id -> list of ConnectionInfo
        self.active_connections: Dict[str, List[ConnectionInfo]] = {}
        
        # Connection tracking
        self.connection_count = 0
        self.total_connections = 0
        
        # Message queues for offline users
        self.pending_messages: Dict[str, List[AlertMessage]] = {}
        
        # Connection statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'failed_sends': 0,
            'disconnections': 0
        }
    
    async def connect(self, websocket: WebSocket, user_id: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Accept a new WebSocket connection and register it
        
        Args:
            websocket: WebSocket instance
            user_id: User identifier
            metadata: Optional connection metadata
            
        Returns:
            Connection ID for tracking
        """
        try:
            await websocket.accept()
            
            # Generate unique connection ID
            self.connection_count += 1
            connection_id = f"conn_{user_id}_{self.connection_count}_{int(datetime.now().timestamp())}"
            
            # Create connection info
            connection_info = ConnectionInfo(
                websocket=websocket,
                user_id=user_id,
                connected_at=datetime.now(),
                last_activity=datetime.now(),
                connection_id=connection_id,
                metadata=metadata or {}
            )
            
            # Store connection
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
            self.active_connections[user_id].append(connection_info)
            
            # Update statistics
            self.stats['total_connections'] += 1
            self.stats['active_connections'] = self._count_active_connections()
            
            logger.info(f"WebSocket connected: user={user_id}, connection_id={connection_id}")
            
            # Send any pending messages
            await self._send_pending_messages(user_id)
            
            # Send connection confirmation
            await self._send_system_message(websocket, {
                'type': 'connection_established',
                'connection_id': connection_id,
                'timestamp': datetime.now().isoformat(),
                'message': 'WebSocket connection established successfully'
            })
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket for user {user_id}: {str(e)}")
            raise
    
    def disconnect(self, websocket: WebSocket, user_id: str) -> bool:
        """
        Disconnect and remove a WebSocket connection
        
        Args:
            websocket: WebSocket instance to disconnect
            user_id: User identifier
            
        Returns:
            True if connection was found and removed
        """
        try:
            if user_id not in self.active_connections:
                return False
            
            # Find and remove the connection
            connections = self.active_connections[user_id]
            for i, conn_info in enumerate(connections):
                if conn_info.websocket == websocket:
                    removed_conn = connections.pop(i)
                    logger.info(f"WebSocket disconnected: user={user_id}, connection_id={removed_conn.connection_id}")
                    
                    # Remove user if no more connections
                    if not connections:
                        del self.active_connections[user_id]
                    
                    # Update statistics
                    self.stats['disconnections'] += 1
                    self.stats['active_connections'] = self._count_active_connections()
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket for user {user_id}: {str(e)}")
            return False
    
    async def send_alert(self, user_id: str, alert: AlertMessage) -> bool:
        """
        Send an alert to a specific user
        
        Args:
            user_id: Target user ID
            alert: Alert message to send
            
        Returns:
            True if sent successfully to at least one connection
        """
        try:
            if user_id not in self.active_connections:
                # User not connected, queue the message
                await self._queue_message(user_id, alert)
                logger.info(f"Alert queued for offline user {user_id}: {alert.alert_id}")
                return False
            
            connections = self.active_connections[user_id]
            sent_count = 0
            failed_count = 0
            
            # Send to all user connections
            for conn_info in connections[:]:  # Copy list to avoid modification during iteration
                try:
                    message_data = {
                        'type': 'alert',
                        'alert_id': alert.alert_id,
                        'alert_type': alert.alert_type,
                        'title': alert.title,
                        'message': alert.message,
                        'severity': alert.severity,
                        'data': alert.data,
                        'timestamp': alert.timestamp,
                        'connection_id': conn_info.connection_id,
                        'acknowledgment_id': alert.acknowledgment_id,
                        'requires_acknowledgment': alert.requires_acknowledgment,
                        'timeout_minutes': alert.timeout_minutes
                    }
                    
                    await conn_info.websocket.send_text(json.dumps(message_data))
                    conn_info.last_activity = datetime.now()
                    sent_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to send alert to connection {conn_info.connection_id}: {str(e)}")
                    failed_count += 1
                    # Remove broken connection
                    connections.remove(conn_info)
            
            # Update statistics
            self.stats['messages_sent'] += sent_count
            self.stats['failed_sends'] += failed_count
            
            if sent_count > 0:
                logger.info(f"Alert sent to user {user_id}: {alert.alert_id} (sent to {sent_count} connections)")
                return True
            else:
                # All connections failed, queue the message
                await self._queue_message(user_id, alert)
                return False
                
        except Exception as e:
            logger.error(f"Error sending alert to user {user_id}: {str(e)}")
            return False
    
    async def broadcast_system_message(self, message: Dict[str, Any]) -> int:
        """
        Broadcast a system message to all connected users
        
        Args:
            message: System message to broadcast
            
        Returns:
            Number of successful sends
        """
        try:
            sent_count = 0
            message_data = {
                'type': 'system_message',
                'timestamp': datetime.now().isoformat(),
                **message
            }
            
            for user_id, connections in self.active_connections.items():
                for conn_info in connections[:]:
                    try:
                        await conn_info.websocket.send_text(json.dumps(message_data))
                        conn_info.last_activity = datetime.now()
                        sent_count += 1
                    except Exception as e:
                        logger.error(f"Failed to broadcast to connection {conn_info.connection_id}: {str(e)}")
                        connections.remove(conn_info)
            
            logger.info(f"System message broadcasted to {sent_count} connections")
            return sent_count
            
        except Exception as e:
            logger.error(f"Error broadcasting system message: {str(e)}")
            return 0
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        return {
            **self.stats,
            'active_connections': self._count_active_connections(),
            'active_users': len(self.active_connections),
            'pending_message_queues': len(self.pending_messages),
            'total_pending_messages': sum(len(msgs) for msgs in self.pending_messages.values()),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_user_connections(self, user_id: str) -> List[Dict[str, Any]]:
        """Get information about a user's active connections"""
        if user_id not in self.active_connections:
            return []
        
        return [
            {
                'connection_id': conn.connection_id,
                'connected_at': conn.connected_at.isoformat(),
                'last_activity': conn.last_activity.isoformat(),
                'metadata': conn.metadata
            }
            for conn in self.active_connections[user_id]
        ]
    
    async def send_acknowledgment_response(self, user_id: str, acknowledgment_id: str, 
                                         status: str, message: Optional[str] = None) -> bool:
        """
        Send acknowledgment response to user's devices
        
        Args:
            user_id: Target user ID
            acknowledgment_id: Acknowledgment ID
            status: Acknowledgment status (acknowledged, dismissed, escalated, timeout)
            message: Optional response message
            
        Returns:
            True if sent successfully to at least one connection
        """
        try:
            if user_id not in self.active_connections:
                return False
            
            connections = self.active_connections[user_id]
            sent_count = 0
            
            response_data = {
                'type': 'acknowledgment_response',
                'acknowledgment_id': acknowledgment_id,
                'status': status,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            
            for conn_info in connections[:]:
                try:
                    await conn_info.websocket.send_text(json.dumps(response_data))
                    conn_info.last_activity = datetime.now()
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to send acknowledgment response to connection {conn_info.connection_id}: {str(e)}")
                    connections.remove(conn_info)
            
            if sent_count > 0:
                logger.info(f"Acknowledgment response sent to user {user_id}: {acknowledgment_id} ({status})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending acknowledgment response to user {user_id}: {str(e)}")
            return False
    
    async def send_sync_notification(self, user_id: str, sync_token: str, 
                                   acknowledgment_data: Dict[str, Any]) -> bool:
        """
        Send synchronization notification to user's devices
        
        Args:
            user_id: Target user ID
            sync_token: Synchronization token
            acknowledgment_data: Acknowledgment data to sync
            
        Returns:
            True if sent successfully to at least one connection
        """
        try:
            if user_id not in self.active_connections:
                return False
            
            connections = self.active_connections[user_id]
            sent_count = 0
            
            sync_data = {
                'type': 'acknowledgment_sync',
                'sync_token': sync_token,
                'acknowledgment_data': acknowledgment_data,
                'timestamp': datetime.now().isoformat()
            }
            
            for conn_info in connections[:]:
                try:
                    await conn_info.websocket.send_text(json.dumps(sync_data))
                    conn_info.last_activity = datetime.now()
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to send sync notification to connection {conn_info.connection_id}: {str(e)}")
                    connections.remove(conn_info)
            
            if sent_count > 0:
                logger.info(f"Sync notification sent to user {user_id}: {sync_token}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending sync notification to user {user_id}: {str(e)}")
            return False
    
    async def handle_acknowledgment_message(self, user_id: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming acknowledgment message from WebSocket
        
        Args:
            user_id: User ID sending the message
            message_data: Message data containing acknowledgment info
            
        Returns:
            Response data
        """
        try:
            message_type = message_data.get('type')
            acknowledgment_id = message_data.get('acknowledgment_id')
            
            if not acknowledgment_id:
                return {'error': 'Missing acknowledgment_id'}
            
            # Import here to avoid circular imports
            from app.services.acknowledgment_service import get_acknowledgment_service
            service = get_acknowledgment_service()
            
            if message_type == 'acknowledge':
                success = await service.acknowledge_alert(
                    acknowledgment_id=int(acknowledgment_id),
                    user_id=user_id,
                    response_message=message_data.get('response_message'),
                    response_data=message_data.get('response_data'),
                    device_id=message_data.get('device_id')
                )
                
                if success:
                    # Notify all user's devices
                    await self.send_acknowledgment_response(
                        user_id, acknowledgment_id, 'acknowledged', 
                        message_data.get('response_message')
                    )
                    return {'success': True, 'action': 'acknowledged'}
                else:
                    return {'error': 'Failed to acknowledge alert'}
            
            elif message_type == 'dismiss':
                success = await service.dismiss_alert(
                    acknowledgment_id=int(acknowledgment_id),
                    user_id=user_id,
                    reason=message_data.get('reason')
                )
                
                if success:
                    await self.send_acknowledgment_response(
                        user_id, acknowledgment_id, 'dismissed',
                        message_data.get('reason')
                    )
                    return {'success': True, 'action': 'dismissed'}
                else:
                    return {'error': 'Failed to dismiss alert'}
            
            elif message_type == 'escalate':
                success = await service.escalate_alert(
                    acknowledgment_id=int(acknowledgment_id),
                    user_id=user_id,
                    escalation_reason=message_data.get('escalation_reason')
                )
                
                if success:
                    await self.send_acknowledgment_response(
                        user_id, acknowledgment_id, 'escalated',
                        message_data.get('escalation_reason')
                    )
                    return {'success': True, 'action': 'escalated'}
                else:
                    return {'error': 'Failed to escalate alert'}
            
            elif message_type == 'snooze':
                success = await service.snooze_alert(
                    acknowledgment_id=int(acknowledgment_id),
                    user_id=user_id,
                    snooze_minutes=message_data.get('snooze_minutes', 30)
                )
                
                if success:
                    await self.send_acknowledgment_response(
                        user_id, acknowledgment_id, 'snoozed',
                        f"Snoozed for {message_data.get('snooze_minutes', 30)} minutes"
                    )
                    return {'success': True, 'action': 'snoozed'}
                else:
                    return {'error': 'Failed to snooze alert'}
            
            else:
                return {'error': f'Unknown message type: {message_type}'}
                
        except Exception as e:
            logger.error(f"Error handling acknowledgment message: {str(e)}")
            return {'error': f'Internal error: {str(e)}'}
    
    async def _send_pending_messages(self, user_id: str) -> None:
        """Send any pending messages to a newly connected user"""
        if user_id not in self.pending_messages:
            return
        
        messages = self.pending_messages[user_id]
        sent_count = 0
        
        for message in messages[:]:  # Copy to avoid modification during iteration
            success = await self.send_alert(user_id, message)
            if success:
                messages.remove(message)
                sent_count += 1
        
        # Clean up empty queue
        if not messages:
            del self.pending_messages[user_id]
        
        if sent_count > 0:
            logger.info(f"Sent {sent_count} pending messages to user {user_id}")
    
    async def _queue_message(self, user_id: str, alert: AlertMessage) -> None:
        """Queue a message for an offline user"""
        if user_id not in self.pending_messages:
            self.pending_messages[user_id] = []
        
        self.pending_messages[user_id].append(alert)
        
        # Limit queue size to prevent memory issues
        if len(self.pending_messages[user_id]) > 100:
            self.pending_messages[user_id] = self.pending_messages[user_id][-100:]
            logger.warning(f"Trimmed pending message queue for user {user_id} to 100 messages")
    
    async def _send_system_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Send a system message to a specific WebSocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send system message: {str(e)}")
    
    def _count_active_connections(self) -> int:
        """Count total active connections across all users"""
        return sum(len(connections) for connections in self.active_connections.values())


# Global connection manager instance
connection_manager = ConnectionManager()


async def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance"""
    return connection_manager


# Utility functions for creating alert messages
def create_alert_message(alert_id: str, user_id: str, alert_type: str,
                        title: str, message: str, severity: str = "medium",
                        data: Optional[Dict[str, Any]] = None,
                        acknowledgment_id: Optional[str] = None,
                        requires_acknowledgment: bool = True,
                        timeout_minutes: int = 15) -> AlertMessage:
    """
    Create an AlertMessage instance
    
    Args:
        alert_id: Unique alert identifier
        user_id: Target user ID
        alert_type: Type of alert (market, news, portfolio, etc.)
        title: Alert title
        message: Alert message
        severity: Alert severity (low, medium, high, critical)
        data: Additional alert data
        acknowledgment_id: Optional acknowledgment ID for tracking
        requires_acknowledgment: Whether alert requires user acknowledgment
        timeout_minutes: Minutes before acknowledgment times out
        
    Returns:
        AlertMessage instance
    """
    return AlertMessage(
        alert_id=alert_id,
        user_id=user_id,
        alert_type=alert_type,
        title=title,
        message=message,
        severity=severity,
        data=data or {},
        timestamp=datetime.now().isoformat(),
        acknowledgment_id=acknowledgment_id,
        requires_acknowledgment=requires_acknowledgment,
        timeout_minutes=timeout_minutes
    )


def create_market_alert(alert_id: str, user_id: str, asset_symbol: str,
                       impact_score: float, sentiment: str, 
                       reasoning: str) -> AlertMessage:
    """
    Create a market impact alert message
    
    Args:
        alert_id: Unique alert identifier
        user_id: Target user ID
        asset_symbol: Asset symbol affected
        impact_score: Market impact score (1-10)
        sentiment: Sentiment (positive/negative/neutral)
        reasoning: Reasoning for the alert
        
    Returns:
        AlertMessage instance
    """
    severity = "critical" if impact_score > 8 else "high" if impact_score > 6 else "medium"
    
    return AlertMessage(
        alert_id=alert_id,
        user_id=user_id,
        alert_type="market_impact",
        title=f"Market Impact Alert: {asset_symbol}",
        message=f"{sentiment.title()} sentiment detected for {asset_symbol} (impact: {impact_score:.1f}/10)",
        severity=severity,
        data={
            'asset_symbol': asset_symbol,
            'impact_score': impact_score,
            'sentiment': sentiment,
            'reasoning': reasoning
        },
        timestamp=datetime.now().isoformat()
    )
