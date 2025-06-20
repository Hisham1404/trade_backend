"""
WebSocket Routes

This module provides WebSocket endpoints for real-time alert delivery,
including user authentication, connection management, and live communication.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException
from fastapi.responses import JSONResponse

from app.websockets.manager import connection_manager, create_alert_message, create_market_alert
from app.models.user import User
from app.database.connection import get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ws", tags=["WebSocket"])


async def get_current_user_ws(websocket: WebSocket, token: str = Query(...)) -> Optional[str]:
    """
    Get current user for WebSocket connection (simplified authentication)
    In production, this should validate JWT tokens properly
    
    Args:
        websocket: WebSocket instance
        token: Authentication token from query params
        
    Returns:
        User ID if authenticated, None otherwise
    """
    try:
        # Simplified authentication - in production, validate JWT token
        # For demo purposes, accept any token and extract user_id
        if token.startswith("user_"):
            user_id = token.replace("user_", "")
            return user_id
        else:
            # Default demo user
            return "demo_user"
    except Exception as e:
        logger.error(f"WebSocket authentication failed: {str(e)}")
        return None


@router.websocket("/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket, 
                                  user_id: str = Depends(get_current_user_ws)):
    """
    WebSocket endpoint for real-time alert delivery
    
    Usage:
        ws://localhost:8000/api/v1/ws/alerts?token=user_123
    """
    if not user_id:
        await websocket.close(code=4001, reason="Authentication failed")
        return
    
    # Connection metadata
    metadata = {
        'client_type': 'web',
        'user_agent': websocket.headers.get('user-agent', 'unknown'),
        'origin': websocket.headers.get('origin', 'unknown')
    }
    
    try:
        # Accept and register the connection
        connection_id = await connection_manager.connect(websocket, user_id, metadata)
        logger.info(f"WebSocket connection established: user={user_id}, connection_id={connection_id}")
        
        # Send welcome message
        welcome_message = {
            'type': 'welcome',
            'message': f'Connected to real-time alerts for user {user_id}',
            'connection_id': connection_id,
            'features': ['alerts', 'market_updates', 'portfolio_notifications'],
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send_text(json.dumps(welcome_message))
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_text()
                
                # Parse client message
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    # Send error response for invalid JSON
                    error_response = {
                        'type': 'error',
                        'message': 'Invalid JSON format',
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(error_response))
                    continue
                
                # Handle different message types from client
                await handle_client_message(websocket, user_id, connection_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: user={user_id}, connection_id={connection_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket message handling: {str(e)}")
                # Send error response
                error_response = {
                    'type': 'error',
                    'message': 'Internal server error',
                    'timestamp': datetime.now().isoformat()
                }
                try:
                    await websocket.send_text(json.dumps(error_response))
                except:
                    # Connection likely broken
                    break
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    
    finally:
        # Clean up connection
        connection_manager.disconnect(websocket, user_id)


async def handle_client_message(websocket: WebSocket, user_id: str, 
                               connection_id: str, message: Dict[str, Any]):
    """
    Handle incoming messages from WebSocket clients
    
    Args:
        websocket: WebSocket instance
        user_id: User ID
        connection_id: Connection ID
        message: Parsed message from client
    """
    try:
        message_type = message.get('type', 'unknown')
        
        if message_type == 'ping':
            # Respond to ping with pong
            pong_response = {
                'type': 'pong',
                'timestamp': datetime.now().isoformat(),
                'connection_id': connection_id
            }
            await websocket.send_text(json.dumps(pong_response))
            
        elif message_type == 'subscribe':
            # Handle subscription to specific alert types
            alert_types = message.get('alert_types', [])
            assets = message.get('assets', [])
            
            # Store subscription preferences (in production, save to database)
            subscription_response = {
                'type': 'subscription_confirmed',
                'alert_types': alert_types,
                'assets': assets,
                'message': f'Subscribed to {len(alert_types)} alert types for {len(assets)} assets',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(subscription_response))
            
        elif message_type == 'get_stats':
            # Send connection statistics
            stats = connection_manager.get_connection_stats()
            stats_response = {
                'type': 'stats',
                'data': stats,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(stats_response))
            
        elif message_type == 'heartbeat':
            # Update last activity and respond
            heartbeat_response = {
                'type': 'heartbeat_ack',
                'timestamp': datetime.now().isoformat(),
                'connection_id': connection_id
            }
            await websocket.send_text(json.dumps(heartbeat_response))
            
        elif message_type in ['acknowledge', 'dismiss', 'escalate', 'snooze']:
            # Handle acknowledgment-related messages
            response = await connection_manager.handle_acknowledgment_message(user_id, message)
            
            # Send response back to client
            acknowledgment_response = {
                'type': 'acknowledgment_result',
                'action': message_type,
                'result': response,
                'timestamp': datetime.now().isoformat(),
                'connection_id': connection_id
            }
            await websocket.send_text(json.dumps(acknowledgment_response))
            
        elif message_type == 'sync_request':
            # Handle synchronization request
            sync_token = message.get('sync_token')
            device_id = message.get('device_id', connection_id)
            
            if sync_token:
                # Import here to avoid circular imports
                from app.services.acknowledgment_service import get_acknowledgment_service
                service = get_acknowledgment_service()
                
                success = await service.sync_acknowledgment_across_devices(sync_token, device_id)
                
                sync_response = {
                    'type': 'sync_result',
                    'sync_token': sync_token,
                    'success': success,
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(sync_response))
            else:
                error_response = {
                    'type': 'sync_error',
                    'message': 'Missing sync_token',
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(error_response))
            
        else:
            # Unknown message type
            unknown_response = {
                'type': 'unknown_message_type',
                'received_type': message_type,
                'message': f'Unknown message type: {message_type}',
                'supported_types': ['ping', 'subscribe', 'get_stats', 'heartbeat', 'acknowledge', 'dismiss', 'escalate', 'snooze', 'sync_request'],
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(unknown_response))
            
    except Exception as e:
        logger.error(f"Error handling client message: {str(e)}")
        error_response = {
            'type': 'message_handling_error',
            'message': 'Failed to process client message',
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send_text(json.dumps(error_response))


# REST API endpoints for WebSocket management
@router.get("/connections/stats")
async def get_connection_stats():
    """Get WebSocket connection statistics"""
    try:
        stats = connection_manager.get_connection_stats()
        return JSONResponse(content={
            'success': True,
            'data': stats
        })
    except Exception as e:
        logger.error(f"Error getting connection stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get connection statistics")


@router.get("/connections/user/{user_id}")
async def get_user_connections(user_id: str):
    """Get connection information for a specific user"""
    try:
        connections = connection_manager.get_user_connections(user_id)
        return JSONResponse(content={
            'success': True,
            'user_id': user_id,
            'connections': connections,
            'connection_count': len(connections)
        })
    except Exception as e:
        logger.error(f"Error getting user connections: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user connections")


@router.post("/send-alert")
async def send_test_alert(user_id: str, alert_type: str = "test", 
                         title: str = "Test Alert", message: str = "This is a test alert"):
    """
    Send a test alert to a specific user (for testing purposes)
    
    Args:
        user_id: Target user ID
        alert_type: Type of alert
        title: Alert title
        message: Alert message
    """
    try:
        # Create test alert
        alert = create_alert_message(
            alert_id=f"test_{int(datetime.now().timestamp())}",
            user_id=user_id,
            alert_type=alert_type,
            title=title,
            message=message,
            severity="medium",
            data={'test': True, 'source': 'manual'}
        )
        
        # Send alert
        success = await connection_manager.send_alert(user_id, alert)
        
        return JSONResponse(content={
            'success': success,
            'alert_id': alert.alert_id,
            'user_id': user_id,
            'message': 'Alert sent successfully' if success else 'User not connected, alert queued'
        })
        
    except Exception as e:
        logger.error(f"Error sending test alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send test alert")


@router.post("/broadcast")
async def broadcast_system_message(title: str, message: str, 
                                  message_type: str = "system"):
    """
    Broadcast a system message to all connected users
    
    Args:
        title: Message title
        message: Message content
        message_type: Type of message
    """
    try:
        # Broadcast message
        sent_count = await connection_manager.broadcast_system_message({
            'title': title,
            'message': message,
            'message_type': message_type
        })
        
        return JSONResponse(content={
            'success': True,
            'sent_to_connections': sent_count,
            'message': f'Message broadcasted to {sent_count} connections'
        })
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to broadcast message")


@router.post("/send-market-alert")
async def send_market_alert_endpoint(user_id: str, asset_symbol: str, 
                                   impact_score: float, sentiment: str,
                                   reasoning: str):
    """
    Send a market impact alert to a specific user
    
    Args:
        user_id: Target user ID
        asset_symbol: Asset symbol
        impact_score: Market impact score (1-10)
        sentiment: Sentiment (positive/negative/neutral)
        reasoning: Reasoning for the alert
    """
    try:
        # Create market alert
        alert = create_market_alert(
            alert_id=f"market_{asset_symbol}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            asset_symbol=asset_symbol,
            impact_score=impact_score,
            sentiment=sentiment,
            reasoning=reasoning
        )
        
        # Send alert
        success = await connection_manager.send_alert(user_id, alert)
        
        return JSONResponse(content={
            'success': success,
            'alert_id': alert.alert_id,
            'user_id': user_id,
            'asset_symbol': asset_symbol,
            'message': 'Market alert sent successfully' if success else 'User not connected, alert queued'
        })
        
    except Exception as e:
        logger.error(f"Error sending market alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send market alert") 