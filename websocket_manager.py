"""
WebSocket Manager for Real-time GDP Data Updates
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import redis.asyncio as redis

from core.config import settings
from services.auth_service import verify_websocket_token

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.connection_subscriptions: Dict[str, Set[str]] = {}  # connection_id -> topics
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize Redis connection for pub/sub"""
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
            await self.redis_client.ping()
            logger.info("WebSocket manager initialized with Redis")
        except Exception as e:
            logger.error(f"Failed to initialize Redis for WebSocket: {e}")
    
    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str = None) -> str:
        """Accept WebSocket connection and register user"""
        await websocket.accept()
        
        connection_id = connection_id or str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        # Associate connection with user
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Initialize subscriptions
        self.connection_subscriptions[connection_id] = set()
        
        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection",
            "status": "connected",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
        
        return connection_id
    
    def disconnect(self, connection_id: str, user_id: str = None):
        """Remove WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if connection_id in self.connection_subscriptions:
            del self.connection_subscriptions[connection_id]
        
        # Remove from user connections
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def send_to_user(self, message: Dict[str, Any], user_id: str):
        """Send message to all connections of a user"""
        if user_id in self.user_connections:
            connection_ids = self.user_connections[user_id].copy()
            for connection_id in connection_ids:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_topic(self, message: Dict[str, Any], topic: str):
        """Broadcast message to all connections subscribed to a topic"""
        message["topic"] = topic
        message["timestamp"] = datetime.utcnow().isoformat()
        
        for connection_id, subscriptions in self.connection_subscriptions.items():
            if topic in subscriptions:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        message["timestamp"] = datetime.utcnow().isoformat()
        
        connection_ids = list(self.active_connections.keys())
        for connection_id in connection_ids:
            await self.send_personal_message(message, connection_id)
    
    def subscribe_to_topic(self, connection_id: str, topic: str):
        """Subscribe connection to a topic"""
        if connection_id in self.connection_subscriptions:
            self.connection_subscriptions[connection_id].add(topic)
            logger.info(f"Connection {connection_id} subscribed to {topic}")
    
    def unsubscribe_from_topic(self, connection_id: str, topic: str):
        """Unsubscribe connection from a topic"""
        if connection_id in self.connection_subscriptions:
            self.connection_subscriptions[connection_id].discard(topic)
            logger.info(f"Connection {connection_id} unsubscribed from {topic}")
    
    async def handle_redis_message(self, channel: str, message: str):
        """Handle message from Redis pub/sub"""
        try:
            data = json.loads(message)
            await self.broadcast_to_topic(data, channel)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from Redis channel {channel}: {message}")
    
    async def publish_to_redis(self, topic: str, message: Dict[str, Any]):
        """Publish message to Redis for distribution across instances"""
        if self.redis_client:
            try:
                await self.redis_client.publish(topic, json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Error publishing to Redis: {e}")


# Global connection manager instance
manager = ConnectionManager()


class WebSocketHandler:
    """WebSocket event handlers for GDP platform"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
    
    async def handle_message(self, websocket: WebSocket, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        message_type = message.get("type")
        
        try:
            if message_type == "subscribe":
                await self.handle_subscribe(connection_id, message)
            elif message_type == "unsubscribe":
                await self.handle_unsubscribe(connection_id, message)
            elif message_type == "gdp_query":
                await self.handle_gdp_query(connection_id, message)
            elif message_type == "forecast_request":
                await self.handle_forecast_request(connection_id, message)
            elif message_type == "ping":
                await self.handle_ping(connection_id, message)
            else:
                await self.send_error(connection_id, "Unknown message type", message_type)
        
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.send_error(connection_id, "Internal server error", str(e))
    
    async def handle_subscribe(self, connection_id: str, message: Dict[str, Any]):
        """Handle subscription request"""
        topic = message.get("topic")
        if not topic:
            await self.send_error(connection_id, "Topic is required for subscription")
            return
        
        # Validate topic
        valid_topics = [
            "gdp_updates",
            "forecast_updates", 
            "data_quality_alerts",
            "system_notifications",
            f"country_{message.get('country_code', '').upper()}"
        ]
        
        if topic not in valid_topics and not topic.startswith("country_"):
            await self.send_error(connection_id, f"Invalid topic: {topic}")
            return
        
        self.manager.subscribe_to_topic(connection_id, topic)
        
        await self.manager.send_personal_message({
            "type": "subscription_confirmed",
            "topic": topic,
            "status": "subscribed"
        }, connection_id)
    
    async def handle_unsubscribe(self, connection_id: str, message: Dict[str, Any]):
        """Handle unsubscription request"""
        topic = message.get("topic")
        if not topic:
            await self.send_error(connection_id, "Topic is required for unsubscription")
            return
        
        self.manager.unsubscribe_from_topic(connection_id, topic)
        
        await self.manager.send_personal_message({
            "type": "unsubscription_confirmed",
            "topic": topic,
            "status": "unsubscribed"
        }, connection_id)
    
    async def handle_gdp_query(self, connection_id: str, message: Dict[str, Any]):
        """Handle real-time GDP query"""
        query_data = message.get("data", {})
        
        # Process query asynchronously
        await self.manager.send_personal_message({
            "type": "query_processing",
            "query_id": message.get("query_id"),
            "status": "processing"
        }, connection_id)
        
        # In a real implementation, this would trigger actual GDP calculation
        # For now, send a mock response after a short delay
        await asyncio.sleep(1)
        
        await self.manager.send_personal_message({
            "type": "query_result",
            "query_id": message.get("query_id"),
            "status": "completed",
            "data": {
                "gdp_value": 25000.0,
                "country_code": query_data.get("country_code"),
                "period": query_data.get("period"),
                "method": query_data.get("method", "expenditure")
            }
        }, connection_id)
    
    async def handle_forecast_request(self, connection_id: str, message: Dict[str, Any]):
        """Handle real-time forecast request"""
        forecast_data = message.get("data", {})
        
        await self.manager.send_personal_message({
            "type": "forecast_processing",
            "request_id": message.get("request_id"),
            "status": "processing"
        }, connection_id)
        
        # Mock forecast processing
        await asyncio.sleep(2)
        
        await self.manager.send_personal_message({
            "type": "forecast_result",
            "request_id": message.get("request_id"),
            "status": "completed",
            "data": {
                "predictions": [25100.0, 25250.0, 25400.0, 25600.0],
                "confidence_intervals": [
                    [24800.0, 25400.0],
                    [24900.0, 25600.0],
                    [25000.0, 25800.0],
                    [25200.0, 26000.0]
                ],
                "country_code": forecast_data.get("country_code"),
                "horizon": forecast_data.get("horizon", 4)
            }
        }, connection_id)
    
    async def handle_ping(self, connection_id: str, message: Dict[str, Any]):
        """Handle ping/keepalive message"""
        await self.manager.send_personal_message({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)
    
    async def send_error(self, connection_id: str, error_message: str, details: str = None):
        """Send error message to connection"""
        await self.manager.send_personal_message({
            "type": "error",
            "message": error_message,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }, connection_id)


# Global WebSocket handler instance
websocket_handler = WebSocketHandler(manager)


# WebSocket endpoint function
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    """WebSocket endpoint for real-time communication"""
    
    # Verify authentication token
    try:
        user_data = verify_websocket_token(token) if token else None
        user_id = user_data.get("user_id", "anonymous") if user_data else "anonymous"
    except Exception as e:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return
    
    connection_id = None
    
    try:
        # Connect WebSocket
        connection_id = await manager.connect(websocket, user_id)
        
        # Main message loop
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await websocket_handler.handle_message(websocket, connection_id, message)
            except json.JSONDecodeError:
                await websocket_handler.send_error(
                    connection_id, 
                    "Invalid JSON format"
                )
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if connection_id:
            manager.disconnect(connection_id, user_id)


# Background task for broadcasting updates
async def broadcast_gdp_updates():
    """Background task to broadcast GDP data updates"""
    while True:
        try:
            # In a real implementation, this would check for new GDP data
            # and broadcast updates to subscribed connections
            
            # Mock update every 30 seconds
            await asyncio.sleep(30)
            
            sample_update = {
                "type": "gdp_data_update",
                "data": {
                    "country_code": "USA",
                    "latest_gdp": 25100.0,
                    "change": 0.4,
                    "period": "2024-Q2"
                }
            }
            
            await manager.broadcast_to_topic(sample_update, "gdp_updates")
            await manager.publish_to_redis("gdp_updates", sample_update)
            
        except Exception as e:
            logger.error(f"Error in broadcast task: {e}")
            await asyncio.sleep(5)


# Initialize manager on startup
async def initialize_websocket_manager():
    """Initialize WebSocket manager"""
    await manager.initialize()
    
    # Start background broadcast task
    asyncio.create_task(broadcast_gdp_updates())