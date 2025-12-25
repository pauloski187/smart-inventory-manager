"""
WebSocket Manager for real-time client connections.
Provides pub/sub pattern for broadcasting events to connected clients.
"""

import asyncio
import logging
import json
from typing import Dict, Set, Optional, Any, List
from datetime import datetime
from enum import Enum
from fastapi import WebSocket, WebSocketDisconnect

from .events import InventoryEvent, EventType

logger = logging.getLogger(__name__)


class SubscriptionChannel(str, Enum):
    """Available subscription channels for clients."""
    ALL = "all"
    ORDERS = "orders"
    STOCK = "stock"
    ALERTS = "alerts"
    FORECASTS = "forecasts"
    ANALYTICS = "analytics"


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts events to subscribers.
    Supports channel-based subscriptions for filtering events.
    """

    def __init__(self):
        # Map of channel -> set of connected websockets
        self._subscriptions: Dict[SubscriptionChannel, Set[WebSocket]] = {
            channel: set() for channel in SubscriptionChannel
        }
        # Map of websocket -> set of subscribed channels
        self._client_channels: Dict[WebSocket, Set[SubscriptionChannel]] = {}
        # Connection metadata
        self._connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        # Message queue for buffering
        self._message_queue: asyncio.Queue = asyncio.Queue()
        # Background broadcast task
        self._broadcast_task: Optional[asyncio.Task] = None

    async def connect(
        self,
        websocket: WebSocket,
        channels: Optional[List[SubscriptionChannel]] = None,
        client_id: Optional[str] = None
    ):
        """
        Accept a new WebSocket connection and subscribe to channels.

        Args:
            websocket: The WebSocket connection
            channels: List of channels to subscribe to (default: ALL)
            client_id: Optional client identifier for tracking
        """
        await websocket.accept()

        async with self._lock:
            # Default to ALL channel if none specified
            subscribe_channels = channels or [SubscriptionChannel.ALL]

            self._client_channels[websocket] = set(subscribe_channels)
            self._connection_info[websocket] = {
                "client_id": client_id,
                "connected_at": datetime.utcnow().isoformat(),
                "channels": [c.value for c in subscribe_channels]
            }

            for channel in subscribe_channels:
                self._subscriptions[channel].add(websocket)

        logger.info(
            f"WebSocket connected: {client_id or 'anonymous'} "
            f"subscribed to {[c.value for c in subscribe_channels]}"
        )

        # Send welcome message
        await self._send_to_client(websocket, {
            "type": "connected",
            "message": "Connected to inventory event stream",
            "channels": [c.value for c in subscribe_channels],
            "timestamp": datetime.utcnow().isoformat()
        })

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection from all subscriptions."""
        async with self._lock:
            channels = self._client_channels.pop(websocket, set())
            info = self._connection_info.pop(websocket, {})

            for channel in channels:
                self._subscriptions[channel].discard(websocket)

        logger.info(f"WebSocket disconnected: {info.get('client_id', 'anonymous')}")

    async def subscribe(self, websocket: WebSocket, channel: SubscriptionChannel):
        """Subscribe a client to an additional channel."""
        async with self._lock:
            if websocket in self._client_channels:
                self._client_channels[websocket].add(channel)
                self._subscriptions[channel].add(websocket)

                # Update connection info
                if websocket in self._connection_info:
                    self._connection_info[websocket]["channels"].append(channel.value)

    async def unsubscribe(self, websocket: WebSocket, channel: SubscriptionChannel):
        """Unsubscribe a client from a channel."""
        async with self._lock:
            if websocket in self._client_channels:
                self._client_channels[websocket].discard(channel)
                self._subscriptions[channel].discard(websocket)

    async def broadcast(self, event: InventoryEvent):
        """
        Broadcast an event to all relevant subscribers.

        Args:
            event: The inventory event to broadcast
        """
        # Determine which channel this event belongs to
        channels = self._get_channels_for_event(event)

        message = {
            "type": "event",
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data
        }

        # Collect unique websockets to notify
        websockets_to_notify: Set[WebSocket] = set()

        async with self._lock:
            # Always include ALL channel subscribers
            websockets_to_notify.update(self._subscriptions[SubscriptionChannel.ALL])

            # Add channel-specific subscribers
            for channel in channels:
                websockets_to_notify.update(self._subscriptions[channel])

        # Send to all relevant clients
        await self._broadcast_to_clients(websockets_to_notify, message)

    async def broadcast_message(
        self,
        message: Dict[str, Any],
        channels: Optional[List[SubscriptionChannel]] = None
    ):
        """
        Broadcast a custom message to subscribers.

        Args:
            message: The message to broadcast
            channels: Optional list of channels to broadcast to
        """
        target_channels = channels or [SubscriptionChannel.ALL]

        websockets_to_notify: Set[WebSocket] = set()

        async with self._lock:
            for channel in target_channels:
                websockets_to_notify.update(self._subscriptions[channel])

        await self._broadcast_to_clients(websockets_to_notify, message)

    def _get_channels_for_event(self, event: InventoryEvent) -> List[SubscriptionChannel]:
        """Determine which channels an event should be broadcast to."""
        event_type = event.event_type.value

        if event_type.startswith("order."):
            return [SubscriptionChannel.ORDERS]
        elif event_type.startswith("stock."):
            return [SubscriptionChannel.STOCK]
        elif event_type.startswith("alert."):
            return [SubscriptionChannel.ALERTS]
        elif event_type.startswith("forecast.") or event_type.startswith("model."):
            return [SubscriptionChannel.FORECASTS]
        elif event_type.startswith("analytics."):
            return [SubscriptionChannel.ANALYTICS]
        else:
            return []

    async def _send_to_client(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a single client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to client: {e}")
            await self.disconnect(websocket)

    async def _broadcast_to_clients(
        self,
        websockets: Set[WebSocket],
        message: Dict[str, Any]
    ):
        """Broadcast a message to multiple clients."""
        disconnected = []

        for websocket in websockets:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws)

    async def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Send an alert notification to all alert subscribers."""
        alert_message = {
            "type": "alert",
            "level": level,
            "title": title,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.broadcast_message(
            alert_message,
            [SubscriptionChannel.ALERTS, SubscriptionChannel.ALL]
        )

    async def send_stock_update(
        self,
        product_id: str,
        product_name: str,
        previous_stock: int,
        current_stock: int,
        change_reason: str = "update"
    ):
        """Send a stock update notification."""
        stock_message = {
            "type": "stock_update",
            "product_id": product_id,
            "product_name": product_name,
            "previous_stock": previous_stock,
            "current_stock": current_stock,
            "change": current_stock - previous_stock,
            "reason": change_reason,
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.broadcast_message(
            stock_message,
            [SubscriptionChannel.STOCK, SubscriptionChannel.ALL]
        )

    def get_connection_count(self) -> int:
        """Get the total number of connected clients."""
        return len(self._client_channels)

    def get_channel_stats(self) -> Dict[str, int]:
        """Get subscriber count per channel."""
        return {
            channel.value: len(websockets)
            for channel, websockets in self._subscriptions.items()
        }

    async def ping_clients(self):
        """Send a ping to all connected clients to check connectivity."""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        }

        all_clients = set(self._client_channels.keys())
        await self._broadcast_to_clients(all_clients, ping_message)


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


async def handle_websocket_connection(
    websocket: WebSocket,
    channels: Optional[List[str]] = None,
    client_id: Optional[str] = None
):
    """
    Handle a WebSocket connection lifecycle.

    Args:
        websocket: The WebSocket connection
        channels: List of channel names to subscribe to
        client_id: Optional client identifier
    """
    # Parse channel names to enum values
    subscribe_channels = None
    if channels:
        try:
            subscribe_channels = [SubscriptionChannel(c) for c in channels]
        except ValueError:
            subscribe_channels = [SubscriptionChannel.ALL]

    await websocket_manager.connect(websocket, subscribe_channels, client_id)

    try:
        while True:
            # Handle incoming messages from client
            data = await websocket.receive_json()

            # Process client commands
            if data.get("action") == "subscribe":
                channel = data.get("channel")
                if channel:
                    try:
                        await websocket_manager.subscribe(
                            websocket,
                            SubscriptionChannel(channel)
                        )
                        await websocket.send_json({
                            "type": "subscribed",
                            "channel": channel
                        })
                    except ValueError:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Invalid channel: {channel}"
                        })

            elif data.get("action") == "unsubscribe":
                channel = data.get("channel")
                if channel:
                    try:
                        await websocket_manager.unsubscribe(
                            websocket,
                            SubscriptionChannel(channel)
                        )
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "channel": channel
                        })
                    except ValueError:
                        pass

            elif data.get("action") == "pong":
                # Client responded to ping
                pass

    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket_manager.disconnect(websocket)
