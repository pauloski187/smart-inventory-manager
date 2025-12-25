"""
Streaming API endpoints for real-time updates.
Provides WebSocket connections and Server-Sent Events (SSE) for live data.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
from datetime import datetime
import asyncio
import json
import logging

from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..streaming.websocket_manager import (
    websocket_manager,
    handle_websocket_connection,
    SubscriptionChannel
)
from ..streaming.producer import get_producer, publish_alert_event
from ..streaming.events import EventType, InventoryEvent
from ..streaming.consumer import get_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Streaming"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==================== WebSocket Endpoints ====================

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    channels: Optional[str] = Query(None, description="Comma-separated channels: orders,stock,alerts,forecasts,analytics"),
    client_id: Optional[str] = Query(None, description="Client identifier for tracking")
):
    """
    WebSocket endpoint for real-time event streaming.

    Connect to receive live updates for inventory events.

    **Channels:**
    - `all` - Receive all events (default)
    - `orders` - Order-related events only
    - `stock` - Stock level changes
    - `alerts` - Alert notifications
    - `forecasts` - Forecast updates
    - `analytics` - Analytics summaries

    **Client Commands:**
    ```json
    {"action": "subscribe", "channel": "alerts"}
    {"action": "unsubscribe", "channel": "orders"}
    ```

    **Example Connection:**
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/stream/ws?channels=orders,alerts');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Event:', data);
    };
    ```
    """
    # Parse channels from query parameter
    channel_list = None
    if channels:
        channel_list = [c.strip() for c in channels.split(",")]

    await handle_websocket_connection(websocket, channel_list, client_id)


@router.websocket("/ws/orders")
async def websocket_orders(websocket: WebSocket):
    """WebSocket endpoint specifically for order events."""
    await handle_websocket_connection(
        websocket,
        channels=["orders"],
        client_id=None
    )


@router.websocket("/ws/stock")
async def websocket_stock(websocket: WebSocket):
    """WebSocket endpoint specifically for stock updates."""
    await handle_websocket_connection(
        websocket,
        channels=["stock"],
        client_id=None
    )


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint specifically for alerts."""
    await handle_websocket_connection(
        websocket,
        channels=["alerts"],
        client_id=None
    )


# ==================== Server-Sent Events (SSE) ====================

async def event_generator(channels: List[SubscriptionChannel]):
    """
    Generator for Server-Sent Events.
    Alternative to WebSocket for clients that prefer SSE.
    """
    # Create a queue for this SSE connection
    event_queue: asyncio.Queue = asyncio.Queue()

    # Subscribe to events (simplified - in production, integrate with Kafka consumer)
    async def forward_event(event: InventoryEvent):
        await event_queue.put(event)

    # Send initial connection event
    yield f"event: connected\ndata: {json.dumps({'message': 'Connected to event stream', 'channels': [c.value for c in channels]})}\n\n"

    # Keep-alive ping interval
    ping_interval = 30  # seconds
    last_ping = datetime.utcnow()

    try:
        while True:
            try:
                # Wait for events with timeout for keep-alive
                event = await asyncio.wait_for(
                    event_queue.get(),
                    timeout=ping_interval
                )

                # Format as SSE
                event_data = {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data
                }
                yield f"event: {event.event_type.value}\ndata: {json.dumps(event_data)}\n\n"

            except asyncio.TimeoutError:
                # Send keep-alive ping
                yield f"event: ping\ndata: {json.dumps({'timestamp': datetime.utcnow().isoformat()})}\n\n"

    except asyncio.CancelledError:
        yield f"event: disconnected\ndata: {json.dumps({'message': 'Connection closed'})}\n\n"


@router.get("/events")
async def sse_events(
    channels: Optional[str] = Query("all", description="Comma-separated channels")
):
    """
    Server-Sent Events endpoint for real-time updates.

    Alternative to WebSocket for environments where WebSocket is not available.

    **Usage:**
    ```javascript
    const eventSource = new EventSource('/stream/events?channels=orders,alerts');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Event:', data);
    };
    eventSource.addEventListener('order.created', (event) => {
        console.log('New order:', JSON.parse(event.data));
    });
    ```
    """
    # Parse channels
    channel_list = [SubscriptionChannel.ALL]
    if channels and channels != "all":
        try:
            channel_list = [SubscriptionChannel(c.strip()) for c in channels.split(",")]
        except ValueError:
            channel_list = [SubscriptionChannel.ALL]

    return StreamingResponse(
        event_generator(channel_list),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# ==================== Stream Management Endpoints ====================

@router.get("/status")
async def get_stream_status():
    """
    Get the current status of streaming connections.

    Returns:
        Connection counts and channel statistics
    """
    producer = await get_producer()

    return {
        "websocket": {
            "connected_clients": websocket_manager.get_connection_count(),
            "channel_stats": websocket_manager.get_channel_stats()
        },
        "kafka": {
            "enabled": producer.is_connected,
            "fallback_queue_size": producer.fallback_queue_size
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/broadcast/alert")
async def broadcast_alert(
    level: str = Query(..., regex="^(info|warning|critical)$"),
    title: str = Query(..., min_length=1, max_length=200),
    message: str = Query(..., min_length=1, max_length=1000)
):
    """
    Broadcast an alert to all connected clients.

    Useful for system-wide notifications.

    Args:
        level: Alert level (info, warning, critical)
        title: Alert title
        message: Alert message
    """
    await websocket_manager.send_alert(level, title, message)

    # Also publish to Kafka if enabled
    await publish_alert_event(
        event_type=EventType.ALERT_LOW_STOCK if level == "warning" else EventType.ALERT_REORDER_NEEDED,
        alert_level=level,
        title=title,
        message=message
    )

    return {
        "status": "broadcast_sent",
        "recipients": websocket_manager.get_connection_count(),
        "alert": {
            "level": level,
            "title": title,
            "message": message
        }
    }


@router.post("/broadcast/stock-update")
async def broadcast_stock_update(
    product_id: str = Query(...),
    product_name: str = Query(...),
    previous_stock: int = Query(..., ge=0),
    current_stock: int = Query(..., ge=0),
    reason: str = Query("manual_update")
):
    """
    Broadcast a stock update to connected clients.

    Typically called after inventory adjustments.
    """
    await websocket_manager.send_stock_update(
        product_id=product_id,
        product_name=product_name,
        previous_stock=previous_stock,
        current_stock=current_stock,
        change_reason=reason
    )

    return {
        "status": "broadcast_sent",
        "recipients": websocket_manager.get_connection_count(),
        "update": {
            "product_id": product_id,
            "product_name": product_name,
            "change": current_stock - previous_stock
        }
    }


@router.get("/fallback-queue")
async def get_fallback_queue(
    max_count: int = Query(100, ge=1, le=1000)
):
    """
    Retrieve events from the fallback queue.

    When Kafka is unavailable, events are stored in a fallback queue.
    Use this endpoint to retrieve and process those events.
    """
    producer = await get_producer()
    events = await producer.get_fallback_events(max_count)

    return {
        "count": len(events),
        "remaining": producer.fallback_queue_size,
        "events": events
    }


# ==================== Test Endpoints ====================

@router.post("/test/emit-event")
async def test_emit_event(
    event_type: str = Query("stock.updated"),
    product_id: str = Query("TEST-001"),
    message: str = Query("Test event")
):
    """
    Emit a test event for debugging WebSocket connections.

    Use this to verify that clients are receiving events correctly.
    """
    try:
        evt_type = EventType(event_type)
    except ValueError:
        evt_type = EventType.STOCK_UPDATED

    event = InventoryEvent(
        event_type=evt_type,
        data={
            "product_id": product_id,
            "message": message,
            "test": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    # Broadcast via WebSocket
    await websocket_manager.broadcast(event)

    # Publish to Kafka
    producer = await get_producer()
    await producer.publish(event)

    return {
        "status": "event_emitted",
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "websocket_recipients": websocket_manager.get_connection_count()
    }


@router.get("/test/simulate-low-stock")
async def simulate_low_stock_alert():
    """
    Simulate a low stock alert for testing.

    Broadcasts a fake low stock alert to all connected clients.
    """
    from ..streaming.events import AlertEvent

    event = AlertEvent.create(
        event_type=EventType.ALERT_LOW_STOCK,
        alert_level="warning",
        title="Low Stock Alert (Test)",
        message="Product 'Test Widget' has fallen below reorder threshold",
        affected_products=[
            {
                "product_id": "TEST-001",
                "product_name": "Test Widget",
                "current_stock": 5,
                "reorder_threshold": 10
            }
        ]
    )

    await websocket_manager.broadcast(event)

    producer = await get_producer()
    await producer.publish(event)

    return {
        "status": "alert_simulated",
        "event_id": event.event_id,
        "recipients": websocket_manager.get_connection_count()
    }
