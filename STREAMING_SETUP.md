# Real-Time Streaming Setup Guide

## Smart Inventory Manager - Kafka & WebSocket Integration

---

## Overview

The Smart Inventory Manager now supports real-time streaming updates via:
- **Apache Kafka** - Event streaming for distributed systems
- **WebSocket** - Real-time browser connections
- **Server-Sent Events (SSE)** - Alternative to WebSocket for simpler clients

---

## Quick Start

### Option 1: Without Kafka (WebSocket Only)

The system works without Kafka - events are handled in-memory and broadcast via WebSocket.

```bash
# Start the API server
cd backend
python main.py

# WebSocket is available at:
# ws://localhost:8000/stream/ws
```

### Option 2: With Kafka (Full Streaming)

For production or distributed systems:

```bash
# Start Kafka infrastructure
docker-compose -f docker-compose.kafka.yml up -d

# Wait for services to be healthy
docker-compose -f docker-compose.kafka.yml ps

# Enable Kafka in the app
export KAFKA_ENABLED=true
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Start the API server
cd backend
python main.py
```

---

## Configuration

### Environment Variables

Create a `.env` file in the `backend` directory:

```env
# Kafka Configuration
KAFKA_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_ORDERS=inventory.orders
KAFKA_TOPIC_STOCK=inventory.stock
KAFKA_TOPIC_ALERTS=inventory.alerts
KAFKA_TOPIC_FORECASTS=inventory.forecasts
KAFKA_CONSUMER_GROUP=inventory-manager

# Redis Configuration (optional)
REDIS_ENABLED=false
REDIS_URL=redis://localhost:6379

# Streaming Settings
STREAM_BATCH_SIZE=100
STREAM_FLUSH_INTERVAL=1.0
```

### Default Topics

| Topic | Purpose | Partitions |
|-------|---------|------------|
| `inventory.orders` | Order events | 3 |
| `inventory.stock` | Stock updates | 3 |
| `inventory.alerts` | Alert notifications | 1 |
| `inventory.forecasts` | Forecast updates | 1 |

---

## WebSocket API

### Connection

```javascript
// Connect to WebSocket with channel subscription
const ws = new WebSocket('ws://localhost:8000/stream/ws?channels=orders,stock,alerts');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data);
};

ws.onclose = () => {
  console.log('Disconnected');
  // Implement reconnection logic
};
```

### Channels

| Channel | Events |
|---------|--------|
| `all` | All events (default) |
| `orders` | order.created, order.updated, order.delivered |
| `stock` | stock.updated, stock.low, stock.out |
| `alerts` | alert.low_stock, alert.dead_stock, alert.reorder_needed |
| `forecasts` | forecast.generated, model.retrained |
| `analytics` | analytics.daily_summary, analytics.monthly_report |

### Client Commands

```javascript
// Subscribe to additional channel
ws.send(JSON.stringify({ action: 'subscribe', channel: 'forecasts' }));

// Unsubscribe from channel
ws.send(JSON.stringify({ action: 'unsubscribe', channel: 'orders' }));

// Respond to ping (keep-alive)
ws.send(JSON.stringify({ action: 'pong' }));
```

### Event Format

```json
{
  "type": "event",
  "event_id": "uuid-here",
  "event_type": "stock.low",
  "timestamp": "2024-12-25T10:30:00Z",
  "data": {
    "product_id": "P001",
    "product_name": "Yoga Mat",
    "current_stock": 5,
    "reorder_threshold": 10
  }
}
```

---

## Server-Sent Events (SSE)

Alternative to WebSocket for simpler implementations:

```javascript
const eventSource = new EventSource('/stream/events?channels=orders,alerts');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data);
};

eventSource.addEventListener('stock.low', (event) => {
  console.log('Low stock alert:', JSON.parse(event.data));
});

eventSource.onerror = (error) => {
  console.error('SSE error:', error);
};
```

---

## REST Endpoints

### Get Stream Status

```bash
curl http://localhost:8000/stream/status
```

Response:
```json
{
  "websocket": {
    "connected_clients": 5,
    "channel_stats": {
      "all": 2,
      "orders": 3,
      "stock": 1,
      "alerts": 4
    }
  },
  "kafka": {
    "enabled": true,
    "fallback_queue_size": 0
  },
  "timestamp": "2024-12-25T10:30:00Z"
}
```

### Broadcast Alert

```bash
curl -X POST "http://localhost:8000/stream/broadcast/alert?level=warning&title=Low%20Stock&message=Product%20X%20below%20threshold"
```

### Test Event

```bash
curl -X POST "http://localhost:8000/stream/test/emit-event?event_type=stock.updated&product_id=TEST-001&message=Test%20event"
```

---

## Event Types Reference

### Order Events

| Event | Description | Data |
|-------|-------------|------|
| `order.created` | New order placed | order_id, product_id, quantity, total_amount |
| `order.updated` | Order status changed | order_id, status |
| `order.cancelled` | Order cancelled | order_id, reason |
| `order.delivered` | Order delivered | order_id, delivery_date |

### Stock Events

| Event | Description | Data |
|-------|-------------|------|
| `stock.updated` | Stock level changed | product_id, previous_stock, current_stock |
| `stock.low` | Below reorder threshold | product_id, current_stock, threshold |
| `stock.out` | Zero inventory | product_id |
| `stock.replenished` | Stock added | product_id, quantity_added, new_stock |

### Alert Events

| Event | Description | Data |
|-------|-------------|------|
| `alert.low_stock` | Low stock warning | affected_products[], alert_level |
| `alert.dead_stock` | Dead stock detected | product_id, days_since_sale, value |
| `alert.reorder_needed` | Reorder recommended | product_id, recommended_quantity |
| `alert.stockout_risk` | Stockout prediction | product_id, days_until_stockout |

### Forecast Events

| Event | Description | Data |
|-------|-------------|------|
| `forecast.generated` | New forecast ready | category, horizon, predicted_demand |
| `forecast.updated` | Forecast revised | category, old_forecast, new_forecast |
| `model.retrained` | Model updated | category, metrics |

---

## Integration Examples

### React Hook

```typescript
import { useEffect, useState } from 'react';

function useInventoryStream(channels: string[] = ['all']) {
  const [events, setEvents] = useState<any[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(
      `ws://localhost:8000/stream/ws?channels=${channels.join(',')}`
    );

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'event') {
        setEvents((prev) => [data, ...prev.slice(0, 99)]);
      }
    };

    return () => ws.close();
  }, [channels.join(',')]);

  return { events, connected };
}
```

### Python Client

```python
import asyncio
import websockets
import json

async def listen_to_stream():
    uri = "ws://localhost:8000/stream/ws?channels=orders,alerts"

    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            event = json.loads(message)
            print(f"Received: {event['event_type']}")

            if event.get('event_type') == 'stock.low':
                handle_low_stock_alert(event['data'])

asyncio.run(listen_to_stream())
```

---

## Kafka UI (Optional)

To monitor Kafka topics and messages:

```bash
# Start Kafka with UI
docker-compose -f docker-compose.kafka.yml --profile ui up -d

# Access UI at:
# http://localhost:8080
```

---

## Troubleshooting

### WebSocket Not Connecting

1. Check if server is running: `curl http://localhost:8000/health`
2. Check CORS settings if connecting from different origin
3. Browser console for WebSocket errors

### Kafka Connection Failed

1. Check if Kafka is running: `docker-compose -f docker-compose.kafka.yml ps`
2. Verify KAFKA_ENABLED=true in environment
3. Check logs: `docker-compose -f docker-compose.kafka.yml logs kafka`

### Events Not Being Received

1. Verify channel subscription in WebSocket URL
2. Check stream status: `GET /stream/status`
3. Test with: `POST /stream/test/emit-event`

### Fallback Mode

If Kafka is unavailable, the system automatically falls back to in-memory queue:
- Events are still broadcast via WebSocket
- Fallback queue can be checked via `/stream/fallback-queue`
- No data loss for connected clients

---

## Production Recommendations

1. **Kafka**: Use managed Kafka (AWS MSK, Confluent Cloud) for production
2. **Redis**: Add Redis for WebSocket pub/sub across multiple API instances
3. **Load Balancing**: Use sticky sessions for WebSocket connections
4. **Monitoring**: Set up alerts for Kafka lag and WebSocket connection counts
5. **Security**: Add authentication to WebSocket connections in production

---

*Documentation updated: December 25, 2024*
*Version: 2.1.0*
