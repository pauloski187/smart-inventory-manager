"""
Kafka Producer for publishing inventory events.
Handles connection management and message serialization.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from ..config.config import config
from .events import InventoryEvent, EventType

logger = logging.getLogger(__name__)

# Global producer instance
_producer: Optional["KafkaEventProducer"] = None


class KafkaEventProducer:
    """
    Async Kafka producer for publishing inventory events.
    Falls back to in-memory queue if Kafka is not available.
    """

    def __init__(self):
        self._producer = None
        self._started = False
        self._fallback_queue: asyncio.Queue = asyncio.Queue()
        self._use_fallback = not config.kafka_enabled

    async def start(self):
        """Initialize the Kafka producer connection."""
        if self._started:
            return

        if config.kafka_enabled:
            try:
                from aiokafka import AIOKafkaProducer

                self._producer = AIOKafkaProducer(
                    bootstrap_servers=config.kafka_bootstrap_servers,
                    value_serializer=lambda v: v.to_bytes() if isinstance(v, InventoryEvent) else v,
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    retries=3,
                    retry_backoff_ms=100,
                )
                await self._producer.start()
                self._use_fallback = False
                logger.info(f"Kafka producer connected to {config.kafka_bootstrap_servers}")
            except Exception as e:
                logger.warning(f"Failed to connect to Kafka: {e}. Using fallback mode.")
                self._use_fallback = True
        else:
            logger.info("Kafka disabled. Using in-memory fallback queue.")
            self._use_fallback = True

        self._started = True

    async def stop(self):
        """Close the Kafka producer connection."""
        if self._producer:
            await self._producer.stop()
            self._producer = None
        self._started = False
        logger.info("Kafka producer stopped")

    def _get_topic_for_event(self, event: InventoryEvent) -> str:
        """Determine the Kafka topic based on event type."""
        event_type = event.event_type.value

        if event_type.startswith("order."):
            return config.kafka_topic_orders
        elif event_type.startswith("stock."):
            return config.kafka_topic_stock
        elif event_type.startswith("alert."):
            return config.kafka_topic_alerts
        elif event_type.startswith("forecast.") or event_type.startswith("model."):
            return config.kafka_topic_forecasts
        elif event_type.startswith("analytics."):
            return config.kafka_topic_alerts  # Analytics go to alerts topic
        else:
            return config.kafka_topic_alerts  # Default topic

    async def publish(
        self,
        event: InventoryEvent,
        topic: Optional[str] = None,
        key: Optional[str] = None
    ) -> bool:
        """
        Publish an event to Kafka.

        Args:
            event: The event to publish
            topic: Optional topic override
            key: Optional partition key

        Returns:
            True if published successfully
        """
        if not self._started:
            await self.start()

        target_topic = topic or self._get_topic_for_event(event)
        partition_key = key or event.data.get("product_id") or event.event_id

        if self._use_fallback:
            # Use fallback queue
            await self._fallback_queue.put({
                "topic": target_topic,
                "key": partition_key,
                "event": event.model_dump()
            })
            logger.debug(f"Event {event.event_type} queued in fallback (queue size: {self._fallback_queue.qsize()})")
            return True

        try:
            await self._producer.send_and_wait(
                topic=target_topic,
                value=event,
                key=partition_key
            )
            logger.debug(f"Published event {event.event_type} to {target_topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            # Fall back to queue on failure
            await self._fallback_queue.put({
                "topic": target_topic,
                "key": partition_key,
                "event": event.model_dump()
            })
            return False

    async def publish_batch(self, events: list[InventoryEvent]) -> int:
        """
        Publish multiple events.

        Returns:
            Number of successfully published events
        """
        success_count = 0
        for event in events:
            if await self.publish(event):
                success_count += 1
        return success_count

    async def get_fallback_events(self, max_count: int = 100) -> list[Dict[str, Any]]:
        """Retrieve events from the fallback queue."""
        events = []
        try:
            while len(events) < max_count:
                event = self._fallback_queue.get_nowait()
                events.append(event)
        except asyncio.QueueEmpty:
            pass
        return events

    @property
    def is_connected(self) -> bool:
        """Check if connected to Kafka."""
        return self._started and not self._use_fallback

    @property
    def fallback_queue_size(self) -> int:
        """Get the size of the fallback queue."""
        return self._fallback_queue.qsize()


# Helper functions for publishing specific event types

async def publish_order_event(
    event_type: EventType,
    order_id: str,
    product_id: str,
    quantity: int,
    total_amount: float,
    customer_id: Optional[str] = None,
    **extra_data
):
    """Convenience function to publish order events."""
    from .events import OrderEvent

    producer = await get_producer()
    event = OrderEvent.create(
        event_type=event_type,
        order_id=order_id,
        product_id=product_id,
        quantity=quantity,
        total_amount=total_amount,
        customer_id=customer_id,
        **extra_data
    )
    return await producer.publish(event)


async def publish_stock_event(
    event_type: EventType,
    product_id: str,
    product_name: str,
    previous_stock: int,
    current_stock: int,
    reorder_threshold: int,
    **extra_data
):
    """Convenience function to publish stock events."""
    from .events import StockEvent

    producer = await get_producer()
    event = StockEvent.create(
        event_type=event_type,
        product_id=product_id,
        product_name=product_name,
        previous_stock=previous_stock,
        current_stock=current_stock,
        reorder_threshold=reorder_threshold,
        **extra_data
    )
    return await producer.publish(event)


async def publish_alert_event(
    event_type: EventType,
    alert_level: str,
    title: str,
    message: str,
    affected_products: Optional[list] = None,
    **extra_data
):
    """Convenience function to publish alert events."""
    from .events import AlertEvent

    producer = await get_producer()
    event = AlertEvent.create(
        event_type=event_type,
        alert_level=alert_level,
        title=title,
        message=message,
        affected_products=affected_products,
        **extra_data
    )
    return await producer.publish(event)


async def publish_forecast_event(
    category: str,
    forecast_horizon_days: int,
    predicted_demand: float,
    confidence_lower: float,
    confidence_upper: float,
    **extra_data
):
    """Convenience function to publish forecast events."""
    from .events import ForecastEvent

    producer = await get_producer()
    event = ForecastEvent.create(
        event_type=EventType.FORECAST_GENERATED,
        category=category,
        forecast_horizon_days=forecast_horizon_days,
        predicted_demand=predicted_demand,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        **extra_data
    )
    return await producer.publish(event)


async def get_producer() -> KafkaEventProducer:
    """Get or create the global producer instance."""
    global _producer
    if _producer is None:
        _producer = KafkaEventProducer()
        await _producer.start()
    return _producer


async def shutdown_producer():
    """Shutdown the global producer."""
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
