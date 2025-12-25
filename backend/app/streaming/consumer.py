"""
Kafka Consumer for processing inventory events.
Handles event consumption and dispatches to appropriate handlers.
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime

from ..config.config import config
from .events import InventoryEvent, EventType

logger = logging.getLogger(__name__)


class KafkaEventConsumer:
    """
    Async Kafka consumer for processing inventory events.
    Supports multiple topics and event handlers.
    """

    def __init__(self, topics: Optional[List[str]] = None, group_id: Optional[str] = None):
        self.topics = topics or [
            config.kafka_topic_orders,
            config.kafka_topic_stock,
            config.kafka_topic_alerts,
            config.kafka_topic_forecasts,
        ]
        self.group_id = group_id or config.kafka_consumer_group
        self._consumer = None
        self._started = False
        self._handlers: Dict[EventType, List[Callable]] = {}
        self._default_handlers: List[Callable] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Initialize the Kafka consumer connection."""
        if self._started:
            return

        if not config.kafka_enabled:
            logger.info("Kafka disabled. Consumer not starting.")
            return

        try:
            from aiokafka import AIOKafkaConsumer

            self._consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=config.kafka_bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda v: InventoryEvent.from_bytes(v),
                auto_offset_reset='latest',
                enable_auto_commit=True,
            )
            await self._consumer.start()
            self._started = True
            logger.info(f"Kafka consumer connected, subscribed to: {self.topics}")
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise

    async def stop(self):
        """Stop the consumer and close connections."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

        self._started = False
        logger.info("Kafka consumer stopped")

    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[[InventoryEvent], Any]
    ):
        """
        Register a handler for a specific event type.

        Args:
            event_type: The event type to handle
            handler: Async function to process the event
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")

    def register_default_handler(self, handler: Callable[[InventoryEvent], Any]):
        """Register a handler that receives all events."""
        self._default_handlers.append(handler)

    async def _dispatch_event(self, event: InventoryEvent):
        """Dispatch event to registered handlers."""
        handlers = self._handlers.get(event.event_type, []) + self._default_handlers

        if not handlers:
            logger.debug(f"No handlers for event type: {event.event_type}")
            return

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")

    async def consume(self):
        """Start consuming messages in a loop."""
        if not self._started:
            await self.start()

        if not self._consumer:
            logger.warning("Consumer not available (Kafka disabled?)")
            return

        self._running = True
        logger.info("Starting message consumption loop")

        try:
            async for message in self._consumer:
                if not self._running:
                    break

                try:
                    event = message.value
                    logger.debug(
                        f"Received event: {event.event_type} "
                        f"from {message.topic}:{message.partition}:{message.offset}"
                    )
                    await self._dispatch_event(event)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled")
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
            raise

    def start_consuming(self) -> asyncio.Task:
        """Start consuming in a background task."""
        if self._task:
            return self._task
        self._task = asyncio.create_task(self.consume())
        return self._task


class EventProcessor:
    """
    High-level event processor that coordinates consumers and handlers.
    Provides built-in handlers for common inventory operations.
    """

    def __init__(self):
        self.consumer = KafkaEventConsumer()
        self._db_session_factory = None

    def set_db_session_factory(self, factory):
        """Set the database session factory for handlers that need DB access."""
        self._db_session_factory = factory

    async def start(self):
        """Start the event processor."""
        # Register built-in handlers
        self._register_builtin_handlers()
        await self.consumer.start()
        self.consumer.start_consuming()
        logger.info("Event processor started")

    async def stop(self):
        """Stop the event processor."""
        await self.consumer.stop()
        logger.info("Event processor stopped")

    def _register_builtin_handlers(self):
        """Register built-in event handlers."""
        # Stock alert handlers
        self.consumer.register_handler(
            EventType.STOCK_LOW,
            self._handle_low_stock
        )
        self.consumer.register_handler(
            EventType.STOCK_OUT,
            self._handle_stockout
        )

        # Order handlers
        self.consumer.register_handler(
            EventType.ORDER_CREATED,
            self._handle_order_created
        )

        # Log all events
        self.consumer.register_default_handler(self._log_event)

    async def _log_event(self, event: InventoryEvent):
        """Default handler that logs all events."""
        logger.info(
            f"Event processed: {event.event_type} | "
            f"ID: {event.event_id} | "
            f"Data: {event.data}"
        )

    async def _handle_low_stock(self, event: InventoryEvent):
        """Handle low stock events - could trigger notifications."""
        product_id = event.data.get("product_id")
        product_name = event.data.get("product_name")
        current_stock = event.data.get("current_stock")
        threshold = event.data.get("reorder_threshold")

        logger.warning(
            f"LOW STOCK ALERT: {product_name} ({product_id}) "
            f"has {current_stock} units (threshold: {threshold})"
        )
        # Could integrate with notification service here

    async def _handle_stockout(self, event: InventoryEvent):
        """Handle stockout events - critical alert."""
        product_id = event.data.get("product_id")
        product_name = event.data.get("product_name")

        logger.critical(
            f"STOCKOUT: {product_name} ({product_id}) is out of stock!"
        )
        # Could trigger emergency reorder or notification

    async def _handle_order_created(self, event: InventoryEvent):
        """Handle new order events - update metrics."""
        order_id = event.data.get("order_id")
        total = event.data.get("total_amount")

        logger.info(f"New order {order_id}: ${total:.2f}")
        # Could update real-time dashboard metrics


# Singleton processor instance
_processor: Optional[EventProcessor] = None


async def get_processor() -> EventProcessor:
    """Get or create the global event processor."""
    global _processor
    if _processor is None:
        _processor = EventProcessor()
    return _processor


async def start_event_processing():
    """Start the global event processor."""
    processor = await get_processor()
    await processor.start()


async def stop_event_processing():
    """Stop the global event processor."""
    global _processor
    if _processor:
        await _processor.stop()
        _processor = None
