# Streaming module for real-time updates
from .events import EventType, InventoryEvent
from .producer import KafkaEventProducer, get_producer
from .consumer import KafkaEventConsumer
from .websocket_manager import WebSocketManager, websocket_manager

__all__ = [
    "EventType",
    "InventoryEvent",
    "KafkaEventProducer",
    "get_producer",
    "KafkaEventConsumer",
    "WebSocketManager",
    "websocket_manager",
]
