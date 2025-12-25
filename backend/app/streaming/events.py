"""
Event models for streaming/Kafka integration.
Defines the structure of events published to Kafka topics.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import json
import uuid


class EventType(str, Enum):
    # Order Events
    ORDER_CREATED = "order.created"
    ORDER_UPDATED = "order.updated"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_DELIVERED = "order.delivered"

    # Stock Events
    STOCK_UPDATED = "stock.updated"
    STOCK_LOW = "stock.low"
    STOCK_OUT = "stock.out"
    STOCK_REPLENISHED = "stock.replenished"

    # Alert Events
    ALERT_LOW_STOCK = "alert.low_stock"
    ALERT_DEAD_STOCK = "alert.dead_stock"
    ALERT_REORDER_NEEDED = "alert.reorder_needed"
    ALERT_STOCKOUT_RISK = "alert.stockout_risk"

    # Forecast Events
    FORECAST_GENERATED = "forecast.generated"
    FORECAST_UPDATED = "forecast.updated"
    MODEL_RETRAINED = "model.retrained"

    # Analytics Events
    DAILY_SUMMARY = "analytics.daily_summary"
    WEEKLY_SUMMARY = "analytics.weekly_summary"
    MONTHLY_REPORT = "analytics.monthly_report"


class InventoryEvent(BaseModel):
    """Base event model for all streaming events."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "inventory-manager"
    version: str = "1.0"

    # Event payload
    data: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return self.model_dump_json()

    def to_bytes(self) -> bytes:
        """Serialize event to bytes for Kafka."""
        return self.to_json().encode('utf-8')

    @classmethod
    def from_json(cls, json_str: str) -> "InventoryEvent":
        """Deserialize event from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_bytes(cls, data: bytes) -> "InventoryEvent":
        """Deserialize event from bytes."""
        return cls.from_json(data.decode('utf-8'))


# Specialized Event Classes

class OrderEvent(InventoryEvent):
    """Event for order-related updates."""

    @classmethod
    def create(
        cls,
        event_type: EventType,
        order_id: str,
        product_id: str,
        quantity: int,
        total_amount: float,
        customer_id: Optional[str] = None,
        **extra_data
    ) -> "OrderEvent":
        return cls(
            event_type=event_type,
            data={
                "order_id": order_id,
                "product_id": product_id,
                "quantity": quantity,
                "total_amount": total_amount,
                "customer_id": customer_id,
                **extra_data
            }
        )


class StockEvent(InventoryEvent):
    """Event for stock-related updates."""

    @classmethod
    def create(
        cls,
        event_type: EventType,
        product_id: str,
        product_name: str,
        previous_stock: int,
        current_stock: int,
        reorder_threshold: int,
        **extra_data
    ) -> "StockEvent":
        return cls(
            event_type=event_type,
            data={
                "product_id": product_id,
                "product_name": product_name,
                "previous_stock": previous_stock,
                "current_stock": current_stock,
                "reorder_threshold": reorder_threshold,
                "stock_change": current_stock - previous_stock,
                **extra_data
            }
        )


class AlertEvent(InventoryEvent):
    """Event for alert notifications."""

    @classmethod
    def create(
        cls,
        event_type: EventType,
        alert_level: str,  # 'info', 'warning', 'critical'
        title: str,
        message: str,
        affected_products: Optional[List[Dict]] = None,
        **extra_data
    ) -> "AlertEvent":
        return cls(
            event_type=event_type,
            data={
                "alert_level": alert_level,
                "title": title,
                "message": message,
                "affected_products": affected_products or [],
                **extra_data
            }
        )


class ForecastEvent(InventoryEvent):
    """Event for forecast updates."""

    @classmethod
    def create(
        cls,
        event_type: EventType,
        category: str,
        forecast_horizon_days: int,
        predicted_demand: float,
        confidence_lower: float,
        confidence_upper: float,
        **extra_data
    ) -> "ForecastEvent":
        return cls(
            event_type=event_type,
            data={
                "category": category,
                "forecast_horizon_days": forecast_horizon_days,
                "predicted_demand": predicted_demand,
                "confidence_interval": {
                    "lower": confidence_lower,
                    "upper": confidence_upper
                },
                **extra_data
            }
        )


class AnalyticsEvent(InventoryEvent):
    """Event for analytics summaries."""

    @classmethod
    def create(
        cls,
        event_type: EventType,
        period: str,
        period_start: datetime,
        period_end: datetime,
        metrics: Dict[str, Any],
        **extra_data
    ) -> "AnalyticsEvent":
        return cls(
            event_type=event_type,
            data={
                "period": period,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "metrics": metrics,
                **extra_data
            }
        )
