from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Config(BaseSettings):
    database_url: str = "sqlite:///./inventory.db"
    secret_key: str = "your-secret-key-here-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Data loading settings
    csv_chunk_size: int = 1000
    dead_stock_days: int = 90  # Days without sales to consider dead stock

    # Kafka Configuration
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_orders: str = "inventory.orders"
    kafka_topic_stock: str = "inventory.stock"
    kafka_topic_alerts: str = "inventory.alerts"
    kafka_topic_forecasts: str = "inventory.forecasts"
    kafka_consumer_group: str = "inventory-manager"
    kafka_enabled: bool = False  # Set to True when Kafka is available

    # Redis Configuration (for WebSocket pub/sub fallback)
    redis_url: str = "redis://localhost:6379"
    redis_enabled: bool = False  # Set to True when Redis is available

    # Streaming Settings
    stream_batch_size: int = 100
    stream_flush_interval: float = 1.0  # seconds

    model_config = SettingsConfigDict(env_file=".env")


config = Config()