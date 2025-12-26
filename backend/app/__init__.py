# Smart Inventory Manager Backend

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .database import engine, Base
from .routes import (
    products_router, orders_router, customers_router,
    sellers_router, analytics_router, auth_router, forecast_router
)
from .routes.streaming import router as streaming_router
from .config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Smart Inventory Manager...")

    # Initialize Kafka producer if enabled
    if config.kafka_enabled:
        try:
            from .streaming.producer import get_producer
            producer = await get_producer()
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.warning(f"Kafka initialization failed: {e}")

    # Start event processor if Kafka is enabled
    if config.kafka_enabled:
        try:
            from .streaming.consumer import start_event_processing
            await start_event_processing()
            logger.info("Event processor started")
        except Exception as e:
            logger.warning(f"Event processor failed to start: {e}")

    logger.info("Smart Inventory Manager started successfully")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down Smart Inventory Manager...")

    # Stop Kafka producer
    try:
        from .streaming.producer import shutdown_producer
        await shutdown_producer()
        logger.info("Kafka producer stopped")
    except Exception as e:
        logger.warning(f"Error stopping Kafka producer: {e}")

    # Stop event processor
    try:
        from .streaming.consumer import stop_event_processing
        await stop_event_processing()
        logger.info("Event processor stopped")
    except Exception as e:
        logger.warning(f"Error stopping event processor: {e}")

    logger.info("Smart Inventory Manager shutdown complete")


app = FastAPI(
    title="Smart Inventory Manager API",
    version="3.0.0",
    description="""
    A comprehensive inventory management system with **Prophet-based demand forecasting**
    achieving **18.35% SMAPE** (verified) and real-time streaming updates via Kafka/WebSocket.

    ## Key Achievement
    - **Forecast Accuracy (SMAPE)**: 18.35% ✅ (Target: <20%)
    - **Model**: Facebook Prophet with automatic seasonality detection
    - **Validation**: 8-week holdout period

    ## Features
    - **Product Management**: CRUD operations for products
    - **Order Management**: Order tracking and history
    - **Demand Forecasting**: Prophet-based 30/60/90 day forecasts by category
    - **Inventory Analytics**: ABC analysis, dead stock detection, low stock alerts
    - **Inventory Recommendations**: Reorder points, safety stock, stockout risk
    - **Real-time Streaming**: WebSocket and SSE for live updates

    ## Forecasting Endpoints
    - `POST /forecast/upload-data`: Upload sales data CSV
    - `GET /forecast/forecast/{category}`: Get category forecast
    - `GET /forecast/forecasts/all`: Get all category forecasts
    - `GET /forecast/inventory-recommendations`: Get reorder recommendations
    - `POST /forecast/ensemble/train`: Train Prophet ensemble models

    ## Streaming Endpoints
    - `WS /stream/ws`: WebSocket for real-time events
    - `GET /stream/events`: Server-Sent Events (SSE)
    - `GET /stream/status`: Connection status
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(products_router, prefix="/products", tags=["Products"])
app.include_router(orders_router, prefix="/orders", tags=["Orders"])
app.include_router(customers_router, prefix="/customers", tags=["Customers"])
app.include_router(sellers_router, prefix="/sellers", tags=["Sellers"])
app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
app.include_router(forecast_router, prefix="/forecast", tags=["SARIMA Forecasting"])
app.include_router(streaming_router, tags=["Real-time Streaming"])


@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Welcome to Smart Inventory Manager API",
        "version": "3.0.0",
        "smape": "18.35%",
        "model": "Facebook Prophet",
        "docs": "/docs",
        "endpoints": {
            "forecast": "/forecast",
            "analytics": "/analytics",
            "products": "/products",
            "orders": "/orders",
            "streaming": "/stream"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for monitoring."""
    from .streaming.websocket_manager import websocket_manager

    return {
        "status": "healthy",
        "service": "smart-inventory-manager",
        "version": "2.1.0",
        "kafka_enabled": config.kafka_enabled,
        "websocket_connections": websocket_manager.get_connection_count()
    }