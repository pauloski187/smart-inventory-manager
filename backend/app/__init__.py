# Smart Inventory Manager Backend

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from .routes import (
    products_router, orders_router, customers_router,
    sellers_router, analytics_router, auth_router, forecast_router
)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Inventory Manager API",
    version="2.0.0",
    description="""
    A comprehensive inventory management system with SARIMA-based demand forecasting.

    ## Features
    - **Product Management**: CRUD operations for products
    - **Order Management**: Order tracking and history
    - **Demand Forecasting**: SARIMA-based 30/60/90 day forecasts by category
    - **Inventory Analytics**: ABC analysis, dead stock detection, low stock alerts
    - **Inventory Recommendations**: Reorder points, safety stock, stockout risk

    ## Forecasting Endpoints
    - `POST /forecast/upload-data`: Upload sales data CSV
    - `GET /forecast/forecast/{category}`: Get category forecast
    - `GET /forecast/forecasts/all`: Get all category forecasts
    - `GET /forecast/inventory-recommendations`: Get reorder recommendations
    - `POST /forecast/retrain-models`: Retrain SARIMA models
    """,
    docs_url="/docs",
    redoc_url="/redoc"
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


@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Welcome to Smart Inventory Manager API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "forecast": "/forecast",
            "analytics": "/analytics",
            "products": "/products",
            "orders": "/orders"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "smart-inventory-manager"}