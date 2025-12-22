# Smart Inventory Manager Backend

from fastapi import FastAPI
from .database import engine, Base
from .routes import products_router, orders_router, customers_router, sellers_router, analytics_router, auth_router

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Inventory Manager API",
    version="1.0.0",
    description="A comprehensive inventory management system with analytics and forecasting"
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(products_router, prefix="/products", tags=["Products"])
app.include_router(orders_router, prefix="/orders", tags=["Orders"])
app.include_router(customers_router, prefix="/customers", tags=["Customers"])
app.include_router(sellers_router, prefix="/sellers", tags=["Sellers"])
app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Smart Inventory Manager API"}