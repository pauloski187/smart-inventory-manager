# Smart Inventory Manager Backend

from fastapi import FastAPI
from .config.settings import settings
from .routes import products, suppliers, sales, auth, reports

app = FastAPI(title="Smart Inventory Manager API", version="1.0.0")

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(products.router, prefix="/products", tags=["Products"])
app.include_router(suppliers.router, prefix="/suppliers", tags=["Suppliers"])
app.include_router(sales.router, prefix="/sales", tags=["Sales"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Smart Inventory Manager API"}