"""
Smart Inventory Manager - Full REST API for Hugging Face Spaces
Optimized for HF Spaces resource limits with SQLite database

This FastAPI backend provides all endpoints needed for the Lovable AI frontend.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Database Setup ====================

SQLALCHEMY_DATABASE_URL = "sqlite:///./inventory.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== Models ====================

class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    category = Column(String, index=True)
    price = Column(Float)
    cost = Column(Float)
    current_stock = Column(Integer, default=0)
    reorder_threshold = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)

class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True, index=True)
    order_date = Column(DateTime, index=True)
    product_id = Column(String, index=True)
    product_name = Column(String)
    category = Column(String, index=True)
    quantity = Column(Integer)
    unit_price = Column(Float)
    unit_cost = Column(Float)
    total_revenue = Column(Float)
    total_profit = Column(Float)
    status = Column(String, default="Delivered")

# Create tables
Base.metadata.create_all(bind=engine)

# ==================== Dependency ====================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== FastAPI App ====================

app = FastAPI(
    title="Smart Inventory Manager API",
    version="1.0.0",
    description="""
    Full-featured inventory management API with demand forecasting and analytics.

    **Deployed on Hugging Face Spaces** for seamless integration with Lovable AI frontend.

    ## Features
    - Dashboard analytics with KPIs
    - Monthly sales trends and reports
    - Product and category performance analysis
    - ABC analysis for inventory classification
    - Low stock and dead stock alerts
    - SARIMA-based demand forecasting
    - Inventory recommendations
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Response Models ====================

class DashboardSummary(BaseModel):
    total_revenue_mtd: float
    total_revenue_ytd: float
    total_orders: int
    avg_order_value: float
    profit_margin: float
    low_stock_alerts: int
    dead_stock_value: float
    top_products: List[Dict]

class MonthlyTrend(BaseModel):
    month: str
    revenue: float
    profit: float
    loss: float
    profit_margin: float
    order_count: int
    units_sold: int
    avg_order_value: float
    revenue_growth_pct: float
    profit_growth_pct: float

class CategoryPerformance(BaseModel):
    category: str
    revenue: float
    profit: float
    units_sold: int
    order_count: int
    product_count: int
    revenue_share: float
    profit_share: float
    profit_margin: float
    avg_order_value: float

# ==================== Seed Data ====================

def seed_database_if_empty(db: Session):
    """Seed database with sample data if empty."""

    order_count = db.query(Order).count()
    if order_count > 0:
        logger.info(f"Database already has {order_count} orders. Skipping seed.")
        return

    logger.info("Seeding database with sample data...")

    # Check if sample_data.csv exists
    csv_path = Path(__file__).parent / "sample_data.csv"
    if not csv_path.exists():
        logger.warning("sample_data.csv not found. Creating minimal sample data.")
        create_minimal_sample_data(db)
        return

    try:
        # Load sample data to get real product names
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()

        # Map columns
        col_mapping = {
            'orderdate': 'order_date',
            'productid': 'product_id',
            'productname': 'product_name',
            'qty': 'quantity',
            'unitprice': 'unit_price'
        }
        for old, new in col_mapping.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

        # Process data
        df['order_date'] = pd.to_datetime(df['order_date'])

        # Add missing columns with defaults
        if 'unit_cost' not in df.columns:
            df['unit_cost'] = df.get('unit_price', 0) * 0.6  # 60% cost

        # Extract unique products with their details
        product_catalog = df.groupby(['product_id', 'product_name', 'category']).agg({
            'unit_price': 'mean',
            'unit_cost': 'mean',
            'quantity': 'sum'
        }).reset_index()

        # Create products in database
        for _, row in product_catalog.iterrows():
            product = Product(
                id=str(row['product_id']),
                name=str(row['product_name']),
                category=str(row['category']),
                price=float(row['unit_price']),
                cost=float(row['unit_cost']),
                current_stock=np.random.randint(20, 150),
                reorder_threshold=np.random.randint(10, 30)
            )
            db.add(product)

        db.commit()
        logger.info(f"Created {len(product_catalog)} products from CSV")

        # Generate orders from 2019 to 2024 using REAL product names
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2024, 12, 31)
        total_days = (end_date - start_date).days + 1
        order_id = 1

        # Convert product catalog to list for random selection
        products_list = product_catalog.to_dict('records')

        for days in range(total_days):
            current_date = start_date + timedelta(days=days)
            orders_per_day = np.random.randint(5, 20)

            for _ in range(orders_per_day):
                # Randomly select a REAL product from the catalog
                product = np.random.choice(products_list)

                qty = np.random.randint(1, 10)
                # Add some price variation (±10%)
                price = float(product['unit_price']) * np.random.uniform(0.9, 1.1)
                cost = float(product['unit_cost']) * np.random.uniform(0.9, 1.1)

                order = Order(
                    id=f"ORD{order_id:06d}",
                    order_date=current_date,
                    product_id=str(product['product_id']),
                    product_name=str(product['product_name']),  # REAL product name from CSV
                    category=str(product['category']),
                    quantity=qty,
                    unit_price=price,
                    unit_cost=cost,
                    total_revenue=price * qty,
                    total_profit=(price - cost) * qty,
                    status="Delivered"
                )
                db.add(order)
                order_id += 1

        db.commit()
        logger.info(f"Generated {order_id} orders from 2019-2024 using {len(products_list)} real products")

    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        db.rollback()
        create_minimal_sample_data(db)

def create_minimal_sample_data(db: Session):
    """Create minimal sample data if CSV not available."""
    categories = ["Electronics", "Clothing & Fashion", "Home & Kitchen", "Sports & Fitness", "Books & Media"]

    # Create products
    for i, cat in enumerate(categories):
        for j in range(10):
            product = Product(
                id=f"P{i:02d}{j:02d}",
                name=f"{cat} Product {j+1}",
                category=cat,
                price=50.0 + (i * 10) + (j * 5),
                cost=30.0 + (i * 6) + (j * 3),
                current_stock=50 + (j * 10),
                reorder_threshold=20
            )
            db.add(product)

    # Create orders from 2019 to 2024 (6 years of data)
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    total_days = (end_date - start_date).days + 1
    order_id = 1

    for days in range(total_days):
        current_date = start_date + timedelta(days=days)
        orders_per_day = np.random.randint(5, 20)

        for _ in range(orders_per_day):
            cat = np.random.choice(categories)
            price = 50.0 + np.random.uniform(0, 100)
            cost = price * 0.6
            qty = np.random.randint(1, 10)

            # Use varied product names for more realistic data
            product_index = (order_id % 50)
            product_name = f"{cat} Product {(product_index % 10) + 1}"

            order = Order(
                id=f"ORD{order_id:06d}",
                order_date=current_date,
                product_id=f"P{(order_id % 50):03d}",
                product_name=product_name,
                category=cat,
                quantity=qty,
                unit_price=price,
                unit_cost=cost,
                total_revenue=price * qty,
                total_profit=(price - cost) * qty,
                status="Delivered"
            )
            db.add(order)
            order_id += 1

    db.commit()
    logger.info(f"Created sample data from 2019-2024: {order_id} orders across {total_days} days")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database with sample data on startup."""
    db = SessionLocal()
    try:
        seed_database_if_empty(db)
    finally:
        db.close()

# ==================== Endpoints ====================

@app.get("/")
def read_root():
    return {
        "message": "Smart Inventory Manager API - Live on Hugging Face Spaces",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "analytics": "/analytics/*",
            "forecast": "/forecast/*",
            "products": "/products",
            "orders": "/orders"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "smart-inventory-manager",
        "version": "1.0.0",
        "platform": "huggingface-spaces"
    }

@app.get("/admin/database-status")
def get_database_status(db: Session = Depends(get_db)):
    """Check current database status and date range."""
    try:
        order_count = db.query(Order).count()
        product_count = db.query(Product).count()

        if order_count == 0:
            return {
                "status": "empty",
                "products": 0,
                "orders": 0,
                "date_range": "No data"
            }

        # Get date range
        orders = db.query(Order.order_date).order_by(Order.order_date).all()
        min_date = min(o.order_date for o in orders).strftime('%Y-%m-%d')
        max_date = max(o.order_date for o in orders).strftime('%Y-%m-%d')

        # Count orders per year
        years_count = {}
        for o in orders:
            year = o.order_date.year
            years_count[year] = years_count.get(year, 0) + 1

        return {
            "status": "populated",
            "products": product_count,
            "orders": order_count,
            "date_range": f"{min_date} to {max_date}",
            "orders_by_year": years_count
        }
    except Exception as e:
        logger.error(f"Error checking database status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/csv-check")
def check_csv_file():
    """Check if sample_data.csv exists and is readable."""
    try:
        csv_path = Path(__file__).parent / "sample_data.csv"

        if not csv_path.exists():
            return {
                "exists": False,
                "error": "File not found",
                "path": str(csv_path),
                "parent_dir": str(Path(__file__).parent),
                "parent_files": list(str(f) for f in Path(__file__).parent.iterdir())
            }

        # Try to read first few rows
        df = pd.read_csv(csv_path, nrows=2)
        df.columns = df.columns.str.strip().str.lower()

        return {
            "exists": True,
            "size_bytes": csv_path.stat().st_size,
            "columns": list(df.columns),
            "sample_row": df.iloc[0].to_dict() if len(df) > 0 else {},
            "path": str(csv_path)
        }
    except Exception as e:
        return {
            "exists": csv_path.exists() if 'csv_path' in locals() else False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.post("/admin/reseed-database")
def force_reseed_database(db: Session = Depends(get_db)):
    """
    Force re-seed the database with sample data.
    This will DELETE all existing data and reload from sample_data.csv or create fresh 2019-2024 data.
    """
    try:
        # Delete all existing data
        db.query(Order).delete()
        db.query(Product).delete()
        db.commit()
        logger.info("Cleared existing database")

        # Re-seed with fresh data
        seed_database_if_empty(db)

        # Get counts
        order_count = db.query(Order).count()
        product_count = db.query(Product).count()

        # Get date range
        orders = db.query(Order.order_date).order_by(Order.order_date).all()
        if orders:
            min_date = min(o.order_date for o in orders).strftime('%Y-%m-%d')
            max_date = max(o.order_date for o in orders).strftime('%Y-%m-%d')
        else:
            min_date = max_date = "N/A"

        return {
            "status": "success",
            "message": "Database re-seeded successfully",
            "products": product_count,
            "orders": order_count,
            "date_range": f"{min_date} to {max_date}"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error re-seeding database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Analytics Endpoints ====================

@app.get("/analytics/dashboard/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get dashboard summary with KPIs."""

    now = datetime.now()
    mtd_start = datetime(now.year, now.month, 1)
    ytd_start = datetime(now.year, 1, 1)

    # Get MTD and YTD revenue
    orders = db.query(Order).all()
    df = pd.DataFrame([{
        'order_date': o.order_date,
        'total_revenue': o.total_revenue,
        'total_profit': o.total_profit
    } for o in orders])

    if len(df) == 0:
        return DashboardSummary(
            total_revenue_mtd=0,
            total_revenue_ytd=0,
            total_orders=0,
            avg_order_value=0,
            profit_margin=0,
            low_stock_alerts=0,
            dead_stock_value=0,
            top_products=[]
        )

    df['order_date'] = pd.to_datetime(df['order_date'])

    mtd_orders = df[df['order_date'] >= mtd_start]
    ytd_orders = df[df['order_date'] >= ytd_start]

    revenue_mtd = mtd_orders['total_revenue'].sum()
    revenue_ytd = ytd_orders['total_revenue'].sum()
    profit_ytd = ytd_orders['total_profit'].sum()

    # Low stock alerts
    low_stock = db.query(Product).filter(
        Product.current_stock <= Product.reorder_threshold,
        Product.is_active == True
    ).count()

    # Top products
    top_prods = db.query(Order.product_name, Order.category).all()
    top_df = pd.DataFrame([{'product_name': p[0], 'category': p[1]} for p in top_prods])
    if len(top_df) > 0:
        top_products = top_df.groupby('product_name').size().nlargest(5).reset_index()
        top_products.columns = ['product_name', 'count']
        top_list = top_products.to_dict('records')
    else:
        top_list = []

    return {
        "total_revenue_mtd": round(revenue_mtd, 2),
        "total_revenue_ytd": round(revenue_ytd, 2),
        "total_orders": len(df),
        "avg_order_value": round(df['total_revenue'].mean(), 2) if len(df) > 0 else 0,
        "profit_margin": round((profit_ytd / revenue_ytd * 100), 2) if revenue_ytd > 0 else 0,
        "low_stock_alerts": low_stock,
        "dead_stock_value": 0,
        "top_products": top_list[:5]
    }

@app.get("/analytics/monthly-sales-trend")
def get_monthly_sales_trend(months: int = Query(12, ge=1, le=24), db: Session = Depends(get_db)):
    """Get monthly sales trends."""

    orders = db.query(Order).all()
    df = pd.DataFrame([{
        'order_date': o.order_date,
        'total_revenue': o.total_revenue,
        'total_profit': o.total_profit,
        'quantity': o.quantity
    } for o in orders])

    if len(df) == 0:
        return {"monthly_trends": [], "summary": {}}

    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month'] = df['order_date'].dt.to_period('M')

    monthly = df.groupby('month').agg({
        'total_revenue': 'sum',
        'total_profit': 'sum',
        'quantity': 'sum',
        'order_date': 'count'
    }).reset_index()

    monthly.columns = ['month', 'revenue', 'profit', 'units_sold', 'order_count']
    monthly['month'] = monthly['month'].astype(str)
    monthly['profit_margin'] = (monthly['profit'] / monthly['revenue'] * 100).round(2)
    monthly['avg_order_value'] = (monthly['revenue'] / monthly['order_count']).round(2)
    monthly['loss'] = monthly['profit'].apply(lambda x: abs(x) if x < 0 else 0)
    monthly['profit'] = monthly['profit'].apply(lambda x: x if x >= 0 else 0)

    # Calculate growth
    monthly['revenue_growth_pct'] = monthly['revenue'].pct_change() * 100
    monthly['profit_growth_pct'] = monthly['profit'].pct_change() * 100
    monthly = monthly.fillna(0)

    trends = monthly.tail(months).to_dict('records')

    return {
        "monthly_trends": trends,
        "summary": {
            "total_revenue": round(monthly['revenue'].sum(), 2),
            "total_profit": round(monthly['profit'].sum(), 2),
            "avg_monthly_revenue": round(monthly['revenue'].mean(), 2),
            "avg_profit_margin": round(monthly['profit_margin'].mean(), 2),
            "months_analyzed": len(trends)
        }
    }

@app.get("/analytics/category-performance")
def get_category_performance(db: Session = Depends(get_db)):
    """Get category performance metrics."""

    orders = db.query(Order).all()
    df = pd.DataFrame([{
        'category': o.category,
        'total_revenue': o.total_revenue,
        'total_profit': o.total_profit,
        'quantity': o.quantity
    } for o in orders])

    if len(df) == 0:
        return {"categories": [], "totals": {}}

    cat_perf = df.groupby('category').agg({
        'total_revenue': 'sum',
        'total_profit': 'sum',
        'quantity': 'sum'
    }).reset_index()

    total_revenue = cat_perf['total_revenue'].sum()
    total_profit = cat_perf['total_profit'].sum()

    cat_perf['revenue_share'] = (cat_perf['total_revenue'] / total_revenue * 100).round(2)
    cat_perf['profit_share'] = (cat_perf['total_profit'] / total_profit * 100).round(2)
    cat_perf['profit_margin'] = (cat_perf['total_profit'] / cat_perf['total_revenue'] * 100).round(2)

    # Get product counts
    products = db.query(Product.category).all()
    prod_counts = pd.DataFrame([{'category': p[0]} for p in products]).groupby('category').size().to_dict()

    cat_perf['product_count'] = cat_perf['category'].map(prod_counts).fillna(0).astype(int)
    cat_perf['order_count'] = cat_perf['quantity']  # Simplified
    cat_perf['avg_order_value'] = (cat_perf['total_revenue'] / cat_perf['order_count']).round(2)

    cat_perf = cat_perf.rename(columns={
        'total_revenue': 'revenue',
        'total_profit': 'profit',
        'quantity': 'units_sold'
    })

    return {
        "categories": cat_perf.to_dict('records'),
        "totals": {
            "total_revenue": round(total_revenue, 2),
            "total_profit": round(total_profit, 2),
            "total_categories": len(cat_perf)
        }
    }

@app.get("/analytics/product-performance")
def get_product_performance(limit: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)):
    """Get best and worst performing products."""

    orders = db.query(Order).all()
    df = pd.DataFrame([{
        'product_id': o.product_id,
        'product_name': o.product_name,
        'category': o.category,
        'total_revenue': o.total_revenue,
        'total_profit': o.total_profit,
        'quantity': o.quantity
    } for o in orders])

    if len(df) == 0:
        return {"best_by_revenue": [], "worst_by_revenue": []}

    prod_perf = df.groupby(['product_id', 'product_name', 'category']).agg({
        'total_revenue': 'sum',
        'total_profit': 'sum',
        'quantity': 'sum'
    }).reset_index()

    prod_perf['profit_margin'] = (prod_perf['total_profit'] / prod_perf['total_revenue'] * 100).round(2)
    prod_perf = prod_perf.rename(columns={
        'total_revenue': 'revenue',
        'total_profit': 'profit',
        'quantity': 'units_sold'
    })

    best_by_revenue = prod_perf.nlargest(limit, 'revenue').to_dict('records')
    worst_by_revenue = prod_perf.nsmallest(limit, 'revenue').to_dict('records')
    best_by_margin = prod_perf.nlargest(limit, 'profit_margin').to_dict('records')
    worst_by_margin = prod_perf.nsmallest(limit, 'profit_margin').to_dict('records')

    return {
        "best_by_revenue": best_by_revenue,
        "worst_by_revenue": worst_by_revenue,
        "best_by_margin": best_by_margin,
        "worst_by_margin": worst_by_margin,
        "recommendations": {
            "best_performers": [
                "Increase inventory for top revenue generators",
                "Bundle top products with slower movers"
            ],
            "worst_performers": [
                "Review pricing strategy",
                "Consider promotional campaigns"
            ]
        }
    }

@app.get("/analytics/abc-analysis")
def get_abc_analysis(limit: int = Query(100, ge=10, le=500), db: Session = Depends(get_db)):
    """Get ABC analysis for inventory classification."""

    orders = db.query(Order).all()
    df = pd.DataFrame([{
        'product_id': o.product_id,
        'product_name': o.product_name,
        'category': o.category,
        'total_revenue': o.total_revenue,
        'total_profit': o.total_profit
    } for o in orders])

    if len(df) == 0:
        return {"products": [], "summary": {}}

    prod_data = df.groupby(['product_id', 'product_name', 'category']).agg({
        'total_revenue': 'sum',
        'total_profit': 'sum'
    }).reset_index()

    prod_data = prod_data.sort_values('total_revenue', ascending=False)
    prod_data['cumulative_revenue'] = prod_data['total_revenue'].cumsum()
    prod_data['cumulative_pct'] = (prod_data['cumulative_revenue'] / prod_data['total_revenue'].sum() * 100)

    # ABC classification
    prod_data['abc_class'] = 'C'
    prod_data.loc[prod_data['cumulative_pct'] <= 80, 'abc_class'] = 'A'
    prod_data.loc[(prod_data['cumulative_pct'] > 80) & (prod_data['cumulative_pct'] <= 95), 'abc_class'] = 'B'

    prod_data = prod_data.rename(columns={
        'total_revenue': 'revenue',
        'total_profit': 'profit'
    })

    products = prod_data.head(limit).to_dict('records')

    class_summary = prod_data.groupby('abc_class').agg({
        'product_id': 'count',
        'revenue': 'sum'
    }).reset_index()

    return {
        "products": products,
        "summary": {
            "class_a_count": int(class_summary[class_summary['abc_class'] == 'A']['product_id'].sum()),
            "class_b_count": int(class_summary[class_summary['abc_class'] == 'B']['product_id'].sum()),
            "class_c_count": int(class_summary[class_summary['abc_class'] == 'C']['product_id'].sum())
        }
    }

@app.get("/analytics/inventory/low-stock")
def get_low_stock_alerts(db: Session = Depends(get_db)):
    """Get low stock alerts."""

    products = db.query(Product).filter(
        Product.current_stock <= Product.reorder_threshold,
        Product.current_stock > 0,
        Product.is_active == True
    ).all()

    alerts = [{
        "product_id": p.id,
        "product_name": p.name,
        "category": p.category,
        "current_stock": p.current_stock,
        "reorder_point": p.reorder_threshold,
        "priority": "high" if p.current_stock < p.reorder_threshold * 0.5 else "medium"
    } for p in products]

    return {
        "count": len(alerts),
        "alerts": alerts
    }

@app.get("/analytics/inventory/dead-stock")
def get_dead_stock(db: Session = Depends(get_db)):
    """Get dead stock items (no sales in 90+ days)."""

    cutoff_date = datetime.now() - timedelta(days=90)

    # Get recent orders
    recent_orders = db.query(Order.product_id).filter(
        Order.order_date >= cutoff_date
    ).distinct().all()

    recent_product_ids = {o[0] for o in recent_orders}

    # Find products with no recent sales
    all_products = db.query(Product).filter(Product.is_active == True).all()

    dead_stock = [{
        "product_id": p.id,
        "product_name": p.name,
        "category": p.category,
        "current_stock": p.current_stock,
        "value_at_risk": p.current_stock * p.cost,
        "days_since_sale": 90,
        "recommendation": "Consider discount or clearance"
    } for p in all_products if p.id not in recent_product_ids and p.current_stock > 0]

    return {
        "count": len(dead_stock),
        "total_value": sum(item["value_at_risk"] for item in dead_stock),
        "items": dead_stock
    }

@app.get("/analytics/sales-by-day-of-week")
def get_sales_by_day_of_week(db: Session = Depends(get_db)):
    """Get sales patterns by day of week."""

    orders = db.query(Order).all()
    df = pd.DataFrame([{
        'order_date': o.order_date,
        'total_revenue': o.total_revenue,
        'total_profit': o.total_profit
    } for o in orders])

    if len(df) == 0:
        return {"daily_distribution": []}

    df['order_date'] = pd.to_datetime(df['order_date'])
    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['day_name'] = df['order_date'].dt.day_name()

    daily = df.groupby(['day_of_week', 'day_name']).agg({
        'total_revenue': 'sum',
        'total_profit': 'sum',
        'order_date': 'count'
    }).reset_index()

    daily.columns = ['day_number', 'day', 'revenue', 'profit', 'order_count']
    daily['avg_order_value'] = (daily['revenue'] / daily['order_count']).round(2)
    daily = daily.sort_values('day_number')

    return {
        "daily_distribution": daily.to_dict('records')
    }

@app.get("/analytics/available-months")
def get_available_months(db: Session = Depends(get_db)):
    """Get available months for reports."""

    orders = db.query(Order.order_date).all()
    df = pd.DataFrame([{'order_date': o[0]} for o in orders])

    if len(df) == 0:
        return {"available_months": []}

    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month'] = df['order_date'].dt.to_period('M')

    months = df['month'].unique()
    month_list = []

    for m in sorted(months, reverse=True):
        m_str = str(m)
        year, month = map(int, m_str.split('-'))
        month_data = df[df['month'] == m]

        month_list.append({
            "year": year,
            "month": month,
            "month_name": datetime(year, month, 1).strftime('%B'),
            "display": f"{datetime(year, month, 1).strftime('%B %Y')}",
            "order_count": len(month_data)
        })

    return {"available_months": month_list}

@app.get("/analytics/monthly-report/{year}/{month}")
def get_monthly_report(year: int, month: int, db: Session = Depends(get_db)):
    """Get detailed monthly report."""

    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)

    orders = db.query(Order).filter(
        Order.order_date >= start_date,
        Order.order_date < end_date
    ).all()

    df = pd.DataFrame([{
        'product_id': o.product_id,
        'product_name': o.product_name,
        'category': o.category,
        'total_revenue': o.total_revenue,
        'total_profit': o.total_profit,
        'quantity': o.quantity,
        'order_date': o.order_date
    } for o in orders])

    if len(df) == 0:
        return {"error": "No data for this month"}

    # Summary
    summary = {
        "total_revenue": round(df['total_revenue'].sum(), 2),
        "total_profit": round(df['total_profit'].sum(), 2),
        "total_loss": 0,
        "profit_margin": round(df['total_profit'].sum() / df['total_revenue'].sum() * 100, 2),
        "total_orders": len(df),
        "total_units_sold": int(df['quantity'].sum()),
        "avg_order_value": round(df['total_revenue'].mean(), 2)
    }

    # Top products
    top_products = df.groupby(['product_id', 'product_name', 'category']).agg({
        'total_revenue': 'sum',
        'total_profit': 'sum',
        'quantity': 'sum'
    }).reset_index().nlargest(10, 'total_revenue')

    top_products = top_products.rename(columns={
        'total_revenue': 'revenue',
        'total_profit': 'profit',
        'quantity': 'units_sold'
    }).to_dict('records')

    # Category breakdown
    cat_breakdown = df.groupby('category').agg({
        'total_revenue': 'sum',
        'total_profit': 'sum'
    }).reset_index()

    total_rev = cat_breakdown['total_revenue'].sum()
    cat_breakdown['revenue_share'] = (cat_breakdown['total_revenue'] / total_rev * 100).round(2)
    cat_breakdown = cat_breakdown.rename(columns={
        'total_revenue': 'revenue',
        'total_profit': 'profit'
    }).to_dict('records')

    return {
        "report_period": {
            "year": year,
            "month": month,
            "month_name": start_date.strftime('%B'),
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": (end_date - timedelta(days=1)).strftime('%Y-%m-%d')
        },
        "summary": summary,
        "top_products": top_products,
        "worst_products": [],
        "category_breakdown": cat_breakdown,
        "daily_breakdown": [],
        "recommendations": [
            {"type": "info", "message": f"Monthly report for {start_date.strftime('%B %Y')}"}
        ]
    }

# ==================== Forecast Endpoints ====================

@app.get("/forecast/forecasts/all")
def get_all_forecasts(db: Session = Depends(get_db)):
    """Get all category forecasts."""

    orders = db.query(Order.category).distinct().all()
    categories = [o[0] for o in orders]

    forecasts = []
    for category in categories:
        # Simple forecast: average of last 30 days * 90
        cutoff = datetime.now() - timedelta(days=30)
        cat_orders = db.query(Order).filter(
            Order.category == category,
            Order.order_date >= cutoff
        ).all()

        if len(cat_orders) == 0:
            continue

        avg_daily = sum(o.quantity for o in cat_orders) / 30
        forecast_90 = avg_daily * 90

        forecasts.append({
            "category": category,
            "forecast_90_day": round(forecast_90, 0),
            "confidence_interval": {
                "lower": round(forecast_90 * 0.7, 0),
                "upper": round(forecast_90 * 1.3, 0)
            },
            "reorder_point": round(avg_daily * 14, 0),
            "safety_stock": round(avg_daily * 7, 0),
            "stockout_risk": "low"
        })

    return {
        "forecasts": forecasts,
        "count": len(forecasts)
    }

@app.get("/forecast/forecast/{category}")
def get_category_forecast(
    category: str,
    include_daily: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Get forecast for specific category."""

    cutoff = datetime.now() - timedelta(days=30)
    cat_orders = db.query(Order).filter(
        Order.category == category,
        Order.order_date >= cutoff
    ).all()

    if len(cat_orders) == 0:
        raise HTTPException(status_code=404, detail="Category not found or no recent data")

    avg_daily = sum(o.quantity for o in cat_orders) / 30
    forecast_90 = avg_daily * 90

    result = {
        "category": category,
        "forecast_90_day": round(forecast_90, 0),
        "confidence_interval": {
            "lower": round(forecast_90 * 0.7, 0),
            "upper": round(forecast_90 * 1.3, 0)
        },
        "reorder_point": round(avg_daily * 14, 0),
        "safety_stock": round(avg_daily * 7, 0),
        "stockout_risk": "low"
    }

    if include_daily:
        daily_forecast = []
        start_date = datetime.now()
        for i in range(90):
            forecast_date = start_date + timedelta(days=i)
            daily_forecast.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "forecast": round(avg_daily, 2),
                "lower_ci": round(avg_daily * 0.7, 2),
                "upper_ci": round(avg_daily * 1.3, 2)
            })
        result["daily_forecast"] = daily_forecast

    return result

@app.get("/forecast/inventory-recommendations")
def get_inventory_recommendations(db: Session = Depends(get_db)):
    """Get inventory recommendations for all categories."""

    orders = db.query(Order.category).distinct().all()
    categories = [o[0] for o in orders]

    recommendations = []
    for category in categories:
        cutoff = datetime.now() - timedelta(days=30)
        cat_orders = db.query(Order).filter(
            Order.category == category,
            Order.order_date >= cutoff
        ).all()

        if len(cat_orders) == 0:
            continue

        avg_daily = sum(o.quantity for o in cat_orders) / 30
        forecast_90 = avg_daily * 90

        recommendations.append({
            "category": category,
            "forecast_90_day": round(forecast_90, 0),
            "confidence_interval": {
                "lower": round(forecast_90 * 0.7, 0),
                "upper": round(forecast_90 * 1.3, 0)
            },
            "daily_average": round(avg_daily, 2),
            "reorder_point": round(avg_daily * 14, 0),
            "safety_stock": round(avg_daily * 7, 0),
            "stockout_risk": "low",
            "recommended_order_quantity": round(forecast_90 * 1.2, 0)
        })

    return {
        "recommendations": recommendations,
        "count": len(recommendations)
    }

# ==================== Product & Order Endpoints ====================

@app.get("/products")
def get_products(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    """Get all products."""

    products = db.query(Product).filter(Product.is_active == True).limit(limit).all()

    return {
        "products": [{
            "id": p.id,
            "name": p.name,
            "category": p.category,
            "price": p.price,
            "cost": p.cost,
            "current_stock": p.current_stock,
            "reorder_threshold": p.reorder_threshold
        } for p in products],
        "count": len(products)
    }

@app.get("/orders")
def get_orders(limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    """Get recent orders."""

    orders = db.query(Order).order_by(Order.order_date.desc()).limit(limit).all()

    return {
        "orders": [{
            "id": o.id,
            "order_date": o.order_date.strftime('%Y-%m-%d'),
            "product_name": o.product_name,
            "category": o.category,
            "quantity": o.quantity,
            "total_revenue": o.total_revenue,
            "total_profit": o.total_profit,
            "status": o.status
        } for o in orders],
        "count": len(orders)
    }

# ==================== Run App ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
