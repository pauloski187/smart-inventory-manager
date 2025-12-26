"""
Database Seeding Script

Loads the Amazon dataset into SQLite database for the dashboard to display.
Run this script to populate the database with initial data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import engine, Base, SessionLocal
from app.models.product import Product
from app.models.customer import Customer
from app.models.seller import Seller
from app.models.order import Order

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to dataset (check multiple locations)
DATA_PATHS = [
    Path(__file__).parent.parent / 'data' / 'olist_public_dataset_v2.csv',  # backend/data/
    Path(__file__).parent.parent.parent / 'ml' / 'data' / 'raw' / 'olist_public_dataset_v2.csv',  # ml/data/raw/
]

def get_data_path():
    """Find the dataset file."""
    for path in DATA_PATHS:
        if path.exists():
            return path
    return DATA_PATHS[0]  # Default

DATA_PATH = get_data_path()


def seed_database(csv_path: str = None, clear_existing: bool = True):
    """
    Seed the database with data from CSV file.

    Args:
        csv_path: Path to CSV file (defaults to olist_public_dataset_v2.csv)
        clear_existing: Whether to clear existing data before seeding
    """
    if csv_path is None:
        csv_path = DATA_PATH

    logger.info(f"Loading data from {csv_path}")

    # Check if file exists
    if not Path(csv_path).exists():
        logger.error(f"Data file not found: {csv_path}")
        return False

    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records")

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Get session
    db = SessionLocal()

    try:
        if clear_existing:
            logger.info("Clearing existing data...")
            db.query(Order).delete()
            db.query(Product).delete()
            db.query(Customer).delete()
            db.query(Seller).delete()
            db.commit()

        # Seed Sellers
        logger.info("Seeding sellers...")
        sellers = df['SellerID'].unique()
        for seller_id in sellers:
            seller = Seller(
                id=seller_id,
                name=f"Seller {seller_id.split('-')[1]}",
                is_active=True
            )
            db.merge(seller)
        db.commit()
        logger.info(f"Added {len(sellers)} sellers")

        # Seed Customers (drop duplicates by CustomerID, keep first)
        logger.info("Seeding customers...")
        customers_df = df[['CustomerID', 'CustomerName', 'City', 'State', 'Country']].drop_duplicates(subset=['CustomerID'], keep='first')
        for _, row in customers_df.iterrows():
            customer = Customer(
                id=row['CustomerID'],
                name=row['CustomerName'] if pd.notna(row['CustomerName']) else 'Unknown',
                city=row['City'] if pd.notna(row['City']) else None,
                state=row['State'] if pd.notna(row['State']) else None,
                country=row['Country'] if pd.notna(row['Country']) else 'USA',
                is_active=True
            )
            db.merge(customer)
        db.commit()
        logger.info(f"Added {len(customers_df)} customers")

        # Seed Products
        logger.info("Seeding products...")
        products_df = df[['ProductID', 'ProductName', 'Category', 'Brand', 'UnitPrice']].drop_duplicates(subset=['ProductID'])

        # Calculate initial stock based on order frequency
        product_orders = df.groupby('ProductID')['Quantity'].sum().to_dict()

        for _, row in products_df.iterrows():
            # Set initial stock as 2x total ordered (simulating adequate inventory)
            total_ordered = product_orders.get(row['ProductID'], 0)
            initial_stock = max(int(total_ordered * 0.3), 50)  # At least 50 or 30% of total ordered
            current_stock = max(int(initial_stock * 0.4), 20)  # 40% remaining

            product = Product(
                id=row['ProductID'],
                name=row['ProductName'],
                category=row['Category'],
                brand=row['Brand'],
                unit_price=row['UnitPrice'],
                cost_price=row['UnitPrice'] * 0.6,  # Assume 40% margin
                initial_stock=initial_stock,
                current_stock=current_stock,
                reorder_threshold=max(int(initial_stock * 0.2), 10),
                reorder_quantity=max(int(initial_stock * 0.5), 25),
                stock_status="In Stock" if current_stock > 10 else "Low Stock",
                is_active=True
            )
            db.merge(product)
        db.commit()
        logger.info(f"Added {len(products_df)} products")

        # Seed Orders
        logger.info("Seeding orders...")
        order_count = 0
        batch_size = 1000

        for i, row in df.iterrows():
            # Calculate profit
            cost = row['UnitPrice'] * 0.6 * row['Quantity']
            revenue = row['TotalAmount']
            profit = revenue - cost - row['ShippingCost']
            profit_margin = (profit / revenue * 100) if revenue > 0 else 0

            order = Order(
                id=row['OrderID'],
                order_date=pd.to_datetime(row['OrderDate']),
                customer_id=row['CustomerID'],
                product_id=row['ProductID'],
                seller_id=row['SellerID'],
                quantity=row['Quantity'],
                unit_price=row['UnitPrice'],
                discount=row['Discount'],
                tax=row['Tax'],
                shipping_cost=row['ShippingCost'],
                total_amount=row['TotalAmount'],
                profit=profit,
                profit_margin=profit_margin,
                payment_method=row['PaymentMethod'],
                order_status=row['OrderStatus'],
                returned=False
            )
            db.merge(order)
            order_count += 1

            # Commit in batches
            if order_count % batch_size == 0:
                db.commit()
                logger.info(f"Processed {order_count} orders...")

        db.commit()
        logger.info(f"Added {order_count} orders")

        # Summary
        logger.info("\n" + "="*50)
        logger.info("DATABASE SEEDING COMPLETE")
        logger.info("="*50)
        logger.info(f"Sellers:   {db.query(Seller).count()}")
        logger.info(f"Customers: {db.query(Customer).count()}")
        logger.info(f"Products:  {db.query(Product).count()}")
        logger.info(f"Orders:    {db.query(Order).count()}")
        logger.info("="*50)

        return True

    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def check_database():
    """Check if database has data."""
    db = SessionLocal()
    try:
        order_count = db.query(Order).count()
        product_count = db.query(Product).count()
        return order_count > 0 and product_count > 0
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed the database with initial data")
    parser.add_argument("--csv", help="Path to CSV file", default=None)
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing data")

    args = parser.parse_args()

    success = seed_database(
        csv_path=args.csv,
        clear_existing=not args.no_clear
    )

    sys.exit(0 if success else 1)
