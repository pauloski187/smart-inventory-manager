#!/usr/bin/env python3
"""
Data Loading Script for Smart Inventory Manager

Loads normalized CSV data files into the SQLite database.
CSV files are located in: ../ml/data/processed/
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, engine
from app.models import Product, Customer, Order, Seller, InventoryMovement
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "ml" / "data" / "processed"


def create_tables():
    """Create all database tables."""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")


def load_sellers(session) -> int:
    """Load sellers from CSV."""
    csv_path = DATA_DIR / "sellers.csv"
    logger.info(f"Loading sellers from {csv_path}")

    df = pd.read_csv(csv_path)
    count = 0

    for _, row in df.iterrows():
        seller = Seller(
            id=str(row['SellerID']),
            name=f"Seller {row['SellerID'].split('-')[1]}",  # Generate name from ID
            is_active=True
        )
        session.merge(seller)
        count += 1

    session.commit()
    logger.info(f"Loaded {count} sellers.")
    return count


def safe_str(value, default="Unknown"):
    """Convert value to string, handling NaN/None."""
    if pd.isna(value):
        return default
    return str(value)


def load_customers(session) -> int:
    """Load customers from CSV."""
    csv_path = DATA_DIR / "customers.csv"
    logger.info(f"Loading customers from {csv_path}")

    df = pd.read_csv(csv_path)
    count = 0

    for _, row in df.iterrows():
        customer = Customer(
            id=str(row['CustomerID']),
            name=safe_str(row.get('CustomerName'), f"Customer {row['CustomerID']}"),
            customer_type=safe_str(row.get('Customer_Type'), 'New'),
            city=safe_str(row.get('City'), None),
            state=safe_str(row.get('State'), None),
            country=safe_str(row.get('Country'), 'USA'),
            is_active=True
        )
        session.merge(customer)
        count += 1

    session.commit()
    logger.info(f"Loaded {count} customers.")
    return count


def load_products(session) -> int:
    """Load products from CSV."""
    csv_path = DATA_DIR / "products.csv"
    logger.info(f"Loading products from {csv_path}")

    df = pd.read_csv(csv_path)
    count = 0

    for _, row in df.iterrows():
        product = Product(
            id=str(row['ProductID']),
            name=safe_str(row.get('ProductName'), f"Product {row['ProductID']}"),
            category=safe_str(row.get('Category'), None),
            brand=safe_str(row.get('Brand'), None),
            cost_price=float(row['Cost_Price']) if pd.notna(row.get('Cost_Price')) else None,
            unit_price=float(row['Cost_Price']) * 1.3 if pd.notna(row.get('Cost_Price')) else 0.0,
            current_stock=0,
            is_active=True
        )
        session.merge(product)
        count += 1

    session.commit()
    logger.info(f"Loaded {count} products.")
    return count


def load_inventory(session) -> int:
    """Load inventory data and update product stock levels."""
    csv_path = DATA_DIR / "inventory.csv"
    logger.info(f"Loading inventory from {csv_path}")

    df = pd.read_csv(csv_path)
    count = 0

    for _, row in df.iterrows():
        product_id = str(row['ProductID'])
        product = session.query(Product).filter(Product.id == product_id).first()

        if product:
            product.initial_stock = int(row['Initial_Stock']) if pd.notna(row.get('Initial_Stock')) else 0
            product.current_stock = int(row['Current_Stock']) if pd.notna(row.get('Current_Stock')) else 0
            product.reorder_threshold = int(row['Reorder_Level']) if pd.notna(row.get('Reorder_Level')) else 10
            product.reorder_quantity = int(row['Restock_Quantity']) if pd.notna(row.get('Restock_Quantity')) else 50
            product.stock_status = row.get('Stock_Status', 'In Stock')
            count += 1

    session.commit()
    logger.info(f"Updated inventory for {count} products.")
    return count


def load_orders(session) -> int:
    """Load orders from orders.csv and order_items.csv."""
    orders_path = DATA_DIR / "orders.csv"
    items_path = DATA_DIR / "order_items.csv"

    logger.info(f"Loading orders from {orders_path} and {items_path}")

    # Load both CSVs
    orders_df = pd.read_csv(orders_path)
    items_df = pd.read_csv(items_path)

    # Merge orders with order items
    merged_df = pd.merge(orders_df, items_df, on='OrderID', how='inner')

    count = 0
    batch_size = 5000

    for i, (_, row) in enumerate(merged_df.iterrows()):
        # Parse dates
        order_date = pd.to_datetime(row['OrderDate'])
        delivery_date = pd.to_datetime(row['Delivery_Date']) if pd.notna(row.get('Delivery_Date')) else None

        # Handle returned field
        returned = False
        if pd.notna(row.get('Returned')):
            returned = str(row['Returned']).lower() in ('true', '1', 'yes')

        order = Order(
            id=str(row['OrderID']),
            order_date=order_date,
            customer_id=str(row['CustomerID']),
            product_id=str(row['ProductID']),
            seller_id=str(row['SellerID']),
            quantity=int(row['Quantity']),
            unit_price=float(row['UnitPrice']) if pd.notna(row.get('UnitPrice')) else 0.0,
            discount=float(row['Discount']) if pd.notna(row.get('Discount')) else 0.0,
            tax=float(row['Tax']) if pd.notna(row.get('Tax')) else 0.0,
            shipping_cost=float(row['ShippingCost']) if pd.notna(row.get('ShippingCost')) else 0.0,
            total_amount=float(row['TotalAmount']) if pd.notna(row.get('TotalAmount')) else 0.0,
            profit=float(row['Profit']) if pd.notna(row.get('Profit')) else 0.0,
            profit_margin=float(row['Profit_Margin']) if pd.notna(row.get('Profit_Margin')) else 0.0,
            payment_method=row.get('PaymentMethod'),
            order_status=row.get('OrderStatus'),
            delivery_date=delivery_date,
            returned=returned,
            refund_amount=float(row['Refund_Amount']) if pd.notna(row.get('Refund_Amount')) else 0.0
        )
        session.merge(order)
        count += 1

        # Commit in batches for performance
        if count % batch_size == 0:
            session.commit()
            logger.info(f"Processed {count} orders...")

    session.commit()
    logger.info(f"Loaded {count} orders.")
    return count


def get_unit_price_from_items(items_df: pd.DataFrame) -> dict:
    """Extract unit prices from order items for products."""
    prices = {}
    for _, row in items_df.iterrows():
        product_id = str(row['ProductID'])
        if product_id not in prices and pd.notna(row.get('UnitPrice')):
            prices[product_id] = float(row['UnitPrice'])
    return prices


def update_product_prices(session, items_df: pd.DataFrame) -> int:
    """Update product unit prices from order items."""
    logger.info("Updating product unit prices from order data...")

    prices = get_unit_price_from_items(items_df)
    count = 0

    for product_id, price in prices.items():
        product = session.query(Product).filter(Product.id == product_id).first()
        if product and product.unit_price == 0.0:
            product.unit_price = price
            count += 1

    session.commit()
    logger.info(f"Updated prices for {count} products.")
    return count


def print_summary(session):
    """Print summary of loaded data."""
    print("\n" + "="*50)
    print("DATA LOADING SUMMARY")
    print("="*50)

    sellers = session.query(Seller).count()
    customers = session.query(Customer).count()
    products = session.query(Product).count()
    orders = session.query(Order).count()

    low_stock = session.query(Product).filter(
        Product.current_stock <= Product.reorder_threshold
    ).count()

    out_of_stock = session.query(Product).filter(
        Product.current_stock == 0
    ).count()

    print(f"Sellers:        {sellers:,}")
    print(f"Customers:      {customers:,}")
    print(f"Products:       {products:,}")
    print(f"Orders:         {orders:,}")
    print("-"*50)
    print(f"Low Stock:      {low_stock:,}")
    print(f"Out of Stock:   {out_of_stock:,}")
    print("="*50 + "\n")


def main():
    """Main entry point for data loading."""
    print("\n" + "="*50)
    print("SMART INVENTORY MANAGER - DATA LOADER")
    print("="*50 + "\n")

    # Check if data directory exists
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    # Check for required CSV files
    required_files = [
        "sellers.csv",
        "customers.csv",
        "products.csv",
        "inventory.csv",
        "orders.csv",
        "order_items.csv"
    ]

    for file in required_files:
        if not (DATA_DIR / file).exists():
            logger.error(f"Required file not found: {DATA_DIR / file}")
            sys.exit(1)

    logger.info(f"All required CSV files found in {DATA_DIR}")

    # Create database session
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Create tables
        create_tables()

        # Load data in order (respecting foreign key constraints)
        load_sellers(session)
        load_customers(session)
        load_products(session)
        load_inventory(session)

        # Load order items to get unit prices
        items_df = pd.read_csv(DATA_DIR / "order_items.csv")
        update_product_prices(session, items_df)

        # Load orders
        load_orders(session)

        # Print summary
        print_summary(session)

        logger.info("Data loading completed successfully!")

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
