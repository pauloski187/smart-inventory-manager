"""
Data Ingestion Module

Handles reading and parsing CSV files for the Smart Inventory Manager.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _safe_str(value, default: Optional[str] = None) -> Optional[str]:
    """Convert value to string, handling NaN/None."""
    if pd.isna(value):
        return default
    return str(value)


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float, handling NaN/None."""
    if pd.isna(value):
        return default
    return float(value)


def _safe_int(value, default: int = 0) -> int:
    """Convert value to int, handling NaN/None."""
    if pd.isna(value):
        return default
    return int(value)


def ingest_customers(csv_path: str) -> pd.DataFrame:
    """
    Ingest customers from CSV file.

    Args:
        csv_path: Path to customers.csv

    Returns:
        DataFrame with cleaned customer data
    """
    logger.info(f"Ingesting customers from {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column names
    column_mapping = {
        'CustomerID': 'customer_id',
        'CustomerName': 'customer_name',
        'Customer_Type': 'customer_type',
        'City': 'city',
        'State': 'state',
        'Country': 'country'
    }
    df = df.rename(columns=column_mapping)

    # Clean data
    df['customer_name'] = df['customer_name'].fillna('Unknown Customer')
    df['customer_type'] = df['customer_type'].fillna('New')
    df['country'] = df['country'].fillna('USA')

    logger.info(f"Ingested {len(df)} customers")
    return df


def ingest_products(csv_path: str) -> pd.DataFrame:
    """
    Ingest products from CSV file.

    Args:
        csv_path: Path to products.csv

    Returns:
        DataFrame with cleaned product data
    """
    logger.info(f"Ingesting products from {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column names
    column_mapping = {
        'ProductID': 'product_id',
        'ProductName': 'product_name',
        'Category': 'category',
        'Brand': 'brand',
        'Cost_Price': 'cost_price'
    }
    df = df.rename(columns=column_mapping)

    # Clean data
    df['product_name'] = df['product_name'].fillna('Unknown Product')
    df['cost_price'] = pd.to_numeric(df['cost_price'], errors='coerce').fillna(0.0)

    # Calculate unit price with default markup
    df['unit_price'] = df['cost_price'] * 1.3

    logger.info(f"Ingested {len(df)} products")
    return df


def ingest_inventory(csv_path: str) -> pd.DataFrame:
    """
    Ingest inventory data from CSV file.

    Args:
        csv_path: Path to inventory.csv

    Returns:
        DataFrame with cleaned inventory data
    """
    logger.info(f"Ingesting inventory from {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column names
    column_mapping = {
        'ProductID': 'product_id',
        'Initial_Stock': 'initial_stock',
        'Current_Stock': 'current_stock',
        'Reorder_Level': 'reorder_level',
        'Restock_Quantity': 'restock_quantity',
        'Stock_Status': 'stock_status'
    }
    df = df.rename(columns=column_mapping)

    # Clean data - ensure non-negative integers
    for col in ['initial_stock', 'current_stock', 'reorder_level', 'restock_quantity']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        df[col] = df[col].clip(lower=0)

    df['stock_status'] = df['stock_status'].fillna('Unknown')

    logger.info(f"Ingested {len(df)} inventory records")
    return df


def ingest_orders(csv_path: str) -> pd.DataFrame:
    """
    Ingest orders from CSV file.

    Args:
        csv_path: Path to orders.csv

    Returns:
        DataFrame with cleaned order data
    """
    logger.info(f"Ingesting orders from {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column names
    column_mapping = {
        'OrderID': 'order_id',
        'OrderDate': 'order_date',
        'CustomerID': 'customer_id',
        'PaymentMethod': 'payment_method',
        'OrderStatus': 'order_status',
        'Delivery_Date': 'delivery_date',
        'Created_At': 'created_at',
        'Updated_At': 'updated_at'
    }
    df = df.rename(columns=column_mapping)

    # Parse dates
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')

    logger.info(f"Ingested {len(df)} orders")
    return df


def ingest_order_items(csv_path: str) -> pd.DataFrame:
    """
    Ingest order items from CSV file.

    Args:
        csv_path: Path to order_items.csv

    Returns:
        DataFrame with cleaned order item data
    """
    logger.info(f"Ingesting order items from {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column names
    column_mapping = {
        'OrderID': 'order_id',
        'ProductID': 'product_id',
        'SellerID': 'seller_id',
        'Quantity': 'quantity',
        'UnitPrice': 'unit_price',
        'Discount': 'discount',
        'Tax': 'tax',
        'ShippingCost': 'shipping_cost',
        'TotalAmount': 'total_amount',
        'Profit': 'profit',
        'Profit_Margin': 'profit_margin',
        'Returned': 'returned',
        'Refund_Amount': 'refund_amount'
    }
    df = df.rename(columns=column_mapping)

    # Clean numeric columns
    numeric_cols = ['quantity', 'unit_price', 'discount', 'tax',
                    'shipping_cost', 'total_amount', 'profit',
                    'profit_margin', 'refund_amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Ensure quantity is positive integer
    df['quantity'] = df['quantity'].astype(int).clip(lower=1)

    # Parse returned column
    if 'returned' in df.columns:
        df['returned'] = df['returned'].apply(
            lambda x: str(x).lower() in ('true', '1', 'yes') if pd.notna(x) else False
        )

    logger.info(f"Ingested {len(df)} order items")
    return df


def ingest_sellers(csv_path: str) -> pd.DataFrame:
    """
    Ingest sellers from CSV file.

    Args:
        csv_path: Path to sellers.csv

    Returns:
        DataFrame with cleaned seller data
    """
    logger.info(f"Ingesting sellers from {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column names
    column_mapping = {
        'SellerID': 'seller_id',
        'SellerName': 'seller_name',
        'SellerRating': 'seller_rating',
        'Location': 'location'
    }
    df = df.rename(columns=column_mapping)

    # Generate seller names if not present
    if 'seller_name' not in df.columns:
        df['seller_name'] = df['seller_id'].apply(
            lambda x: f"Seller {x.split('-')[1]}" if '-' in str(x) else f"Seller {x}"
        )

    logger.info(f"Ingested {len(df)} sellers")
    return df


def merge_orders_with_items(orders_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge orders with their line items.

    Args:
        orders_df: DataFrame from ingest_orders
        items_df: DataFrame from ingest_order_items

    Returns:
        Merged DataFrame with complete order information
    """
    merged = pd.merge(orders_df, items_df, on='order_id', how='inner')
    logger.info(f"Merged {len(merged)} order records")
    return merged
