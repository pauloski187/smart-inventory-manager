import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func
from ..models import Product, Customer, Order, Seller
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, db: Session):
        self.db = db

    def load_csv_data(self, csv_path: str, column_mapping: dict = None):
        """
        Load data from CSV and populate database.
        column_mapping allows remapping CSV columns to expected format.
        """
        df = pd.read_csv(csv_path)

        # Apply column mapping if provided
        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_columns = [
            'OrderID', 'OrderDate', 'CustomerID', 'CustomerName',
            'ProductID', 'ProductName', 'Category', 'Brand',
            'Quantity', 'UnitPrice', 'Discount', 'Tax',
            'ShippingCost', 'TotalAmount', 'PaymentMethod',
            'OrderStatus', 'City', 'State', 'Country', 'SellerID'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Process data
        self._load_products(df)
        self._load_customers(df)
        self._load_sellers(df)
        self._load_orders(df)

        logger.info(f"Successfully loaded {len(df)} orders from {csv_path}")

    def _load_products(self, df: pd.DataFrame):
        """Extract and load unique products"""
        products_df = df[['ProductID', 'ProductName', 'Category', 'Brand', 'UnitPrice']].drop_duplicates('ProductID')

        for _, row in products_df.iterrows():
            product = Product(
                id=str(row['ProductID']),
                name=row['ProductName'],
                category=row['Category'],
                brand=row['Brand'],
                unit_price=float(row['UnitPrice']),
                current_stock=0  # Will be calculated from orders
            )
            self.db.merge(product)  # Use merge to handle duplicates

        self.db.commit()

    def _load_customers(self, df: pd.DataFrame):
        """Extract and load unique customers"""
        customers_df = df[['CustomerID', 'CustomerName', 'City', 'State', 'Country']].drop_duplicates('CustomerID')

        for _, row in customers_df.iterrows():
            customer = Customer(
                id=str(row['CustomerID']),
                name=row['CustomerName'],
                city=row['City'],
                state=row['State'],
                country=row['Country']
            )
            self.db.merge(customer)

        self.db.commit()

    def _load_sellers(self, df: pd.DataFrame):
        """Extract and load unique sellers"""
        sellers_df = df[['SellerID']].drop_duplicates('SellerID')

        for _, row in sellers_df.iterrows():
            seller = Seller(id=str(row['SellerID']))
            self.db.merge(seller)

        self.db.commit()

    def _load_orders(self, df: pd.DataFrame):
        """Load all orders"""
        for _, row in df.iterrows():
            order = Order(
                id=str(row['OrderID']),
                order_date=pd.to_datetime(row['OrderDate']),
                customer_id=str(row['CustomerID']),
                product_id=str(row['ProductID']),
                seller_id=str(row['SellerID']),
                quantity=int(row['Quantity']),
                unit_price=float(row['UnitPrice']),
                discount=float(row['Discount']) if pd.notna(row['Discount']) else 0.0,
                tax=float(row['Tax']) if pd.notna(row['Tax']) else 0.0,
                shipping_cost=float(row['ShippingCost']) if pd.notna(row['ShippingCost']) else 0.0,
                total_amount=float(row['TotalAmount']),
                payment_method=row['PaymentMethod'],
                order_status=row['OrderStatus'],
                city=row['City'],
                state=row['State'],
                country=row['Country']
            )
            self.db.add(order)

        self.db.commit()

    def calculate_initial_stock(self):
        """Calculate initial stock levels based on historical orders"""
        # This is a simplified approach - in reality, you'd need initial stock data
        # For now, assume all stock starts at some level and subtract sales

        products = self.db.query(Product).all()
        for product in products:
            # Calculate total sold
            total_sold = self.db.query(func.sum(Order.quantity)).filter(Order.product_id == product.id).scalar() or 0
            # Assume initial stock was total_sold + some buffer
            initial_stock = total_sold + 100  # Arbitrary buffer
            product.current_stock = initial_stock - total_sold

        self.db.commit()
        logger.info("Calculated initial stock levels for all products")