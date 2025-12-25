"""
Feature Engineering Module

Creates derived metrics for business intelligence.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import logging

from ..models import Product, Order, Customer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for inventory and sales analytics."""

    def __init__(self, db: Session):
        self.db = db

    # ==================== Inventory Metrics ====================

    def calculate_current_stock(self, product_id: str) -> int:
        """
        Calculate real-time stock level for a product.

        In a real system, this would consider initial stock minus orders
        plus restocks. Here we return the stored current_stock value.
        """
        product = self.db.query(Product).filter(Product.id == product_id).first()
        return product.current_stock if product else 0

    def calculate_stock_velocity(self, product_id: str, days: int = 30) -> float:
        """
        Calculate units sold per day over the last N days.

        Args:
            product_id: The product to analyze
            days: Number of days to look back (default 30)

        Returns:
            Average daily units sold
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        total_sold = self.db.query(func.sum(Order.quantity)).filter(
            Order.product_id == product_id,
            Order.order_date >= cutoff_date,
            Order.order_status != 'Cancelled'
        ).scalar() or 0

        return round(total_sold / days, 2)

    def calculate_inventory_turnover(self, product_id: str, days: int = 365) -> float:
        """
        Calculate inventory turnover ratio.

        Turnover = Cost of Goods Sold / Average Inventory

        Args:
            product_id: The product to analyze
            days: Period for calculation

        Returns:
            Inventory turnover ratio
        """
        product = self.db.query(Product).filter(Product.id == product_id).first()
        if not product or product.current_stock == 0:
            return 0.0

        cutoff_date = datetime.now() - timedelta(days=days)

        # Calculate total units sold
        total_sold = self.db.query(func.sum(Order.quantity)).filter(
            Order.product_id == product_id,
            Order.order_date >= cutoff_date,
            Order.order_status == 'Delivered'
        ).scalar() or 0

        # Cost of goods sold
        cost_price = product.cost_price or (product.unit_price * 0.7)
        cogs = total_sold * cost_price

        # Average inventory (simplified: use current stock)
        avg_inventory_value = product.current_stock * cost_price

        if avg_inventory_value == 0:
            return 0.0

        return round(cogs / avg_inventory_value, 2)

    def days_since_last_sale(self, product_id: str) -> int:
        """
        Calculate days since product was last ordered.

        Args:
            product_id: The product to check

        Returns:
            Number of days since last sale, or -1 if never sold
        """
        last_order = self.db.query(Order).filter(
            Order.product_id == product_id,
            Order.order_status != 'Cancelled'
        ).order_by(desc(Order.order_date)).first()

        if not last_order:
            return -1

        delta = datetime.now() - last_order.order_date
        return delta.days

    def calculate_days_until_stockout(self, product_id: str) -> Optional[int]:
        """
        Estimate days until product runs out of stock.

        Based on current stock and recent sales velocity.
        """
        product = self.db.query(Product).filter(Product.id == product_id).first()
        if not product or product.current_stock <= 0:
            return 0

        velocity = self.calculate_stock_velocity(product_id, days=30)
        if velocity <= 0:
            return None  # No recent sales, can't estimate

        return int(product.current_stock / velocity)

    # ==================== Demand Metrics ====================

    def calculate_rolling_average_demand(
        self, product_id: str, window: int = 7
    ) -> float:
        """
        Calculate N-day rolling average demand.

        Args:
            product_id: The product to analyze
            window: Rolling window in days (7 or 30 typically)

        Returns:
            Average daily demand over the window
        """
        return self.calculate_stock_velocity(product_id, days=window)

    def calculate_demand_trend(self, product_id: str) -> str:
        """
        Determine if demand is increasing, stable, or declining.

        Compares recent 30-day demand to prior 30-day period.

        Returns:
            'Increasing', 'Stable', or 'Declining'
        """
        now = datetime.now()
        recent_start = now - timedelta(days=30)
        prior_start = now - timedelta(days=60)

        # Recent period demand
        recent_demand = self.db.query(func.sum(Order.quantity)).filter(
            Order.product_id == product_id,
            Order.order_date >= recent_start,
            Order.order_status != 'Cancelled'
        ).scalar() or 0

        # Prior period demand
        prior_demand = self.db.query(func.sum(Order.quantity)).filter(
            Order.product_id == product_id,
            Order.order_date >= prior_start,
            Order.order_date < recent_start,
            Order.order_status != 'Cancelled'
        ).scalar() or 0

        if prior_demand == 0:
            return 'Stable' if recent_demand == 0 else 'Increasing'

        change_pct = (recent_demand - prior_demand) / prior_demand

        if change_pct > 0.1:
            return 'Increasing'
        elif change_pct < -0.1:
            return 'Declining'
        else:
            return 'Stable'

    # ==================== Revenue Metrics ====================

    def calculate_revenue_contribution(self, product_id: str) -> float:
        """
        Calculate percentage of total revenue from this product.

        Returns:
            Revenue contribution as a percentage (0-100)
        """
        product_revenue = self.db.query(func.sum(Order.total_amount)).filter(
            Order.product_id == product_id,
            Order.order_status == 'Delivered'
        ).scalar() or 0

        total_revenue = self.db.query(func.sum(Order.total_amount)).filter(
            Order.order_status == 'Delivered'
        ).scalar() or 1  # Avoid division by zero

        return round((product_revenue / total_revenue) * 100, 2)

    def calculate_profit_contribution(self, product_id: str) -> float:
        """
        Calculate total profit from this product.

        Returns:
            Total profit amount
        """
        total_profit = self.db.query(func.sum(Order.profit)).filter(
            Order.product_id == product_id,
            Order.order_status == 'Delivered'
        ).scalar() or 0

        return round(total_profit, 2)

    # ==================== Customer Metrics ====================

    def calculate_customer_lifetime_value(self, customer_id: str) -> float:
        """
        Calculate total revenue from a customer.

        Returns:
            Customer lifetime value (total revenue)
        """
        total_revenue = self.db.query(func.sum(Order.total_amount)).filter(
            Order.customer_id == customer_id,
            Order.order_status == 'Delivered'
        ).scalar() or 0

        return round(total_revenue, 2)

    def segment_customers(self) -> pd.DataFrame:
        """
        Perform RFM (Recency, Frequency, Monetary) segmentation.

        Returns:
            DataFrame with customer segmentation
        """
        now = datetime.now()

        # Get customer metrics
        customer_data = []

        customers = self.db.query(Customer).all()
        for customer in customers:
            orders = self.db.query(Order).filter(
                Order.customer_id == customer.id,
                Order.order_status == 'Delivered'
            ).all()

            if not orders:
                continue

            # Recency: days since last order
            last_order = max(orders, key=lambda o: o.order_date)
            recency = (now - last_order.order_date).days

            # Frequency: total number of orders
            frequency = len(orders)

            # Monetary: total spend
            monetary = sum(o.total_amount for o in orders)

            customer_data.append({
                'customer_id': customer.id,
                'customer_name': customer.name,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary
            })

        if not customer_data:
            return pd.DataFrame()

        df = pd.DataFrame(customer_data)

        # Calculate RFM scores (1-5 scale)
        df['r_score'] = pd.qcut(df['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        df['m_score'] = pd.qcut(df['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        # Create combined RFM score
        df['rfm_score'] = df['r_score'].astype(int) + df['f_score'].astype(int) + df['m_score'].astype(int)

        # Segment customers
        def get_segment(row):
            if row['rfm_score'] >= 12:
                return 'VIP'
            elif row['rfm_score'] >= 9:
                return 'Regular'
            elif row['rfm_score'] >= 6:
                return 'At-Risk'
            else:
                return 'Lost'

        df['segment'] = df.apply(get_segment, axis=1)

        return df

    # ==================== Product Analysis ====================

    def get_product_metrics(self, product_id: str) -> dict:
        """
        Get comprehensive metrics for a product.

        Returns:
            Dictionary with all product metrics
        """
        product = self.db.query(Product).filter(Product.id == product_id).first()
        if not product:
            return {}

        return {
            'product_id': product_id,
            'product_name': product.name,
            'current_stock': product.current_stock,
            'reorder_threshold': product.reorder_threshold,
            'stock_velocity': self.calculate_stock_velocity(product_id),
            'days_until_stockout': self.calculate_days_until_stockout(product_id),
            'days_since_last_sale': self.days_since_last_sale(product_id),
            'inventory_turnover': self.calculate_inventory_turnover(product_id),
            'demand_trend': self.calculate_demand_trend(product_id),
            'revenue_contribution': self.calculate_revenue_contribution(product_id),
            'profit_contribution': self.calculate_profit_contribution(product_id),
            'low_stock_alert': product.current_stock <= product.reorder_threshold
        }

    def get_all_product_metrics(self, limit: int = 100) -> List[dict]:
        """Get metrics for all products."""
        products = self.db.query(Product).limit(limit).all()
        return [self.get_product_metrics(p.id) for p in products]
