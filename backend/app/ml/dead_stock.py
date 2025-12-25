"""
Dead Stock Detection Module

Identifies products with no recent sales activity.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import logging

from ..models import Order, Product

logger = logging.getLogger(__name__)


class DeadStockDetector:
    """
    Detects dead and slow-moving inventory.

    Dead Stock: Products with no sales in specified period
    Slow-Moving: Products with very low sales velocity
    """

    def __init__(self, db: Session, threshold_days: int = 90):
        """
        Initialize detector.

        Args:
            db: Database session
            threshold_days: Days without sales to consider dead stock
        """
        self.db = db
        self.threshold_days = threshold_days

    def get_last_sale_dates(self) -> Dict[str, datetime]:
        """
        Get the last sale date for each product.

        Returns:
            Dictionary mapping product_id to last sale date
        """
        results = self.db.query(
            Order.product_id,
            func.max(Order.order_date).label('last_sale')
        ).filter(
            Order.order_status != 'Cancelled'
        ).group_by(
            Order.product_id
        ).all()

        return {r.product_id: r.last_sale for r in results}

    def detect_dead_stock(
        self,
        threshold_days: Optional[int] = None
    ) -> List[Dict]:
        """
        Identify products with no sales in the threshold period.

        Args:
            threshold_days: Override default threshold

        Returns:
            List of dead stock products with details
        """
        threshold = threshold_days or self.threshold_days
        cutoff_date = datetime.now() - timedelta(days=threshold)

        # Get last sale dates
        last_sales = self.get_last_sale_dates()

        # Get all products with stock
        products = self.db.query(Product).filter(
            Product.current_stock > 0,
            Product.is_active == True
        ).all()

        dead_stock = []

        for product in products:
            last_sale = last_sales.get(product.id)

            # No sales ever, or last sale before cutoff
            if last_sale is None or last_sale < cutoff_date:
                days_since_sale = (datetime.now() - last_sale).days if last_sale else None
                stock_value = product.current_stock * (product.cost_price or product.unit_price * 0.7)

                dead_stock.append({
                    'product_id': product.id,
                    'product_name': product.name,
                    'category': product.category,
                    'brand': product.brand,
                    'current_stock': product.current_stock,
                    'last_sale_date': last_sale.strftime('%Y-%m-%d') if last_sale else 'Never',
                    'days_since_sale': days_since_sale,
                    'stock_value': round(stock_value, 2),
                    'unit_price': product.unit_price,
                    'recommended_action': self._get_recommendation(days_since_sale, stock_value)
                })

        # Sort by stock value (highest first)
        dead_stock.sort(key=lambda x: x['stock_value'], reverse=True)

        return dead_stock

    def detect_slow_moving_stock(
        self,
        velocity_threshold: float = 0.1,
        days: int = 30
    ) -> List[Dict]:
        """
        Identify products with very low sales velocity.

        Args:
            velocity_threshold: Units per day below which product is slow-moving
            days: Period for calculating velocity

        Returns:
            List of slow-moving products
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        # Get sales velocity per product
        results = self.db.query(
            Order.product_id,
            func.sum(Order.quantity).label('total_sold')
        ).filter(
            Order.order_date >= cutoff_date,
            Order.order_status != 'Cancelled'
        ).group_by(
            Order.product_id
        ).all()

        sales_velocity = {r.product_id: r.total_sold / days for r in results}

        # Get products with stock
        products = self.db.query(Product).filter(
            Product.current_stock > 0,
            Product.is_active == True
        ).all()

        slow_moving = []

        for product in products:
            velocity = sales_velocity.get(product.id, 0)

            if velocity < velocity_threshold and velocity > 0:
                stock_value = product.current_stock * (product.cost_price or product.unit_price * 0.7)
                days_of_stock = int(product.current_stock / velocity) if velocity > 0 else float('inf')

                slow_moving.append({
                    'product_id': product.id,
                    'product_name': product.name,
                    'category': product.category,
                    'current_stock': product.current_stock,
                    'velocity': round(velocity, 3),
                    'days_of_stock': days_of_stock if days_of_stock < 10000 else 'Infinite',
                    'stock_value': round(stock_value, 2),
                    'recommended_action': 'Consider promotional pricing'
                })

        # Sort by velocity (lowest first)
        slow_moving.sort(key=lambda x: x['velocity'])

        return slow_moving

    def get_summary(self) -> Dict:
        """
        Get summary statistics for dead and slow-moving stock.

        Returns:
            Dictionary with summary metrics
        """
        dead_stock = self.detect_dead_stock()
        slow_moving = self.detect_slow_moving_stock()

        total_products = self.db.query(Product).filter(
            Product.current_stock > 0,
            Product.is_active == True
        ).count()

        dead_stock_value = sum(item['stock_value'] for item in dead_stock)
        slow_moving_value = sum(item['stock_value'] for item in slow_moving)

        return {
            'total_active_products': total_products,
            'dead_stock': {
                'count': len(dead_stock),
                'percentage': round(len(dead_stock) / total_products * 100, 1) if total_products > 0 else 0,
                'total_value': round(dead_stock_value, 2),
                'threshold_days': self.threshold_days
            },
            'slow_moving': {
                'count': len(slow_moving),
                'percentage': round(len(slow_moving) / total_products * 100, 1) if total_products > 0 else 0,
                'total_value': round(slow_moving_value, 2)
            },
            'at_risk_value': round(dead_stock_value + slow_moving_value, 2)
        }

    def get_category_breakdown(self) -> List[Dict]:
        """
        Get dead stock breakdown by category.

        Returns:
            List of category summaries
        """
        dead_stock = self.detect_dead_stock()

        categories = {}
        for item in dead_stock:
            cat = item['category'] or 'Unknown'
            if cat not in categories:
                categories[cat] = {
                    'category': cat,
                    'product_count': 0,
                    'total_value': 0,
                    'total_units': 0
                }
            categories[cat]['product_count'] += 1
            categories[cat]['total_value'] += item['stock_value']
            categories[cat]['total_units'] += item['current_stock']

        # Convert to list and sort by value
        result = list(categories.values())
        result.sort(key=lambda x: x['total_value'], reverse=True)

        return result

    def _get_recommendation(
        self,
        days_since_sale: Optional[int],
        stock_value: float
    ) -> str:
        """
        Get action recommendation based on dead stock characteristics.
        """
        if days_since_sale is None:
            return 'Review for discontinuation - never sold'

        if days_since_sale > 180:
            if stock_value > 1000:
                return 'Urgent: Consider liquidation or donation'
            else:
                return 'Consider removal from inventory'

        if days_since_sale > 120:
            return 'Run clearance promotion (50%+ discount)'

        if days_since_sale > 90:
            if stock_value > 500:
                return 'Create bundle offers with popular items'
            else:
                return 'Apply promotional discount (20-30%)'

        return 'Monitor closely'

    def get_clearance_candidates(
        self,
        min_value: float = 100,
        max_items: int = 50
    ) -> List[Dict]:
        """
        Get products recommended for clearance sale.

        Args:
            min_value: Minimum stock value to include
            max_items: Maximum items to return

        Returns:
            List of clearance candidates sorted by priority
        """
        dead_stock = self.detect_dead_stock()

        # Filter by minimum value
        candidates = [
            item for item in dead_stock
            if item['stock_value'] >= min_value
        ]

        # Add suggested discount
        for item in candidates:
            days = item['days_since_sale']
            if days is None or days > 180:
                item['suggested_discount'] = 70
            elif days > 120:
                item['suggested_discount'] = 50
            else:
                item['suggested_discount'] = 30

            item['clearance_price'] = round(
                item['unit_price'] * (1 - item['suggested_discount'] / 100), 2
            )

        # Sort by suggested discount (highest priority first)
        candidates.sort(key=lambda x: x['suggested_discount'], reverse=True)

        return candidates[:max_items]
