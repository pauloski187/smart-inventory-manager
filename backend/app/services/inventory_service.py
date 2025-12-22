from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from ..models import Product, Order, InventoryMovement, Customer, Seller
from ..schemas import ProductCreate, ProductUpdate, OrderCreate
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd

class InventoryService:
    def __init__(self, db: Session):
        self.db = db

    def create_product(self, product: ProductCreate) -> Product:
        db_product = Product(**product.dict())
        self.db.add(db_product)
        self.db.commit()
        self.db.refresh(db_product)
        return db_product

    def get_product(self, product_id: str) -> Product:
        return self.db.query(Product).filter(Product.id == product_id).first()

    def update_product(self, product_id: str, product_update: ProductUpdate) -> Product:
        db_product = self.get_product(product_id)
        if db_product:
            for key, value in product_update.dict(exclude_unset=True).items():
                setattr(db_product, key, value)
            self.db.commit()
            self.db.refresh(db_product)
        return db_product

    def record_order(self, order: OrderCreate) -> Order:
        # Create order
        db_order = Order(**order.dict())
        self.db.add(db_order)

        # Update product stock
        product = self.get_product(order.product_id)
        if product:
            product.current_stock -= order.quantity
            # Record inventory movement
            movement = InventoryMovement(
                product_id=order.product_id,
                movement_type='sale',
                quantity=-order.quantity,
                reason=f'Order {order.id}'
            )
            self.db.add(movement)

        self.db.commit()
        self.db.refresh(db_order)
        return db_order

    def get_inventory_alerts(self) -> List[Dict[str, Any]]:
        alerts = []
        products = self.db.query(Product).filter(Product.is_active == True).all()
        for product in products:
            if product.current_stock <= product.reorder_threshold:
                alerts.append({
                    'product_id': product.id,
                    'product_name': product.name,
                    'current_stock': product.current_stock,
                    'reorder_threshold': product.reorder_threshold,
                    'alert_type': 'low_stock'
                })
        return alerts

    def get_sales_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        start_date = datetime.utcnow() - timedelta(days=days)
        trends = self.db.query(
            func.date(Order.order_date).label('date'),
            func.sum(Order.total_amount).label('total_sales'),
            func.count(Order.id).label('total_orders')
        ).filter(Order.order_date >= start_date).group_by(func.date(Order.order_date)).all()

        return [{
            'period': str(trend.date),
            'total_sales': float(trend.total_sales),
            'total_orders': trend.total_orders,
            'avg_order_value': float(trend.total_sales) / trend.total_orders if trend.total_orders > 0 else 0
        } for trend in trends]

    def perform_abc_analysis(self) -> List[Dict[str, Any]]:
        # Calculate total revenue per product
        product_revenue = self.db.query(
            Product.id,
            Product.name,
            Product.category,
            func.sum(Order.total_amount).label('total_revenue'),
            func.sum(Order.quantity).label('total_quantity')
        ).join(Order).group_by(Product.id).order_by(desc(func.sum(Order.total_amount))).all()

        total_revenue = sum(p.total_revenue for p in product_revenue)
        cumulative_revenue = 0
        abc_analysis = []

        for product in product_revenue:
            cumulative_revenue += product.total_revenue
            percentage = (cumulative_revenue / total_revenue) * 100

            if percentage <= 80:
                abc_class = 'A'
            elif percentage <= 95:
                abc_class = 'B'
            else:
                abc_class = 'C'

            abc_analysis.append({
                'product_id': product.id,
                'product_name': product.name,
                'category': product.category,
                'total_revenue': float(product.total_revenue),
                'total_quantity': product.total_quantity,
                'abc_class': abc_class
            })

        return abc_analysis

    def simple_demand_forecast(self, product_id: str, days_ahead: int = 30) -> Dict[str, Any]:
        # Simple moving average forecast
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=90)  # Use last 90 days for forecast

        daily_sales = self.db.query(
            func.date(Order.order_date).label('date'),
            func.sum(Order.quantity).label('quantity')
        ).filter(
            Order.product_id == product_id,
            Order.order_date >= start_date
        ).group_by(func.date(Order.order_date)).order_by(func.date(Order.order_date)).all()

        if not daily_sales:
            return {'product_id': product_id, 'predicted_demand': 0}

        # Calculate average daily demand
        total_quantity = sum(s.quantity for s in daily_sales)
        avg_daily_demand = total_quantity / len(daily_sales)

        return {
            'product_id': product_id,
            'forecast_period': f'next_{days_ahead}_days',
            'predicted_demand': avg_daily_demand * days_ahead
        }