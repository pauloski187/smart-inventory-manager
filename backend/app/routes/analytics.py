"""
Analytics & Reporting API Endpoints

Provides inventory analytics, forecasting, and business intelligence.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime, timedelta

from ..services.inventory_service import InventoryService
from ..ml.forecasting import DemandForecaster
from ..ml.abc_analysis import ABCAnalyzer
from ..ml.dead_stock import DeadStockDetector
from ..etl.transform import FeatureEngineer
from ..models import Product, Order, Customer
from ..schemas.analytics import AnalyticsResponse
from ..database import get_db

router = APIRouter()


# ==================== Response Models ====================

class InventoryStatusResponse(BaseModel):
    total_products: int
    low_stock_count: int
    dead_stock_count: int
    out_of_stock_count: int
    in_stock_count: int


class ProductInventoryResponse(BaseModel):
    product_id: str
    product_name: str
    current_stock: int
    reorder_point: int
    stock_velocity: float
    days_until_stockout: Optional[int]
    low_stock_alert: bool
    recommended_reorder_quantity: int


# ==================== Inventory Endpoints ====================

@router.get("/inventory/status", response_model=InventoryStatusResponse)
def get_inventory_status(db: Session = Depends(get_db)):
    """Get overall inventory health metrics."""
    total = db.query(Product).filter(Product.is_active == True).count()

    low_stock = db.query(Product).filter(
        Product.is_active == True,
        Product.current_stock <= Product.reorder_threshold,
        Product.current_stock > 0
    ).count()

    out_of_stock = db.query(Product).filter(
        Product.is_active == True,
        Product.current_stock == 0
    ).count()

    detector = DeadStockDetector(db)
    dead_stock = len(detector.detect_dead_stock())

    return InventoryStatusResponse(
        total_products=total,
        low_stock_count=low_stock,
        dead_stock_count=dead_stock,
        out_of_stock_count=out_of_stock,
        in_stock_count=total - out_of_stock
    )


@router.get("/inventory/product/{product_id}", response_model=ProductInventoryResponse)
def get_product_inventory(product_id: str, db: Session = Depends(get_db)):
    """Get detailed inventory for a specific product."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    engineer = FeatureEngineer(db)
    velocity = engineer.calculate_stock_velocity(product_id)
    days_until_stockout = engineer.calculate_days_until_stockout(product_id)

    return ProductInventoryResponse(
        product_id=product.id,
        product_name=product.name,
        current_stock=product.current_stock,
        reorder_point=product.reorder_threshold,
        stock_velocity=velocity,
        days_until_stockout=days_until_stockout,
        low_stock_alert=product.current_stock <= product.reorder_threshold,
        recommended_reorder_quantity=max(product.reorder_quantity, int(velocity * 30))
    )


@router.get("/inventory/low-stock")
def get_low_stock_products(
    threshold: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get products below reorder point."""
    query = db.query(Product).filter(
        Product.is_active == True,
        Product.current_stock > 0
    )

    if threshold:
        query = query.filter(Product.current_stock <= threshold)
    else:
        query = query.filter(Product.current_stock <= Product.reorder_threshold)

    products = query.order_by(Product.current_stock).all()

    engineer = FeatureEngineer(db)
    results = []
    for p in products:
        velocity = engineer.calculate_stock_velocity(p.id)
        results.append({
            'product_id': p.id,
            'product_name': p.name,
            'category': p.category,
            'current_stock': p.current_stock,
            'reorder_point': p.reorder_threshold,
            'stock_velocity': velocity,
            'priority': 'High' if p.current_stock < p.reorder_threshold / 2 else 'Medium'
        })

    return {'low_stock_products': results, 'count': len(results)}


@router.get("/inventory/dead-stock")
def get_dead_stock(
    threshold_days: int = Query(90, ge=30, le=365),
    db: Session = Depends(get_db)
):
    """Get products with no sales in threshold days."""
    detector = DeadStockDetector(db, threshold_days=threshold_days)
    dead_stock = detector.detect_dead_stock()
    summary = detector.get_summary()

    return {
        'dead_stock_products': dead_stock,
        'count': len(dead_stock),
        'total_value': summary['dead_stock']['total_value'],
        'threshold_days': threshold_days
    }


# ==================== Forecasting Endpoints ====================

@router.get("/forecast/product/{product_id}")
def get_product_forecast(
    product_id: str,
    days: int = Query(30, ge=7, le=90),
    method: str = Query('auto'),
    db: Session = Depends(get_db)
):
    """Get demand forecast for a product."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    forecaster = DemandForecaster(db)
    forecast = forecaster.predict_demand(product_id, days_ahead=days, method=method)

    return forecast


@router.get("/forecast/category/{category}")
def get_category_forecast(
    category: str,
    days: int = Query(30, ge=7, le=90),
    db: Session = Depends(get_db)
):
    """Get aggregated forecast for a product category."""
    forecaster = DemandForecaster(db)
    forecast = forecaster.forecast_category(category, days_ahead=days)

    if 'error' in forecast:
        raise HTTPException(status_code=404, detail=forecast['error'])

    return forecast


# ==================== Analytics Endpoints ====================

@router.get("/abc-analysis")
def get_abc_analysis(
    limit: int = Query(100, ge=10, le=1000),
    db: Session = Depends(get_db)
):
    """Get ABC classification for products."""
    analyzer = ABCAnalyzer(db)
    analysis = analyzer.to_dict_list()[:limit]
    summary = analyzer.get_class_summary()
    recommendations = analyzer.get_recommendations()

    return {
        'analysis': analysis,
        'summary': summary,
        'recommendations': recommendations
    }


@router.get("/top-products")
def get_top_products(
    limit: int = Query(10, ge=1, le=100),
    metric: str = Query('revenue'),
    db: Session = Depends(get_db)
):
    """Get top N products by specified metric (revenue, profit, quantity, velocity)."""
    if metric == 'revenue':
        results = db.query(
            Product.id,
            Product.name,
            Product.category,
            func.sum(Order.total_amount).label('value')
        ).join(Order).filter(
            Order.order_status == 'Delivered'
        ).group_by(Product.id).order_by(desc('value')).limit(limit).all()

    elif metric == 'profit':
        results = db.query(
            Product.id,
            Product.name,
            Product.category,
            func.sum(Order.profit).label('value')
        ).join(Order).filter(
            Order.order_status == 'Delivered'
        ).group_by(Product.id).order_by(desc('value')).limit(limit).all()

    elif metric == 'quantity':
        results = db.query(
            Product.id,
            Product.name,
            Product.category,
            func.sum(Order.quantity).label('value')
        ).join(Order).filter(
            Order.order_status == 'Delivered'
        ).group_by(Product.id).order_by(desc('value')).limit(limit).all()

    else:  # velocity
        engineer = FeatureEngineer(db)
        products = db.query(Product).limit(500).all()
        product_velocities = []
        for p in products:
            velocity = engineer.calculate_stock_velocity(p.id)
            product_velocities.append({
                'product_id': p.id,
                'product_name': p.name,
                'category': p.category,
                'value': velocity
            })
        product_velocities.sort(key=lambda x: x['value'], reverse=True)
        return {'top_products': product_velocities[:limit], 'metric': metric}

    return {
        'top_products': [
            {
                'product_id': r.id,
                'product_name': r.name,
                'category': r.category,
                'value': round(float(r.value), 2)
            }
            for r in results
        ],
        'metric': metric
    }


@router.get("/sales-trends")
def get_sales_trends(
    period: str = Query('daily'),
    days: int = Query(30, ge=7, le=365),
    db: Session = Depends(get_db)
):
    """Get sales trends over time (daily, weekly, monthly)."""
    start_date = datetime.now() - timedelta(days=days)

    if period == 'daily':
        group_by = func.date(Order.order_date)
    elif period == 'weekly':
        group_by = func.strftime('%Y-%W', Order.order_date)
    else:  # monthly
        group_by = func.strftime('%Y-%m', Order.order_date)

    trends = db.query(
        group_by.label('period'),
        func.sum(Order.total_amount).label('total_sales'),
        func.count(Order.id).label('total_orders'),
        func.sum(Order.profit).label('total_profit'),
        func.avg(Order.total_amount).label('avg_order_value')
    ).filter(
        Order.order_date >= start_date,
        Order.order_status == 'Delivered'
    ).group_by(group_by).order_by(group_by).all()

    return {
        'trends': [
            {
                'period': str(t.period),
                'total_sales': round(float(t.total_sales), 2),
                'total_orders': t.total_orders,
                'total_profit': round(float(t.total_profit or 0), 2),
                'avg_order_value': round(float(t.avg_order_value), 2)
            }
            for t in trends
        ],
        'period_type': period,
        'days_covered': days
    }


# ==================== Customer Endpoints ====================

@router.get("/customers/segments")
def get_customer_segments(db: Session = Depends(get_db)):
    """Get customer segmentation (RFM analysis)."""
    engineer = FeatureEngineer(db)
    segments_df = engineer.segment_customers()

    if segments_df.empty:
        return {'segments': [], 'summary': {}}

    summary = segments_df.groupby('segment').agg({
        'customer_id': 'count',
        'monetary': 'sum',
        'frequency': 'mean'
    }).to_dict('index')

    return {
        'segments': segments_df.to_dict('records'),
        'summary': summary
    }


@router.get("/customers/{customer_id}/profile")
def get_customer_profile(customer_id: str, db: Session = Depends(get_db)):
    """Get customer purchase history and metrics."""
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    orders = db.query(Order).filter(
        Order.customer_id == customer_id,
        Order.order_status == 'Delivered'
    ).all()

    if not orders:
        return {
            'customer_id': customer_id,
            'customer_name': customer.name,
            'total_orders': 0,
            'lifetime_value': 0,
            'avg_order_value': 0,
            'last_purchase_date': None
        }

    total_value = sum(o.total_amount for o in orders)
    last_order = max(orders, key=lambda o: o.order_date)

    return {
        'customer_id': customer_id,
        'customer_name': customer.name,
        'customer_type': customer.customer_type,
        'city': customer.city,
        'state': customer.state,
        'total_orders': len(orders),
        'lifetime_value': round(total_value, 2),
        'avg_order_value': round(total_value / len(orders), 2),
        'last_purchase_date': last_order.order_date.strftime('%Y-%m-%d')
    }


# ==================== Dashboard Summary ====================

@router.get("/dashboard/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get high-level KPIs for dashboard."""
    now = datetime.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start_of_year = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    revenue_mtd = db.query(func.sum(Order.total_amount)).filter(
        Order.order_date >= start_of_month,
        Order.order_status == 'Delivered'
    ).scalar() or 0

    revenue_ytd = db.query(func.sum(Order.total_amount)).filter(
        Order.order_date >= start_of_year,
        Order.order_status == 'Delivered'
    ).scalar() or 0

    total_orders = db.query(Order).filter(Order.order_status == 'Delivered').count()
    total_revenue = db.query(func.sum(Order.total_amount)).filter(
        Order.order_status == 'Delivered'
    ).scalar() or 1

    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    low_stock = db.query(Product).filter(
        Product.is_active == True,
        Product.current_stock <= Product.reorder_threshold
    ).count()

    detector = DeadStockDetector(db)
    summary = detector.get_summary()

    top_products = db.query(
        Product.id,
        Product.name,
        func.sum(Order.total_amount).label('revenue')
    ).join(Order).filter(
        Order.order_status == 'Delivered'
    ).group_by(Product.id).order_by(desc('revenue')).limit(5).all()

    return {
        'total_revenue_mtd': round(float(revenue_mtd), 2),
        'total_revenue_ytd': round(float(revenue_ytd), 2),
        'total_orders': total_orders,
        'avg_order_value': round(float(avg_order_value), 2),
        'low_stock_alerts': low_stock,
        'dead_stock_value': summary['dead_stock']['total_value'],
        'top_products': [
            {'product_id': p.id, 'product_name': p.name, 'revenue': round(float(p.revenue), 2)}
            for p in top_products
        ]
    }


# ==================== Monthly Sales Trends & Reports ====================

@router.get("/monthly-sales-trend")
def get_monthly_sales_trend(
    months: int = Query(12, ge=3, le=60),
    db: Session = Depends(get_db)
):
    """
    Get monthly sales trend with revenue, profit, and loss data.
    Perfect for line chart visualization.
    """
    # Get monthly aggregated data
    monthly_data = db.query(
        func.strftime('%Y-%m', Order.order_date).label('month'),
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.count(Order.id).label('order_count'),
        func.sum(Order.quantity).label('units_sold'),
        func.avg(Order.total_amount).label('avg_order_value')
    ).filter(
        Order.order_status == 'Delivered'
    ).group_by(
        func.strftime('%Y-%m', Order.order_date)
    ).order_by(
        func.strftime('%Y-%m', Order.order_date)
    ).all()

    # Calculate loss (negative profit) and growth rates
    trends = []
    prev_revenue = None
    prev_profit = None

    for m in monthly_data[-months:]:
        revenue = float(m.revenue or 0)
        profit = float(m.profit or 0)
        loss = abs(profit) if profit < 0 else 0

        # Calculate month-over-month growth
        revenue_growth = None
        profit_growth = None
        if prev_revenue and prev_revenue > 0:
            revenue_growth = round(((revenue - prev_revenue) / prev_revenue) * 100, 2)
        if prev_profit and prev_profit != 0:
            profit_growth = round(((profit - prev_profit) / abs(prev_profit)) * 100, 2)

        trends.append({
            'month': m.month,
            'revenue': round(revenue, 2),
            'profit': round(profit, 2),
            'loss': round(loss, 2),
            'profit_margin': round((profit / revenue * 100), 2) if revenue > 0 else 0,
            'order_count': m.order_count,
            'units_sold': m.units_sold,
            'avg_order_value': round(float(m.avg_order_value or 0), 2),
            'revenue_growth_pct': revenue_growth,
            'profit_growth_pct': profit_growth
        })

        prev_revenue = revenue
        prev_profit = profit

    # Summary statistics
    total_revenue = sum(t['revenue'] for t in trends)
    total_profit = sum(t['profit'] for t in trends)
    avg_monthly_revenue = total_revenue / len(trends) if trends else 0

    return {
        'monthly_trends': trends,
        'summary': {
            'total_revenue': round(total_revenue, 2),
            'total_profit': round(total_profit, 2),
            'avg_monthly_revenue': round(avg_monthly_revenue, 2),
            'avg_profit_margin': round((total_profit / total_revenue * 100), 2) if total_revenue > 0 else 0,
            'months_analyzed': len(trends)
        }
    }


@router.get("/monthly-report/{year}/{month}")
def get_monthly_report(
    year: int,
    month: int,
    db: Session = Depends(get_db)
):
    """
    Generate comprehensive monthly report for a specific month.
    """
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)

    # Previous month for comparison
    if month == 1:
        prev_start = datetime(year - 1, 12, 1)
        prev_end = start_date
    else:
        prev_start = datetime(year, month - 1, 1)
        prev_end = start_date

    # Current month metrics
    current_orders = db.query(Order).filter(
        Order.order_date >= start_date,
        Order.order_date < end_date,
        Order.order_status == 'Delivered'
    ).all()

    current_revenue = sum(o.total_amount for o in current_orders)
    current_profit = sum(o.profit for o in current_orders)
    current_units = sum(o.quantity for o in current_orders)

    # Previous month metrics
    prev_orders = db.query(Order).filter(
        Order.order_date >= prev_start,
        Order.order_date < prev_end,
        Order.order_status == 'Delivered'
    ).all()

    prev_revenue = sum(o.total_amount for o in prev_orders)
    prev_profit = sum(o.profit for o in prev_orders)

    # Top products this month
    top_products = db.query(
        Product.id,
        Product.name,
        Product.category,
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.sum(Order.quantity).label('units')
    ).join(Order).filter(
        Order.order_date >= start_date,
        Order.order_date < end_date,
        Order.order_status == 'Delivered'
    ).group_by(Product.id).order_by(desc('revenue')).limit(10).all()

    # Worst products this month
    worst_products = db.query(
        Product.id,
        Product.name,
        Product.category,
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.sum(Order.quantity).label('units')
    ).join(Order).filter(
        Order.order_date >= start_date,
        Order.order_date < end_date,
        Order.order_status == 'Delivered'
    ).group_by(Product.id).order_by('revenue').limit(10).all()

    # Category breakdown
    category_breakdown = db.query(
        Product.category,
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.count(Order.id).label('orders')
    ).join(Order).filter(
        Order.order_date >= start_date,
        Order.order_date < end_date,
        Order.order_status == 'Delivered'
    ).group_by(Product.category).order_by(desc('revenue')).all()

    # Daily breakdown for the month
    daily_breakdown = db.query(
        func.date(Order.order_date).label('date'),
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.count(Order.id).label('orders')
    ).filter(
        Order.order_date >= start_date,
        Order.order_date < end_date,
        Order.order_status == 'Delivered'
    ).group_by(func.date(Order.order_date)).order_by('date').all()

    # Calculate growth
    revenue_growth = ((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
    profit_growth = ((current_profit - prev_profit) / abs(prev_profit) * 100) if prev_profit != 0 else 0

    # Generate recommendations
    recommendations = []
    if current_profit < 0:
        recommendations.append({
            'type': 'critical',
            'message': f'Month operated at a loss of ${abs(current_profit):,.2f}. Review pricing and costs.'
        })
    if revenue_growth < -10:
        recommendations.append({
            'type': 'warning',
            'message': f'Revenue declined {abs(revenue_growth):.1f}% from previous month. Consider promotional activities.'
        })
    if revenue_growth > 20:
        recommendations.append({
            'type': 'success',
            'message': f'Strong revenue growth of {revenue_growth:.1f}%. Consider increasing inventory for top performers.'
        })

    # Add product-specific recommendations
    if top_products:
        top_product = top_products[0]
        recommendations.append({
            'type': 'info',
            'message': f'Top performer: {top_product.name} generated ${float(top_product.revenue):,.2f} revenue.'
        })

    return {
        'report_period': {
            'year': year,
            'month': month,
            'month_name': start_date.strftime('%B'),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': (end_date - timedelta(days=1)).strftime('%Y-%m-%d')
        },
        'summary': {
            'total_revenue': round(current_revenue, 2),
            'total_profit': round(current_profit, 2),
            'total_loss': round(abs(current_profit), 2) if current_profit < 0 else 0,
            'profit_margin': round((current_profit / current_revenue * 100), 2) if current_revenue > 0 else 0,
            'total_orders': len(current_orders),
            'total_units_sold': current_units,
            'avg_order_value': round(current_revenue / len(current_orders), 2) if current_orders else 0
        },
        'comparison': {
            'revenue_change': round(current_revenue - prev_revenue, 2),
            'revenue_growth_pct': round(revenue_growth, 2),
            'profit_change': round(current_profit - prev_profit, 2),
            'profit_growth_pct': round(profit_growth, 2),
            'prev_month_revenue': round(prev_revenue, 2),
            'prev_month_profit': round(prev_profit, 2)
        },
        'top_products': [
            {
                'product_id': p.id,
                'product_name': p.name,
                'category': p.category,
                'revenue': round(float(p.revenue), 2),
                'profit': round(float(p.profit), 2),
                'units_sold': p.units
            }
            for p in top_products
        ],
        'worst_products': [
            {
                'product_id': p.id,
                'product_name': p.name,
                'category': p.category,
                'revenue': round(float(p.revenue), 2),
                'profit': round(float(p.profit), 2),
                'units_sold': p.units
            }
            for p in worst_products
        ],
        'category_breakdown': [
            {
                'category': c.category,
                'revenue': round(float(c.revenue), 2),
                'profit': round(float(c.profit), 2),
                'orders': c.orders,
                'revenue_share': round(float(c.revenue) / current_revenue * 100, 2) if current_revenue > 0 else 0
            }
            for c in category_breakdown
        ],
        'daily_breakdown': [
            {
                'date': str(d.date),
                'revenue': round(float(d.revenue), 2),
                'profit': round(float(d.profit), 2),
                'orders': d.orders
            }
            for d in daily_breakdown
        ],
        'recommendations': recommendations
    }


@router.get("/product-performance")
def get_product_performance(
    limit: int = Query(10, ge=5, le=50),
    db: Session = Depends(get_db)
):
    """
    Get best and worst performing products with detailed metrics.
    Includes recommendations for each category.
    """
    # Best performers by revenue
    best_by_revenue = db.query(
        Product.id,
        Product.name,
        Product.category,
        Product.cost_price,
        Product.unit_price,
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.sum(Order.quantity).label('units_sold'),
        func.count(Order.id).label('order_count')
    ).join(Order).filter(
        Order.order_status == 'Delivered'
    ).group_by(Product.id).order_by(desc('revenue')).limit(limit).all()

    # Worst performers by revenue (with at least some sales)
    worst_by_revenue = db.query(
        Product.id,
        Product.name,
        Product.category,
        Product.cost_price,
        Product.unit_price,
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.sum(Order.quantity).label('units_sold'),
        func.count(Order.id).label('order_count')
    ).join(Order).filter(
        Order.order_status == 'Delivered'
    ).group_by(Product.id).order_by('revenue').limit(limit).all()

    # Best by profit margin
    best_by_margin = db.query(
        Product.id,
        Product.name,
        Product.category,
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.sum(Order.quantity).label('units_sold'),
        (func.sum(Order.profit) / func.sum(Order.total_amount) * 100).label('margin')
    ).join(Order).filter(
        Order.order_status == 'Delivered'
    ).group_by(Product.id).having(
        func.sum(Order.total_amount) > 1000  # Minimum revenue threshold
    ).order_by(desc('margin')).limit(limit).all()

    # Worst by profit margin (or negative margin)
    worst_by_margin = db.query(
        Product.id,
        Product.name,
        Product.category,
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.sum(Order.quantity).label('units_sold'),
        (func.sum(Order.profit) / func.sum(Order.total_amount) * 100).label('margin')
    ).join(Order).filter(
        Order.order_status == 'Delivered'
    ).group_by(Product.id).having(
        func.sum(Order.total_amount) > 100
    ).order_by('margin').limit(limit).all()

    def format_product(p, include_margin=False):
        result = {
            'product_id': p.id,
            'product_name': p.name,
            'category': p.category,
            'revenue': round(float(p.revenue), 2),
            'profit': round(float(p.profit), 2),
            'units_sold': p.units_sold,
            'profit_margin': round(float(p.profit) / float(p.revenue) * 100, 2) if float(p.revenue) > 0 else 0
        }
        if hasattr(p, 'order_count'):
            result['order_count'] = p.order_count
        return result

    # Generate recommendations
    recommendations = {
        'best_performers': [
            'Increase inventory for top revenue generators to avoid stockouts',
            'Consider bundling top products with slower movers',
            'Analyze what makes these products successful - apply learnings elsewhere'
        ],
        'worst_performers': [
            'Review pricing strategy for low performers',
            'Consider promotional campaigns or discounts',
            'Evaluate if products should be discontinued'
        ],
        'margin_optimization': [
            'Products with high margin but low volume may benefit from promotion',
            'Low margin products may need cost reduction or price increase',
            'Consider renegotiating supplier costs for high-volume, low-margin items'
        ]
    }

    return {
        'best_by_revenue': [format_product(p) for p in best_by_revenue],
        'worst_by_revenue': [format_product(p) for p in worst_by_revenue],
        'best_by_margin': [format_product(p) for p in best_by_margin],
        'worst_by_margin': [format_product(p) for p in worst_by_margin],
        'recommendations': recommendations
    }


@router.get("/category-performance")
def get_category_performance(db: Session = Depends(get_db)):
    """
    Get performance metrics by category for pie/bar chart visualization.
    """
    categories = db.query(
        Product.category,
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.sum(Order.quantity).label('units_sold'),
        func.count(Order.id).label('order_count'),
        func.count(func.distinct(Product.id)).label('product_count')
    ).join(Order).filter(
        Order.order_status == 'Delivered'
    ).group_by(Product.category).order_by(desc('revenue')).all()

    total_revenue = sum(float(c.revenue) for c in categories)
    total_profit = sum(float(c.profit) for c in categories)

    return {
        'categories': [
            {
                'category': c.category,
                'revenue': round(float(c.revenue), 2),
                'profit': round(float(c.profit), 2),
                'units_sold': c.units_sold,
                'order_count': c.order_count,
                'product_count': c.product_count,
                'revenue_share': round(float(c.revenue) / total_revenue * 100, 2) if total_revenue > 0 else 0,
                'profit_share': round(float(c.profit) / total_profit * 100, 2) if total_profit > 0 else 0,
                'profit_margin': round(float(c.profit) / float(c.revenue) * 100, 2) if float(c.revenue) > 0 else 0,
                'avg_order_value': round(float(c.revenue) / c.order_count, 2) if c.order_count > 0 else 0
            }
            for c in categories
        ],
        'totals': {
            'total_revenue': round(total_revenue, 2),
            'total_profit': round(total_profit, 2),
            'total_categories': len(categories)
        }
    }


@router.get("/sales-by-day-of-week")
def get_sales_by_day_of_week(db: Session = Depends(get_db)):
    """
    Get sales distribution by day of week for pattern analysis.
    """
    # SQLite uses strftime with %w (0=Sunday, 6=Saturday)
    day_sales = db.query(
        func.strftime('%w', Order.order_date).label('day_num'),
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.count(Order.id).label('order_count'),
        func.avg(Order.total_amount).label('avg_order_value')
    ).filter(
        Order.order_status == 'Delivered'
    ).group_by(
        func.strftime('%w', Order.order_date)
    ).order_by('day_num').all()

    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    return {
        'daily_distribution': [
            {
                'day': day_names[int(d.day_num)],
                'day_number': int(d.day_num),
                'revenue': round(float(d.revenue), 2),
                'profit': round(float(d.profit), 2),
                'order_count': d.order_count,
                'avg_order_value': round(float(d.avg_order_value), 2)
            }
            for d in day_sales
        ]
    }


@router.get("/revenue-vs-profit-trend")
def get_revenue_profit_trend(
    period: str = Query('monthly', regex='^(daily|weekly|monthly)$'),
    limit: int = Query(30, ge=7, le=365),
    db: Session = Depends(get_db)
):
    """
    Get revenue vs profit trend for dual-axis line chart.
    Shows both metrics over time for comparison.
    """
    if period == 'daily':
        group_by = func.date(Order.order_date)
        date_format = '%Y-%m-%d'
    elif period == 'weekly':
        group_by = func.strftime('%Y-%W', Order.order_date)
        date_format = '%Y-W%W'
    else:  # monthly
        group_by = func.strftime('%Y-%m', Order.order_date)
        date_format = '%Y-%m'

    trends = db.query(
        group_by.label('period'),
        func.sum(Order.total_amount).label('revenue'),
        func.sum(Order.profit).label('profit'),
        func.count(Order.id).label('orders')
    ).filter(
        Order.order_status == 'Delivered'
    ).group_by(group_by).order_by(desc(group_by)).limit(limit).all()

    # Reverse to chronological order
    trends = list(reversed(trends))

    return {
        'period_type': period,
        'data': [
            {
                'period': str(t.period),
                'revenue': round(float(t.revenue), 2),
                'profit': round(float(t.profit), 2),
                'profit_margin': round(float(t.profit) / float(t.revenue) * 100, 2) if float(t.revenue) > 0 else 0,
                'orders': t.orders
            }
            for t in trends
        ]
    }


@router.get("/available-months")
def get_available_months(db: Session = Depends(get_db)):
    """
    Get list of months that have data for report generation.
    """
    months = db.query(
        func.strftime('%Y', Order.order_date).label('year'),
        func.strftime('%m', Order.order_date).label('month'),
        func.count(Order.id).label('order_count'),
        func.sum(Order.total_amount).label('revenue')
    ).filter(
        Order.order_status == 'Delivered'
    ).group_by(
        func.strftime('%Y-%m', Order.order_date)
    ).order_by(
        desc(func.strftime('%Y-%m', Order.order_date))
    ).all()

    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    return {
        'available_months': [
            {
                'year': int(m.year),
                'month': int(m.month),
                'month_name': month_names[int(m.month)],
                'display': f"{month_names[int(m.month)]} {m.year}",
                'order_count': m.order_count,
                'revenue': round(float(m.revenue), 2)
            }
            for m in months
        ]
    }


# ==================== Legacy endpoints ====================

@router.get("/inventory-alerts")
def get_inventory_alerts(db: Session = Depends(get_db)):
    service = InventoryService(db)
    return {"alerts": service.get_inventory_alerts()}


@router.get("/demand-forecast/{product_id}")
def get_demand_forecast(product_id: str, days_ahead: int = 30, db: Session = Depends(get_db)):
    service = InventoryService(db)
    return service.simple_demand_forecast(product_id, days_ahead)


@router.get("/dashboard", response_model=AnalyticsResponse)
def get_dashboard_analytics(db: Session = Depends(get_db)):
    service = InventoryService(db)
    return AnalyticsResponse(
        sales_trends=service.get_sales_trends(30),
        inventory_alerts=service.get_inventory_alerts(),
        abc_analysis=service.perform_abc_analysis(),
        demand_forecasts=[]
    )
