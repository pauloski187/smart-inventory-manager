# Inventory Management Logic

## Stock Tracking

### Current Stock Calculation

Stock levels are maintained in the `Product.current_stock` field and updated automatically when orders are processed:

```python
# When an order is created
product.current_stock -= order.quantity

# Record inventory movement
movement = InventoryMovement(
    product_id=order.product_id,
    movement_type='sale',
    quantity=-order.quantity,
    reason=f'Order {order.id}'
)
```

### Initial Stock Setup

For existing datasets, initial stock is calculated as:
```
initial_stock = total_sold + buffer (100 units)
current_stock = initial_stock - total_sold
```

## Inventory Alerts

### Low Stock Alerts

Triggered when `current_stock <= reorder_threshold`:

```python
alerts = []
for product in products:
    if product.current_stock <= product.reorder_threshold:
        alerts.append({
            'type': 'low_stock',
            'product_id': product.id,
            'current_stock': product.current_stock,
            'threshold': product.reorder_threshold
        })
```

### Dead Stock Detection

Products with no sales in the last N days (default: 90 days):

```python
dead_stock_period = 90  # days
cutoff_date = datetime.now() - timedelta(days=dead_stock_period)

dead_products = db.query(Product).filter(
    ~Product.orders.any(Order.order_date >= cutoff_date)
).all()
```

## ABC Analysis

### Revenue-based Classification

1. Calculate total revenue per product
2. Sort products by revenue descending
3. Apply Pareto principle:
   - **A items**: Top 20% of products, ~80% of revenue
   - **B items**: Next 30% of products, ~15% of revenue
   - **C items**: Bottom 50% of products, ~5% of revenue

```python
def perform_abc_analysis(products_data):
    total_revenue = sum(p['revenue'] for p in products_data)
    cumulative_revenue = 0

    for product in sorted(products_data, key=lambda x: x['revenue'], reverse=True):
        cumulative_revenue += product['revenue']
        percentage = (cumulative_revenue / total_revenue) * 100

        if percentage <= 80:
            product['class'] = 'A'
        elif percentage <= 95:
            product['class'] = 'B'
        else:
            product['class'] = 'C'
```

## Demand Forecasting

### Simple Moving Average

For short-term forecasting using recent sales data:

```python
def simple_forecast(product_id, days_ahead=30, lookback_days=90):
    # Get sales data for the last lookback_days
    sales_data = get_product_sales(product_id, lookback_days)

    if not sales_data:
        return 0

    # Calculate average daily demand
    total_quantity = sum(sale['quantity'] for sale in sales_data)
    avg_daily_demand = total_quantity / len(sales_data)

    # Forecast for days_ahead
    return avg_daily_demand * days_ahead
```

### Seasonal Adjustment

Consider hourly/daily/weekly patterns:

```python
# Group sales by hour/day/week
hourly_patterns = df.groupby(df['order_date'].dt.hour)['quantity'].mean()
daily_patterns = df.groupby(df['order_date'].dt.dayofweek)['quantity'].mean()
```

## Inventory Optimization

### Reorder Point Calculation

```
Reorder Point = (Average Daily Demand × Lead Time) + Safety Stock
```

Where:
- **Lead Time**: Time to receive new stock (configurable)
- **Safety Stock**: Buffer for demand variability

### Economic Order Quantity (EOQ)

```
EOQ = √(2 × Annual Demand × Ordering Cost / Holding Cost)
```

## Data Quality Checks

### Stock Level Validation

- Ensure stock levels never go negative
- Validate reorder thresholds are reasonable
- Check for data consistency across related tables

### Order Validation

- Verify product exists before order creation
- Check sufficient stock before order fulfillment
- Validate customer and seller references

## Performance Optimization

### Database Indexing

Key fields for indexing:
- `products.id` (primary key)
- `orders.product_id` (foreign key)
- `orders.order_date` (time-based queries)
- `orders.customer_id` (customer analytics)

### Query Optimization

- Use `selectinload` for related data fetching
- Implement pagination for large result sets
- Cache frequently accessed analytics data

## Monitoring and Alerts

### Real-time Metrics

- Current stock levels across all products
- Daily/weekly/monthly sales trends
- Alert counts (low stock, dead stock)
- System performance metrics

### Automated Reports

- Daily inventory summary
- Weekly sales and stock reports
- Monthly ABC analysis updates
- Quarterly forecasting accuracy assessment