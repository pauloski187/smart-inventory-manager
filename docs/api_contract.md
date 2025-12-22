# API Contract - Smart Inventory Manager

## Base URL
```
http://localhost:8000  (development)
https://api.smartinventory.com  (production)
```

## Authentication
All endpoints except `/auth/token` require JWT authentication:
```
Authorization: Bearer <jwt_token>
```

## Response Format
All responses follow this structure:
```json
{
  "data": <response_data>,
  "message": "Optional success/error message",
  "errors": ["Optional error details"]
}
```

## Error Codes
- `200`: Success
- `400`: Bad Request
- `401`: Unauthorized
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

## Endpoints

### Authentication

#### POST /auth/token
Login and get access token.

**Request:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "token_type": "bearer"
}
```

### Products

#### GET /products
List products with optional filtering.

**Query Parameters:**
- `skip`: int (default: 0)
- `limit`: int (default: 100)
- `category`: string
- `brand`: string
- `low_stock`: boolean

**Response:**
```json
{
  "data": [
    {
      "id": "string",
      "name": "string",
      "category": "string",
      "brand": "string",
      "unit_price": 0.0,
      "current_stock": 0,
      "reorder_threshold": 0,
      "is_active": true,
      "created_at": "datetime",
      "updated_at": "datetime"
    }
  ]
}
```

#### POST /products
Create a new product.

**Request:**
```json
{
  "id": "string",
  "name": "string",
  "category": "string",
  "brand": "string",
  "unit_price": 0.0,
  "current_stock": 0,
  "reorder_threshold": 10
}
```

#### GET /products/{product_id}
Get product details.

#### PUT /products/{product_id}
Update product information.

### Orders

#### GET /orders
List orders with pagination.

**Query Parameters:**
- `skip`: int
- `limit`: int
- `customer_id`: string
- `product_id`: string
- `start_date`: date
- `end_date`: date

#### POST /orders
Create a new order (automatically updates inventory).

**Request:**
```json
{
  "id": "string",
  "order_date": "datetime",
  "customer_id": "string",
  "product_id": "string",
  "seller_id": "string",
  "quantity": 1,
  "unit_price": 0.0,
  "discount": 0.0,
  "tax": 0.0,
  "shipping_cost": 0.0,
  "total_amount": 0.0,
  "payment_method": "string",
  "order_status": "string",
  "city": "string",
  "state": "string",
  "country": "string"
}
```

### Analytics

#### GET /analytics/inventory-alerts
Get current inventory alerts.

**Response:**
```json
{
  "data": [
    {
      "product_id": "string",
      "product_name": "string",
      "current_stock": 0,
      "reorder_threshold": 0,
      "alert_type": "low_stock"
    }
  ]
}
```

#### GET /analytics/sales-trends
Get sales trends over time.

**Query Parameters:**
- `days`: int (default: 30)

**Response:**
```json
{
  "data": [
    {
      "period": "2024-01-01",
      "total_sales": 0.0,
      "total_orders": 0,
      "avg_order_value": 0.0
    }
  ]
}
```

#### GET /analytics/abc-analysis
Get ABC analysis for products.

**Response:**
```json
{
  "data": [
    {
      "product_id": "string",
      "product_name": "string",
      "category": "string",
      "total_revenue": 0.0,
      "total_quantity": 0,
      "abc_class": "A"
    }
  ]
}
```

#### GET /analytics/demand-forecast/{product_id}
Get demand forecast for a product.

**Query Parameters:**
- `days_ahead`: int (default: 30)

**Response:**
```json
{
  "data": {
    "product_id": "string",
    "forecast_period": "next_30_days",
    "predicted_demand": 0.0
  }
}
```

#### GET /analytics/dashboard
Get comprehensive dashboard data.

**Response:**
```json
{
  "data": {
    "sales_trends": [...],
    "inventory_alerts": [...],
    "abc_analysis": [...],
    "demand_forecasts": [...]
  }
}
```

### Customers

#### GET /customers
List customers.

#### GET /customers/{customer_id}
Get customer details with order history.

### Sellers

#### GET /sellers
List sellers.

#### GET /sellers/{seller_id}
Get seller details.

## Rate Limiting
- 100 requests per minute per IP
- 1000 requests per hour per authenticated user

## Pagination
All list endpoints support pagination:
```json
{
  "data": [...],
  "pagination": {
    "skip": 0,
    "limit": 100,
    "total": 1000
  }
}
```

## Data Validation
- All inputs validated using Pydantic models
- Date formats: ISO 8601
- Currency values: 2 decimal places
- String lengths: Reasonable limits applied

## Versioning
API version included in URL path: `/v1/`
Future versions will be `/v2/`, etc.