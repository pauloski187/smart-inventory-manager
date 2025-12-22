# Smart Inventory Manager - Architecture

## Overview

The Smart Inventory Manager is a full-stack application designed for small businesses to manage inventory, track sales, and optimize stock levels using data-driven insights.

## System Architecture

### Backend Architecture (FastAPI)

```
backend/
├── main.py                 # Application entry point
├── config.py              # Configuration management
├── database.py            # Database connection and session management
├── models/                # SQLAlchemy ORM models
│   ├── product.py
│   ├── customer.py
│   ├── order.py
│   ├── seller.py
│   └── inventory_movement.py
├── schemas/               # Pydantic schemas for API
│   ├── product.py
│   ├── order.py
│   ├── customer.py
│   └── analytics.py
├── routes/                # API route handlers
│   ├── products.py
│   ├── orders.py
│   ├── customers.py
│   ├── sellers.py
│   ├── analytics.py
│   └── auth.py
├── services/              # Business logic layer
│   └── inventory_service.py
├── middleware/            # Authentication middleware
├── utils/                 # Utility functions
│   └── data_loader.py
└── requirements.txt
```

### ML/Analytics Architecture

```
ml/
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01_data_overview.ipynb
│   ├── 02_inventory_eda.ipynb
│   └── ...
├── data/                  # Data storage
│   ├── raw/              # Raw input data
│   ├── processed/        # Cleaned data
│   └── features/         # Feature engineering outputs
├── models/               # Trained ML models
└── pipelines/            # ML pipelines and scripts
```

## Data Flow

1. **Data Ingestion**: CSV files loaded via `data_loader.py`
2. **Data Processing**: Normalization into relational database
3. **API Layer**: RESTful endpoints for frontend consumption
4. **Analytics**: Real-time calculations and ML-based forecasting
5. **Frontend**: Loveable AI-generated UI consuming backend APIs

## Database Schema

### Core Entities

- **Products**: Inventory items with stock levels and pricing
- **Customers**: Buyer information and order history
- **Orders**: Transaction records with line items
- **Sellers**: Supplier/vendor information
- **Inventory Movements**: Stock change tracking

### Relationships

```
Product 1:N Order
Customer 1:N Order
Seller 1:N Order
Product 1:N InventoryMovement
```

## API Design

### RESTful Endpoints

- `GET /products` - List products with filtering/pagination
- `POST /orders` - Create new orders (updates inventory)
- `GET /analytics/inventory-alerts` - Low stock alerts
- `GET /analytics/sales-trends` - Sales analytics
- `GET /analytics/abc-analysis` - Product classification
- `GET /analytics/demand-forecast/{product_id}` - Demand prediction

### Response Format

```json
{
  "data": [...],
  "meta": {
    "total": 100,
    "page": 1,
    "limit": 20
  }
}
```

## Analytics Features

### Inventory Optimization

- **Low Stock Alerts**: Automatic notifications when stock < reorder_threshold
- **Dead Stock Detection**: Products with no sales in configurable period
- **ABC Analysis**: Pareto classification (A=80% revenue, B=15%, C=5%)

### Demand Forecasting

- **Simple Moving Average**: Historical average for short-term forecasting
- **Seasonal Trends**: Time-based demand patterns
- **Product-specific Models**: Individual forecasting per product

## Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS configuration for frontend integration
- Input validation with Pydantic

## Deployment

- **Backend**: FastAPI with Uvicorn server
- **Database**: SQLite (development) / PostgreSQL (production)
- **ML Models**: Pickle serialization for scikit-learn models
- **Containerization**: Docker support planned

## Scalability Considerations

- Database indexing on frequently queried fields
- Pagination for large datasets
- Background job processing for heavy analytics
- Caching layer for frequently accessed data