---
title: Smart Inventory Manager API
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
license: mit
short_description: Inventory management API with analytics & forecasting
---

# Smart Inventory Manager - Full REST API

A comprehensive inventory management REST API with analytics, forecasting, and business intelligence features.

**Live on Hugging Face Spaces** for seamless integration with frontend applications built on Lovable AI.

## 🚀 Features

- **Dashboard Analytics**: Real-time KPIs and business metrics
- **Sales Trends**: Monthly revenue and profit analysis
- **Product Performance**: Best/worst performing products
- **Category Analysis**: Category-wise performance metrics
- **ABC Classification**: Inventory prioritization using ABC analysis
- **Inventory Alerts**: Low stock and dead stock detection
- **Demand Forecasting**: Time-series based demand predictions
- **Inventory Recommendations**: Automated reorder points and safety stock

## 📡 API Endpoints

### Analytics
- `GET /analytics/dashboard/summary` - Dashboard KPIs
- `GET /analytics/monthly-sales-trend?months=12` - Monthly trends
- `GET /analytics/monthly-report/{year}/{month}` - Detailed monthly report
- `GET /analytics/product-performance?limit=10` - Product performance
- `GET /analytics/category-performance` - Category metrics
- `GET /analytics/abc-analysis?limit=100` - ABC classification
- `GET /analytics/inventory/low-stock` - Low stock alerts
- `GET /analytics/inventory/dead-stock` - Dead stock items
- `GET /analytics/sales-by-day-of-week` - Weekly patterns
- `GET /analytics/available-months` - Available report months

### Forecasting
- `GET /forecast/forecasts/all` - All category forecasts
- `GET /forecast/forecast/{category}?include_daily=true` - Single category forecast
- `GET /forecast/inventory-recommendations` - Reorder recommendations

### Data
- `GET /products?limit=100` - List products
- `GET /orders?limit=100` - List orders
- `GET /health` - Health check

## 🔌 Integration with Lovable AI

This API is designed to work seamlessly with Lovable AI frontends.

### Quick Start

1. **API Base URL**:
   ```
   https://huggingface.co/spaces/pauloski07/sales-inventory-forecasting
   ```

2. **CORS**: Enabled for all origins

3. **No Authentication Required**: Open API for easy integration

### Example Usage

```javascript
// JavaScript/TypeScript example
const API_BASE = 'https://pauloski07-sales-inventory-forecasting.hf.space';

// Get dashboard summary
const response = await fetch(`${API_BASE}/analytics/dashboard/summary`);
const data = await response.json();

// Get monthly sales trend
const trends = await fetch(`${API_BASE}/analytics/monthly-sales-trend?months=12`);
const trendData = await trends.json();

// Get forecast for a category
const forecast = await fetch(`${API_BASE}/forecast/forecast/Electronics?include_daily=true`);
const forecastData = await forecast.json();
```

### Python Example

```python
import requests

API_BASE = "https://pauloski07-sales-inventory-forecasting.hf.space"

# Get dashboard summary
response = requests.get(f"{API_BASE}/analytics/dashboard/summary")
data = response.json()

# Get inventory recommendations
recommendations = requests.get(f"{API_BASE}/forecast/inventory-recommendations")
rec_data = recommendations.json()
```

## 📊 Data

The API comes pre-loaded with sample e-commerce data including:
- 100k+ order records
- 10+ product categories
- Multiple years of historical data
- Realistic sales patterns

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **Database**: SQLite (embedded)
- **Data Processing**: Pandas, NumPy
- **Analytics**: SciPy, Scikit-learn
- **Deployment**: Hugging Face Spaces

## 📖 API Documentation

Interactive API documentation available at:
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

## 🔧 Configuration

The API runs on port 7860 (HF Spaces default) and uses SQLite for data persistence.

## 🎯 Use Cases

Perfect for:
- E-commerce inventory management dashboards
- Demand forecasting applications
- Business intelligence tools
- Supply chain optimization platforms
- Retail analytics solutions

## 📝 License

MIT License - Free to use for commercial and personal projects

## 🤝 Contributing

Built for the community. Feel free to fork and customize for your needs.

---

**Need help?** Check the `/docs` endpoint for complete API documentation.

**Frontend Integration?** Use this API with Lovable AI to build beautiful dashboards in minutes!
