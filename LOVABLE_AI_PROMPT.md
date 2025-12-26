# Lovable AI Frontend Prompt

## Smart Inventory Manager Dashboard

---

## Project Overview

Build a modern, responsive dashboard for a Smart Inventory Management System that connects to an existing FastAPI backend. The system provides **Prophet-based demand forecasting** with **18.35% SMAPE** (verified), ABC analysis, comprehensive reporting, and inventory recommendations.

**Backend API Base URL**: `https://smart-inventory-api.onrender.com` (Production)
**Local Development**: `http://localhost:8000`
**API Documentation**: `https://smart-inventory-api.onrender.com/docs`

### Key ML Achievement
| Metric | Value |
|--------|-------|
| **Forecast Accuracy (SMAPE)** | **18.35%** ✅ |
| Target | <20% |
| Model | Facebook Prophet |
| Validation | 8-week holdout |

---

## WHAT TO BUILD

### 1. Dashboard Home Page (`/`)
Create a KPI dashboard with:
- **Revenue MTD/YTD cards** - Display total revenue with trend indicator
- **Total Orders count** - Number of completed orders
- **Profit Margin** - Percentage with color coding (green >20%, yellow 10-20%, red <10%)
- **Low Stock Alerts** - Count with warning styling
- **Dead Stock Value** - Amount tied up in non-moving inventory
- **Top 5 Products chart** - Bar chart showing revenue by product
- **Quick links** to reports and alerts

**API Endpoint**: `GET /analytics/dashboard/summary`

---

### 2. Monthly Sales Trend Page (`/sales-trends`) - NEW
**Primary Visualization - Line Chart**:
- Dual-axis line chart showing:
  - Revenue (left Y-axis, blue line)
  - Profit/Loss (right Y-axis, green line for profit, red for loss)
- X-axis: Months
- Highlight loss months in red
- Show month-over-month growth percentages
- Toggle between 6/12/24 month views

**Additional Charts**:
- **Revenue vs Profit Comparison** - Dual bar chart
- **Profit Margin Trend** - Area chart showing margin % over time
- **Month-over-Month Growth** - Waterfall chart

**Summary Cards**:
- Total Revenue (period)
- Total Profit
- Average Monthly Revenue
- Average Profit Margin %

**API Endpoint**: `GET /analytics/monthly-sales-trend?months=12`

---

### 3. Monthly Reports Page (`/reports`) - NEW
**Month Selector**:
- Dropdown with available months (fetched from API)
- Quick navigation: Previous/Next month buttons

**Report Dashboard for Selected Month**:

**Summary Cards**:
- Total Revenue with comparison to previous month (arrow up green / arrow down red)
- Total Profit/Loss
- Total Orders
- Average Order Value
- Units Sold

**Comparison Section**:
- Revenue change amount and percentage
- Profit change amount and percentage
- Visual indicator (up/down arrows with colors)

**Top 10 Products Table** (best performers):
- Product Name, Category, Revenue, Profit, Units Sold
- Sortable columns
- Bar visualization in revenue column

**Bottom 10 Products Table** (worst performers):
- Same columns as top products
- Flag for products needing attention

**Category Breakdown**:
- Pie chart showing revenue share by category
- Table with category metrics

**Daily Breakdown Chart**:
- Line chart showing daily revenue/profit for the month
- Identify peak and low days

**Recommendations Section**:
- Display auto-generated recommendations
- Color-coded by type: critical (red), warning (yellow), success (green), info (blue)

**Export Button**: Download report as PDF

**API Endpoints**:
- `GET /analytics/available-months` - Get list of months with data
- `GET /analytics/monthly-report/{year}/{month}` - Get full report

---

### 4. Product Performance Page (`/products/performance`) - NEW
**Best Performers Section**:
- **By Revenue** - Bar chart + table (top 10)
- **By Profit Margin** - Bar chart + table (top 10)
- Highlight star performers with badges

**Worst Performers Section**:
- **By Revenue** - Bar chart + table (bottom 10)
- **By Profit Margin** - Bar chart + table (lowest margin)
- Warning indicators for products needing attention

**Recommendations Panel**:
- Best performers: Inventory tips
- Worst performers: Action suggestions
- Margin optimization advice

**API Endpoint**: `GET /analytics/product-performance?limit=10`

---

### 5. Category Performance Page (`/categories`) - NEW
**Visualizations**:
- **Revenue by Category** - Pie chart with percentages
- **Profit by Category** - Pie chart
- **Revenue Share Bar** - Horizontal stacked bar

**Category Table**:
- Category name
- Revenue (with % share)
- Profit (with margin %)
- Units Sold
- Order Count
- Average Order Value

**Insights**:
- Top performing category highlight
- Categories below average margin flagged

**API Endpoint**: `GET /analytics/category-performance`

---

### 6. Sales Patterns Page (`/patterns`) - NEW
**Day of Week Analysis**:
- Bar chart showing revenue by day of week
- Highlight busiest and slowest days
- Show average order value per day

**Use Cases**:
- Staffing recommendations
- Promotion timing suggestions

**API Endpoint**: `GET /analytics/sales-by-day-of-week`

---

### 7. Demand Forecast Page (`/forecast`)
- **Category selector dropdown** - List all product categories
- **Forecast display** showing:
  - 30/60/90 day total forecast
  - Daily forecast line chart with confidence intervals (shaded area)
  - Reorder point indicator
  - Safety stock level
  - Stockout risk badge (low=green, medium=yellow, high=red)

**API Endpoints**:
- `GET /forecast/forecasts/all` - Get all category summaries
- `GET /forecast/forecast/{category}?include_daily=true` - Get detailed forecast

---

### 8. ABC Analysis Page (`/abc-analysis`)
- **Summary cards** for Class A, B, C showing:
  - Product count
  - Revenue percentage
  - Recommended actions
- **Product table** with columns:
  - Product ID, Name, Category, Revenue, Profit, ABC Class
  - Sortable and filterable
- **Pie chart** showing revenue distribution by class

**API Endpoint**: `GET /analytics/abc-analysis?limit=100`

---

### 9. Inventory Alerts Page (`/inventory`)
**Low Stock Tab**:
- Table of products below reorder threshold
- Columns: Product, Category, Current Stock, Reorder Point, Priority
- Priority badge (High=red, Medium=yellow)
- Quick reorder button

**Dead Stock Tab**:
- Products with no sales in 90+ days
- Show days since last sale
- Inventory value at risk
- Recommendation action

**API Endpoints**:
- `GET /analytics/inventory/low-stock`
- `GET /analytics/inventory/dead-stock`

---

### 10. Inventory Recommendations Page (`/recommendations`)
- Card grid showing each category with:
  - 90-day forecast
  - Confidence interval range
  - Reorder point
  - Safety stock
  - Stockout risk indicator
  - Recommended order quantity

**API Endpoint**: `GET /forecast/inventory-recommendations`

---

### 11. Data Upload Page (`/upload`)
- File upload component for CSV
- Upload progress indicator
- Validation result display
- Success message with:
  - Records processed
  - Categories found
  - Date range
- Button to trigger model retraining

**API Endpoints**:
- `POST /forecast/upload-data` - Upload CSV
- `POST /forecast/retrain-models` - Retrain models
- `GET /forecast/model-status` - Check model status

---

### 12. Navigation
**Sidebar navigation** with icons and grouping:

**Overview**
- Dashboard (home icon)

**Analytics** - NEW SECTION
- Sales Trends (line chart icon)
- Monthly Reports (calendar icon)
- Product Performance (trophy icon)
- Category Performance (pie chart icon)
- Sales Patterns (bar chart icon)

**Inventory**
- Forecast (trending icon)
- ABC Analysis (layers icon)
- Alerts (bell icon)
- Recommendations (lightbulb icon)

**Settings**
- Upload Data (upload icon)

Features:
- Responsive: collapse to hamburger on mobile
- Active state highlighting
- Collapsible sidebar option

---

## CHART SPECIFICATIONS

### Monthly Sales Trend (Line Chart)
```
Type: Multi-line with dual Y-axis
Data: revenue, profit (from /analytics/monthly-sales-trend)
X-axis: month (format: "Jan 2024")
Left Y-axis: Revenue ($)
Right Y-axis: Profit ($)
Colors:
  - Revenue: #3B82F6 (blue)
  - Profit (positive): #10B981 (green)
  - Profit (negative): #EF4444 (red)
Features:
  - Tooltips with full values
  - Loss months highlighted
  - Growth indicators on hover
```

### Revenue vs Profit Bar Chart
```
Type: Grouped bar chart
Data: monthly_trends from /analytics/revenue-vs-profit-trend
Bars: Revenue (blue) | Profit (green/red)
Features:
  - Clickable to drill into month
```

### Category Performance Pie Chart
```
Type: Donut chart
Data: categories from /analytics/category-performance
Values: revenue_share
Colors: Distinct color per category
Features:
  - Legend with values
  - Center text: Total Revenue
```

### Product Performance Bar Chart
```
Type: Horizontal bar chart
Data: best_by_revenue from /analytics/product-performance
X-axis: Revenue ($)
Y-axis: Product names
Features:
  - Profit margin shown as secondary metric
  - Color intensity based on margin
```

### Day of Week Analysis
```
Type: Vertical bar chart
Data: daily_distribution from /analytics/sales-by-day-of-week
X-axis: Day names
Y-axis: Revenue
Features:
  - Average line
  - Highlight max/min
```

---

## DESIGN REQUIREMENTS

### Color Scheme
| Purpose | Color | Hex |
|---------|-------|-----|
| Primary | Blue | `#3B82F6` |
| Success/Profit/Low Risk | Green | `#10B981` |
| Warning/Medium Risk | Amber | `#F59E0B` |
| Danger/Loss/High Risk | Red | `#EF4444` |
| Info | Indigo | `#6366F1` |
| Background | Light gray | `#F9FAFB` |
| Card Background | White | `#FFFFFF` |
| Text Primary | Dark gray | `#111827` |
| Text Secondary | Gray | `#6B7280` |

### Typography
- Headings: Inter or system font, bold
- Body: Regular weight, readable size (16px base)
- Numbers/Data: Tabular numerals for alignment
- Metrics: Large, bold for emphasis

### Components
- Use shadcn/ui or Radix UI components
- Cards with subtle shadows
- Tables with hover states and sticky headers
- Charts using Recharts or Chart.js
- Loading skeletons while fetching data
- Toast notifications for actions
- Breadcrumbs for navigation

### Responsive Design
- Desktop: Full sidebar, multi-column layouts, detailed tables
- Tablet: Collapsible sidebar, 2-column grids
- Mobile: Bottom navigation, single column, swipeable cards, simplified charts

---

## WHAT NOT TO DO

### Technical Don'ts
- **DON'T** implement authentication (not required for MVP)
- **DON'T** create a database - use the API
- **DON'T** implement the forecasting logic - it's in the backend
- **DON'T** hardcode data - always fetch from API
- **DON'T** use deprecated dependencies
- **DON'T** skip error handling for API calls

### Design Don'ts
- **DON'T** use heavy animations that slow the UI
- **DON'T** use more than 3 colors for risk levels
- **DON'T** hide important data behind modals
- **DON'T** make charts too small to read
- **DON'T** forget loading and error states
- **DON'T** use inconsistent spacing

### UX Don'ts
- **DON'T** require many clicks to see key data
- **DON'T** auto-refresh without user control
- **DON'T** delete data without confirmation
- **DON'T** show raw error messages to users
- **DON'T** skip empty state designs

---

## API RESPONSE EXAMPLES

### Monthly Sales Trend - NEW
```json
GET /analytics/monthly-sales-trend?months=6

{
  "monthly_trends": [
    {
      "month": "2024-07",
      "revenue": 425000.50,
      "profit": 85000.25,
      "loss": 0,
      "profit_margin": 20.0,
      "order_count": 2500,
      "units_sold": 8500,
      "avg_order_value": 170.00,
      "revenue_growth_pct": 5.2,
      "profit_growth_pct": 8.1
    }
  ],
  "summary": {
    "total_revenue": 2500000.00,
    "total_profit": 500000.00,
    "avg_monthly_revenue": 416666.67,
    "avg_profit_margin": 20.0,
    "months_analyzed": 6
  }
}
```

### Monthly Report - NEW
```json
GET /analytics/monthly-report/2024/6

{
  "report_period": {
    "year": 2024,
    "month": 6,
    "month_name": "June",
    "start_date": "2024-06-01",
    "end_date": "2024-06-30"
  },
  "summary": {
    "total_revenue": 394353.54,
    "total_profit": 78870.71,
    "total_loss": 0,
    "profit_margin": 20.0,
    "total_orders": 2150,
    "total_units_sold": 7500,
    "avg_order_value": 183.42
  },
  "comparison": {
    "revenue_change": 15000.00,
    "revenue_growth_pct": 3.95,
    "profit_change": 5000.00,
    "profit_growth_pct": 6.77,
    "prev_month_revenue": 379353.54,
    "prev_month_profit": 73870.71
  },
  "top_products": [
    {
      "product_id": "P001",
      "product_name": "Premium Yoga Mat",
      "category": "Sports & Fitness",
      "revenue": 12500.00,
      "profit": 4500.00,
      "units_sold": 250
    }
  ],
  "worst_products": [...],
  "category_breakdown": [
    {
      "category": "Sports & Fitness",
      "revenue": 85000.00,
      "profit": 17000.00,
      "orders": 450,
      "revenue_share": 21.5
    }
  ],
  "daily_breakdown": [
    {
      "date": "2024-06-01",
      "revenue": 12500.00,
      "profit": 2500.00,
      "orders": 75
    }
  ],
  "recommendations": [
    {"type": "success", "message": "Strong revenue growth of 3.95%. Consider increasing inventory for top performers."},
    {"type": "info", "message": "Top performer: Premium Yoga Mat generated $12,500 revenue."}
  ]
}
```

### Product Performance - NEW
```json
GET /analytics/product-performance?limit=10

{
  "best_by_revenue": [
    {
      "product_id": "P001",
      "product_name": "Premium Yoga Mat",
      "category": "Sports & Fitness",
      "revenue": 125000.00,
      "profit": 45000.00,
      "units_sold": 2500,
      "profit_margin": 36.0,
      "order_count": 1850
    }
  ],
  "worst_by_revenue": [...],
  "best_by_margin": [...],
  "worst_by_margin": [...],
  "recommendations": {
    "best_performers": [
      "Increase inventory for top revenue generators to avoid stockouts",
      "Consider bundling top products with slower movers",
      "Analyze what makes these products successful"
    ],
    "worst_performers": [
      "Review pricing strategy for low performers",
      "Consider promotional campaigns or discounts",
      "Evaluate if products should be discontinued"
    ],
    "margin_optimization": [
      "Products with high margin but low volume may benefit from promotion",
      "Low margin products may need cost reduction or price increase"
    ]
  }
}
```

### Category Performance - NEW
```json
GET /analytics/category-performance

{
  "categories": [
    {
      "category": "Sports & Fitness",
      "revenue": 2500000.00,
      "profit": 500000.00,
      "units_sold": 45000,
      "order_count": 25000,
      "product_count": 450,
      "revenue_share": 22.5,
      "profit_share": 25.0,
      "profit_margin": 20.0,
      "avg_order_value": 100.00
    }
  ],
  "totals": {
    "total_revenue": 11000000.00,
    "total_profit": 2000000.00,
    "total_categories": 10
  }
}
```

### Sales by Day of Week - NEW
```json
GET /analytics/sales-by-day-of-week

{
  "daily_distribution": [
    {
      "day": "Sunday",
      "day_number": 0,
      "revenue": 1200000.00,
      "profit": 240000.00,
      "order_count": 10000,
      "avg_order_value": 120.00
    },
    {
      "day": "Monday",
      "day_number": 1,
      "revenue": 1500000.00,
      "profit": 300000.00,
      "order_count": 12500,
      "avg_order_value": 120.00
    }
  ]
}
```

### Available Months for Reports - NEW
```json
GET /analytics/available-months

{
  "available_months": [
    {
      "year": 2024,
      "month": 12,
      "month_name": "December",
      "display": "December 2024",
      "order_count": 2500,
      "revenue": 425000.00
    }
  ]
}
```

### Dashboard Summary
```json
GET /analytics/dashboard/summary

{
  "total_revenue_mtd": 125000.50,
  "total_revenue_ytd": 1500000.00,
  "total_orders": 89109,
  "avg_order_value": 156.75,
  "low_stock_alerts": 42,
  "dead_stock_value": 15000.00,
  "top_products": [
    {"product_id": "P001", "product_name": "Yoga Mat", "revenue": 45000.00}
  ]
}
```

### Category Forecast
```json
GET /forecast/forecast/Sports%20%26%20Fitness?include_daily=true

{
  "category": "Sports & Fitness",
  "forecast_90_day": 3357,
  "confidence_interval": {
    "lower": 1117,
    "upper": 5597
  },
  "daily_forecast": [
    {"date": "2025-01-01", "forecast": 37.5, "lower_ci": 12.5, "upper_ci": 62.5}
  ],
  "reorder_point": 450,
  "safety_stock": 120,
  "stockout_risk": "low"
}
```

---

## TECHNICAL NOTES

### CORS
The backend has CORS enabled for all origins. No proxy needed.

### Error Handling
All API errors return:
```json
{
  "detail": "Error message here"
}
```
Display user-friendly error messages, not raw API errors.

### Data Refresh
- Dashboard: Auto-refresh every 5 minutes (optional toggle)
- Reports: On-demand refresh button
- Forecast: Refresh on category change
- Inventory: Manual refresh button

### Date Formatting
- Display dates as: "Jan 15, 2024" or "January 2024"
- API returns: "2024-01-15" or "2024-01"

### Number Formatting
- Currency: $1,234.56
- Percentages: 25.5%
- Large numbers: 1.2M or 12K

---

## PRIORITY ORDER

Build in this order for best workflow:

**Phase 1 - Core**
1. Navigation and layout
2. Dashboard home page

**Phase 2 - Analytics (NEW)**
3. Monthly Sales Trend (primary visualization)
4. Monthly Reports page
5. Product Performance

**Phase 3 - Inventory**
6. Inventory alerts
7. Forecast page
8. Recommendations

**Phase 4 - Additional**
9. ABC analysis
10. Category Performance
11. Sales Patterns
12. Data upload

---

## SUMMARY OF ALL ENDPOINTS

### New Endpoints
| Endpoint | Description | Chart Type |
|----------|-------------|------------|
| `GET /analytics/monthly-sales-trend` | Revenue & profit by month | Line chart |
| `GET /analytics/monthly-report/{year}/{month}` | Full monthly report | Multiple |
| `GET /analytics/product-performance` | Best/worst products | Bar charts |
| `GET /analytics/category-performance` | Category metrics | Pie chart |
| `GET /analytics/sales-by-day-of-week` | Weekly patterns | Bar chart |
| `GET /analytics/revenue-vs-profit-trend` | R&P comparison | Dual line |
| `GET /analytics/available-months` | Report month list | Dropdown |

### Existing Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /analytics/dashboard/summary` | KPI dashboard |
| `GET /analytics/abc-analysis` | ABC classification |
| `GET /analytics/inventory/low-stock` | Low stock alerts |
| `GET /analytics/inventory/dead-stock` | Dead stock detection |
| `GET /forecast/forecasts/all` | All category forecasts |
| `GET /forecast/forecast/{category}` | Single category forecast |
| `GET /forecast/inventory-recommendations` | Reorder recommendations |
| `POST /forecast/upload-data` | Upload sales data |
| `POST /forecast/retrain-models` | Retrain SARIMA models |

---

## REAL-TIME STREAMING (NEW - v2.1.0)

### WebSocket Connection
The backend now supports real-time updates via WebSocket connections.

**WebSocket Endpoint**: `ws://localhost:8000/stream/ws`

### Available Channels
Subscribe to specific event channels:
- `all` - Receive all events (default)
- `orders` - Order-related events
- `stock` - Stock level changes
- `alerts` - Alert notifications
- `forecasts` - Forecast updates
- `analytics` - Analytics summaries

### Connection Example
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/stream/ws?channels=orders,alerts&client_id=dashboard');

ws.onopen = () => {
  console.log('Connected to inventory stream');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'connected':
      console.log('Subscribed to channels:', data.channels);
      break;
    case 'event':
      handleInventoryEvent(data);
      break;
    case 'alert':
      showAlertNotification(data);
      break;
    case 'stock_update':
      updateStockDisplay(data);
      break;
  }
};

// Subscribe to additional channel
ws.send(JSON.stringify({ action: 'subscribe', channel: 'stock' }));

// Unsubscribe from channel
ws.send(JSON.stringify({ action: 'unsubscribe', channel: 'forecasts' }));
```

### Event Types
| Event | Description |
|-------|-------------|
| `order.created` | New order placed |
| `order.delivered` | Order delivered |
| `stock.updated` | Stock level changed |
| `stock.low` | Stock fell below threshold |
| `stock.out` | Product out of stock |
| `alert.low_stock` | Low stock warning |
| `alert.dead_stock` | Dead stock detected |
| `forecast.generated` | New forecast available |

### Streaming Endpoints
| Endpoint | Description |
|----------|-------------|
| `WS /stream/ws` | WebSocket for real-time events |
| `GET /stream/events` | Server-Sent Events (SSE alternative) |
| `GET /stream/status` | Connection status and stats |
| `POST /stream/broadcast/alert` | Broadcast alert to clients |
| `POST /stream/test/emit-event` | Test event emission |

### Dashboard Real-time Features
- **Live stock updates** - Show stock changes as they happen
- **Alert notifications** - Toast notifications for critical alerts
- **Order feed** - Real-time order activity stream
- **Connection indicator** - Show WebSocket connection status

### Implementation Notes
- Reconnect automatically on disconnect (exponential backoff)
- Buffer messages during reconnection
- Show connection status indicator in UI
- Handle both WebSocket and SSE for fallback

---

---

## FORECAST ACCURACY BY CATEGORY

The Prophet model achieves the following verified SMAPE scores:

| Category | SMAPE | Status |
|----------|-------|--------|
| Health & Personal Care | 9.64% | ✅ Excellent |
| Toys & Games | 13.53% | ✅ Excellent |
| Electronics | 15.40% | ✅ Good |
| Office Products | 16.05% | ✅ Good |
| Sports & Fitness | 18.21% | ✅ Good |
| Tools & Home Improvement | 18.92% | ✅ Good |
| Books & Media | 20.00% | ⚠️ Acceptable |
| Home & Kitchen | 21.55% | ⚠️ Acceptable |
| Grocery & Gourmet Food | 25.01% | ⚠️ Moderate |
| Clothing & Fashion | 25.20% | ⚠️ Moderate |
| **AVERAGE** | **18.35%** | **✅ Target Met** |

Display these accuracy metrics in the forecast dashboard to build user confidence.

---

## PRODUCTION DEPLOYMENT

### Backend Deployment (Render.com)

The FastAPI backend is deployed on Render.com:

**Production URL**: `https://smart-inventory-api.onrender.com`

### Environment Variables (Render)
```
DATABASE_URL=postgresql://...
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Frontend Deployment (Lovable/Vercel)

Configure the frontend to use the production API:

```javascript
// config.js
const API_BASE_URL = process.env.NODE_ENV === 'production'
  ? 'https://smart-inventory-api.onrender.com'
  : 'http://localhost:8000';
```

### Health Check Endpoint
`GET /health` - Returns `{"status": "healthy"}` for monitoring

---

*Prompt updated: December 26, 2024*
*Version: 3.0.0*
*Key Update: Prophet model achieves 18.35% SMAPE (target <20% achieved!)*
*Features: Monthly trends, reports, product/category performance, recommendations, real-time streaming via Kafka/WebSocket*
