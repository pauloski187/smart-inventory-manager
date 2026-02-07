# Smart Inventory Manager - Complete Feature Overview

## What Is It?

Smart Inventory Manager is a full-stack, AI-powered inventory and sales analytics platform built for e-commerce and retail businesses. It combines real-time dashboards, machine learning demand forecasting, and intelligent inventory classification to help businesses make data-driven decisions about stock management, purchasing, and sales strategy.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 16, React 18, TypeScript, Tailwind CSS |
| **Backend** | FastAPI (Python), SQLAlchemy ORM, SQLite/PostgreSQL |
| **ML/AI** | SARIMA, Facebook Prophet, LSTM (Ensemble Forecasting) |
| **Charts** | Recharts (Line, Bar, Pie) |
| **Deployment** | Vercel (Frontend), HuggingFace Spaces (Backend API) |
| **Real-time** | WebSocket + Server-Sent Events (SSE) |

---

## Core Features

### 1. Executive Dashboard
- **4 KPI Cards**: Total Orders, Average Order Value, Profit Margin, Low Stock Alerts
- **Top 5 Products**: Ranked by order count with badge indicators
- **Quick Actions**: One-click navigation to trends, alerts, and product performance
- Real-time data pulled from backend API on each page load

### 2. Sales Trend Analysis
- **Interactive Line Chart**: Revenue and profit plotted over time
- **Time Range Selector**: View last 3, 6, 12, or 24 months
- **Summary Cards**: Total Revenue, Total Profit, Avg Monthly Revenue, Avg Profit Margin
- **Monthly Details Table**: Revenue, profit, margin, order count, and month-over-month growth percentage for every period

### 3. Monthly Reports
- **Month Selector Dropdown**: Choose any available reporting month
- **Report Header Card**: Displays selected period with date range
- **4 Financial KPIs**: Revenue, Profit, Avg Order Value, Total Loss
- **Daily Performance Chart**: Day-by-day revenue and profit line chart
- **Revenue by Category**: Pie chart showing category distribution
- **Category Breakdown Table**: Revenue, profit, and share per category
- **Top Performing Products**: Green-highlighted cards with revenue and units sold
- **Underperforming Products**: Red-highlighted cards flagging weak performers
- **AI-Generated Insights**: 6 types of auto-generated business recommendations:
  - Profit margin assessment
  - Revenue concentration risk analysis
  - Star product spotlight
  - Underperformer alerts
  - Order efficiency metrics
  - Loss detection and analysis

### 4. Product Performance
- **Top 10 Products by Revenue**: Ranked table with profit margins and badges
- **Bottom 10 Products by Revenue**: Identifies underperformers needing attention
- Category and margin breakdowns for each product

### 5. Category Performance
- **3 Summary Cards**: Total Revenue, Total Profit, Number of Categories
- **Revenue Share Pie Chart**: Visual breakdown of category contribution
- **Category Performance Cards**: Top 5 categories with order count and profit margin
- **Full Category Table**: All categories with revenue, profit, units, orders, margin, and market share

### 6. Sales Patterns (Day-of-Week Analysis)
- **3 Summary Cards**: Busiest Day, Slowest Day, Daily Average
- **Line Chart**: Revenue plotted across all 7 days of the week
- **Detailed Breakdown Table**: Revenue, profit, orders, avg order value, and % vs average for each day
- **3 Actionable Insights**:
  - Staffing recommendation (schedule more staff on peak days)
  - Promotion opportunity (run deals on slow days)
  - Inventory planning (stock up before high-demand days)

### 7. AI-Powered Demand Forecasting
- **Category Selector**: Choose any product category to forecast
- **Forecast Horizon**: Adjustable from 1 to 90 days
- **SARIMA Model**: Seasonal AutoRegressive Integrated Moving Average trained on historical data
- **4 Forecast KPIs**: Total Forecast (units), Daily Average, Reorder Point, Stockout Risk Level
- **Daily Forecast Chart**: Line chart showing predicted demand per day
- **95% Confidence Interval**: Lower bound, expected forecast, and upper bound with visual bars
- **Inventory Recommendations**:
  - Safety Stock level (buffer for demand variability)
  - Reorder Point (when to place new orders)
  - Daily Average demand
  - Stockout Risk assessment (Low / Medium / High) with color-coded alerts
- **Forecasting Insights**: Model explanation, action-required alerts for high-risk categories, and summary

### 8. ABC Analysis (Inventory Classification)
- **Pareto Principle Classification**:
  - **Class A** (Critical): Top 20% of products generating 80% of revenue
  - **Class B** (Moderate): Next 30% of products generating 15% of revenue
  - **Class C** (Low Priority): Remaining 50% generating 5% of revenue
- **3 Summary Cards**: Count per class with color-coded borders
- **Pie Chart**: Visual product distribution across A/B/C classes
- **Management Guidelines**: Specific inventory control recommendations per class
- **Filter Buttons**: Toggle between All, Class A, Class B, Class C
- **Product Classification Table**: Every product with its ABC class, revenue, profit, and margin

### 9. Inventory Alerts
- **4-Zone Color-Coded Alert System**:
  - **Red Zone** (Order Now): Out of stock items requiring immediate ordering
  - **Orange Zone** (Order This Week): Very low stock, urgent attention needed
  - **Yellow Zone** (Watch Closely): Below reorder point, monitor closely
  - **Green Zone** (Healthy): Stock levels are optimal
- **Critical Alerts Table**: Red-themed table with product, category, stock level, reorder point, and URGENT action tags
- **Urgent Alerts Table**: Orange-themed with current stock vs reorder point
- **Warning Alerts Table**: Yellow-themed for items below reorder threshold
- **Low Stock Summary**: Combined view of all items needing attention with priority badges
- **Dead Stock Detection**: Products with no sales in 90+ days, showing value at risk, days since last sale, and liquidation recommendations

### 10. Products Catalog
- **4 Summary Cards**: Total Products, Low Stock Items, Total Inventory Value, Avg Profit Margin
- **Search**: Filter by product name, category, or ID
- **Category Filter**: Dropdown to filter by any category
- **Stock Status Filter**: All / Low Stock / Normal Stock
- **Products Table**: ID, name, category, price, cost, margin (color-coded), stock level with threshold, and status badges (In Stock / Low Stock)
- **Quick Actions**: View Low Stock, Filter by Category, Reset All Filters

### 11. Orders History
- **4 Summary Cards**: Total Orders, Total Revenue, Total Profit, Categories Count
- **Search**: Filter by product name, category, or order ID
- **Category Filter**: Dropdown selection
- **Date Range Filter**: All Time, Last 7 Days, Last 30 Days, Last 90 Days
- **Live Filter Stats**: Shows filtered count, filtered revenue, and filtered profit
- **Orders Table**: Order ID, date, product, category, quantity, revenue, profit (color-coded by margin), and status badges (completed/pending/cancelled)
- **Export to CSV**: Download filtered orders as a CSV file
- **Quick Stats Panel**: Recent Activity shortcuts, Top Categories list, Reset Filters and Export buttons

---

## Design & UX

- **Dark Theme**: Professional glassmorphism design with deep navy backgrounds
- **Color System**: Cyan-blue primary, emerald accent, amber warnings, red danger
- **Responsive Layout**: Sidebar navigation with grid-based card layouts
- **Interactive Tables**: Hover effects, color-coded values, sortable data
- **Loading States**: Skeleton loaders and spinners for async data
- **Error Handling**: Graceful error messages with retry guidance

---

## Backend Capabilities

- **50+ API Endpoints** covering products, orders, customers, analytics, and forecasting
- **4 ML Models**: SARIMA, Prophet, LSTM, and Ensemble forecasting
- **ETL Pipeline**: CSV ingestion, data validation, and feature engineering
- **Real-time Streaming**: WebSocket and SSE support for live updates
- **Authentication**: OAuth2 with JWT tokens
- **Database**: SQLAlchemy ORM supporting SQLite (dev) and PostgreSQL (production)

---

## Deployment

- **Frontend**: Deployed on Vercel with automatic builds from GitHub
- **Backend API**: Deployed on HuggingFace Spaces with Docker
- **Data**: Brazilian e-commerce dataset (5 years of sales history, 150 products)

---

## Business Value

This platform helps businesses:
1. **Reduce stockouts** by forecasting demand 90 days ahead
2. **Cut dead stock** by identifying products with no sales in 90+ days
3. **Optimize purchasing** with ABC classification and reorder point recommendations
4. **Increase revenue** by identifying top-performing products and categories
5. **Save time** with automated monthly reports and AI-generated insights
6. **Make better decisions** with real-time dashboards and trend analysis
