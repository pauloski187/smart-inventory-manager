# Smart Inventory Manager - Technical Project Report

## Complete Documentation: From Concept to Production

---

## Executive Summary

This document provides a comprehensive technical walkthrough of the **Smart Sales & Inventory Management System** - an end-to-end machine learning solution for demand forecasting and inventory optimization. The system processes 100,000+ sales transactions spanning 5 years (2020-2024) to deliver actionable business intelligence through a production-ready FastAPI backend.

**Key Achievement**: Successfully implemented SARIMA-based demand forecasting after rigorously validating that traditional Linear Regression failed statistical assumption tests - demonstrating proper data science methodology.

---

## 1. Project Overview & Business Problem

### 1.1 The Problem

Retail and e-commerce businesses face critical inventory challenges:
- **Overstocking**: Ties up capital, increases storage costs, risks obsolescence
- **Understocking**: Lost sales, customer dissatisfaction, damaged reputation
- **No visibility**: Inability to predict future demand leads to reactive (not proactive) decisions

### 1.2 The Solution

An intelligent inventory management system that:
1. **Predicts demand** using time series forecasting (SARIMA)
2. **Classifies products** using ABC analysis (Pareto principle)
3. **Identifies problems** through dead stock and low stock detection
4. **Generates recommendations** for optimal inventory levels

### 1.3 Dataset Characteristics

| Metric | Value |
|--------|-------|
| Total Orders | 100,000 |
| Date Range | January 2020 - December 2024 (5 years) |
| Products | 9,000 unique items |
| Categories | 10 distinct categories |
| Customers | 14,549 |
| Sellers | 900 |
| Orders Used (after cleaning) | 98,009 (cancelled orders removed) |

**Categories covered:**
- Electronics
- Clothing & Fashion
- Home & Kitchen
- Books & Media
- Sports & Fitness
- Health & Personal Care
- Toys & Games
- Grocery & Gourmet Food
- Office Products
- Tools & Home Improvement

---

## 2. Data Processing Pipeline

### 2.1 ETL Architecture

```
Raw CSV Files → Ingestion → Validation → Transformation → Database
     ↓              ↓            ↓              ↓            ↓
  orders.csv    Type conv    Quality checks  Features    SQLite
  products.csv  Date parse   Null handling   Aggregation  Tables
  customers.csv Encoding     Range checks    Time series  Indexes
```

### 2.2 Data Ingestion (`backend/app/etl/ingestion.py`)

**Key operations:**
- CSV parsing with proper encoding (UTF-8)
- Date parsing with multiple format handling
- Type conversion (strings to floats, integers)
- ID standardization across tables

```python
# Example: Date parsing with fallback
def parse_date(date_str):
    formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None
```

### 2.3 Data Validation (`backend/app/etl/validation.py`)

**Validation rules implemented:**
1. **Completeness**: Check for required fields (ProductID, OrderDate, Quantity)
2. **Range validation**: Prices > 0, Quantities > 0, Dates within expected range
3. **Referential integrity**: Product IDs exist, Customer IDs valid
4. **Business logic**: Total = Price × Quantity, Profit = Revenue - Cost

**Quality issues handled:**
- 1,991 cancelled orders excluded from analysis
- Null values in optional fields (brand, category) defaulted appropriately
- Duplicate detection and handling

### 2.4 Feature Engineering (`backend/app/etl/transform.py`)

**Time-based features created:**
```python
# Daily demand aggregation per category
daily_demand = orders.groupby(['date', 'category']).agg({
    'quantity': 'sum',
    'total_amount': 'sum',
    'profit': 'sum'
}).reset_index()
```

**Features generated:**
| Feature | Description | Purpose |
|---------|-------------|---------|
| `day_of_week` | 0-6 (Mon-Sun) | Weekly seasonality |
| `month` | 1-12 | Monthly patterns |
| `quarter` | 1-4 | Quarterly trends |
| `is_weekend` | Boolean | Weekend effects |
| `days_since_start` | Integer | Trend capture |
| `rolling_7d_avg` | Float | Smoothing |
| `rolling_30d_avg` | Float | Monthly baseline |

---

## 3. Exploratory Data Analysis

### 3.1 Analysis Notebooks Created

| Notebook | Purpose | Key Findings |
|----------|---------|--------------|
| `01_data_cleaning_validation.ipynb` | Data quality assessment | 98% data completeness |
| `02_exploratory_data_analysis.ipynb` | Distribution analysis | Right-skewed sales |
| `03_sales_trend_analysis.ipynb` | Time series patterns | Strong weekly seasonality |
| `04_product_performance_analysis.ipynb` | Product metrics | 80/20 Pareto confirmed |
| `05_customer_segmentation_analysis.ipynb` | Customer behavior | RFM segmentation |
| `06_inventory_optimization.ipynb` | Stock analysis | 400+ low stock items |
| `07_demand_forecasting_experiments.ipynb` | Initial forecasting | Linear regression tested |
| `07a_regression_assumptions_validation.ipynb` | Statistical validation | Assumptions violated |
| `07b_sarima_demand_forecasting.ipynb` | SARIMA implementation | Final model |
| `08_abc_analysis_development.ipynb` | ABC classification | Priority categorization |
| `09_model_evaluation_comparison.ipynb` | Model comparison | SARIMA selected |

### 3.2 Key Patterns Discovered

**1. Weekly Seasonality**
- Clear day-of-week patterns in all categories
- Weekends show different buying behavior
- Tuesday-Thursday typically highest volume

**2. Category Performance Variation**
- Sports & Fitness: Highest demand variability
- Clothing & Fashion: Most predictable (lowest error)
- Books & Media: Highest forecast uncertainty

**3. Pareto Principle Confirmed**
- Top 20% of products generate ~80% of revenue
- Justifies ABC classification approach

---

## 4. Model Development Journey

### 4.1 Initial Approach: Linear Regression

**Why we tried it first:**
- Simple, interpretable baseline
- Fast to train and deploy
- Common starting point

**Features used:**
```python
features = ['day_of_week', 'month', 'quarter', 'is_weekend',
            'days_since_start', 'rolling_7d_avg', 'rolling_30d_avg']
```

### 4.2 The Critical Discovery: Assumption Violations

**Statistical tests performed:**

| Test | Purpose | Result | Pass? |
|------|---------|--------|-------|
| Durbin-Watson | Autocorrelation | 0.68 | ❌ FAIL |
| Breusch-Pagan | Heteroscedasticity | p < 0.001 | ❌ FAIL |
| VIF | Multicollinearity | > 85 | ❌ FAIL |
| Shapiro-Wilk | Normality of residuals | p < 0.01 | ❌ FAIL |

**What these failures mean:**
1. **Durbin-Watson = 0.68** (should be ~2.0): Strong positive autocorrelation - today's demand is correlated with yesterday's
2. **Breusch-Pagan p < 0.001**: Variance of errors changes over time - predictions less reliable in some periods
3. **VIF > 85**: Time-based features are highly correlated with each other
4. **Non-normal residuals**: Confidence intervals would be unreliable

**Conclusion**: Linear Regression is statistically INVALID for this time series data.

### 4.3 The Solution: SARIMA

**Model specification: SARIMA(1,1,1)(1,1,1,7)**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| p | 1 | One autoregressive lag |
| d | 1 | First differencing for stationarity |
| q | 1 | One moving average term |
| P | 1 | Seasonal autoregressive (1 week lag) |
| D | 1 | Seasonal differencing |
| Q | 1 | Seasonal moving average |
| s | 7 | Weekly seasonality (7 days) |

**Why SARIMA works:**
1. **Handles autocorrelation**: AR terms explicitly model correlation between consecutive observations
2. **Captures seasonality**: Seasonal components model weekly patterns
3. **Differencing**: Makes non-stationary data stationary
4. **Provides confidence intervals**: Proper uncertainty quantification

### 4.4 Stationarity Validation (ADF Test)

All 10 categories passed the Augmented Dickey-Fuller test:

| Category | ADF Statistic | p-value | Stationary? |
|----------|---------------|---------|-------------|
| Books & Media | -5.71 | 0.000001 | ✓ YES |
| Clothing & Fashion | -6.84 | 0.000000 | ✓ YES |
| Electronics | -5.07 | 0.000016 | ✓ YES |
| Grocery & Gourmet Food | -5.06 | 0.000017 | ✓ YES |
| Health & Personal Care | -5.40 | 0.000003 | ✓ YES |
| Home & Kitchen | -5.20 | 0.000009 | ✓ YES |
| Office Products | -5.48 | 0.000002 | ✓ YES |
| Sports & Fitness | -4.91 | 0.000033 | ✓ YES |
| Tools & Home Improvement | -6.54 | 0.000000 | ✓ YES |
| Toys & Games | -5.52 | 0.000002 | ✓ YES |

### 4.5 Residual Diagnostics (Ljung-Box Test)

All models passed - no autocorrelation in residuals:

| Category | Residual Mean | Residual Std | LB p-value | Pass? |
|----------|---------------|--------------|------------|-------|
| Books & Media | 0.006 | 4.92 | 0.518 | ✓ |
| Clothing & Fashion | -0.012 | 3.89 | 0.584 | ✓ |
| Electronics | -0.016 | 5.38 | 0.636 | ✓ |
| Sports & Fitness | 0.045 | 7.84 | 0.815 | ✓ |
| ... | ... | ... | ... | ✓ |

**10/10 categories passed** (p > 0.05 indicates no residual autocorrelation)

---

## 5. Model Performance

### 5.1 Performance Metrics

| Category | MAE | RMSE | SMAPE (%) |
|----------|-----|------|-----------|
| Clothing & Fashion | 4.59 | 5.29 | 75.36 |
| Tools & Home Improvement | 9.00 | 9.85 | 82.94 |
| Books & Media | 10.95 | 11.80 | 92.83 |
| Electronics | 11.21 | 12.20 | 83.75 |
| Grocery & Gourmet Food | 11.23 | 12.25 | 83.46 |
| Toys & Games | 11.27 | 12.30 | 79.62 |
| Home & Kitchen | 12.32 | 13.50 | 82.88 |
| Health & Personal Care | 15.02 | 16.02 | 86.86 |
| Office Products | 14.70 | 16.03 | 81.76 |
| Sports & Fitness | 21.28 | 22.69 | 84.23 |
| **AVERAGE** | **12.16** | **13.19** | **83.37** |

**Interpretation:**
- **MAE (Mean Absolute Error)**: On average, predictions are off by ~12 units
- **RMSE**: Penalizes large errors more - average ~13 units
- **SMAPE**: Symmetric percentage error - ~83% (higher for volatile series)

### 5.2 90-Day Forecast Results

| Category | 30-Day | 60-Day | 90-Day | 95% CI Range |
|----------|--------|--------|--------|--------------|
| Sports & Fitness | 1,108 | 2,227 | 3,357 | 1,117 - 5,597 |
| Office Products | 857 | 1,729 | 2,620 | 940 - 4,300 |
| Home & Kitchen | 639 | 1,285 | 1,937 | 601 - 3,273 |
| Toys & Games | 596 | 1,201 | 1,816 | 513 - 3,118 |
| Health & Personal Care | 593 | 1,192 | 1,796 | 285 - 3,325 |
| Electronics | 559 | 1,132 | 1,720 | 426 - 3,013 |
| **TOTAL (90-day)** | **6,213** | **12,521** | **18,914** | - |

---

## 6. ABC Analysis Implementation

### 6.1 Methodology

Based on Pareto Principle (80/20 rule):

| Class | Revenue Contribution | Product % | Priority |
|-------|---------------------|-----------|----------|
| A | Top 80% | ~18% | High |
| B | Next 15% (80-95%) | ~27% | Medium |
| C | Bottom 5% | ~55% | Low |

### 6.2 Implementation

```python
def perform_abc_analysis(products_with_revenue):
    # Sort by revenue descending
    sorted_products = sorted(products_with_revenue,
                            key=lambda x: x['revenue'],
                            reverse=True)

    total_revenue = sum(p['revenue'] for p in sorted_products)
    cumulative = 0

    for product in sorted_products:
        cumulative += product['revenue']
        cumulative_pct = (cumulative / total_revenue) * 100

        if cumulative_pct <= 80:
            product['abc_class'] = 'A'
        elif cumulative_pct <= 95:
            product['abc_class'] = 'B'
        else:
            product['abc_class'] = 'C'

    return sorted_products
```

### 6.3 Recommendations by Class

| Class | Reorder Frequency | Safety Stock | Monitoring | Forecasting |
|-------|-------------------|--------------|------------|-------------|
| A | Weekly | 2-3 weeks supply | Daily | SARIMA (advanced) |
| B | Bi-weekly | 1-2 weeks supply | Weekly | Standard |
| C | Monthly | Minimal | Monthly review | Simple average |

---

## 7. Low Stock & Dead Stock Detection

### 7.1 Low Stock Analysis

**Detection criteria:**
- Current stock < Reorder threshold

**Priority levels:**
- **High**: Stock < 50% of reorder threshold
- **Medium**: Stock between 50-100% of threshold

**Current status (from database):**
| Status | Count |
|--------|-------|
| In Stock | ~8,500 |
| Low Stock | ~400 |
| Out of Stock | ~100 |

### 7.2 Dead Stock Analysis

**Detection criteria:**
- No sales in last 90 days (configurable)

**Recommended actions:**
| Days Since Sale | Action |
|-----------------|--------|
| 90-120 | Monitor closely |
| 120-180 | Markdown/promotion |
| 180-270 | Deep discount or bundle |
| 270+ | Liquidate or write off |

---

## 8. API Architecture

### 8.1 Technology Stack

| Layer | Technology |
|-------|------------|
| Framework | FastAPI 2.0 |
| ORM | SQLAlchemy 2.0 |
| Database | SQLite (dev) / PostgreSQL (prod) |
| ML | statsmodels, scikit-learn |
| Data | pandas, numpy |
| Validation | Pydantic v2 |
| Testing | pytest |

### 8.2 API Endpoints Summary

**Forecasting Endpoints (`/forecast/`)**
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload-data` | Upload sales CSV for training |
| GET | `/forecast/{category}` | Get category forecast with CI |
| GET | `/forecasts/all` | All category forecasts |
| GET | `/inventory-recommendations` | Reorder recommendations |
| POST | `/retrain-models` | Trigger model retraining |
| GET | `/model-status` | Check model status |

**Analytics Endpoints (`/analytics/`)**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/dashboard/summary` | KPI dashboard |
| GET | `/abc-analysis` | ABC classification |
| GET | `/inventory/low-stock` | Low stock alerts |
| GET | `/inventory/dead-stock` | Dead stock detection |
| GET | `/monthly-sales-trend` | Revenue/profit trends |
| GET | `/monthly-report/{year}/{month}` | Monthly report |
| GET | `/product-performance` | Best/worst products |
| GET | `/category-performance` | Category metrics |
| GET | `/sales-by-day-of-week` | Weekly patterns |
| GET | `/revenue-vs-profit-trend` | R&P comparison |
| GET | `/available-months` | Report month list |

### 8.3 Response Example

```json
{
  "category": "Sports & Fitness",
  "forecast_90_day": 3357,
  "confidence_interval": {
    "lower": 1117,
    "upper": 5597
  },
  "model_info": {
    "type": "SARIMA",
    "parameters": "(1,1,1)(1,1,1,7)",
    "last_trained": "2024-12-25"
  },
  "recommendations": {
    "reorder_point": 450,
    "safety_stock": 120,
    "stockout_risk": "low"
  }
}
```

---

## 9. Technical Challenges & Solutions

### 9.1 Challenge: Model Selection

**Problem**: Initial Linear Regression showed good R² but failed assumption tests.

**Solution**:
1. Implemented comprehensive assumption testing
2. Switched to SARIMA which is designed for time series
3. Documented the reasoning for future reference

**Lesson**: Good R² does not mean valid model. Always validate assumptions.

### 9.2 Challenge: Category-Specific Variability

**Problem**: Some categories (Sports & Fitness) had much higher variance than others.

**Solution**:
1. Train separate model per category
2. Use upper confidence interval for safety stock
3. Adjust recommendations based on category volatility

### 9.3 Challenge: Seasonality Detection

**Problem**: Multiple overlapping seasonal patterns (daily, weekly, monthly).

**Solution**:
1. Used ADF test to confirm stationarity after differencing
2. Set seasonal period s=7 for weekly patterns
3. Validated with Ljung-Box test on residuals

---

## 10. Testing & Quality Assurance

### 10.1 Test Results

```
======================== 26 passed in 4.14s ========================
```

### 10.2 Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| ETL Pipeline | 14 | ✓ Pass |
| ML Models | 9 | ✓ Pass |
| API Endpoints | 3 | ✓ Pass |

### 10.3 Test Categories

**Unit Tests:**
- Data validation functions
- Feature engineering
- ABC classification logic
- Forecast calculations

**Integration Tests:**
- Database operations
- API endpoint responses
- Model training/prediction pipeline

---

## 11. Production Readiness

### 11.1 Completed

- [x] CORS configuration for frontend integration
- [x] Health check endpoint (`/health`)
- [x] Comprehensive error handling
- [x] Input validation (Pydantic)
- [x] API documentation (Swagger/ReDoc at `/docs`)
- [x] Model persistence (pickle files)
- [x] Background task support
- [x] Logging infrastructure
- [x] Unit tests (26 passing)

### 11.2 Recommended for Production Deployment

- [ ] Environment-based configuration (dev/staging/prod)
- [ ] Rate limiting
- [ ] Authentication for sensitive endpoints
- [ ] PostgreSQL for production scale
- [ ] Model versioning and A/B testing
- [ ] Monitoring and alerting
- [ ] Container deployment (Docker)
- [ ] CI/CD pipeline

---

## 12. Project Structure

```
smart-inventory-manager/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # FastAPI app initialization
│   │   ├── database.py          # SQLAlchemy configuration
│   │   ├── config/              # Settings & configuration
│   │   ├── models/              # SQLAlchemy ORM models
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── routes/              # API endpoint handlers
│   │   │   ├── analytics.py     # Analytics & visualization endpoints
│   │   │   ├── forecast.py      # SARIMA forecast endpoints
│   │   │   ├── products.py      # Product CRUD
│   │   │   └── orders.py        # Order management
│   │   ├── ml/                  # Machine Learning modules
│   │   │   ├── sarima_forecaster.py  # SARIMA implementation
│   │   │   ├── abc_analysis.py       # ABC classification
│   │   │   └── dead_stock.py         # Dead stock detection
│   │   ├── etl/                 # Data pipeline
│   │   │   ├── ingestion.py     # CSV loading
│   │   │   ├── transform.py     # Feature engineering
│   │   │   └── validation.py    # Data quality checks
│   │   └── services/            # Business logic layer
│   ├── models/                  # Trained SARIMA models (.pkl)
│   ├── notebooks/               # Jupyter analysis notebooks (9)
│   ├── tests/                   # pytest test suite
│   └── requirements.txt         # Python dependencies
├── ml/
│   ├── data/processed/          # Cleaned datasets
│   └── scripts/                 # Training scripts
├── PROJECT_REPORT.md            # Project summary
├── DATA_SCIENCE_ANALYSIS_REPORT.md  # ML results
└── LOVABLE_AI_PROMPT.md         # Frontend specifications
```

---

## 13. Key Learnings & Reflections

### What Worked Well

1. **Rigorous statistical validation**: Testing Linear Regression assumptions before discovering SARIMA was the right approach
2. **Category-specific modeling**: Each category has different patterns - separate models captured this
3. **Comprehensive API design**: RESTful design with clear documentation enables easy frontend integration
4. **Modular architecture**: Separation of ETL, ML, and API layers enables independent testing and updates

### What Could Be Improved

1. **Real-time updates**: Current batch approach; could add streaming predictions
2. **External factors**: Model doesn't account for promotions, holidays, or external events
3. **Ensemble methods**: Could combine SARIMA with ML models for potentially better accuracy
4. **Automated retraining**: Manual retraining; could automate based on prediction drift

### Key Technical Learnings

1. **Time series ≠ Regression**: Autocorrelation makes standard regression invalid
2. **SMAPE for imbalanced data**: Better than MAPE when dealing with low values
3. **Confidence intervals matter**: Point forecasts without uncertainty are incomplete
4. **ABC classification scales**: Works regardless of business size

---

## 14. Conclusion

This project successfully delivers a **production-ready inventory management system** with:

1. **Validated forecasting**: SARIMA properly validated after Linear Regression was proven inappropriate
2. **Comprehensive analytics**: ABC analysis, dead stock, low stock, and performance metrics
3. **Business-ready API**: 38+ endpoints covering all inventory management needs
4. **Solid foundation**: Clean architecture, tested code, documented API

The system transforms raw sales data into actionable inventory recommendations, enabling businesses to:
- Reduce overstock by predicting demand
- Prevent stockouts with early warnings
- Optimize inventory investment with ABC prioritization
- Make data-driven decisions with confidence intervals

---

*Technical Report Generated: December 25, 2024*
*Total Development Time: Multi-phase iterative development*
*Lines of Code: ~5,000+ (Python)*
*API Endpoints: 38+*
*Test Coverage: 26 automated tests*
