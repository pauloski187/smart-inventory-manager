# Data Science Analysis Report

## Smart Inventory Manager - ML/Analytics Results

---

## 1. SARIMA Demand Forecasting Results

### Model Specification
```
Model: SARIMA(1,1,1)(1,1,1,7)
Parameters:
  - p=1: Autoregressive term (1 lag)
  - d=1: First differencing for stationarity
  - q=1: Moving average term (1 lag)
  - P=1: Seasonal autoregressive term
  - D=1: Seasonal differencing
  - Q=1: Seasonal moving average term
  - s=7: Weekly seasonality (7-day cycle)
```

### Data Overview
| Metric | Value |
|--------|-------|
| Records Used | 98,009 (after removing cancelled) |
| Date Range | 2020-01-01 to 2024-12-31 (5 years) |
| Categories | 10 |
| Aggregation | Daily demand per category |
| Train Period | 2020-01-01 to 2023-12-31 (80%) |
| Test Period | 2024-01-01 to 2024-12-31 (20%) |

### Stationarity Check (ADF Test)
| Category | ADF Statistic | p-value | Stationary |
|----------|---------------|---------|------------|
| Books & Media | -5.7069 | 0.000001 | ✓ YES |
| Clothing & Fashion | -6.8363 | 0.000000 | ✓ YES |
| Electronics | -5.0745 | 0.000016 | ✓ YES |
| Grocery & Gourmet Food | -5.0593 | 0.000017 | ✓ YES |
| Health & Personal Care | -5.3986 | 0.000003 | ✓ YES |
| Home & Kitchen | -5.1999 | 0.000009 | ✓ YES |
| Office Products | -5.4837 | 0.000002 | ✓ YES |
| Sports & Fitness | -4.9103 | 0.000033 | ✓ YES |
| Tools & Home Improvement | -6.5429 | 0.000000 | ✓ YES |
| Toys & Games | -5.5172 | 0.000002 | ✓ YES |

**Result**: ALL 10/10 categories are stationary (p < 0.05)

### Residual Diagnostics (Ljung-Box Test)
| Category | Res. Mean | Res. Std | LB p-value | Pass |
|----------|-----------|----------|------------|------|
| Books & Media | 0.0063 | 4.9212 | 0.5178 | ✓ |
| Clothing & Fashion | -0.0120 | 3.8873 | 0.5838 | ✓ |
| Electronics | -0.0162 | 5.3750 | 0.6360 | ✓ |
| Grocery & Gourmet Food | -0.0280 | 5.8124 | 0.6512 | ✓ |
| Health & Personal Care | -0.0119 | 6.0493 | 0.7115 | ✓ |
| Home & Kitchen | 0.0297 | 5.5985 | 0.6911 | ✓ |
| Office Products | -0.0123 | 6.5146 | 0.1168 | ✓ |
| Sports & Fitness | 0.0454 | 7.8351 | 0.8152 | ✓ |
| Tools & Home Improvement | -0.0122 | 4.9663 | 0.4298 | ✓ |
| Toys & Games | 0.0328 | 5.7431 | 0.9416 | ✓ |

**Result**: ALL 10/10 models passed (p > 0.05 = no autocorrelation in residuals)

### Model Performance Metrics
| Category | MAE | RMSE | SMAPE (%) |
|----------|-----|------|-----------|
| Sports & Fitness | 21.28 | 22.69 | 84.23 |
| Office Products | 14.70 | 16.03 | 81.76 |
| Health & Personal Care | 15.02 | 16.02 | 86.86 |
| Home & Kitchen | 12.32 | 13.50 | 82.88 |
| Toys & Games | 11.27 | 12.30 | 79.62 |
| Grocery & Gourmet Food | 11.23 | 12.25 | 83.46 |
| Electronics | 11.21 | 12.20 | 83.75 |
| Books & Media | 10.95 | 11.80 | 92.83 |
| Tools & Home Improvement | 9.00 | 9.85 | 82.94 |
| Clothing & Fashion | 4.59 | 5.29 | 75.36 |
| **AVERAGE** | **12.16** | **13.19** | **83.37** |

### 90-Day Forecast Results
| Category | 30-Day | 60-Day | 90-Day | 95% CI Range |
|----------|--------|--------|--------|--------------|
| Sports & Fitness | 1,108 | 2,227 | 3,357 | 1,117 - 5,597 |
| Office Products | 857 | 1,729 | 2,620 | 940 - 4,300 |
| Home & Kitchen | 639 | 1,285 | 1,937 | 601 - 3,273 |
| Toys & Games | 596 | 1,201 | 1,816 | 513 - 3,118 |
| Health & Personal Care | 593 | 1,192 | 1,796 | 285 - 3,325 |
| Electronics | 559 | 1,132 | 1,720 | 426 - 3,013 |
| Tools & Home Improvement | 557 | 1,121 | 1,691 | 566 - 2,817 |
| Grocery & Gourmet Food | 525 | 1,058 | 1,597 | 256 - 2,943 |
| Books & Media | 476 | 963 | 1,458 | 312 - 2,604 |
| Clothing & Fashion | 304 | 612 | 923 | 154 - 1,691 |
| **TOTAL** | **6,213** | **12,521** | **18,914** | - |

---

## 2. ABC Analysis Results

### Classification Methodology
- **Class A**: Products contributing to top 80% of revenue
- **Class B**: Products contributing to next 15% of revenue (80-95%)
- **Class C**: Products contributing to bottom 5% of revenue

### Classification Summary
| Class | Product Count | Product % | Revenue | Revenue % |
|-------|---------------|-----------|---------|-----------|
| A | ~18% of products | 18% | 80% | High priority |
| B | ~27% of products | 27% | 15% | Medium priority |
| C | ~55% of products | 55% | 5% | Low priority |

### Pareto Validation
- Top 20% of products generate approximately 80% of revenue
- Pareto principle (80/20 rule) confirmed in the data

### Recommendations by Class

#### Class A (High Priority)
- **Reorder Frequency**: Weekly
- **Safety Stock**: High (2-3 weeks supply)
- **Monitoring**: Daily
- **Supplier Strategy**: Multiple reliable suppliers
- **Forecasting**: Advanced models (SARIMA) required

#### Class B (Medium Priority)
- **Reorder Frequency**: Bi-weekly
- **Safety Stock**: Medium (1-2 weeks supply)
- **Monitoring**: Weekly
- **Supplier Strategy**: Primary + backup supplier
- **Forecasting**: Standard forecasting

#### Class C (Low Priority)
- **Reorder Frequency**: Monthly
- **Safety Stock**: Low (minimal)
- **Monitoring**: Monthly review
- **Supplier Strategy**: Single supplier acceptable
- **Forecasting**: Simple moving average

---

## 3. Low Stock Analysis Results

### Current Inventory Status
Based on database analysis:

| Status | Count | Description |
|--------|-------|-------------|
| In Stock | ~8,500 | Above reorder threshold |
| Low Stock | ~400 | Below reorder threshold |
| Out of Stock | ~100 | Zero inventory |
| Dead Stock | Variable | No sales in 90+ days |

### Low Stock Priority Levels
- **High Priority**: Current stock < 50% of reorder threshold
- **Medium Priority**: Current stock between 50-100% of reorder threshold

### Reorder Recommendations
For each low stock product:
- Calculate stock velocity (units/day)
- Estimate days until stockout
- Recommend reorder quantity (30 days supply + safety stock)

---

## 4. Dead Stock Analysis

### Detection Criteria
- **Threshold**: No sales in 90 days (configurable)
- **Value at Risk**: Total inventory value tied up in dead stock

### Dead Stock Recommendations
| Days Since Last Sale | Recommendation |
|---------------------|----------------|
| 90-120 days | Monitor closely |
| 120-180 days | Consider markdown/promotion |
| 180-270 days | Deep discount or bundle |
| 270+ days | Liquidate or write off |

---

## 5. Key Insights & Findings

### Demand Patterns
1. **Weekly Seasonality**: Strong day-of-week patterns detected
2. **Trend**: Stable demand with slight growth trend
3. **Variability**: High daily variability (SMAPE ~83%)
4. **Stationarity**: All categories are stationary (suitable for SARIMA)

### Category Performance
1. **Highest Demand**: Sports & Fitness (3,357 units/90 days)
2. **Most Predictable**: Clothing & Fashion (lowest MAE of 4.59)
3. **Highest Variability**: Books & Media (highest SMAPE of 92.83%)

### Model Validation
- **Linear Regression**: FAILED assumption checks (autocorrelation, heteroscedasticity)
- **SARIMA**: PASSED all diagnostic tests
- **Conclusion**: SARIMA is the appropriate model for this data

---

## 6. Business Recommendations

### Inventory Planning
1. Use **upper confidence interval** for safety stock calculations
2. Prioritize **Sports & Fitness** and **Office Products** (highest demand)
3. Apply weekly review cycle for Class A products

### Forecasting Improvements
1. Consider **weekly aggregation** for more stable forecasts
2. Add **holiday/promotion effects** as exogenous variables
3. Retrain models **monthly** with new data

### Stock Optimization
1. Reduce safety stock for Class C products
2. Implement automatic reorder triggers based on forecast
3. Monitor dead stock weekly and take action within 90 days

---

## 7. Technical Notes

### Why SARIMA Over Linear Regression?

Linear Regression assumptions violated:
- **Independence**: Durbin-Watson = 0.68 (strong autocorrelation)
- **Homoscedasticity**: Breusch-Pagan p < 0.001 (heteroscedasticity)
- **Multicollinearity**: VIF > 85 for time-based features

SARIMA advantages:
- Explicitly models autocorrelation (AR terms)
- Handles seasonality (seasonal differencing)
- Provides prediction intervals
- No independence assumption required

### Model Persistence
- Models saved as pickle files in `/backend/models/`
- Format: `sarima_{category_name}.pkl`
- Retrain via POST `/forecast/retrain-models`

---

*Analysis Report generated: December 25, 2024*
