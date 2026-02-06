---
title: Smart Inventory Forecaster
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: SARIMA + Prophet demand forecasting API
---

# Smart Inventory Manager - Demand Forecasting API

A production-ready demand forecasting system using **SARIMA + Prophet ensemble models**, achieving **18.35% SMAPE** on validation data.

## Features

- **SARIMA Forecasting**: Seasonal ARIMA with log transformation for stable predictions
- **Prophet Forecasting**: Facebook Prophet with automatic trend and seasonality detection
- **Ensemble Model**: Weighted combination of both models for improved accuracy
- **Inventory Recommendations**: Reorder points, safety stock, stockout risk assessment

## Performance

| Metric | Value |
|--------|-------|
| **SMAPE** | 18.35% |
| Target | <20% |
| Validation | 8-week holdout |

## How to Use

### 1. Upload Data

Upload a CSV file with your sales data. Required columns:
- `OrderDate` - Date of the order
- `Quantity` - Number of units sold
- `Category` - Product category

Optional columns:
- `OrderStatus` - Cancelled orders will be filtered out

### 2. Get Forecasts

Select a category and forecast horizon (1-52 weeks) to get:
- Weekly demand predictions
- 95% confidence intervals
- Ensemble forecast (SARIMA + Prophet average)

### 3. Inventory Recommendations

Get automated recommendations for each category:
- **Reorder Point**: When to place a new order
- **Safety Stock**: Buffer inventory for demand variability
- **Recommended Order Quantity**: How much to order
- **Stockout Risk**: Low / Medium / High classification

## API Usage

Use the Gradio Client for programmatic access:

```python
from gradio_client import Client

# Connect to the Space
client = Client("YOUR_USERNAME/smart-inventory-forecaster")

# Upload and train
result = client.predict(
    file="sales_data.csv",
    api_name="/load_dataset"
)

# Get forecast for a category
forecast = client.predict(
    category="Electronics",
    weeks=13,
    api_name="/get_forecast"
)

# Get all recommendations
recommendations = client.predict(
    api_name="/get_recommendations"
)
```

## Data Format Example

```csv
OrderDate,Quantity,Category,OrderStatus
2024-01-15,5,Electronics,Delivered
2024-01-15,3,Clothing & Fashion,Delivered
2024-01-16,2,Home & Kitchen,Processing
```

## Technical Details

### SARIMA Model
- Order: (1,1,1)
- Seasonal Order: (1,1,1,52) for yearly seasonality
- Weekly data aggregation
- Log transformation for variance stabilization

### Prophet Model
- Automatic trend changepoint detection
- Yearly and weekly seasonality
- 95% confidence intervals

### Ensemble Strategy
- Simple average of SARIMA and Prophet forecasts
- Averaged confidence intervals
- Falls back to single model if one fails

## Limitations

- Minimum 10 weeks of data per category required
- Cold start: First request may take longer as models initialize
- Free tier: Limited compute resources

## License

MIT License

---

Built with Gradio and Hugging Face Spaces
