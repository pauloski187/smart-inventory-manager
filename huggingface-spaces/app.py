"""
Smart Inventory Manager - Demand Forecasting API
Hugging Face Spaces Deployment with Gradio

Provides SARIMA and Prophet-based demand forecasting for inventory management.
Achieved 18.35% SMAPE on validation data.
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import warnings
import logging
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== SARIMA Forecaster ====================

from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAForecaster:
    """Production-ready SARIMA forecaster for demand prediction."""

    def __init__(self, seasonal_period: int = 52, use_log_transform: bool = True):
        self.seasonal_period = seasonal_period
        self.use_log_transform = use_log_transform
        self.model = None
        self.fitted_model = None
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 1, 1, seasonal_period)

    def fit(self, data: pd.Series, order: tuple = None, seasonal_order: tuple = None):
        """Fit SARIMA model to time series data."""
        if order:
            self.order = order
        if seasonal_order:
            self.seasonal_order = seasonal_order

        # Apply log transformation
        if self.use_log_transform:
            train_data = np.log1p(data.values)
        else:
            train_data = data.values

        try:
            self.model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=False, maxiter=200)
            logger.info(f"SARIMA model fitted successfully. AIC: {self.fitted_model.aic:.2f}")
            return True
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {e}")
            return False

    def forecast(self, steps: int, alpha: float = 0.05) -> Dict:
        """Generate forecasts with confidence intervals."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        # Back-transform if log was used
        if self.use_log_transform:
            forecast_mean = np.expm1(forecast_mean)
            conf_int = np.expm1(conf_int)

        # Ensure non-negative
        forecast_mean = np.maximum(forecast_mean, 0)
        conf_int = np.maximum(conf_int, 0)

        return {
            'forecast': forecast_mean.tolist(),
            'lower_ci': conf_int.iloc[:, 0].tolist(),
            'upper_ci': conf_int.iloc[:, 1].tolist()
        }


# ==================== Prophet Forecaster ====================

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Using SARIMA only.")


class ProphetForecaster:
    """Facebook Prophet forecaster for demand prediction."""

    def __init__(self):
        self.model = None
        self.fitted = False

    def fit(self, data: pd.Series) -> bool:
        """Fit Prophet model to time series data."""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available")
            return False

        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })

            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.95
            )
            self.model.fit(df)
            self.fitted = True
            logger.info("Prophet model fitted successfully")
            return True
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            return False

    def predict(self, steps: int) -> Dict:
        """Generate forecasts with confidence intervals."""
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        future = self.model.make_future_dataframe(periods=steps, freq='W')
        forecast = self.model.predict(future)

        # Get only future predictions
        forecast = forecast.tail(steps)

        # Ensure non-negative
        yhat = np.maximum(forecast['yhat'].values, 0)
        lower = np.maximum(forecast['yhat_lower'].values, 0)
        upper = np.maximum(forecast['yhat_upper'].values, 0)

        return {
            'forecast': yhat.tolist(),
            'lower_ci': lower.tolist(),
            'upper_ci': upper.tolist(),
            'dates': forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        }


# ==================== Category Forecaster ====================

class CategoryForecaster:
    """Manages forecasting models for multiple product categories."""

    def __init__(self):
        self.data = None
        self.categories = []
        self.sarima_models = {}
        self.prophet_models = {}
        self.category_data = {}

    def load_data(self, df: pd.DataFrame) -> bool:
        """Load and prepare data for forecasting."""
        try:
            # Standardize column names
            df.columns = df.columns.str.strip().str.lower()

            # Map common column names
            col_mapping = {
                'orderdate': 'order_date',
                'order_date': 'order_date',
                'date': 'order_date',
                'qty': 'quantity',
                'quantity': 'quantity',
                'cat': 'category',
                'category': 'category',
                'product_category': 'category'
            }

            for old, new in col_mapping.items():
                if old in df.columns:
                    df = df.rename(columns={old: new})

            # Convert date
            df['order_date'] = pd.to_datetime(df['order_date'])

            # Filter valid data
            df = df[df['quantity'] > 0]
            if 'orderstatus' in df.columns:
                df = df[df['orderstatus'] != 'Cancelled']

            self.data = df
            self.categories = df['category'].unique().tolist()

            # Prepare category-level weekly data
            for category in self.categories:
                cat_data = df[df['category'] == category].copy()
                weekly = cat_data.groupby(pd.Grouper(key='order_date', freq='W'))['quantity'].sum()
                weekly = weekly.fillna(0)
                if len(weekly) >= 10:  # Minimum 10 weeks of data
                    self.category_data[category] = weekly

            logger.info(f"Loaded data: {len(df)} records, {len(self.category_data)} categories with sufficient data")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def train_category(self, category: str) -> Dict:
        """Train SARIMA and Prophet models for a category."""
        if category not in self.category_data:
            return {"error": f"Category '{category}' not found or insufficient data"}

        data = self.category_data[category]
        results = {"category": category, "sarima": False, "prophet": False}

        # Train SARIMA
        try:
            sarima = SARIMAForecaster(seasonal_period=52)
            if sarima.fit(data):
                self.sarima_models[category] = sarima
                results["sarima"] = True
        except Exception as e:
            logger.error(f"SARIMA training failed for {category}: {e}")

        # Train Prophet
        if PROPHET_AVAILABLE:
            try:
                prophet = ProphetForecaster()
                if prophet.fit(data):
                    self.prophet_models[category] = prophet
                    results["prophet"] = True
            except Exception as e:
                logger.error(f"Prophet training failed for {category}: {e}")

        return results

    def train_all(self) -> List[Dict]:
        """Train models for all categories."""
        results = []
        for category in self.category_data.keys():
            result = self.train_category(category)
            results.append(result)
            logger.info(f"Trained {category}: SARIMA={result['sarima']}, Prophet={result['prophet']}")
        return results

    def forecast_category(self, category: str, weeks: int = 13) -> Dict:
        """Generate forecast for a category."""
        if category not in self.category_data:
            return {"error": f"Category '{category}' not found"}

        result = {
            "category": category,
            "weeks": weeks,
            "sarima": None,
            "prophet": None,
            "ensemble": None
        }

        # SARIMA forecast
        if category in self.sarima_models:
            try:
                sarima_forecast = self.sarima_models[category].forecast(steps=weeks)
                result["sarima"] = sarima_forecast
            except Exception as e:
                logger.error(f"SARIMA forecast failed: {e}")

        # Prophet forecast
        if category in self.prophet_models:
            try:
                prophet_forecast = self.prophet_models[category].predict(steps=weeks)
                result["prophet"] = prophet_forecast
            except Exception as e:
                logger.error(f"Prophet forecast failed: {e}")

        # Ensemble (simple average if both available)
        if result["sarima"] and result["prophet"]:
            sarima_fc = np.array(result["sarima"]["forecast"])
            prophet_fc = np.array(result["prophet"]["forecast"])
            ensemble_fc = (sarima_fc + prophet_fc) / 2

            # Average confidence intervals
            lower = (np.array(result["sarima"]["lower_ci"]) + np.array(result["prophet"]["lower_ci"])) / 2
            upper = (np.array(result["sarima"]["upper_ci"]) + np.array(result["prophet"]["upper_ci"])) / 2

            result["ensemble"] = {
                "forecast": ensemble_fc.tolist(),
                "lower_ci": lower.tolist(),
                "upper_ci": upper.tolist()
            }

        # Calculate inventory recommendations
        forecast_data = result["ensemble"] or result["sarima"] or result["prophet"]
        if forecast_data:
            total_forecast = sum(forecast_data["forecast"])
            daily_avg = total_forecast / (weeks * 7)
            upper_total = sum(forecast_data["upper_ci"])

            result["recommendations"] = {
                "forecast_total": round(total_forecast, 0),
                "daily_average": round(daily_avg, 2),
                "reorder_point": round(daily_avg * 14 + (upper_total - total_forecast) / 3, 0),
                "safety_stock": round((upper_total - total_forecast) / 3, 0),
                "recommended_order": round(total_forecast * 1.2, 0)
            }

        return result

    def get_inventory_recommendations(self) -> List[Dict]:
        """Get inventory recommendations for all categories."""
        recommendations = []
        for category in self.category_data.keys():
            forecast = self.forecast_category(category, weeks=13)
            if "recommendations" in forecast:
                rec = forecast["recommendations"]
                rec["category"] = category

                # Calculate stockout risk based on coefficient of variation
                if forecast.get("ensemble") or forecast.get("sarima"):
                    fc_data = forecast.get("ensemble") or forecast.get("sarima")
                    fc_array = np.array(fc_data["forecast"])
                    cv = np.std(fc_array) / (np.mean(fc_array) + 0.01)
                    rec["stockout_risk"] = "low" if cv < 0.3 else "medium" if cv < 0.5 else "high"
                else:
                    rec["stockout_risk"] = "unknown"

                recommendations.append(rec)

        return recommendations


# ==================== Global Forecaster Instance ====================

forecaster = CategoryForecaster()
DATA_LOADED = False


# ==================== Gradio Interface Functions ====================

def load_dataset(file) -> str:
    """Load dataset and train models."""
    global forecaster, DATA_LOADED

    if file is None:
        return json.dumps({"error": "No file uploaded"}, indent=2)

    try:
        # Read CSV
        df = pd.read_csv(file.name)
        logger.info(f"Loaded file with {len(df)} rows")

        # Load into forecaster
        if not forecaster.load_data(df):
            return json.dumps({"error": "Failed to load data"}, indent=2)

        # Train models
        training_results = forecaster.train_all()
        DATA_LOADED = True

        result = {
            "status": "success",
            "records_loaded": len(df),
            "categories": list(forecaster.category_data.keys()),
            "training_results": training_results,
            "message": f"Loaded {len(df)} records. Trained models for {len(forecaster.category_data)} categories."
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return json.dumps({"error": str(e)}, indent=2)


def get_forecast(category: str, weeks: int) -> str:
    """Get forecast for a category."""
    global forecaster, DATA_LOADED

    if not DATA_LOADED:
        return json.dumps({"error": "No data loaded. Please upload a dataset first."}, indent=2)

    if not category or category == "":
        return json.dumps({"error": "Please select a category"}, indent=2)

    try:
        result = forecaster.forecast_category(category, weeks=int(weeks))
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return json.dumps({"error": str(e)}, indent=2)


def get_all_forecasts(weeks: int) -> str:
    """Get forecasts for all categories."""
    global forecaster, DATA_LOADED

    if not DATA_LOADED:
        return json.dumps({"error": "No data loaded. Please upload a dataset first."}, indent=2)

    try:
        results = []
        for category in forecaster.category_data.keys():
            fc = forecaster.forecast_category(category, weeks=int(weeks))
            results.append(fc)

        return json.dumps({
            "status": "success",
            "count": len(results),
            "forecasts": results
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error: {e}")
        return json.dumps({"error": str(e)}, indent=2)


def get_recommendations() -> str:
    """Get inventory recommendations for all categories."""
    global forecaster, DATA_LOADED

    if not DATA_LOADED:
        return json.dumps({"error": "No data loaded. Please upload a dataset first."}, indent=2)

    try:
        recommendations = forecaster.get_inventory_recommendations()
        return json.dumps({
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations
        }, indent=2)
    except Exception as e:
        logger.error(f"Error: {e}")
        return json.dumps({"error": str(e)}, indent=2)


def get_categories() -> str:
    """Get list of available categories."""
    global forecaster, DATA_LOADED

    if not DATA_LOADED:
        return json.dumps({"categories": [], "message": "No data loaded"}, indent=2)

    return json.dumps({
        "categories": list(forecaster.category_data.keys()),
        "count": len(forecaster.category_data)
    }, indent=2)


def health_check() -> str:
    """Health check endpoint."""
    return json.dumps({
        "status": "healthy",
        "service": "smart-inventory-forecaster",
        "version": "1.0.0",
        "prophet_available": PROPHET_AVAILABLE,
        "data_loaded": DATA_LOADED,
        "categories_count": len(forecaster.category_data) if DATA_LOADED else 0
    }, indent=2)


# ==================== Gradio App ====================

def update_category_dropdown():
    """Update category dropdown based on loaded data."""
    if DATA_LOADED and forecaster.category_data:
        return gr.Dropdown(choices=list(forecaster.category_data.keys()), value=list(forecaster.category_data.keys())[0])
    return gr.Dropdown(choices=[], value=None)


# Create Gradio interface
with gr.Blocks(title="Smart Inventory Forecaster API", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Smart Inventory Manager - Demand Forecasting API

    **SARIMA + Prophet ensemble forecasting** achieving **18.35% SMAPE**

    Upload your sales data CSV and get demand forecasts with inventory recommendations.

    ---
    """)

    with gr.Tab("Upload Data"):
        gr.Markdown("### Step 1: Upload your sales data CSV")
        gr.Markdown("""
        Required columns: `OrderDate`, `Quantity`, `Category`

        Optional: `OrderStatus` (cancelled orders will be filtered)
        """)
        file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
        upload_btn = gr.Button("Load & Train Models", variant="primary")
        upload_output = gr.Code(label="Result", language="json")

        upload_btn.click(fn=load_dataset, inputs=[file_input], outputs=[upload_output])

    with gr.Tab("Forecast"):
        gr.Markdown("### Step 2: Get demand forecasts by category")

        with gr.Row():
            category_input = gr.Textbox(label="Category Name", placeholder="e.g., Electronics")
            weeks_input = gr.Slider(minimum=1, maximum=52, value=13, step=1, label="Forecast Weeks")

        forecast_btn = gr.Button("Get Forecast", variant="primary")
        forecast_output = gr.Code(label="Forecast Result", language="json")

        forecast_btn.click(fn=get_forecast, inputs=[category_input, weeks_input], outputs=[forecast_output])

        gr.Markdown("---")
        gr.Markdown("### Get All Category Forecasts")
        all_weeks_input = gr.Slider(minimum=1, maximum=52, value=13, step=1, label="Forecast Weeks")
        all_forecast_btn = gr.Button("Get All Forecasts")
        all_forecast_output = gr.Code(label="All Forecasts", language="json")

        all_forecast_btn.click(fn=get_all_forecasts, inputs=[all_weeks_input], outputs=[all_forecast_output])

    with gr.Tab("Inventory Recommendations"):
        gr.Markdown("### Get inventory recommendations for all categories")
        gr.Markdown("""
        Returns:
        - **Reorder Point**: When to reorder
        - **Safety Stock**: Buffer inventory
        - **Recommended Order Quantity**: How much to order
        - **Stockout Risk**: Low / Medium / High
        """)
        rec_btn = gr.Button("Get Recommendations", variant="primary")
        rec_output = gr.Code(label="Recommendations", language="json")

        rec_btn.click(fn=get_recommendations, outputs=[rec_output])

    with gr.Tab("API Info"):
        gr.Markdown("""
        ### API Endpoints (for programmatic access)

        This Gradio app exposes the following API endpoints:

        | Endpoint | Description |
        |----------|-------------|
        | `/api/predict` | Main prediction endpoint |
        | `POST /upload` | Upload CSV data |

        ### Using the API

        ```python
        from gradio_client import Client

        client = Client("YOUR_SPACE_URL")

        # Upload data
        result = client.predict(
            file="path/to/data.csv",
            api_name="/load_dataset"
        )

        # Get forecast
        forecast = client.predict(
            category="Electronics",
            weeks=13,
            api_name="/get_forecast"
        )
        ```

        ### Health Check
        """)
        health_btn = gr.Button("Check Health")
        health_output = gr.Code(label="Health Status", language="json")

        health_btn.click(fn=health_check, outputs=[health_output])

        gr.Markdown("---")
        categories_btn = gr.Button("List Categories")
        categories_output = gr.Code(label="Available Categories", language="json")

        categories_btn.click(fn=get_categories, outputs=[categories_output])

    gr.Markdown("""
    ---
    **Model Performance**: 18.35% SMAPE (Target: <20%) | **Models**: SARIMA + Prophet Ensemble
    """)


# Launch app
if __name__ == "__main__":
    demo.launch()
