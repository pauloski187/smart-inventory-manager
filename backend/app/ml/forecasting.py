"""
Demand Forecasting Module

Implements time-series forecasting for product demand prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from sklearn.linear_model import LinearRegression
import logging

from ..models import Order, Product

logger = logging.getLogger(__name__)


class DemandForecaster:
    """
    Demand forecasting for products.

    Implements multiple forecasting methods:
    - Moving Average
    - Linear Regression
    - Exponential Smoothing (simple)
    """

    def __init__(self, db: Session):
        self.db = db
        self._models: Dict[str, Any] = {}

    def get_historical_demand(
        self,
        product_id: str,
        days: int = 90
    ) -> pd.DataFrame:
        """
        Get historical daily demand for a product.

        Args:
            product_id: Product to analyze
            days: Number of days of history

        Returns:
            DataFrame with date and quantity columns
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        orders = self.db.query(
            func.date(Order.order_date).label('date'),
            func.sum(Order.quantity).label('quantity')
        ).filter(
            Order.product_id == product_id,
            Order.order_date >= cutoff_date,
            Order.order_status != 'Cancelled'
        ).group_by(
            func.date(Order.order_date)
        ).all()

        if not orders:
            return pd.DataFrame(columns=['date', 'quantity'])

        df = pd.DataFrame(orders, columns=['date', 'quantity'])
        df['date'] = pd.to_datetime(df['date'])

        # Fill in missing dates with 0
        date_range = pd.date_range(
            start=cutoff_date,
            end=datetime.now(),
            freq='D'
        )
        full_df = pd.DataFrame({'date': date_range})
        df = full_df.merge(df, on='date', how='left').fillna(0)

        return df.sort_values('date')

    def moving_average_forecast(
        self,
        product_id: str,
        window: int = 7,
        days_ahead: int = 30
    ) -> Dict:
        """
        Simple moving average forecast.

        Args:
            product_id: Product to forecast
            window: Moving average window (days)
            days_ahead: Days to forecast

        Returns:
            Dictionary with forecast results
        """
        historical = self.get_historical_demand(product_id, days=90)

        if len(historical) < window:
            return {
                'product_id': product_id,
                'method': 'moving_average',
                'forecast_dates': [],
                'forecast_values': [],
                'error': 'Insufficient historical data'
            }

        # Calculate moving average
        ma = historical['quantity'].rolling(window=window).mean().iloc[-1]

        # Generate forecast dates
        last_date = historical['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )

        # Forecast is just the moving average for each day
        forecast_values = [round(ma, 2)] * days_ahead

        return {
            'product_id': product_id,
            'method': 'moving_average',
            'window': window,
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'forecast_values': forecast_values,
            'daily_average': round(ma, 2),
            'total_forecast': round(sum(forecast_values), 2)
        }

    def linear_regression_forecast(
        self,
        product_id: str,
        days_ahead: int = 30
    ) -> Dict:
        """
        Linear regression trend forecast.

        Captures trend in demand over time.

        Args:
            product_id: Product to forecast
            days_ahead: Days to forecast

        Returns:
            Dictionary with forecast results
        """
        historical = self.get_historical_demand(product_id, days=90)

        if len(historical) < 14:
            return {
                'product_id': product_id,
                'method': 'linear_regression',
                'forecast_dates': [],
                'forecast_values': [],
                'error': 'Insufficient historical data'
            }

        # Prepare features
        X = np.arange(len(historical)).reshape(-1, 1)
        y = historical['quantity'].values

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Store model
        self._models[product_id] = model

        # Generate forecast
        last_idx = len(historical)
        future_X = np.arange(last_idx, last_idx + days_ahead).reshape(-1, 1)
        predictions = model.predict(future_X)

        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)

        # Generate forecast dates
        last_date = historical['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )

        # Determine trend
        slope = model.coef_[0]
        if slope > 0.1:
            trend = 'Increasing'
        elif slope < -0.1:
            trend = 'Declining'
        else:
            trend = 'Stable'

        return {
            'product_id': product_id,
            'method': 'linear_regression',
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'forecast_values': [round(v, 2) for v in predictions],
            'trend': trend,
            'slope': round(slope, 4),
            'total_forecast': round(sum(predictions), 2)
        }

    def exponential_smoothing_forecast(
        self,
        product_id: str,
        alpha: float = 0.3,
        days_ahead: int = 30
    ) -> Dict:
        """
        Simple exponential smoothing forecast.

        Gives more weight to recent observations.

        Args:
            product_id: Product to forecast
            alpha: Smoothing factor (0-1)
            days_ahead: Days to forecast

        Returns:
            Dictionary with forecast results
        """
        historical = self.get_historical_demand(product_id, days=90)

        if len(historical) < 7:
            return {
                'product_id': product_id,
                'method': 'exponential_smoothing',
                'forecast_dates': [],
                'forecast_values': [],
                'error': 'Insufficient historical data'
            }

        # Calculate exponential smoothing
        values = historical['quantity'].values
        smoothed = [values[0]]

        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])

        # Last smoothed value is our forecast
        forecast_value = smoothed[-1]

        # Generate forecast dates
        last_date = historical['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )

        forecast_values = [round(forecast_value, 2)] * days_ahead

        return {
            'product_id': product_id,
            'method': 'exponential_smoothing',
            'alpha': alpha,
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'forecast_values': forecast_values,
            'daily_average': round(forecast_value, 2),
            'total_forecast': round(sum(forecast_values), 2)
        }

    def predict_demand(
        self,
        product_id: str,
        days_ahead: int = 30,
        method: str = 'auto'
    ) -> Dict:
        """
        Predict future demand using the best available method.

        Args:
            product_id: Product to forecast
            days_ahead: Number of days to forecast
            method: 'auto', 'moving_average', 'linear_regression', or 'exponential_smoothing'

        Returns:
            Dictionary with forecast results
        """
        if method == 'auto':
            # Try linear regression first, fall back to simpler methods
            result = self.linear_regression_forecast(product_id, days_ahead)
            if 'error' not in result:
                return result

            result = self.exponential_smoothing_forecast(product_id, days_ahead=days_ahead)
            if 'error' not in result:
                return result

            return self.moving_average_forecast(product_id, days_ahead=days_ahead)

        method_map = {
            'moving_average': self.moving_average_forecast,
            'linear_regression': self.linear_regression_forecast,
            'exponential_smoothing': self.exponential_smoothing_forecast
        }

        if method not in method_map:
            raise ValueError(f"Unknown method: {method}")

        return method_map[method](product_id, days_ahead=days_ahead)

    def forecast_category(
        self,
        category: str,
        days_ahead: int = 30
    ) -> Dict:
        """
        Forecast demand for an entire product category.

        Aggregates forecasts for all products in the category.

        Args:
            category: Product category name
            days_ahead: Days to forecast

        Returns:
            Dictionary with aggregated forecast
        """
        products = self.db.query(Product).filter(
            Product.category == category
        ).all()

        if not products:
            return {
                'category': category,
                'error': 'No products found in category'
            }

        total_forecast = []
        product_forecasts = []

        for product in products:
            forecast = self.predict_demand(product.id, days_ahead)
            if 'error' not in forecast:
                product_forecasts.append(forecast)
                if not total_forecast:
                    total_forecast = forecast['forecast_values'].copy()
                else:
                    for i, val in enumerate(forecast['forecast_values']):
                        total_forecast[i] += val

        if not product_forecasts:
            return {
                'category': category,
                'error': 'Could not generate forecasts for any products'
            }

        return {
            'category': category,
            'products_count': len(product_forecasts),
            'forecast_dates': product_forecasts[0]['forecast_dates'],
            'forecast_values': [round(v, 2) for v in total_forecast],
            'total_forecast': round(sum(total_forecast), 2)
        }

    def get_forecast_accuracy(
        self,
        product_id: str,
        days_back: int = 30
    ) -> Dict:
        """
        Calculate forecast accuracy by comparing predictions to actuals.

        Uses historical data to validate the model.
        """
        # Get historical data
        historical = self.get_historical_demand(product_id, days=60)

        if len(historical) < 45:
            return {'error': 'Insufficient data for accuracy calculation'}

        # Split into training and test
        train = historical.iloc[:-days_back]
        test = historical.iloc[-days_back:]

        # Make predictions on training data
        X_train = np.arange(len(train)).reshape(-1, 1)
        y_train = train['quantity'].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict test period
        X_test = np.arange(len(train), len(historical)).reshape(-1, 1)
        predictions = model.predict(X_test)
        actuals = test['quantity'].values

        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1))) * 100

        return {
            'product_id': product_id,
            'test_days': days_back,
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 2)
        }
