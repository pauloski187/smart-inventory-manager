"""
Prophet Forecasting Module

Facebook Prophet implementation for demand forecasting.
Handles trend changes, seasonality, and holidays automatically.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Prophet-based forecaster for demand prediction.

    Automatically detects:
    - Trend changes (changepoint detection)
    - Weekly seasonality
    - Yearly seasonality
    - Holiday effects

    Attributes:
        changepoint_prior_scale: Flexibility of trend changes
        seasonality_prior_scale: Flexibility of seasonality
        yearly_seasonality: Enable yearly patterns
        weekly_seasonality: Enable weekly patterns
    """

    def __init__(self,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False):
        """
        Initialize Prophet forecaster.

        Args:
            changepoint_prior_scale: Controls trend flexibility (higher = more flexible)
            seasonality_prior_scale: Controls seasonality strength
            yearly_seasonality: Enable yearly seasonal patterns
            weekly_seasonality: Enable weekly seasonal patterns
            daily_seasonality: Enable daily patterns (usually False for aggregated data)
        """
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        self.category = None

    def fit(self, data: pd.Series) -> 'ProphetForecaster':
        """
        Fit Prophet model to time series data.

        Args:
            data: Time series with DatetimeIndex

        Returns:
            Self for method chaining
        """
        # Prepare data in Prophet format (ds, y)
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })

        # Initialize Prophet model
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=0.95  # 95% confidence interval
        )

        # Fit the model
        logger.info(f"Fitting Prophet model with {len(df)} observations")
        self.model.fit(df)

        return self

    def predict(self, steps: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate forecast for specified number of steps.

        Args:
            steps: Number of periods to forecast

        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps, freq='W')

        # Generate forecast
        forecast = self.model.predict(future)

        # Extract predictions (only future values)
        forecast_df = forecast.tail(steps)

        # Ensure non-negative predictions
        yhat = np.maximum(forecast_df['yhat'].values, 0)
        yhat_lower = np.maximum(forecast_df['yhat_lower'].values, 0)
        yhat_upper = np.maximum(forecast_df['yhat_upper'].values, 0)

        # Create series with proper index
        forecast_index = forecast_df['ds'].values

        return (
            pd.Series(yhat, index=forecast_index),
            pd.Series(yhat_lower, index=forecast_index),
            pd.Series(yhat_upper, index=forecast_index)
        )

    def evaluate(self, train_data: pd.Series, test_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            train_data: Training time series
            test_data: Test time series

        Returns:
            Dictionary with MAE, RMSE, SMAPE metrics
        """
        # Fit on training data
        self.fit(train_data)

        # Predict for test period
        forecast, _, _ = self.predict(steps=len(test_data))

        # Calculate metrics
        mae = mean_absolute_error(test_data.values, forecast.values)
        rmse = np.sqrt(mean_squared_error(test_data.values, forecast.values))
        smape = self._calculate_smape(test_data.values, forecast.values)

        return {
            'mae': mae,
            'rmse': rmse,
            'smape': smape
        }

    def _calculate_smape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.

        SMAPE = (100/n) * Σ(|actual - predicted| / ((|actual| + |predicted|) / 2))
        """
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1, denominator)
        smape = np.mean(np.abs(actual - predicted) / denominator) * 100
        return smape

    def get_components(self) -> pd.DataFrame:
        """
        Get decomposed time series components (trend, seasonality).

        Returns:
            DataFrame with trend, weekly, yearly components
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting components")

        # Get forecast with components
        future = self.model.make_future_dataframe(periods=0, freq='W')
        forecast = self.model.predict(future)

        return forecast[['ds', 'trend', 'weekly', 'yearly']]
