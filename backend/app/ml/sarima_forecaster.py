"""
SARIMA Demand Forecasting Module

Production-ready SARIMA model for category-level demand forecasting.
Implements time series analysis with seasonal patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import logging
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Model storage path
MODEL_DIR = Path(__file__).parent.parent.parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)


class SARIMAForecaster:
    """
    Production-ready SARIMA forecaster for demand prediction.

    SARIMA(p,d,q)(P,D,Q,s) model with weekly seasonality.

    Attributes:
        seasonal_period: Seasonality period (7 for weekly)
        order: Non-seasonal ARIMA order (p,d,q)
        seasonal_order: Seasonal order (P,D,Q,s)
    """

    def __init__(self, seasonal_period: int = 7):
        """Initialize forecaster with default weekly seasonality."""
        self.seasonal_period = seasonal_period
        self.order = (1, 1, 1)  # Default ARIMA order
        self.seasonal_order = (1, 1, 1, seasonal_period)
        self.model = None
        self.fitted_model = None
        self.category = None

    def fit(self, data: pd.Series, order: Tuple = None,
            seasonal_order: Tuple = None) -> 'SARIMAForecaster':
        """
        Fit SARIMA model to time series data.

        Args:
            data: Time series with DatetimeIndex
            order: ARIMA order (p,d,q), default (1,1,1)
            seasonal_order: Seasonal order (P,D,Q,s), default (1,1,1,7)

        Returns:
            Self for method chaining
        """
        if order:
            self.order = order
        if seasonal_order:
            self.seasonal_order = seasonal_order

        try:
            self.model = SARIMAX(
                data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=False, maxiter=200)
            logger.info(f"SARIMA model fitted: AIC={self.fitted_model.aic:.2f}")
        except Exception as e:
            logger.error(f"Error fitting SARIMA: {e}")
            raise

        return self

    def forecast(self, steps: int, alpha: float = 0.05) -> pd.DataFrame:
        """
        Generate forecast with confidence intervals.

        Args:
            steps: Number of periods to forecast
            alpha: Significance level for CI (default 5%)

        Returns:
            DataFrame with Forecast, Lower_CI, Upper_CI columns
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        forecast_df = pd.DataFrame({
            'Forecast': forecast_mean.values,
            'Lower_CI': conf_int.iloc[:, 0].values,
            'Upper_CI': conf_int.iloc[:, 1].values
        }, index=forecast_mean.index)

        # Demand cannot be negative
        forecast_df['Forecast'] = forecast_df['Forecast'].clip(lower=0)
        forecast_df['Lower_CI'] = forecast_df['Lower_CI'].clip(lower=0)

        return forecast_df

    def evaluate(self, test_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            test_data: Actual test values

        Returns:
            Dictionary with MAE, RMSE, MAPE, SMAPE metrics
        """
        forecast_df = self.forecast(steps=len(test_data))
        predictions = forecast_df['Forecast'].values
        actual = test_data.values

        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))

        # SMAPE (handles zeros better)
        smape = np.mean(2 * np.abs(actual - predictions) /
                       (np.abs(actual) + np.abs(predictions) + 1)) * 100

        # MAPE excluding zeros
        mask = actual > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predictions[mask]) /
                          actual[mask])) * 100
        else:
            mape = np.nan

        return {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2) if not np.isnan(mape) else None,
            'SMAPE': round(smape, 2)
        }

    def check_residuals(self) -> Dict[str, Any]:
        """
        Perform residual diagnostics.

        Returns:
            Dictionary with Ljung-Box test results
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        residuals = self.fitted_model.resid
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].values[0]

        return {
            'residual_mean': round(residuals.mean(), 4),
            'residual_std': round(residuals.std(), 4),
            'ljung_box_pvalue': round(lb_pvalue, 4),
            'no_autocorrelation': lb_pvalue > 0.05
        }

    def save(self, filepath: Path) -> None:
        """Save fitted model to disk."""
        if self.fitted_model is None:
            raise ValueError("No model to save")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'fitted_model': self.fitted_model,
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'category': self.category
            }, f)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'SARIMAForecaster':
        """Load fitted model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        forecaster = cls()
        forecaster.fitted_model = data['fitted_model']
        forecaster.order = data['order']
        forecaster.seasonal_order = data['seasonal_order']
        forecaster.category = data.get('category')
        return forecaster


class CategoryForecaster:
    """
    Manager for category-level SARIMA forecasts.

    Handles data preparation, model training, and forecasting
    for product categories.
    """

    def __init__(self, df: pd.DataFrame = None):
        """
        Initialize with order data.

        Args:
            df: DataFrame with OrderDate, Quantity, Category columns
        """
        self.df = df
        self.models: Dict[str, SARIMAForecaster] = {}
        self.forecasts: Dict[str, pd.DataFrame] = {}

    def prepare_category_data(self, category: str) -> pd.Series:
        """
        Prepare daily demand time series for a category.

        Args:
            category: Category name

        Returns:
            Series with daily demand indexed by date
        """
        if self.df is None:
            raise ValueError("No data loaded")

        cat_df = self.df[self.df['Category'] == category].copy()

        if 'OrderDate' in cat_df.columns:
            date_col = 'OrderDate'
        elif 'order_date' in cat_df.columns:
            date_col = 'order_date'
        else:
            raise ValueError("No date column found")

        cat_df[date_col] = pd.to_datetime(cat_df[date_col])

        qty_col = 'Quantity' if 'Quantity' in cat_df.columns else 'quantity'

        daily = cat_df.groupby(cat_df[date_col].dt.date)[qty_col].sum()
        daily.index = pd.to_datetime(daily.index)

        # Fill missing dates with 0
        date_range = pd.date_range(
            start=daily.index.min(),
            end=daily.index.max(),
            freq='D'
        )
        daily = daily.reindex(date_range, fill_value=0)
        daily.name = 'Quantity'

        return daily

    def check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test.

        Args:
            series: Time series data

        Returns:
            Dictionary with ADF test results
        """
        result = adfuller(series.dropna(), autolag='AIC')

        return {
            'adf_statistic': round(result[0], 4),
            'p_value': round(result[1], 6),
            'is_stationary': result[1] < 0.05,
            'critical_values': {k: round(v, 4) for k, v in result[4].items()}
        }

    def train_category(self, category: str,
                       train_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Train SARIMA model for a category.

        Args:
            category: Category name
            train_ratio: Train/test split ratio

        Returns:
            Dictionary with model info and metrics
        """
        data = self.prepare_category_data(category)

        # Train-test split
        split_idx = int(len(data) * train_ratio)
        train = data[:split_idx]
        test = data[split_idx:]

        # Train model
        forecaster = SARIMAForecaster(seasonal_period=7)
        forecaster.category = category
        forecaster.fit(train)

        # Evaluate
        metrics = forecaster.evaluate(test)
        diagnostics = forecaster.check_residuals()

        # Store model
        self.models[category] = forecaster

        # Save model to disk
        model_path = MODEL_DIR / f'sarima_{category.replace(" ", "_").replace("&", "and")}.pkl'
        forecaster.save(model_path)

        return {
            'category': category,
            'train_size': len(train),
            'test_size': len(test),
            'metrics': metrics,
            'diagnostics': diagnostics,
            'model_path': str(model_path)
        }

    def train_all_categories(self) -> List[Dict[str, Any]]:
        """Train models for all categories."""
        if self.df is None:
            raise ValueError("No data loaded")

        categories = self.df['Category'].unique() if 'Category' in self.df.columns \
                     else self.df['category'].unique()

        results = []
        for category in categories:
            try:
                result = self.train_category(category)
                results.append(result)
                logger.info(f"Trained model for {category}")
            except Exception as e:
                logger.error(f"Error training {category}: {e}")
                results.append({
                    'category': category,
                    'error': str(e)
                })

        return results

    def forecast_category(self, category: str,
                         horizons: List[int] = [30, 60, 90]) -> Dict[str, Any]:
        """
        Generate forecasts for a category.

        Args:
            category: Category name
            horizons: List of forecast horizons (days)

        Returns:
            Dictionary with forecasts for each horizon
        """
        # Load model if not in memory
        if category not in self.models:
            model_path = MODEL_DIR / f'sarima_{category.replace(" ", "_").replace("&", "and")}.pkl'
            if model_path.exists():
                self.models[category] = SARIMAForecaster.load(model_path)
            else:
                # Train on the fly
                self.train_category(category)

        forecaster = self.models[category]

        # Get the longest horizon forecast
        max_horizon = max(horizons)
        forecast_df = forecaster.forecast(steps=max_horizon)

        results = {
            'category': category,
            'generated_at': datetime.now().isoformat(),
            'forecasts': {}
        }

        for horizon in horizons:
            horizon_forecast = forecast_df.head(horizon)
            results['forecasts'][f'{horizon}_day'] = {
                'total_forecast': round(horizon_forecast['Forecast'].sum(), 0),
                'daily_average': round(horizon_forecast['Forecast'].mean(), 2),
                'lower_ci_total': round(horizon_forecast['Lower_CI'].sum(), 0),
                'upper_ci_total': round(horizon_forecast['Upper_CI'].sum(), 0),
                'daily_forecast': [
                    {
                        'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                        'forecast': round(row['Forecast'], 2),
                        'lower_ci': round(row['Lower_CI'], 2),
                        'upper_ci': round(row['Upper_CI'], 2)
                    }
                    for idx, row in horizon_forecast.iterrows()
                ]
            }

        self.forecasts[category] = results
        return results

    def forecast_all_categories(self, horizons: List[int] = [30, 60, 90]) -> Dict[str, Any]:
        """Generate forecasts for all categories."""
        if self.df is None:
            # Try to load from stored models
            stored_models = list(MODEL_DIR.glob('sarima_*.pkl'))
            if not stored_models:
                raise ValueError("No data or models available")

            all_forecasts = {}
            for model_path in stored_models:
                try:
                    forecaster = SARIMAForecaster.load(model_path)
                    category = forecaster.category or model_path.stem.replace('sarima_', '').replace('_', ' ')
                    self.models[category] = forecaster
                    forecast_df = forecaster.forecast(steps=max(horizons))

                    result = {'category': category, 'forecasts': {}}
                    for horizon in horizons:
                        horizon_forecast = forecast_df.head(horizon)
                        result['forecasts'][f'{horizon}_day'] = {
                            'total_forecast': round(horizon_forecast['Forecast'].sum(), 0),
                            'lower_ci_total': round(horizon_forecast['Lower_CI'].sum(), 0),
                            'upper_ci_total': round(horizon_forecast['Upper_CI'].sum(), 0)
                        }
                    all_forecasts[category] = result
                except Exception as e:
                    logger.error(f"Error loading {model_path}: {e}")

            return all_forecasts

        categories = self.df['Category'].unique() if 'Category' in self.df.columns \
                     else self.df['category'].unique()

        all_forecasts = {}
        for category in categories:
            try:
                all_forecasts[category] = self.forecast_category(category, horizons)
            except Exception as e:
                logger.error(f"Error forecasting {category}: {e}")
                all_forecasts[category] = {'error': str(e)}

        return all_forecasts

    def get_inventory_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate inventory recommendations based on forecasts.

        Returns:
            List of recommendations by category
        """
        recommendations = []

        for category, forecaster in self.models.items():
            try:
                forecast_df = forecaster.forecast(steps=90)

                # Calculate metrics
                forecast_30 = forecast_df.head(30)['Forecast'].sum()
                forecast_90 = forecast_df['Forecast'].sum()
                daily_avg = forecast_df['Forecast'].mean()
                upper_ci_90 = forecast_df['Upper_CI'].sum()

                # Safety stock: difference between upper CI and forecast
                safety_stock = round((upper_ci_90 - forecast_90) / 3, 0)  # ~1 month safety

                # Reorder point: 2 weeks demand + safety stock
                reorder_point = round(daily_avg * 14 + safety_stock, 0)

                # Risk assessment
                ci_width = upper_ci_90 - forecast_df['Lower_CI'].sum()
                cv = ci_width / forecast_90 if forecast_90 > 0 else 0

                if cv < 0.3:
                    stockout_risk = 'low'
                elif cv < 0.5:
                    stockout_risk = 'medium'
                else:
                    stockout_risk = 'high'

                recommendations.append({
                    'category': category,
                    'forecast_90_day': round(forecast_90, 0),
                    'confidence_interval': {
                        'lower': round(forecast_df['Lower_CI'].sum(), 0),
                        'upper': round(upper_ci_90, 0)
                    },
                    'daily_average': round(daily_avg, 2),
                    'reorder_point': reorder_point,
                    'safety_stock': safety_stock,
                    'stockout_risk': stockout_risk,
                    'recommended_order_quantity': round(forecast_30 * 1.2, 0)  # 30 days + 20% buffer
                })

            except Exception as e:
                logger.error(f"Error generating recommendation for {category}: {e}")
                recommendations.append({
                    'category': category,
                    'error': str(e)
                })

        return recommendations


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate CSV data for forecasting.

    Args:
        filepath: Path to CSV file

    Returns:
        Cleaned DataFrame ready for forecasting
    """
    df = pd.read_csv(filepath)

    # Standardize column names
    column_map = {
        'order_date': 'OrderDate',
        'orderdate': 'OrderDate',
        'date': 'OrderDate',
        'quantity': 'Quantity',
        'qty': 'Quantity',
        'product_id': 'ProductID',
        'productid': 'ProductID',
        'category': 'Category',
        'order_status': 'OrderStatus',
        'orderstatus': 'OrderStatus'
    }

    df.columns = [column_map.get(c.lower(), c) for c in df.columns]

    # Parse dates
    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])

    # Remove cancelled orders if status column exists
    if 'OrderStatus' in df.columns:
        df = df[df['OrderStatus'] != 'Cancelled']

    return df
