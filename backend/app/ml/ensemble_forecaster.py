"""
Hybrid Ensemble Forecasting Module

Combines SARIMA + Prophet + LSTM for superior demand forecasting.
Uses weighted averaging with dynamic weight optimization based on validation performance.

Target: SMAPE < 20%
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

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Import individual forecasters
from .sarima_forecaster import SARIMAForecaster
from .prophet_forecaster import ProphetForecaster
from .lstm_forecaster import LSTMForecaster

logger = logging.getLogger(__name__)

# Model storage path
MODEL_DIR = Path(__file__).parent.parent.parent / 'models' / 'ensemble'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class EnsembleForecaster:
    """
    Hybrid Ensemble Forecaster combining SARIMA, Prophet, and LSTM.

    Uses intelligent weight optimization:
    1. Train each model on historical data
    2. Validate on holdout period
    3. Assign weights inversely proportional to validation SMAPE
    4. Combine forecasts using optimized weights

    Attributes:
        sarima_weight: Weight for SARIMA predictions (0-1)
        prophet_weight: Weight for Prophet predictions (0-1)
        lstm_weight: Weight for LSTM predictions (0-1)
        optimization_method: 'validation' or 'equal'
    """

    def __init__(self,
                 optimization_method: str = 'validation',
                 validation_size: int = 8,  # 8 weeks for validation
                 use_log_transform: bool = True):
        """
        Initialize Ensemble Forecaster.

        Args:
            optimization_method: 'validation' (optimize weights) or 'equal' (1/3 each)
            validation_size: Number of periods for validation (weeks)
            use_log_transform: Apply log transformation for SARIMA
        """
        self.optimization_method = optimization_method
        self.validation_size = validation_size
        self.use_log_transform = use_log_transform

        # Initialize individual forecasters
        self.sarima = SARIMAForecaster(
            seasonal_period=52,
            use_log_transform=use_log_transform,
            resample_freq='W'
        )
        self.prophet = ProphetForecaster(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            yearly_seasonality=True,
            weekly_seasonality=True
        )
        self.lstm = LSTMForecaster(
            lookback=12,
            lstm_units=64,
            dropout_rate=0.2,
            epochs=100,
            batch_size=32
        )

        # Weights (will be optimized during fit)
        self.sarima_weight = 1/3
        self.prophet_weight = 1/3
        self.lstm_weight = 1/3

        # Model status
        self.is_fitted = False
        self.category = None
        self.model_performances = {}

    def _optimize_weights(self, train_data: pd.Series, val_data: pd.Series) -> Dict[str, float]:
        """
        Optimize ensemble weights based on validation performance.

        Uses inverse SMAPE weighting: better models get higher weights.

        Args:
            train_data: Training time series
            val_data: Validation time series

        Returns:
            Dictionary with optimized weights
        """
        logger.info("Optimizing ensemble weights on validation set...")

        smapes = {}
        predictions = {}

        # SARIMA validation
        try:
            sarima_temp = SARIMAForecaster(
                seasonal_period=52,
                use_log_transform=self.use_log_transform,
                resample_freq='W'
            )
            sarima_temp.fit(train_data)
            sarima_pred, _, _ = sarima_temp.predict(steps=len(val_data))
            sarima_pred.index = val_data.index
            smapes['sarima'] = self._calculate_smape(val_data.values, sarima_pred.values)
            predictions['sarima'] = sarima_pred
            logger.info(f"SARIMA validation SMAPE: {smapes['sarima']:.2f}%")
        except Exception as e:
            logger.warning(f"SARIMA validation failed: {e}")
            smapes['sarima'] = 100  # Penalty for failure

        # Prophet validation
        try:
            prophet_temp = ProphetForecaster(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            prophet_temp.fit(train_data)
            prophet_pred, _, _ = prophet_temp.predict(steps=len(val_data))
            prophet_pred.index = val_data.index
            smapes['prophet'] = self._calculate_smape(val_data.values, prophet_pred.values)
            predictions['prophet'] = prophet_pred
            logger.info(f"Prophet validation SMAPE: {smapes['prophet']:.2f}%")
        except Exception as e:
            logger.warning(f"Prophet validation failed: {e}")
            smapes['prophet'] = 100

        # LSTM validation
        try:
            lstm_temp = LSTMForecaster(lookback=12, lstm_units=64, epochs=50)
            lstm_temp.fit(train_data)
            lstm_pred, _, _ = lstm_temp.predict(steps=len(val_data))
            lstm_pred.index = val_data.index
            smapes['lstm'] = self._calculate_smape(val_data.values, lstm_pred.values)
            predictions['lstm'] = lstm_pred
            logger.info(f"LSTM validation SMAPE: {smapes['lstm']:.2f}%")
        except Exception as e:
            logger.warning(f"LSTM validation failed: {e}")
            smapes['lstm'] = 100

        # Calculate inverse SMAPE weights (lower SMAPE = higher weight)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        inverse_smapes = {k: 1 / (v + epsilon) for k, v in smapes.items()}
        total_inverse = sum(inverse_smapes.values())

        weights = {
            'sarima': inverse_smapes['sarima'] / total_inverse,
            'prophet': inverse_smapes['prophet'] / total_inverse,
            'lstm': inverse_smapes['lstm'] / total_inverse
        }

        # Store performances
        self.model_performances = {
            'smapes': smapes,
            'weights': weights,
            'predictions': predictions
        }

        logger.info(f"Optimized weights - SARIMA: {weights['sarima']:.3f}, "
                    f"Prophet: {weights['prophet']:.3f}, LSTM: {weights['lstm']:.3f}")

        return weights

    def fit(self, data: pd.Series, category: str = None) -> 'EnsembleForecaster':
        """
        Fit all ensemble models to time series data.

        Args:
            data: Time series with DatetimeIndex (weekly frequency)
            category: Category name for logging

        Returns:
            Self for method chaining
        """
        self.category = category or "Unknown"
        logger.info(f"Fitting ensemble model for {self.category}")

        # Ensure data is sorted
        data = data.sort_index()

        # Split data for weight optimization
        if self.optimization_method == 'validation' and len(data) > self.validation_size + 20:
            train_data = data[:-self.validation_size]
            val_data = data[-self.validation_size:]

            # Optimize weights
            weights = self._optimize_weights(train_data, val_data)
            self.sarima_weight = weights['sarima']
            self.prophet_weight = weights['prophet']
            self.lstm_weight = weights['lstm']

        # Fit all models on full data
        logger.info("Fitting SARIMA on full data...")
        try:
            self.sarima.fit(data)
        except Exception as e:
            logger.error(f"SARIMA fit failed: {e}")
            self.sarima_weight = 0

        logger.info("Fitting Prophet on full data...")
        try:
            self.prophet.fit(data)
        except Exception as e:
            logger.error(f"Prophet fit failed: {e}")
            self.prophet_weight = 0

        logger.info("Fitting LSTM on full data...")
        try:
            self.lstm.fit(data)
        except Exception as e:
            logger.error(f"LSTM fit failed: {e}")
            self.lstm_weight = 0

        # Renormalize weights if any model failed
        total_weight = self.sarima_weight + self.prophet_weight + self.lstm_weight
        if total_weight > 0 and total_weight != 1:
            self.sarima_weight /= total_weight
            self.prophet_weight /= total_weight
            self.lstm_weight /= total_weight

        self.is_fitted = True
        logger.info(f"Ensemble fitting complete for {self.category}")

        return self

    def predict(self, steps: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate ensemble forecast for specified number of steps.

        Combines predictions from all models using optimized weights.

        Args:
            steps: Number of periods to forecast (weeks)

        Returns:
            Tuple of (ensemble_forecast, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        forecasts = []
        lowers = []
        uppers = []
        weights = []

        # SARIMA prediction
        if self.sarima_weight > 0:
            try:
                sarima_pred, sarima_lower, sarima_upper = self.sarima.predict(steps=steps)
                forecasts.append(sarima_pred.values)
                lowers.append(sarima_lower.values)
                uppers.append(sarima_upper.values)
                weights.append(self.sarima_weight)
                forecast_index = sarima_pred.index
            except Exception as e:
                logger.warning(f"SARIMA prediction failed: {e}")

        # Prophet prediction
        if self.prophet_weight > 0:
            try:
                prophet_pred, prophet_lower, prophet_upper = self.prophet.predict(steps=steps)
                forecasts.append(prophet_pred.values)
                lowers.append(prophet_lower.values)
                uppers.append(prophet_upper.values)
                weights.append(self.prophet_weight)
                if 'forecast_index' not in dir():
                    forecast_index = prophet_pred.index
            except Exception as e:
                logger.warning(f"Prophet prediction failed: {e}")

        # LSTM prediction
        if self.lstm_weight > 0:
            try:
                lstm_pred, lstm_lower, lstm_upper = self.lstm.predict(steps=steps)
                forecasts.append(lstm_pred.values)
                lowers.append(lstm_lower.values)
                uppers.append(lstm_upper.values)
                weights.append(self.lstm_weight)
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")

        if not forecasts:
            raise ValueError("All models failed to produce predictions")

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted ensemble combination
        ensemble_forecast = np.zeros(steps)
        ensemble_lower = np.zeros(steps)
        ensemble_upper = np.zeros(steps)

        for i, (f, l, u, w) in enumerate(zip(forecasts, lowers, uppers, weights)):
            ensemble_forecast += w * f
            ensemble_lower += w * l
            ensemble_upper += w * u

        # Ensure non-negative
        ensemble_forecast = np.maximum(ensemble_forecast, 0)
        ensemble_lower = np.maximum(ensemble_lower, 0)
        ensemble_upper = np.maximum(ensemble_upper, 0)

        return (
            pd.Series(ensemble_forecast, index=forecast_index if 'forecast_index' in dir() else None),
            pd.Series(ensemble_lower, index=forecast_index if 'forecast_index' in dir() else None),
            pd.Series(ensemble_upper, index=forecast_index if 'forecast_index' in dir() else None)
        )

    def evaluate(self, train_data: pd.Series, test_data: pd.Series) -> Dict[str, Any]:
        """
        Evaluate ensemble and individual model performance.

        Args:
            train_data: Training time series
            test_data: Test time series

        Returns:
            Dictionary with metrics for ensemble and each model
        """
        # Fit on training data
        self.fit(train_data)

        # Predict for test period
        ensemble_pred, _, _ = self.predict(steps=len(test_data))
        ensemble_pred.index = test_data.index

        # Calculate ensemble metrics
        ensemble_metrics = {
            'mae': mean_absolute_error(test_data.values, ensemble_pred.values),
            'rmse': np.sqrt(mean_squared_error(test_data.values, ensemble_pred.values)),
            'smape': self._calculate_smape(test_data.values, ensemble_pred.values)
        }

        return {
            'ensemble': ensemble_metrics,
            'weights': {
                'sarima': self.sarima_weight,
                'prophet': self.prophet_weight,
                'lstm': self.lstm_weight
            },
            'individual_smapes': self.model_performances.get('smapes', {})
        }

    def _calculate_smape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.

        SMAPE = (100/n) * Σ(|actual - predicted| / ((|actual| + |predicted|) / 2))
        """
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
        denominator = np.where(denominator == 0, 1, denominator)
        smape = np.mean(np.abs(actual - predicted) / denominator) * 100
        return smape

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return {
            'sarima': self.sarima_weight,
            'prophet': self.prophet_weight,
            'lstm': self.lstm_weight
        }

    def save(self, filepath: str = None) -> str:
        """
        Save ensemble model to disk.

        Args:
            filepath: Optional custom path

        Returns:
            Path where model was saved
        """
        if filepath is None:
            filepath = MODEL_DIR / f"ensemble_{self.category}_{datetime.now().strftime('%Y%m%d')}.pkl"

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Ensemble model saved to {filepath}")
        return str(filepath)

    @classmethod
    def load(cls, filepath: str) -> 'EnsembleForecaster':
        """
        Load ensemble model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded EnsembleForecaster instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Ensemble model loaded from {filepath}")
        return model


class CategoryEnsembleForecaster:
    """
    Manages ensemble forecasters for multiple product categories.

    Trains and maintains separate ensemble models for each category,
    enabling category-specific forecasting with optimized weights.
    """

    def __init__(self, validation_size: int = 8):
        """
        Initialize Category Ensemble Forecaster.

        Args:
            validation_size: Weeks for validation when optimizing weights
        """
        self.validation_size = validation_size
        self.forecasters: Dict[str, EnsembleForecaster] = {}
        self.results: Dict[str, Dict] = {}

    def fit_all_categories(self, data: pd.DataFrame,
                           date_col: str = 'order_date',
                           category_col: str = 'category',
                           value_col: str = 'quantity') -> 'CategoryEnsembleForecaster':
        """
        Fit ensemble models for all categories in the data.

        Args:
            data: DataFrame with date, category, and quantity columns
            date_col: Name of date column
            category_col: Name of category column
            value_col: Name of value column (quantity/demand)

        Returns:
            Self for method chaining
        """
        # Aggregate to weekly by category
        data[date_col] = pd.to_datetime(data[date_col])

        categories = data[category_col].unique()
        logger.info(f"Training ensemble models for {len(categories)} categories")

        for category in categories:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training ensemble for: {category}")
            logger.info(f"{'='*50}")

            # Filter and aggregate
            cat_data = data[data[category_col] == category].copy()
            weekly_demand = cat_data.groupby(pd.Grouper(key=date_col, freq='W'))[value_col].sum()
            weekly_demand = weekly_demand[weekly_demand > 0]  # Remove zero weeks

            if len(weekly_demand) < 30:  # Need enough data
                logger.warning(f"Skipping {category}: insufficient data ({len(weekly_demand)} weeks)")
                continue

            # Create and fit ensemble
            ensemble = EnsembleForecaster(
                optimization_method='validation',
                validation_size=self.validation_size
            )

            try:
                ensemble.fit(weekly_demand, category=category)
                self.forecasters[category] = ensemble

                # Store results
                self.results[category] = {
                    'weights': ensemble.get_model_weights(),
                    'data_points': len(weekly_demand),
                    'status': 'success'
                }
            except Exception as e:
                logger.error(f"Failed to train ensemble for {category}: {e}")
                self.results[category] = {'status': 'failed', 'error': str(e)}

        return self

    def predict_category(self, category: str, steps: int = 4) -> Dict[str, Any]:
        """
        Generate forecast for a specific category.

        Args:
            category: Category name
            steps: Number of weeks to forecast

        Returns:
            Dictionary with forecast, bounds, and weights
        """
        if category not in self.forecasters:
            raise ValueError(f"No model found for category: {category}")

        forecaster = self.forecasters[category]
        forecast, lower, upper = forecaster.predict(steps=steps)

        return {
            'category': category,
            'forecast': forecast.tolist(),
            'lower_bound': lower.tolist(),
            'upper_bound': upper.tolist(),
            'weights': forecaster.get_model_weights(),
            'horizon_weeks': steps
        }

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.

        Returns:
            DataFrame with category, weights, and status
        """
        rows = []
        for category, result in self.results.items():
            if result['status'] == 'success':
                rows.append({
                    'category': category,
                    'sarima_weight': result['weights']['sarima'],
                    'prophet_weight': result['weights']['prophet'],
                    'lstm_weight': result['weights']['lstm'],
                    'data_points': result['data_points'],
                    'status': 'success'
                })
            else:
                rows.append({
                    'category': category,
                    'sarima_weight': None,
                    'prophet_weight': None,
                    'lstm_weight': None,
                    'data_points': None,
                    'status': 'failed'
                })

        return pd.DataFrame(rows)
