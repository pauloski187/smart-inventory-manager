"""
LSTM Forecasting Module

Deep learning LSTM implementation for demand forecasting.
Captures complex non-linear patterns and long-term dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class LSTMForecaster:
    """
    LSTM-based forecaster for demand prediction.

    Uses sliding window approach to create sequences for training.
    Captures complex temporal patterns and non-linear relationships.

    Attributes:
        lookback: Number of past timesteps to use for prediction
        lstm_units: Number of LSTM units in each layer
        dropout_rate: Dropout rate for regularization
        epochs: Maximum training epochs
        batch_size: Training batch size
    """

    def __init__(self,
                 lookback: int = 12,  # 12 weeks lookback
                 lstm_units: int = 64,
                 dropout_rate: float = 0.2,
                 epochs: int = 100,
                 batch_size: int = 32,
                 validation_split: float = 0.2):
        """
        Initialize LSTM forecaster.

        Args:
            lookback: Number of past timesteps to use (weeks)
            lstm_units: Number of LSTM units per layer
            dropout_rate: Dropout rate for regularization
            epochs: Maximum training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.category = None
        self._last_sequence = None  # For iterative forecasting

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            data: 1D array of time series values

        Returns:
            Tuple of (X, y) where X has shape (samples, lookback, 1)
        """
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback])

        return np.array(X), np.array(y)

    def _build_model(self) -> keras.Model:
        """
        Build LSTM neural network architecture.

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(self.lstm_units, return_sequences=True,
                 input_shape=(self.lookback, 1)),
            Dropout(self.dropout_rate),

            # Second LSTM layer
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),

            # Dense output layer
            Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, data: pd.Series) -> 'LSTMForecaster':
        """
        Fit LSTM model to time series data.

        Args:
            data: Time series with DatetimeIndex

        Returns:
            Self for method chaining
        """
        # Scale data to [0, 1]
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))

        # Create sequences
        X, y = self._create_sequences(scaled_data)

        if len(X) < self.lookback:
            raise ValueError(f"Not enough data. Need at least {self.lookback + 1} points.")

        # Reshape X for LSTM [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build model
        self.model = self._build_model()

        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        logger.info(f"Training LSTM with {len(X)} sequences")
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stop],
            verbose=0
        )

        # Store last sequence for future predictions
        self._last_sequence = scaled_data[-self.lookback:].flatten()

        logger.info(f"LSTM training completed. Final loss: {history.history['loss'][-1]:.4f}")

        return self

    def predict(self, steps: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate forecast for specified number of steps.

        Uses iterative approach: predict one step, append to sequence, repeat.

        Args:
            steps: Number of periods to forecast

        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        current_sequence = self._last_sequence.copy()

        # Iterative forecasting
        for _ in range(steps):
            # Reshape for prediction
            X = current_sequence.reshape((1, self.lookback, 1))

            # Predict next value
            pred_scaled = self.model.predict(X, verbose=0)[0, 0]

            # Store prediction
            predictions.append(pred_scaled)

            # Update sequence (shift and append new prediction)
            current_sequence = np.append(current_sequence[1:], pred_scaled)

        # Inverse transform to original scale
        predictions_array = np.array(predictions).reshape(-1, 1)
        forecast = self.scaler.inverse_transform(predictions_array).flatten()

        # Ensure non-negative predictions
        forecast = np.maximum(forecast, 0)

        # Calculate confidence intervals (±20% for LSTM uncertainty)
        std_dev = np.std(forecast) * 0.2
        lower_bound = np.maximum(forecast - 1.96 * std_dev, 0)
        upper_bound = forecast + 1.96 * std_dev

        # Create series (index will be set by ensemble)
        return (
            pd.Series(forecast),
            pd.Series(lower_bound),
            pd.Series(upper_bound)
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
