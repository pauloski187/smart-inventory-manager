#!/usr/bin/env python3
"""
Demand Forecasting Model Training Pipeline

This script trains and saves the demand forecasting model for inventory optimization.
It includes data preprocessing, feature engineering, model training, and evaluation.

Usage:
    python scripts/train_forecast_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the raw sales data.

    Args:
        data_path (Path): Path to the raw data file

    Returns:
        pd.DataFrame: Preprocessed data
    """
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Convert date column
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Basic preprocessing
    df = df.dropna()  # Remove missing values
    df = df[df['quantity'] > 0]  # Remove negative quantities

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def create_time_features(df):
    """
    Create time-based features for forecasting.

    Args:
        df (pd.DataFrame): Input dataframe with order_date

    Returns:
        pd.DataFrame: DataFrame with additional time features
    """
    df = df.copy()

    # Extract time components
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['day'] = df['order_date'].dt.day
    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['quarter'] = df['order_date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Create lag features (7-day and 30-day lags)
    df = df.sort_values(['product_id', 'order_date'])

    # Group by product and create lag features
    df['quantity_lag_7'] = df.groupby('product_id')['quantity'].shift(7)
    df['quantity_lag_30'] = df.groupby('product_id')['quantity'].shift(30)

    # Rolling statistics
    df['quantity_rolling_mean_7'] = df.groupby('product_id')['quantity'].rolling(7).mean().reset_index(0, drop=True)
    df['quantity_rolling_std_7'] = df.groupby('product_id')['quantity'].rolling(7).std().reset_index(0, drop=True)
    df['quantity_rolling_mean_30'] = df.groupby('product_id')['quantity'].rolling(30).mean().reset_index(0, drop=True)

    # Fill NaN values created by lag features
    df = df.fillna(0)

    return df

def create_product_features(df):
    """
    Create product-specific features.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: DataFrame with product features
    """
    df = df.copy()

    # Product statistics
    product_stats = df.groupby('product_id').agg({
        'quantity': ['mean', 'std', 'min', 'max', 'sum'],
        'unit_price': ['mean', 'std'],
        'order_id': 'count'
    }).fillna(0)

    # Flatten column names
    product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns]
    product_stats = product_stats.reset_index()

    # Merge back to main dataframe
    df = df.merge(product_stats, on='product_id', how='left')

    return df

def prepare_features_and_target(df, target_days_ahead=7):
    """
    Prepare features and target variable for forecasting.

    Args:
        df (pd.DataFrame): Input dataframe
        target_days_ahead (int): Days ahead to forecast

    Returns:
        tuple: (X, y) features and target
    """
    # Create target: future demand
    df = df.sort_values(['product_id', 'order_date'])
    df[f'quantity_future_{target_days_ahead}d'] = df.groupby('product_id')['quantity'].shift(-target_days_ahead)

    # Remove rows where target is NaN (last target_days_ahead days)
    df = df.dropna(subset=[f'quantity_future_{target_days_ahead}d'])

    # Select features
    feature_columns = [
        # Time features
        'month', 'day', 'day_of_week', 'quarter', 'is_weekend',

        # Lag features
        'quantity_lag_7', 'quantity_lag_30',

        # Rolling features
        'quantity_rolling_mean_7', 'quantity_rolling_std_7', 'quantity_rolling_mean_30',

        # Product statistics
        'quantity_mean', 'quantity_std', 'quantity_min', 'quantity_max', 'quantity_sum',
        'unit_price_mean', 'unit_price_std', 'order_id_count',

        # Current values
        'unit_price', 'quantity'
    ]

    X = df[feature_columns]
    y = df[f'quantity_future_{target_days_ahead}d']

    return X, y

def train_model(X_train, y_train):
    """
    Train the forecasting model.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target

    Returns:
        Pipeline: Trained model pipeline
    """
    print("Training model...")

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    print("Model training completed.")
    return pipeline

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target

    Returns:
        dict: Evaluation metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

    print("Evaluation Results:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return metrics

def save_model_and_data(model, X_train, X_test, y_train, y_test, metrics, output_dir):
    """
    Save the trained model and processed data.

    Args:
        model: Trained model
        X_train, X_test, y_train, y_test: Train/test data
        metrics (dict): Evaluation metrics
        output_dir (Path): Output directory
    """
    print("Saving model and data...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / 'demand_forecast_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save processed data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_path = output_dir / 'train_features.csv'
    test_path = output_dir / 'test_features.csv'

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Training data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")

    # Save metrics
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['metric', 'value'])
    metrics_path = output_dir / 'model_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

def main():
    """Main training pipeline."""
    print("=== Demand Forecasting Model Training Pipeline ===\n")

    # Define paths
    data_path = Path('../data/raw/amazon_orders.csv')
    output_dir = Path('../models')

    # Check if data exists
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    try:
        # Load and preprocess data
        df = load_and_preprocess_data(data_path)

        # Create features
        print("Creating features...")
        df = create_time_features(df)
        df = create_product_features(df)

        # Prepare features and target
        print("Preparing features and target...")
        X, y = prepare_features_and_target(df)

        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Save everything
        save_model_and_data(model, X_train, X_test, y_train, y_test, metrics, output_dir)

        print("\n✅ Training pipeline completed successfully!")
        print(f"Model performance: MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()