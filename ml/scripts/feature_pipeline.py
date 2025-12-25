#!/usr/bin/env python3
"""
Feature Engineering Pipeline

This script creates and processes features for the demand forecasting model.
It includes time-based features, lag features, rolling statistics, and product-specific features.

Usage:
    python scripts/feature_pipeline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class FeaturePipeline:
    """
    Feature engineering pipeline for demand forecasting.
    """

    def __init__(self, target_days_ahead=7):
        """
        Initialize the feature pipeline.

        Args:
            target_days_ahead (int): Days ahead to forecast
        """
        self.target_days_ahead = target_days_ahead
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None

    def load_data(self, data_path):
        """
        Load and perform initial data cleaning.

        Args:
            data_path (Path): Path to raw data

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        print("Loading data...")
        df = pd.read_csv(data_path)

        # Convert date column
        df['order_date'] = pd.to_datetime(df['order_date'])

        # Basic data cleaning
        df = df.dropna()  # Remove missing values
        df = df[df['quantity'] > 0]  # Remove negative quantities
        df = df[df['unit_price'] > 0]  # Remove invalid prices

        # Sort by product and date
        df = df.sort_values(['product_id', 'order_date'])

        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
        print(f"Unique products: {df['product_id'].nunique()}")

        return df

    def create_time_features(self, df):
        """
        Create time-based features.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: DataFrame with time features
        """
        df = df.copy()
        print("Creating time-based features...")

        # Basic time components
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        df['day_of_week'] = df['order_date'].dt.dayofweek
        df['quarter'] = df['order_date'].dt.quarter
        df['day_of_year'] = df['order_date'].dt.dayofyear

        # Weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Month-end and quarter-end indicators
        df['is_month_end'] = df['order_date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['order_date'].dt.is_quarter_end.astype(int)

        # Cyclic encoding for month and day of week
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def create_lag_features(self, df, lags=[1, 3, 7, 14, 30]):
        """
        Create lag features for quantity.

        Args:
            df (pd.DataFrame): Input dataframe
            lags (list): List of lag periods in days

        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        df = df.copy()
        print(f"Creating lag features for periods: {lags}")

        for lag in lags:
            df[f'quantity_lag_{lag}'] = df.groupby('product_id')['quantity'].shift(lag)
            df[f'unit_price_lag_{lag}'] = df.groupby('product_id')['unit_price'].shift(lag)

        # Fill NaN values with 0 (no previous data)
        lag_columns = [col for col in df.columns if 'lag' in col]
        df[lag_columns] = df[lag_columns].fillna(0)

        return df

    def create_rolling_features(self, df, windows=[3, 7, 14, 30]):
        """
        Create rolling window features.

        Args:
            df (pd.DataFrame): Input dataframe
            windows (list): List of rolling window sizes

        Returns:
            pd.DataFrame: DataFrame with rolling features
        """
        df = df.copy()
        print(f"Creating rolling features for windows: {windows}")

        for window in windows:
            # Rolling statistics for quantity
            df[f'quantity_rolling_mean_{window}'] = (
                df.groupby('product_id')['quantity']
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

            df[f'quantity_rolling_std_{window}'] = (
                df.groupby('product_id')['quantity']
                .rolling(window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )

            df[f'quantity_rolling_min_{window}'] = (
                df.groupby('product_id')['quantity']
                .rolling(window, min_periods=1)
                .min()
                .reset_index(0, drop=True)
            )

            df[f'quantity_rolling_max_{window}'] = (
                df.groupby('product_id')['quantity']
                .rolling(window, min_periods=1)
                .max()
                .reset_index(0, drop=True)
            )

            # Rolling statistics for price
            df[f'price_rolling_mean_{window}'] = (
                df.groupby('product_id')['unit_price']
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

        # Fill NaN values
        rolling_columns = [col for col in df.columns if 'rolling' in col]
        df[rolling_columns] = df[rolling_columns].fillna(0)

        return df

    def create_product_features(self, df):
        """
        Create product-specific statistical features.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: DataFrame with product features
        """
        df = df.copy()
        print("Creating product-specific features...")

        # Calculate product-level statistics
        product_stats = df.groupby('product_id').agg({
            'quantity': ['count', 'mean', 'std', 'min', 'max', 'sum'],
            'unit_price': ['mean', 'std', 'min', 'max'],
            'order_date': ['min', 'max']
        }).fillna(0)

        # Flatten column names
        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns]
        product_stats = product_stats.reset_index()

        # Calculate additional metrics
        product_stats['product_lifespan_days'] = (
            product_stats['order_date_max'] - product_stats['order_date_min']
        ).dt.days

        product_stats['avg_orders_per_day'] = (
            product_stats['quantity_count'] / product_stats['product_lifespan_days'].clip(lower=1)
        )

        product_stats['price_volatility'] = (
            product_stats['unit_price_std'] / product_stats['unit_price_mean'].clip(lower=0.01)
        )

        # Merge back to main dataframe
        df = df.merge(product_stats, on='product_id', how='left')

        return df

    def create_seasonal_features(self, df):
        """
        Create seasonal and trend features.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: DataFrame with seasonal features
        """
        df = df.copy()
        print("Creating seasonal and trend features...")

        # Seasonal decomposition components
        df['quantity_trend'] = df.groupby('product_id')['quantity'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )

        # Month-over-month growth
        df['quantity_mom_growth'] = df.groupby('product_id')['quantity'].pct_change(30)

        # Year-over-year growth (if data spans multiple years)
        df['quantity_yoy_growth'] = df.groupby('product_id')['quantity'].pct_change(365)

        # Seasonal indicators
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)  # Nov-Dec
        df['is_back_to_school'] = df['month'].isin([8, 9]).astype(int)    # Aug-Sep

        # Fill NaN values
        seasonal_columns = ['quantity_mom_growth', 'quantity_yoy_growth']
        df[seasonal_columns] = df[seasonal_columns].fillna(0)

        # Clip extreme values
        df[seasonal_columns] = df[seasonal_columns].clip(-10, 10)

        return df

    def encode_categorical_features(self, df):
        """
        Encode categorical features.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        df = df.copy()
        print("Encoding categorical features...")

        # Label encode product_id
        if 'product_id' not in self.label_encoders:
            self.label_encoders['product_id'] = LabelEncoder()

        df['product_id_encoded'] = self.label_encoders['product_id'].fit_transform(df['product_id'])

        return df

    def create_target_variable(self, df):
        """
        Create the target variable for forecasting.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: DataFrame with target variable
        """
        df = df.copy()
        print(f"Creating target variable (forecasting {self.target_days_ahead} days ahead)...")

        # Create target: future demand
        df[f'quantity_future_{self.target_days_ahead}d'] = (
            df.groupby('product_id')['quantity'].shift(-self.target_days_ahead)
        )

        # Remove rows where target is NaN (last target_days_ahead days)
        initial_rows = len(df)
        df = df.dropna(subset=[f'quantity_future_{self.target_days_ahead}d'])
        final_rows = len(df)

        print(f"Removed {initial_rows - final_rows} rows with missing target values")

        return df

    def select_features(self, df):
        """
        Select final feature set for modeling.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            tuple: (X, y) features and target
        """
        # Define feature columns (exclude non-feature columns)
        exclude_columns = [
            'order_id', 'order_date', 'product_id',
            f'quantity_future_{self.target_days_ahead}d',  # This is the target
            'order_date_min', 'order_date_max'  # Date columns
        ]

        # Also exclude the original target column
        exclude_columns.append('quantity')

        self.feature_columns = [col for col in df.columns if col not in exclude_columns]

        X = df[self.feature_columns]
        y = df[f'quantity_future_{self.target_days_ahead}d']

        print(f"Selected {len(self.feature_columns)} features:")
        print(self.feature_columns)

        return X, y

    def fit_scaler(self, X):
        """
        Fit the feature scaler.

        Args:
            X (pd.DataFrame): Feature matrix
        """
        print("Fitting feature scaler...")
        self.scaler.fit(X)

    def transform_features(self, X):
        """
        Transform features using fitted scaler.

        Args:
            X (pd.DataFrame): Feature matrix

        Returns:
            pd.DataFrame: Scaled features
        """
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def save_pipeline(self, output_path):
        """
        Save the feature pipeline.

        Args:
            output_path (Path): Path to save pipeline
        """
        pipeline_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_days_ahead': self.target_days_ahead
        }

        joblib.dump(pipeline_data, output_path)
        print(f"Feature pipeline saved to: {output_path}")

    def load_pipeline(self, input_path):
        """
        Load a saved feature pipeline.

        Args:
            input_path (Path): Path to load pipeline from
        """
        pipeline_data = joblib.load(input_path)

        self.scaler = pipeline_data['scaler']
        self.label_encoders = pipeline_data['label_encoders']
        self.feature_columns = pipeline_data['feature_columns']
        self.target_days_ahead = pipeline_data['target_days_ahead']

        print(f"Feature pipeline loaded from: {input_path}")

def main():
    """Main feature engineering pipeline."""
    print("=== Feature Engineering Pipeline ===\\n")

    # Define paths
    data_path = Path('../data/raw/amazon_orders.csv')
    output_dir = Path('../data/processed')
    pipeline_path = Path('../models/feature_pipeline.pkl')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if data exists
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    try:
        # Initialize pipeline
        pipeline = FeaturePipeline(target_days_ahead=7)

        # Load data
        df = pipeline.load_data(data_path)

        # Create all features
        df = pipeline.create_time_features(df)
        df = pipeline.create_lag_features(df)
        df = pipeline.create_rolling_features(df)
        df = pipeline.create_product_features(df)
        df = pipeline.create_seasonal_features(df)
        df = pipeline.encode_categorical_features(df)

        # Create target
        df = pipeline.create_target_variable(df)

        # Select features
        X, y = pipeline.select_features(df)

        # Fit scaler
        pipeline.fit_scaler(X)

        # Transform features
        X_scaled = pipeline.transform_features(X)

        # Combine features and target for saving
        final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

        # Save processed data
        processed_data_path = output_dir / 'processed_features.csv'
        final_df.to_csv(processed_data_path, index=False)
        print(f"\\nProcessed data saved to: {processed_data_path}")
        print(f"Dataset shape: {final_df.shape}")

        # Save pipeline
        pipeline.save_pipeline(pipeline_path)

        # Print summary
        print("\\n=== FEATURE ENGINEERING SUMMARY ===")
        print(f"Original features: {len(df.columns) - len([col for col in df.columns if 'future' in col])}")
        print(f"Engineered features: {len(pipeline.feature_columns)}")
        print(f"Target variable: quantity_future_{pipeline.target_days_ahead}d")
        print(f"Total samples: {len(final_df)}")

        # Feature categories summary
        time_features = [col for col in pipeline.feature_columns if any(x in col for x in ['month', 'day', 'year', 'weekend'])]
        lag_features = [col for col in pipeline.feature_columns if 'lag' in col]
        rolling_features = [col for col in pipeline.feature_columns if 'rolling' in col]
        product_features = [col for col in pipeline.feature_columns if any(x in col for x in ['count', 'mean', 'std', 'lifespan'])]

        print(f"\\nFeature breakdown:")
        print(f"• Time features: {len(time_features)}")
        print(f"• Lag features: {len(lag_features)}")
        print(f"• Rolling features: {len(rolling_features)}")
        print(f"• Product features: {len(product_features)}")

        print("\\n✅ Feature engineering completed successfully!")

    except Exception as e:
        print(f"Error during feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    main()