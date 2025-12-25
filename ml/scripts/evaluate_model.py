#!/usr/bin/env python3
"""
Model Evaluation and Validation Script

This script performs comprehensive evaluation of the trained demand forecasting model,
including performance metrics, cross-validation, and business impact analysis.

Usage:
    python scripts/evaluate_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_model_and_data(model_path, test_data_path):
    """
    Load the trained model and test data.

    Args:
        model_path (Path): Path to the saved model
        test_data_path (Path): Path to test data

    Returns:
        tuple: (model, X_test, y_test)
    """
    print("Loading model and test data...")

    # Load model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Load test data
    test_df = pd.read_csv(test_data_path)
    print(f"Test data loaded: {test_df.shape}")

    # Separate features and target
    target_col = [col for col in test_df.columns if 'future' in col.lower() or col == 'quantity_future_7d']
    if not target_col:
        target_col = test_df.columns[-1]  # Assume last column is target

    feature_cols = [col for col in test_df.columns if col != target_col[0]]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col[0]]

    return model, X_test, y_test

def calculate_comprehensive_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true (array): True values
        y_pred (array): Predicted values

    Returns:
        dict: Dictionary of metrics
    """
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Percentage-based metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

    # Scaled metrics
    naive_errors = np.abs(y_true[1:] - y_true[:-1])
    mean_naive_error = np.mean(naive_errors) if len(naive_errors) > 0 else 1
    mase = mae / mean_naive_error if mean_naive_error > 0 else np.inf

    # Additional metrics
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    residual_std = np.std(residuals)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'SMAPE': smape,
        'MASE': mase,
        'Mean_Residual': mean_residual,
        'Residual_Std': residual_std
    }

    return metrics

def perform_cross_validation(model, X, y, n_splits=5):
    """
    Perform time series cross-validation.

    Args:
        model: Trained model
        X (pd.DataFrame): Features
        y (pd.Series): Target
        n_splits (int): Number of CV splits

    Returns:
        dict: Cross-validation results
    """
    print(f"Performing {n_splits}-fold time series cross-validation...")

    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Define metrics to evaluate
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

    cv_results = {}
    for metric in scoring:
        scores = cross_val_score(model, X, y, cv=tscv, scoring=metric)

        if 'neg_' in metric:
            scores = -scores  # Convert to positive
            metric_name = metric.replace('neg_', '').replace('_', ' ').title()
        else:
            metric_name = metric.upper()

        cv_results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
            'scores': scores
        }

        print(f"{metric_name}: {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results

def create_evaluation_plots(y_true, y_pred, output_dir):
    """
    Create comprehensive evaluation plots.

    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        output_dir (Path): Directory to save plots
    """
    residuals = y_true - y_pred

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Evaluation Plots', fontsize=16, fontweight='bold')

    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residual Plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residual Distribution
    axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('Residual Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Residual Distribution')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Time Series Comparison (first 100 points)
    sample_size = min(100, len(y_true))
    axes[1, 0].plot(range(sample_size), y_true[:sample_size],
                    label='Actual', linewidth=2, color='blue')
    axes[1, 0].plot(range(sample_size), y_pred[:sample_size],
                    label='Predicted', linewidth=2, color='red', alpha=0.8)
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Demand Quantity')
    axes[1, 0].set_title('Time Series Comparison (Sample)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Error by Magnitude
    error_pct = np.abs(residuals / y_true) * 100
    axes[1, 1].scatter(y_true, error_pct, alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Absolute Percentage Error (%)')
    axes[1, 1].set_title('Error Distribution by Magnitude')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot of Residuals')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'model_evaluation_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Evaluation plots saved to: {plot_path}")

def analyze_feature_importance(model, feature_names, output_dir):
    """
    Analyze and plot feature importance.

    Args:
        model: Trained model
        feature_names (list): List of feature names
        output_dir (Path): Directory to save plots
    """
    if not hasattr(model.named_steps['regressor'], 'feature_importances_'):
        print("Model does not support feature importance analysis.")
        return

    # Get feature importance
    importance = model.named_steps['regressor'].feature_importances_

    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Plot top 10 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(10)
    bars = plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', ha='left', va='center')

    plt.tight_layout()
    importance_plot_path = output_dir / 'feature_importance.png'
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot saved to: {importance_plot_path}")

    # Save feature importance data
    importance_data_path = output_dir / 'feature_importance.csv'
    feature_importance_df.to_csv(importance_data_path, index=False)
    print(f"Feature importance data saved to: {importance_data_path}")

def assess_business_impact(metrics, y_test, output_dir):
    """
    Assess the business impact of the model.

    Args:
        metrics (dict): Model performance metrics
        y_test (pd.Series): Test target values
        output_dir (Path): Directory to save results
    """
    print("\\n=== BUSINESS IMPACT ASSESSMENT ===")

    # Assumptions
    avg_price_per_unit = 50  # Assume average price
    carrying_cost_rate = 0.25  # 25% annual carrying cost
    stockout_cost_per_incident = 20  # Cost per stock-out incident
    baseline_stockout_rate = 0.05  # 5% baseline stock-out rate
    forecast_improvement_factor = 0.3  # 30% improvement from better forecasting

    mae = metrics['MAE']
    total_predictions = len(y_test)

    # Calculate potential benefits
    monthly_inventory_savings = mae * avg_price_per_unit * total_predictions * carrying_cost_rate / 12
    monthly_stockout_savings = (baseline_stockout_rate * forecast_improvement_factor *
                               stockout_cost_per_incident * total_predictions)
    total_monthly_savings = monthly_inventory_savings + monthly_stockout_savings

    # ROI calculation
    development_cost = 5000  # Estimated development cost
    annual_benefits = total_monthly_savings * 12
    roi = (annual_benefits - development_cost) / development_cost * 100
    payback_months = development_cost / total_monthly_savings if total_monthly_savings > 0 else float('inf')

    # Create business impact report
    impact_data = {
        'Metric': [
            'Monthly Inventory Carrying Cost Savings',
            'Monthly Stock-out Cost Reduction',
            'Total Monthly Savings',
            'Annual Benefits',
            'Development Cost',
            'ROI (%)',
            'Payback Period (months)'
        ],
        'Value': [
            f"${monthly_inventory_savings:,.2f}",
            f"${monthly_stockout_savings:,.2f}",
            f"${total_monthly_savings:,.2f}",
            f"${annual_benefits:,.2f}",
            f"${development_cost:,.2f}",
            f"{roi:.1f}%",
            f"{payback_months:.1f}"
        ]
    }

    impact_df = pd.DataFrame(impact_data)
    impact_path = output_dir / 'business_impact_analysis.csv'
    impact_df.to_csv(impact_path, index=False)

    print("Business Impact Summary:")
    for _, row in impact_df.iterrows():
        print(f"  {row['Metric']}: {row['Value']}")

    print(f"\\nBusiness impact analysis saved to: {impact_path}")

def save_evaluation_results(metrics, cv_results, output_dir):
    """
    Save all evaluation results to files.

    Args:
        metrics (dict): Performance metrics
        cv_results (dict): Cross-validation results
        output_dir (Path): Output directory
    """
    # Save comprehensive metrics
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_path = output_dir / 'comprehensive_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\\nComprehensive metrics saved to: {metrics_path}")

    # Save CV results
    cv_summary = []
    for metric_name, results in cv_results.items():
        cv_summary.append({
            'Metric': metric_name,
            'Mean': results['mean'],
            'Std': results['std'],
            'Min': results['min'],
            'Max': results['max']
        })

    cv_df = pd.DataFrame(cv_summary)
    cv_path = output_dir / 'cross_validation_results.csv'
    cv_df.to_csv(cv_path, index=False)
    print(f"Cross-validation results saved to: {cv_path}")

def main():
    """Main evaluation pipeline."""
    print("=== Model Evaluation and Validation Pipeline ===\\n")

    # Define paths
    model_path = Path('../models/demand_forecast_model.pkl')
    test_data_path = Path('../models/test_features.csv')
    train_data_path = Path('../models/train_features.csv')
    output_dir = Path('../models')

    # Check if required files exist
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    if not test_data_path.exists():
        print(f"Error: Test data file not found at {test_data_path}")
        return

    try:
        # Load model and data
        model, X_test, y_test = load_model_and_data(model_path, test_data_path)

        # Make predictions
        print("Generating predictions...")
        y_pred = model.predict(X_test)

        # Calculate comprehensive metrics
        print("\\nCalculating evaluation metrics...")
        metrics = calculate_comprehensive_metrics(y_test, y_pred)

        print("\\n=== MODEL PERFORMANCE METRICS ===")
        for metric, value in metrics.items():
            if metric in ['MAE', 'RMSE', 'Mean_Residual', 'Residual_Std']:
                print(f"{metric}: {value:.4f}")
            elif metric in ['MAPE', 'SMAPE']:
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value:.4f}")

        # Perform cross-validation
        if train_data_path.exists():
            train_df = pd.read_csv(train_data_path)
            target_col = [col for col in train_df.columns if 'future' in col.lower() or col == 'quantity_future_7d']
            if target_col:
                X_train = train_df.drop(target_col[0], axis=1)
                y_train = train_df[target_col[0]]
                cv_results = perform_cross_validation(model, X_train, y_train)
            else:
                print("Could not identify target column for cross-validation")
                cv_results = None
        else:
            print("Training data not found - skipping cross-validation")
            cv_results = None

        # Create evaluation plots
        create_evaluation_plots(y_test, y_pred, output_dir)

        # Analyze feature importance
        feature_names = X_test.columns.tolist()
        analyze_feature_importance(model, feature_names, output_dir)

        # Assess business impact
        assess_business_impact(metrics, y_test, output_dir)

        # Save all results
        save_evaluation_results(metrics, cv_results or {}, output_dir)

        print("\\n✅ Model evaluation completed successfully!")
        print("\\nKey Findings:")
        print(f"• Model MAE: {metrics['MAE']:.4f}")
        print(f"• Model MAPE: {metrics['MAPE']:.2f}%")
        print(f"• Model R²: {metrics['R²']:.4f}")

        # Performance interpretation
        mape = metrics['MAPE']
        if mape < 10:
            print("• Performance Rating: Excellent (MAPE < 10%)")
        elif mape < 20:
            print("• Performance Rating: Good (MAPE 10-20%)")
        elif mape < 30:
            print("• Performance Rating: Acceptable (MAPE 20-30%)")
        else:
            print("• Performance Rating: Needs Improvement (MAPE > 30%)")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()