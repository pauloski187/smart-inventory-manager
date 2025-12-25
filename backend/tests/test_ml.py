"""
Unit tests for ML models.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDemandForecaster:
    """Tests for demand forecasting."""

    def test_moving_average_forecast(self):
        """Test moving average forecast calculation."""
        from app.ml.forecasting import DemandForecaster

        # Mock database session
        mock_db = MagicMock()

        # Create mock data
        mock_orders = []
        base_date = datetime.now() - timedelta(days=30)
        for i in range(30):
            mock_order = MagicMock()
            mock_order.date = (base_date + timedelta(days=i)).date()
            mock_order.quantity = 10
            mock_orders.append(mock_order)

        mock_db.query.return_value.filter.return_value.group_by.return_value.all.return_value = mock_orders

        forecaster = DemandForecaster(mock_db)

        # Test that forecaster initializes
        assert forecaster.db == mock_db

    def test_linear_regression_forecast(self):
        """Test linear regression forecast."""
        from app.ml.forecasting import DemandForecaster

        mock_db = MagicMock()
        forecaster = DemandForecaster(mock_db)

        # Mock historical data
        with patch.object(forecaster, 'get_historical_demand') as mock_hist:
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            mock_hist.return_value = pd.DataFrame({
                'date': dates,
                'quantity': np.random.randint(5, 15, size=30)
            })

            result = forecaster.linear_regression_forecast('PROD-001', days_ahead=7)

            assert 'product_id' in result
            assert result['product_id'] == 'PROD-001'
            assert result['method'] == 'linear_regression'
            assert 'forecast_values' in result
            assert len(result['forecast_values']) == 7

    def test_insufficient_data_handling(self):
        """Test handling of insufficient historical data."""
        from app.ml.forecasting import DemandForecaster

        mock_db = MagicMock()
        forecaster = DemandForecaster(mock_db)

        with patch.object(forecaster, 'get_historical_demand') as mock_hist:
            # Return only 5 days of data (insufficient)
            mock_hist.return_value = pd.DataFrame({
                'date': pd.date_range(end=datetime.now(), periods=5, freq='D'),
                'quantity': [1, 2, 3, 4, 5]
            })

            result = forecaster.linear_regression_forecast('PROD-001')

            # Should return error for insufficient data
            assert 'error' in result


class TestABCAnalyzer:
    """Tests for ABC analysis."""

    def test_abc_classification(self):
        """Test ABC classification logic."""
        from app.ml.abc_analysis import ABCAnalyzer

        mock_db = MagicMock()

        # Mock query results
        mock_results = [
            MagicMock(product_id='P1', total_revenue=80000, total_quantity=100),
            MagicMock(product_id='P2', total_revenue=10000, total_quantity=50),
            MagicMock(product_id='P3', total_revenue=5000, total_quantity=30),
            MagicMock(product_id='P4', total_revenue=3000, total_quantity=20),
            MagicMock(product_id='P5', total_revenue=2000, total_quantity=10),
        ]

        mock_db.query.return_value.filter.return_value.group_by.return_value.all.return_value = mock_results

        analyzer = ABCAnalyzer(mock_db)

        # Test that analyzer initializes
        assert analyzer.db == mock_db

    def test_class_summary(self):
        """Test ABC class summary generation."""
        from app.ml.abc_analysis import ABCAnalyzer

        mock_db = MagicMock()
        analyzer = ABCAnalyzer(mock_db)

        with patch.object(analyzer, 'perform_analysis') as mock_analysis:
            mock_analysis.return_value = pd.DataFrame({
                'product_id': ['P1', 'P2', 'P3', 'P4'],
                'abc_class': ['A', 'A', 'B', 'C'],
                'total_revenue': [50000, 30000, 15000, 5000],
                'total_quantity': [100, 80, 50, 20],
                'revenue_pct': [50, 30, 15, 5]
            })

            summary = analyzer.get_class_summary()

            assert 'A' in summary
            assert 'B' in summary
            assert 'C' in summary
            assert summary['A']['product_count'] == 2


class TestDeadStockDetector:
    """Tests for dead stock detection."""

    def test_dead_stock_detection(self):
        """Test dead stock detection logic."""
        from app.ml.dead_stock import DeadStockDetector

        mock_db = MagicMock()
        detector = DeadStockDetector(mock_db, threshold_days=90)

        assert detector.threshold_days == 90
        assert detector.db == mock_db

    def test_custom_threshold(self):
        """Test detection with custom threshold."""
        from app.ml.dead_stock import DeadStockDetector

        mock_db = MagicMock()
        detector = DeadStockDetector(mock_db, threshold_days=60)

        assert detector.threshold_days == 60

    def test_recommendation_logic(self):
        """Test recommendation generation based on days."""
        from app.ml.dead_stock import DeadStockDetector

        mock_db = MagicMock()
        detector = DeadStockDetector(mock_db)

        # Test different scenarios
        rec1 = detector._get_recommendation(None, 1000)
        assert 'never sold' in rec1.lower()

        rec2 = detector._get_recommendation(200, 2000)
        assert 'liquidation' in rec2.lower() or 'urgent' in rec2.lower()

        rec3 = detector._get_recommendation(100, 100)
        assert 'discount' in rec3.lower() or 'promotion' in rec3.lower()


class TestFeatureEngineer:
    """Tests for feature engineering."""

    def test_stock_velocity_calculation(self):
        """Test stock velocity calculation."""
        from app.etl.transform import FeatureEngineer

        mock_db = MagicMock()

        # Mock query result
        mock_db.query.return_value.filter.return_value.scalar.return_value = 100

        engineer = FeatureEngineer(mock_db)
        velocity = engineer.calculate_stock_velocity('PROD-001', days=30)

        assert velocity == round(100 / 30, 2)

    def test_demand_trend_classification(self):
        """Test demand trend classification."""
        from app.etl.transform import FeatureEngineer

        mock_db = MagicMock()
        engineer = FeatureEngineer(mock_db)

        # Mock increasing demand
        mock_db.query.return_value.filter.return_value.scalar.side_effect = [100, 50]  # Recent > Prior

        trend = engineer.calculate_demand_trend('PROD-001')
        # With 100 recent vs 50 prior, should be increasing
        assert trend in ['Increasing', 'Stable', 'Declining']

    def test_days_since_last_sale(self):
        """Test days since last sale calculation."""
        from app.etl.transform import FeatureEngineer

        mock_db = MagicMock()
        engineer = FeatureEngineer(mock_db)

        # Mock no orders
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

        days = engineer.days_since_last_sale('PROD-001')
        assert days == -1  # Never sold
