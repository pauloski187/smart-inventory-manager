"""
Unit tests for ETL pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.etl.ingestion import (
    ingest_customers,
    ingest_products,
    ingest_inventory,
    ingest_orders,
    ingest_order_items,
    ingest_sellers
)
from app.etl.validation import validate_data, ValidationResult


# Test data directory
DATA_DIR = Path(__file__).parent.parent.parent / 'ml' / 'data' / 'processed'


class TestIngestion:
    """Tests for data ingestion functions."""

    def test_ingest_customers(self):
        """Test customer ingestion."""
        if not (DATA_DIR / 'customers.csv').exists():
            pytest.skip("Test data not available")

        df = ingest_customers(str(DATA_DIR / 'customers.csv'))

        assert len(df) > 0
        assert 'customer_id' in df.columns
        assert 'customer_name' in df.columns
        assert df['customer_name'].isna().sum() == 0  # No missing names

    def test_ingest_products(self):
        """Test product ingestion."""
        if not (DATA_DIR / 'products.csv').exists():
            pytest.skip("Test data not available")

        df = ingest_products(str(DATA_DIR / 'products.csv'))

        assert len(df) > 0
        assert 'product_id' in df.columns
        assert 'product_name' in df.columns
        assert 'cost_price' in df.columns
        assert (df['cost_price'] >= 0).all()  # No negative prices

    def test_ingest_inventory(self):
        """Test inventory ingestion."""
        if not (DATA_DIR / 'inventory.csv').exists():
            pytest.skip("Test data not available")

        df = ingest_inventory(str(DATA_DIR / 'inventory.csv'))

        assert len(df) > 0
        assert 'product_id' in df.columns
        assert 'current_stock' in df.columns
        assert (df['current_stock'] >= 0).all()  # No negative stock

    def test_ingest_orders(self):
        """Test order ingestion."""
        if not (DATA_DIR / 'orders.csv').exists():
            pytest.skip("Test data not available")

        df = ingest_orders(str(DATA_DIR / 'orders.csv'))

        assert len(df) > 0
        assert 'order_id' in df.columns
        assert 'order_date' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['order_date'])

    def test_ingest_order_items(self):
        """Test order items ingestion."""
        if not (DATA_DIR / 'order_items.csv').exists():
            pytest.skip("Test data not available")

        df = ingest_order_items(str(DATA_DIR / 'order_items.csv'))

        assert len(df) > 0
        assert 'order_id' in df.columns
        assert 'product_id' in df.columns
        assert 'quantity' in df.columns
        assert (df['quantity'] >= 1).all()  # Quantity must be positive

    def test_ingest_sellers(self):
        """Test seller ingestion."""
        if not (DATA_DIR / 'sellers.csv').exists():
            pytest.skip("Test data not available")

        df = ingest_sellers(str(DATA_DIR / 'sellers.csv'))

        assert len(df) > 0
        assert 'seller_id' in df.columns


class TestValidation:
    """Tests for data validation functions."""

    def test_validate_customers(self):
        """Test customer validation."""
        df = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'customer_name': ['John', 'Jane', None]
        })

        result = validate_data(df, 'customers')

        assert isinstance(result, ValidationResult)
        assert result.entity == 'customers'
        assert result.total_records == 3
        assert len(result.warnings) > 0  # Should warn about missing name

    def test_validate_products(self):
        """Test product validation."""
        df = pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003'],
            'product_name': ['Widget', 'Gadget', 'Thing'],
            'cost_price': [10.0, 20.0, -5.0]  # One negative price
        })

        result = validate_data(df, 'products')

        assert result.entity == 'products'
        assert result.is_valid == False  # Should fail due to negative price
        assert result.invalid_records > 0

    def test_validate_inventory(self):
        """Test inventory validation."""
        df = pd.DataFrame({
            'product_id': ['P001', 'P002'],
            'current_stock': [100, -10],  # One negative stock
            'reorder_level': [10, 5]
        })

        result = validate_data(df, 'inventory')

        assert result.is_valid == False  # Should fail due to negative stock

    def test_validate_orders(self):
        """Test order validation."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002'],
            'order_date': pd.to_datetime(['2024-01-01', '2024-01-02'])
        })

        result = validate_data(df, 'orders')

        assert result.is_valid == True
        assert result.total_records == 2

    def test_validate_order_items(self):
        """Test order items validation."""
        df = pd.DataFrame({
            'order_id': ['O001', 'O002', 'O003'],
            'product_id': ['P001', 'P002', 'P003'],
            'quantity': [1, 0, 5],  # One zero quantity
            'unit_price': [10.0, 20.0, 30.0]
        })

        result = validate_data(df, 'order_items')

        assert result.is_valid == False  # Should fail due to zero quantity

    def test_validate_unknown_entity(self):
        """Test validation with unknown entity type."""
        df = pd.DataFrame({'col': [1, 2, 3]})

        with pytest.raises(ValueError):
            validate_data(df, 'unknown_entity')


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_error_rate(self):
        """Test error rate calculation."""
        result = ValidationResult(
            is_valid=False,
            entity='test',
            total_records=100,
            valid_records=80,
            invalid_records=20
        )

        assert result.error_rate == 0.2

    def test_error_rate_zero_records(self):
        """Test error rate with zero records."""
        result = ValidationResult(
            is_valid=True,
            entity='test',
            total_records=0,
            valid_records=0,
            invalid_records=0
        )

        assert result.error_rate == 0.0
