"""
Data Validation Module

Validates data integrity for the Smart Inventory Manager.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from data validation."""
    is_valid: bool
    entity: str
    total_records: int
    valid_records: int
    invalid_records: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.invalid_records / self.total_records


def validate_data(df: pd.DataFrame, entity: str) -> ValidationResult:
    """
    Validate data based on entity type.

    Args:
        df: DataFrame to validate
        entity: Type of entity ('customers', 'products', 'inventory', 'orders', 'order_items', 'sellers')

    Returns:
        ValidationResult with validation details
    """
    validators = {
        'customers': _validate_customers,
        'products': _validate_products,
        'inventory': _validate_inventory,
        'orders': _validate_orders,
        'order_items': _validate_order_items,
        'sellers': _validate_sellers
    }

    if entity not in validators:
        raise ValueError(f"Unknown entity type: {entity}")

    return validators[entity](df)


def _validate_customers(df: pd.DataFrame) -> ValidationResult:
    """Validate customer data."""
    errors = []
    warnings = []
    invalid_count = 0

    # Check for required columns
    required_cols = ['customer_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationResult(
            is_valid=False,
            entity='customers',
            total_records=len(df),
            valid_records=0,
            invalid_records=len(df),
            errors=errors
        )

    # Check for duplicate customer IDs
    duplicates = df['customer_id'].duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate customer IDs")

    # Check for missing names
    if 'customer_name' in df.columns:
        missing_names = df['customer_name'].isna().sum()
        if missing_names > 0:
            warnings.append(f"Found {missing_names} customers with missing names")
            invalid_count += missing_names

    valid_count = len(df) - invalid_count

    return ValidationResult(
        is_valid=len(errors) == 0,
        entity='customers',
        total_records=len(df),
        valid_records=valid_count,
        invalid_records=invalid_count,
        errors=errors,
        warnings=warnings
    )


def _validate_products(df: pd.DataFrame) -> ValidationResult:
    """Validate product data."""
    errors = []
    warnings = []
    invalid_count = 0

    # Check for required columns
    required_cols = ['product_id', 'product_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationResult(
            is_valid=False,
            entity='products',
            total_records=len(df),
            valid_records=0,
            invalid_records=len(df),
            errors=errors
        )

    # Check for duplicate product IDs
    duplicates = df['product_id'].duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate product IDs")

    # Check for missing product names
    missing_names = df['product_name'].isna().sum()
    if missing_names > 0:
        errors.append(f"Found {missing_names} products with missing names (NOT NULL constraint)")
        invalid_count += missing_names

    # Validate cost_price if present
    if 'cost_price' in df.columns:
        negative_prices = (df['cost_price'] < 0).sum()
        if negative_prices > 0:
            errors.append(f"Found {negative_prices} products with negative cost prices")
            invalid_count += negative_prices

    valid_count = len(df) - invalid_count

    return ValidationResult(
        is_valid=len(errors) == 0,
        entity='products',
        total_records=len(df),
        valid_records=valid_count,
        invalid_records=invalid_count,
        errors=errors,
        warnings=warnings
    )


def _validate_inventory(df: pd.DataFrame) -> ValidationResult:
    """Validate inventory data."""
    errors = []
    warnings = []
    invalid_count = 0

    required_cols = ['product_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationResult(
            is_valid=False,
            entity='inventory',
            total_records=len(df),
            valid_records=0,
            invalid_records=len(df),
            errors=errors
        )

    # Check for negative stock levels
    if 'current_stock' in df.columns:
        negative_stock = (df['current_stock'] < 0).sum()
        if negative_stock > 0:
            errors.append(f"Found {negative_stock} items with negative stock levels")
            invalid_count += negative_stock

    # Check for negative reorder points
    if 'reorder_level' in df.columns:
        negative_reorder = (df['reorder_level'] < 0).sum()
        if negative_reorder > 0:
            errors.append(f"Found {negative_reorder} items with negative reorder points")
            invalid_count += negative_reorder

    valid_count = len(df) - invalid_count

    return ValidationResult(
        is_valid=len(errors) == 0,
        entity='inventory',
        total_records=len(df),
        valid_records=valid_count,
        invalid_records=invalid_count,
        errors=errors,
        warnings=warnings
    )


def _validate_orders(df: pd.DataFrame) -> ValidationResult:
    """Validate order data."""
    errors = []
    warnings = []
    invalid_count = 0

    required_cols = ['order_id', 'order_date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationResult(
            is_valid=False,
            entity='orders',
            total_records=len(df),
            valid_records=0,
            invalid_records=len(df),
            errors=errors
        )

    # Check for duplicate order IDs
    duplicates = df['order_id'].duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate order IDs")

    # Check for invalid dates
    if df['order_date'].dtype == 'datetime64[ns]':
        invalid_dates = df['order_date'].isna().sum()
        if invalid_dates > 0:
            errors.append(f"Found {invalid_dates} orders with invalid dates")
            invalid_count += invalid_dates

        # Check for future dates
        future_dates = (df['order_date'] > datetime.now()).sum()
        if future_dates > 0:
            warnings.append(f"Found {future_dates} orders with future dates")

    valid_count = len(df) - invalid_count

    return ValidationResult(
        is_valid=len(errors) == 0,
        entity='orders',
        total_records=len(df),
        valid_records=valid_count,
        invalid_records=invalid_count,
        errors=errors,
        warnings=warnings
    )


def _validate_order_items(df: pd.DataFrame) -> ValidationResult:
    """Validate order item data."""
    errors = []
    warnings = []
    invalid_count = 0

    required_cols = ['order_id', 'product_id', 'quantity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationResult(
            is_valid=False,
            entity='order_items',
            total_records=len(df),
            valid_records=0,
            invalid_records=len(df),
            errors=errors
        )

    # Check for non-positive quantities
    non_positive_qty = (df['quantity'] <= 0).sum()
    if non_positive_qty > 0:
        errors.append(f"Found {non_positive_qty} items with quantity <= 0")
        invalid_count += non_positive_qty

    # Check for negative prices
    if 'unit_price' in df.columns:
        negative_prices = (df['unit_price'] < 0).sum()
        if negative_prices > 0:
            errors.append(f"Found {negative_prices} items with negative unit prices")
            invalid_count += negative_prices

    # Check for negative total amounts
    if 'total_amount' in df.columns:
        negative_totals = (df['total_amount'] < 0).sum()
        if negative_totals > 0:
            warnings.append(f"Found {negative_totals} items with negative total amounts (may be refunds)")

    valid_count = len(df) - invalid_count

    return ValidationResult(
        is_valid=len(errors) == 0,
        entity='order_items',
        total_records=len(df),
        valid_records=valid_count,
        invalid_records=invalid_count,
        errors=errors,
        warnings=warnings
    )


def _validate_sellers(df: pd.DataFrame) -> ValidationResult:
    """Validate seller data."""
    errors = []
    warnings = []
    invalid_count = 0

    required_cols = ['seller_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationResult(
            is_valid=False,
            entity='sellers',
            total_records=len(df),
            valid_records=0,
            invalid_records=len(df),
            errors=errors
        )

    # Check for duplicate seller IDs
    duplicates = df['seller_id'].duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate seller IDs")

    valid_count = len(df) - invalid_count

    return ValidationResult(
        is_valid=len(errors) == 0,
        entity='sellers',
        total_records=len(df),
        valid_records=valid_count,
        invalid_records=invalid_count,
        errors=errors,
        warnings=warnings
    )
