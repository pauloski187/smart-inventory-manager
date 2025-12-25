# Schemas package
from .product import ProductBase, ProductCreate, ProductUpdate, Product
from .order import OrderBase, OrderCreate, OrderUpdate, Order
from .customer import CustomerBase, CustomerCreate, CustomerUpdate, Customer
from .analytics import (
    InventoryAlert,
    SalesTrend,
    ABCAnalysis,
    DemandForecast,
    AnalyticsResponse
)

__all__ = [
    "ProductBase", "ProductCreate", "ProductUpdate", "Product",
    "OrderBase", "OrderCreate", "OrderUpdate", "Order",
    "CustomerBase", "CustomerCreate", "CustomerUpdate", "Customer",
    "InventoryAlert", "SalesTrend", "ABCAnalysis", "DemandForecast", "AnalyticsResponse"
]