from pydantic import BaseModel
from typing import Optional, Dict, Any

class SalesTrend(BaseModel):
    period: str
    total_sales: float
    total_orders: int
    avg_order_value: float

class InventoryAlert(BaseModel):
    product_id: str
    product_name: str
    current_stock: int
    reorder_threshold: int
    alert_type: str  # 'low_stock', 'dead_stock'

class ABCAnalysis(BaseModel):
    product_id: str
    product_name: str
    category: str
    total_revenue: float
    total_quantity: int
    abc_class: str  # 'A', 'B', 'C'

class DemandForecast(BaseModel):
    product_id: str
    forecast_period: str
    predicted_demand: float
    confidence_interval: Optional[Dict[str, float]] = None

class AnalyticsResponse(BaseModel):
    sales_trends: list[SalesTrend] = []
    inventory_alerts: list[InventoryAlert] = []
    abc_analysis: list[ABCAnalysis] = []
    demand_forecasts: list[DemandForecast] = []