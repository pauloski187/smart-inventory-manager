from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..services.inventory_service import InventoryService
from ..schemas.analytics import AnalyticsResponse
from ..database import get_db

router = APIRouter()

@router.get("/inventory-alerts")
def get_inventory_alerts(db: Session = Depends(get_db)):
    service = InventoryService(db)
    return {"alerts": service.get_inventory_alerts()}

@router.get("/sales-trends")
def get_sales_trends(days: int = 30, db: Session = Depends(get_db)):
    service = InventoryService(db)
    return {"trends": service.get_sales_trends(days)}

@router.get("/abc-analysis")
def get_abc_analysis(db: Session = Depends(get_db)):
    service = InventoryService(db)
    return {"analysis": service.perform_abc_analysis()}

@router.get("/demand-forecast/{product_id}")
def get_demand_forecast(product_id: str, days_ahead: int = 30, db: Session = Depends(get_db)):
    service = InventoryService(db)
    return service.simple_demand_forecast(product_id, days_ahead)

@router.get("/dashboard", response_model=AnalyticsResponse)
def get_dashboard_analytics(db: Session = Depends(get_db)):
    service = InventoryService(db)
    return AnalyticsResponse(
        sales_trends=service.get_sales_trends(30),
        inventory_alerts=service.get_inventory_alerts(),
        abc_analysis=service.perform_abc_analysis(),
        demand_forecasts=[]  # Could add forecasts for top products
    )