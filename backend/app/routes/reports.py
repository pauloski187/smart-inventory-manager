from fastapi import APIRouter

router = APIRouter()

@router.get("/inventory-summary")
def get_inventory_summary():
    return {"summary": "Inventory data"}

@router.get("/sales-report")
def get_sales_report():
    return {"report": "Sales data"}

# Add more analytics endpoints