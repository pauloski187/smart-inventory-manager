from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..services.inventory_service import InventoryService
from ..schemas.order import OrderCreate, OrderUpdate, Order
from ..database import get_db

router = APIRouter()

@router.get("/", response_model=list[Order])
def read_orders(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    orders = db.query(Order).offset(skip).limit(limit).all()
    return orders

@router.post("/", response_model=Order)
def create_order(order: OrderCreate, db: Session = Depends(get_db)):
    service = InventoryService(db)
    return service.record_order(order)

@router.get("/{order_id}", response_model=Order)
def read_order(order_id: str, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()
    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return order