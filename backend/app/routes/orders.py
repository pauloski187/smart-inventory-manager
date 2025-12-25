from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from ..services.inventory_service import InventoryService
from ..schemas.order import OrderCreate, OrderUpdate, Order as OrderSchema
from ..models.order import Order as OrderModel
from ..database import get_db

router = APIRouter()


@router.get("/", response_model=list[OrderSchema])
def read_orders(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    orders = db.query(OrderModel).offset(skip).limit(limit).all()
    return orders


@router.get("/recent", response_model=list[OrderSchema])
def get_recent_orders(limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    """Get most recent orders."""
    orders = db.query(OrderModel).order_by(desc(OrderModel.order_date)).limit(limit).all()
    return orders


@router.post("/", response_model=OrderSchema)
def create_order(order: OrderCreate, db: Session = Depends(get_db)):
    service = InventoryService(db)
    return service.record_order(order)


@router.get("/{order_id}", response_model=OrderSchema)
def read_order(order_id: str, db: Session = Depends(get_db)):
    order = db.query(OrderModel).filter(OrderModel.id == order_id).first()
    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return order