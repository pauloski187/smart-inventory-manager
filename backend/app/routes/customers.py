from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..models import Customer
from ..database import get_db

router = APIRouter()

@router.get("/")
def read_customers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    customers = db.query(Customer).offset(skip).limit(limit).all()
    return customers

@router.get("/{customer_id}")
def read_customer(customer_id: str, db: Session = Depends(get_db)):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    return customer