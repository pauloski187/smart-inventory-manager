from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..models import Seller
from ..database import get_db

router = APIRouter()

@router.get("/")
def read_sellers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    sellers = db.query(Seller).offset(skip).limit(limit).all()
    return sellers

@router.get("/{seller_id}")
def read_seller(seller_id: str, db: Session = Depends(get_db)):
    seller = db.query(Seller).filter(Seller.id == seller_id).first()
    return seller