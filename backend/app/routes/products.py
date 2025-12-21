from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..controllers.inventory_service import get_products, create_product, update_product, delete_product
from ..models.product import Product
from ..config.settings import settings
from ..middleware.auth_middleware import get_current_user

router = APIRouter()

@router.get("/")
def read_products(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    products = get_products(db, skip=skip, limit=limit)
    return products

@router.post("/")
def create_new_product(product: Product, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return create_product(db=db, product=product)

@router.put("/{product_id}")
def update_existing_product(product_id: int, product: Product, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return update_product(db=db, product_id=product_id, product=product)

@router.delete("/{product_id}")
def delete_existing_product(product_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return delete_product(db=db, product_id=product_id)

# Note: get_db function needs to be defined, probably in a database session utility