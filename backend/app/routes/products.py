from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..services.inventory_service import InventoryService
from ..schemas.product import ProductCreate, ProductUpdate, Product as ProductSchema
from ..models.product import Product as ProductModel
from ..database import get_db

router = APIRouter()

@router.get("/", response_model=list[ProductSchema])
def read_products(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    products = db.query(ProductModel).offset(skip).limit(limit).all()
    return products

@router.post("/", response_model=ProductSchema)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    service = InventoryService(db)
    return service.create_product(product)

@router.get("/{product_id}", response_model=ProductSchema)
def read_product(product_id: str, db: Session = Depends(get_db)):
    service = InventoryService(db)
    db_product = service.get_product(product_id)
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return db_product

@router.put("/{product_id}", response_model=ProductSchema)
def update_product(product_id: str, product: ProductUpdate, db: Session = Depends(get_db)):
    service = InventoryService(db)
    db_product = service.update_product(product_id, product)
    if db_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return db_product