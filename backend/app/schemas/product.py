from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ProductBase(BaseModel):
    id: str
    name: str
    category: Optional[str] = None
    brand: Optional[str] = None
    unit_price: float
    current_stock: int = 0
    reorder_threshold: int = 10
    is_active: bool = True

class ProductCreate(ProductBase):
    pass

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    unit_price: Optional[float] = None
    current_stock: Optional[int] = None
    reorder_threshold: Optional[int] = None
    is_active: Optional[bool] = None

class Product(ProductBase):
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True