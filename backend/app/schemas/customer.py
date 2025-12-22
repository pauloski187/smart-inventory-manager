from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from .order import Order

class CustomerBase(BaseModel):
    id: str
    name: str
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    is_active: bool = True

class CustomerCreate(CustomerBase):
    pass

class CustomerUpdate(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    is_active: Optional[bool] = None

class Customer(CustomerBase):
    created_at: datetime
    orders: List[Order] = []

    class Config:
        from_attributes = True