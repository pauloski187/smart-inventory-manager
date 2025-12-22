from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class OrderBase(BaseModel):
    id: str
    order_date: datetime
    customer_id: str
    product_id: str
    seller_id: str
    quantity: int
    unit_price: float
    discount: float = 0.0
    tax: float = 0.0
    shipping_cost: float = 0.0
    total_amount: float
    payment_method: Optional[str] = None
    order_status: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

class OrderCreate(OrderBase):
    pass

class OrderUpdate(BaseModel):
    order_status: Optional[str] = None

class Order(OrderBase):
    created_at: datetime

    class Config:
        from_attributes = True