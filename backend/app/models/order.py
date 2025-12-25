from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from ..database import Base


class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True)  # OrderID from CSV
    order_date = Column(DateTime, nullable=False)
    customer_id = Column(String, ForeignKey("customers.id"))
    product_id = Column(String, ForeignKey("products.id"))
    seller_id = Column(String, ForeignKey("sellers.id"))
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    discount = Column(Float, default=0.0)
    tax = Column(Float, default=0.0)
    shipping_cost = Column(Float, default=0.0)
    total_amount = Column(Float, nullable=False)
    profit = Column(Float, default=0.0)
    profit_margin = Column(Float, default=0.0)
    payment_method = Column(String)
    order_status = Column(String)
    delivery_date = Column(DateTime)
    returned = Column(Boolean, default=False)
    refund_amount = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="orders")
    product = relationship("Product", back_populates="orders")
    seller = relationship("Seller", back_populates="orders")