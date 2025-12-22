from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

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
    payment_method = Column(String)
    order_status = Column(String)
    city = Column(String)
    state = Column(String)
    country = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="orders")
    product = relationship("Product", back_populates="orders")
    seller = relationship("Seller", back_populates="orders")