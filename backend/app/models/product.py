from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from ..database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True)  # ProductID from CSV
    name = Column(String, nullable=False)
    category = Column(String)
    brand = Column(String)
    unit_price = Column(Float, nullable=False)
    cost_price = Column(Float)  # Cost price from inventory
    current_stock = Column(Integer, default=0)
    initial_stock = Column(Integer, default=0)
    reorder_threshold = Column(Integer, default=10)
    reorder_quantity = Column(Integer, default=50)
    stock_status = Column(String, default="In Stock")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    orders = relationship("Order", back_populates="product")
    inventory_movements = relationship("InventoryMovement", back_populates="product")