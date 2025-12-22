from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True)  # ProductID from CSV
    name = Column(String, nullable=False)
    category = Column(String)
    brand = Column(String)
    unit_price = Column(Float, nullable=False)
    current_stock = Column(Integer, default=0)
    reorder_threshold = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    orders = relationship("Order", back_populates="product")
    inventory_movements = relationship("InventoryMovement", back_populates="product")