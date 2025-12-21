from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    sku = Column(String, unique=True, nullable=False)
    category = Column(String)
    price = Column(Float, nullable=False)
    cost = Column(Float)
    stock_quantity = Column(Integer, default=0)
    min_stock_level = Column(Integer, default=0)
    max_stock_level = Column(Integer)
    supplier_id = Column(Integer)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)