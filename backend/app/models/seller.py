from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Seller(Base):
    __tablename__ = "sellers"

    id = Column(String, primary_key=True)  # SellerID from CSV
    name = Column(String)  # Can be derived or set
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    orders = relationship("Order", back_populates="seller")