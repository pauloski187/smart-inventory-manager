from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from ..database import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(String, primary_key=True)  # CustomerID from CSV
    name = Column(String, nullable=False)
    customer_type = Column(String)  # New, Returning, VIP
    city = Column(String)
    state = Column(String)
    country = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    orders = relationship("Order", back_populates="customer")