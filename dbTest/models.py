from database import Base
from sqlalchemy import Column, Integer, String

class Expenses(Base):
    __tablename__ = "expenses"

    id = Column(Integer, primary_key=True, index=True)
    item_name = Column(String)
    category = Column(String)
    price = Column(Integer)