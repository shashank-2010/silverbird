from sqlalchemy import Column, Integer, String
from app.database import Base 

class Nifty50Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    company_name = Column(String, nullable=False, index=True)
    industry = Column(String, nullable=True)
    symbol = Column(String, nullable=False, unique=True, index=True)
    series = Column(String, nullable=True)
    isin_code = Column(String, nullable=True)
