from pydantic import BaseModel
from typing import List
from datetime import date
from pydantic import RootModel

class CurrentPriceResponse(BaseModel):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None 

class HistoricalPriceEntry(BaseModel):
    symbol:str
    date: date  
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None  # Optional

class HistoricalPriceResponse(RootModel[List[HistoricalPriceEntry]]):
    pass
