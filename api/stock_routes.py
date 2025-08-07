# stock_routes.py
from fastapi import APIRouter, Depends, Query
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from app.controller.stock_controller import StockController
controller = StockController()

from app.schemas.ohlc import CurrentPriceResponse, HistoricalPriceResponse
from app.database import SessionLocal

router = APIRouter(prefix="/stocks", tags=["Stocks"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/price", response_model=List[CurrentPriceResponse])
def get_current_price(symbol: str = Query(..., description="Ticker symbol (e.g., AAPL)")):
    result = controller.get_price_data(symbol=symbol)
    if result is None:
        return {"error": "No data found for the given symbol."}
    return result

@router.get("/historical", response_model=HistoricalPriceResponse)
def get_historical_price(
    symbol: str = Query(...),
    start_date: str = Query(...),
    end_date: Optional[str] = Query(None)
):
    result = controller.get_price_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if not result:
        return []
    return result

@router.get("/predict", response_model=Dict[str, Any])
def get_model_predictions(
    symbol: str = Query(..., description="Ticker symbol (e.g., AAPL)"),
    days: Optional[List[int]] = Query([1, 3, 5], description="List of days ahead to predict (e.g., 1,3,5)")
):
    result = controller.get_predictions(symbol=symbol, days_list=days)

    if not result or not result.get("predictions"):
        return {"error": "No prediction found for the given symbol and days."}
    return result