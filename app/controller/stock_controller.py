import os
import sys
import datetime
from fastapi import Depends
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from app.service.StockService import StockService
from app.repository.CompanyRepository import CompanyRepository
from app.database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class StockController:
    def __init__(self):
        self.db = next(get_db())
        self.service = StockService()
        self.company_repo = CompanyRepository(db=self.db)

    def get_price_data(self, symbol:str, start_date:str =None, end_date:str = None):
        nse_symbols = symbol + ".NS"
        if start_date:
            return self.service.get_historical_data(nse_symbols, start_date, end_date)
        else:
            return self.service.get_current_price(nse_symbols)
        
    def get_predictions(self, symbol: str, days_list: list):
        try:
            all_preds = {}
            for day in days_list:
                prediction, _date = self.service.get_prediction_from_db(symbol, day, date=None)
                all_preds[f"{day}_day"] = prediction
            return {
                "symbol": symbol,
                "date": _date,
                "predictions": all_preds
            }
        except Exception as e:
            return {"message":"We dont have data for this symbol currently. Be patient, and we will give you seemless information very soon"}
    
    def process_prediction_summary(self, prediction_data: dict) -> dict:
        if not prediction_data.get("predictions"):
            return prediction_data

        cleaned_predictions = {}
        for day_key, day_data in prediction_data["predictions"].items():
            cleaned_predictions[day_key] = {
                "predicted_price": day_data["predicted_price"],
                "direction": day_data["direction"]
            }

        prediction_data["predictions"] = cleaned_predictions
        return prediction_data
    
    def get_market_trends(self, direction: str, days: list[int]):
        symbols = self.company_repo.get_all_symbols()
        result = []

        for symbol in symbols:
            preds = self.service.get_market_predictions(symbol, days_list=days)
            for day in days:
                day_key = f"{day}_day"
                if preds["predictions"].get(day_key, {}).get("direction") == direction:
                    result.append({
                        "symbol": symbol,
                        "day": day,
                        "predicted_price": preds["predictions"][day_key]["predicted_price"],
                        "direction":direction
                    })
        if not result:
            return {
                "message": f"No stocks found trending '{direction}' in the next {days} day(s)."
            }

        return result
    
    def get_past_predictions(self, symbol: str, days_list: list, date):
        try:
            all_preds = {}
            for day in days_list:
                prediction, _date = self.service.get_prediction_from_db(symbol, day, date)
                all_preds[f"{day}_day"] = prediction
            return {
                    "symbol": symbol,
                    "date": _date,
                    "predictions": all_preds
                }
        except Exception as e:
            return {"message":"We dont have data for this symbol currently. Be patient, and we will give you seemless information very soon"}
        
