import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import yfinance as yf
from datetime import datetime, timedelta
from tinydb import TinyDB, Query

from app.service.PredictionResult import PredictionResult

class StockService:
    def __init__(self, path = 'prediction_store.json'):
        self.path = path
    
    def get_current_price(self, symbol:str):
        data = yf.Ticker(symbol)
        hist = data.history(period='1d')
        if hist.empty:
            return None
        ohlc = hist[['Open','High','Low','Close']].iloc[-1]
        return [{
        "symbol": symbol.replace(".NS", ""),  
        "open": float(ohlc["Open"]),
        "high": float(ohlc["High"]),
        "low": float(ohlc["Low"]),
        "close": float(ohlc["Close"])
    }]
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str = None):
        if end_date is None:
            end_date = date.today().strftime('%Y-%m-%d')

        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

        if data.empty:
            return []

        data.columns = [col[0] for col in data.columns] 
        
        records = []
        for row in data.itertuples():
            records.append({
                "symbol": symbol.replace(".NS", ""),
                "date": row.Index.strftime('%Y-%m-%d'),
                "open": float(row.Open),
                "high": float(row.High),
                "low": float(row.Low),
                "close": float(row.Close),
                "volume": float(row.Volume),
            })

        return records

        
    def predict_stock_price(self, symbol:str):
            return PredictionResult.predict_for_periods(symbol=symbol)
    
    def get_prediction_from_db(self, symbol, days):
        today = datetime.now().strftime("%Y-%m-%d")
        db = TinyDB(self.path)
        Symbol = Query()
        result = db.get(Symbol.id == f"{symbol}_{today}")

        if result:
            return result['predictions'].get(f"{days}_day"), result['date']
        else:
            return None
        
    def get_market_predictions(self, symbol: str, days_list: list):
        all_preds = {}
        _date = None 

        for day in days_list:
            result = self.get_prediction_from_db(symbol, day)

            if result is None:
                continue  

            prediction, _date = result
            all_preds[f"{day}_day"] = prediction

        return {
            "symbol": symbol,
            "date": _date,
            "predictions": all_preds
        }

        




