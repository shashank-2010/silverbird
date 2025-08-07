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
from datetime import datetime, timedelta
from app.service.StockPredictor import StockPricePredictor
from app.feature_engineering.FeatureEngineering import FeatureEngineering

from tinydb import TinyDB, Query
from datetime import datetime
import os

class PredictionResult:
    def __init__(self, csv_path, db_path='prediction_store_exp.json'):
        self.df = self._load_features(csv_path)
        input_dim = self.df.drop(columns=['symbol', 'date']).shape[1]
        self.models = StockPricePredictor(input_dim)
        self.days_list = [1, 3]

        self.db = TinyDB(db_path)

    def _load_features(self, csv_path):
        engineer = FeatureEngineering(csv_path)
        return engineer.load_and_prepare()

    def predict_for_periods(self, symbol):
        results = {}
        today = datetime.now().strftime("%Y-%m-%d")
        doc_id = f"{symbol}_{today}"

        for days in self.days_list:
            lstm_pred = self.models.predict_lstm(symbol, days, self.df)
            attn_lstm_pred = self.models.predict_attentional_lstm(symbol, days, self.df)
            rnn_pred = self.models.predict_rnn(symbol, days, self.df)

            votes = [lstm_pred['direction'], attn_lstm_pred['direction'], rnn_pred['direction']]
            majority_direction = 'up' if votes.count('up') > votes.count('down') else 'down'

            majority_prices = [
                p['predicted_price']
                for p in [lstm_pred, attn_lstm_pred, rnn_pred]
                if p['direction'] == majority_direction
            ]
            average_price = sum(majority_prices) / len(majority_prices) if majority_prices else None

            results[f"{days}_day"] = {
                'predicted_price': average_price,
                'direction': majority_direction,
                'models': {
                    'lstm': lstm_pred,
                    'attentional_lstm': attn_lstm_pred,
                    'rnn': rnn_pred
                }
            }

        # Store in DB (update if already exists)
        entry = {
            'id': doc_id,
            'symbol': symbol,
            'date': today,
            'predictions': results
        }

        Symbol = Query()
        self.db.upsert(entry, Symbol.id == doc_id)

        return results
    

if __name__ == "__main__":
    symbols = ["ADANIPORTS"]  
    data_dir = "D:/company" 

    for symbol in symbols:
        file_name = f"{symbol}_OHLC_10_years_daily.csv"
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            print(f"[ERROR] CSV file not found for {symbol}: {file_path}")
            continue

        print(f"[INFO] Processing prediction for {symbol}...")
        predictor = PredictionResult(csv_path=file_path)
        result = predictor.predict_for_periods(symbol)

        print(f"[SUCCESS] Prediction stored for {symbol}:")
        for day, output in result.items():
            print(f"  {day} → price: {output['predicted_price']:.2f}, direction: {output['direction']}")
