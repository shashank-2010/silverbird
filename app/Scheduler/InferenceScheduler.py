import os
import sys
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
import time
import schedule
from datetime import datetime
from app.service.StockPredictor import StockPricePredictor
from app.feature_engineering.FeatureEngineering import FeatureEngineering
from tinydb import TinyDB, Query

# Add parent paths
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
for path in [os.getcwd(), parent_dir, grandparent_dir]:
    if path not in sys.path:
        sys.path.append(path)

# Global symbols list
SYMBOLS = ["ADANIPORTS", "APOLLOHOSP","AXISBANK",
    "BAJFINANCE", "BEL", "BHARTIARTL","COALINDIA",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
     "HINDUNILVR", "ICICIBANK", "ITC",
    "KOTAKBANK","MARUTI","NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "TCS",
     "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN",
]
#SYMBOLS = ["ADANIENT"]


class PredictionResult:
    def __init__(self, csv_path, db_path='prediction_store.json'):
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
            #rnn_pred = self.models.predict_rnn(symbol, days, self.df)
            cnn_lstm_pred = self.models.predict_cnn_lstm(symbol, days, self.df)

            votes = [lstm_pred['direction'], attn_lstm_pred['direction'], cnn_lstm_pred['direction']]
            majority_direction = 'up' if votes.count('up') > votes.count('down') else 'down'

            majority_prices = [
                p['predicted_price']
                for p in [lstm_pred, attn_lstm_pred, cnn_lstm_pred]
                if p['direction'] == majority_direction
            ]
            average_price = sum(majority_prices) / len(majority_prices) if majority_prices else None

            results[f"{days}_day"] = {
                'predicted_price': average_price,
                'direction': majority_direction,
                'models': {
                    'lstm': lstm_pred,
                    'attentional_lstm': attn_lstm_pred,
                    "CNN-LSTM": cnn_lstm_pred
                }
            }

        # Store in DB
        entry = {
            'id': doc_id,
            'symbol': symbol,
            'date': today,
            'predictions': results
        }
        Symbol = Query()
        self.db.upsert(entry, Symbol.id == doc_id)

        return results


def run_prediction_job():
    data_dir = "D:/company"
    print(f"\n[INFO] Running scheduled prediction job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for symbol in SYMBOLS:
        file_path = os.path.join(data_dir, f"{symbol}_OHLC_10_years_daily.csv")

        if not os.path.exists(file_path):
            print(f"[WARNING] CSV not found for {symbol}: {file_path}")
            continue

        try:
            predictor = PredictionResult(csv_path=file_path)
            result = predictor.predict_for_periods(symbol)

            print(f"[SUCCESS] Prediction stored for {symbol}:")
            for day, output in result.items():
                print(f"  {day} → price: {output['predicted_price']:.2f}, direction: {output['direction']}")
        except Exception as e:
            print(f"[ERROR] Prediction failed for {symbol}: {str(e)}")


if __name__ == "__main__":
    # Schedule to run daily at 7:00 AM
    schedule.every().day.at("17:46").do(run_prediction_job)

    print("[INFO] Scheduler is active. Waiting for the next scheduled prediction run.")
    while True:
        schedule.run_pending()
        time.sleep(60)
