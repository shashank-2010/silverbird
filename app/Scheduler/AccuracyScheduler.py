import os
import sys
import time
import schedule
from datetime import datetime
from tinydb import TinyDB, Query


# Path setup
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))

for p in [parent_dir, grandparent_dir, os.getcwd()]:
    if p not in sys.path:
        sys.path.append(p)

import numpy as np
import torch
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from app.ai_models_class.LSTMModel import LSTMModel
from app.ai_models_class.ALSTMModel import AttentionLSTM
from app.ai_models_class.TCNNModel import CNNLSTMModel
from app.feature_engineering.FeatureEngineering import FeatureEngineering
from app.feature_engineering.features_collinearity import Multicollinearity
from sklearn.model_selection import TimeSeriesSplit

# Constants
SEQ_LEN = 30
BATCH_SIZE = 32
SYMBOLS = [
    "ADANIPORTS", "APOLLOHOSP", "AXISBANK", "BAJFINANCE", "BEL", "BHARTIARTL",
    "COALINDIA", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDUNILVR",
    "ICICIBANK", "ITC", "KOTAKBANK", "MARUTI", "NESTLEIND", "ONGC", "POWERGRID",
    "RELIANCE", "TCS", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN"
]
FORECAST_DAYS = [1, 3]
DB_PATH = "daily_forecast_results.json"
DATA_FOLDER = r"D:\company"

class ModelEvaluator:
    def __init__(self, model_class, df, forecast_days=1, test_ratio=0.2, device=None):
        self.model_class = model_class
        self.df = df.sort_values("date").drop(columns=["symbol", "date"]).copy()
        self.forecast_days = forecast_days
        self.test_ratio = test_ratio
        self.scaler = MinMaxScaler()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = self.df.shape[1]

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - SEQ_LEN - self.forecast_days):
            X.append(data[i:i + SEQ_LEN])
            y.append(data[i + SEQ_LEN + self.forecast_days - 1][3])  # 4th col is 'close'
        return np.array(X), np.array(y)

    def cross_validate(self, splits=5, epochs=50, lr=0.001):
        scaled_data = self.scaler.fit_transform(self.df)
        X, y = self.create_sequences(scaled_data)

        tscv = TimeSeriesSplit(n_splits=splits)
        all_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            print(f"\nFold {fold + 1}/{splits}")
            model = self.train(X_train, y_train, epochs=epochs, lr=lr)
            metrics = self.evaluate(model, X_test, y_test)
            all_metrics.append(metrics)
            print(f"Fold {fold + 1} Metrics: {metrics}")

        avg_metrics = {
            "MSE": round(np.mean([m["MSE"] for m in all_metrics]), 4),
            "MAE": round(np.mean([m["MAE"] for m in all_metrics]), 4),
            "Directional Accuracy": round(np.mean([m["Directional Accuracy"] for m in all_metrics]), 2)
        }

        print(f"\nAverage Cross-Validated Metrics: {avg_metrics}")
        return avg_metrics

    def split_data(self):
        scaled_data = self.scaler.fit_transform(self.df)
        X, y = self.create_sequences(scaled_data)

        split_idx = int(len(X) * (1 - self.test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, epochs=50, lr=0.001):
        model = self.model_class(self.input_dim).to(self.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in trange(epochs, desc=f"Training {self.model_class.__name__} ({self.forecast_days}-day)"):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        return model

    def directional_accuracy_fixed(self, y_true_full, y_pred):
        forecast_days = self.forecast_days
        y_full = np.array(y_true_full)
        y_pred = np.array(y_pred)

        max_valid_len = min(len(y_full) - forecast_days, len(y_pred))

        y_now = y_full[:max_valid_len]
        y_future = y_full[forecast_days:forecast_days + max_valid_len]
        y_pred_trimmed = y_pred[:max_valid_len]

        actual_direction = np.sign(y_future - y_now)
        predicted_direction = np.sign(y_pred_trimmed - y_now)

        return np.mean(actual_direction == predicted_direction)

    def evaluate(self, model, X_test, y_test_full):
        forecast_days = self.forecast_days
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32).to(self.device)).squeeze().cpu().numpy()

        y_test_aligned = np.array(y_test_full[:len(preds)])

        mse = mean_squared_error(y_test_aligned, preds)
        mae = mean_absolute_error(y_test_aligned, preds)
        dir_acc = self.directional_accuracy_fixed(y_test_full, preds)

        return {
            "MSE": round(mse, 4),
            "MAE": round(mae, 4),
            "Directional Accuracy": round(dir_acc * 100, 2)
        }

    def run(self, epochs=50, lr=0.001, use_cv=False):
        self.compute_feature_correlations()
        checker = Multicollinearity(self.df, target_columns='close', threshold=0.80)
        checker.report()
        if use_cv:
            return self.cross_validate(epochs=epochs, lr=lr)
        else:
            X_train, y_train, X_test, y_test = self.split_data()
            model = self.train(X_train, y_train, epochs, lr)
            metrics = self.evaluate(model, X_test, y_test)
            return metrics

    def compute_feature_correlations(self):
        df = self.df.copy()
        target_shift = SEQ_LEN + self.forecast_days - 1
        future_close = self.df["close"].shift(-target_shift)
        df["target_close"] = future_close
        df = df.dropna()
        correlations = df.drop(columns=["target_close"]).corrwith(df["target_close"])

        print(f"\nFeature correlations with {self.forecast_days}-day future close:")
        for feature, corr in correlations.items():
            print(f" - {feature:>10}: {corr:.4f}")
        return correlations


def evaluate_all_models(csv_path, symbol, forecast_days=3, epochs=50):
    print(f"\nEvaluating models for symbol: {symbol} | Forecast: {forecast_days} days")

    engineer = FeatureEngineering(csv_path)
    df = engineer.load_and_prepare()
    df_symbol = df[df["symbol"] == symbol]

    models = {
        "LSTM": LSTMModel,
        "Attention LSTM": AttentionLSTM,
        "CNN-LSTM": CNNLSTMModel
    }

    results = {}

    for model_name, model_class in models.items():
        print(f"\nEvaluating {model_name}...")
        evaluator = ModelEvaluator(model_class, df_symbol, forecast_days=forecast_days)
        metrics = evaluator.run(epochs=epochs, use_cv=False)
        print(f"{model_name} Metrics: {metrics}")
        results[model_name] = metrics

    return results

def run_accuracy_job():
    db = TinyDB(DB_PATH)
    symbol_query = Query()
    today_str = datetime.now().strftime("%Y-%m-%d")

    print(f"[{datetime.now()}] Running daily prediction job...")

    for symbol in SYMBOLS:
        result_doc = {
            "_id": f"{symbol}_{today_str}",
            "symbol": symbol,
            "forecast_results": {}
        }

        for forecast_day in FORECAST_DAYS:
            csv_path = os.path.join(DATA_FOLDER, f"{symbol}_OHLC_10_years_daily.csv")
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                continue

            try:
                results = evaluate_all_models(csv_path=csv_path, symbol=symbol, forecast_days=forecast_day)

                # Get the best directional accuracy
                best_dir_acc = max(m["Directional Accuracy"] for m in results.values())
                result_doc["forecast_results"][str(forecast_day)] = {
                    "directional_accuracy": best_dir_acc
                }

                print(f"{symbol} | {forecast_day}-day | Best Dir. Accuracy: {best_dir_acc}")

            except Exception as e:
                print(f"Error processing {symbol} ({forecast_day}-day): {e}")

        # Remove existing document with same _id (if re-run)
        db.remove((symbol_query._id == result_doc["_id"]))

        # Insert updated result
        db.insert(result_doc)

    print(f"[{datetime.now()}] Daily prediction job completed.\n")

if __name__ == "__main__":
    # Schedule to run daily at 04:35 AM
    schedule.every().day.at("10:15").do(run_accuracy_job)

    print("[INFO] Scheduler is active. Waiting for the next scheduled prediction run...")
    while True:
        schedule.run_pending()
        time.sleep(60)

