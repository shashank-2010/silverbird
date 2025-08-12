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
import joblib
import schedule
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
from app.feature_engineering.FeatureEngineering import FeatureEngineering
from app.ai_models_class.ALSTMModel import AttentionLSTM
from app.ai_models_class.LSTMModel import LSTMModel
from app.ai_models_class.RNNModel import RNNModel
from app.ai_models_class.TCNNModel import CNNLSTMModel
from sklearn.metrics import mean_squared_error
import pandas as pd

# Constants
SEQ_LEN = 30
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_DIR = "D:/models"
SCALAR_SAVE_DIR = "D:/scalar"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

PERIODS = [1, 3]
MODELS = {
    "lstm": LSTMModel,
    "attn_lstm": AttentionLSTM,
    "CNN-LSTM": CNNLSTMModel
}

# SYMBOLS = [
#     "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
#     "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
#     "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
#     "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL",
#     "JIOFIN", "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
#     "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS",
#     "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"
# ]

SYMBOLS = ["ADANIPORTS", "APOLLOHOSP","AXISBANK",
    "BAJFINANCE", "BEL", "BHARTIARTL","COALINDIA",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
     "HINDUNILVR", "ICICIBANK", "ITC",
    "KOTAKBANK","MARUTI","NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "TCS",
     "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN",
]
# SYMBOLS = ["ADANIPORTS"]

def directional_accuracy(y_true_seq, y_pred, forecast_days):
    """
    y_true_seq: original array of target prices (sequence aligned with model inputs)
    y_pred: predicted prices (same length as y_true after seq_len + forecast_days)
    forecast_days: the forecast horizon used
    """

    y_true_seq = np.array(y_true_seq)
    y_pred = np.array(y_pred)

    # Get the actual "current" values by shifting backward
    y_true_now = y_true_seq[:-forecast_days]
    y_true_future = y_true_seq[forecast_days:]

    # Make sure lengths match predicted values
    y_true_now = y_true_now[-len(y_pred):]
    y_true_future = y_true_future[-len(y_pred):]

    # Direction of actual change
    actual_dir = np.sign(y_true_future - y_true_now)
    predicted_dir = np.sign(y_pred - y_true_now)

    correct = actual_dir == predicted_dir
    return np.mean(correct)


def evaluate_model(model, data_loader, device, y_true_full, forecast_days):
    model.eval()
    predictions = []

    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            pred = model(xb).squeeze(-1)
            predictions.extend(pred.cpu().numpy())

    # Slice y_true to match predictions
    y_true = y_true_full[:len(predictions)]

    mse = mean_squared_error(y_true, predictions)
    dir_acc = directional_accuracy(y_true_full, predictions, forecast_days)

    return {
        "MSE": mse,
        "Directional_Accuracy": dir_acc
    }


def create_sequences(data, seq_len, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - forecast_horizon):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len + forecast_horizon - 1][3])  # 'close' column
    return np.array(X), np.array(y)


def train_model_entry(args):
    symbol, model_name, forecast_days, raw_data = args

    df = pd.DataFrame(raw_data)
    df = df[df['symbol'] == symbol].copy()
    df = df.sort_values('date')
    df = df.drop(columns=['symbol', 'date'])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaler_path = os.path.join(SCALAR_SAVE_DIR, f"{symbol}_{model_name}_{forecast_days}_scaler.save")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Saved scaler: {scaler_path}")

    X, y = create_sequences(scaled_data, SEQ_LEN, forecast_days)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = MODELS[model_name]
    model = model_class(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in trange(EPOCHS, desc=f"{symbol}-{model_name}-{forecast_days}", leave=False):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{symbol}_day-{forecast_days}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Saved model: {model_path}")
    # metrics = evaluate_model(model, loader, device)
    # print(f"[EVALUATION] {symbol}-{model_name}-{forecast_days}: {metrics}")


def parallel_train_all_models(csv_path, symbol):
    engineer = FeatureEngineering(csv_path)
    df = engineer.load_and_prepare()

    raw_data = df.to_dict(orient='records')  # Make it serializable
    tasks = [
        (symbol, model_name, period, raw_data)
        for model_name in MODELS
        for period in PERIODS
    ]

    with ProcessPoolExecutor() as executor:
        executor.map(train_model_entry, tasks)


def run_training_job():
    print("[SCHEDULER] Triggered training job for all symbols...")
    for symbol in SYMBOLS:
        file_path = os.path.join("D:/company", f"{symbol}_OHLC_10_years_daily.csv")
        if os.path.exists(file_path):
            parallel_train_all_models(file_path, symbol)
        else:
            print(f"[WARNING] CSV not found for {symbol} -> {file_path}")


if __name__ == "__main__":
    schedule.every().day.at("00:10").do(run_training_job)

    print("[SCHEDULER] Started. Waiting for next run...")
    while True:
        schedule.run_pending()
        time.sleep(60)