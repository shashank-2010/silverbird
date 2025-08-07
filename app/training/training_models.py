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


import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from app.feature_engineering.FeatureEngineering import FeatureEngineering
from app.ai_models_class.ALSTMModel import AttentionLSTM
from app.ai_models_class.LSTMModel import LSTMModel
from app.ai_models_class.RNNModel import RNNModel
from multiprocessing import Process

SEQ_LEN = 30
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_DIR = "D:/models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

PERIODS = [1, 3, 5]
MODELS = {
    "lstm": LSTMModel,
    "attn_lstm": AttentionLSTM,
    "rnn": RNNModel
}


def create_sequences(data, seq_len, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - forecast_horizon):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len + forecast_horizon - 1][3])  
    return np.array(X), np.array(y)


def train_model(symbol, model_name, forecast_days, df):
    print(f"Training {model_name} for {symbol} with {forecast_days}-day horizon...")

    df = df[df['symbol'] == symbol].copy()
    df = df.sort_values('date')
    df = df.drop(columns=['symbol', 'date'])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
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

    for epoch in trange(EPOCHS, desc=f"Training {model_name} ({forecast_days}-day)"):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f"[{model_name}] Day-{forecast_days} Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss:.4f}")

    model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{symbol}_day-{forecast_days}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")


def parallel_train_all_models(csv_path, symbol):
    # Load and prepare features
    engineer = FeatureEngineering(csv_path)
    df = engineer.load_and_prepare()

    jobs = []
    for model_name in MODELS.keys():
        for period in PERIODS:
            p = Process(target=train_model, args=(symbol, model_name, period, df))
            p.start()
            jobs.append(p)

    for job in jobs:
        job.join()


if __name__ == "__main__":
    symbols = ["ADANIENT"]
    for symbol in symbols:
        file_name = f"{symbol}_OHLC_10_years_daily.csv"
        file_path = os.path.join("D:/company", file_name)
        parallel_train_all_models(file_path, symbol=symbol)
