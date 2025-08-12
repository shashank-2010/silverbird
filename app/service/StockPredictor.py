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

from app.ai_models_class.ALSTMModel import AttentionLSTM
from app.ai_models_class.LSTMModel import LSTMModel
from app.ai_models_class.RNNModel import RNNModel
from app.ai_models_class.TCNNModel import CNNLSTMModel
    
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

class StockPricePredictor:
    def __init__(self, input_dim, model_dir="D:/models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.scalar_dir = "D:/scalar"
        self.seq_len = 30
        self.input_dim = input_dim
        self.scaler = MinMaxScaler()

    def _load_model(self, model_name, symbol, days):
        path = os.path.join(self.model_dir, f"{model_name}_{symbol}_day-{days}.pth")
        if model_name == "lstm":
            model = LSTMModel(self.input_dim).to(self.device)
        elif model_name == "attn_lstm":
            model = AttentionLSTM(self.input_dim).to(self.device)
        elif model_name == "rnn":
            model = RNNModel(self.input_dim).to(self.device)
        elif model_name == "CNN-LSTM":
            model = CNNLSTMModel(self.input_dim).to(self.device)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"Model not found at: {path}")

    def _load_scaler(self, symbol,model_name,days):
        scaler_path = os.path.join(self.scalar_dir, f"{symbol}_{model_name}_{days}_scaler.save")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

    def preprocess(self, df, symbol):
        df = df[df['symbol'] == symbol].copy().sort_values('date')
        df = df.drop(columns=['symbol', 'date'])

        scaled = self.scaler.transform(df)
        data = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        if len(data) < self.seq_len:
            raise ValueError("Not enough data for prediction.")
        return data[-self.seq_len:].unsqueeze(0)  # (1, seq_len, input_dim)

    def predict_model(self, model_name, symbol, days, df):
        self._load_scaler(symbol,model_name,days)
        model = self._load_model(model_name, symbol, days)
        input_tensor = self.preprocess(df, symbol)

        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        print(self.scaler.feature_names_in_.tolist())
        close_index = self.scaler.feature_names_in_.tolist().index('close')
        dummy = np.zeros((1, self.input_dim))
        dummy[0][close_index] = prediction
        prediction_unscaled = self.scaler.inverse_transform(dummy)[0][close_index]

        # dummy = np.zeros((1, self.input_dim))
        # dummy[0][3] = prediction  
        # prediction_unscaled = self.scaler.inverse_transform(dummy)[0][3]

        current_price = df[df['symbol'] == symbol].iloc[-1]['close']
        direction = "up" if prediction_unscaled > current_price else "down"

        return {
            "predicted_price": round(prediction_unscaled, 2),
            "direction": direction
        }

    def predict_lstm(self, symbol, days, df):
        return self.predict_model("lstm", symbol, days, df)

    def predict_attentional_lstm(self, symbol, days, df):
        return self.predict_model("attn_lstm", symbol, days, df)

    def predict_rnn(self, symbol, days, df):
        return self.predict_model("rnn", symbol, days, df)
    
    def predict_cnn_lstm(self, symbol, days, df):
        return self.predict_model("CNN-LSTM", symbol, days, df)

