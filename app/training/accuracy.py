# import os
# import sys

# parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# if grandparent_dir not in sys.path:
#     sys.path.append(grandparent_dir)

# if os.getcwd() not in sys.path:
#     sys.path.append(os.getcwd())

# import numpy as np
# import torch
# from tqdm import trange
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import MinMaxScaler
# from app.ai_models_class.LSTMModel import LSTMModel
# from app.ai_models_class.ALSTMModel import AttentionLSTM
# from app.ai_models_class.RNNModel import RNNModel
# from app.ai_models_class.TCNNModel import CNNLSTMModel
# from app.feature_engineering.FeatureEngineering import FeatureEngineering
# from app.feature_engineering.features_collinearity import Multicollinearity
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# SEQ_LEN = 30
# BATCH_SIZE = 32

# class ModelEvaluator:
#     def __init__(self, model_class, df, forecast_days=1, test_ratio=0.2, device=None):
#         self.model_class = model_class
#         self.df = df.sort_values("date").drop(columns=["symbol", "date"]).copy()
#         self.forecast_days = forecast_days
#         self.test_ratio = test_ratio
#         self.scaler = MinMaxScaler()
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.input_dim = self.df.shape[1]

#     def create_sequences(self, data):
#         X, y = [], []
#         for i in range(len(data) - SEQ_LEN - self.forecast_days):
#             X.append(data[i:i + SEQ_LEN])
#             y.append(data[i + SEQ_LEN + self.forecast_days - 1][3])  
#         return np.array(X), np.array(y)
    
#     def cross_validate(self, splits=5, epochs=50, lr=0.001):
#         scaled_data = self.scaler.fit_transform(self.df)
#         X, y = self.create_sequences(scaled_data)

#         tscv = TimeSeriesSplit(n_splits=splits)
#         all_metrics = []

#         for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
#             X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y[train_idx], y[test_idx]

#             print(f"\n⏱️ Fold {fold + 1}/{splits}")
#             model = self.train(X_train, y_train, epochs=epochs, lr=lr)
#             metrics = self.evaluate(model, X_test, y_test)
#             all_metrics.append(metrics)
#             print(f"📈 Fold {fold + 1} Metrics: {metrics}")

#         # Aggregate metrics
#         avg_metrics = {
#             "MSE": round(np.mean([m["MSE"] for m in all_metrics]), 4),
#             "MAE": round(np.mean([m["MAE"] for m in all_metrics]), 4),
#             "Directional Accuracy": round(np.mean([m["Directional Accuracy"] for m in all_metrics]), 2)
#         }

#         print(f"\n📊 Average Cross-Validated Metrics: {avg_metrics}")
#         return avg_metrics

#     def split_data(self):
#         scaled_data = self.scaler.fit_transform(self.df)
#         X, y = self.create_sequences(scaled_data)

#         split_idx = int(len(X) * (1 - self.test_ratio))
#         X_train, X_test = X[:split_idx], X[split_idx:]
#         y_train, y_test = y[:split_idx], y[split_idx:]

#         return X_train, y_train, X_test, y_test

#     def train(self, X_train, y_train, epochs=50, lr=0.001):
#         model = self.model_class(self.input_dim).to(self.device)
#         criterion = torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#         train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#                                       torch.tensor(y_train, dtype=torch.float32))
#         loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

#         for epoch in trange(epochs, desc=f"Training {self.model_class.__name__} ({self.forecast_days}-day)"):
#             model.train()
#             epoch_loss = 0
#             for xb, yb in loader:
#                 xb, yb = xb.to(self.device), yb.to(self.device)
#                 optimizer.zero_grad()
#                 preds = model(xb).squeeze()
#                 loss = criterion(preds, yb)
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
#             #print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

#         return model



#     def directional_accuracy_fixed(self,y_true_full, y_pred, forecast_days):
#         """
#         y_true_full: full unshifted target values from the dataset
#         y_pred: predicted prices from the model
#         forecast_days: forecast horizon (how many days ahead you're predicting)
#         """
#         y_true_full = np.array(y_true_full)
#         y_pred = np.array(y_pred)

#         # Get true current prices and future prices
#         y_now = y_true_full[:-forecast_days]
#         y_future = y_true_full[forecast_days:]

#         # Align with prediction length
#         y_now = y_now[-len(y_pred):]
#         y_future = y_future[-len(y_pred):]

#         actual_direction = np.sign(y_future - y_now)
#         predicted_direction = np.sign(y_pred - y_now)

#         return np.mean(actual_direction == predicted_direction)

#     def evaluate(self, model, X_test, y_test_full, forecast_days):
#         """
#         model       : trained PyTorch model
#         X_test      : test input features (numpy array)
#         y_test_full : full sequence of target prices used in test (not sliced)
#         forecast_days : number of steps ahead the model was trained to predict
#         """

#         model.eval()
#         with torch.no_grad():
#             preds = model(torch.tensor(X_test, dtype=torch.float32).to(self.device)).squeeze().cpu().numpy()

#         y_test_aligned = np.array(y_test_full[:len(preds)])  # actual future prices

#         mse = mean_squared_error(y_test_aligned, preds)
#         mae = mean_absolute_error(y_test_aligned, preds)

#         dir_acc = self.directional_accuracy_fixed(y_test_full, preds, forecast_days)

#         return {
#             "MSE": round(mse, 4),
#             "MAE": round(mae, 4),
#             "Directional Accuracy": round(dir_acc * 100, 2)
#         }


#     def run(self, epochs=50, lr=0.001, use_cv=False):
#         self.compute_feature_correlations()
#         checker = Multicollinearity(self.df, target_columns='close', threshold=0.80)
#         checker.report()
#         if use_cv:
#             return self.cross_validate(epochs=epochs, lr=lr)
#         else:
#             X_train, y_train, X_test, y_test = self.split_data()
#             model = self.train(X_train, y_train, epochs, lr)
#             metrics = self.evaluate(model, X_test, y_test)
#             return metrics

#     def compute_feature_correlations(self):
#         df = self.df.copy()
#         target_shift = SEQ_LEN + self.forecast_days - 1

#         # Reconstruct the target (future close price)
#         future_close = self.df["close"].shift(-target_shift)
#         df["target_close"] = future_close

#         # Drop rows with NaNs (from shifting)
#         df = df.dropna()

#         # Compute correlation of each input feature with target
#         correlations = df.drop(columns=["target_close"]).corrwith(df["target_close"])

#         print(f"\n📊 Feature correlations with {self.forecast_days}-day future close:")
#         for feature, corr in correlations.items():
#             print(f" - {feature:>10}: {corr:.4f}")
        
#         return correlations

    
# def evaluate_all_models(csv_path, symbol, forecast_days=3, epochs=50):
#         print(f"\n🔍 Evaluating models for symbol: {symbol} | Forecast: {forecast_days} days")

#         # Load and prepare data
#         engineer = FeatureEngineering(csv_path)
#         df = engineer.load_and_prepare()
#         print(df.columns)
#         df_symbol = df[df["symbol"] == symbol]

#         models = {
#             "LSTM": LSTMModel,
#             "Attention LSTM": AttentionLSTM,
#             "RNN": RNNModel,
#             "CNN-LSTM": CNNLSTMModel
#         }

#         results = {}

#         for model_name, model_class in models.items():
#             print(f"\n▶️ Evaluating {model_name}...")
#             evaluator = ModelEvaluator(model_class, df_symbol, forecast_days=forecast_days)
#             metrics = evaluator.run(epochs=epochs,use_cv=False)
#             print(f"{model_name} Metrics: {metrics}")
#             results[model_name] = metrics

#         return results

# if __name__ == "__main__":
#     evaluate_all_models(csv_path=r"D:\company\ADANIPORTS_OHLC_10_years_daily.csv", symbol="ADANIPORTS", forecast_days=3)

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
from sklearn.metrics import mean_squared_error, mean_absolute_error

SEQ_LEN = 30
BATCH_SIZE = 32

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
            y.append(data[i + SEQ_LEN + self.forecast_days - 1][3])  
        return np.array(X), np.array(y)
    
    def cross_validate(self, splits=5, epochs=50, lr=0.001):
        scaled_data = self.scaler.fit_transform(self.df)
        X, y = self.create_sequences(scaled_data)

        tscv = TimeSeriesSplit(n_splits=splits)
        all_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            print(f"\n⏱️ Fold {fold + 1}/{splits}")
            model = self.train(X_train, y_train, epochs=epochs, lr=lr)
            metrics = self.evaluate(model, X_test, y_test)
            all_metrics.append(metrics)
            print(f"📈 Fold {fold + 1} Metrics: {metrics}")

        # Aggregate metrics
        avg_metrics = {
            "MSE": round(np.mean([m["MSE"] for m in all_metrics]), 4),
            "MAE": round(np.mean([m["MAE"] for m in all_metrics]), 4),
            "Directional Accuracy": round(np.mean([m["Directional Accuracy"] for m in all_metrics]), 2)
        }

        print(f"\n📊 Average Cross-Validated Metrics: {avg_metrics}")
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
            epoch_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            #print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

        return model



    def directional_accuracy_fixed(self,y_true_full, y_pred, forecast_days):
        """
        y_true_full: full unshifted target values from the dataset
        y_pred: predicted prices from the model
        forecast_days: forecast horizon (how many days ahead you're predicting)
        """
        y_full = np.array(y_true_full)
        y_pred = np.array(y_pred)

        # Compute how many usable steps are available
        max_valid_len = min(len(y_full) - forecast_days, len(y_pred))

        y_now = y_full[:max_valid_len]
        y_future = y_full[forecast_days:forecast_days + max_valid_len]
        y_pred_trimmed = y_pred[:max_valid_len]

        actual_direction = np.sign(y_future - y_now)
        predicted_direction = np.sign(y_pred_trimmed - y_now)

        return np.mean(actual_direction == predicted_direction)

    def evaluate(self, model, X_test, y_test_full):
        """
        model       : trained PyTorch model
        X_test      : test input features (numpy array)
        y_test_full : full sequence of target prices used in test (not sliced)
        forecast_days : number of steps ahead the model was trained to predict
        """
        forecast_days = self.forecast_days
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32).to(self.device)).squeeze().cpu().numpy()

        y_test_aligned = np.array(y_test_full[:len(preds)])  # actual future prices

        mse = mean_squared_error(y_test_aligned, preds)
        mae = mean_absolute_error(y_test_aligned, preds)

        dir_acc = self.directional_accuracy_fixed(y_test_full, preds, forecast_days)

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

        # Reconstruct the target (future close price)
        future_close = self.df["close"].shift(-target_shift)
        df["target_close"] = future_close

        # Drop rows with NaNs (from shifting)
        df = df.dropna()

        # Compute correlation of each input feature with target
        correlations = df.drop(columns=["target_close"]).corrwith(df["target_close"])

        print(f"\n📊 Feature correlations with {self.forecast_days}-day future close:")
        for feature, corr in correlations.items():
            print(f" - {feature:>10}: {corr:.4f}")
        
        return correlations

    
def evaluate_all_models(csv_path, symbol, forecast_days=3, epochs=50):
        print(f"\n🔍 Evaluating models for symbol: {symbol} | Forecast: {forecast_days} days")

        # Load and prepare data
        engineer = FeatureEngineering(csv_path)
        df = engineer.load_and_prepare()
        print(df.columns)
        df_symbol = df[df["symbol"] == symbol]

        models = {
            "LSTM": LSTMModel,
            "Attention LSTM": AttentionLSTM,
            "CNN-LSTM": CNNLSTMModel
        }

        results = {}

        for model_name, model_class in models.items():
            print(f"\n▶️ Evaluating {model_name}...")
            evaluator = ModelEvaluator(model_class, df_symbol, forecast_days=forecast_days)
            metrics = evaluator.run(epochs=epochs,use_cv=False)
            print(f"{model_name} Metrics: {metrics}")
            results[model_name] = metrics

        return results

if __name__ == "__main__":
    evaluate_all_models(csv_path=r"D:\company\ADANIPORTS_OHLC_10_years_daily.csv", symbol="ADANIPORTS", forecast_days=3)

