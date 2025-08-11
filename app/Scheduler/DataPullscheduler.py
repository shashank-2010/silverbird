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
import pandas as pd
import os
import schedule
import time
from datetime import datetime, timedelta
from app.Utility.utility import log_execution

class DailyDataPullScheduler:
    def __init__(self, symbols: list, folder: str, run_time="07:21"):
        self.symbols = symbols 
        self.folder = folder
        self.run_time = run_time

    def get_csv_path(self, symbol: str):
        return os.path.join(self.folder, f"{symbol}_OHLC_10_years_daily.csv")

    def get_last_date_from_csv(self, csv_path: str):
        if not os.path.exists(csv_path):
            return None
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['date'])
            df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='coerce')
            df = df.dropna(subset=['date'])
            if df.empty:
                return None
            return df['date'].max().date()
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return None

    @log_execution
    def fetch_data(self, symbol: str, start_date: datetime):
        end_date = datetime.combine(datetime.today().date(), datetime.min.time())
        if start_date >= end_date:
            return pd.DataFrame()

        yf_symbol = symbol + ".NS"
        data = yf.download(yf_symbol, start=start_date + timedelta(days=1), end=end_date)
        if data.empty:
            return pd.DataFrame()

        data.reset_index(inplace=True)
        data = data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
        data['date'] = pd.to_datetime(data['date']).dt.strftime('%d-%m-%Y')
        clean_symbol = yf_symbol.replace('.NS', '')
        data.insert(0, 'symbol', clean_symbol)
        return data
    @log_execution
    def append_to_csv(self, data: pd.DataFrame, csv_path: str):
        if data.empty:
            print(f"No new data for {os.path.basename(csv_path)}.")
            return
        if not os.path.exists(csv_path):
            data.to_csv(csv_path, index=False)
        else:
            data.to_csv(csv_path, mode='a', index=False, header=False)
        print(f"Appended {len(data)} rows to {csv_path}")

    @log_execution
    def run_job(self):
        print(f"\nRunning job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for symbol in self.symbols:
            print(f"Updating {symbol}...")
            csv_path = self.get_csv_path(symbol)
            last_date = self.get_last_date_from_csv(csv_path)
            start_date = datetime(2000, 1, 1) if last_date is None else datetime.combine(last_date, datetime.min.time())
            data = self.fetch_data(symbol, start_date)
            self.append_to_csv(data, csv_path)

    def start_scheduler(self):
        schedule.every().day.at(self.run_time).do(self.run_job)
        print(f"Scheduler started. Will run daily at {self.run_time}.")
        while True:
            schedule.run_pending()
            time.sleep(60)


# 🧾 List of Symbols
symbols = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL",
    "JIOFIN", "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO",
    "WIPRO"
]

# 🛠️ Start Scheduler
if __name__ == "__main__":
    folder_path = r"D:\company"
    scheduler = DailyDataPullScheduler(symbols=symbols, folder=folder_path, run_time="07:21")
    scheduler.start_scheduler()
