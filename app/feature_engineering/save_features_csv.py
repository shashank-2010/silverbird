import pandas as pd
import os
import pandas_ta as ta

class StoreFeatureEngineering:
    def __init__(self,symbol):
        self.symbol = symbol;
        self.base_path = "D:/company"
        self.enriched_path = "D:/enriched_company_csv/"
        self.raw_file = os.path.join(self.base_path, f"{self.symbol}_OHLC_10_years_daily.csv")
        self.enriched_file = os.path.join(self.enriched_path, f"{self.symbol}_OHLC_enriched.csv")

    def _generate_enriched_csv_with_features(self) -> str:
        if not os.path.exists(self.raw_file):
            raise FileNotFoundError(f"Raw OHLCV file not found for symbol: {self.symbol}")

        df = pd.read_csv(self.raw_file)
        df.columns = df.columns.str.lower()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # === Start Feature Engineering ===
        df['rsi_14'] = df.ta.rsi(length=14)
        df['ema_13'] = df.ta.ema(length=13)
        df['ema_21'] = df.ta.ema(length=21)
        df['ema_diff'] = df['ema_13'] - df['ema_21']
        df['price_above_ema_21'] = (df['close'] > df['ema_21']).astype(int)
        df['roc_5'] = df.ta.roc(length=5)
        df['roc_10'] = df.ta.roc(length=10)
        df['stoch_d'] = df.ta.stoch()['STOCHd_14_3_3']
        df['consecutive_up'] = (df['close'] > df['close'].shift(1)).rolling(3).sum()

        macd = df.ta.macd()
        df['macd_hist'] = macd['MACDh_12_26_9']
        df['macd_cross'] = macd['MACD_12_26_9'] - macd['MACDs_12_26_9']

        df['atr_14'] = df.ta.atr(length=14)
        df['natr'] = 100 * df['atr_14'] / df['close']
        df['rolling_std_5'] = df['close'].rolling(5).std()
        df['rolling_std_20'] = df['close'].rolling(20).std()

        bb = df.ta.bbands(length=20, std=2)
        df['bb_percent_b'] = bb['BBP_20_2.0']
        df['adx_14'] = df.ta.adx(length=14)['ADX_14']
        df['cmf'] = df.ta.cmf()

        df = df.set_index('date')
        df['vwap'] = df.ta.vwap()
        df = df.reset_index()

        df['candle_body'] = (df['close'] - df['open']).abs()
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        df['3d_return'] = df['close'].pct_change(3)
        df['intraday_reversal'] = (df['close'] < df['open']).astype(int)

        df = df.dropna()
        df.to_csv(self.enriched_file, index=False)
        return self.enriched_file
