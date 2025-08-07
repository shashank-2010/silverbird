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


import pandas as pd
import pandas_ta as ta


class FeatureEngineering:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.db = None

    def load_and_prepare(self):
        self.df = pd.read_csv(self.csv_path)
        #self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['date'] = pd.to_datetime(self.df['date'], format='mixed', dayfirst=True, errors='coerce')
        self.df = self.df.sort_values(by = ['date']).reset_index(drop=True)
        self._add_features()
        self.df.dropna(inplace=True)
        return self.df
    
    # def _add_features(self):
    #     df = self.df

    #     df['ma_5'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    #     #df['ma_10'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(10).mean())
    #     df['price_vs_ma_5'] = df['close'] / df['ma_5']
    #     #df['price_vs_ma_10'] = df['close'] / df['ma_10']

    #     df['rsi_14'] = df.groupby('symbol')['close'].transform(lambda x: RSIIndicator(x, window=14).rsi())
    #     #df['rsi_3'] = df.groupby('symbol')['close'].transform(lambda x: RSIIndicator(x, window=3).rsi())

    #     #df['macd_hist'] = df.groupby('symbol')['close'].transform(lambda x: MACD(x).macd_diff())
    #     #df['macd_hist_slope'] = df.groupby('symbol')['macd_hist'].transform(lambda x: x.diff().rolling(3).mean())

    #     df['bb_percent_b'] = df.groupby('symbol')['close'].transform(lambda x: BollingerBands(x, window=20, window_dev=2).bollinger_pband())

    #     df['atr_5'] = AverageTrueRange(df['high'], df['low'], df['close'], window=5).average_true_range()
    #     #df['atr_5_norm'] = df['atr_5'] / df['close']

    #     df['prev_close'] = df.groupby('symbol')['close'].shift(1)
    #     df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']

    #     df['volume_10_avg'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(10).mean())
    #     #df['volume_vs_avg'] = df['volume'] / df['volume_10_avg']

    #     df['intraday_reversal'] = (df['close'] < df['open']).astype(int)

    #     typical_price = (df['high'] + df['low'] + df['close']) / 3
    #     df['vwap'] = (typical_price * df['volume']) / df['volume']
    #     #df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

    #    # Exponential Moving Average
    #     #df['ema_21'] = EMAIndicator(close=df['close'], window=5).ema_indicator()

    #     # # ADX (Trend Strength)
    #     # df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=7).adx()

    #     # # CCI
    #     # #df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=7).cci()

    #     # # Rate of Change
    #     # df['roc_5'] = ROCIndicator(close=df['close'], window=5).roc()
    #     # df['roc_10'] = ROCIndicator(close=df['close'], window=10).roc()

    #     # # Stochastic Oscillator
    #     # stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14)
    #     # #df['stoch_k'] = stoch.stoch()
    #     # df['stoch_d'] = stoch.stoch_signal()

    #     # # Awesome Oscillator
    #     # df['ao'] = AwesomeOscillatorIndicator(high=df['high'], low=df['low']).awesome_oscillator()

    #     # # Williams %R
    #     # #df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r()

    #     # # On-Balance Volume
    #     # #df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

    #     # # Chaikin Money Flow (CMF)
    #     # df['cmf'] = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).chaikin_money_flow()

    #     # # Accumulation/Distribution Index
    #     # df['acc_dist'] = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()

    #     # # Donchian Channel Width
    #     # # donchian = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=20)
    #     # # df['donchian_upper'] = donchian.donchian_channel_hband()
    #     # # df['donchian_lower'] = donchian.donchian_channel_lband()
    #     # # df['donchian_width'] = df['donchian_upper'] - df['donchian_lower']

    #     # # Normalized ATR (manually, since ta doesn't have NormalizedATR)
    #     # atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    #     # df['natr'] = 100 * atr / df['close']        
        
    #     df.drop(columns=['prev_close','open','high','low','ma_5'], inplace=True)
    #     df.dropna(inplace=True)

    #     self.df = df

    def _add_features(self):
        df = self.df
        # RSI
        df['rsi_14'] = df.ta.rsi(length=14)

        # ========== Trend Features ==========
        df['ema_13'] = df.ta.ema(close='close', length=13)
        df['ema_21'] = df.ta.ema(close='close', length=21)

        df['ema_diff'] = df['ema_13'] - df['ema_21']
        df['price_above_ema_21'] = (df['close'] > df['ema_21']).astype(int)

        # ========== Momentum Features ==========
        df['roc_5'] = df.ta.roc(close = 'close', length = 5)
        df['roc_10'] = df.ta.roc(close = 'close', length = 10)
        df['stoch_d'] = df.ta.stoch()['STOCHd_14_3_3']
        df['consecutive_up'] = (df['close'] > df['close'].shift(1)).rolling(3).sum()

        # MACD Histogram and Cross
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['macd_hist'] = macd['MACDh_12_26_9']
        df['macd_cross'] = macd['MACD_12_26_9'] - macd['MACDs_12_26_9']

        # ATR and Normalized ATR
        df['atr_14'] = df.ta.atr(length=14)
        df['natr'] = 100 * df['atr_14'] / df['close']

        df['rolling_std_5'] = df['close'].rolling(5).std()
        df['rolling_std_20'] = df['close'].rolling(20).std()

        # Bollinger Bands
        bb = df.ta.bbands(length=20, std=2)
        #print(bb.columns)
        #df['bb_width'] = bb['BBU_20_2.0'] - bb['BBL_20_2.0']
        df['bb_percent_b'] = bb['BBP_20_2.0']

        # ADX (trend strength)
        df['adx_14'] = df.ta.adx(length=14)['ADX_14']

        # CCI
        #df['cci_14'] = df.ta.cci(length=14)

        # Stochastic Oscillator (%D line)
        df['stoch_d'] = df.ta.stoch()['STOCHd_14_3_3']

        # OBV (On-Balance Volume)
        #df['obv'] = df.ta.obv()

        # CMF (Chaikin Money Flow)
        df['cmf'] = df.ta.cmf()

        # VWAP
        df = df.set_index('date')
        df['vwap'] = df.ta.vwap(high='high', low='low', close='close', volume='volume')
        df = df.reset_index()

        # Candle body & wick metrics
        df['candle_body'] = (df['close'] - df['open']).abs()
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']

        # Gap %
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']

        # 3-day return
        df['3d_return'] = df['close'].pct_change(3)

        # Intraday reversal (close < open)
        df['intraday_reversal'] = (df['close'] < df['open']).astype(int)

        # === 🧹 Clean Up === #
        df.drop(columns=['prev_close', 'open', 'high', 'low','atr_14','ema_13','ema_21','bb_percent_b'], inplace=True)
        df.dropna(inplace=True)

        self.df = df
        

