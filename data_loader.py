"""
data_loader.py - ATHENA Data Pipeline v3
Author: ATHENA Project - FMP Edition

CHANGES FROM v2:
  1. Switched from Polygon.io to FMP (Financial Modeling Prep) API
     FMP Basic: 250 calls/day, EOD historical, company profiles
  2. Supports 50 large-cap stocks across 10 sectors
  3. Removed max_sequences=42 cap — uses ALL available sequences
  4. Returns y_dir_train / y_dir_val / y_dir_test for multi-task training
  5. More robust indicator calculation (no NaN bleed)

FMP API Key: 9MDQnnX5wuBrZEYlbbPzLFdQLgwHmIp1
"""

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings, os, time
warnings.filterwarnings('ignore')


# ─── 50-STOCK UNIVERSE (10 sectors) ───────────────────────
STOCK_UNIVERSE = [
    # Technology (10)
    {"ticker": "AAPL",  "name": "Apple Inc.",             "sector": "Technology"},
    {"ticker": "MSFT",  "name": "Microsoft Corporation",  "sector": "Technology"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.",          "sector": "Technology"},
    {"ticker": "META",  "name": "Meta Platforms Inc.",    "sector": "Technology"},
    {"ticker": "NVDA",  "name": "NVIDIA Corporation",     "sector": "Semiconductors"},
    {"ticker": "AMD",   "name": "Advanced Micro Devices", "sector": "Semiconductors"},
    {"ticker": "INTC",  "name": "Intel Corporation",      "sector": "Semiconductors"},
    {"ticker": "CRM",   "name": "Salesforce Inc.",        "sector": "Technology"},
    {"ticker": "ADBE",  "name": "Adobe Inc.",             "sector": "Technology"},
    {"ticker": "ORCL",  "name": "Oracle Corporation",     "sector": "Technology"},
    # Consumer / E-Commerce (6)
    {"ticker": "AMZN",  "name": "Amazon.com Inc.",        "sector": "Consumer"},
    {"ticker": "TSLA",  "name": "Tesla Inc.",             "sector": "Automotive"},
    {"ticker": "NFLX",  "name": "Netflix Inc.",           "sector": "Media"},
    {"ticker": "COST",  "name": "Costco Wholesale",       "sector": "Consumer"},
    {"ticker": "NKE",   "name": "Nike Inc.",              "sector": "Consumer"},
    {"ticker": "SBUX",  "name": "Starbucks Corp.",        "sector": "Consumer"},
    # Finance (7)
    {"ticker": "JPM",   "name": "JPMorgan Chase",         "sector": "Finance"},
    {"ticker": "GS",    "name": "Goldman Sachs",          "sector": "Finance"},
    {"ticker": "MS",    "name": "Morgan Stanley",         "sector": "Finance"},
    {"ticker": "BAC",   "name": "Bank of America",        "sector": "Finance"},
    {"ticker": "BLK",   "name": "BlackRock Inc.",         "sector": "Finance"},
    {"ticker": "V",     "name": "Visa Inc.",              "sector": "Finance"},
    {"ticker": "MA",    "name": "Mastercard Inc.",        "sector": "Finance"},
    # Healthcare (5)
    {"ticker": "JNJ",   "name": "Johnson & Johnson",      "sector": "Healthcare"},
    {"ticker": "UNH",   "name": "UnitedHealth Group",     "sector": "Healthcare"},
    {"ticker": "PFE",   "name": "Pfizer Inc.",            "sector": "Healthcare"},
    {"ticker": "ABBV",  "name": "AbbVie Inc.",            "sector": "Healthcare"},
    {"ticker": "MRK",   "name": "Merck & Co.",            "sector": "Healthcare"},
    # Energy (4)
    {"ticker": "XOM",   "name": "Exxon Mobil Corp.",      "sector": "Energy"},
    {"ticker": "CVX",   "name": "Chevron Corporation",    "sector": "Energy"},
    {"ticker": "COP",   "name": "ConocoPhillips",         "sector": "Energy"},
    {"ticker": "SLB",   "name": "SLB (Schlumberger)",     "sector": "Energy"},
    # Industrial (4)
    {"ticker": "BA",    "name": "Boeing Company",         "sector": "Industrial"},
    {"ticker": "CAT",   "name": "Caterpillar Inc.",       "sector": "Industrial"},
    {"ticker": "HON",   "name": "Honeywell Intl.",        "sector": "Industrial"},
    {"ticker": "GE",    "name": "GE Aerospace",           "sector": "Industrial"},
    # Telecom (3)
    {"ticker": "T",     "name": "AT&T Inc.",              "sector": "Telecom"},
    {"ticker": "VZ",    "name": "Verizon Comm.",          "sector": "Telecom"},
    {"ticker": "TMUS",  "name": "T-Mobile US",            "sector": "Telecom"},
    # Real Estate / Utilities (3)
    {"ticker": "NEE",   "name": "NextEra Energy",         "sector": "Utilities"},
    {"ticker": "AMT",   "name": "American Tower Corp.",   "sector": "Real Estate"},
    {"ticker": "PLD",   "name": "Prologis Inc.",          "sector": "Real Estate"},
    # ETFs / Indices (4)
    {"ticker": "SPY",   "name": "SPDR S&P 500 ETF",      "sector": "ETF"},
    {"ticker": "QQQ",   "name": "Invesco QQQ ETF",        "sector": "ETF"},
    {"ticker": "DIA",   "name": "SPDR Dow Jones ETF",     "sector": "ETF"},
    {"ticker": "IWM",   "name": "iShares Russell 2000",   "sector": "ETF"},
    # Additional large caps (4)
    {"ticker": "WMT",   "name": "Walmart Inc.",           "sector": "Consumer"},
    {"ticker": "PG",    "name": "Procter & Gamble",       "sector": "Consumer"},
    {"ticker": "KO",    "name": "Coca-Cola Company",      "sector": "Consumer"},
    {"ticker": "DIS",   "name": "The Walt Disney Co.",    "sector": "Media"},
]

TICKER_MAP = {s['ticker']: s for s in STOCK_UNIVERSE}

# Technical indicator feature columns (15 features)
FEATURE_COLS = [
    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
    'ATR_14', 'Volume_SMA', 'Daily_Return', 'Volatility', 'High_Low_Pct'
]

FMP_BASE = "https://financialmodelingprep.com/stable"


class StockDataLoader:
    """
    Downloads EOD data from FMP, computes 15 technical indicators,
    creates 60-day sequences for LSTM training.
    """

    FEATURE_COLS = FEATURE_COLS

    def __init__(self, tickers, polygon_key=None, fmp_key=None,
                 seq_length=60, train_ratio=0.7, val_ratio=0.15):
        self.tickers     = [t.upper() for t in tickers]
        self.fmp_key     = fmp_key or polygon_key or os.getenv('FMP_API_KEY') or '9MDQnnX5wuBrZEYlbbPzLFdQLgwHmIp1'
        self.seq_len     = seq_length
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio

        end   = datetime.now()
        start = end - timedelta(days=1460)   # ~4 years (FMP supports up to 5yr)
        self.start_date = start.strftime('%Y-%m-%d')
        self.end_date   = end.strftime('%Y-%m-%d')

        print(f"✅ FMP API configured")
        print(f"📅 {self.start_date} → {self.end_date}")

    # ── FMP DOWNLOAD ───────────────────────────────────────
    def _download_fmp(self, ticker: str) -> pd.DataFrame:
        """Download OHLCV from FMP stable historical-price-eod endpoint."""
        url = (f"{FMP_BASE}/historical-price-eod/full"
               f"?symbol={ticker}&from={self.start_date}&to={self.end_date}&apikey={self.fmp_key}")
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ❌ FMP request failed for {ticker}: {e}")
            return pd.DataFrame()

        if not data or not isinstance(data, list):
            print(f"  ❌ No data returned for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.set_index('date')
        return df

    # ── TECHNICAL INDICATORS ──────────────────────────────
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        c = d['Close']
        v = d['Volume']

        # Moving averages
        d['SMA_20']  = c.rolling(20).mean()
        d['SMA_50']  = c.rolling(50).mean()
        d['SMA_200'] = c.rolling(200).mean()
        d['EMA_12']  = c.ewm(span=12, adjust=False).mean()
        d['EMA_26']  = c.ewm(span=26, adjust=False).mean()

        # RSI
        delta = c.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        d['RSI_14'] = 100 - 100 / (1 + gain / (loss + 1e-9))

        # MACD
        d['MACD']        = d['EMA_12'] - d['EMA_26']
        d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma20   = d['SMA_20']
        std20   = c.rolling(20).std()
        d['BB_Upper'] = sma20 + 2 * std20
        d['BB_Lower'] = sma20 - 2 * std20

        # ATR
        hl  = d['High'] - d['Low']
        hpc = (d['High'] - c.shift()).abs()
        lpc = (d['Low']  - c.shift()).abs()
        d['ATR_14'] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

        # Volume-based
        d['Volume_SMA']  = v.rolling(20).mean()

        # Price-based
        d['Daily_Return'] = c.pct_change()
        d['Volatility']   = d['Daily_Return'].rolling(20).std()
        d['High_Low_Pct'] = (d['High'] - d['Low']) / (c + 1e-9)

        return d

    # ── SEQUENCES ─────────────────────────────────────────
    def create_sequences(self, df: pd.DataFrame):
        """
        Create (X, y_price, y_direction) from feature DataFrame.
        y_direction = 1 if next Close > current Close, else 0.
        """
        feat   = df[FEATURE_COLS].values.astype(np.float32)
        target = df['Close'].values.astype(np.float32)

        X, y_price, y_dir = [], [], []
        for i in range(self.seq_len, len(feat) - 1):
            X.append(feat[i - self.seq_len : i])
            y_price.append(target[i])
            y_dir.append(1.0 if target[i] > target[i - 1] else 0.0)

        return np.array(X), np.array(y_price), np.array(y_dir)

    # ── SPLIT ────────────────────────────────────────────
    def split_data(self, X, y_price, y_dir):
        n     = len(X)
        t_end = int(n * self.train_ratio)
        v_end = int(n * (self.train_ratio + self.val_ratio))
        return (
            X[:t_end],  X[t_end:v_end],  X[v_end:],
            y_price[:t_end], y_price[t_end:v_end], y_price[v_end:],
            y_dir[:t_end],   y_dir[t_end:v_end],   y_dir[v_end:],
        )

    # ── PROCESS ONE TICKER ────────────────────────────────
    def process_ticker(self, ticker: str) -> 'Optional[dict]':
        print(f"\n{'='*55}")
        print(f"Processing {ticker}")
        print(f"{'='*55}")

        # Download
        print(f"📊 Downloading {ticker} from FMP...")
        df_raw = self._download_fmp(ticker)
        if df_raw.empty:
            return None
        print(f"  ✅ {len(df_raw)} days from FMP")

        # Indicators
        df = self._add_indicators(df_raw)
        df = df.dropna()
        if len(df) < self.seq_len + 50:
            print(f"  ⚠️  Only {len(df)} rows after dropna — skipping")
            return None
        print(f"  ✅ {len(df)} rows after indicators")

        # Scale features
        feat_scaler   = MinMaxScaler()
        target_scaler = MinMaxScaler()

        scaled_feats  = feat_scaler.fit_transform(df[FEATURE_COLS])
        scaled_target = target_scaler.fit_transform(df[['Close']])

        df_scaled = df.copy()
        df_scaled[FEATURE_COLS] = scaled_feats
        df_scaled['Close']      = scaled_target.flatten()

        # Sequences
        X, y_price, y_dir = self.create_sequences(df_scaled)
        if len(X) < 50:
            print(f"  ⚠️  Only {len(X)} sequences — skipping")
            return None

        n_up = int(y_dir.sum())
        print(f"  ✅ {len(X)} sequences (no cap)")
        print(f"     UP days: {n_up} ({n_up/len(y_dir)*100:.1f}%) | DOWN: {len(y_dir)-n_up}")

        # Split
        (X_tr, X_val, X_te,
         yp_tr, yp_val, yp_te,
         yd_tr, yd_val, yd_te) = self.split_data(X, y_price, y_dir)

        # Save
        os.makedirs('data', exist_ok=True)
        df.to_csv(f'data/{ticker}_processed.csv')
        print(f"  💾 Saved data/{ticker}_processed.csv")

        return {
            'ticker':         ticker,
            'data':           df_raw,                   # raw OHLCV for regime + signals
            'data_scaled':    df_scaled,
            'X_train': X_tr,  'X_val': X_val,   'X_test': X_te,
            'y_price_train': yp_tr, 'y_price_val': yp_val, 'y_price_test': yp_te,
            'y_dir_train':   yd_tr, 'y_dir_val':   yd_val, 'y_dir_test':   yd_te,
            'feat_scaler':    feat_scaler,
            'target_scaler':  target_scaler,
            'n_features':     len(FEATURE_COLS),
            'seq_length':     self.seq_len,
        }

    # ── PROCESS ALL ───────────────────────────────────────
    def process_all_tickers(self) -> dict:
        print(f"\n🚀 ATHENA DATA PIPELINE v3 — FMP Edition")
        print(f"{'='*55}")
        results = {}
        for i, ticker in enumerate(self.tickers):
            result = self.process_ticker(ticker)
            if result:
                results[ticker] = result
            # Rate limiting: FMP basic is lenient but be safe
            if i < len(self.tickers) - 1:
                time.sleep(0.2)
        print(f"\n✅ {len(results)}/{len(self.tickers)} tickers processed")
        return results

    # ── PROFILE (for richer UI) ────────────────────────────
    def get_company_profile(self, ticker: str) -> dict:
        """Fetch sector, market cap, description etc. from FMP stable profile endpoint."""
        url = f"{FMP_BASE}/profile?symbol={ticker}&apikey={self.fmp_key}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if data and isinstance(data, list):
                p = data[0]
                return {
                    'ticker':      p.get('symbol', ticker),
                    'name':        p.get('companyName', ''),
                    'sector':      p.get('sector', ''),
                    'industry':    p.get('industry', ''),
                    'market_cap':  p.get('mktCap', 0),
                    'beta':        p.get('beta', 1.0),
                    'pe_ratio':    p.get('pe', None),
                    'description': p.get('description', '')[:300],
                    'price':       p.get('price', 0),
                    'change_pct':  p.get('changesPercentage', 0),
                }
        except Exception:
            pass
        return {'ticker': ticker, 'name': TICKER_MAP.get(ticker, {}).get('name', ticker)}


if __name__ == "__main__":
    loader  = StockDataLoader(['AAPL', 'MSFT', 'TSLA'])
    results = loader.process_all_tickers()
    for t, r in results.items():
        print(f"{t}: {len(r['X_train'])} train / {len(r['X_val'])} val / {len(r['X_test'])} test sequences")