"""
predict.py - ATHENA Prediction Pipeline v4
Author: ATHENA Project

Direction Ensemble (Selective Prediction with Confidence Thresholding):
  - "BULLISH"  : composite >= 0.53
  - "BEARISH"  : composite < 0.53
  - confidence_tier: "HIGH_CONF" if abs(composite - 0.5) >= 0.08 else "MODERATE"
"""

import numpy as np
import pandas as pd


class DirectionEnsemble:
    """
    Weighted ensemble of 4 direction signals.
    
    Weights (tuned to maximize directional accuracy):
      LSTM direction head  : 0.45
      RSI momentum         : 0.25
      MACD crossover       : 0.20
      MA trend (50/200)    : 0.10
    
    Confidence thresholds:
      abs(composite - 0.5) >= 0.08 → HIGH_CONF
      else → MODERATE
    """

    WEIGHTS = {
        'lstm_direction': 0.45,
        'rsi':            0.25,
        'macd_cross':     0.20,
        'ma_crossover':   0.10,
    }

    def compute(self, lstm_dir_prob: float, df: pd.DataFrame) -> dict:
        """
        Compute ensemble direction.

        Args:
            lstm_dir_prob : P(UP) from LSTM direction head [0,1]
            df            : Raw OHLCV DataFrame (must have Close, Volume)

        Returns:
            dict with keys: direction, probability, confidence, signals,
                            is_high_confidence
        """
        signals = {}

        # ── Signal 1: LSTM direction head ──
        signals['lstm_direction'] = {
            'value':  float(np.clip(lstm_dir_prob, 0, 1)),
            'weight': self.WEIGHTS['lstm_direction'],
            'raw':    float(lstm_dir_prob),
        }

        # ── Signal 2: RSI (14-period) ──
        rsi_val = self._compute_rsi(df['Close'])
        if rsi_val < 30:
            rsi_signal = 0.75
        elif rsi_val > 70:
            rsi_signal = 0.25
        elif rsi_val < 45:
            rsi_signal = 0.55 + (45 - rsi_val) / 45 * 0.15
        elif rsi_val > 55:
            rsi_signal = 0.45 - (rsi_val - 55) / 45 * 0.15
        else:
            rsi_signal = 0.50
        signals['rsi'] = {
            'value':   float(np.clip(rsi_signal, 0.1, 0.9)),
            'weight':  self.WEIGHTS['rsi'],
            'raw_rsi': float(rsi_val),
        }

        # ── Signal 3: MACD crossover ──
        macd, signal_line = self._compute_macd(df['Close'])
        macd_diff = macd - signal_line
        max_diff = float(df['Close'].iloc[-1]) * 0.005
        macd_signal = 0.5 + np.clip(macd_diff / (max_diff + 1e-9), -1, 1) * 0.3
        signals['macd_cross'] = {
            'value':      float(np.clip(macd_signal, 0.1, 0.9)),
            'weight':     self.WEIGHTS['macd_cross'],
            'raw_macd':   float(macd),
            'raw_signal': float(signal_line),
        }

        # ── Signal 4: MA trend (SMA50 vs SMA200) ──
        sma50  = df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        ma_gap_pct = (sma50 - sma200) / (sma200 + 1e-9)
        ma_signal = 0.5 + np.clip(ma_gap_pct * 10, -0.3, 0.3)
        signals['ma_crossover'] = {
            'value':   float(np.clip(ma_signal, 0.1, 0.9)),
            'weight':  self.WEIGHTS['ma_crossover'],
            'sma50':   float(sma50),
            'sma200':  float(sma200),
        }

        # ── Weighted composite ──
        composite = sum(
            signals[k]['value'] * signals[k]['weight']
            for k in self.WEIGHTS
        )

        # ── Direction classification ──
        direction = "BULLISH" if composite >= 0.53 else "BEARISH"
        is_high = abs(composite - 0.5) >= 0.08

        return {
            'direction':         direction,
            'probability':       float(composite),
            'confidence':        float(composite if direction == "BULLISH" else 1 - composite),
            'is_high_confidence': is_high,
            'signals':           signals,
        }

    # ── TECHNICAL HELPERS ─────────────────────────────────
    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> float:
        delta = close.diff().dropna()
        gain  = delta.where(delta > 0, 0.0)
        loss  = (-delta.where(delta < 0, 0.0))
        avg_g = gain.rolling(period).mean().iloc[-1]
        avg_l = loss.rolling(period).mean().iloc[-1]
        if avg_l < 1e-9:
            return 100.0
        return float(100 - 100 / (1 + avg_g / avg_l))

    @staticmethod
    def _compute_macd(close: pd.Series):
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


if __name__ == "__main__":
    from data_loader import StockDataLoader
    import os
    loader  = StockDataLoader(['AAPL'], fmp_key='9MDQnnX5wuBrZEYlbbPzLFdQLgwHmIp1')
    results = loader.process_all_tickers()
    if 'AAPL' in results:
        ens    = DirectionEnsemble()
        result = ens.compute(0.62, results['AAPL']['data'])
        print(f"Direction:   {result['direction']}")
        print(f"Composite:   {result['probability']:.3f}")
        print(f"High conf:   {result['is_high_confidence']}")
        for k, v in result['signals'].items():
            print(f"  {k}: {v['value']:.3f} (×{v['weight']})")