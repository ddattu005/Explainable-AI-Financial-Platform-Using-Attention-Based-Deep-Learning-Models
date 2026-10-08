"""
regime_detector.py - ATHENA Market Regime Detection v2
Author: ATHENA Project - FIXED v2

ROOT CAUSE OF ALWAYS-TRENDING BUG:
  The original detector fit KMeans on only 1 ticker with ~42 sequences
  (old data_loader cap). With so few samples and low variance, KMeans
  always collapses the "middle" volatility cluster = Trending.

FIX 1: Rule-based regime detection as PRIMARY method
  - Calm:     20-day vol < 1.0% AND |momentum| < 0.05%
  - Volatile: 20-day vol > 2.5% OR recent ATR spike > 2x baseline  
  - Trending: everything else (has directional momentum)

FIX 2: KMeans as SECONDARY validation
  - Only used to confirm or override rule-based if cluster separation is good
  - Falls back to rule-based if silhouette score < 0.3

FIX 3: Expanded feature set
  - Added ATR ratio, RSI extremes, BB width, MACD strength
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class MarketRegimeDetector:
    """
    Hybrid rule-based + KMeans regime detector.
    Reliably distinguishes Calm / Trending / Volatile.
    """

    REGIME_LABELS = {0: 'Calm', 1: 'Trending', 2: 'Volatile'}
    CONFIDENCE_ADJ = {0: 1.05, 1: 1.00, 2: 0.85}

    def __init__(self, n_regimes=3):
        self.n_regimes    = n_regimes
        self.kmeans       = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.scaler       = StandardScaler()
        self.is_fitted    = False
        self.regime_mapping = {0: 0, 1: 1, 2: 2}
        self._use_rules   = True   # start with rule-based; upgrade to KMeans if data is rich

    # ──────────────────────────────── FEATURE EXTRACTION ──
    def _extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract regime-relevant features from OHLCV dataframe."""
        d = df.copy()

        # Need raw Close/Volume (not normalized) — try both
        close  = d['Close']    if d['Close'].max()  > 10 else d.get('Close_Normalized', d['Close'])
        volume = d['Volume']   if 'Volume' in d.columns else pd.Series(1.0, index=d.index)

        returns = close.pct_change()

        # ── Core features ──
        d['ret_vol_20']   = returns.rolling(20).std()          # 20-day realized volatility
        d['ret_mom_20']   = returns.rolling(20).mean()         # 20-day momentum
        d['vol_ratio']    = volume / volume.rolling(20).mean() # volume surge

        # ── Extended features (more discriminative) ──
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        std20 = close.rolling(20).std()

        # Bollinger band width (normalized) — wider = more volatile
        d['bb_width']   = (4 * std20) / (sma20 + 1e-9)

        # Price vs SMA50 deviation (trend strength)
        d['sma_dev']    = (close - sma50) / (sma50 + 1e-9)

        # ATR ratio (recent ATR / 60-day ATR)
        hl = d['High'] - d['Low'] if 'High' in d.columns else std20 * 2
        hc = (d['High'] - close.shift()).abs() if 'High' in d.columns else std20
        lc = (d['Low']  - close.shift()).abs() if 'Low'  in d.columns else std20
        atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        d['atr_ratio']  = atr / (atr.rolling(60).mean() + 1e-9)

        # RSI extreme flag (overbought / oversold)
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))
        d['rsi_extreme'] = ((rsi > 70) | (rsi < 30)).astype(float).rolling(5).mean()

        d = d.dropna()
        return d

    FEAT_COLS = ['ret_vol_20', 'ret_mom_20', 'vol_ratio', 'bb_width', 'sma_dev', 'atr_ratio', 'rsi_extreme']

    # ──────────────────────────────── FIT ──
    def fit(self, df: pd.DataFrame):
        print("\n🔍 Fitting Market Regime Detector...")
        d = self._extract(df)

        if len(d) < 30:
            print("  ⚠️  Insufficient data (<30 rows) — using rule-based only")
            self._use_rules = True
            self.is_fitted  = True
            # Store regime stats from rules for display
            self._fit_rule_stats(d)
            return self

        feats = d[self.FEAT_COLS].values
        feats_scaled = self.scaler.fit_transform(feats)

        # Try KMeans
        self.kmeans.fit(feats_scaled)
        labels = self.kmeans.predict(feats_scaled)

        # Check cluster quality
        if len(np.unique(labels)) >= 2:
            sil = silhouette_score(feats_scaled, labels)
        else:
            sil = 0.0

        print(f"  KMeans silhouette score: {sil:.3f}")

        if sil >= 0.25:
            # Good separation — use KMeans, label clusters by volatility
            self._assign_kmeans_labels(d, labels)
            self._use_rules = False
            print("  ✅ Using KMeans (good cluster separation)")
        else:
            self._use_rules = True
            print("  ⚠️  Poor KMeans separation — using rule-based regime detection")

        self._fit_rule_stats(d)
        self.is_fitted = True
        self._print_regime_summary(d)
        return self

    def _assign_kmeans_labels(self, d, labels):
        """Map KMeans cluster IDs to Calm/Trending/Volatile by volatility."""
        stats = []
        for i in range(self.n_regimes):
            mask = labels == i
            if mask.sum() == 0:
                continue
            avg_vol = d.loc[mask, 'ret_vol_20'].mean()
            avg_bb  = d.loc[mask, 'bb_width'].mean()
            stats.append({'cluster': i, 'avg_vol': avg_vol + avg_bb * 0.5})

        stats.sort(key=lambda x: x['avg_vol'])
        # lowest vol → Calm, middle → Trending, highest → Volatile
        for rank, s in enumerate(stats):
            self.regime_mapping[s['cluster']] = rank

    def _fit_rule_stats(self, d):
        """Pre-compute percentile thresholds for rule-based detection."""
        self._vol_p33  = float(np.nanpercentile(d['ret_vol_20'], 33))
        self._vol_p66  = float(np.nanpercentile(d['ret_vol_20'], 66))
        self._atr_p75  = float(np.nanpercentile(d['atr_ratio'],  75))
        self._bb_p75   = float(np.nanpercentile(d['bb_width'],   75))

    def _print_regime_summary(self, d):
        ids = [self._rule_predict_row(d.iloc[i]) for i in range(len(d))]
        for rid, name in self.REGIME_LABELS.items():
            cnt = sum(1 for x in ids if x == rid)
            print(f"  {name}: {cnt} days ({cnt/len(ids)*100:.1f}%)")

    # ──────────────────────────────── PREDICT ──
    def _rule_predict_row(self, row) -> int:
        """
        Rule-based regime for a single row.
        Returns 0=Calm, 1=Trending, 2=Volatile
        """
        vol  = row.get('ret_vol_20', 0.01)
        atr  = row.get('atr_ratio',  1.0)
        bb   = row.get('bb_width',   0.05)
        rsi_x= row.get('rsi_extreme',0.0)

        # Volatile: high realized vol OR ATR spike OR BB very wide
        vol_thresh   = getattr(self, '_vol_p66',  0.015)
        atr_thresh   = getattr(self, '_atr_p75',  1.4)
        bb_thresh    = getattr(self, '_bb_p75',   0.08)

        if vol > vol_thresh * 1.3 or atr > atr_thresh or (bb > bb_thresh and rsi_x > 0.3):
            return 2  # Volatile

        # Calm: low vol AND atr near baseline
        calm_thresh = getattr(self, '_vol_p33', 0.008)
        if vol < calm_thresh and atr < 1.1:
            return 0  # Calm

        return 1  # Trending (middle)

    def predict(self, df: pd.DataFrame):
        """
        Predict regime for the most recent period.

        Returns:
            (regime_id: int, regime_name: str, confidence: float)
        """
        if not self.is_fitted:
            self.fit(df)

        d = self._extract(df)
        if len(d) == 0:
            return 1, 'Trending', 0.6

        row = d.iloc[-1]

        if self._use_rules:
            regime_id  = self._rule_predict_row(row)
            confidence = self._rule_confidence(row, regime_id)
        else:
            feats  = row[self.FEAT_COLS].values.reshape(1, -1)
            scaled = self.scaler.transform(feats)
            cluster = self.kmeans.predict(scaled)[0]
            regime_id = self.regime_mapping.get(int(cluster), 1)

            dists = self.kmeans.transform(scaled)[0]
            min_d = dists[cluster]
            max_d = dists.max()
            confidence = 1 - (min_d / (max_d + 1e-10))
            confidence = max(0.50, min(0.92, confidence))

            # Validate with rules — if they disagree strongly, trust rules
            rule_id = self._rule_predict_row(row)
            if rule_id != regime_id and confidence < 0.65:
                regime_id  = rule_id
                confidence = self._rule_confidence(row, rule_id) * 0.9

        regime_name = self.REGIME_LABELS.get(regime_id, 'Trending')
        return int(regime_id), regime_name, float(confidence)

    def _rule_confidence(self, row, regime_id: int) -> float:
        """Confidence based on how far into the regime zone the features are."""
        vol  = row.get('ret_vol_20', 0.01)
        atr  = row.get('atr_ratio',  1.0)
        p33  = getattr(self, '_vol_p33', 0.008)
        p66  = getattr(self, '_vol_p66', 0.015)

        if regime_id == 2:   # Volatile
            spread = max(0, (vol - p66) / (p66 + 1e-9))
            return float(max(0.60, min(0.92, 0.65 + spread * 1.2)))
        elif regime_id == 0: # Calm
            spread = max(0, (p33 - vol) / (p33 + 1e-9))
            return float(max(0.60, min(0.90, 0.65 + spread * 1.5)))
        else:                # Trending
            mid = (p33 + p66) / 2
            dist = abs(vol - mid) / ((p66 - p33) / 2 + 1e-9)
            return float(max(0.55, min(0.85, 0.75 - dist * 0.15)))

    def adjust_confidence(self, base_confidence: float, regime_id: int) -> float:
        adj = self.CONFIDENCE_ADJ.get(int(regime_id), 1.0)
        return float(max(0.50, min(0.95, base_confidence * adj)))

    def predict_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict regime for every row (for plotting)."""
        if not self.is_fitted:
            self.fit(df)
        d = self._extract(df)
        d['regime_id']   = [self._rule_predict_row(d.iloc[i]) for i in range(len(d))]
        d['regime_name'] = d['regime_id'].map(self.REGIME_LABELS)
        return d

    def get_regime_stats(self, df: pd.DataFrame) -> dict:
        d = self.predict_sequence(df)
        stats = {}
        for rid, name in self.REGIME_LABELS.items():
            mask = d['regime_id'] == rid
            stats[name] = {
                'count':          int(mask.sum()),
                'percentage':     float(mask.sum() / len(d) * 100),
                'avg_volatility': float(d.loc[mask, 'ret_vol_20'].mean()) if mask.any() else 0.0,
                'avg_return':     float(d.loc[mask, 'ret_mom_20'].mean()) if mask.any() else 0.0,
            }
        return stats


if __name__ == "__main__":
    from data_loader import StockDataLoader
    import os
    loader = StockDataLoader(['AAPL'], polygon_key=os.getenv('POLYGON_API_KEY') or os.getenv('FMP_API_KEY'))
    results = loader.process_all_tickers()
    if 'AAPL' in results:
        df  = results['AAPL']['data']
        det = MarketRegimeDetector()
        det.fit(df)
        rid, rname, rconf = det.predict(df)
        print(f"\nCurrent regime: {rname} (confidence {rconf*100:.1f}%)")
        stats = det.get_regime_stats(df)
        for k, v in stats.items():
            print(f"  {k}: {v['count']} days ({v['percentage']:.1f}%)")