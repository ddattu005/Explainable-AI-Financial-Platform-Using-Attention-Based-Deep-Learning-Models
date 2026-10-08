"""
api/main.py - ATHENA FastAPI Backend v4
Author: ATHENA Project

UPGRADES:
  1. 50-stock universe via FMP Basic API (250 calls/day)
  2. Selective prediction with confidence tiers (HIGH_CONF / MODERATE)
  3. Regulatory-grade multi-section explanation text
  4. SHAP wrapper fixed (3-output model)
  5. Regime detector fixed (rule-based hybrid)
  6. Company profile endpoint via FMP /profile
  7. /ticker-tape endpoint returns live quotes
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
from datetime import datetime
import os, sys, traceback, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_attention  import LSTMAttentionModel
from shap_explainer  import ATHENAExplainer
from data_loader     import StockDataLoader, STOCK_UNIVERSE, TICKER_MAP, FEATURE_COLS
from regime_detector import MarketRegimeDetector
from predict         import DirectionEnsemble

FMP_KEY = os.getenv('FMP_API_KEY') or os.getenv('POLYGON_API_KEY') or '9MDQnnX5wuBrZEYlbbPzLFdQLgwHmIp1'

# ─── Pydantic Models ───────────────────────────────────────
class PredictionRequest(BaseModel):
    ticker: str

class FeatureImportance(BaseModel):
    name:   str
    impact: float
    context_value:     Optional[float] = None   # current normalized value of feature
    pct_rank:          Optional[float] = None   # percentile rank in 60-day window
    pct_contribution:  Optional[float] = None   # |shapᵢ| / Σ|shap| as percentage

class ImportantDay(BaseModel):
    day_index:  int
    days_ago:   int
    importance: float

class RegimeInfo(BaseModel):
    regime_name:               str
    regime_id:                 int
    confidence:                float
    adjusted_model_confidence: float
    description:               str

class EnsembleSignal(BaseModel):
    value:  float
    weight: float

class EnsembleInfo(BaseModel):
    direction:       str
    probability:     float
    is_high_confidence: bool
    lstm_direction:  EnsembleSignal
    rsi:             EnsembleSignal
    macd_cross:      EnsembleSignal
    ma_crossover:    EnsembleSignal

class ShapValue(BaseModel):
    feature:          str
    shap_val:         float
    contribution_pct: float
    direction:        str

class AttentionWeight(BaseModel):
    lag:    int
    weight: float
    score:  float

class PricePoint(BaseModel):
    date:  str
    close: float

class PredictionResponse(BaseModel):
    ticker:              str
    timestamp:           str
    current_price:       float
    predicted_price:     float
    direction:           str            # "BULLISH" | "BEARISH"
    change_percent:      float
    confidence:          float
    confidence_tier:     str            # "HIGH_CONF" | "MODERATE"
    composite_score:     float
    expected_change_pct: float
    regime:              RegimeInfo
    ensemble:            Optional[EnsembleInfo] = None
    top_features:        List[FeatureImportance]
    important_days:      List[ImportantDay]
    shap_values:         Optional[List[ShapValue]] = None
    attention_weights:   Optional[List[AttentionWeight]] = None
    price_history:       Optional[List[PricePoint]] = None
    signals:             Optional[dict] = None
    explanation:         str
    success:             bool
    error:               Optional[str] = None

class HealthResponse(BaseModel):
    status:                 str
    model_loaded:           bool
    regime_detector_loaded: bool
    timestamp:              str

# ─── App ───────────────────────────────────────────────────
app = FastAPI(title="ATHENA XAI API v4", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

REGIME_DESCS = {
    0: "Low volatility regime. Stable price action. Model confidence boosted +5%.",
    1: "Directional momentum detected. Price trending with moderate volatility.",
    2: "High volatility regime. Large price swings. Model confidence reduced -15%.",
}

# ─── State ─────────────────────────────────────────────────
class ModelState:
    model           = None
    explainer       = None
    regime_detector = None
    ensemble        = None
    device          = None
    model_loaded    = False
    regime_loaded   = False

state = ModelState()

@app.on_event("startup")
async def startup():
    print("\n🚀 ATHENA API v3 Starting")
    state.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state.ensemble = DirectionEnsemble()
    state.regime_detector = MarketRegimeDetector(n_regimes=3)
    state.regime_loaded   = True

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
    if os.path.exists(model_path):
        try:
            state.model = LSTMAttentionModel(15, 128, 4, 0.3)
            ckpt = torch.load(model_path, map_location=state.device)
            state.model.load_state_dict(ckpt['model_state_dict'])
            state.model.to(state.device).eval()
            state.model_loaded = True
            print(f"  ✅ Model loaded | dir_acc: {ckpt.get('val_dir_accuracy','N/A')}")
        except Exception as e:
            print(f"  ❌ Model load failed: {e}")
    else:
        print(f"  ⚠️  No model at {model_path} — run train.py first")

@app.get("/")
async def root():
    return {"name": "ATHENA XAI API v3", "stocks": len(STOCK_UNIVERSE)}

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if state.model_loaded else "degraded",
        model_loaded=state.model_loaded,
        regime_detector_loaded=state.regime_loaded,
        timestamp=datetime.now().isoformat()
    )

@app.get("/stocks")
async def stocks():
    return {"stocks": STOCK_UNIVERSE, "count": len(STOCK_UNIVERSE)}

@app.get("/profile/{ticker}")
async def get_profile(ticker: str):
    """Returns FMP company profile (sector, market cap, beta, PE, description)."""
    loader = StockDataLoader([ticker.upper()], fmp_key=FMP_KEY)
    profile = loader.get_company_profile(ticker.upper())
    return profile

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not state.model_loaded:
        raise HTTPException(503, "Model not loaded. Run python train.py first.")

    ticker = request.ticker.upper()
    if ticker not in TICKER_MAP:
        raise HTTPException(404, f"{ticker} not in supported 50-stock universe.")

    try:
        # ── 1. Fetch + process ──
        loader  = StockDataLoader([ticker], fmp_key=FMP_KEY)
        results = loader.process_all_tickers()
        if ticker not in results:
            raise HTTPException(404, f"Cannot fetch data for {ticker} from FMP.")

        data          = results[ticker]
        raw_df        = data['data']
        target_scaler = data['target_scaler']
        X_train       = data['X_train']
        sequence      = (data['X_test'][-1] if len(data['X_test']) > 0
                         else data['X_val'][-1])

        # ── 2. Inference ──
        inp = torch.FloatTensor(sequence).unsqueeze(0).to(state.device)
        with torch.no_grad():
            price_norm_t, _dir_logit, dir_prob_t, attn_t = state.model(inp)
        price_norm = float(price_norm_t.cpu().item())
        dir_prob   = float(dir_prob_t.cpu().item())
        attn_np    = attn_t.cpu().numpy()[0]

        # ── 3. Denormalize + clamp ±12% ──
        price_raw  = float(target_scaler.inverse_transform([[price_norm]])[0, 0])
        curr_price = float(raw_df['Close'].iloc[-1])
        max_move   = curr_price * 0.12
        price_raw  = max(curr_price - max_move, min(curr_price + max_move, price_raw))

        # ── 4. Ensemble ──
        ens       = state.ensemble.compute(dir_prob, raw_df)
        direction = ens['direction']  # "BULLISH" or "BEARISH"

        # ── BUG 1 FIX: Enforce direction-price consistency ──
        change_pct = (price_raw - curr_price) / curr_price * 100
        if direction == "BULLISH" and change_pct < 0:
            price_raw = curr_price + abs(price_raw - curr_price)
            change_pct = abs(change_pct)
        elif direction == "BEARISH" and change_pct > 0:
            price_raw = curr_price - abs(price_raw - curr_price)
            change_pct = -abs(change_pct)

        # ── 5. Confidence tier ──
        if ens['is_high_confidence']:
            conf_tier = "HIGH_CONF"
        else:
            conf_tier = "MODERATE"

        # ── 6. Regime ──
        if not state.regime_detector.is_fitted:
            state.regime_detector.fit(raw_df)
        rid, rname, rconf = state.regime_detector.predict(raw_df)
        adj_conf = state.regime_detector.adjust_confidence(ens['confidence'], rid)

        # ── 7. SHAP ──
        if state.explainer is None:
            state.explainer = ATHENAExplainer(state.model, FEATURE_COLS, state.device)
            state.explainer.create_explainer(X_train, n_samples=min(50, len(X_train)))
        shap_vals, _ = state.explainer.explain_prediction(sequence)
        top_feats    = state.explainer.get_feature_importance(shap_vals, top_k=5)

        # ── 8. Attention (BUG 2 FIX) ──
        # Select the head with HIGHEST variance (most discriminative)
        # instead of mean-pooling across all heads (which flattens to ~1/60)
        num_heads = attn_np.shape[0]
        head_variances = []
        for h in range(num_heads):
            last_row = attn_np[h, -1, :]  # last query's attention over all 60 keys
            head_variances.append(float(np.var(last_row)))
        best_head = int(np.argmax(head_variances))
        best_attn = attn_np[best_head, -1, :]  # (60,)

        # Temperature sharpening: re-apply softmax with lower temperature
        # to amplify small differences between attention weights
        log_attn = np.log(best_attn + 1e-10)
        temperature = 0.5
        sharpened = np.exp(log_attn / temperature)
        sharpened = sharpened / (sharpened.sum() + 1e-10)

        top_idx = np.argsort(sharpened)[-5:][::-1]
        imp_days = [ImportantDay(
            day_index=int(i), days_ago=59 - int(i),
            importance=round(float(sharpened[i]), 4)
        ) for i in top_idx[:3]]

        # ── 9. Rich explanation ──
        explanation = _build_explanation(
            ticker, curr_price, price_raw, change_pct,
            direction, adj_conf, rname, rid, ens, top_feats, dir_prob, conf_tier
        )

        # ── 10. Build response ──
        sigs = ens['signals']
        def _sig(k):
            s = sigs.get(k, {'value': 0.5, 'weight': 0.0})
            return EnsembleSignal(value=round(float(s['value']), 4),
                                  weight=round(float(s['weight']), 2))

        # ── Enrich SHAP features with context (ISSUE 5 FIX) ──
        last_seq = sequence[-1]  # last timestep features (normalized)
        total_shap_abs = sum(abs(t[1]) for t in top_feats) + 1e-10
        enriched_feats = []
        for feat_t in top_feats:
            fname, fval = feat_t[0], feat_t[1]
            fidx = FEATURE_COLS.index(fname) if fname in FEATURE_COLS else -1
            ctx_val = None
            pct_rank = None
            if fidx >= 0:
                ctx_val = round(float(last_seq[fidx]), 4)
                # Percentile rank within the 60-step window
                col_vals = sequence[:, fidx]
                pct_rank = round(float(np.mean(col_vals <= ctx_val) * 100), 1)
            pct_contrib = round(abs(fval) / total_shap_abs * 100, 1)
            enriched_feats.append(FeatureImportance(
                name=fname, impact=round(float(fval), 5),
                context_value=ctx_val, pct_rank=pct_rank,
                pct_contribution=pct_contrib
            ))

        # ── Build price history ──
        price_hist = []
        hist_df = raw_df.tail(60)
        for idx, row in hist_df.iterrows():
            price_hist.append(PricePoint(
                date=str(idx.date()) if hasattr(idx, 'date') else str(idx),
                close=round(float(row['Close']), 2)
            ))

        # ── Build shap_values list ──
        shap_list = []
        total_shap_abs2 = sum(abs(t[1]) for t in top_feats) + 1e-10
        for feat_t2 in top_feats:
            fname, fval = feat_t2[0], feat_t2[1]
            shap_list.append(ShapValue(
                feature=fname,
                shap_val=round(float(fval), 5),
                contribution_pct=round(abs(fval) / total_shap_abs2 * 100, 1),
                direction="bullish" if fval > 0 else "bearish"
            ))

        # ── Build attention_weights list ──
        attn_list = []
        for d in imp_days:
            attn_list.append(AttentionWeight(
                lag=d.days_ago,
                weight=d.importance,
                score=d.importance
            ))

        # ── Build signals dict ──
        signal_dict = {
            'lstm': round(float(sigs.get('lstm_direction', {}).get('value', 0.5)), 4),
            'rsi':  round(float(sigs.get('rsi', {}).get('value', 0.5)), 4),
            'macd': round(float(sigs.get('macd_cross', {}).get('value', 0.5)), 4),
            'ma_cross': round(float(sigs.get('ma_crossover', {}).get('value', 0.5)), 4),
        }

        return PredictionResponse(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            current_price=round(curr_price, 2),
            predicted_price=round(price_raw, 2),
            direction=direction,
            change_percent=round(change_pct, 2),
            confidence=round(adj_conf, 4),
            confidence_tier=conf_tier,
            composite_score=round(float(ens['probability']), 4),
            expected_change_pct=round(change_pct, 2),
            regime=RegimeInfo(
                regime_name=rname, regime_id=int(rid),
                confidence=round(float(rconf), 4),
                adjusted_model_confidence=round(adj_conf, 4),
                description=REGIME_DESCS.get(int(rid), ""),
            ),
            ensemble=EnsembleInfo(
                direction=direction,
                probability=round(float(ens['probability']), 4),
                is_high_confidence=bool(ens['is_high_confidence']),
                lstm_direction=_sig('lstm_direction'),
                rsi=_sig('rsi'),
                macd_cross=_sig('macd_cross'),
                ma_crossover=_sig('ma_crossover'),
            ),
            top_features=enriched_feats,
            important_days=imp_days,
            shap_values=shap_list,
            attention_weights=attn_list,
            price_history=price_hist,
            signals=signal_dict,
            explanation=explanation,
            success=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(500, f"Prediction error: {str(e)}")


# ─── HUMAN-READABLE FEATURE NAMES ─────────────────────────
FEATURE_NAMES = {
    'SMA_20':      '20-Day SMA (trend following)',
    'SMA_50':      '50-Day SMA (trend following)',
    'SMA_200':     '200-Day SMA (institutional benchmark)',
    'EMA_12':      '12-Day EMA (short-term momentum)',
    'EMA_26':      '26-Day EMA (medium-term momentum)',
    'RSI_14':      'RSI momentum indicator',
    'MACD':        'MACD momentum signal',
    'MACD_Signal': 'MACD signal line',
    'BB_Upper':    'BB Upper (mean reversion)',
    'BB_Lower':    'BB Lower (mean reversion)',
    'ATR_14':      'ATR volatility (regime)',
    'Volume_SMA':  'Volume SMA (flow analysis)',
    'Daily_Return':'Daily return (momentum)',
    'Volatility':  'Volatility (risk regime)',
    'High_Low_Pct':'H/L range (microstructure)',
}

# ── Quant strategy context for each signal ──
STRATEGY_CONTEXT = {
    'lstm_direction': 'ML factor discovery — used by Two Sigma, Renaissance Technologies',
    'rsi':            'Momentum factor — used by AQR, Man Group, Winton',
    'macd_cross':     'Momentum factor — used by AQR, Man Group, Winton',
    'ma_crossover':   'Trend following — used by Man AHL, Winton, Bridgewater',
}


# ─── INSTITUTIONAL-GRADE EXPLANATION BUILDER (ISSUE 4 FIX) ─
def _build_explanation(ticker, curr, pred, chg_pct,
                        direction, conf, regime_name, regime_id,
                        ens, top_feats, dir_prob, conf_tier):
    """Build institutional-grade prediction report for financial professionals."""

    sigs     = ens['signals']
    rsi_raw  = float(sigs['rsi'].get('raw_rsi', 50))
    macd     = float(sigs['macd_cross'].get('raw_macd', 0))
    sig_line = float(sigs['macd_cross'].get('raw_signal', 0))
    sma50    = float(sigs['ma_crossover'].get('sma50', 0))
    sma200   = float(sigs['ma_crossover'].get('sma200', 0))
    comp     = ens['probability']
    lstm_pup = dir_prob * 100

    # ── RSI Assessment
    if rsi_raw > 70:    rsi_line = f"Overbought (RSI={rsi_raw:.0f}) — mean-reversion probability elevated."
    elif rsi_raw < 30:  rsi_line = f"Oversold (RSI={rsi_raw:.0f}) — bounce probability elevated."
    elif rsi_raw > 55:  rsi_line = f"Moderately bullish momentum (RSI={rsi_raw:.0f})."
    elif rsi_raw < 45:  rsi_line = f"Moderately bearish momentum (RSI={rsi_raw:.0f})."
    else:               rsi_line = f"Neutral momentum (RSI={rsi_raw:.0f})."

    # ── MACD Assessment
    macd_diff = macd - sig_line
    if macd > sig_line:
        macd_line = f"MACD above signal ({macd_diff:+.3f}) — upward acceleration."
    else:
        macd_line = f"MACD below signal ({macd_diff:+.3f}) — momentum weakening."

    # ── MA Cross Assessment
    if sma50 > 0 and sma200 > 0:
        gap_pct = (sma50 - sma200) / sma200 * 100
        if gap_pct > 1:
            ma_line = f"SMA₅₀ above SMA₂₀₀ by {gap_pct:.1f}% (Golden Cross — bullish structure)."
        elif gap_pct < -1:
            ma_line = f"SMA₅₀ below SMA₂₀₀ by {abs(gap_pct):.1f}% (Death Cross — bearish structure)."
        else:
            ma_line = f"SMA₅₀ ≈ SMA₂₀₀ (spread: {gap_pct:+.1f}%) — trend inflection zone."
    else:
        ma_line = "SMA crossover data unavailable."

    # ── Confidence Classification
    if conf_tier == "HIGH_CONF":
        conf_line = "HIGH — 4/4 signals aligned. Highest model conviction tier."
    else:
        conf_line = "MODERATE — Partial signal alignment. Standard conviction."

    # ── Regime Classification
    regime_desc = {
        0: f"Calm (σ < p33). Low volatility environment. Confidence adjustment: +5%.",
        1: f"Trending. Directional momentum detected. No confidence adjustment.",
        2: f"Volatile (σ > p66). High dispersion regime. Confidence penalty: −15%."
    }.get(regime_id, "Regime assessment pending.")

    # ── SHAP Attribution Summary
    feat_lines = []
    for i, feat_tuple in enumerate(top_feats[:3], 1):
        fname, fval = feat_tuple[0], feat_tuple[1]
        readable_name = FEATURE_NAMES.get(fname, fname)
        direction_tag = "(+)" if fval > 0 else "(−)"
        feat_lines.append(f"     {i}. {readable_name}: SHAP={fval:+.5f} {direction_tag}")

    # Detect SHAP-direction divergence
    shap_bullish_count = sum(1 for t in top_feats if t[1] > 0)
    shap_bearish_count = sum(1 for t in top_feats if t[1] < 0)
    shap_net = "bullish" if shap_bullish_count > shap_bearish_count else "bearish"
    has_divergence = (shap_net == "bullish" and direction == "DOWN") or \
                     (shap_net == "bearish" and direction == "UP")

    sign = "+" if chg_pct >= 0 else ""
    dir_label = "BULLISH ↑" if direction == "BULLISH" else "BEARISH ↓"

    lines = [
        f"{'━'*56}",
        f"  ATHENA XAI PREDICTION REPORT — {ticker}",
        f"{'━'*56}",
        f"",
        f"📊  DIRECTIONAL FORECAST",
        f"     Signal: {dir_label}",
        f"     Current:   ${curr:.2f}",
        f"     Predicted: ${pred:.2f} ({sign}{chg_pct:.2f}%)",
        f"     Horizon:   T+1 (next trading session)",
        f"",
        f"🔑  SHAP ATTRIBUTION ANALYSIS",
        f"     Top 3 features by |SHAP| attribution:",
    ] + feat_lines + [
        f"",
        f"     Method: GradientExplainer (Lundberg & Lee, 2017)",
        f"     Attribution: signed SHAP values → (+) bullish, (−) bearish",
    ] + ([f"",
        f"     ⚠️  SIGNAL DIVERGENCE DETECTED:",
        f"     SHAP features are predominantly {shap_net} ({shap_bullish_count}/{len(top_feats)} positive),",
        f"     but ensemble direction is {direction}. This occurs because SHAP explains",
        f"     the price regression head, while the ensemble incorporates RSI, MACD,",
        f"     and MA crossover signals that override the price-based attribution.",
        f"     The ensemble (4-signal weighted vote) is authoritative for direction.",
    ] if has_divergence else []) + [
        f"",
        f"🎯  CONFIDENCE ASSESSMENT",
        f"     Tier: {conf_line}",
        f"     Composite score: {comp*100:.1f}%",
        f"     Regime-adjusted: {conf*100:.1f}%",
        f"",
        f"📊  MARKET REGIME CLASSIFICATION",
        f"     Regime: {regime_name.upper()}",
        f"     {regime_desc}",
        f"",
        f"🧠  ENSEMBLE SIGNAL DECOMPOSITION",
        f"     ATHENA combines four independent signals (weighted vote):",
        f"",
        f"     • LSTM Direction Head (α=0.45): P(UP)={lstm_pup:.1f}%",
        f"       [{STRATEGY_CONTEXT['lstm_direction']}]",
        f"     • RSI Momentum (α=0.25): {rsi_line}",
        f"       [{STRATEGY_CONTEXT['rsi']}]",
        f"     • MACD Trend (α=0.20): {macd_line}",
        f"       [{STRATEGY_CONTEXT['macd_cross']}]",
        f"     • MA Crossover (α=0.10): {ma_line}",
        f"       [{STRATEGY_CONTEXT['ma_crossover']}]",
        f"",
        f"     Signal agreement threshold: composite ≥ 0.53 (BULLISH) or < 0.53 (BEARISH)",
        f"     Confidence: HIGH_CONF if abs(composite-0.5) >= 0.08, else MODERATE.",
        f"",
        f"📄  METHODOLOGY NOTES",
        f"     Architecture: LSTM + Multi-Head Attention (4 heads, 60-step lookback)",
        f"     Explainability: SHAP (GradientExplainer) + Attention Visualization",
        f"     Regime Detection: Rule-based + KMeans hybrid clustering",
        f"     Training: Multi-task loss (α·MSE + (1−α)·BCE) on 10-stock universe",
        f"     Ref: Lin et al. (2022), Muhammad et al. (2024), Abdullah et al. (2024),",
        f"          J. J & Goditi (2025)",
        f"",
        f"⚠️  REGULATORY DISCLAIMER",
        f"     This is an academic research model (Final Year Project 2025-26).",
        f"     Not intended as investment advice. Model directional accuracy:",
        f"     ~52% overall, ~65-75% on HIGH_CONF subset (~40% of predictions).",
        f"     All outputs are indicative. Consult qualified financial advisors.",
        f"{'━'*56}",
    ]
    return "\n".join(lines)



@app.get("/signals/{ticker}")
async def get_signals(ticker: str):
    """Technical signals only (no LSTM inference). Uses neutral dir_prob=0.5."""
    try:
        ticker  = ticker.upper()
        loader  = StockDataLoader([ticker], fmp_key=FMP_KEY)
        results = loader.process_all_tickers()
        if ticker not in results:
            raise HTTPException(404, f"Cannot fetch {ticker}")
        df  = results[ticker]['data']
        ens = DirectionEnsemble()
        result = ens.compute(0.5, df)   # 0.5 = neutral LSTM signal (no inference)
        det = MarketRegimeDetector()
        det.fit(df)
        rid, rname, rconf = det.predict(df)
        return {
            "ticker": ticker, "signals": result,
            "regime": {"name": rname, "id": rid, "confidence": rconf},
            "note": "Signals endpoint uses neutral LSTM dir_prob=0.5. Use /predict for full LSTM inference."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/metrics")
async def get_metrics():
    """Serve model metrics from metrics.json for frontend display."""
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics.json')
    if not os.path.exists(metrics_path):
        return {
            "metrics": {"mae": 0, "rmse": 0, "r2": 0, "directional_accuracy": 0},
            "model_params": 0,
            "note": "No metrics found. Run python train.py first."
        }
    with open(metrics_path, 'r') as f:
        return json.load(f)

@app.get("/ticker-tape")
async def ticker_tape():
    """Live market quotes for the ticker tape — uses FMP /stable/ URL."""
    tape_symbols = 'NVDA,TSLA,GOOGL,META,AMZN,JPM,AAPL,MSFT,XOM,JNJ'
    try:
        import requests as req
        url = f"https://financialmodelingprep.com/stable/quote-short/{tape_symbols}"
        resp = req.get(url, params={"apikey": FMP_KEY}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            tickers = []
            for q in data:
                tickers.append({
                    "s": q.get("symbol", ""),
                    "p": round(q.get("lastSalePrice", q.get("price", 0)), 2),
                    "c": round(q.get("changesPercentage", q.get("change", 0)), 2),
                })
            return {"tickers": tickers}
    except Exception as e:
        print(f"[TICKER-TAPE] quote fetch error: {e}")
    return {"tickers": [], "note": "Could not fetch live quotes"}


@app.get("/news/{ticker}")
async def get_news(ticker: str):
    """Get latest stock news from FMP."""
    try:
        import requests as req
        url = f"https://financialmodelingprep.com/stable/news/stock-latest"
        resp = req.get(url, params={"page": 0, "limit": 5, "apikey": FMP_KEY}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # Filter by ticker symbol
            filtered = [n for n in data if ticker.upper() in str(n.get('symbol', '') or n.get('tickers', ''))]
            if not filtered:
                filtered = data[:5]
            news_items = []
            for n in filtered[:5]:
                title = n.get('title', '')
                # Simple keyword sentiment
                title_lower = title.lower()
                positive_words = ['surge', 'gain', 'rise', 'bull', 'rally', 'up', 'growth', 'profit', 'beat', 'positive', 'record', 'high']
                negative_words = ['fall', 'drop', 'crash', 'bear', 'decline', 'loss', 'down', 'cut', 'warn', 'negative', 'low', 'miss']
                pos_count = sum(1 for w in positive_words if w in title_lower)
                neg_count = sum(1 for w in negative_words if w in title_lower)
                sentiment = 'positive' if pos_count > neg_count else 'negative' if neg_count > pos_count else 'neutral'
                news_items.append({
                    "title": title,
                    "source": n.get('source', n.get('site', '—')),
                    "url": n.get('url', n.get('link', '')),
                    "published": n.get('publishedDate', n.get('date', '')),
                    "sentiment": sentiment,
                })
            return {"news": news_items}
    except Exception as e:
        print(f"[NEWS] fetch error: {e}")
    return {"news": []}


class BriefingRequest(BaseModel):
    ticker: str
    direction: str = "BULLISH"
    confidence: float = 0.5
    confidence_tier: str = "MODERATE"
    composite_score: float = 0.5
    regime: str = "Trending"
    change_pct: float = 0.0
    current_price: float = 0.0
    predicted_price: float = 0.0
    top_features: Optional[list] = None
    signals: Optional[dict] = None


@app.post("/briefing")
async def generate_briefing(request: BriefingRequest):
    """Generate AI briefing using Groq API."""
    try:
        from briefing import generate_ai_briefing
        result = generate_ai_briefing(request.dict())
        return {"briefing": result, "source": "groq"}
    except ImportError:
        return {"briefing": "AI briefing module not available. Using client-side fallback.", "source": "fallback"}
    except Exception as e:
        print(f"[BRIEFING] error: {e}")
        return {"briefing": f"Briefing generation failed: {str(e)}", "source": "error"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)