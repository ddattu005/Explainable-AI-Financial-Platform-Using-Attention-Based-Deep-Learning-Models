# ATHENA XAI Financial Platform — Full Overhaul

## Codebase Summary (5 sentences)

The ATHENA project is a React + FastAPI stock prediction platform using LSTM+Attention (15 features, 60-day lookback, 368K params) trained on 10 tickers. The backend ([api/main.py](file:///Users/teja/ATHENA/api/main.py)) serves predictions via `/predict` using a 4-signal ensemble ([predict.py](file:///Users/teja/ATHENA/predict.py)), SHAP explanations ([shap_explainer.py](file:///Users/teja/ATHENA/shap_explainer.py)), and regime detection ([regime_detector.py](file:///Users/teja/ATHENA/regime_detector.py)). The frontend has 15 components organized as Dashboard/Watchlist/Audit tabs, with [TickerTape.js](file:///Users/teja/ATHENA/frontend/src/components/TickerTape.js) polling a `/markets` endpoint (incorrect URL), [WatchlistScanner.js](file:///Users/teja/ATHENA/frontend/src/components/WatchlistScanner.js) doing sequential staggered scans, and [AIBriefingPanel.js](file:///Users/teja/ATHENA/frontend/src/components/AIBriefingPanel.js) generating briefings client-side only. The model currently has class imbalance (DOWN F1 = 0%, predicts UP for everything) and NEUTRAL tier logic is spread across 8+ files. The ticker tape uses the wrong FMP API base URL (`/api/v3/` instead of `/stable/`).

---

## Phase 1 — Critical Fixes

### FIX 1 — Remove Welcome Card

#### [MODIFY] [App.js](file:///Users/teja/ATHENA/frontend/src/App.js)
Delete the `{!prediction && !loading && !error && (...)}` block (lines 256–287) that renders the "EXPLAINABLE AI" welcome section with feature cards.

---

### FIX 2 — Delete NEUTRAL Tier

#### [MODIFY] [predict.py](file:///Users/teja/ATHENA/predict.py)
- Change thresholds: `direction = "BULLISH" if composite >= 0.53 else "BEARISH"`
- `confidence_tier = "HIGH_CONF" if abs(composite - 0.5) >= 0.08 else "MODERATE"`
- Remove `is_neutral` field entirely, replace with `is_neutral: False` always
- Remove NEUTRAL from docstrings and comments

#### [MODIFY] [main.py](file:///Users/teja/ATHENA/api/main.py)
- Remove `is_neutral` from [EnsembleInfo](file:///Users/teja/ATHENA/api/main.py#61-70) model (line 65)
- Remove NEUTRAL from `PredictionResponse.confidence_tier` comment (line 79)
- Remove `elif ens['is_neutral']: conf_tier = "NEUTRAL"` block (lines 221–222)
- Remove NEUTRAL references from [_build_explanation()](file:///Users/teja/ATHENA/api/main.py#363-501) (lines 406–407, 483)

#### [MODIFY] [EnsemblePanel.js](file:///Users/teja/ATHENA/frontend/src/components/EnsemblePanel.js)
- Remove `is_neutral` tier label mapping (line 92)
- Remove NEUTRAL footer text (lines 219–228)

#### [MODIFY] [WatchlistScanner.js](file:///Users/teja/ATHENA/frontend/src/components/WatchlistScanner.js)
- Remove `NEUTRAL` from `TIER_STYLES` (line 16)
- Legend will auto-update since it iterates `TIER_STYLES`

#### [MODIFY] [AuditLogPanel.js](file:///Users/teja/ATHENA/frontend/src/components/AuditLogPanel.js)
- Remove `NEUTRAL` from `TIER_STYLES` (line 13)
- Update compliance item about NEUTRAL (line 21)

#### [MODIFY] [AIBriefingPanel.js](file:///Users/teja/ATHENA/frontend/src/components/AIBriefingPanel.js)
- Remove NEUTRAL tier in [generateBriefing()](file:///Users/teja/ATHENA/frontend/src/components/AIBriefingPanel.js#29-98) paragraph logic (lines 63–64, 82–83, 91)
- Remove NEUTRAL from context badge (line 141 shows `· {tier}` — will show MODERATE/HIGH_CONF only)

#### [MODIFY] [PredictionCard.js](file:///Users/teja/ATHENA/frontend/src/components/PredictionCard.js)
- Remove `NEUTRAL` from `tierStyles` (line 37)
- Remove `isNeutral` variable (line 12)

#### [MODIFY] [App.js](file:///Users/teja/ATHENA/frontend/src/App.js)
- Change [addAuditEntry](file:///Users/teja/ATHENA/frontend/src/App.js#79-97): default tier from `'NEUTRAL'` to `'MODERATE'` (line 88)

---

### FIX 3 — Remove MODEL ANALYSIS Panel

#### [MODIFY] [App.js](file:///Users/teja/ATHENA/frontend/src/App.js)
- Remove `<ExplanationPanel explanation={prediction.explanation} />` (line 247)
- Remove `import ExplanationPanel` (line 20)

---

### FIX 4 — PREDICT Button Pipeline
Already works: [StockSelector](file:///Users/teja/ATHENA/frontend/src/components/StockSelector.js#25-82) calls `onPredict` → [handlePredict](file:///Users/teja/ATHENA/frontend/src/App.js#98-127) → `predictStock()` → `POST /predict` → sets [prediction](file:///Users/teja/ATHENA/shap_explainer.py#84-115) state → all child components read from [prediction](file:///Users/teja/ATHENA/shap_explainer.py#84-115) props. The response shape from `/predict` already includes direction, composite_score (as `confidence`), confidence_tier, regime, ensemble signals, top_features, and important_days. I'll add `price_history` data to the response and map the response field names to match the required shape.

#### [MODIFY] [main.py](file:///Users/teja/ATHENA/api/main.py)
- Add `price_history` field to [PredictionResponse](file:///Users/teja/ATHENA/api/main.py#71-87) — last 60 days of close prices
- Clean up `direction` field to return `"BULLISH"` / `"BEARISH"` only (currently returns `"UP ⬆️"`)
- Add `composite_score` field (same as `confidence`)
- Add `expected_change_pct` field (same as `change_percent`)
- Add `shap_values` in the new shape with [feature](file:///Users/teja/ATHENA/shap_explainer.py#133-141), `shap_val`, `contribution_pct`, `direction`
- Add [attention_weights](file:///Users/teja/ATHENA/lstm_attention.py#174-178) in the new shape with `lag`, [weight](file:///Users/teja/ATHENA/lstm_attention.py#174-178), `score`

---

## Phase 2 — Connectivity

### FIX 5 — Ticker Tape

#### [MODIFY] [main.py](file:///Users/teja/ATHENA/api/main.py)
Add `GET /ticker-tape` endpoint that fetches from FMP `/stable/quote-short/{tickers}` with the 10 tickers specified.

#### [MODIFY] [TickerTape.js](file:///Users/teja/ATHENA/frontend/src/components/TickerTape.js)
- Change endpoint from `/markets` to `/ticker-tape`
- Change poll interval from 5 minutes to 8 seconds

---

### FIX 6 — Watchlist SCAN ALL

#### [MODIFY] [WatchlistScanner.js](file:///Users/teja/ATHENA/frontend/src/components/WatchlistScanner.js)
- Replace sequential `setTimeout` stagger in [scanAll](file:///Users/teja/ATHENA/frontend/src/components/WatchlistScanner.js#49-57) with `Promise.all`
- Add pulse skeleton animation while scanning
- Color rows: green (HIGH_CONF BULLISH), red (HIGH_CONF BEARISH), yellow (MODERATE)

---

### FIX 7 — AI Briefing with Groq API

#### [NEW] [briefing.py](file:///Users/teja/ATHENA/briefing.py)
- Use `groq` SDK with the provided key
- `POST /briefing` accepts full prediction context
- Returns 3-paragraph institutional report using `llama-3.3-70b-versatile` model

#### [MODIFY] [main.py](file:///Users/teja/ATHENA/api/main.py)
- Import briefing.py and add `POST /briefing` endpoint

#### [MODIFY] [AIBriefingPanel.js](file:///Users/teja/ATHENA/frontend/src/components/AIBriefingPanel.js)
- Call `/briefing` on button click instead of client-side generation
- Keep existing [generateBriefing()](file:///Users/teja/ATHENA/frontend/src/components/AIBriefingPanel.js#29-98) as fallback if API fails

---

## Phase 3 — ML Model Fixes

#### [MODIFY] [train.py](file:///Users/teja/ATHENA/train.py)
- **ML FIX 1**: Replace `nn.BCELoss()` with `nn.BCEWithLogitsLoss(pos_weight=...)` in [MultiTaskLoss](file:///Users/teja/ATHENA/train.py#33-60)
- **ML FIX 2**: Add label noise filtering — threshold returns > 0.003, mask X and y
- **ML FIX 5**: Change gradient clipping from 0.5 to 1.0

#### [MODIFY] [lstm_attention.py](file:///Users/teja/ATHENA/lstm_attention.py)
- **ML FIX 3**: Add temperature scaling `attention_scores = attention_scores / 0.5` before softmax

#### [MODIFY] [shap_explainer.py](file:///Users/teja/ATHENA/shap_explainer.py)
- **ML FIX 4**: Normalize SHAP values to percentages after computing raw values

---

## Phase 4 — Polish

#### [MODIFY] [FeatureImportance.js](file:///Users/teja/ATHENA/frontend/src/components/FeatureImportance.js)
- SHAP bars: green if `shap_val >= 0`, red if negative (already done — bars use `positive`/`negative` classes)

#### [MODIFY] [AttentionHeatmap.js](file:///Users/teja/ATHENA/frontend/src/components/AttentionHeatmap.js)
- Top 3 lags: bright purple `#A855F7`, rest: `#4B3F6A`

#### [MODIFY] [PriceChart.js](file:///Users/teja/ATHENA/frontend/src/components/PriceChart.js)
- Replace spike with dashed `ReferenceLine` at predicted price + shaded `ReferenceArea` (±1.5% bounds)

#### [MODIFY] [StockSelector.js](file:///Users/teja/ATHENA/frontend/src/components/StockSelector.js)
- PREDICT button: show `"⟳ SCANNING..."` with pulse while loading

#### [NEW] News panel component below price chart
- `GET /stable/news/stock-latest?page=0&limit=5&apikey={key}`
- Filter by ticker, show headline, source, time ago, sentiment badge

---

## Verification Plan

### Automated Tests
1. **Grep verification**: After Phase 1 FIX 2:
   ```bash
   grep -r "NEUTRAL" /Users/teja/ATHENA/frontend/src/
   grep -r "NEUTRAL" /Users/teja/ATHENA/*.py
   grep -r "NEUTRAL" /Users/teja/ATHENA/api/
   ```
   All must return zero results.

2. **Backend curl tests** after Phase 2:
   ```bash
   curl http://localhost:8000/ticker-tape
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"ticker": "AAPL"}'
   curl -X POST http://localhost:8000/briefing -H "Content-Type: application/json" -d '{"ticker": "AAPL", "direction": "BULLISH", "confidence": 0.65}'
   ```

3. **Existing test suite**: `python /Users/teja/ATHENA/api/test_api.py` — tests root, health, stocks, predict, error handling

### Manual Verification
1. **Phase 1**: Open `localhost:3000` — verify no welcome card on initial load, no NEUTRAL text anywhere on Dashboard/Watchlist/Audit tabs, no MODEL ANALYSIS text panel
2. **Phase 2**: Verify ticker tape updates every 8 seconds, SCAN ALL fires all 8 simultaneously, AI Briefing calls Groq API
3. **Phase 3**: After retraining, verify DOWN F1 > 0 in classification report
4. **Phase 4**: Visual check — SHAP bars colored correctly, attention weights colored, price chart shows prediction bands
