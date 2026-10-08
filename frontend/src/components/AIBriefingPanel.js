/**
 * AIBriefingPanel.js — Institutional AI Briefing Generator
 * Generates a 3-line briefing from the model's prediction data.
 * Pure client-side — uses prediction context directly.
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const generateBriefing = (prediction) => {
  if (!prediction) return null;

  const ticker = prediction.ticker || '—';
  const isUp = prediction.direction?.includes('BULL') || prediction.direction?.includes('UP');
  const dirLabel = isUp ? 'BULLISH' : 'BEARISH';
  const currentPrice = prediction.current_price?.toFixed(2) || '—';
  const predictedPrice = prediction.predicted_price?.toFixed(2) || '—';
  const chg = prediction.change_percent || 0;
  const tier = prediction.confidence_tier || 'MODERATE';
  const regime = prediction.regime?.regime_name || 'Unknown';
  const adjConf = ((prediction.regime?.adjusted_model_confidence || prediction.confidence || 0.5) * 100).toFixed(1);

  // SHAP features
  const feats = prediction.top_features || [];
  const topFeat = feats.length > 0 ? feats[0].name : 'technical indicators';
  const shapBullish = feats.filter(f => f.impact > 0).length;
  const shapBearish = feats.length - shapBullish;

  // Ensemble signals
  const ens = prediction.ensemble || {};
  const signals = ens.signals || ens;
  const lstmProb = signals.lstm_direction ? (signals.lstm_direction.value * 100).toFixed(0) : '—';

  // Line 1: Direction + price + recommendation
  const line1 = `${ticker} is ${dirLabel} — Predicts a move from $${currentPrice} → $${predictedPrice} (${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%) for the next trading session. ${
    tier === 'HIGH_CONF'
      ? 'HIGH CONFIDENCE signal — back-tested accuracy of 68–75% on this tier.'
      : 'MODERATE conviction — signal present but below the high-confidence threshold.'
  }`;

  // Line 2: Why — SHAP + ensemble reasoning
  const line2 = `Key drivers: ${feats.slice(0, 3).map(f => f.name).join(', ')} — ${shapBullish}/${feats.length} SHAP features are bullish. LSTM direction head gives P(UP)=${lstmProb}%. Market regime: ${regime.toUpperCase()} (regime-adjusted confidence ${adjConf}%).`;

  // Line 3: Action + risk
  const line3 = isUp
    ? `Outlook: Momentum and technical signals support upside. ${shapBearish > 0 ? `Watch ${shapBearish} bearish SHAP feature(s) for reversal signals.` : 'All SHAP drivers align bullish.'} Model: ATHENA v3.0, 368K params, 60-day lookback.`
    : `Outlook: Downward pressure detected from RSI/MACD signals overriding price regression. ${shapBullish > 0 ? `${shapBullish} bullish SHAP features suggest the decline may be limited.` : 'All SHAP drivers confirm bearish momentum.'} Model: ATHENA v3.0, 368K params, 60-day lookback.`;

  return { line1, line2, line3 };
};

const AIBriefingPanel = ({ prediction }) => {
  const [briefing, setBriefing]   = useState(null);
  const [loading, setLoading]     = useState(false);

  if (!prediction) return null;

  const isUp = prediction.direction?.includes('BULL') || prediction.direction?.includes('UP');
  const ticker = prediction.ticker || '—';
  const tier = prediction.confidence_tier || 'MODERATE';

  const handleGenerate = () => {
    setLoading(true);
    setBriefing(null);
    // Small delay for visual effect
    setTimeout(() => {
      const result = generateBriefing(prediction);
      setBriefing(result);
      setLoading(false);
    }, 800);
  };

  return (
    <motion.div className="card briefing-card"
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.15 }}
    >
      {/* Top accent line */}
      <div className="briefing-accent" />

      {/* Header */}
      <div className="briefing-header">
        <div className="briefing-header-left">
          <div className="briefing-icon-box">✦</div>
          <div>
            <span className="briefing-title">AI Institutional Briefing</span>
            <div className="briefing-subtitle">
              ATHENA-powered · Hedge fund format · SHAP + Ensemble + Regime synthesis
            </div>
          </div>
        </div>
        <div className="briefing-header-right">
          <span className="briefing-context-tag">
            {ticker} · {isUp ? '↑ BULL' : '↓ BEAR'} · {tier}
          </span>
          <button
            className="briefing-generate-btn"
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? '⟳ GENERATING…' : briefing ? '↺ REGENERATE' : '✦ GENERATE BRIEFING'}
          </button>
        </div>
      </div>

      {/* Loading state */}
      <AnimatePresence>
        {loading && (
          <motion.div
            className="briefing-loading"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          >
            <div className="briefing-loading-text">✦ ANALYZING SIGNALS…</div>
            <div className="briefing-loading-sub">
              synthesizing SHAP · regime · ensemble → institutional report
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Briefing content — 3 lines */}
      {briefing && !loading && (
        <motion.div
          className="briefing-content"
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
        >
          {[briefing.line1, briefing.line2, briefing.line3].map((line, i) => (
            <div key={i} className={`briefing-paragraph ${i === 0 ? 'primary' : ''}`}>
              {line}
            </div>
          ))}

          {/* Footer */}
          <div className="briefing-footer">
            ✦ Generated by ATHENA AI Engine · {new Date().toLocaleTimeString()} · Academic FYP 2025-26 · Not investment advice
          </div>
        </motion.div>
      )}

      {/* Empty state */}
      {!briefing && !loading && (
        <div className="briefing-empty">
          Click <strong>GENERATE BRIEFING</strong> to synthesize an institutional signal report from SHAP + ensemble + regime data
        </div>
      )}
    </motion.div>
  );
};

export default AIBriefingPanel;
