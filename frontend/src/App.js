/**
 * App.js — ATHENA XAI Stock Predictor
 * Connects to FastAPI backend at localhost:8000
 * 3-tab layout: Dashboard | Watchlist | Audit Log
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';

import Header           from './components/Header';
import StockSelector    from './components/StockSelector';
import PredictionCard   from './components/PredictionCard';
import FeatureImportance from './components/FeatureImportance';
import AttentionHeatmap  from './components/AttentionHeatmap';
import EnsemblePanel     from './components/EnsemblePanel';
import RegimeCard        from './components/RegimeCard';
import MetricsCard       from './components/MetricsCard';
import PriceChart        from './components/PriceChart';

import LoadingSpinner    from './components/LoadingSpinner';
import TickerTape        from './components/TickerTape';
import AIBriefingPanel   from './components/AIBriefingPanel';
import WatchlistScanner  from './components/WatchlistScanner';
import AuditLogPanel     from './components/AuditLogPanel';

import { predictStock, getHealth, getSupportedStocks, getMetrics } from './services/api';

function App() {
  const [selectedStock, setSelectedStock]     = useState('AAPL');
  const [prediction,    setPrediction]         = useState(null);
  const [loading,       setLoading]            = useState(false);
  const [loadStep,      setLoadStep]           = useState(-1);
  const [error,         setError]              = useState(null);
  const [apiHealth,     setApiHealth]          = useState(null);
  const [supportedStocks, setSupportedStocks]  = useState([]);
  const [metrics,       setMetrics]            = useState(null);
  const [activeTab,     setActiveTab]          = useState('dashboard');
  const [auditLog,      setAuditLog]           = useState([]);
  const stepTimer = useRef(null);

  useEffect(() => {
    checkApiHealth();
    fetchSupportedStocks();
    fetchMetrics();
  }, []);

  const checkApiHealth = async () => {
    try {
      const health = await getHealth();
      setApiHealth(health);
      if (!health.model_loaded) {
        setError('Model not loaded on backend — run python train.py first.');
      }
    } catch {
      setApiHealth({ model_loaded: false, status: 'offline' });
      setError('Cannot connect to API at http://localhost:8000 — start the backend first.');
    }
  };

  const fetchSupportedStocks = async () => {
    try {
      const data = await getSupportedStocks();
      setSupportedStocks(data.stocks || []);
    } catch {
      // silently fall back to hardcoded list in StockSelector
    }
  };

  const fetchMetrics = async () => {
    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch {
      // metrics unavailable — Header/MetricsCard will show dashes
    }
  };

  // ─── Add entry to audit log ───
  const addAuditEntry = (result) => {
    const now = new Date();
    const isUp = result.direction?.includes('UP');
    const entry = {
      timestamp: result.timestamp || now.toISOString(),
      timeStr: now.toLocaleTimeString('en-US', { hour12: false }),
      ticker: result.ticker,
      direction: isUp ? 'UP' : 'DOWN',
      tier: result.confidence_tier || 'MODERATE',
      composite: `${((result.confidence || 0.5) * 100).toFixed(1)}%`,
      regime: result.regime?.regime_name || '—',
      currentPrice: result.current_price,
      predictedPrice: result.predicted_price,
      changePct: result.change_percent,
    };
    setAuditLog(prev => [entry, ...prev].slice(0, 50)); // max 50 entries
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setLoadStep(0);

    // Animate loading steps while waiting for API
    let s = 0;
    stepTimer.current = setInterval(() => {
      s = Math.min(s + 1, 3);
      setLoadStep(s);
      if (s >= 3) clearInterval(stepTimer.current);
    }, 900);

    try {
      const result = await predictStock(selectedStock);
      clearInterval(stepTimer.current);
      setLoadStep(3);
      await new Promise(r => setTimeout(r, 300));
      setPrediction(result);
      addAuditEntry(result);
    } catch (err) {
      clearInterval(stepTimer.current);
      setError(err.response?.data?.detail || 'Prediction failed — check the backend console.');
    } finally {
      setLoading(false);
      setLoadStep(-1);
    }
  };

  // ─── Watchlist: when user clicks a scanned row ───
  const handleWatchlistSelect = (pred) => {
    setPrediction(pred);
    setSelectedStock(pred.ticker);
    setActiveTab('dashboard');
  };

  return (
    <div className="App">
      <TickerTape />

      <Header apiHealth={apiHealth} metrics={metrics}>
        {/* Tab navigation injected into header */}
        <div className="tab-bar">
          {[
            { key: 'dashboard', label: 'Dashboard' },
            { key: 'watchlist', label: 'Watchlist' },
            { key: 'audit',     label: 'Audit Log' },
          ].map(t => (
            <button
              key={t.key}
              className={`tab-btn ${activeTab === t.key ? 'active' : ''}`}
              onClick={() => setActiveTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </Header>

      <main className="main-content">
        <div className="container">

          {/* ═══════════ DASHBOARD TAB ═══════════ */}
          {activeTab === 'dashboard' && (
            <>
              {/* STOCK SELECTOR */}
              <motion.div
                initial={{ opacity: 0, y: -16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45 }}
                className="selector-section"
              >
                <StockSelector
                  selectedStock={selectedStock}
                  onStockChange={(t) => { setSelectedStock(t); setPrediction(null); setError(null); }}
                  onPredict={handlePredict}
                  loading={loading}
                  supportedStocks={supportedStocks}
                />
              </motion.div>

              {/* ERROR */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    key="error"
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="error-banner"
                  >
                    <span className="error-icon">⚠️</span>
                    <span className="error-text">{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* LANDING PAGE — shown when no prediction loaded */}
              {!prediction && !loading && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.2 }}
                  style={{ textAlign: 'center', padding: '60px 20px 40px' }}
                >
                  <h2 style={{
                    fontFamily: "'Courier New', monospace",
                    fontSize: 42, fontWeight: 900, letterSpacing: '0.15em',
                    background: 'linear-gradient(135deg, #00d2ff, #7c3aed, #ff3b6b, #f5a623)',
                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                    marginBottom: 12,
                  }}>EXPLAINABLE AI</h2>
                  <p style={{
                    fontFamily: "'Courier New', monospace",
                    fontSize: 13, color: '#5a6a82', marginBottom: 40, letterSpacing: '0.04em',
                  }}>Predict. Understand. Trust. — Every decision backed by transparent AI reasoning.</p>

                  <div style={{
                    display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
                    gap: 16, maxWidth: 700, margin: '0 auto 40px',
                  }}>
                    {[
                      { icon: '🧠', title: 'LSTM + ATTENTION', desc: '60-day lookback with multi-head temporal focus' },
                      { icon: '🔑', title: 'SHAP ANALYSIS', desc: 'Game-theory feature attribution per prediction' },
                      { icon: '📊', title: 'REGIME DETECTION', desc: 'KMeans: Calm · Trending · Volatile regimes' },
                      { icon: '⚡', title: 'DIRECTION ENSEMBLE', desc: 'LSTM head + RSI + MACD + MA cross signals' },
                    ].map((c, i) => (
                      <motion.div
                        key={c.title}
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 + i * 0.1 }}
                        style={{
                          background: 'rgba(0,210,255,0.04)',
                          border: '1px solid rgba(0,210,255,0.1)',
                          borderRadius: 10, padding: '20px 14px',
                          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
                        }}
                      >
                        <span style={{ fontSize: 28 }}>{c.icon}</span>
                        <span style={{
                          fontFamily: "'Courier New', monospace",
                          fontSize: 10, fontWeight: 700, color: '#00d2ff',
                          letterSpacing: '0.08em',
                        }}>{c.title}</span>
                        <span style={{
                          fontFamily: "'Courier New', monospace",
                          fontSize: 9, color: '#5a6a82', lineHeight: 1.4, textAlign: 'center',
                        }}>{c.desc}</span>
                      </motion.div>
                    ))}
                  </div>

                  <p style={{
                    fontFamily: "'Courier New', monospace",
                    fontSize: 11, color: '#3a4558', fontStyle: 'italic',
                  }}>Select a stock above and click PREDICT to begin</p>
                </motion.div>
              )}

              {/* LOADING */}
              <AnimatePresence>
                {loading && (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <LoadingSpinner currentStep={loadStep} />
                  </motion.div>
                )}
              </AnimatePresence>

              {/* RESULTS */}
              <AnimatePresence>
                {prediction && !loading && (
                  <motion.div
                    key="results"
                    className="results-section"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.4 }}
                  >
                    {/* HERO */}
                    <PredictionCard prediction={prediction} />

                    {/* AI BRIEFING — appears right after prediction hero */}
                    <AIBriefingPanel prediction={prediction} />

                    {/* ROW 1: SHAP + Attention */}
                    <div className="analysis-grid">
                      <FeatureImportance features={prediction.top_features} />
                      <AttentionHeatmap  days={prediction.important_days} />
                    </div>

                    {/* ROW 2: Ensemble + Regime + Metrics */}
                    <div className="analysis-grid-3">
                      <EnsemblePanel ensemble={prediction.ensemble} />
                      <RegimeCard    regime={prediction.regime} />
                      <MetricsCard metrics={metrics} />
                    </div>

                    {/* ROW 3: Chart full-width */}
                    <PriceChart
                      currentPrice={prediction.current_price}
                      predictedPrice={prediction.predicted_price}
                      direction={prediction.direction}
                    />


                  </motion.div>
                )}
              </AnimatePresence>


            </>
          )}

          {/* ═══════════ WATCHLIST TAB ═══════════ */}
          {activeTab === 'watchlist' && (
            <WatchlistScanner
              onSelectPrediction={handleWatchlistSelect}
              onAddAuditEntry={addAuditEntry}
            />
          )}

          {/* ═══════════ AUDIT LOG TAB ═══════════ */}
          {activeTab === 'audit' && (
            <AuditLogPanel auditLog={auditLog} />
          )}

        </div>
      </main>

      <footer className="footer">
        <p>ATHENA · XAI Stock Predictor · Final Year Project 2025–26</p>
        <span>LSTM + Attention + SHAP + KMeans · PyTorch · FastAPI · React · Claude AI</span>
      </footer>
    </div>
  );
}

export default App;