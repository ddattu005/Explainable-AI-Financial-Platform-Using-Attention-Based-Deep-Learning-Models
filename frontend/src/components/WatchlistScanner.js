/**
 * WatchlistScanner.js — Bloomberg-style multi-stock signal screener
 * Scans up to 8 tickers simultaneously via /predict endpoint.
 * Color-coded by confidence tier. Click any row to load full prediction.
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { predictStock } from '../services/api';

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'META', 'JPM', 'AMZN'];

const TIER_STYLES = {
  HIGH_CONF: { color: '#00e87a', bg: 'rgba(0,232,122,0.1)',  border: 'rgba(0,232,122,0.3)', label: 'HIGH CONF' },
  MODERATE:  { color: '#f5a623', bg: 'rgba(245,166,35,0.1)', border: 'rgba(245,166,35,0.3)', label: 'MODERATE'  },
};

const REGIME_COLORS = { Calm: '#7cc674', Trending: '#00d2ff', Volatile: '#ff3b6b' };

const WatchlistScanner = ({ onSelectPrediction, onAddAuditEntry }) => {
  const [scanResults, setScanResults] = useState({});
  const [scanStates, setScanStates]   = useState({});
  const [scanning, setScanning]       = useState(false);

  const scanRow = async (ticker) => {
    setScanStates(s => ({ ...s, [ticker]: 'loading' }));
    try {
      const result = await predictStock(ticker);
      const topShap = result.top_features?.[0]?.name || '—';
      const isUp = result.direction?.includes('UP');
      setScanResults(s => ({ ...s, [ticker]: {
        prediction: result,
        dir: isUp ? 'UP' : 'DOWN',
        tier: result.confidence_tier || 'MODERATE',
        comp: (result.confidence || 0.5) * 100,
        chg: result.change_percent || 0,
        regime: result.regime?.regime_name || '—',
        shap: topShap,
      }}));
      setScanStates(s => ({ ...s, [ticker]: 'done' }));
      if (onAddAuditEntry) onAddAuditEntry(result);
    } catch (err) {
      console.error(`Scan failed for ${ticker}:`, err);
      setScanStates(s => ({ ...s, [ticker]: 'error' }));
    }
  };

  const scanAll = async () => {
    setScanning(true);
    // Fire all scans simultaneously
    await Promise.all(DEFAULT_TICKERS.map(ticker => scanRow(ticker)));
    setScanning(false);
  };

  const handleRowClick = (ticker) => {
    const data = scanResults[ticker];
    if (data?.prediction && onSelectPrediction) {
      onSelectPrediction(data.prediction);
    }
  };

  // Summary stats
  const scannedResults = Object.values(scanResults);
  const bullCount   = scannedResults.filter(r => r.dir === 'UP').length;
  const highCount   = scannedResults.filter(r => r.tier === 'HIGH_CONF').length;
  const avgComp     = scannedResults.length > 0
    ? (scannedResults.reduce((a, r) => a + r.comp, 0) / scannedResults.length).toFixed(1)
    : '—';
  const regimeCounts = {};
  scannedResults.forEach(r => { regimeCounts[r.regime] = (regimeCounts[r.regime] || 0) + 1; });
  const dominantRegime = Object.entries(regimeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || '—';

  return (
    <div className="watchlist-section">
      {/* Main Table Card */}
      <motion.div className="card wl-card"
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
      >
        {/* Header */}
        <div className="wl-header">
          <div className="wl-header-left">
            <span className="wl-icon">📡</span>
            <span className="wl-title">Watchlist Scanner</span>
            <span className="wl-subtitle">{DEFAULT_TICKERS.length} STOCKS · LIVE SIGNAL SCAN</span>
          </div>
          <div className="wl-header-right">
            <span className="wl-info-tag">ℹ Click any row to load full XAI prediction · FINRA 17a-4 compliant</span>
            <button className="wl-scan-all-btn" onClick={scanAll} disabled={scanning}>
              {scanning ? '⟳ SCANNING…' : '⚡ SCAN ALL'}
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="wl-table-wrap">
          <table className="wl-table">
            <thead>
              <tr>
                {['TICKER', 'DIRECTION', 'Δ PRICE', 'CONFIDENCE TIER', 'COMPOSITE', 'REGIME', 'TOP SHAP DRIVER', 'ACTION'].map(h => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {DEFAULT_TICKERS.map((ticker, i) => {
                  const st   = scanStates[ticker];
                  const data = scanResults[ticker];
                  const tier = data ? (TIER_STYLES[data.tier] || TIER_STYLES.MODERATE) : null;
                  const isUp = data?.dir === 'UP';

                  return (
                    <motion.tr
                      key={ticker}
                      className="wl-row"
                      onClick={() => handleRowClick(ticker)}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.04 }}
                    >
                      <td className="wl-ticker">{ticker}</td>
                      <td className={`wl-dir ${st === 'done' ? (isUp ? 'up' : 'down') : ''}`}>
                        {st === 'loading'
                          ? <span className="wl-scanning">scanning…</span>
                          : st === 'done'
                          ? (isUp ? '↑ BULL' : '↓ BEAR')
                          : st === 'error' ? '✗ ERR' : '—'}
                      </td>
                      <td className={`wl-chg ${st === 'done' ? (data.chg >= 0 ? 'up' : 'down') : ''}`}>
                        {st === 'done' ? `${data.chg >= 0 ? '+' : ''}${data.chg.toFixed(2)}%` : '—'}
                      </td>
                      <td>
                        {st === 'done' && tier
                          ? <span className="wl-tier-badge" style={{ color: tier.color, background: tier.bg, borderColor: tier.border }}>{tier.label}</span>
                          : <span className="wl-dash">—</span>}
                      </td>
                      <td className="wl-comp">
                        {st === 'done' ? `${data.comp.toFixed(1)}%` : '—'}
                      </td>
                      <td style={{ color: st === 'done' ? (REGIME_COLORS[data.regime] || 'var(--dim)') : 'var(--muted)' }}>
                        {st === 'done' ? data.regime : '—'}
                      </td>
                      <td className="wl-shap">
                        {st === 'done' ? data.shap : '—'}
                      </td>
                      <td>
                        <button
                          className="wl-row-scan-btn"
                          onClick={(e) => { e.stopPropagation(); scanRow(ticker); }}
                          disabled={st === 'loading'}
                        >
                          {st === 'loading' ? '…' : st === 'done' ? '↺' : 'SCAN'}
                        </button>
                      </td>
                    </motion.tr>
                  );
                })}
              </AnimatePresence>
            </tbody>
          </table>
        </div>

        {/* Legend */}
        <div className="wl-legend">
          {Object.entries(TIER_STYLES).map(([k, v]) => (
            <span key={k} className="wl-legend-item" style={{ color: v.color }}>
              <span className="wl-legend-dot" style={{ background: v.bg, borderColor: v.border }} />
              {k.replace('_', ' ')}
            </span>
          ))}
          <span className="wl-legend-note">FMP API · Click SCAN ALL to run batch · Click ↺ to refresh row</span>
        </div>
      </motion.div>

      {/* Summary Stats */}
      {scannedResults.length > 0 && (
        <div className="wl-stats-grid">
          {[
            { label: 'BULLISH SIGNALS', value: `${bullCount} / ${DEFAULT_TICKERS.length}`, color: 'var(--green)', bg: 'rgba(0,232,122,0.05)' },
            { label: 'HIGH CONF COUNT', value: `${highCount} / ${DEFAULT_TICKERS.length}`, color: 'var(--cyan)',  bg: 'rgba(0,210,255,0.05)' },
            { label: 'AVG COMPOSITE',   value: `${avgComp}%`,                              color: 'var(--amber)', bg: 'rgba(245,166,35,0.05)' },
            { label: 'DOMINANT REGIME', value: dominantRegime,                              color: 'var(--purple)',bg: 'rgba(155,109,255,0.05)' },
          ].map(s => (
            <motion.div key={s.label} className="card wl-stat-card"
              style={{ background: s.bg }}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            >
              <div className="wl-stat-label">{s.label}</div>
              <div className="wl-stat-value" style={{ color: s.color }}>{s.value}</div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
};

export default WatchlistScanner;
