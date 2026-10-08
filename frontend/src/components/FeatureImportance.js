import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const mono = { fontFamily: "'JetBrains Mono', monospace" };

const FEATURE_NAMES = {
  'SMA_20':      '20-Day SMA',
  'SMA_50':      '50-Day SMA',
  'SMA_200':     '200-Day SMA',
  'EMA_12':      '12-Day EMA',
  'EMA_26':      '26-Day EMA',
  'RSI_14':      'RSI (14)',
  'MACD':        'MACD',
  'MACD_Signal': 'MACD Signal',
  'BB_Upper':    'BB UPPER',
  'BB_Lower':    'BB LOWER',
  'ATR_14':      'ATR (14)',
  'Volume_SMA':  'Vol. SMA',
  'Daily_Return':'Daily Ret.',
  'Volatility':  'Volatility',
  'High_Low_Pct':'H/L Range',
};

// Plain-english XAI detail for each feature type
const FEATURE_DETAIL = {
  'BB_Upper':    'Price near upper Bollinger Band — historically precedes mean-reversion pullback. LSTM interpreted proximity as resistance signal.',
  'BB_Lower':    'Lower band provides wide bandwidth signal. Suggests elevated volatility window — LSTM used this as range estimate input.',
  'EMA_12':      'Short-term EMA trending, near-crossover zone — LSTM weighted this as near-term momentum continuation.',
  'EMA_26':      'Medium-term EMA trending. Price above EMA providing dynamic support. Contributes positive momentum signal to LSTM price head.',
  'SMA_20':      'SMA-20 closely tracking predicted price. LSTM price regression anchors near this level — strong mean-pull factor.',
  'SMA_50':      '50-day moving average acting as dynamic support/resistance. Institutional benchmark for trend direction.',
  'SMA_200':     'Long-term trend benchmark. Relative position to SMA-200 signals structural bull/bear market context.',
  'RSI_14':      'Momentum exhaustion indicator. Extremes (>70 or <30) trigger mean-reversion signals in the ensemble.',
  'MACD':        'MACD line position relative to signal line. Crossover direction determines momentum bias in ensemble weight.',
  'MACD_Signal': 'Signal line acts as MACD trigger. Divergence from MACD line indicates momentum shift.',
  'ATR_14':      'Average True Range indicates volatility regime. Higher ATR = wider predicted price range.',
  'Volume_SMA':  'Volume vs 20-day average. Confirms price moves — high volume breakouts carry stronger institutional conviction.',
  'Daily_Return':'Recent daily return momentum. Short-term return direction feeds LSTM sequence as momentum factor.',
  'Volatility':  'Rolling 20-day volatility. High volatility triggers regime penalty — reduces model confidence.',
  'High_Low_Pct':'Intraday high/low range as % of price. Wide ranges signal uncertainty and potential reversal zones.',
};

const TagBtn = ({ active, onClick, children }) => (
  <button onClick={onClick} style={{
    ...mono, fontSize: 8, padding: '2px 8px', borderRadius: 3, cursor: 'pointer', transition: 'all 0.2s',
    border: `1px solid ${active ? 'rgba(0,210,255,0.4)' : 'rgba(255,255,255,0.07)'}`,
    background: active ? 'rgba(0,210,255,0.08)' : 'transparent',
    color: active ? '#00d2ff' : '#5a6a82',
  }}>{children}</button>
);

const FeatureImportance = ({ features }) => {
  const [expandedRow, setExpandedRow] = useState(null);
  const [sortMode, setSortMode]       = useState('impact'); // 'impact' | 'name'

  if (!features?.length) return null;

  const maxAbs = Math.max(...features.map(f => Math.abs(f.impact)), 1e-9);

  const sorted = [...features].sort((a, b) =>
    sortMode === 'impact'
      ? Math.abs(b.impact) - Math.abs(a.impact)
      : (FEATURE_NAMES[a.name] || a.name).localeCompare(FEATURE_NAMES[b.name] || b.name)
  );

  // Detect SHAP-direction split for warning
  const bullCount = features.filter(f => f.impact > 0).length;
  const bearCount = features.length - bullCount;
  const shapNet   = bullCount > bearCount ? 'bullish' : 'bearish';

  return (
    <motion.div
      className="card feature-importance-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.1 }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>
          <div className="card-ico ico-c">🔑</div>
          <span className="card-title-text">SHAP Feature Attribution</span>
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <TagBtn active={sortMode === 'impact'} onClick={() => setSortMode('impact')}>By Impact</TagBtn>
          <TagBtn active={sortMode === 'name'}   onClick={() => setSortMode('name')}>By Name</TagBtn>
        </div>
      </div>

      {/* Click hint banner */}
      <div style={{
        background: 'rgba(245,166,35,0.06)', border: '1px solid rgba(245,166,35,0.2)',
        borderRadius: 5, padding: '7px 10px', marginBottom: 12,
        display: 'flex', gap: 8, alignItems: 'flex-start',
      }}>
        <span>💡</span>
        <div style={{ ...mono, fontSize: 9, color: '#f5a623', lineHeight: 1.65 }}>
          <strong>Click any feature</strong> to see its XAI explanation.{' '}
          {bullCount}/{features.length} bullish (LSTM price: {bullCount > bearCount ? 'UP' : 'DOWN'}),
          RSI+MACD ensemble: {shapNet === 'bullish' ? 'BEARISH (divergence)' : 'BEARISH'}.
        </div>
      </div>

      {/* Table header */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 80px 52px 44px',
        gap: 6, marginBottom: 8, paddingBottom: 7,
        borderBottom: '1px solid rgba(0,210,255,0.07)',
      }}>
        {['FEATURE', 'SHAP VALUE', 'CONTRIB%', 'DIR'].map(h => (
          <span key={h} style={{ ...mono, fontSize: 7, color: '#3a4558', fontWeight: 700, letterSpacing: '0.08em' }}>{h}</span>
        ))}
      </div>

      {/* Rows */}
      <div className="feature-list" style={{ gap: 6 }}>
        {sorted.map((f, i) => {
          const isPos     = f.impact >= 0;
          const pct       = (Math.abs(f.impact) / maxAbs) * 100;
          const displayName = FEATURE_NAMES[f.name] || f.name;
          const isOpen    = expandedRow === f.name;
          const detail    = FEATURE_DETAIL[f.name] || `This feature contributed ${isPos ? 'positively' : 'negatively'} to the LSTM price regression output.`;

          return (
            <motion.div
              key={f.name}
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 + i * 0.06 }}
              style={{
                borderRadius: 6, overflow: 'hidden',
                border: isOpen ? '1px solid rgba(0,210,255,0.15)' : '1px solid transparent',
                transition: 'border 0.2s',
              }}
            >
              {/* Main row — click to expand */}
              <div
                onClick={() => setExpandedRow(isOpen ? null : f.name)}
                style={{
                  display: 'grid', gridTemplateColumns: '1fr 80px 52px 44px',
                  gap: 6, alignItems: 'center', padding: '6px 6px 4px',
                  background: isOpen ? 'rgba(0,210,255,0.06)' : 'transparent',
                  cursor: 'pointer', borderRadius: isOpen ? 0 : 6, transition: 'background 0.2s',
                }}
                className="shap-row"
              >
                <span style={{ ...mono, fontSize: 11, color: '#d8e0ec' }}>
                  {i + 1}. {displayName}{' '}
                  <span style={{ color: '#5a6a82', fontSize: 9 }}>{isOpen ? '▲' : '▼'}</span>
                </span>
                <span style={{ ...mono, fontSize: 10, color: isPos ? '#00e87a' : '#ff3b6b' }}>
                  {isPos ? '+' : ''}{f.impact.toFixed(5)}
                </span>
                <span style={{ ...mono, fontSize: 9, color: '#5a6a82' }}>
                  {f.pct_contribution != null ? `${f.pct_contribution.toFixed(1)}%` : `${(Math.abs(f.impact) / maxAbs * 100).toFixed(1)}%`}
                </span>
                <span style={{
                  ...mono, fontSize: 9, fontWeight: 700,
                  color: isPos ? '#00e87a' : '#ff3b6b',
                  background: isPos ? 'rgba(0,232,122,0.08)' : 'rgba(255,59,107,0.08)',
                  border: `1px solid ${isPos ? 'rgba(0,232,122,0.2)' : 'rgba(255,59,107,0.2)'}`,
                  borderRadius: 3, padding: '1px 5px', textAlign: 'center',
                }}>
                  {isPos ? '↗' : '↘'}
                </span>
              </div>

              {/* Bar */}
              <div style={{ padding: '0 6px 4px' }}>
                <motion.div
                  className={`feature-bar`}
                  style={{ height: 5 }}
                >
                  <motion.div
                    className={`feature-bar-fill ${isPos ? 'positive' : 'negative'}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.9, delay: 0.2 + i * 0.06 }}
                  />
                </motion.div>
              </div>

              {/* Context row (pct_rank, val) */}
              {(f.pct_rank != null || f.context_value != null) && (
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, padding: '0 6px 4px', ...mono, fontSize: 9, color: '#5a6a82' }}>
                  {f.pct_rank != null     && <span title="Percentile rank in 60-day window">P{f.pct_rank.toFixed(0)}</span>}
                  {f.context_value != null && <span title="Current normalized value">val={f.context_value.toFixed(3)}</span>}
                </div>
              )}

              {/* Expanded detail */}
              <AnimatePresence>
                {isOpen && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.22 }}
                    style={{ overflow: 'hidden' }}
                  >
                    <div style={{
                      padding: '10px 10px 8px',
                      background: 'rgba(0,210,255,0.04)',
                      borderTop: '1px solid rgba(0,210,255,0.1)',
                    }}>
                      <div style={{ ...mono, fontSize: 9, color: '#d8e0ec', lineHeight: 1.75 }}>{detail}</div>
                      <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
                        <div style={{ flex: 1, background: 'rgba(255,255,255,0.02)', borderRadius: 5, padding: '5px 8px' }}>
                          <div style={{ fontSize: 7, color: '#3a4558', fontWeight: 700, marginBottom: 3 }}>SHAP IMPACT</div>
                          <div style={{ ...mono, fontSize: 14, color: isPos ? '#00e87a' : '#ff3b6b', fontWeight: 700 }}>
                            {isPos ? '+' : ''}{f.impact.toFixed(5)}
                          </div>
                        </div>
                        <div style={{ flex: 1, background: 'rgba(255,255,255,0.02)', borderRadius: 5, padding: '5px 8px' }}>
                          <div style={{ fontSize: 7, color: '#3a4558', fontWeight: 700, marginBottom: 3 }}>SHARE</div>
                          <div style={{ ...mono, fontSize: 14, color: '#00d2ff', fontWeight: 700 }}>
                            {f.pct_contribution != null ? f.pct_contribution.toFixed(1) : (Math.abs(f.impact) / maxAbs * 100).toFixed(1)}%
                          </div>
                        </div>
                        {f.pct_rank != null && (
                          <div style={{ flex: 1, background: 'rgba(255,255,255,0.02)', borderRadius: 5, padding: '5px 8px' }}>
                            <div style={{ fontSize: 7, color: '#3a4558', fontWeight: 700, marginBottom: 3 }}>PERCENTILE</div>
                            <div style={{ ...mono, fontSize: 14, color: '#9b6dff', fontWeight: 700 }}>P{f.pct_rank.toFixed(0)}</div>
                          </div>
                        )}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      <div className="tip-box" style={{ marginTop: 12 }}>
        💡 SHAP (+) = pushed LSTM price prediction UP. Click features for details. P<em>n</em> = percentile rank in 60-day window.
      </div>

      <style>{`.shap-row:hover { background: rgba(0,210,255,0.04) !important; }`}</style>
    </motion.div>
  );
};

export default FeatureImportance;