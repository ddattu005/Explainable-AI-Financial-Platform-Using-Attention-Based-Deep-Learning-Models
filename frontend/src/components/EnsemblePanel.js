import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const mono = { fontFamily: "'JetBrains Mono', monospace" };

const SIG_LABELS = {
  lstm_direction: 'LSTM Direction Head',
  rsi:            'RSI Momentum',
  macd_cross:     'MACD Crossover',
  ma_crossover:   'MA Crossover',
};

const SIG_WEIGHTS = {
  lstm_direction: 45,
  rsi:            25,
  macd_cross:     20,
  ma_crossover:   10,
};

const SIG_DETAIL = {
  lstm_direction: '60-day sequence model. Direction head converts price regression into P(UP). Dominant signal at 45% weight — captures non-linear temporal patterns invisible to rule-based indicators.',
  rsi:            'RSI momentum exhaustion indicator. Values below 50 midline indicate bearish momentum. Consistent with distribution phase before pullback. 25% ensemble weight.',
  macd_cross:     'MACD line position vs signal line. Bearish crossover confirmed — histogram narrowing suggests bear momentum decelerating. 20% ensemble weight.',
  ma_crossover:   '50-day vs 200-day SMA crossover. Golden Cross structure is bullish long-term but low weight (10%) for short-term T+1 prediction horizon.',
};

/* ── Animated confidence ring ── */
function ConfRing({ value, label, size = 70 }) {
  const [dash, setDash] = useState(0);
  const r    = size / 2 - 6;
  const circ = 2 * Math.PI * r;
  const color = value >= 58 ? '#00e87a' : value <= 42 ? '#ff3b6b'
              : value >= 53 ? '#7cc674' : value <= 47 ? '#e0854c' : '#5a6a82';

  useEffect(() => {
    const t = setTimeout(() => setDash(circ * (value / 100)), 200);
    return () => clearTimeout(t);
  }, [value, circ]);

  return (
    <div style={{ position: 'relative', width: size, height: size, flexShrink: 0 }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={6} />
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={6}
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={circ - dash}
          style={{ transition: 'stroke-dashoffset 1.2s ease, stroke 0.3s ease' }}
        />
      </svg>
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{ ...mono, fontSize: size > 80 ? 15 : 12, color, fontWeight: 700 }}>{value.toFixed(1)}%</span>
        <span style={{ fontSize: 7, color: '#5a6a82', letterSpacing: '0.12em', fontWeight: 700, marginTop: 2 }}>{label}</span>
      </div>
    </div>
  );
}

const EnsemblePanel = ({ ensemble }) => {
  const [disabled, setDisabled]     = useState({});
  const [expandedRow, setExpandedRow] = useState(null);

  if (!ensemble) return null;

  const { signals, probability, is_high_confidence, direction } = ensemble;

  const toggleRow = (key) => {
    setDisabled(p => ({ ...p, [key]: !p[key] }));
    setExpandedRow(prev => prev === key ? null : key);
  };

  // Re-compute composite with enabled signals only
  const enabledKeys = Object.keys(signals || {}).filter(k => !disabled[k]);
  const totalW = enabledKeys.reduce((s, k) => s + (SIG_WEIGHTS[k] || 0), 0);
  const liveComposite = totalW === 0
    ? 50
    : enabledKeys.reduce((s, k) => {
        const val = signals[k]?.value ?? 0.5;
        const w   = SIG_WEIGHTS[k] || 0;
        return s + val * 100 * (w / totalW);
      }, 0);

  const someDisabled = Object.values(disabled).some(Boolean);
  // When all signals enabled, use backend's authoritative direction (uses 53% threshold)
  // Only recompute locally when user has toggled signals off
  const isBull = someDisabled ? liveComposite >= 53 : direction?.includes('BULL');

  const tierLabel = is_high_confidence
    ? { text: 'HIGH CONFIDENCE', color: '#00e87a' }
    : { text: 'MODERATE', color: '#f5a623' };

  return (
    <motion.div
      className="card ensemble-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.1 }}
    >
      <div className="card-title">
        <div className="card-ico ico-g">⚡</div>
        <span className="card-title-text">Direction Ensemble</span>
      </div>

      {/* Live direction + ring */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div>
          <div style={{
            fontSize: 14, fontWeight: 700, letterSpacing: '0.06em',
            color: isBull ? '#00e87a' : '#ff3b6b',
          }}>
            {isBull ? '↑ BULLISH' : '↓ BEARISH'}
          </div>
          <div style={{ ...mono, fontSize: 9, color: '#5a6a82', marginTop: 3 }}>
            {someDisabled ? '⚠ Some signals disabled' : 'All signals active'}
          </div>
        </div>
        <ConfRing value={+liveComposite.toFixed(1)} label="LIVE ADJ." size={66} />
      </div>

      <div style={{ ...mono, fontSize: 8, color: '#3a4558', marginBottom: 8 }}>
        Click rows to toggle signals — watch composite update live ↓
      </div>

      {/* Signal table */}
      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 8 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid rgba(0,210,255,0.08)' }}>
            {['SIGNAL SOURCE', 'WT%', 'READING'].map(h => (
              <th key={h} style={{ fontSize: 7, color: '#3a4558', textAlign: 'left', padding: '3px 5px', letterSpacing: '0.08em', fontWeight: 700 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(signals || {}).map(([key, sig], i) => {
            const isOff  = disabled[key];
            const isOpen = expandedRow === key && !isOff;
            const isBullSig = sig.value >= 0.5;
            const pct = sig.value * 100;

            return (
              <React.Fragment key={key}>
                <tr
                  onClick={() => toggleRow(key)}
                  style={{
                    borderBottom: '1px solid rgba(255,255,255,0.03)',
                    opacity: isOff ? 0.35 : 1,
                    transition: 'opacity 0.2s',
                    cursor: 'pointer',
                    background: isOpen ? 'rgba(0,210,255,0.04)' : 'transparent',
                  }}
                  className="ens-row"
                >
                  <td style={{ ...mono, fontSize: 10, color: isOff ? '#3a4558' : '#d8e0ec', padding: '7px 5px', textDecoration: isOff ? 'line-through' : 'none' }}>
                    {isOff ? '✗ ' : ''}{SIG_LABELS[key] || key}
                    <span style={{ color: '#5a6a82', fontSize: 9, marginLeft: 4 }}>{isOpen ? '▲' : '▼'}</span>
                  </td>
                  <td style={{ ...mono, fontSize: 10, color: '#00d2ff', padding: '7px 5px' }}>{SIG_WEIGHTS[key] || sig.weight * 100 || '—'}%</td>
                  <td style={{ ...mono, fontSize: 9, color: isBullSig ? '#00e87a' : '#ff3b6b', padding: '7px 5px' }}>
                    {pct.toFixed(0)}%
                  </td>
                </tr>

                {/* Expanded detail */}
                {isOpen && (
                  <tr>
                    <td colSpan={3} style={{ padding: 0 }}>
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        style={{ overflow: 'hidden' }}
                      >
                        <div style={{
                          padding: '9px 10px',
                          background: 'rgba(245,166,35,0.04)',
                          border: '1px solid rgba(245,166,35,0.15)',
                          borderRadius: 5, margin: '4px 0 6px',
                        }}>
                          <div style={{ ...mono, fontSize: 9, color: '#d8e0ec', lineHeight: 1.75 }}>
                            {SIG_DETAIL[key] || `Signal value: ${pct.toFixed(1)}%, weight: ${SIG_WEIGHTS[key]}%.`}
                          </div>
                          {/* Mini bar */}
                          <div style={{ marginTop: 8 }}>
                            <div style={{ height: 4, background: 'rgba(255,255,255,0.04)', borderRadius: 2, overflow: 'hidden' }}>
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${pct}%` }}
                                transition={{ duration: 0.8 }}
                                style={{
                                  height: '100%', borderRadius: 2,
                                  background: isBullSig
                                    ? 'linear-gradient(90deg,rgba(0,232,122,0.5),#00e87a)'
                                    : 'linear-gradient(90deg,rgba(255,59,107,0.5),#ff3b6b)',
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>

      {/* Composite score footer */}
      <div style={{
        marginTop: 4, padding: '8px 12px', borderRadius: 7,
        background: is_high_confidence ? 'rgba(0,232,122,0.06)' : 'rgba(245,166,35,0.06)',
        border: `1px solid ${tierLabel.color}22`,
      }}>
        <div style={{ ...mono, fontSize: 9, color: tierLabel.color }}>{tierLabel.text}</div>
      </div>

      <style>{`.ens-row:hover { background: rgba(255,255,255,0.03) !important; }`}</style>
    </motion.div>
  );
};

export default EnsemblePanel;