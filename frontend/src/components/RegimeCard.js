import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

/* ── Animated confidence ring (matches preview exactly) ── */
function ConfRing({ value, label, size = 90 }) {
  const [dash, setDash] = useState(0);
  const r    = size / 2 - 6;
  const circ = 2 * Math.PI * r;
  const color = value >= 58 ? '#00e87a'
              : value <= 42 ? '#ff3b6b'
              : value >= 53 ? '#7cc674'
              : value <= 47 ? '#e0854c'
              : '#5a6a82';

  useEffect(() => {
    const t = setTimeout(() => setDash(circ * (value / 100)), 200);
    return () => clearTimeout(t);
  }, [value, circ]);

  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={6}
        />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke={color} strokeWidth={6}
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
        <span style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: size > 80 ? 15 : 12, color, fontWeight: 700,
        }}>
          {value.toFixed(1)}%
        </span>
        <span style={{
          fontSize: 7, color: '#5a6a82',
          letterSpacing: '0.12em', fontWeight: 700, marginTop: 2,
        }}>
          {label}
        </span>
      </div>
    </div>
  );
}

const DESCS = {
  Calm:     'Low volatility — stable conditions. Confidence boosted +5%.',
  Trending: 'Directional momentum detected. Confidence unchanged.',
  Volatile: 'High uncertainty. Confidence reduced −15% for safety.',
};

const REGIME_COLORS = {
  Calm:     '#7cc674',
  Trending: '#00d2ff',
  Volatile: '#ff3b6b',
};

const RegimeCard = ({ regime }) => {
  if (!regime) return null;

  const name    = regime.regime_name || 'Trending';
  const adjConf = (regime.adjusted_model_confidence || 0.7) * 100;

  return (
    <motion.div
      className="card regime-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.18 }}
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 18px' }}
    >
      <div className="card-title" style={{ width: '100%' }}>
        <div className="card-ico ico-a">🌐</div>
        <span className="card-title-text">Market Regime</span>
      </div>

      {/* Animated ring */}
      <ConfRing value={adjConf} label="ADJ. CONF" size={90} />

      {/* Regime name */}
      <div style={{
        fontFamily: "'Orbitron', monospace",
        fontSize: 20, fontWeight: 700,
        letterSpacing: '0.06em',
        color: REGIME_COLORS[name] || '#00d2ff',
        textShadow: `0 0 20px ${REGIME_COLORS[name] || '#00d2ff'}66`,
        margin: '10px 0 6px',
        textAlign: 'center',
      }}>
        {name.toUpperCase()}
      </div>

      {/* Description */}
      <div style={{
        fontSize: 11, color: 'var(--dim)',
        textAlign: 'center', maxWidth: 180,
        lineHeight: 1.65, marginBottom: 10,
      }}>
        {DESCS[name] || ''}
      </div>

      {/* Calm / Trending / Volatile selector pills */}
      <div style={{ display: 'flex', gap: 7 }}>
        {['Calm', 'Trending', 'Volatile'].map(r => {
          const isActive = r === name;
          const col      = REGIME_COLORS[r];
          return (
            <div key={r} style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 9, fontWeight: 700,
              color: isActive ? col : 'var(--muted)',
              background: isActive ? `${col}14` : 'transparent',
              border: `1px solid ${isActive ? `${col}55` : 'rgba(255,255,255,0.04)'}`,
              borderRadius: 4, padding: '2px 9px',
              transition: 'all 0.25s',
            }}>
              {r}
            </div>
          );
        })}
      </div>
    </motion.div>
  );
};

export default RegimeCard;