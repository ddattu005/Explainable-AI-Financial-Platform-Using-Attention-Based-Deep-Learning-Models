import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const PredictionCard = ({ prediction }) => {
  const {
    ticker, timestamp, current_price, predicted_price,
    direction, change_percent, confidence, regime,
    confidence_tier, top_features
  } = prediction;

  const isUp = direction?.includes('BULLISH') || direction?.includes('UP');
  const [confWidth, setConfWidth] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => setConfWidth(confidence * 100), 200);
    return () => clearTimeout(t);
  }, [confidence]);

  const fmt = iso => new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
  });

  const regimeCls  = `regime-${(regime?.regime_name || 'trending').toLowerCase()}`;

  // Detect SHAP-direction divergence
  // All SHAP features bullish but direction bearish (or vice versa)
  const shapBullish = top_features?.filter(f => f.impact > 0).length || 0;
  const shapBearish = top_features?.filter(f => f.impact < 0).length || 0;
  const shapNet = shapBullish > shapBearish ? 'bullish' : 'bearish';
  const hasDivergence = (shapNet === 'bullish' && !isUp) || (shapNet === 'bearish' && isUp);

  // Confidence tier badge styles
  const tierStyles = {
    HIGH_CONF: { bg: 'rgba(0,232,122,0.1)',  border: 'rgba(0,232,122,0.4)', color: '#00e87a', label: '⬡ HIGH CONFIDENCE' },
    MODERATE:  { bg: 'rgba(245,166,35,0.1)', border: 'rgba(245,166,35,0.4)', color: '#f5a623', label: '◈ MODERATE' },
  };
  const tier = tierStyles[confidence_tier] || tierStyles['MODERATE'];

  // Direction badge: always show direction. SHAP divergence adds a sub-label.
  const getBadge = () => {
    return (
      <div className={`direction-badge ${isUp ? 'up' : 'down'}`}>
        <span className="dir-arrow">{isUp ? '↑' : '↓'}</span>
        <span>{isUp ? 'BULLISH' : 'BEARISH'}</span>
        {hasDivergence && (
          <span style={{
            display: 'block', fontSize: 9, marginTop: 2,
            color: 'var(--amber)', fontFamily: 'JetBrains Mono,monospace'
          }}></span>
        )}
      </div>
    );
  };

  // Confidence threshold positions (for visual markers)
  const thresholds = [
    { pos: 42, label: 'BEAR' },
    { pos: 47, label: '' },
    { pos: 53, label: '' },
    { pos: 58, label: 'BULL' },
  ];

  return (
    <motion.div className="prediction-card"
      initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      {/* Top row */}
      <div className="pred-top">
        <div>
          <div className="prediction-ticker">{ticker}</div>
          <div className="pred-company" style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 4 }}>
            <span style={{ fontSize: 12, color: 'var(--dim)' }}>{regime?.regime_name || ''} regime</span>
          </div>
          <div className="prediction-timestamp">{fmt(timestamp)}</div>
        </div>
        {getBadge()}
      </div>

      {/* Price boxes */}
      <div className="prediction-prices">
        {[
          { label: 'Current Price',    val: `$${current_price.toFixed(2)}`,  cls: 'current' },
          { label: 'Predicted Price',  val: `$${predicted_price.toFixed(2)}`,cls: 'predicted' },
          { label: 'Expected Change',  val: `${change_percent >= 0 ? '+' : ''}${change_percent.toFixed(2)}%`, cls: isUp ? 'up' : 'down' },
          { label: 'Market Regime',    val: regime?.regime_name || '—',       cls: regimeCls },
        ].map((b, i) => (
          <motion.div key={i} className="price-box"
            initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.08 * i }}
          >
            <div className="price-label">{b.label}</div>
            <div className={`price-value ${b.cls}`}>{b.val}</div>
          </motion.div>
        ))}
      </div>

      {/* Confidence bar with threshold markers */}
      <div>
        <div className="conf-header">
          <span className="conf-label">Model Confidence</span>
          <span className="conf-value">{(confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="confidence-progress" style={{ position: 'relative' }}>
          <div className="confidence-fill" style={{ width: `${confWidth}%` }} />
          {/* Threshold markers */}
          {thresholds.map((t, i) => (
            <div key={i} style={{
              position: 'absolute', left: `${t.pos}%`, top: -2,
              width: 1, height: 'calc(100% + 4px)',
              background: 'rgba(255,255,255,0.2)', zIndex: 2
            }}>
              {t.label && <span style={{
                position: 'absolute', top: -14, left: -10,
                fontSize: 7, color: 'var(--dim)',
                fontFamily: 'JetBrains Mono,monospace',
                letterSpacing: '.04em', whiteSpace: 'nowrap'
              }}>{t.label}</span>}
            </div>
          ))}
        </div>
      </div>

      {/* SHAP divergence warning banner */}
      {hasDivergence && (
        <div style={{
          marginTop: 10, padding: '6px 12px', borderRadius: 6,
          background: 'rgba(245,166,35,0.08)', border: '1px solid rgba(245,166,35,0.25)',
          fontSize: 10, color: 'var(--amber)', fontFamily: 'JetBrains Mono,monospace',
          lineHeight: 1.5
        }}>
          ⚠️ SHAP features are predominantly {shapNet} ({shapBullish}/{(top_features?.length||0)} positive),
          but ensemble direction is {isUp ? 'UP' : 'DOWN'}. SHAP explains the price regression head;
          ensemble incorporates RSI + MACD + MA signals.
        </div>
      )}
    </motion.div>
  );
};

export default PredictionCard;