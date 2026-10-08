import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';

const mono = { fontFamily: "'JetBrains Mono', monospace" };

const AttnTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(5,7,13,0.97)', border: '1px solid rgba(155,109,255,0.3)',
      borderRadius: 6, padding: '8px 12px', ...mono, fontSize: 10,
    }}>
      <div style={{ color: '#5a6a82', marginBottom: 3 }}>T-{label}</div>
      <div style={{ color: '#9b6dff' }}>Attention: <strong>{payload[0]?.value?.toFixed(4)}</strong></div>
    </div>
  );
};

const AttentionHeatmap = ({ days }) => {
  const [hoveredIdx, setHoveredIdx] = useState(null);

  if (!days?.length) return null;

  const maxImp  = Math.max(...days.map(d => d.importance), 1e-9);
  const minImp  = Math.min(...days.map(d => d.importance));
  const totalImp = days.reduce((s, d) => s + d.importance, 0);

  // Entropy / focus score
  const entropy = days.reduce((s, d) => {
    const p = d.importance / (totalImp + 1e-10);
    return s - (p > 0 ? p * Math.log2(p + 1e-10) : 0);
  }, 0);
  const maxEntropy = Math.log2(days.length);
  const focusScore = Math.max(0, (1 - entropy / maxEntropy) * 100);
  const range      = maxImp - minImp;
  const isUniform  = range < maxImp * 0.05 || focusScore < 5;

  // Build chart data — reverse so oldest is left
  const chartData = [...days].reverse().map(d => ({
    lag:   d.days_ago,
    score: d.importance,
  }));

  return (
    <motion.div
      className="card attention-heatmap-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.18 }}
    >
      {/* Header */}
      <div className="card-title">
        <div className="card-ico ico-p">👁️</div>
        <span className="card-title-text">Temporal Attention Weights</span>
        <span style={{
          marginLeft: 'auto', ...mono, fontSize: 9, fontWeight: 700,
          color: isUniform ? '#f5a623' : focusScore > 30 ? '#00e87a' : '#5a6a82',
          letterSpacing: '0.06em',
        }}>
          60-DAY WINDOW
        </span>
      </div>

      {/* Uniform warning */}
      {isUniform && (
        <div style={{
          padding: '7px 10px', marginBottom: 10, borderRadius: 5,
          background: 'rgba(245,166,35,0.06)', border: '1px solid rgba(245,166,35,0.2)',
          ...mono, fontSize: 9, color: '#f5a623', lineHeight: 1.6, display: 'flex', gap: 7,
        }}>
          <span>⚠️</span>
          <div>
            <strong>Near-uniform attention</strong> — hover rows to inspect.{' '}
            FOCUS: {focusScore.toFixed(0)}%. Retraining required.
          </div>
        </div>
      )}

      {/* Mini area chart */}
      <div style={{ marginBottom: 12 }}>
        <ResponsiveContainer width="100%" height={90}>
          <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -30, bottom: 0 }}>
            <defs>
              <linearGradient id="attnGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#9b6dff" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#9b6dff" stopOpacity={0}   />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="lag" tick={{ fill: '#5a6a82', fontSize: 7, fontFamily: "'JetBrains Mono',monospace" }}
              tickFormatter={v => `T-${v}`} tickLine={false} axisLine={false}
            />
            <YAxis hide domain={[minImp * 0.98, maxImp * 1.02]} />
            <Tooltip content={<AttnTooltip />} />
            <Area
              type="monotone" dataKey="score"
              stroke="#9b6dff" strokeWidth={1.5}
              fill="url(#attnGrad)"
              dot={{ r: 2, fill: '#9b6dff' }}
              activeDot={{ r: 4, fill: '#c4b5fd' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Table header */}
      <div style={{
        display: 'grid', gridTemplateColumns: '28px 90px 1fr 60px',
        gap: 8, marginBottom: 8, paddingBottom: 7,
        borderBottom: '1px solid rgba(0,210,255,0.07)',
      }}>
        {['#', 'LAG', 'WEIGHT', 'SCORE'].map(h => (
          <span key={h} style={{ ...mono, fontSize: 7, color: '#3a4558', fontWeight: 700 }}>{h}</span>
        ))}
      </div>

      {/* Rows */}
      <div className="day-list" style={{ maxHeight: 220, overflowY: 'auto' }}>
        {days.map((day, i) => {
          const pct    = (day.importance / maxImp) * 100;
          const label  = day.days_ago === 0 ? 'Today' : `T-${day.days_ago}`;
          const isHov  = hoveredIdx === i;

          return (
            <motion.div
              key={day.day_index}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.18 + i * 0.06 }}
              style={{
                display: 'grid', gridTemplateColumns: '28px 90px 1fr 60px',
                gap: 8, alignItems: 'center', marginBottom: 8,
                padding: '4px 5px', borderRadius: 5, cursor: 'default',
                background: isHov ? 'rgba(155,109,255,0.07)' : 'transparent',
                transition: 'background 0.15s',
              }}
            >
              <span style={{ ...mono, fontSize: 10, color: i < 3 ? '#c4b5fd' : '#00d2ff', fontWeight: 700 }}>
                {i < 3 ? '◉' : `#${i + 1}`}
              </span>

              <div>
                <div style={{ ...mono, fontSize: 11, color: isHov ? '#c4b5fd' : '#d8e0ec' }}>{label}</div>
                {isHov && (
                  <div style={{ ...mono, fontSize: 8, color: '#5a6a82' }}>{day.days_ago} days back</div>
                )}
              </div>

              <div style={{
                height: isHov ? 8 : 5,
                background: 'rgba(255,255,255,0.04)',
                borderRadius: 3, overflow: 'hidden', transition: 'height 0.2s',
              }}>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.9, delay: 0.25 + i * 0.07 }}
                  style={{
                    height: '100%', borderRadius: 3,
                    background: isHov
                      ? '#c4b5fd'
                      : i < 3
                        ? 'linear-gradient(90deg,#9b6dff,#c4b5fd)'
                        : 'linear-gradient(90deg,#7c3aed,#00d2ff)',
                    transition: 'background 0.2s',
                  }}
                />
              </div>

              <span style={{ ...mono, fontSize: 11, color: isHov ? '#c4b5fd' : '#9b6dff', fontWeight: 600 }}>
                {day.importance.toFixed(4)}
              </span>
            </motion.div>
          );
        })}
      </div>

      <div className="tip-box" style={{ marginTop: 12 }}>
        💡 Weights from highest-variance attention head, re-sharpened (τ=0.5). Hover rows to inspect.
        Uniform scores across window = attention regularization needed.
      </div>
    </motion.div>
  );
};

export default AttentionHeatmap;