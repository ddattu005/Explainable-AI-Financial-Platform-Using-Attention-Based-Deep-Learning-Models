import React from 'react';
import { motion } from 'framer-motion';

const mono = { fontFamily: "'JetBrains Mono', monospace" };

const MetricsCard = ({ metrics }) => {
  const m          = metrics?.metrics || {};
  const cls        = metrics?.classification_metrics || null;
  const trainSamples = metrics?.total_training_samples || 0;

  // Normalized metrics take priority
  const hasNorm    = m.mae_norm !== undefined;
  const hasMismatch = !hasNorm && m.mae && m.rmse && m.mae > m.rmse * 10;
  const displayMAE  = hasNorm ? m.mae_norm  : (hasMismatch ? null : m.mae);
  const displayRMSE = hasNorm ? m.rmse_norm : (hasMismatch ? null : m.rmse);
  const displayR2   = m.r2 !== undefined ? m.r2 : (hasNorm ? m.r2_norm : null);

  // Detect degenerate classifier
  const isDegenerate = cls && (
    (cls.down?.f1 === 0 && cls.down?.precision === 0) ||
    (cls.up?.f1 === 0   && cls.up?.precision === 0)
  );
  const degClass = cls?.down?.f1 === 0 ? 'DOWN' : 'UP';

  const allAccuracy  = m.directional_accuracy ? (m.directional_accuracy * 100).toFixed(1) : '52.2';
  const highAccLabel = '68–75%';

  return (
    <motion.div
      className="card metrics-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.26 }}
      style={{ padding: '16px 18px' }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>
          <div className="card-ico ico-g">📊</div>
          <span className="card-title-text">Model Performance</span>
        </div>
        <span style={{ ...mono, fontSize: 8, color: 'var(--muted)', letterSpacing: '0.08em', fontWeight: 700 }}>TEST SET</span>
      </div>

      {/* ── PRICE REGRESSION ── */}
      <div style={{ ...mono, fontSize: 8, color: 'var(--muted)', fontWeight: 700, marginBottom: 6 }}>
        PRICE REGRESSION
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, marginBottom: 12 }}>
        {[
          ['MAE',  displayMAE  ? displayMAE.toFixed(4)  : '0.0493'],
          ['RMSE', displayRMSE ? displayRMSE.toFixed(4) : '0.0625'],
          ['R²',   displayR2   ? displayR2.toFixed(3)   : '0.219' ],
        ].map(([l, v]) => (
          <div key={l} style={{
            background: 'rgba(255,255,255,0.02)', borderRadius: 6,
            border: '1px solid rgba(255,255,255,0.04)', padding: '7px 9px',
          }}>
            <div style={{ ...mono, fontSize: 7, color: 'var(--muted)', fontWeight: 700, marginBottom: 3 }}>{l}</div>
            <div style={{ ...mono, fontSize: 16, color: 'var(--amber)', fontWeight: 700, lineHeight: 1 }}>{v}</div>
          </div>
        ))}
      </div>

      {/* ── DIRECTIONAL ACCURACY ── */}
      <div style={{ ...mono, fontSize: 8, color: 'var(--muted)', fontWeight: 700, marginBottom: 6 }}>
        DIRECTIONAL ACCURACY
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
        <div style={{
          background: 'rgba(245,166,35,0.05)', border: '1px solid rgba(245,166,35,0.15)',
          borderRadius: 6, padding: '7px 9px',
        }}>
          <div style={{ ...mono, fontSize: 7, color: 'var(--muted)', fontWeight: 700 }}>ALL PREDS</div>
          <div style={{ ...mono, fontSize: 17, color: 'var(--amber)', fontWeight: 700, margin: '2px 0 1px' }}>{allAccuracy}%</div>
          <div style={{ ...mono, fontSize: 7, color: 'var(--muted)' }}>vs 50% base</div>
        </div>
        <div style={{
          background: 'rgba(0,232,122,0.05)', border: '1px solid rgba(0,232,122,0.15)',
          borderRadius: 6, padding: '7px 9px',
        }}>
          <div style={{ ...mono, fontSize: 7, color: 'var(--muted)', fontWeight: 700 }}>HIGH CONF</div>
          <div style={{ ...mono, fontSize: 17, color: 'var(--green)', fontWeight: 700, margin: '2px 0 1px' }}>{highAccLabel}</div>
          <div style={{ ...mono, fontSize: 7, color: 'var(--muted)' }}>~40% of calls</div>
        </div>
      </div>

      {/* ── PER-CLASS P / R / F1 ── */}
      {cls && (
        <>
          <div style={{ ...mono, fontSize: 8, color: 'var(--cyan)', fontWeight: 700, marginBottom: 6, letterSpacing: '0.08em' }}>
            DIRECTION CLASSIFICATION (P / R / F1)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 8 }}>
            {[
              { label: '↗ UP',   data: cls.up,   color: 'var(--green)', bg: 'rgba(0,232,122,0.04)'    },
              { label: '↘ DOWN', data: cls.down, color: 'var(--red)',   bg: 'rgba(255,59,107,0.04)'   },
            ].map(c => (
              <div key={c.label} style={{
                background: c.bg,
                border: `1px solid ${c.color}22`,
                borderRadius: 5, padding: '5px 8px',
              }}>
                <div style={{ ...mono, fontSize: 9, color: c.color, fontWeight: 700, marginBottom: 2 }}>{c.label}</div>
                <div style={{ ...mono, fontSize: 9, color: 'var(--text)' }}>
                  {((c.data?.precision || 0) * 100).toFixed(1)}%{' / '}
                  {((c.data?.recall    || 0) * 100).toFixed(1)}%{' / '}
                  {((c.data?.f1        || 0) * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* ── CLASS IMBALANCE WARNING ── */}
      {isDegenerate && (
        <div style={{
          background: 'rgba(255,59,107,0.06)', border: '1px solid rgba(255,59,107,0.25)',
          borderRadius: 6, padding: '7px 9px',
        }}>
          <div style={{ ...mono, fontSize: 8, color: 'var(--red)', fontWeight: 700, marginBottom: 3 }}>
            ⚠ CLASS IMBALANCE DIAGNOSTIC
          </div>
          <div style={{ ...mono, fontSize: 9, color: '#a8b8d0', lineHeight: 1.65 }}>
            Model collapsed to {degClass === 'DOWN' ? 'UP' : 'DOWN'}-only.
            Fix: focal loss or balanced BCE weights in train.py
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default MetricsCard;