import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

function LiveClock() {
  const [t, setT] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setT(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: 'var(--dim)' }}>
      {t.toLocaleTimeString('en-US', { hour12: false })} EST
    </span>
  );
}

const Header = ({ apiHealth, metrics, children }) => {
  const status = !apiHealth
    ? 'checking'
    : apiHealth.model_loaded
    ? 'ready'
    : 'offline';

  const labels = { ready: 'Model Ready', offline: 'Model Offline', checking: 'Connecting…' };

  const m = metrics?.metrics || {};
  const params = metrics?.model_params || 0;
  const paramsLabel = params > 0 ? `${Math.round(params / 1000)}K` : '368K';
  const r2Label = m.r2 ? `R² ${m.r2.toFixed(3)}` : 'R² 0.219';

  return (
    <header className="header">
      <div className="header-content">
        <motion.div
          className="header-left"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.45 }}
        >
          {/* Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div>
              <div className="logo-name">XAI Financial Platform</div>
            </div>
          </div>

          {/* Tab bar injected from App.js */}
          {children}
        </motion.div>

        <motion.div
          className="header-right"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.45, delay: 0.15 }}
        >
          <LiveClock />
          <div className="hdr-stat">
            <span className="sv">{paramsLabel}</span>
            <span className="sl">Parameters</span>
          </div>
          <div className="hdr-stat">
            <span className="sv">{r2Label}</span>
            <span className="sl">Model Score</span>
          </div>
          <div className="hdr-stat">
            <span className="sv">60-day</span>
            <span className="sl">Lookback</span>
          </div>
          <div className={`health-badge ${status}`}>
            <div className="h-dot" />
            <span>{labels[status]}</span>
          </div>
        </motion.div>
      </div>
    </header>
  );
};

export default Header;