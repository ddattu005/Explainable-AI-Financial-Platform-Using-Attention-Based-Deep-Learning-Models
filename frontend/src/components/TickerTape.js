import React, { useState, useEffect } from 'react';

const FALLBACK_TICKERS = [
  { s: 'AAPL',  p: null, c: null },
  { s: 'MSFT',  p: null, c: null },
  { s: 'NVDA',  p: null, c: null },
  { s: 'TSLA',  p: null, c: null },
  { s: 'GOOGL', p: null, c: null },
  { s: 'META',  p: null, c: null },
  { s: 'AMZN',  p: null, c: null },
  { s: 'JPM',   p: null, c: null },
];

const TickerTape = () => {
  const [tickers, setTickers] = useState(FALLBACK_TICKERS);

  useEffect(() => {
    const fetchMarket = async () => {
      try {
        const BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        const res = await fetch(`${BASE}/ticker-tape`);
        if (res.ok) {
          const data = await res.json();
          if (data.tickers?.length) {
            setTickers(data.tickers);
          }
        }
      } catch {
        // Use fallback — will show dashes
      }
    };
    fetchMarket();
    // Refresh every 8 seconds
    const interval = setInterval(fetchMarket, 8000);
    return () => clearInterval(interval);
  }, []);

  const items = [...tickers, ...tickers]; // duplicate for seamless loop
  return (
    <div className="ticker-tape">
      <div className="ticker-scroll">
        {items.map((t, i) => (
          <span key={i} className="t-item">
            <span className="t-sym">{t.s}</span>
            <span className="t-px">
              {t.p != null ? `$${t.p.toLocaleString()}` : '—'}
            </span>
            {t.c != null && (
              <span className={t.c >= 0 ? 't-up' : 't-dn'}>
                {t.c >= 0 ? '▲' : '▼'}{Math.abs(t.c).toFixed(2)}%
              </span>
            )}
            <span className="t-sep">│</span>
          </span>
        ))}
      </div>
    </div>
  );
};

export default TickerTape;