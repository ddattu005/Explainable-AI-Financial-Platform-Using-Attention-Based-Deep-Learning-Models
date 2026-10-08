import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const mono = { fontFamily: "'JetBrains Mono', monospace" };

const SENTIMENT_COLORS = {
  bullish:  { bg: 'rgba(0,232,122,0.08)', border: 'rgba(0,232,122,0.2)', text: '#00e87a', label: '↗ BULLISH' },
  bearish:  { bg: 'rgba(255,59,107,0.08)', border: 'rgba(255,59,107,0.2)', text: '#ff3b6b', label: '↘ BEARISH' },
  neutral:  { bg: 'rgba(90,106,130,0.08)', border: 'rgba(90,106,130,0.2)', text: '#5a6a82', label: '— NEUTRAL' },
};

const NewsPanel = ({ ticker }) => {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    setError(null);

    const BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    fetch(`${BASE}/news/${ticker}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch');
        return res.json();
      })
      .then(data => {
        setArticles(data.articles || []);
        setLoading(false);
      })
      .catch(err => {
        console.warn('News fetch failed:', err);
        setError('Unable to load news');
        setLoading(false);
      });
  }, [ticker]);

  if (!ticker) return null;

  return (
    <motion.div
      className="card"
      style={{ padding: '16px 20px' }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.25 }}
    >
      {/* Header */}
      <div className="card-title" style={{ marginBottom: 12 }}>
        <div className="card-ico ico-c">📰</div>
        <span className="card-title-text">Market News</span>
        <span style={{ marginLeft: 'auto', ...mono, fontSize: 9, color: '#5a6a82' }}>
          {ticker} · {articles.length} articles
        </span>
      </div>

      {/* Loading */}
      {loading && (
        <div style={{ ...mono, fontSize: 10, color: '#5a6a82', textAlign: 'center', padding: '24px 0' }}>
          ⟳ Loading news for {ticker}…
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ ...mono, fontSize: 10, color: '#f5a623', textAlign: 'center', padding: '24px 0' }}>
          ⚠ {error}
        </div>
      )}

      {/* Articles */}
      <AnimatePresence>
        {!loading && articles.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {articles.slice(0, 8).map((article, i) => {
              const sentiment = SENTIMENT_COLORS[article.sentiment] || SENTIMENT_COLORS.neutral;
              const timeAgo = article.publishedDate
                ? formatTimeAgo(article.publishedDate)
                : '';

              return (
                <motion.a
                  key={i}
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.05 * i }}
                  style={{
                    display: 'block', padding: '10px 12px', borderRadius: 6,
                    background: sentiment.bg,
                    border: `1px solid ${sentiment.border}`,
                    textDecoration: 'none', cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                  className="news-article-row"
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ ...mono, fontSize: 11, color: '#d8e0ec', fontWeight: 600, lineHeight: 1.4, marginBottom: 4 }}>
                        {article.title}
                      </div>
                      <div style={{ ...mono, fontSize: 9, color: '#5a6a82', lineHeight: 1.4 }}>
                        {article.site || 'Unknown'} {timeAgo && `· ${timeAgo}`}
                      </div>
                    </div>
                    <span style={{
                      ...mono, fontSize: 8, fontWeight: 700,
                      color: sentiment.text,
                      background: sentiment.bg,
                      border: `1px solid ${sentiment.border}`,
                      borderRadius: 3, padding: '2px 6px',
                      whiteSpace: 'nowrap', flexShrink: 0,
                    }}>
                      {sentiment.label}
                    </span>
                  </div>
                </motion.a>
              );
            })}
          </div>
        )}
      </AnimatePresence>

      {/* Empty state */}
      {!loading && !error && articles.length === 0 && (
        <div style={{ ...mono, fontSize: 10, color: '#3a4558', textAlign: 'center', padding: '24px 0' }}>
          No recent news for {ticker}
        </div>
      )}

      <style>{`
        .news-article-row:hover {
          transform: translateX(3px);
          box-shadow: 0 2px 12px rgba(0,0,0,0.2);
        }
      `}</style>
    </motion.div>
  );
};

function formatTimeAgo(dateStr) {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now - date;
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

export default NewsPanel;
