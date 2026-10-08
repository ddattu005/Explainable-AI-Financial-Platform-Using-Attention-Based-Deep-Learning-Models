import React from 'react';
import { motion } from 'framer-motion';

// Full 50-stock universe — matches backend STOCK_UNIVERSE
const STOCKS_BY_SECTOR = {
  Technology:    ['AAPL','MSFT','GOOGL','META','CRM','ADBE','ORCL'],
  Semiconductors:['NVDA','AMD','INTC'],
  Consumer:      ['AMZN','COST','NKE','SBUX','WMT','PG','KO'],
  Automotive:    ['TSLA'],
  Media:         ['NFLX','DIS'],
  Finance:       ['JPM','GS','MS','BAC','BLK','V','MA'],
  Healthcare:    ['JNJ','UNH','PFE','ABBV','MRK'],
  Energy:        ['XOM','CVX','COP','SLB'],
  Industrial:    ['BA','CAT','HON','GE'],
  Telecom:       ['T','VZ','TMUS'],
  Utilities:     ['NEE'],
  'Real Estate': ['AMT','PLD'],
  ETF:           ['SPY','QQQ','DIA','IWM'],
};

const ALL_STOCKS = Object.entries(STOCKS_BY_SECTOR).flatMap(([sector, tickers]) =>
  tickers.map(t => ({ ticker: t, sector }))
);

const StockSelector = ({ selectedStock, onStockChange, onPredict, loading, supportedStocks }) => {
  // Use backend-provided list if available, else use our 50-stock default
  const stocks = supportedStocks?.length ? supportedStocks : ALL_STOCKS.map(s => ({ ticker: s.ticker, name: s.ticker }));

  // Group options by sector for display
  const grouped = supportedStocks?.length
    ? null
    : STOCKS_BY_SECTOR;

  return (
    <div className="selector-row">
      {/* Main dropdown */}
      <div className="sel-main-card">
        <div className="sel-lbl">Select Stock — {stocks.length} stocks available</div>
        <div className="sel-wrap">
          <select
            className="stock-select"
            value={selectedStock}
            onChange={e => onStockChange(e.target.value)}
            disabled={loading}
          >
            {grouped ? (
              Object.entries(grouped).map(([sector, tickers]) => (
                <optgroup key={sector} label={`── ${sector} ──`}>
                  {tickers.map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </optgroup>
              ))
            ) : (
              stocks.map(s => (
                <option key={s.ticker} value={s.ticker}>
                  {s.ticker}{s.name && s.name !== s.ticker ? ` — ${s.name}` : ''}
                </option>
              ))
            )}
          </select>
          <span className="sel-arrow">▾</span>
        </div>
      </div>

      {/* Predict button */}
      <motion.button
        className="predict-button"
        onClick={onPredict}
        disabled={loading}
        whileHover={{ scale: loading ? 1 : 1.04 }}
        whileTap={{ scale: loading ? 1 : 0.96 }}
      >
        {loading
          ? <><span className="spin-sm" /><span style={{ animation: 'pulse 1.2s infinite' }}>⟳ SCANNING…</span></>
          : <><span>⚡</span><span>PREDICT</span></>
        }
      </motion.button>
    </div>
  );
};

export default StockSelector;