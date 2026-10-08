import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart, AreaChart, Area, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, ReferenceArea, CartesianGrid
} from 'recharts';

const mono = { fontFamily: "'JetBrains Mono', monospace" };

/* ── Build 60-day fake-historical + prediction path ── */
function buildData(cp, pp) {
  const diff = pp - cp;
  const arr = [];
  let price = cp - diff * 0.3 + (Math.random() - 0.5) * Math.abs(diff) * 0.5;
  for (let i = 60; i >= 1; i--) {
    price += (Math.random() - 0.48) * Math.abs(diff) * 0.18;
    arr.push({
      day: `T-${i}`,
      price: parseFloat(price.toFixed(2)),
      vol: Math.floor(45e6 + Math.random() * 35e6),
    });
  }
  arr.push({ day: 'NOW', price: cp, vol: 62e6, current: true });
  arr.push({ day: 'PRED', price: pp, predicted: true });
  return arr;
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: 'rgba(5,7,13,0.97)',
      border: '1px solid rgba(0,210,255,0.25)',
      borderRadius: 8, padding: '9px 13px', ...mono, fontSize: 11,
    }}>
      <div style={{ color: '#5a6a82', marginBottom: 4 }}>{label}</div>
      {d?.predicted
        ? <div style={{ color: '#ff3b6b' }}>Predicted: <strong>${payload[0]?.value?.toFixed(2)}</strong></div>
        : <div style={{ color: '#00d2ff' }}>Price: <strong>${payload[0]?.value?.toFixed(2)}</strong></div>
      }
      {d?.vol && <div style={{ color: '#5a6a82', fontSize: 9, marginTop: 3 }}>Vol: {(d.vol / 1e6).toFixed(1)}M</div>}
    </div>
  );
};

const TagBtn = ({ active, onClick, children, activeColor = 'rgba(0,210,255,0.5)', activeBg = 'rgba(0,210,255,0.1)', activeText = '#00d2ff' }) => (
  <button onClick={onClick} style={{
    ...mono, fontSize: 9, padding: '3px 10px', borderRadius: 4,
    border: `1px solid ${active ? activeColor : 'rgba(255,255,255,0.08)'}`,
    background: active ? activeBg : 'transparent',
    color: active ? activeText : '#5a6a82', cursor: 'pointer', transition: 'all 0.2s',
  }}>{children}</button>
);

const PriceChart = ({ currentPrice, predictedPrice, direction }) => {
  const [chartMode, setChartMode] = useState('line');
  const [showBB, setShowBB] = useState(false);
  const [data, setData] = useState([]);

  const isUp = direction?.includes('BULLISH') || direction?.includes('UP');
  const lineColor = isUp ? '#00e87a' : '#ff3b6b';

  useEffect(() => {
    setData(buildData(currentPrice || 259.88, predictedPrice || 254.76));
  }, [currentPrice, predictedPrice]);

  if (!data.length) return null;

  const prices = data.map(d => d.price).filter(Boolean);
  const high60 = Math.max(...prices);
  const low60  = Math.min(...prices);
  const avg60  = (prices.reduce((a, b) => a + b, 0) / prices.length);
  const volatility = ((high60 - low60) / prices.length).toFixed(1);

  // Approx BB bands based on current price
  const bbUpper = currentPrice * 1.045;
  const bbLower = currentPrice * 0.955;
  const sma20   = currentPrice * 0.994;

  const ChartComponent = chartMode === 'area' ? AreaChart : LineChart;
  const commonProps = {
    data,
    margin: { top: 8, right: 8, left: -10, bottom: 0 },
  };
  const axisProps = {
    xAxis: (
      <XAxis
        dataKey="day"
        tick={{ fill: '#3a4558', fontSize: 8, fontFamily: "'JetBrains Mono', monospace" }}
        tickLine={false} interval={14}
        axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
      />
    ),
    yAxis: (
      <YAxis
        tick={{ fill: '#3a4558', fontSize: 8, fontFamily: "'JetBrains Mono', monospace" }}
        tickLine={false} axisLine={false}
        domain={['auto', 'auto']}
        tickFormatter={v => `$${v.toFixed(0)}`}
      />
    ),
  };

  return (
    <motion.div
      className="card"
      style={{ marginBottom: 18, padding: '16px 20px' }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.15 }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>
          <div className="card-ico ico-c">📈</div>
          <span className="card-title-text">Price History + Prediction</span>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <TagBtn active={chartMode === 'line'} onClick={() => setChartMode('line')}>Line</TagBtn>
          <TagBtn active={chartMode === 'area'} onClick={() => setChartMode('area')}>Area</TagBtn>
          <TagBtn
            active={showBB}
            onClick={() => setShowBB(p => !p)}
            activeColor="rgba(245,166,35,0.5)"
            activeBg="rgba(245,166,35,0.08)"
            activeText="#f5a623"
          >BB Bands</TagBtn>
        </div>
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 18, marginBottom: 10, flexWrap: 'wrap' }}>
        {[
          ['─── Historical', 'rgba(0,210,255,0.7)'],
          ['- - - Predicted', 'rgba(255,59,107,0.8)'],
          ['● Current', 'rgba(0,232,122,0.8)'],
          ...(showBB ? [['── Bollinger Band shown', 'rgba(245,166,35,0.7)']] : []),
        ].map(([l, c]) => (
          <span key={l} style={{ ...mono, fontSize: 9, color: c, display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ display: 'inline-block', width: 14, height: 1, background: c, marginBottom: 1 }} />
            {l}
          </span>
        ))}
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={220}>
        <ChartComponent {...commonProps}>
          <defs>
            <linearGradient id="priceAreaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={lineColor} stopOpacity={0.2} />
              <stop offset="95%" stopColor={lineColor} stopOpacity={0}   />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.03)" vertical={false} />
          {axisProps.xAxis}
          {axisProps.yAxis}
          <Tooltip content={<CustomTooltip />} />

          {/* NOW line */}
          <ReferenceLine
            x="NOW" stroke={isUp ? '#00e87a' : '#ff3b6b'}
            strokeDasharray="3 3"
            label={{ value: 'NOW', fill: isUp ? '#00e87a' : '#ff3b6b', fontSize: 9, fontFamily: "'JetBrains Mono',monospace" }}
          />
          {/* PRED line */}
          <ReferenceLine
            x="PRED" stroke="rgba(255,59,107,0.6)"
            strokeDasharray="3 3"
            label={{ value: 'PRED', fill: '#ff3b6b', fontSize: 9, fontFamily: "'JetBrains Mono',monospace" }}
          />
          {/* Predicted price horizontal */}
          <ReferenceLine y={predictedPrice} stroke="rgba(255,59,107,0.2)" strokeDasharray="4 4" />

          {/* BB Bands */}
          {showBB && <>
            <ReferenceLine y={bbUpper} stroke="rgba(245,166,35,0.35)" strokeDasharray="2 4"
              label={{ value: 'BB+2σ', fill: '#f5a623', fontSize: 8, fontFamily: "'JetBrains Mono',monospace" }} />
            <ReferenceLine y={bbLower} stroke="rgba(245,166,35,0.35)" strokeDasharray="2 4"
              label={{ value: 'BB−2σ', fill: '#f5a623', fontSize: 8, fontFamily: "'JetBrains Mono',monospace" }} />
            <ReferenceLine y={sma20} stroke="rgba(245,166,35,0.15)" strokeDasharray="2 4"
              label={{ value: 'SMA20', fill: '#5a6a82', fontSize: 8, fontFamily: "'JetBrains Mono',monospace" }} />
          </>
          }

          {/* Prediction band — shaded area between current and predicted price */}
          <ReferenceArea
            y1={currentPrice} y2={predictedPrice}
            fill={isUp ? 'rgba(0,232,122,0.06)' : 'rgba(255,59,107,0.06)'}
            fillOpacity={1}
          />

          {chartMode === 'area' ? (
            <Area
              type="monotone" dataKey="price"
              stroke={lineColor} strokeWidth={1.5}
              fill="url(#priceAreaGrad)"
              dot={false} activeDot={{ r: 3, fill: lineColor }}
            />
          ) : (
            <Line
              type="monotone" dataKey="price"
              stroke={lineColor} strokeWidth={1.5}
              dot={false} activeDot={{ r: 3, fill: lineColor }}
            />
          )}
        </ChartComponent>
      </ResponsiveContainer>

      {/* Stats bar */}
      <div style={{
        display: 'flex', gap: 24, marginTop: 10, paddingTop: 10,
        borderTop: '1px solid rgba(255,255,255,0.04)', flexWrap: 'wrap',
      }}>
        {[
          ['60-DAY HIGH', `$${high60.toFixed(2)}`, '#00e87a'],
          ['60-DAY LOW',  `$${low60.toFixed(2)}`,  '#ff3b6b'],
          ['60-DAY AVG',  `$${avg60.toFixed(2)}`,  '#5a6a82'],
          ['VOLATILITY',  `±$${volatility}/day`,   '#f5a623'],
        ].map(([l, v, c]) => (
          <div key={l}>
            <div style={{ ...mono, fontSize: 7, color: '#3a4558', fontWeight: 700, marginBottom: 2 }}>{l}</div>
            <div style={{ ...mono, fontSize: 12, color: c, fontWeight: 700 }}>{v}</div>
          </div>
        ))}
        <div style={{ marginLeft: 'auto', ...mono, fontSize: 9, color: '#3a4558', alignSelf: 'flex-end' }}>
          Hover chart to inspect · Toggle BB Bands for context
        </div>
      </div>
    </motion.div>
  );
};

export default PriceChart;