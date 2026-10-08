/**
 * services/api.js — ATHENA API Service
 * Connects to FastAPI backend at localhost:8000 (or REACT_APP_API_URL)
 */

import axios from 'axios';

const BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE,
  headers: { 'Content-Type': 'application/json' },
  timeout: 60000, // 60s — SHAP can be slow on first call
});

api.interceptors.request.use(cfg => {
  console.log(`[ATHENA API] ${cfg.method?.toUpperCase()} ${cfg.url}`);
  return cfg;
});

api.interceptors.response.use(
  res  => res,
  err  => {
    console.error('[ATHENA API Error]', err.response?.data || err.message);
    return Promise.reject(err);
  }
);

export const getHealth         = () => api.get('/health').then(r => r.data);
export const getSupportedStocks = () => api.get('/stocks').then(r => r.data);
export const predictStock      = ticker => api.post('/predict', { ticker }).then(r => r.data);
export const getSignals        = ticker => api.get(`/signals/${ticker}`).then(r => r.data);
export const getMetrics        = () => api.get('/metrics').then(r => r.data);

export default api;