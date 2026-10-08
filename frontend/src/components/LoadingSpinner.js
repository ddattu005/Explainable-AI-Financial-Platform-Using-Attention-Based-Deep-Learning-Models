import React from 'react';
import { motion } from 'framer-motion';

const STEPS = ['Fetching data', 'Running LSTM', 'Computing SHAP', 'Extracting attention'];

const LoadingSpinner = ({ currentStep = -1 }) => (
  <motion.div
    className="loading-container"
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
  >
    <div className="neural-spinner">
      <div className="ns-ring" />
      <div className="ns-ring" />
      <div className="ns-ring" />
    </div>
    <div className="loading-title">NEURAL NETWORK PROCESSING</div>
    <div className="loading-steps">
      {STEPS.map((s, i) => (
        <div
          key={i}
          className={`l-step ${currentStep === i ? 'active' : currentStep > i ? 'done' : ''}`}
        >
          {currentStep > i ? '✓ ' : ''}{s}
        </div>
      ))}
    </div>
  </motion.div>
);

export default LoadingSpinner;