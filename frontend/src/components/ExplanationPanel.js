import React from 'react';

/**
 * ExplanationPanel — renders the institutional-grade XAI prediction report
 * in structured sections with professional headings (ISSUE 4 FIX).
 */

// Map section headings → CSS class + icon (updated for institutional headings)
const SECTION_MAP = [
  // New institutional headings
  { match: 'DIRECTIONAL FORECAST',       cls: 'forecast',    icon: '📊' },
  { match: 'SHAP ATTRIBUTION',           cls: 'evidence',    icon: '🔑' },
  { match: 'CONFIDENCE ASSESSMENT',      cls: 'confidence',  icon: '🎯' },
  { match: 'MARKET REGIME CLASSIF',      cls: 'regime',      icon: '📊' },
  { match: 'ENSEMBLE SIGNAL',            cls: 'decision',    icon: '🧠' },
  { match: 'METHODOLOGY NOTES',          cls: 'methodology', icon: '📄' },
  { match: 'REGULATORY DISCLAIMER',      cls: 'disclaimer',  icon: '⚠️' },
  // Legacy fallback headings (backward compat)
  { match: 'WHAT IS THE MODEL PREDICTING',  cls: 'forecast',    icon: '📈' },
  { match: 'WHY DOES THE MODEL THINK',      cls: 'evidence',    icon: '🔍' },
  { match: 'HOW CONFIDENT IS THE MODEL',    cls: 'confidence',  icon: '🎯' },
  { match: 'WHAT IS THE MARKET DOING',      cls: 'regime',      icon: '📊' },
  { match: 'HOW DOES THE MODEL REACH',      cls: 'decision',    icon: '🧠' },
  { match: 'ABOUT THIS METHODOLOGY',        cls: 'methodology', icon: '📄' },
  { match: 'IMPORTANT DISCLAIMER',          cls: 'disclaimer',  icon: '⚠️' },
  { match: 'PRICE FORECAST',                cls: 'forecast',    icon: '📈' },
  { match: 'TOP SHAP FEATURES',             cls: 'evidence',    icon: '🔑' },
  { match: 'DIRECTION ENSEMBLE',            cls: 'decision',    icon: '⚡' },
  { match: 'XAI METHODOLOGY',               cls: 'methodology', icon: '🧠' },
];

function parseExplanation(text) {
  if (!text) return [];
  const lines = text.split('\n');
  const sections = [];
  let current = null;

  for (const line of lines) {
    // Skip divider lines (both ─ and ━)
    if (/^[─━]+$/.test(line.trim())) continue;

    // Check if this line is a section heading
    const upper = line.toUpperCase();
    const matched = SECTION_MAP.find(s => upper.includes(s.match));

    if (matched) {
      if (current) sections.push(current);
      // Extract the title text (remove emoji prefix)
      const title = line.replace(/^\s*[^\w\s]*\s*/, '').trim();
      current = { title, cls: matched.cls, icon: matched.icon, lines: [] };
    } else if (current) {
      current.lines.push(line);
    } else if (line.trim() && !line.trim().startsWith('ATHENA XAI')) {
      // Before any section, treat as header
      if (!current) {
        current = { title: line.trim(), cls: 'header', icon: '📊', lines: [] };
      }
    }
  }
  if (current) sections.push(current);
  return sections;
}

const ExplanationPanel = ({ explanation }) => {
  if (!explanation) return null;
  const sections = parseExplanation(explanation);

  return (
    <div className="explanation-panel">
      <div className="card-title expl">🧠 MODEL ANALYSIS</div>
      <div className="explanation-sections">
        {sections.map((sec, i) => (
          <div key={i} className={`expl-section ${sec.cls}`}>
            <div className="expl-section-title">
              <span className="expl-icon">{sec.icon}</span>
              {sec.title}
            </div>
            <div className="expl-section-body">
              {sec.lines.map((line, j) => (
                <div key={j} className="expl-line">{line}</div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExplanationPanel;