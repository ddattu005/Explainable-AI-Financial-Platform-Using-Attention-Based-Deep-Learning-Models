"""
briefing.py - ATHENA AI Briefing Generator
Uses Groq API (llama-3.3-70b-versatile) to generate institutional-grade reports.
Author: ATHENA Project
"""

import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_ai_briefing(context: dict) -> str:
    """
    Generate a 3-paragraph institutional briefing from prediction context.

    Args:
        context: dict with keys: ticker, direction, confidence, confidence_tier,
                 composite_score, regime, change_pct, current_price, predicted_price,
                 top_features, signals

    Returns:
        str: 3-paragraph briefing text
    """
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    ticker = context.get("ticker", "UNKNOWN")
    direction = context.get("direction", "BULLISH")
    confidence = context.get("confidence", 0.5)
    tier = context.get("confidence_tier", "MODERATE")
    composite = context.get("composite_score", 0.5)
    regime = context.get("regime", "Trending")
    change_pct = context.get("change_pct", 0.0)
    current_price = context.get("current_price", 0.0)
    predicted_price = context.get("predicted_price", 0.0)
    top_features = context.get("top_features", [])
    signals = context.get("signals", {})

    # Build feature summary
    feature_summary = ""
    if top_features:
        top3 = top_features[:3] if isinstance(top_features, list) else []
        feature_lines = []
        for f in top3:
            if isinstance(f, dict):
                name = f.get("name", f.get("feature", "unknown"))
                impact = f.get("impact", f.get("shap_val", 0))
                feature_lines.append(f"{name} ({'+' if impact > 0 else ''}{impact:.4f})")
        feature_summary = ", ".join(feature_lines)

    # Build signal summary
    signal_summary = ""
    if signals and isinstance(signals, dict):
        parts = []
        for k, v in signals.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v:.3f}")
        signal_summary = ", ".join(parts)

    prompt = f"""You are an institutional equity research analyst writing a concise research note.

PREDICTION DATA:
- Ticker: {ticker}
- Direction: {direction}
- Confidence: {confidence*100:.1f}% ({tier})
- Composite Score: {composite*100:.1f}%
- Market Regime: {regime}
- Current Price: ${current_price:.2f}
- Predicted Price: ${predicted_price:.2f}
- Expected Change: {change_pct:+.2f}%
- Top SHAP Features: {feature_summary or "N/A"}
- Signal Values: {signal_summary or "N/A"}

Write exactly 3 paragraphs:
1. Direction assessment with ensemble composite analysis
2. Technical factor breakdown citing the SHAP features and regime context
3. Risk-adjusted recommendation with confidence tier classification

Rules:
- Use precise financial language (institutional tone)
- Reference specific numbers from the data
- Keep each paragraph to 3-4 sentences max
- Do NOT add disclaimers or headers
- Write as if for a Bloomberg terminal research note"""

    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
    )

    return chat.choices[0].message.content.strip()


if __name__ == "__main__":
    test_ctx = {
        "ticker": "AAPL",
        "direction": "BULLISH",
        "confidence": 0.65,
        "confidence_tier": "HIGH_CONF",
        "composite_score": 0.62,
        "regime": "Trending",
        "change_pct": 1.45,
        "current_price": 198.50,
        "predicted_price": 201.38,
        "top_features": [
            {"name": "EMA_12", "impact": 0.0032},
            {"name": "RSI_14", "impact": -0.0018},
            {"name": "BB_Upper", "impact": 0.0015},
        ],
        "signals": {"lstm": 0.62, "rsi": 0.55, "macd": 0.60, "ma_cross": 0.58},
    }
    print(generate_ai_briefing(test_ctx))
