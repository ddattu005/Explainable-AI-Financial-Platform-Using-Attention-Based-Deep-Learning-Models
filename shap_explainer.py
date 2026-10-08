"""
shap_explainer.py - ATHENA Explainability Module v3
Author: ATHENA Project - FIXED v3

ROOT CAUSE OF SHAP ERROR:
  PredictionOnlyWrapper did `predictions, _ = self.model(x)` — unpacks only 2 values.
  New model returns 3: (price, dir_prob, attention). This causes the unpack crash,
  which then triggers the "does not sum up" fallback warning.

FIXES:
  1. Wrapper correctly unpacks 3 values: `price, _, _ = self.model(x)`
  2. Use shap.GradientExplainer (more stable than DeepExplainer for LSTM+Attention)
  3. Return SIGNED shap values so frontend shows bullish/bearish correctly
  4. Gradient×Input fallback also returns signed values
"""

import torch
import torch.nn as nn
import numpy as np
import shap
import warnings
warnings.filterwarnings('ignore')


class _PriceOnlyWrapper(nn.Module):
    """
    Strips direction + attention outputs — returns only price scalar.
    SHAP requires single scalar output.

    Model signature: forward(x) → (price, dir_prob, attn_weights)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        price, _logit, _dir, _attn = self.model(x)   # unpack all 4 — use only price
        return price                           # (batch, 1)


class ATHENAExplainer:
    """
    SHAP-based explanations for ATHENA's LSTM+Attention model.
    Uses GradientExplainer (more stable than DeepExplainer for custom architectures).
    Falls back to Gradient×Input if SHAP raises errors.
    """

    def __init__(self, model, feature_names, device='cpu'):
        self.model         = model.to(device)
        self.model.eval()
        self.device        = device
        self.feature_names = feature_names
        self.explainer     = None
        self._wrapper      = _PriceOnlyWrapper(self.model).to(device)
        self._wrapper.eval()

    # ── CREATE EXPLAINER ───────────────────────────────────
    def create_explainer(self, background_data: np.ndarray, n_samples: int = 50):
        """
        Build SHAP GradientExplainer.

        Args:
            background_data : (N, seq_len, n_features) — must be VARIED training samples
            n_samples       : how many background samples to use
        """
        print("\n🔧 Creating SHAP GradientExplainer")
        pool = min(len(background_data), n_samples)
        idx  = np.random.choice(len(background_data), pool, replace=False)
        bg   = background_data[idx]

        std = float(np.std(bg))
        print(f"   Background pool: {len(background_data)} samples → using {pool}")
        print(f"   Background uniqueness (std): {std:.6f}", "✅" if std > 1e-3 else "⚠️  LOW")

        bg_tensor = torch.FloatTensor(bg).to(self.device)
        try:
            self.explainer = shap.GradientExplainer(self._wrapper, bg_tensor)
            print("✅ SHAP GradientExplainer ready")
        except Exception as e:
            print(f"   ⚠️  GradientExplainer init failed ({e}) — gradient fallback only")
            self.explainer = None

    # ── EXPLAIN PREDICTION ─────────────────────────────────
    def explain_prediction(self, input_data: np.ndarray):
        """
        Compute signed SHAP values for a single input window.

        Returns:
            feature_shap : (n_features,) signed array
                           Positive = pushed price UP, Negative = pushed price DOWN
            prediction   : raw normalized price scalar
        """
        inp = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

        with torch.no_grad():
            price_t, _logit, _d, _a = self.model(inp)    # unpack 4 outputs correctly
            prediction = float(price_t.cpu().item())

        # Try GradientExplainer first
        if self.explainer is not None:
            try:
                shap_raw = self.explainer.shap_values(inp)
                if isinstance(shap_raw, list):
                    shap_raw = shap_raw[0]
                # shape (1, seq_len, n_features) → mean over time → (n_features,)
                # Use mean (signed) NOT abs-mean, so direction is preserved
                signed = shap_raw[0].mean(axis=0)
                if np.max(np.abs(signed)) > 1e-9:
                    return signed, prediction
                print("   ⚠️  SHAP values collapsed — using gradient fallback")
            except Exception as e:
                print(f"   ⚠️  SHAP error ({str(e)[:80]}) — using gradient fallback")

        return self._gradient_importance(inp, prediction)

    # ── GRADIENT × INPUT FALLBACK ──────────────────────────
    def _gradient_importance(self, inp_tensor: torch.Tensor, prediction: float):
        """Signed gradient × input attribution."""
        x = inp_tensor.detach().clone().requires_grad_(True)
        price_out, _logit, _d, _a = self.model(x)
        price_out.sum().backward()

        grad   = x.grad.cpu().numpy()[0]       # (seq_len, n_features)
        inp_np = inp_tensor.cpu().numpy()[0]
        gi     = (grad * inp_np).mean(axis=0)  # (n_features,) — signed

        # Normalize to ~SHAP scale
        scale = np.max(np.abs(gi)) + 1e-9
        gi    = gi / scale * 0.01
        return gi, prediction

    # ── FEATURE IMPORTANCE ─────────────────────────────────
    def get_feature_importance(self, shap_values: np.ndarray, top_k: int = 5):
        """
        Return top-k features sorted by |SHAP|.
        Values keep their sign: + = bullish, - = bearish.
        """
        # Flatten to 1D — SHAP sometimes returns (n_features, 1) instead of (n_features,)
        sv = shap_values.flatten()
        abs_vals    = np.abs(sv)
        total_abs   = abs_vals.sum() + 1e-10
        top_indices = np.argsort(abs_vals)[-top_k:][::-1]
        result = []
        for idx in top_indices:
            i = int(idx)  # numpy int64 → Python int
            result.append((
                self.feature_names[i],
                float(sv[i]),
                float(abs_vals[i] / total_abs * 100),  # importance pct
            ))
        return result

    # ── ATTENTION HELPER ───────────────────────────────────
    def get_attention_explanation(self, input_data: np.ndarray):
        """Extract attention weights (3-output model compatible)."""
        inp = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _price, _logit, _dir, attn = self.model(inp)
        attn_np = attn.cpu().numpy()[0]
        avg     = attn_np.mean(axis=0)[-1, :]
        top_idx = np.argsort(avg)[-5:][::-1]
        return attn_np, [(int(i), float(avg[i])) for i in top_idx]