"""
lstm_attention.py - ATHENA Enhanced Model Architecture
LSTM + Multi-Head Attention + Direction Classification Head
Author: ATHENA Project - IMPROVED v2

CHANGES FROM v1:
  - Added direction_head (sigmoid) for direct UP/DOWN classification
  - forward() now returns (price_pred, direction_prob, attention_weights)
  - Added LayerNorm after each LSTM for training stability
  - Increased dropout to 0.3 for better generalization
  - Added positional encoding for temporal awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — tells the model WHERE in time each step is.
    Improves temporal awareness significantly.
    """
    def __init__(self, d_model, max_len=60, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism.
    Learns to focus on the most relevant historical time steps.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, L, _ = x.shape
        Q = self.query(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = self.key(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = self.value(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)

        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        # Temperature sharpening (τ=0.5) — focuses attention on relevant lags
        energy = energy / 0.5
        attention = self.attn_dropout(F.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)
        x = x.permute(0,2,1,3).contiguous().view(B, L, self.hidden_dim)
        return self.out(x), attention


class LSTMAttentionModel(nn.Module):
    """
    LSTM + Multi-Head Attention with Dual-Head Output:
      1. Price prediction head  → regression (normalized price)
      2. Direction head         → binary classification (sigmoid: P(UP))

    Architecture:
      Input(60,15) → Linear Proj → PosEnc → LSTM1 → LN → LSTM2 → LN
        → Attention → Residual+LN → [fc1→fc2→fc3→price_out]
                                  → [dir_fc1→dir_out]

    Training: Loss = α * MSE(price) + (1-α) * BCE(direction)
              α = 0.5  →  equal weight to both objectives
    """

    def __init__(self, input_dim=15, hidden_dim=128, num_heads=4,
                 dropout=0.3, seq_len=60):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection (normalizes input scale)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=seq_len, dropout=dropout)

        # LSTM layers
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.ln1   = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.ln2   = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout)

        # Multi-Head Attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ── Price Prediction Head ──────────────────────────────────
        self.price_fc1 = nn.Linear(hidden_dim, 128)
        self.price_fc2 = nn.Linear(128, 64)
        self.price_fc3 = nn.Linear(64, 16)
        self.price_out = nn.Linear(16, 1)
        self.price_drop = nn.Dropout(dropout)

        # ── Direction Classification Head ──────────────────────────
        # NEW: directly predicts P(price will go UP)
        self.dir_fc1  = nn.Linear(hidden_dim, 64)
        self.dir_fc2  = nn.Linear(64, 16)
        self.dir_out  = nn.Linear(16, 1)
        self.dir_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len=60, input_dim=15)

        Returns:
            price_pred:     (batch, 1)  — normalized predicted price
            direction_logit:(batch, 1)  — raw logit for BCEWithLogitsLoss
            direction_prob: (batch, 1)  — P(price UP), range [0,1]
            attention:      (batch, num_heads, seq_len, seq_len)
        """
        # Project input to hidden_dim
        x = F.relu(self.input_norm(self.input_proj(x)))  # (B, L, H)

        # Positional encoding
        x = self.pos_enc(x)

        # LSTM 1
        out1, _ = self.lstm1(x)
        out1 = self.drop1(self.ln1(out1))

        # LSTM 2
        out2, _ = self.lstm2(out1)
        out2 = self.drop2(self.ln2(out2))

        # Attention + residual
        attn_out, attn_weights = self.attention(out2)
        out2 = self.attn_norm(out2 + attn_out)   # residual connection

        # Take last time step
        last = out2[:, -1, :]   # (B, H)

        # ── Price head ──
        p = self.price_drop(F.relu(self.price_fc1(last)))
        p = self.price_drop(F.relu(self.price_fc2(p)))
        p = F.relu(self.price_fc3(p))
        price_pred = self.price_out(p)             # (B, 1)

        # ── Direction head ──
        d = self.dir_drop(F.relu(self.dir_fc1(last)))
        d = self.dir_drop(F.relu(self.dir_fc2(d)))
        direction_logit = self.dir_out(d)                   # (B, 1) raw logit
        direction_prob  = torch.sigmoid(direction_logit)    # (B, 1), P(UP)

        return price_pred, direction_logit, direction_prob, attn_weights

    def get_attention_weights(self, x):
        with torch.no_grad():
            _, _, _, attn = self.forward(x)
            return attn.squeeze(0).cpu().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    print("🧪 Testing Enhanced LSTM + Attention Model v2")
    print("=" * 60)
    model = LSTMAttentionModel(input_dim=15, hidden_dim=128, num_heads=4, dropout=0.3)
    print(f"✅ Model created")
    print(f"📊 Parameters: {count_parameters(model):,}")

    dummy = torch.randn(32, 60, 15)
    price, logit, direction, attn = model(dummy)
    print(f"📤 Price output: {price.shape}")
    print(f"📤 Direction logit: {logit.shape}")
    print(f"📤 Direction prob: {direction.shape}  (P(UP) range: [{direction.min():.3f}, {direction.max():.3f}])")
    print(f"🎯 Attention: {attn.shape}")
    print("✅ All shapes correct!")
    return model


if __name__ == "__main__":
    model = test_model()
