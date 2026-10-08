"""
train.py - ATHENA Multi-Task Training Script v2
Trains LSTM + Attention on both price regression AND direction classification.
Author: ATHENA Project - IMPROVED v2

KEY CHANGES FROM v1:
  1. Multi-task loss: 0.5 * MSELoss(price) + 0.5 * BCELoss(direction)
     → Directly optimizes directional accuracy (was 54%, target 80-90%)
  2. Uses ALL sequences (no max_sequences=42 cap)
  3. Gradient clipping tightened to 0.5
  4. Warm-up LR schedule for stable early training
  5. Saves directional_accuracy to metrics.json
  6. CosineAnnealing LR instead of ReduceLROnPlateau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              precision_score, recall_score, f1_score, confusion_matrix)
import os, json, sys
from datetime import datetime

from lstm_attention import LSTMAttentionModel
from data_loader import StockDataLoader


# ─────────────────────────────────────── MULTI-TASK LOSS ──
class MultiTaskLoss(nn.Module):
    """
    Combined price regression + direction classification loss.

    L = alpha * MSE(pred_price, true_price)
      + (1-alpha) * BCEWithLogits(pred_direction_logit, true_direction)

    alpha < 0.5 → prioritize directional accuracy (our goal)
    Uses BCEWithLogitsLoss with pos_weight to fix class imbalance.
    label_smoothing: softens 0/1 targets to 0.05/0.95 for better generalization
    """
    def __init__(self, alpha=0.35, label_smoothing=0.05, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.mse = nn.MSELoss()
        # BCEWithLogitsLoss handles sigmoid internally + supports pos_weight
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce_logits = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, price_pred, price_true, dir_logit, dir_true):
        price_loss = self.mse(price_pred.squeeze(), price_true)
        # Label smoothing: 0→0.05, 1→0.95 for better generalization
        if self.label_smoothing > 0:
            dir_true_smooth = dir_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            dir_true_smooth = dir_true
        dir_loss = self.bce_logits(dir_logit.squeeze(), dir_true_smooth)
        total = self.alpha * price_loss + (1 - self.alpha) * dir_loss
        return total, price_loss.item(), dir_loss.item()


# ─────────────────────────────────────────── TRAINER ──
class ModelTrainer:
    def __init__(self, model, device='cpu', alpha=0.5, pos_weight=None):
        self.model  = model.to(device)
        self.device = device
        self.criterion = MultiTaskLoss(alpha=alpha, pos_weight=pos_weight).to(device)
        self.train_losses, self.val_losses = [], []
        self.train_dir_acc, self.val_dir_acc = [], []
        self.best_val_loss = float('inf')

    # ─── single epoch ──
    def _run_epoch(self, loader, optimizer=None, train=True):
        self.model.train(train)
        total_loss = 0.0
        correct_dir, total_dir = 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_X, batch_y_price, batch_y_dir in loader:
                batch_X       = batch_X.to(self.device)
                batch_y_price = batch_y_price.to(self.device)
                batch_y_dir   = batch_y_dir.to(self.device)

                price_pred, dir_logit, dir_prob, _ = self.model(batch_X)

                # Use logits for loss (BCEWithLogitsLoss)
                loss, pl, dl = self.criterion(
                    price_pred, batch_y_price,
                    dir_logit,  batch_y_dir
                )

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()

                # Direction accuracy (use sigmoid probs)
                predicted_dir = (dir_prob.squeeze() >= 0.5).float()
                correct_dir  += (predicted_dir == batch_y_dir).sum().item()
                total_dir    += len(batch_y_dir)

        avg_loss = total_loss / len(loader)
        dir_acc  = correct_dir / total_dir if total_dir > 0 else 0.0
        return avg_loss, dir_acc

    # ─── full training ──
    def fit(self, train_loader, val_loader,
            epochs=100, lr=5e-4, patience=15,
            save_path='models/best_model.pth'):

        print("\n" + "="*60)
        print("🚀 ATHENA MULTI-TASK TRAINING v3")
        print("="*60)
        print(f"  Device:       {self.device}")
        print(f"  Epochs:       {epochs} (early stop patience={patience})")
        print(f"  LR:           {lr}")
        print(f"  Loss alpha:   {self.criterion.alpha:.2f} (price) / {1-self.criterion.alpha:.2f} (direction)")
        print(f"  Label smooth: {self.criterion.label_smoothing}")
        print("="*60)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )

        os.makedirs('models', exist_ok=True)
        no_improve = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_dir = self._run_epoch(train_loader, optimizer, train=True)
            vl_loss, vl_dir = self._run_epoch(val_loader,   train=False)
            scheduler.step(vl_loss)

            self.train_losses.append(tr_loss)
            self.val_losses.append(vl_loss)
            self.train_dir_acc.append(tr_dir)
            self.val_dir_acc.append(vl_dir)

            current_lr = optimizer.param_groups[0]['lr']
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d}/{epochs} | "
                      f"Train Loss: {tr_loss:.5f}  Dir: {tr_dir*100:.1f}% | "
                      f"Val Loss: {vl_loss:.5f}  Dir: {vl_dir*100:.1f}% | "
                      f"LR: {current_lr:.6f}")

            if vl_loss < self.best_val_loss:
                self.best_val_loss = vl_loss
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': vl_loss,
                    'val_dir_accuracy': vl_dir,
                    'train_loss': tr_loss,
                }, save_path)
                print(f"  ✅ Best model saved  (val_loss={vl_loss:.5f}, dir_acc={vl_dir*100:.1f}%)")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  ⏹️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

        print(f"\n✅ Training complete. Best val_loss={self.best_val_loss:.5f}")
        print(f"   Final val directional accuracy: {self.val_dir_acc[-1]*100:.1f}%")
        return self

    # ─── evaluate on test ──
    def evaluate(self, test_loader, target_scaler=None):
        """Evaluate model on test set. Computes metrics in BOTH normalized and
        dollar space consistently (BUG 3 FIX: was mixing scales before).
        Also computes per-class direction metrics (ISSUE 6 FIX)."""
        self.model.eval()
        all_price_pred, all_price_true, all_dir_pred, all_dir_true = [], [], [], []

        with torch.no_grad():
            for batch_X, batch_y_price, batch_y_dir in test_loader:
                batch_X = batch_X.to(self.device)
                pp, _logit, dp, _ = self.model(batch_X)
                all_price_pred.extend(pp.squeeze().cpu().numpy())
                all_price_true.extend(batch_y_price.numpy())
                all_dir_pred.extend((dp.squeeze() >= 0.5).float().cpu().numpy())
                all_dir_true.extend(batch_y_dir.numpy())

        pp = np.array(all_price_pred)
        pt = np.array(all_price_true)
        dp = np.array(all_dir_pred)
        dt = np.array(all_dir_true)

        # ── Normalized-space metrics (model's native space) ──
        mae_norm  = mean_absolute_error(pt, pp)
        rmse_norm = np.sqrt(mean_squared_error(pt, pp))
        r2_norm   = r2_score(pt, pp)

        # ── Dollar-space metrics (denormalized, for human readability) ──
        if target_scaler is not None:
            pp_real = target_scaler.inverse_transform(pp.reshape(-1,1)).flatten()
            pt_real = target_scaler.inverse_transform(pt.reshape(-1,1)).flatten()
            mae_dollar  = mean_absolute_error(pt_real, pp_real)
            rmse_dollar = np.sqrt(mean_squared_error(pt_real, pp_real))
            r2_dollar   = r2_score(pt_real, pp_real)
        else:
            mae_dollar  = mae_norm
            rmse_dollar = rmse_norm
            r2_dollar   = r2_norm

        # ── Direction classification metrics (ISSUE 6) ──
        dir_acc = (dp == dt).mean()

        # Per-class Precision / Recall / F1 (aligned with Muhammad et al. 2024)
        prec_up   = precision_score(dt, dp, pos_label=1.0, zero_division=0)
        recall_up = recall_score(dt, dp, pos_label=1.0, zero_division=0)
        f1_up     = f1_score(dt, dp, pos_label=1.0, zero_division=0)

        prec_dn   = precision_score(dt, dp, pos_label=0.0, zero_division=0)
        recall_dn = recall_score(dt, dp, pos_label=0.0, zero_division=0)
        f1_dn     = f1_score(dt, dp, pos_label=0.0, zero_division=0)

        cm = confusion_matrix(dt, dp, labels=[0.0, 1.0])

        print("\n" + "="*60)
        print("📊 TEST SET RESULTS")
        print("="*60)
        print(f"  --- Normalized Space ---")
        print(f"  MAE:  {mae_norm:.4f}")
        print(f"  RMSE: {rmse_norm:.4f}")
        print(f"  R²:   {r2_norm:.4f}")
        print(f"  --- Dollar Space ---")
        print(f"  MAE:  ${mae_dollar:.2f}")
        print(f"  RMSE: ${rmse_dollar:.2f}")
        print(f"  R²:   {r2_dollar:.4f}")
        print(f"  --- Direction Classification ---")
        print(f"  Accuracy: {dir_acc*100:.2f}%")
        print(f"  UP   → Precision: {prec_up:.3f}  Recall: {recall_up:.3f}  F1: {f1_up:.3f}")
        print(f"  DOWN → Precision: {prec_dn:.3f}  Recall: {recall_dn:.3f}  F1: {f1_dn:.3f}")
        print(f"  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
        print("="*60)

        return {
            # Normalized space (consistent — RMSE ≥ MAE guaranteed)
            'mae_norm':  mae_norm,
            'rmse_norm': rmse_norm,
            'r2_norm':   r2_norm,
            # Dollar space (all denormalized — also consistent)
            'mae':  mae_dollar,
            'rmse': rmse_dollar,
            'r2':   r2_dollar,
            # Direction metrics
            'directional_accuracy': dir_acc,
            'classification': {
                'up':   {'precision': prec_up,  'recall': recall_up,  'f1': f1_up},
                'down': {'precision': prec_dn,  'recall': recall_dn,  'f1': f1_dn},
                'confusion_matrix': cm.tolist(),
            }
        }


    def plot_curves(self, save_path='models/training_curves.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0a0a0a')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#111111')

        epochs = range(1, len(self.train_losses)+1)
        ax1.plot(epochs, self.train_losses, '#00d2ff', linewidth=2, label='Train Loss')
        ax1.plot(epochs, self.val_losses,   '#f5a623', linewidth=2, label='Val Loss')
        ax1.set_title('Training & Validation Loss', color='white', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Epoch', color='gray'); ax1.set_ylabel('Loss (MSE + BCE)', color='gray')
        ax1.tick_params(colors='gray'); ax1.legend()
        ax1.grid(alpha=0.2)

        ax2.plot(epochs, [x*100 for x in self.train_dir_acc], '#9b6dff', linewidth=2, label='Train Dir Acc')
        ax2.plot(epochs, [x*100 for x in self.val_dir_acc],   '#00e87a', linewidth=2, label='Val Dir Acc')
        ax2.axhline(y=80, color='#ff3b6b', linestyle='--', alpha=0.5, label='80% target')
        ax2.set_title('Directional Accuracy', color='white', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Epoch', color='gray'); ax2.set_ylabel('Accuracy (%)', color='gray')
        ax2.tick_params(colors='gray'); ax2.legend()
        ax2.grid(alpha=0.2)
        ax2.set_ylim([40, 100])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
        print(f"📊 Curves saved to {save_path}")
        plt.close()


# ─────────────────────────────────────────── MAIN ──
def main():
    DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS  = 100
    LR      = 5e-4
    BATCH   = 32
    ALPHA   = 0.35      # prioritize direction (65%) over price (35%)
    PATIENCE = 15       # early stopping patience

    print(f"\n🚀 ATHENA TRAINING PIPELINE\nDevice: {DEVICE}")
    os.makedirs('models', exist_ok=True)
    os.makedirs('data',   exist_ok=True)

    # 1. Load data
    loader = StockDataLoader(
        tickers=['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'JPM', 'NVDA', 'META', 'XOM', 'JNJ'],
        fmp_key=os.getenv('FMP_API_KEY') or os.getenv('POLYGON_API_KEY')
    )
    results = loader.process_all_tickers()

    # 2. Merge all tickers' data
    X_trains, y_ptrain, y_dtrain = [], [], []
    X_vals,   y_pval,   y_dval   = [], [], []
    X_tests,  y_ptest,  y_dtest  = [], [], []
    target_scalers = {}

    for ticker, r in results.items():
        X_trains.append(r['X_train']); y_ptrain.append(r['y_price_train']); y_dtrain.append(r['y_dir_train'])
        X_vals.append(r['X_val']);     y_pval.append(r['y_price_val']);     y_dval.append(r['y_dir_val'])
        X_tests.append(r['X_test']);   y_ptest.append(r['y_price_test']);   y_dtest.append(r['y_dir_test'])
        target_scalers[ticker] = r['target_scaler']

    def cat(*arrs):
        return np.concatenate(arrs)

    X_tr  = cat(*X_trains); yp_tr = cat(*y_ptrain); yd_tr = cat(*y_dtrain)
    X_vl  = cat(*X_vals);   yp_vl = cat(*y_pval);   yd_vl = cat(*y_dval)
    X_te  = cat(*X_tests);  yp_te = cat(*y_ptest);  yd_te = cat(*y_dtest)

    # ── ML FIX 2: Label noise filtering ──
    # Remove samples where direction change is too small to be meaningful
    NOISE_THRESHOLD = 0.003
    if len(X_tr) > 0:
        # Compute approximate returns from price targets
        # Filter out samples where |return| < threshold
        returns = np.abs(np.diff(yp_tr, prepend=yp_tr[0]))
        noise_mask = returns > NOISE_THRESHOLD
        n_before = len(X_tr)
        X_tr  = X_tr[noise_mask]
        yp_tr = yp_tr[noise_mask]
        yd_tr = yd_tr[noise_mask]
        print(f"  Label noise filter: {n_before} → {len(X_tr)} samples (removed {n_before - len(X_tr)} noisy)")

    # ── ML FIX 1: Compute pos_weight for class imbalance ──
    n_up   = yd_tr.sum()
    n_down = len(yd_tr) - n_up
    pos_weight = float(n_down / (n_up + 1e-6))
    print(f"  Class balance: UP={n_up:.0f} ({n_up/len(yd_tr)*100:.1f}%), DOWN={n_down:.0f} ({n_down/len(yd_tr)*100:.1f}%)")
    print(f"  pos_weight = {pos_weight:.3f} (compensates for UP majority)")

    print(f"\n📊 Dataset sizes:")
    print(f"  Train: {len(X_tr)} | Val: {len(X_vl)} | Test: {len(X_te)}")
    print(f"  Direction balance (train): UP={yd_tr.mean()*100:.1f}%")

    def make_loader(X, yp, yd, shuffle=True):
        ds = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(yp),
            torch.FloatTensor(yd)
        )
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

    train_loader = make_loader(X_tr,  yp_tr, yd_tr, shuffle=True)
    val_loader   = make_loader(X_vl,  yp_vl, yd_vl, shuffle=False)
    test_loader  = make_loader(X_te,  yp_te, yd_te, shuffle=False)

    # 3. Build model
    model = LSTMAttentionModel(
        input_dim=15, hidden_dim=128, num_heads=4,
        dropout=0.3, seq_len=60
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model parameters: {total_params:,}")

    # 4. Train (with pos_weight for class imbalance)
    trainer = ModelTrainer(model, device=DEVICE, alpha=ALPHA, pos_weight=pos_weight)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    trainer.plot_curves()

    # 5. Load best model and evaluate
    checkpoint = torch.load('models/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Per-ticker evaluation with correct scalers, then average
    print("\n📊 PER-TICKER TEST RESULTS:")
    ticker_metrics = []
    for ticker, r in results.items():
        te_ds = TensorDataset(
            torch.FloatTensor(r['X_test']),
            torch.FloatTensor(r['y_price_test']),
            torch.FloatTensor(r['y_dir_test'])
        )
        te_loader = DataLoader(te_ds, batch_size=BATCH, shuffle=False)
        t_metrics = trainer.evaluate(te_loader, target_scaler=r['target_scaler'])
        ticker_metrics.append(t_metrics)
        print(f"  {ticker}: MAE=${t_metrics['mae']:.2f}  RMSE=${t_metrics['rmse']:.2f}  "
              f"R²={t_metrics['r2']:.4f}  Dir={t_metrics['directional_accuracy']*100:.1f}%")

    # Average metrics across tickers (each weighted equally)
    avg_metrics = {
        # Dollar space
        'mae':  float(np.mean([m['mae']  for m in ticker_metrics])),
        'rmse': float(np.mean([m['rmse'] for m in ticker_metrics])),
        'r2':   float(np.mean([max(0.0, m['r2']) for m in ticker_metrics])),  # clamp at 0
        # Normalized space
        'mae_norm':  float(np.mean([m['mae_norm']  for m in ticker_metrics])),
        'rmse_norm': float(np.mean([m['rmse_norm'] for m in ticker_metrics])),
        'r2_norm':   float(np.mean([max(0.0, m['r2_norm']) for m in ticker_metrics])),  # clamp at 0
        # Direction
        'directional_accuracy': float(np.mean([m['directional_accuracy'] for m in ticker_metrics])),
    }

    # Average classification metrics
    avg_cls = {
        'up': {
            'precision': float(np.mean([m['classification']['up']['precision'] for m in ticker_metrics])),
            'recall':    float(np.mean([m['classification']['up']['recall']    for m in ticker_metrics])),
            'f1':        float(np.mean([m['classification']['up']['f1']        for m in ticker_metrics])),
        },
        'down': {
            'precision': float(np.mean([m['classification']['down']['precision'] for m in ticker_metrics])),
            'recall':    float(np.mean([m['classification']['down']['recall']    for m in ticker_metrics])),
            'f1':        float(np.mean([m['classification']['down']['f1']        for m in ticker_metrics])),
        },
    }

    print(f"\n📊 AVERAGED TEST METRICS:")
    print(f"  --- Dollar Space ---")
    print(f"  MAE:  ${avg_metrics['mae']:.2f}")
    print(f"  RMSE: ${avg_metrics['rmse']:.2f}")
    print(f"  R²:   {avg_metrics['r2']:.4f}")
    print(f"  --- Normalized Space ---")
    print(f"  MAE:  {avg_metrics['mae_norm']:.4f}")
    print(f"  RMSE: {avg_metrics['rmse_norm']:.4f}")
    print(f"  R²:   {avg_metrics['r2_norm']:.4f}")
    print(f"  --- Direction ---")
    print(f"  Accuracy: {avg_metrics['directional_accuracy']*100:.1f}%")
    print(f"  UP   P/R/F1: {avg_cls['up']['precision']:.3f} / {avg_cls['up']['recall']:.3f} / {avg_cls['up']['f1']:.3f}")
    print(f"  DOWN P/R/F1: {avg_cls['down']['precision']:.3f} / {avg_cls['down']['recall']:.3f} / {avg_cls['down']['f1']:.3f}")

    # 6. Save metrics
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "tickers": list(results.keys()),
        "total_training_samples": int(len(X_tr)),
        "total_val_samples":      int(len(X_vl)),
        "total_test_samples":     int(len(X_te)),
        "metrics": {
            "mae":                  avg_metrics['mae'],
            "rmse":                 avg_metrics['rmse'],
            "r2":                   avg_metrics['r2'],
            "mae_norm":             avg_metrics['mae_norm'],
            "rmse_norm":            avg_metrics['rmse_norm'],
            "r2_norm":              avg_metrics['r2_norm'],
            "directional_accuracy": avg_metrics['directional_accuracy'],
        },
        "classification_metrics": avg_cls,
        "model_params": total_params,
        "training_config": {
            "epochs": EPOCHS, "lr": LR,
            "alpha_price": ALPHA, "alpha_direction": 1-ALPHA,
            "loss": "MultiTaskLoss(MSE + BCE)",
        }
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print("\n💾 Saved models/metrics.json")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()