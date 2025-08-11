# plot_cv5_dashboard_win.py
# Visualizes 5-fold CV results: per-fold bars, mean±std by dropout, and learning curves per (fold, dropout).
# Tailored for Windows paths and your log file location.

import os, re, math, glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ================== CONFIG (edit if needed) ==================
BASE_DIR = r"C:\Users\paulg\Desktop\ophta\Ophthalmology_project"
LOG_FILE = os.path.join(BASE_DIR, "history_run.txt")   # your combined console log
# Try to auto-detect cv5_summary.csv anywhere under BASE_DIR:
FOUND = glob.glob(os.path.join(BASE_DIR, "**", "cv5_summary.csv"), recursive=True)
SUMMARY_CSV = FOUND[0] if FOUND else None
FIG_DIR = os.path.join(BASE_DIR, "figs_cv5")
os.makedirs(FIG_DIR, exist_ok=True)
# ============================================================

def load_summary(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # expected columns
    need = {'fold', 'dropout', 'best_val_mae'}
    assert need.issubset(df.columns), f"Missing columns in {path} (need {need})"
    return df

# --------- Parse console log ----------
# Matches lines like: [F1 D0.10] Epoch 001/60 | LR ... | Train Loss X MAE Y | Val Loss A MAE B RMSE C
LOG_PATTERN = re.compile(
    r"\[F(?P<fold>\d+)\s+D(?P<drop>\d+\.\d+)\]\s*Epoch\s*(?P<ep>\d+)"
    r"/\d+\s*\|\s*LR\s*(?P<lr>[0-9.eE+\-]+)\s*\|\s*Train Loss\s*(?P<tr_loss>[0-9.]+)\s*MAE\s*(?P<tr_mae>[0-9.]+)\s*\|\s*Val Loss\s*(?P<val_loss>[0-9.]+)\s*MAE\s*(?P<val_mae>[0-9.]+)\s*RMSE\s*(?P<val_rmse>[0-9.]+)"
)

# Matches "-> Saved best to <path>\best_mae.pt (MAE=xx.xxx)" to recover ckpt paths (optional)
BEST_LINE = re.compile(
    r"Saved best to\s+(?P<ckpt>.+?best_mae\.pt)\s+\(MAE=(?P<mae>[0-9.]+)\)"
)

def parse_log_to_df(log_path):
    rows = []
    best_rows = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                d = m.groupdict()
                rows.append({
                    'fold': int(d['fold']),
                    'dropout': float(d['drop']),
                    'epoch': int(d['ep']),
                    'lr': float(d['lr']),
                    'train_loss': float(d['tr_loss']),
                    'train_mae': float(d['tr_mae']),
                    'val_loss': float(d['val_loss']),
                    'val_mae': float(d['val_mae']),
                    'val_rmse': float(d['val_rmse']),
                })
            b = BEST_LINE.search(line)
            if b:
                best_rows.append({'ckpt': b.group('ckpt').strip(), 'mae': float(b.group('mae'))})
    df = pd.DataFrame(rows)
    df_best = pd.DataFrame(best_rows)
    return df, df_best

def build_summary_from_log(df_log, save_path):
    # best val MAE per (fold, dropout)
    df_summary = (
        df_log.groupby(['fold', 'dropout'], as_index=False)['val_mae']
        .min()
        .rename(columns={'val_mae': 'best_val_mae'})
        .sort_values(['fold', 'best_val_mae'])
    )
    df_summary.to_csv(save_path, index=False)
    print(f"[Info] Built summary from log: {save_path}")
    return df_summary

# --------- Plot helpers ----------
def plot_bar_mae_per_fold(df_summary, out_path):
    # Best per fold
    best_per_fold = df_summary.sort_values('best_val_mae').groupby('fold', as_index=False).first()
    # Colors by dropout (optional)
    cmap = {0.1:"#4c78a8", 0.2:"#f58518", 0.3:"#54a24b"}
    colors = [cmap.get(round(d,2), "#333333") for d in best_per_fold['dropout']]
    plt.figure(figsize=(9,4))
    plt.bar(best_per_fold['fold'].astype(int), best_per_fold['best_val_mae'], color=colors)
    for x,y,d in zip(best_per_fold['fold'], best_per_fold['best_val_mae'], best_per_fold['dropout']):
        plt.text(x, y+0.3, f"{y:.1f}\n(d={d:.2f})", ha='center', va='bottom', fontsize=9)
    plt.xlabel("Fold")
    plt.ylabel("Best Val MAE")
    plt.title("Best Val MAE per fold (color = dropout)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_mean_mae_by_dropout(df_summary, out_path):
    g = df_summary.groupby('dropout')['best_val_mae'].agg(['mean','std','count']).reset_index()
    plt.figure(figsize=(7,4))
    plt.errorbar(g['dropout'], g['mean'], yerr=g['std'], fmt='-o', capsize=4)
    for x,m,s,n in zip(g['dropout'], g['mean'], g['std'], g['count']):
        s_show = 0.0 if (pd.isna(s)) else float(s)
        plt.text(x, m + s_show + 0.3, f"{m:.1f}±{s_show:.1f}", ha='center', fontsize=9)
    plt.xlabel("Dropout")
    plt.ylabel("MAE (mean ± std across folds)")
    plt.title("Mean CV MAE by dropout")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_learning_curves(df_log, fig_dir):
    # One figure per (fold, dropout)
    for (fold, d), g in df_log.sort_values('epoch').groupby(['fold','dropout']):
        # MAE curves
        plt.figure(figsize=(7.5,4))
        plt.plot(g['epoch'], g['train_mae'], label='Train MAE')
        plt.plot(g['epoch'], g['val_mae'],   label='Val MAE')
        # mark best val
        idx = g['val_mae'].idxmin()
        if pd.notna(idx):
            ep_best = int(g.loc[idx, 'epoch'])
            mae_best = float(g.loc[idx, 'val_mae'])
            plt.scatter([ep_best], [mae_best], marker='*', s=130, label=f'Best Val MAE={mae_best:.2f} @ep{ep_best}')
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title(f"Fold {fold} – Dropout {d:.2f} : MAE")
        plt.legend()
        plt.grid(alpha=0.3)
        out = os.path.join(fig_dir, f"curves_fold{fold}_drop{str(d).replace('.','p')}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

        # Loss curves
        plt.figure(figsize=(7.5,4))
        plt.plot(g['epoch'], g['train_loss'], label='Train Loss')
        plt.plot(g['epoch'], g['val_loss'],   label='Val Loss')
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Fold {fold} – Dropout {d:.2f} : Loss")
        plt.legend(); plt.grid(alpha=0.3)
        out2 = os.path.join(fig_dir, f"loss_fold{fold}_drop{str(d).replace('.','p')}.png")
        plt.tight_layout(); plt.savefig(out2, dpi=150); plt.close()

def main():
    # 1) Parse log (always)
    if not os.path.exists(LOG_FILE):
        print(f"[Error] Log file not found: {LOG_FILE}")
        return
    df_log, df_best_lines = parse_log_to_df(LOG_FILE)
    if df_log.empty:
        print(f"[Warning] No epoch lines parsed from {LOG_FILE}.")
    else:
        print(f"[Info] Parsed {len(df_log)} epoch rows from log.")

    # 2) Load or build summary
    if SUMMARY_CSV and os.path.exists(SUMMARY_CSV):
        df_sum = load_summary(SUMMARY_CSV)
        print(f"[Info] Loaded summary: {SUMMARY_CSV}")
    else:
        if df_log.empty:
            print("[Error] No summary.csv and cannot build summary (log parse empty).")
            return
        # Build from log
        out_csv = os.path.join(BASE_DIR, "cv5_summary_from_log.csv")
        df_sum = build_summary_from_log(df_log, out_csv)

    # 3) Print overall CV stats (best per fold)
    best_per_fold = df_sum.sort_values('best_val_mae').groupby('fold', as_index=False).first()
    mean_mae = best_per_fold['best_val_mae'].mean()
    std_mae  = best_per_fold['best_val_mae'].std()
    print(best_per_fold)
    print(f"CV mean MAE = {mean_mae:.3f} ± {std_mae:.3f}")

    # 4) Plots from summary
    plot_bar_mae_per_fold(df_sum, os.path.join(FIG_DIR, "bar_mae_per_fold.png"))
    plot_mean_mae_by_dropout(df_sum, os.path.join(FIG_DIR, "mean_mae_by_dropout.png"))
    print(f"[OK] Summary figures saved to: {FIG_DIR}")

    # 5) Learning curves from log (if we parsed any)
    if not df_log.empty:
        plot_learning_curves(df_log, FIG_DIR)
        print(f"[OK] Learning-curve figures saved to: {FIG_DIR}")
    else:
        print("[Info] Skipped learning curves (no log epochs parsed).")

if __name__ == "__main__":
    main()
