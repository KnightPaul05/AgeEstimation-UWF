#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évaluation du biais par âge pour un modèle de prédiction d'âge (MobileViT/ViT via timm).

Fonctionnalités :
- Charge un checkpoint (best-ema.pt ou best.pt) et prédit sur toutes les images.
- Calcule des métriques par tranche d'âge : n, MAE, RMSE, biais (pred−true), MedAE.
- IC 95% bootstrap pour MAE et biais.
- TTA (Test-Time Augmentation) optionnel : moyenne de 5 vues légères, sans réentraînement.
- Déduction automatique de la taille d'entrée du modèle timm. Fallback 256 si non disponible.

Entrées attendues :
- Dossier d'images (ex : crop_images_1square).
- Fichier Excel avec colonnes (casse insensible) : 'filename', 'age'.

Sorties :
- predictions_with_errors.csv
- metrics_by_age_bin.csv (inclut une ligne 'ALL')
- mae_by_age_bin.png
- bias_by_age_bin.png
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
import timm
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as F
from collections import defaultdict

# --------------------------- Binning & Stats ---------------------------

def parse_bins(binedges_str: str):
    """
    Convertit par ex. "0,10,20,30,40,50,60,70,80,90,100" en liste d'entiers.
    Crée des bins [0-10), [10-20), ..., [90-100) et un bin final "100+".
    """
    edges = [int(x) for x in binedges_str.split(",")]
    if len(edges) < 2 or edges != sorted(edges):
        raise ValueError("Arg --bins doit être une liste d'entiers croissante, ex: 0,10,20,...,100")
    return edges

def age_to_bin_label(age, edges):
    """Retourne l'étiquette de bin pour un âge donné et des bornes edges."""
    if pd.isna(age):
        return None
    try:
        a = float(age)
    except Exception:
        return None
    if a < 0:
        return None
    if a >= edges[-1]:
        return f"{edges[-1]}+"
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= a < hi:
            return f"{lo}-{hi}"
    return None

def bootstrap_ci(values, fn=np.mean, n_boot=2000, alpha=0.05, seed=42):
    """IC bootstrap (percentile) pour une statistique (par défaut moyenne)."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    n = vals.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(fn(vals[idx]))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi

# --------------------------- Transforms & TTA ---------------------------

def infer_img_size_from_model(model, fallback=256):
    """
    Essaie de récupérer la taille d'entrée depuis la config timm.
    - timm >= 0.9 : model.pretrained_cfg["input_size"] -> (3, H, W)
    - timm plus ancien : model.default_cfg["input_size"]
    """
    for attr in ["pretrained_cfg", "default_cfg"]:
        cfg = getattr(model, attr, None)
        if isinstance(cfg, dict):
            inp = cfg.get("input_size", None)  # ex: (3, 256, 256)
            if inp and len(inp) == 3 and isinstance(inp[1], (int, float)):
                return int(inp[1])
    return int(fallback)

def build_base_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def tta_views_from_pil(pil_img, img_size: int):
    """
    5 vues TTA légères adaptées fond d'œil :
    - identité
    - flip horizontal
    - rotation +5°
    - rotation −5°
    - légère augmentation de brightness
    On applique ensuite la même normalisation.
    """
    fill = (0, 0, 0)
    views = [
        pil_img,
        F.hflip(pil_img),
        F.rotate(pil_img, 5, fill=fill),
        F.rotate(pil_img, -5, fill=fill),
        F.adjust_brightness(pil_img, 1.10),
    ]
    base = build_base_transform(img_size)
    return [base(v) for v in views]  # liste de tensors (C,H,W)

# --------------------------- Modèle & Checkpoint ---------------------------

def load_model(model_name: str, ckpt_path: str, device: torch.device):
    """
    Crée le modèle timm et charge le state_dict depuis un checkpoint flexible :
    - 'ema_state_dict', 'state_dict', 'model', 'model_state', 'module', 'state'
    - ou directement un dict de poids (state_dict).
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = None
    if isinstance(ckpt, dict):
        for k in ["ema_state_dict", "state_dict", "model", "model_state", "module", "state"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        # parfois le ckpt est directement un state_dict
        if state is None and all(isinstance(k, str) for k in ckpt.keys()):
            state = ckpt
    if state is None:
        raise RuntimeError(f"state_dict introuvable dans le checkpoint : {ckpt_path}")

    # retire un éventuel préfixe "module."
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] Clés manquantes (extrait) :", missing[:8])
    if unexpected:
        print("[warn] Clés inattendues (extrait) :", unexpected[:8])

    model.to(device).eval()
    return model

# --------------------------- Évaluation principale ---------------------------

@torch.no_grad()
def run_eval(args):
    device = torch.device(args.device)
    model = load_model(args.model, args.ckpt, device)

    # Taille d'entrée auto
    img_size = infer_img_size_from_model(model, fallback=256)
    print(f"[info] img_size auto = {img_size}")

    # Lecture métadonnées
    df = pd.read_excel(args.metadata)  # nécessite openpyxl
    cols = {c.lower(): c for c in df.columns}
    if "filename" not in cols or "age" not in cols:
        raise ValueError("Le fichier Excel doit contenir les colonnes 'filename' et 'age'.")
    fn_col, age_col = cols["filename"], cols["age"]

    os.makedirs(args.out_dir, exist_ok=True)

    # Images présentes
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    present = {f for f in os.listdir(args.img_dir) if f.lower().endswith(valid_ext)}

    edges = parse_bins(args.bins)
    base_tfm = build_base_transform(img_size)

    rows = []   # tuples : (filename, age, pred, err, abs_err, bin)
    missing = 0
    skipped = 0

    # Boucle de prédiction
    for _, row in df.iterrows():
        base = os.path.basename(str(row[fn_col]))
        age  = row[age_col]

        if base not in present:
            missing += 1
            continue

        path = os.path.join(args.img_dir, base)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[skip] {base}: {e}")
            skipped += 1
            continue

        # Prédiction (avec ou sans TTA)
        try:
            if args.tta:
                tensors = tta_views_from_pil(img, img_size)
                preds = [model(t.unsqueeze(0).to(device)).squeeze().item() for t in tensors]
                pred = float(np.mean(preds))
            else:
                x = base_tfm(img).unsqueeze(0).to(device)
                pred = model(x).squeeze().item()
        except Exception as e:
            print(f"[skip pred] {base}: {e}")
            skipped += 1
            continue

        # Erreurs & bin
        binlab = age_to_bin_label(age, edges)
        if binlab is None:
            err = np.nan
            abse = np.nan
        else:
            err = float(pred) - float(age)
            abse = abs(err)

        rows.append((base, age, pred, err, abse, binlab))

    # CSV détaillé par image
    pred_csv = os.path.join(args.out_dir, "predictions_with_errors.csv")
    with open(pred_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["filename", "age", "pred_age", "signed_error_pred_minus_true", "abs_error", "age_bin"])
        for r in rows:
            w.writerow(r)

    # Agrégation par bin
    perbin = defaultdict(lambda: {"errs": [], "abserrs": []})
    for _, age, pred, err, abse, binlab in rows:
        if binlab is not None and not (np.isnan(err) or np.isnan(abse)):
            perbin[binlab]["errs"].append(err)
            perbin[binlab]["abserrs"].append(abse)

    # Ordre canonique des bins
    bin_labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)] + [f"{edges[-1]}+"]

    # Tableau de synthèse par bin
    summary = []
    all_errs, all_abserrs = [], []
    for b in bin_labels:
        errs = np.array(perbin[b]["errs"], dtype=float) if b in perbin else np.array([], dtype=float)
        abse = np.array(perbin[b]["abserrs"], dtype=float) if b in perbin else np.array([], dtype=float)
        n = errs.size
        if n > 0:
            mae = float(np.mean(abse))
            rmse = float(np.sqrt(np.mean(errs ** 2)))
            bias = float(np.mean(errs))
            medae = float(np.median(abse))
            mae_lo, mae_hi = bootstrap_ci(abse, fn=np.mean, n_boot=args.boot, alpha=0.05, seed=42)
            bias_lo, bias_hi = bootstrap_ci(errs, fn=np.mean, n_boot=args.boot, alpha=0.05, seed=42)
            summary.append([b, n, round(mae, 4), round(rmse, 4), round(bias, 4), round(medae, 4),
                            round(mae_lo, 4), round(mae_hi, 4), round(bias_lo, 4), round(bias_hi, 4)])
            all_errs.append(errs); all_abserrs.append(abse)
        else:
            summary.append([b, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    # Global "ALL"
    if len(all_errs) > 0:
        all_errs = np.concatenate(all_errs)
        all_abserrs = np.concatenate(all_abserrs)
        g_mae = float(np.mean(all_abserrs))
        g_rmse = float(np.sqrt(np.mean(all_errs ** 2)))
        g_bias = float(np.mean(all_errs))
        g_medae = float(np.median(all_abserrs))
        g_mae_lo, g_mae_hi = bootstrap_ci(all_abserrs, fn=np.mean, n_boot=args.boot, alpha=0.05, seed=42)
        g_bias_lo, g_bias_hi = bootstrap_ci(all_errs, fn=np.mean, n_boot=args.boot, alpha=0.05, seed=42)
    else:
        g_mae = g_rmse = g_bias = g_medae = g_mae_lo = g_mae_hi = g_bias_lo = g_bias_hi = np.nan

    # CSV résumé
    summary_csv = os.path.join(args.out_dir, "metrics_by_age_bin.csv")
    with open(summary_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["age_bin", "n", "mae", "rmse", "bias_pred_minus_true", "medae",
                    "mae_ci95_lo", "mae_ci95_hi", "bias_ci95_lo", "bias_ci95_hi"])
        for row in summary:
            w.writerow(row)
        w.writerow(["ALL", int(np.nansum([r[1] for r in summary])),
                    round(g_mae, 4), round(g_rmse, 4), round(g_bias, 4), round(g_medae, 4),
                    round(g_mae_lo, 4), round(g_mae_hi, 4), round(g_bias_lo, 4), round(g_bias_hi, 4)])

    # --------------------------- Graphiques ---------------------------

    # MAE par bin + IC95 + n
    mae_vals, mae_lo, mae_hi, ns = [], [], [], []
    for b, n, mae, rmse, bias, medae, lo, hi, blo, bhi in summary:
        mae_vals.append(mae)
        mae_lo.append(lo)
        mae_hi.append(hi)
        ns.append(n)

    x = np.arange(len(bin_labels))
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(x, mae_vals)
    for i, (v, lo, hi, n) in enumerate(zip(mae_vals, mae_lo, mae_hi, ns)):
        if not np.isnan(v):
            ax1.text(i, v + 0.2, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=9)
            # barres d'IC (verticales + petits tirets)
            if not (np.isnan(lo) or np.isnan(hi)):
                ax1.vlines(i, lo, hi, linewidth=2)
                ax1.hlines([lo, hi], i - 0.12, i + 0.12, linewidth=2)
    ax1.set_xticks(x); ax1.set_xticklabels(bin_labels, rotation=0)
    ax1.set_ylabel("MAE (années)"); ax1.set_xlabel("Tranche d'âge (ans)")
    ax1.set_title("MAE par tranche d'âge (IC 95% bootstrap)")
    if any([not np.isnan(v) for v in mae_vals]):
        ax1.set_ylim(0, max([v for v in mae_vals if not np.isnan(v)]) * 1.2)
    plt.tight_layout()
    fig1_path = os.path.join(args.out_dir, "mae_by_age_bin.png")
    plt.savefig(fig1_path, dpi=220)

    # Biais signé (pred − true) par bin + IC95
    bias_vals, bias_lo, bias_hi = [], [], []
    for b, n, mae, rmse, bias, medae, lo, hi, blo, bhi in summary:
        bias_vals.append(bias)
        bias_lo.append(blo)
        bias_hi.append(bhi)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.plot(x, bias_vals, marker="o")
    for i, (blo, bhi) in enumerate(zip(bias_lo, bias_hi)):
        if not (np.isnan(blo) or np.isnan(bhi)):
            ax2.vlines(i, blo, bhi, linewidth=2)
    ax2.set_xticks(x); ax2.set_xticklabels(bin_labels)
    ax2.set_ylabel("Biais (pred − âge réel) [années]")
    ax2.set_xlabel("Tranche d'âge (ans)")
    ax2.set_title("Biais par tranche d'âge (IC 95% bootstrap)")
    plt.tight_layout()
    fig2_path = os.path.join(args.out_dir, "bias_by_age_bin.png")
    plt.savefig(fig2_path, dpi=220)

    # Logs finaux
    print(f"[OK] CSV détaillé            : {pred_csv}")
    print(f"[OK] CSV résumé par bin      : {summary_csv}")
    print(f"[OK] Figure MAE              : {fig1_path}")
    print(f"[OK] Figure Biais            : {fig2_path}")

    if missing > 0:
        print(f"[WARN] {missing} fichiers listés dans l'Excel sont introuvables dans {args.img_dir}")
    if skipped > 0:
        print(f"[WARN] {skipped} images ont été ignorées (lecture/prediction).")

# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Évaluer le biais par âge d'un modèle de régression d'âge (timm).")
    ap.add_argument("--ckpt", required=True, help="Chemin vers best-ema.pt ou best.pt")
    ap.add_argument("--model", default="mobilevit_xs", help="Nom du modèle timm (ex: mobilevit_xs)")
    ap.add_argument("--img-dir", required=True, help="Dossier des images")
    ap.add_argument("--metadata", required=True, help="Excel avec colonnes 'filename' et 'age'")
    ap.add_argument("--out-dir", required=True, help="Dossier de sortie")
    ap.add_argument("--bins", default="0,10,20,30,40,50,60,70,80,90,100", help="Bornes des bins (ex: 0,10,...,100)")
    ap.add_argument("--boot", type=int, default=2000, help="Taille bootstrap pour IC (par bin)")
    ap.add_argument("--tta", action="store_true", help="Active le Test-Time Augmentation (5 vues)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda ou cpu")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
