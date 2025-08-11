# analyze_cv5_results.py
import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.model_selection import KFold, GroupKFold
import matplotlib.pyplot as plt

class FundusDS(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.img_dir, str(r['Filename']))).convert('RGB')
        if self.transform: img = self.transform(img)
        age = float(r['Age'])
        return img, torch.tensor(age, dtype=torch.float32).unsqueeze(0), r['Filename']

def get_transforms(img_size):
    mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
    val_tfms = T.Compose([T.Resize((img_size,img_size)), T.ToTensor(), T.Normalize(mean,std)])
    return val_tfms

def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt.get('args', {})
    model_name = args.get('model', 'mobilevit_xs')
    dropout = ckpt.get('dropout', args.get('dropout', 0.0))
    model = timm.create_model(model_name, pretrained=False, num_classes=1, drop_rate=float(dropout))
    model.load_state_dict(ckpt['model_state'], strict=True)
    return model, args, float(dropout)

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    mae_sum=0.0; mse_sum=0.0; n=0
    files=[]; y_true=[]; y_pred=[]
    for x,y,f in loader:
        x=x.to(device, non_blocking=True)
        y=y.to(device, non_blocking=True)
        o=model(x)
        mae_sum += (o-y).abs().sum().item()
        mse_sum += ((o-y)**2).sum().item()
        n += y.shape[0]
        files += list(f)
        y_true += y.squeeze(1).cpu().tolist()
        y_pred += o.squeeze(1).cpu().tolist()
    mae = mae_sum/n
    rmse = math.sqrt(mse_sum/n)
    return mae, rmse, pd.DataFrame({'Filename':files,'Age':y_true,'Pred':y_pred})

def main():
    import argparse
    p=argparse.ArgumentParser("Analyze CV5 results + OOF plots")
    p.add_argument('--summary', required=True, help='Path to cv5_summary.csv')
    p.add_argument('--csv', required=True, help='Original CSV/XLSX with Filename,Age,(optional SubjectID)')
    p.add_argument('--img-dir', required=True)
    p.add_argument('--out', default='./runs/analysis')
    p.add_argument('--num-workers', type=int, default=2)
    args=p.parse_args()

    out_dir=Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Charger summary
    summ=pd.read_csv(args.summary)
    if 'dropout' not in summ.columns:
        raise RuntimeError("cv5_summary.csv doit contenir au moins: fold, dropout, best_val_mae, ckpt")
    print("Résumé des meilleurs MAE par fold/config :")
    print(summ.head())

    # Stats par dropout
    g = summ.groupby('dropout')['best_val_mae'].agg(['mean','std','count']).sort_values('mean')
    print("\nMAE par dropout (mean ± std):")
    print(g)

    # 2) Plot bar mean±std par dropout
    plt.figure()
    xs = [str(d) for d in g.index.values]
    means = g['mean'].values
    stds = g['std'].values
    plt.bar(xs, means, yerr=stds, capsize=4)
    plt.title('Validation MAE (best per fold) par dropout')
    plt.xlabel('Dropout'); plt.ylabel('MAE (ans)')
    plt.tight_layout()
    plt.savefig(out_dir/'mae_by_dropout.png', dpi=150)
    plt.close()

    # 3) Recharger args depuis 1er checkpoint pour connaître model/img_size/seed/folds/subject-col
    first_ckpt = summ['ckpt'].iloc[0]
    model0, ckargs0, _ = load_ckpt(first_ckpt)
    img_size = int(ckargs0.get('img_size', 320))
    folds = int(ckargs0.get('folds', 5))
    seed = int(ckargs0.get('seed', 42))
    subject_col = ckargs0.get('subject_col', '')
    print(f"\nDetected from ckpt -> img_size={img_size}, folds={folds}, seed={seed}, subject_col='{subject_col}'")

    # 4) Charger table complète pour reconstruire les splits
    if args.csv.endswith('.xlsx'):
        df = pd.read_excel(args.csv, engine='openpyxl')
    else:
        df = pd.read_csv(args.csv)
    df.columns=[c.strip() for c in df.columns]
    assert 'Filename' in df.columns and 'Age' in df.columns, "CSV/XLSX doit contenir Filename et Age."

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 5) Refaire les splits
    if subject_col and subject_col in df.columns:
        print(f"Rebuild splits with GroupKFold on '{subject_col}'")
        splitter = GroupKFold(n_splits=folds)
        split_iter = list(splitter.split(df, groups=df[subject_col].astype(str).values))
    else:
        print("Rebuild splits with KFold")
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = list(splitter.split(df))

    val_trans = get_transforms(img_size)

    # 6) Pour chaque fold: charger le meilleur ckpt indiqué dans summary, évaluer sur son fold val
    oof_list=[]
    per_fold=[]
    for fold in range(1, folds+1):
        # sélectionner la meilleure ligne (plus bas MAE) pour ce fold
        rows = summ[summ['fold']==fold].sort_values('best_val_mae')
        if rows.empty:
            print(f"Fold {fold}: introuvable dans summary, on saute."); continue
        row = rows.iloc[0]
        ckpt_path=row['ckpt']
        print(f"\nFold {fold}: load {ckpt_path} (dropout={row['dropout']}, MAE={row['best_val_mae']:.3f})")

        model, ckargs, dprob = load_ckpt(ckpt_path)
        model.to(device)

        tr_idx, va_idx = split_iter[fold-1]
        val_df = df.iloc[va_idx].reset_index(drop=True)

        val_ds = FundusDS(val_df, args.img_dir, val_trans)
        pin = (device.type=='cuda')
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin)

        mae, rmse, df_pred = eval_loader(model, val_loader, device)
        df_pred['fold']=fold; df_pred['dropout']=dprob
        df_pred['Gap'] = df_pred['Age'] - df_pred['Pred']
        oof_list.append(df_pred)
        per_fold.append({'fold':fold, 'mae':mae, 'rmse':rmse})

        # libérer VRAM entre folds
        del model
        if device.type=='cuda': torch.cuda.empty_cache()

    if not oof_list:
        print("Aucune prédiction OOF générée (summary vide ?).")
        return

    oof = pd.concat(oof_list, ignore_index=True)
    oof_path = out_dir/'oof_predictions.csv'
    oof.to_csv(oof_path, index=False)
    print(f"\nOOF enregistré -> {oof_path}")

    per_fold_df = pd.DataFrame(per_fold)
    per_fold_df.to_csv(out_dir/'oof_per_fold_metrics.csv', index=False)
    print(per_fold_df)

    # 7) Plots OOF
    # (a) Scatter Age vs Pred
    plt.figure()
    plt.scatter(oof['Age'], oof['Pred'], s=8, alpha=0.6)
    lim_min = min(oof['Age'].min(), oof['Pred'].min())
    lim_max = max(oof['Age'].max(), oof['Pred'].max())
    plt.plot([lim_min, lim_max],[lim_min, lim_max], '--')
    plt.xlabel('Âge réel'); plt.ylabel('Âge prédit'); plt.title('OOF: Réel vs Prédit')
    plt.tight_layout(); plt.savefig(out_dir/'oof_scatter_age_pred.png', dpi=150); plt.close()

    # (b) Histogramme des résidus (Age - Pred)
    plt.figure()
    plt.hist(oof['Gap'], bins=30)
    plt.title('OOF: Histogramme des écarts (Age - Pred)')
    plt.xlabel('Écart (ans)'); plt.ylabel('N')
    plt.tight_layout(); plt.savefig(out_dir/'oof_residual_hist.png', dpi=150); plt.close()

    # (c) Résidus vs âge (biais)
    plt.figure()
    plt.scatter(oof['Age'], oof['Gap'], s=8, alpha=0.6)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Âge réel'); plt.ylabel('Écart (Age - Pred)')
    plt.title('OOF: Biais selon l’âge')
    plt.tight_layout(); plt.savefig(out_dir/'oof_residual_vs_age.png', dpi=150); plt.close()

    # (d) MAE par fold (bar)
    plt.figure()
    plt.bar(per_fold_df['fold'].astype(str), per_fold_df['mae'])
    plt.xlabel('Fold'); plt.ylabel('MAE (ans)'); plt.title('OOF: MAE par fold')
    plt.tight_layout(); plt.savefig(out_dir/'oof_mae_per_fold.png', dpi=150); plt.close()

    print(f"\nFigures enregistrées dans: {out_dir}")
    print("Astuce: pour visualiser les courbes d’apprentissage par époque, "
          "sauvegarde un CSV d’historique pendant l’entraînement (train/val par epoch), "
          "et on ajoutera un parseur pour les tracer.")

if __name__ == '__main__':
    main()
# This code is part of a script to analyze results from a CV5 training run
# and generate OOF predictions, metrics, and visualizations.