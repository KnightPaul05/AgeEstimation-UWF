# mobilevit_cv5.py
"""
5-Fold CV for Age Regression with MobileViT (timm)
- Strong augmentation option
- Early stopping
- AMP (mixed precision) on CUDA
- Dropout: fixed or grid (e.g., 0.1,0.2,0.3)
- Windows-safe (no local classes inside functions)
Author: you (+ ChatGPT as thesis supervisor)
Attribution: MobileViT models & weights via TIMM (pytorch-image-models)
"""

import argparse, os, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import timm
from sklearn.model_selection import KFold, GroupKFold

# ---------------------------
# Dataset
# ---------------------------
class FundusAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.data = df.reset_index(drop=True).copy()
        self.data.columns = [c.strip() for c in self.data.columns]
        assert 'Filename' in self.data.columns and 'Age' in self.data.columns, \
            "DataFrame must contain 'Filename' and 'Age'."
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['Filename']))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        age = float(row['Age'])
        return image, torch.tensor(age, dtype=torch.float32).unsqueeze(0)

class SubsetWithTransform(Dataset):
    def __init__(self, base_df, indices, img_dir, transform):
        self.df = base_df.iloc[indices].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, str(row['Filename']))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        age = float(row['Age'])
        return image, torch.tensor(age, dtype=torch.float32).unsqueeze(0)

# ---------------------------
# Utils
# ---------------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, v, n=1):
        self.val = v; self.sum += v*n; self.count += n
        self.avg = self.sum / self.count if self.count else 0

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(img_size, strong=False):
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    if strong:
        train_tfms = T.Compose([
            T.Resize(int(img_size * 1.15)),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.2,0.2,0.2,0.05)], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.ToTensor(), T.Normalize(mean, std),
        ])
    else:
        train_tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(), T.Normalize(mean, std),
        ])
    val_tfms = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)])
    return train_tfms, val_tfms

def build_model(model_name='mobilevit_xs', pretrained=True, dropout=0.0):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=1, drop_rate=dropout)

def build_optimizer(model, lr, wd):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def build_scheduler(optimizer, kind, epochs, warmup_epochs=0):
    if kind == 'cosine':
        def lr_lambda(ep):
            if ep < warmup_epochs:
                return float(ep + 1) / float(max(1, warmup_epochs))
            prog = (ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif kind == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    return None

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_grad_norm=None):
    model.train()
    loss_m = AverageMeter(); mae_m = AverageMeter()
    use_amp = (scaler is not None and device.type == 'cuda')
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x); loss = criterion(out, y)
        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if max_grad_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        with torch.no_grad():
            mae = (out - y).abs().mean().item()
        loss_m.update(loss.item(), x.size(0)); mae_m.update(mae, x.size(0))
    return loss_m.avg, mae_m.avg

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_m = AverageMeter(); mae_m = AverageMeter(); rmse_m = AverageMeter()
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        out = model(x); loss = criterion(out, y)
        mae = (out - y).abs().mean().item()
        rmse = torch.sqrt(((out - y) ** 2).mean()).item()
        loss_m.update(loss.item(), x.size(0)); mae_m.update(mae, x.size(0)); rmse_m.update(rmse, x.size(0))
    return loss_m.avg, mae_m.avg, rmse_m.avg

# ---------------------------
# Main (5-fold CV)
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("5-Fold CV MobileViT Age Regression")
    # DATA
    p.add_argument('--csv', type=str, required=True, help='CSV/XLSX with Filename, Age (+optional SubjectID)')
    p.add_argument('--img-dir', type=str, required=True)
    p.add_argument('--subject-col', type=str, default='', help='Optional column to group by subject (prevents leakage)')
    # MODEL & AUG
    p.add_argument('--model', type=str, default='mobilevit_xs', choices=['mobilevit_xxs','mobilevit_xs','mobilevit_s'])
    p.add_argument('--img-size', type=int, default=320)
    p.add_argument('--aug-strong', action='store_true')
    p.add_argument('--dropout', type=float, default=0.2, help='Used if no grid provided')
    p.add_argument('--dropout-grid', type=str, default='', help='Comma list, e.g. "0.1,0.2,0.3"')
    p.add_argument('--no-pretrained', action='store_true')
    # OPTIM
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--scheduler', type=str, default='plateau', choices=['cosine','plateau','none'])
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    # MISC
    p.add_argument('--out', type=str, default='./runs/cv5')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--amp', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load table
    if args.csv.endswith('.xlsx'):
        df = pd.read_excel(args.csv, engine='openpyxl')
    else:
        df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    assert 'Filename' in df.columns and 'Age' in df.columns

    # Folds
    groups = None
    if args.subject_col and args.subject_col in df.columns:
        groups = df[args.subject_col].astype(str).values
        splitter = GroupKFold(n_splits=args.folds)
        index_iter = splitter.split(df, groups=groups)
        print(f"Using GroupKFold on '{args.subject-col}'")
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        index_iter = splitter.split(df)
        print("Using KFold (no subject grouping)")

    dropouts = [args.dropout]
    if args.dropout_grid:
        try:
            dropouts = [float(x) for x in args.dropout_grid.split(',') if x.strip()]
        except:
            print("Could not parse --dropout-grid, falling back to single dropout", args.dropout)

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for fold, (train_idx, val_idx) in enumerate(index_iter, start=1):
        print(f"\n===== Fold {fold}/{args.folds} =====")
        train_tfms, val_tfms = get_transforms(args.img_size, strong=args.aug_strong)

        # Datasets & loaders
        train_ds = SubsetWithTransform(df, train_idx, args.img_dir, train_tfms)
        val_ds   = SubsetWithTransform(df, val_idx,   args.img_dir, val_tfms)

        pin = (torch.cuda.is_available() and device.type == 'cuda')
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin, persistent_workers=(args.num_workers>0))
        val_loader   = DataLoader(val_ds, batch_size=max(8, args.batch_size*2), shuffle=False,
                                  num_workers=args.num_workers, pin_memory=pin, persistent_workers=(args.num_workers>0))

        best_fold_mae = float('inf'); best_cfg = None

        for dprob in dropouts:
            fold_dir = out_root / f"fold{fold}_drop{str(dprob).replace('.','p')}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n-- Dropout={dprob:.2f} -> out: {fold_dir}")

            model = build_model(args.model, pretrained=not args.no_pretrained, dropout=dprob).to(device)
            criterion = nn.SmoothL1Loss(beta=1.0)
            optimizer = build_optimizer(model, args.lr, args.weight_decay)
            scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.warmup_epochs)
            scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=='cuda'))

            no_improve = 0; best_mae = float('inf'); best_ckpt = fold_dir / 'best_mae.pt'

            for epoch in range(1, args.epochs+1):
                t0 = time.time()
                tr_loss, tr_mae = train_one_epoch(
                    model, train_loader, criterion, optimizer, scaler, device,
                    max_grad_norm=(args.max_grad_norm if args.max_grad_norm>0 else None)
                )
                val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device)

                # Step scheduler
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                lr_now = optimizer.param_groups[0]['lr']
                dt = time.time()-t0
                print(f"[F{fold} D{dprob:.2f}] Epoch {epoch:03d}/{args.epochs} | LR {lr_now:.3e} | "
                      f"Train Loss {tr_loss:.4f} MAE {tr_mae:.3f} | "
                      f"Val Loss {val_loss:.4f} MAE {val_mae:.3f} RMSE {val_rmse:.3f} | {dt:.1f}s")

                if val_mae < best_mae:
                    best_mae = val_mae; no_improve = 0
                    torch.save({'model_state': model.state_dict(),
                                'val_mae': val_mae,
                                'args': vars(args),
                                'dropout': dprob,
                                'fold': fold}, best_ckpt)
                    print(f"  -> Saved best to {best_ckpt} (MAE={best_mae:.3f})")
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        print(f"  -> Early stopping (no improve {args.patience} epochs).")
                        break

            # Track best for this dropout
            summary.append({'fold': fold, 'dropout': dprob, 'best_val_mae': best_mae, 'ckpt': str(best_ckpt)})
            if best_mae < best_fold_mae:
                best_fold_mae = best_mae; best_cfg = (dprob, str(best_ckpt))

        print(f"Best on Fold {fold}: MAE={best_fold_mae:.3f} with dropout={best_cfg[0]:.2f}")
        # free CUDA mem between configs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save summary
    summ_df = pd.DataFrame(summary).sort_values(['fold','best_val_mae'])
    summ_path = out_root / "cv5_summary.csv"
    summ_df.to_csv(summ_path, index=False)
    print("\n===== CV summary =====")
    print(summ_df)
    print(f"Saved: {summ_path}")
    # Report per-fold best (first per fold)
    best_per_fold = summ_df.groupby('fold').first()['best_val_mae'].values
    print(f"Per-fold MAE: {best_per_fold} | mean={np.mean(best_per_fold):.3f} ± {np.std(best_per_fold):.3f}")

if __name__ == "__main__":
    main()
