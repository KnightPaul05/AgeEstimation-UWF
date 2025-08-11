"""
MobileViT Age Regression – Finetuning Script
Author: you (+ ChatGPT as thesis supervisor)

This script finetunes a MobileViT model (from `timm`) to predict age from fundus images.
Comments are in ENGLISH and highlight the parts you may want to CHANGE to improve results.

Requirements (install once):
    pip install timm torch torchvision pandas pillow scikit-learn

Usage (example):
    python mobilevit_finetune_age.py \
        --csv /path/to/Demographics_of_the_participants.xlsx \
        --img-dir /path/to/Healthy/images \
        --out ./runs/mobilevit_s \
        --epochs 30 --batch-size 16 --lr 3e-4

Note: this script expects columns 'Filename' and 'Age' in the csv/xlsx.

Attribution: the MobileViT model and pretrained weights are provided by the open-source TIMM library
(https://github.com/huggingface/pytorch-image-models). We use its API to create the model.
"""

import argparse
import os
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T

# External: TIMM provides MobileViT models
import timm  # pip install timm

# Optional but useful
from sklearn.model_selection import train_test_split


# ---------------------------
# Dataset (from your snippet)
# ---------------------------
class FundusAgeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """Args:
        csv_file (str): path to .csv or .xlsx with columns 'Filename' and 'Age'
        img_dir (str): directory with dataset images
        transform (callable, optional): torchvision transforms for PIL Images
        """
        if csv_file.endswith('.xlsx'):
            self.data = pd.read_excel(csv_file)
        else:
            self.data = pd.read_csv(csv_file)

        # Defensive: normalize column names
        self.data.columns = [c.strip() for c in self.data.columns]
        assert 'Filename' in self.data.columns and 'Age' in self.data.columns, \
            "Input file must contain columns 'Filename' and 'Age'"

        self.img_dir = img_dir
        self.transform = transform

        print('Dataset initialised')
        print(f'Number of images: {len(self.data)}')
        print(f'Example rows:\n{self.data.head(3)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = str(row['Filename'])
        age = float(row['Age'])

        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error opening image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        age_tensor = torch.tensor(age, dtype=torch.float32).unsqueeze(0)  # shape: [1]
        return image, age_tensor


# ---------------------------
# Utilities
# ---------------------------
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Model factory
# ---------------------------

def build_model(model_name: str = 'mobilevit_s', pretrained: bool = True, dropout: float = 0.0):
    """Create a MobileViT model for REGRESSION (1 output neuron).

    ### 🔧 You can change `model_name` to: 'mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s'
    Larger models may improve accuracy but need more VRAM/compute.
    """
    # timm will set the final layer to output 1 unit if we pass num_classes=1
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    # Optional: add dropout before the head if the architecture supports it
    # Many timm models expose a classifier/last layer attribute. For MobileViT it's `head`.
    if dropout > 0:
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1)
            )
        # else: keep as is
    return model


# ---------------------------
# Training / Validation loop
# ---------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_grad_norm=None, limit_batches=0):
    model.train()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)  # [B, 1]
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # metrics: MAE
        with torch.no_grad():
            mae = (outputs.detach() - targets).abs().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        mae_meter.update(mae, images.size(0))
        if limit_batches and loss_meter.count >= limit_batches * images.size(0):
            break

    


    return loss_meter.avg, mae_meter.avg


def validate(model, loader, criterion, device, limit_batches=0):
    model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    rmse_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            mae = (outputs - targets).abs().mean().item()
            rmse = torch.sqrt(((outputs - targets) ** 2).mean()).item()

            loss_meter.update(loss.item(), images.size(0))
            mae_meter.update(mae, images.size(0))
            rmse_meter.update(rmse, images.size(0))
            if limit_batches and loss_meter.count >= limit_batches * images.size(0):
                break

    return loss_meter.avg, mae_meter.avg, rmse_meter.avg


# ---------------------------
# Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Finetune MobileViT for Age Regression')

    # DATA
    p.add_argument('--csv', type=str, required=True, help='Path to CSV/XLSX with Filename, Age')
    p.add_argument('--img-dir', type=str, required=True, help='Directory with images')
    p.add_argument('--val-split', type=float, default=0.2, help='Validation split (0-1).')
    p.add_argument('--img-size', type=int, default=288, help='### 🔧 Input image size (e.g., 224/256/288/320)')

    # AUGMENTATION
    p.add_argument('--aug-light', action='store_true', help='Use LIGHT augmentations (default).')
    p.add_argument('--aug-strong', action='store_true', help='Use STRONG augmentations.')

    # MODEL
    p.add_argument('--model', type=str, default='mobilevit_s', help="### 🔧 TIMM model name: mobilevit_xxs/xs/s")
    p.add_argument('--dropout', type=float, default=0.0, help='### 🔧 Extra dropout before head (0-0.5).')
    p.add_argument('--no-pretrained', action='store_true', help='Disable pretrained weights.')

    # OPTIM
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4, help='### 🔧 Learning rate')
    p.add_argument('--weight-decay', type=float, default=1e-4, help='### 🔧 Weight decay (L2)')
    p.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'],
                   help='### 🔧 LR scheduler type')
    p.add_argument('--warmup-epochs', type=int, default=1, help='Warmup epochs for cosine schedule')
    p.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping (None to disable)')

    # MISC
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--out', type=str, default='./runs/mobilevit_exp')
    p.add_argument('--amp', action='store_true', help='Use mixed precision (AMP) if CUDA available')
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs)')

    # TEST / LIMITS
    p.add_argument('--limit-samples', type=int, default=0,
                   help='Use only the first N samples (0 = all).')
    p.add_argument('--limit-train-batches', type=int, default=0,
                   help='Stop each training epoch after B batches (0 = all).')
    p.add_argument('--limit-val-batches', type=int, default=0,
                   help='Stop each validation epoch after B batches (0 = all).')
    p.add_argument('--fast-test', action='store_true',
                   help='Quick smoke test: few samples, 1 epoch, tiny batches.')

    return p.parse_args()


def get_transforms(img_size, aug_light=True, aug_strong=False):
    """Create torchvision transforms for train/val.
    ### 🔧 Tweak augmentations: stronger aug can help generalization.
    """
    # Basic normalization values for ImageNet-pretrained backbones
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if aug_strong:
        train_tfms = T.Compose([
            T.Resize(int(img_size * 1.15)),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        # light/default
        train_tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tfms, val_tfms


def make_loaders(csv_path, img_dir, img_size, val_split, batch_size, num_workers, aug_light=True, aug_strong=False, limit_samples=0):
    train_tfms, val_tfms = get_transforms(img_size, aug_light, aug_strong)
    full_ds = FundusAgeDataset(csv_path, img_dir, transform=None)

    # Split indices for reproducible train/val split
    idx = np.arange(len(full_ds))
    if limit_samples > 0:
        limit = min(limit_samples, len(idx))
        idx = idx[:limit]  # limit to first N samples
    train_idx, val_idx = train_test_split(idx, test_size=val_split, random_state=42, shuffle=True)

    # Wrap into Subset with independent transforms
    class _Wrapper(Dataset):
        def __init__(self, base, indices, transform):
            self.base = base
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            img, age = self.base[self.indices[i]]
            if self.transform:
                # Re-open image to ensure transform is applied to PIL (since base returns transformed=None)
                row = full_ds.data.iloc[self.indices[i]]
                path = os.path.join(full_ds.img_dir, str(row['Filename']))
                image = Image.open(path).convert('RGB')
                img = self.transform(image)
            return img, age

    train_ds = _Wrapper(full_ds, train_idx, train_tfms)
    val_ds = _Wrapper(full_ds, val_idx, val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_optimizer(model, lr, weight_decay):
    """AdamW is a solid default for ViT-like models.
    ### 🔧 You can try: SGD(momentum=0.9), Lion (if available), different wd.
    """
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, scheduler_type, epochs, warmup_epochs=0):
    if scheduler_type == 'cosine':
        # Cosine Annealing with warmup via LambdaLR
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(max(1, warmup_epochs))
            progress = (current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        return None


def main():
    args = parse_args()
    if args.fast_test:
        # sensible defaults for a smoke test
        args.epochs = 1
        args.batch_size = max(2, min(args.batch_size, 4))
        args.limit_samples = max(args.limit_samples, 16)  
        args.limit_train_batches = max(args.limit_train_batches, 2)
        args.limit_val_batches = max(args.limit_val_batches, 2)
        args.num_workers = 0  # évite des soucis de multiprocessing sur petits tests
        print("[FAST-TEST] epochs=1, bs={}, limit_samples={}, limit_train_batches={}, limit_val_batches={}".format(
            args.batch_size, args.limit_samples, args.limit_train_batches, args.limit_val_batches
        ))

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # OUTPUT DIR
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DATA
    aug_light = True
    aug_strong = False
    if args.aug_strong:
        aug_light, aug_strong = False, True

    train_loader, val_loader = make_loaders(
        csv_path=args.csv,
        img_dir=args.img_dir,
        img_size=args.img_size,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_light=aug_light,
        aug_strong=aug_strong,
        limit_samples=args.limit_samples,
    )

    # MODEL
    model = build_model(model_name=args.model, pretrained=not args.no_pretrained, dropout=args.dropout)
    model.to(device)

    # LOSS – SmoothL1 can be more robust than pure MSE for regression
    # ### 🔧 Try `nn.L1Loss()` or `nn.MSELoss()` and compare MAE/RMSE.
    criterion = nn.SmoothL1Loss(beta=1.0)

    # OPT + SCHED
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.warmup_epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')

    best_val_mae = float('inf')
    best_ckpt_path = out_dir / 'best_mae.pt'
    patience = args.patience
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            max_grad_norm=(args.max_grad_norm if args.max_grad_norm > 0 else None),
            limit_batches=args.limit_train_batches 
        )

        val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device,
                                               limit_batches=args.limit_val_batches
        )

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Logging
        lr_current = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | LR {lr_current:.3e} | "
              f"Train Loss {train_loss:.4f} MAE {train_mae:.3f} | "
              f"Val Loss {val_loss:.4f} MAE {val_mae:.3f} RMSE {val_rmse:.3f} | "
              f"{elapsed:.1f}s")

        # Checkpoint on improvement
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'model_state': model.state_dict(),
                'args': vars(args),
                'val_mae': val_mae,
            }, best_ckpt_path)
            print(f"Saved new best model to {best_ckpt_path} (MAE={best_val_mae:.3f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping after {epoch} epochs without improvement.")
            break

    print(f"Best Val MAE: {best_val_mae:.3f} | checkpoint: {best_ckpt_path}")


if __name__ == '__main__':
    main()
