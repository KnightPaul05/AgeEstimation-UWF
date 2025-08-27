# infer_age.py
import argparse, os, csv, math
import torch
from PIL import Image
from torchvision import transforms
import timm
from tqdm import tqdm

def load_model(model_name, ckpt_path, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Essaye d'abord l'état EMA, sinon état standard
    state = None
    for key in ["ema_state_dict", "state_dict", "model", "model_state", "module", "state"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    if state is None:
        # parfois le ckpt est directement le state_dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            state = ckpt
        else:
            raise RuntimeError(f"Impossible de trouver state_dict dans {ckpt_path}")

    # Retire un éventuel préfixe "module."
    state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
    # Certaines têtes peuvent avoir des noms différents; on charge avec strict=False
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warn] Missing keys:", missing[:10], "...")
    if unexpected:
        print("[warn] Unexpected keys:", unexpected[:10], "...")
    model.to(device)
    model.eval()
    return model

def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), # ImageNet
    ])

@torch.no_grad()
def predict_folder(model, transform, img_dir, device, out_csv):
    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))]
    rows = []
    for f in tqdm(sorted(img_files)):
        path = os.path.join(img_dir, f)
        try:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            pred = model(x).squeeze().item()
            rows.append((f, float(pred)))
        except Exception as e:
            print(f"[skip] {f}: {e}")
    with open(out_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["filename","pred_age"])
        w.writerows(rows)
    return out_csv

def maybe_eval_mae(pred_csv, gt_csv):
    gt = {}
    with open(gt_csv, newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            gt[row["filename"]] = float(row["age"])
    abs_errors, n = 0.0, 0
    with open(pred_csv, newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            f = row["filename"]
            if f in gt:
                abs_errors += abs(float(row["pred_age"]) - gt[f])
                n += 1
    if n > 0:
        mae = abs_errors / n
        print(f"MAE on files with ground truth (n={n}): {mae:.3f} years")
    else:
        print("No filename overlap between predictions and ground truth.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best-ema.pt or best.pt")
    ap.add_argument("--model", default="mobilevit_xs")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--out-csv", default="predictions.csv")
    ap.add_argument("--gt-csv", default=None, help="Optional CSV with columns filename,age")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_model(args.model, args.ckpt, args.device)
    tfm = build_transform(args.img_size)
    pred_csv = predict_folder(model, tfm, args.img-dir, args.device, args.out_csv)

    if args.gt_csv:
        maybe_eval_mae(pred_csv, args.gt_csv)

if __name__ == "__main__":
    main()
