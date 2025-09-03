# inferance.py  — robuste aux encodages/délimiteurs/variantes de colonnes
import argparse, os, csv
import torch
from PIL import Image
from torchvision import transforms
import timm
from tqdm import tqdm

# ---------- Modèle ----------
def load_model(model_name, ckpt_path, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Essaye d'abord l'état EMA, sinon état standard
    state = None
    for key in ["ema_state_dict", "state_dict", "model", "model_state", "module", "state"]:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    if state is None:
        # parfois le ckpt est directement le state_dict
        if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
            state = ckpt
        else:
            raise RuntimeError(f"Impossible de trouver state_dict dans {ckpt_path}")

    # Retire un éventuel préfixe "module."
    state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }

    # Certaines têtes peuvent avoir des noms différents; on charge avec strict=False
    res = model.load_state_dict(state, strict=False)
    try:
        missing = res.missing_keys
        unexpected = res.unexpected_keys
    except AttributeError:
        missing, unexpected = res

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

# ---------- IO robustes ----------
def _safe_open_read(path):
    """Ouvre un fichier texte en lecture en essayant plusieurs encodages."""
    # 1) utf-8
    try:
        return open(path, "r", encoding="utf-8", newline="")
    except Exception:
        pass
    # 2) utf-8-sig
    try:
        return open(path, "r", encoding="utf-8-sig", newline="")
    except Exception:
        pass
    # 3) cp1252 (Windows) permissif
    try:
        return open(path, "r", encoding="cp1252", errors="replace", newline="")
    except Exception:
        # dernier recours binaire -> utf-8 replace
        data = open(path, "rb").read().decode("utf-8", errors="replace")
        from io import StringIO
        return StringIO(data)

def _sniff_dialect(fp):
    """Devine le séparateur (',' ou ';')."""
    pos = fp.tell()
    sample = fp.read(4096)
    fp.seek(pos)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    except Exception:
        class SimpleDialect(csv.excel):
            delimiter = ','  # défaut
        dialect = SimpleDialect()
    return dialect

def _norm_key(k: str) -> str:
    """Normalise un nom de colonne: minuscules, trim, enlève BOM, remplace espaces/tirets par underscore."""
    if not isinstance(k, str):
        return k
    k = k.replace("\ufeff", "")  # BOM
    return k.strip().lower().replace(" ", "_").replace("-", "_")

def _get_first(d: dict, candidates):
    """Retourne la première clé existante (après normalisation) parmi candidates."""
    for c in candidates:
        if c in d:  # déjà normalisé en amont
            return d[c]
    return None

# ---------- Prédiction ----------
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
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["filename","pred_age"])
        w.writerows(rows)
    return out_csv

# ---------- Évaluation MAE robuste ----------
def maybe_eval_mae(pred_csv, gt_csv, gt_filename_col=None, gt_age_col=None):
    # 1) charge GT
    gt = {}
    with _safe_open_read(gt_csv) as fp:
        dialect = _sniff_dialect(fp)
        rdr = csv.DictReader(fp, dialect=dialect)
        # Prépare la liste des colonnes normalisées
        field_map = [_norm_key(c) for c in (rdr.fieldnames or [])]
        # debug utile
        print("[info] GT columns detected:", field_map)

        # Normalisation et extraction
        for row in rdr:
            row_n = { _norm_key(k): v for k, v in row.items() }

            # clés candidates
            fname_candidates = []
            age_candidates = []

            if gt_filename_col:
                fname_candidates = [_norm_key(gt_filename_col)]
            else:
                fname_candidates = ["filename","file","image","img","img_name","name"]

            if gt_age_col:
                age_candidates = [_norm_key(gt_age_col)]
            else:
                age_candidates = ["age","ages","label","target","y","gt_age"]

            fn = None
            for key in fname_candidates:
                if key in row_n and row_n[key] not in (None, ""):
                    fn = str(row_n[key]).strip()
                    break

            agestr = None
            for key in age_candidates:
                if key in row_n and row_n[key] not in (None, ""):
                    agestr = str(row_n[key]).strip()
                    break

            if fn is None or agestr is None:
                continue

            # convertit l'âge (gère virgule décimale)
            agestr = agestr.replace(",", ".")
            try:
                gt[fn] = float(agestr)
            except ValueError:
                continue

    # 2) charge prédictions
    abs_errors, n = 0.0, 0
    with _safe_open_read(pred_csv) as fp:
        dialect = _sniff_dialect(fp)
        rdr = csv.DictReader(fp, dialect=dialect)
        # debug
        print("[info] PRED columns detected:", [_norm_key(c) for c in (rdr.fieldnames or [])])
        for row in rdr:
            row_n = { _norm_key(k): v for k, v in row.items() }
            f = row_n.get("filename")
            pa = row_n.get("pred_age")
            if f in gt and pa not in (None, ""):
                try:
                    pa = float(str(pa).replace(",", "."))
                    abs_errors += abs(pa - gt[f])
                    n += 1
                except ValueError:
                    pass

    if n > 0:
        mae = abs_errors / n
        print(f"MAE on files with ground truth (n={n}): {mae:.3f} years")
    else:
        print("No filename overlap or missing/invalid columns/values in GT or predictions.")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best-ema.pt or best.pt")
    ap.add_argument("--model", default="mobilevit_xs")
    ap.add_argument("--img-dir", dest="img_dir", required=True)
    ap.add_argument("--img-size", dest="img_size", type=int, default=256)
    ap.add_argument("--out-csv", dest="out_csv", default="predictions.csv")
    ap.add_argument("--gt-csv", dest="gt_csv", default=None, help="Optional CSV with ground-truth")
    # facultatif : si tu veux forcer les noms des colonnes GT
    ap.add_argument("--gt-filename-col", dest="gt_filename_col", default=None)
    ap.add_argument("--gt-age-col", dest="gt_age_col", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_model(args.model, args.ckpt, args.device)
    tfm = build_transform(args.img_size)
    pred_csv = predict_folder(model, tfm, args.img_dir, args.device, args.out_csv)

    if args.gt_csv:
        maybe_eval_mae(pred_csv, args.gt_csv, args.gt_filename_col, args.gt_age_col)

if __name__ == "__main__":
    main()
