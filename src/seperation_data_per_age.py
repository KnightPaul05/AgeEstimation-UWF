# bin_images_by_age.py
import os
import math
import shutil
import pandas as pd

def age_to_bin(age):
    """Retourne l'étiquette de bin pour un âge (tranches de 10 ans).
       Bins: 0-10, 10-20, ..., 90-100, 100+ (>=100)."""
    if pd.isna(age):
        return None
    try:
        a = float(age)
    except:
        return None
    if a < 0:
        return None
    if a >= 100:
        return "100+"
    lo = int((a // 10) * 10)
    hi = lo + 10
    return f"{lo}-{hi}"

def main(
    img_dir=r"C:\Users\paulg\Desktop\ophta\Ophthalmology_project\data\crop_images_1square",
    excel_path=r"C:\Users\paulg\Desktop\ophta\Ophthalmology_project\data\metadata_healthy_only.xlsx",
    out_dir=r"C:\Users\paulg\Desktop\ophta\Ophthalmology_project\data\by_age_bins"
):
    # 1) Charger le tableau des métadonnées
    df = pd.read_excel(excel_path)

    # Normaliser les noms de colonnes attendus
    cols = {c.lower(): c for c in df.columns}
    if "filename" not in cols or "age" not in cols:
        raise ValueError("Le fichier Excel doit contenir les colonnes 'filename' et 'age'.")

    fn_col = cols["filename"]
    age_col = cols["age"]

    # 2) Création du dossier de sortie
    os.makedirs(out_dir, exist_ok=True)

    # 3) Préparer un set des fichiers présents dans le dossier images
    existing = {f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))}

    missing_files = []
    skipped_no_age = []
    copied = 0

    # 4) Parcourir les lignes et copier dans le bon bin
    for _, row in df.iterrows():
        fname = str(row[fn_col])
        age = row[age_col]

        # Option: si le fichier dans Excel contient un chemin, ne garder que la base
        base = os.path.basename(fname)

        if base not in existing:
            missing_files.append(base)
            continue

        bin_label = age_to_bin(age)
        if bin_label is None:
            skipped_no_age.append(base)
            continue

        src = os.path.join(img_dir, base)
        dst_dir = os.path.join(out_dir, bin_label)
        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, base)
        shutil.copy2(src, dst)
        copied += 1

    # 5) Résumé
    print(f"[OK] Images copiées : {copied}")
    if missing_files:
        print(f"[WARN] Fichiers listés dans l'Excel introuvables dans {img_dir} (n={len(missing_files)}).")
        # Affiche les 10 premiers pour debug
        print("   Exemples:", missing_files[:10])
    if skipped_no_age:
        print(f"[WARN] Images sans âge valide / négatif (n={len(skipped_no_age)}).")
        print("   Exemples:", skipped_no_age[:10])

if __name__ == "__main__":
    main()
