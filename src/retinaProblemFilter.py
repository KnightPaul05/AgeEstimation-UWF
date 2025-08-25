#!/usr/bin/env python3
"""
Retina Problem Filter (OpenCV GUI)

Purpose
-------
Quickly review a folder of fundus images and mark each image as
  - OK (no problem)
  - PROBLEM (e.g., missing/unclear optic disc, artifacts, wrong eye, etc.)

Features
--------
- Keyboard-first workflow (no mouse required)
- Writes a CSV log that can be resumed any time
- Copies every PROBLEM image to a separate folder for manual audit
- Skips images already labeled (resume where you left off)
- Undo the last decision
- Works on large images (auto-fit to window)

Output
------
CSV with columns: Filename, Problem (1=problem, 0=ok), Reason
Copies of problem images are saved to: out_dir

Author: ChatGPT (custom-built for Paul's UWF project)
Source: Original code written for this project (no external code pasted).
"""

import argparse
import os
import sys
import cv2
import csv
import shutil
from pathlib import Path
from typing import List, Tuple

# -----------------------------
# Utilities
# -----------------------------

def list_images(img_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = []
    for e in exts:
        files.extend(img_dir.rglob(f"*{e}"))
    # stable order
    files = sorted(files)
    return files


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_existing(csv_path: Path):
    """Return dict filename->(problem, reason) and ordered list of rows."""
    mapping = {}
    rows = []
    if not csv_path.exists():
        return mapping, rows
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = r.get('Filename') or r.get('filename') or r.get('file')
            prob = r.get('Problem') or r.get('problem')
            reason = r.get('Reason') or r.get('reason') or ''
            if fn is None or prob is None:
                continue
            prob = int(str(prob).strip())
            mapping[fn] = (prob, reason)
            rows.append({'Filename': fn, 'Problem': prob, 'Reason': reason})
    return mapping, rows


def append_row(csv_path: Path, row: dict, write_header_if_needed=True):
    write_header = write_header_if_needed and (not csv_path.exists())
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['Filename', 'Problem', 'Reason'])
        if write_header:
            w.writeheader()
        w.writerow(row)


def rewrite_all(csv_path: Path, rows: List[dict]):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['Filename', 'Problem', 'Reason'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def draw_overlay(img, text_lines, margin=8):
    """Draw semi-transparent text box with helper instructions."""
    overlay = img.copy()
    x, y = margin, margin
    # background box
    (w, h), _ = cv2.getTextSize(" ".join(text_lines), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # compute box height as lines * (h+4)
    box_h = int(len(text_lines) * (h + 8) + 8)
    box_w = max(300, min(img.shape[1] - 2*margin, 20 + max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in text_lines)))
    cv2.rectangle(overlay, (x-6, y-6), (x + box_w, y + box_h), (0, 0, 0), -1)
    alpha = 0.45
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # draw text
    yy = y
    for t in text_lines:
        cv2.putText(img, t, (x, yy + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        yy += h + 8
    return img


def fit_in_window(img, max_wh: int):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_wh:
        scale = max_wh / float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale

# -----------------------------
# Main loop
# -----------------------------

def main():
    ap = argparse.ArgumentParser("Flag PROBLEM images and copy them to a folder")
    ap.add_argument('--img-dir', required=True, type=str, help='Input directory with images')
    ap.add_argument('--out-dir', required=True, type=str, help='Directory where PROBLEM images will be copied')
    ap.add_argument('--csv', type=str, default='', help='Path to CSV log (default: <img-dir>/problem_flags.csv)')
    ap.add_argument('--exts', type=str, default='.jpg,.jpeg,.png,.tif,.tiff,.bmp', help='Comma list of extensions')
    ap.add_argument('--window', type=str, default='ProblemFilter', help='Window name')
    ap.add_argument('--max-size', type=int, default=1400, help='Max window side (px) for display')
    ap.add_argument('--review-all', action='store_true', help='Do not skip images already labeled in CSV')
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv) if args.csv else (img_dir / 'problem_flags.csv')
    ensure_dir(out_dir)

    exts = tuple(e.strip().lower() if e.strip().startswith('.') else '.'+e.strip().lower() for e in args.exts.split(','))

    # Load state
    labeled_map, rows = load_existing(csv_path)

    files = list_images(img_dir, exts)
    if not files:
        print(f"No images found in {img_dir} with extensions {exts}")
        sys.exit(1)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    history = []  # stack for undo; contains (filename, old_row_or_none)

    HELP = [
        'Keys: P=Problem  G=Good  U=Undo  S=Save  Q=Quit',
        'Enter=Next (same as G)  Backspace=Mark Problem',
        'When Problem: a copy is saved to out folder.',
    ]

    for img_path in files:
        rel = str(img_path.relative_to(img_dir))
        if (not args.review_all) and (rel in labeled_map):
            # already labeled, skip
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            # mark as problem with reason for traceability
            row = {'Filename': rel, 'Problem': 1, 'Reason': 'unreadable'}
            rows.append(row); labeled_map[rel] = (1, 'unreadable')
            append_row(csv_path, row)
            continue

        disp, scale = fit_in_window(img.copy(), args['max_size'] if isinstance(args, dict) else args.max_size)
        label_text = f"{rel}  |  size={img.shape[1]}x{img.shape[0]}"
        disp = draw_overlay(disp, [label_text] + HELP)

        while True:
            cv2.imshow(args.window, disp)
            k = cv2.waitKey(50) & 0xFF

            if k in (ord('g'), ord('G'), 13):  # good / Enter
                # record 0
                row = {'Filename': rel, 'Problem': 0, 'Reason': ''}
                history.append((rel, labeled_map.get(rel)))
                labeled_map[rel] = (0, '')
                rows.append(row)
                append_row(csv_path, row)
                break

            elif k in (ord('p'), ord('P'), 8):  # problem / Backspace
                # record 1 and copy
                row = {'Filename': rel, 'Problem': 1, 'Reason': ''}
                history.append((rel, labeled_map.get(rel)))
                labeled_map[rel] = (1, '')
                rows.append(row)
                append_row(csv_path, row)

                # copy to out folder, keep subfolder structure
                dst = out_dir / rel
                ensure_dir(dst.parent)
                try:
                    shutil.copy2(str(img_path), str(dst))
                except Exception as e:
                    print(f"[ERR] copy failed for {img_path}: {e}")
                break

            elif k in (ord('u'), ord('U')):
                if not rows:
                    continue
                last_rel, prev = history.pop() if history else (None, None)
                if last_rel is None:
                    continue
                # remove last occurrence from rows (simple pop)
                last = rows.pop()
                # restore labeled_map
                if prev is None:
                    labeled_map.pop(last_rel, None)
                else:
                    labeled_map[last_rel] = prev
                # rewrite CSV fully for consistency
                rewrite_all(csv_path, rows)
                # refresh overlay (no state change of current image)
                disp = draw_overlay(fit_in_window(img.copy(), args.max_size)[0], [label_text] + HELP)

            elif k in (ord('s'), ord('S')):
                rewrite_all(csv_path, rows)
                print(f"Saved CSV -> {csv_path}")

            elif k in (ord('q'), ord('Q'), 27):
                rewrite_all(csv_path, rows)
                print(f"Saved CSV -> {csv_path}")
                print("Quit.")
                cv2.destroyAllWindows()
                return

            else:
                # refresh periodically
                pass

    # end for
    rewrite_all(csv_path, rows)
    print(f"All done. CSV saved at: {csv_path}")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
