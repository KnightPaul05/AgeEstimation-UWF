#!/usr/bin/env python3
"""
Disc Annotator & Cropper (OpenCV)

Author: ChatGPT (custom-made for Paul's UWF fundus project)
Source: Original code written for this project (no external code copy/paste).

Features
--------
1) annotate: Iterate through images, draw a bounding box around the optic disc with the mouse, and save to CSV.
   - Resume from an existing CSV (skips images already labeled)
   - Keyboard shortcuts shown on-screen
   - Works on large images by auto-fitting to window; saves coordinates in ORIGINAL pixel space
2) crop: Use the CSV of bounding boxes to extract fixed-size crops centered on each box.
   - Pads with black if the crop goes out of image bounds

Dependencies: Python 3.8+, OpenCV (cv2), numpy
Install: pip install opencv-python numpy

Usage (Windows PowerShell examples)
-----------------------------------
# Annotate images in a folder and save CSV alongside
python .\disc_annotator.py annotate `
  --img-dir "C:\\Users\\paulg\\Desktop\\DeepLearning\\Ophthalmology_project\\data\\images" `
  --csv "C:\\Users\\paulg\\Desktop\\DeepLearning\\Ophthalmology_project\\data\\disc_bboxes.csv"

# Later: Crop fixed-size patches centered on the annotated box centers
python .\disc_annotator.py crop `
  --img-dir "C:\\Users\\paulg\\Desktop\\DeepLearning\\Ophthalmology_project\\data\\images" `
  --csv "C:\\Users\\paulg\\Desktop\\DeepLearning\\Ophthalmology_project\\data\\disc_bboxes.csv" `
  --out-dir "C:\\Users\\paulg\\Desktop\\DeepLearning\\Ophthalmology_project\\data\\cropped" `
  --crop-width 384 --crop-height 384

Keyboard (annotation)
---------------------
- Left click & drag: draw rectangle
- SPACE/ENTER: save current rectangle and go to next image
- N: next image (skip without saving a box)
- P/BACKSPACE: previous image
- R/DEL: remove current rectangle
- S: save (without moving)
- Q/ESC: quit (progress auto-saved)

CSV format
----------
filename,x,y,w,h  (absolute pixels in original image)

Notes
-----
- If your images are huge, the viewer scales them to fit; saved coordinates are mapped back to original size.
- You can re-run annotate with the same CSV: already-labeled images are skipped; you can still navigate back and edit.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ----------------------------
# Helpers for CSV persistence
# ----------------------------

BBox = Tuple[int, int, int, int]


def read_csv(csv_path: Path) -> Dict[str, BBox]:
    boxes: Dict[str, BBox] = {}
    if not csv_path.exists():
        return boxes
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                filename = row['filename']
                x = int(float(row['x']))
                y = int(float(row['y']))
                w = int(float(row['w']))
                h = int(float(row['h']))
                boxes[filename] = (x, y, w, h)
            except Exception:
                continue
    return boxes


def write_csv(csv_path: Path, boxes: Dict[str, BBox]) -> None:
    tmp = csv_path.with_suffix('.tmp')
    fieldnames = ['filename', 'x', 'y', 'w', 'h']
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fname, (x, y, w, h) in sorted(boxes.items()):
            writer.writerow({'filename': fname, 'x': x, 'y': y, 'w': w, 'h': h})
    tmp.replace(csv_path)


# ----------------------------
# Annotation UI
# ----------------------------

class Annotator:
    def __init__(self, img_dir: Path, csv_path: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.boxes = read_csv(csv_path)

        self.images: List[Path] = sorted([
            p for p in img_dir.iterdir() if p.suffix.lower() in exts
        ])
        if not self.images:
            print(f"No images found in {img_dir}")
            sys.exit(1)

        # Determine starting index (first unlabeled if possible)
        start = 0
        for i, p in enumerate(self.images):
            if p.name not in self.boxes:
                start = i
                break
        self.idx = start

        # UI state
        self.window = 'Disc Annotator'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        self.view_w = 1600
        self.view_h = 1000
        cv2.resizeWindow(self.window, self.view_w, self.view_h)

        self.dragging = False
        self.start_pt: Optional[Tuple[int, int]] = None
        self.current_rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h in DISPLAY coords

        self.orig_img: Optional[np.ndarray] = None
        self.disp_img: Optional[np.ndarray] = None
        self.scale: float = 1.0
        self.offset: Tuple[int, int] = (0, 0)

        cv2.setMouseCallback(self.window, self._on_mouse)

    # --------- coordinate mapping between display and original ---------
    def _fit_to_window(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        H, W = img.shape[:2]
        scale = min(self.view_w / W, self.view_h / H, 1.0)
        new_W, new_H = int(W * scale), int(H * scale)
        resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        canvas = np.zeros((self.view_h, self.view_w, 3), dtype=np.uint8)
        ox = (self.view_w - new_W) // 2
        oy = (self.view_h - new_H) // 2
        canvas[oy:oy + new_H, ox:ox + new_W] = resized
        return canvas, scale, (ox, oy)

    def _disp_to_orig_rect(self, rect_disp: Tuple[int, int, int, int]) -> BBox:
        x, y, w, h = rect_disp
        ox, oy = self.offset
        x0 = max(0, x - ox)
        y0 = max(0, y - oy)
        # map to original
        x0 = int(round(x0 / self.scale))
        y0 = int(round(y0 / self.scale))
        w0 = int(round(w / self.scale))
        h0 = int(round(h / self.scale))
        # clamp to image bounds
        H, W = self.orig_img.shape[:2]
        x0 = int(np.clip(x0, 0, max(0, W - 1)))
        y0 = int(np.clip(y0, 0, max(0, H - 1)))
        w0 = int(np.clip(w0, 1, W - x0))
        h0 = int(np.clip(h0, 1, H - y0))
        return (x0, y0, w0, h0)

    def _orig_to_disp_rect(self, rect_orig: BBox) -> Tuple[int, int, int, int]:
        x, y, w, h = rect_orig
        ox, oy = self.offset
        x1 = int(round(x * self.scale)) + ox
        y1 = int(round(y * self.scale)) + oy
        w1 = int(round(w * self.scale))
        h1 = int(round(h * self.scale))
        return (x1, y1, w1, h1)

    # ---------------- mouse & drawing ----------------
    def _on_mouse(self, event, x, y, flags, param):
        if self.disp_img is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.start_pt is not None:
            x0, y0 = self.start_pt
            x1, y1 = x, y
            rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self.current_rect = rect
            self._refresh()
        elif event == cv2.EVENT_LBUTTONUP and self.dragging and self.start_pt is not None:
            self.dragging = False
            x0, y0 = self.start_pt
            x1, y1 = x, y
            rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self.current_rect = rect
            self._refresh()

    def _drawHUD(self, img: np.ndarray, text_lines: List[str]) -> None:
        y = 24
        for t in text_lines:
            cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y += 24

    def _refresh(self):
        if self.orig_img is None:
            return
        disp = self.disp_img.copy()
        # Draw existing box if any
        cur_path = self.images[self.idx]
        if cur_path.name in self.boxes:
            x, y, w, h = self._orig_to_disp_rect(self.boxes[cur_path.name])
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 200, 0), 2)
        # Draw current dragging box
        if self.current_rect is not None and self.current_rect[2] > 1 and self.current_rect[3] > 1:
            x, y, w, h = self.current_rect
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 180, 255), 2)

        info = [
            f"Image {self.idx + 1}/{len(self.images)}: {cur_path.name}",
            "[Mouse] drag = draw box | [SPACE/ENTER] save+next | [N] next | [P/BS] prev",
            "[S] save | [R/DEL] clear | [Q/ESC] quit",
        ]
        self._drawHUD(disp, info)
        cv2.imshow(self.window, disp)

    def _load_image(self, idx: int):
        path = self.images[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        self.orig_img = img
        disp, scale, offset = self._fit_to_window(img)
        self.disp_img = disp
        self.scale = scale
        self.offset = offset

        # If we have a saved box, pre-load it into current_rect (display coords)
        if path.name in self.boxes:
            self.current_rect = self._orig_to_disp_rect(self.boxes[path.name])
        else:
            self.current_rect = None
        self._refresh()

    def _save_current(self) -> bool:
        if self.current_rect is None or self.current_rect[2] < 2 or self.current_rect[3] < 2:
            return False
        rect_orig = self._disp_to_orig_rect(self.current_rect)
        fname = self.images[self.idx].name
        self.boxes[fname] = rect_orig
        write_csv(self.csv_path, self.boxes)
        return True

    def _clear_current(self):
        fname = self.images[self.idx].name
        if fname in self.boxes:
            del self.boxes[fname]
            write_csv(self.csv_path, self.boxes)
        self.current_rect = None
        self._refresh()

    def run(self):
        self._load_image(self.idx)
        while True:
            key = cv2.waitKey(50) & 0xFFFF
            if key == 0xFF:  # no key
                continue
            # Normalize some keys
            if key in (13, 32):  # ENTER or SPACE
                saved = self._save_current()
                if not saved:
                    print("No rectangle to save for this image.")
                if self.idx < len(self.images) - 1:
                    self.idx += 1
                    self._load_image(self.idx)
                else:
                    print("Reached last image.")
                    self._refresh()
            elif key in (ord('n'), ord('N')):
                if self.idx < len(self.images) - 1:
                    self.idx += 1
                    self._load_image(self.idx)
                else:
                    print("Reached last image.")
            elif key in (ord('p'), ord('P'), 8):  # 'p' or Backspace
                if self.idx > 0:
                    self.idx -= 1
                    self._load_image(self.idx)
                else:
                    print("At first image.")
            elif key in (ord('s'), ord('S')):
                if self._save_current():
                    print(f"Saved: {self.images[self.idx].name}")
                else:
                    print("Draw a rectangle first.")
            elif key in (ord('r'), ord('R'), 3014656):  # DEL on some systems
                self._clear_current()
                print("Cleared current bounding box.")
            elif key in (ord('q'), ord('Q'), 27):  # ESC
                print("Quitting. Progress saved.")
                break
            else:
                # Unmapped key; just refresh screen (keeps UI responsive)
                self._refresh()
        cv2.destroyAllWindows()


# ----------------------------
# Cropping logic
# ----------------------------

def crop_centered(img: np.ndarray, center_xy: Tuple[float, float], out_w: int, out_h: int) -> np.ndarray:
    H, W = img.shape[:2]
    cx, cy = center_xy
    # Desired integer box (top-left based)
    x0 = int(round(cx - out_w / 2))
    y0 = int(round(cy - out_h / 2))
    x1 = x0 + out_w
    y1 = y0 + out_h

    # Create output canvas
    out = np.zeros((out_h, out_w, 3), dtype=img.dtype)

    # Compute intersection with image
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(W, x1)
    sy1 = min(H, y1)

    if sx1 <= sx0 or sy1 <= sy0:
        return out  # completely outside; return black

    # Corresponding region in the output canvas
    dx0 = sx0 - x0
    dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out


def run_crop(img_dir: Path, csv_path: Path, out_dir: Path, crop_w: int, crop_h: int) -> None:
    boxes = read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    images: List[Path] = sorted([
        p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    ])

    missing = 0
    done = 0
    for p in images:
        if p.name not in boxes:
            missing += 1
            continue
        x, y, w, h = boxes[p.name]
        cx = x + w / 2.0
        cy = y + h / 2.0
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: cannot read {p}")
            continue
        crop = crop_centered(img, (cx, cy), crop_w, crop_h)
        out_path = out_dir / p.name
        cv2.imwrite(str(out_path), crop)
        done += 1

    print(f"Cropped {done} images to {out_dir}")
    if missing:
        print(f"Note: {missing} images had no bbox in CSV; skipped.")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Optic disc annotation and cropping tool")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_ann = sub.add_parser('annotate', help='Launch interactive annotator')
    p_ann.add_argument('--img-dir', type=Path, required=True, help='Folder containing images')
    p_ann.add_argument('--csv', type=Path, required=True, help='CSV path to save/load bounding boxes')

    p_crop = sub.add_parser('crop', help='Crop fixed-size patches centered on annotated boxes')
    p_crop.add_argument('--img-dir', type=Path, required=True, help='Folder containing images')
    p_crop.add_argument('--csv', type=Path, required=True, help='CSV of bounding boxes')
    p_crop.add_argument('--out-dir', type=Path, required=True, help='Where to save cropped images')
    p_crop.add_argument('--crop-width', type=int, required=True, help='Output crop width in pixels')
    p_crop.add_argument('--crop-height', type=int, required=True, help='Output crop height in pixels')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == 'annotate':
        ann = Annotator(args.img_dir, args.csv)
        ann.run()
    elif args.cmd == 'crop':
        run_crop(args.img_dir, args.csv, args.out_dir, args.crop_width, args.crop_height)


if __name__ == '__main__':
    main()
