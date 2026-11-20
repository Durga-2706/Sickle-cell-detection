"""
Step 1 - Cut individual red blood cells out of the erythrocytesIDB smear images.

The dataset gives, for each blood smear photo, one binary mask per cell type
(circular / elongated / other). Each white blob in a mask is one cell (or a few
touching cells). We crop every blob out of the source photo and label it:

    circular  -> "normal"
    elongated -> "sickle"
    other     -> ignored

Every crop keeps the ID of the smear it came from (manifest.csv). We need this
later to split train/val/test BY SMEAR and avoid data leakage.

Usage:
    python src/prepare_dataset.py --idb2 "<path to erythrocytesIDB2 folder>" --out data
"""
import argparse
import os

import cv2
import pandas as pd

CLASSES = {"circular": "normal", "elongated": "sickle"}  # mask name -> our label
MIN_AREA = 150  # ignore tiny specks (in pixels)
PAD = 8         # extra pixels kept around each cell


def find_file(folder, prefix):
    """Find e.g. 'mask-circular.jpg' or 'mask-circular.JPG' (case-insensitive)."""
    for f in sorted(os.listdir(folder)):
        if (not f.startswith(".") and f.lower().startswith(prefix)
                and f.lower().endswith((".jpg", ".jpeg", ".png"))):
            return os.path.join(folder, f)
    return None


def main(idb2_dir, out_dir, check):
    raw_dir = os.path.join(out_dir, "raw")
    for label in CLASSES.values():
        os.makedirs(os.path.join(raw_dir, label), exist_ok=True)
        for old in os.listdir(os.path.join(raw_dir, label)):
            if old.endswith(".png"):
                os.remove(os.path.join(raw_dir, label, old))

    rows = []
    folders = sorted(d for d in os.listdir(idb2_dir)
                     if os.path.isdir(os.path.join(idb2_dir, d)))

    for folder in folders:
        path = os.path.join(idb2_dir, folder)
        source_id = folder[:2]  # '01', '02', ...
        src_file = find_file(path, "source")
        if src_file is None:
            print("Skipping (no source image):", folder)
            continue
        src = cv2.imread(src_file)
        h, w = src.shape[:2]
        overlay = src.copy()

        for mask_name, label in CLASSES.items():
            mask_file = find_file(path, f"mask-{mask_name}")
            if mask_file is None:
                print(f"Skipping {folder}: no mask-{mask_name}")
                continue
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask.shape != (h, w):
                # Masks can be stored at a different size than the photo.
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            if check and folder == folders[0]:
                color = (0, 255, 0) if label == "normal" else (0, 0, 255)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, max(2, w // 400))

            n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
            k = 0
            for i in range(1, n):  # 0 is the background
                x, y, bw, bh, area = stats[i]
                if area < MIN_AREA:
                    continue
                x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
                x1, y1 = min(x + bw + PAD, w), min(y + bh + PAD, h)
                k += 1
                fname = f"{source_id}_{label}_{k:03d}.png"
                cv2.imwrite(os.path.join(raw_dir, label, fname), src[y0:y1, x0:x1])
                rows.append({"filename": fname, "label": label,
                             "source_id": source_id, "area": int(area)})

        if check and folder == folders[0]:
            cv2.imwrite(os.path.join(out_dir, "alignment_check.png"), overlay)
            print("Saved alignment_check.png - open it and confirm outlines sit on the right cells.")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    print("Total crops:", len(manifest))
    print(manifest["label"].value_counts())
    print("Source smears used:", manifest["source_id"].nunique())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idb2", required=True, help="Path to the erythrocytesIDB2 folder")
    ap.add_argument("--out", default="data", help="Output folder (default: data)")
    ap.add_argument("--check", action="store_true", help="Save an outline overlay for the first smear")
    a = ap.parse_args()
    main(a.idb2, a.out, a.check)
