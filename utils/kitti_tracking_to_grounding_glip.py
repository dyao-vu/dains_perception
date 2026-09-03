#!/usr/bin/env python3
"""
Convert KITTI Tracking dataset to GroundingDINO/ODVG JSON.

Expected structure:
  dataset/kitti/training/
    image_02/0000/000000.png
    label_02/0000.txt
Each label file contains all frames for a single sequence.

Usage:
  python3 utils/kitti_tracking_to_grounding_glip.py \
    --img-dir dataset/kitti/training/image_02 \
    --label-dir dataset/kitti/training/label_02 \
    --out dataset/kitti/training/train_grounding.json

Classes merged (2 total):
  - car: Car, Van
  - pedestrian: Pedestrian, Person_sitting, Cyclist
  - All others ignored.
"""

import argparse
import os
import json
import glob
import pathlib

# ======= CONFIG =======
ORDERED_PROMPT = ["car", "pedestrian"]
CLASS_MAP = {
    "Car": "car",
    "Van": "car",
    "Pedestrian": "pedestrian",
    "Person_sitting": "pedestrian",
    "Cyclist": "pedestrian",
}
IGNORE = {"Truck", "Tram", "Misc"}
# ======================


def parse_label_file(path):
    """Parse KITTI tracking label_02 file."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            frame = int(parts[0])
            cls = parts[2]
            if cls in IGNORE or cls not in CLASS_MAP:
                continue
            x1, y1, x2, y2 = map(float, parts[6:10])
            if x2 <= x1 or y2 <= y1:
                continue
            out.setdefault(frame, []).append(
                (CLASS_MAP[cls], x1, y1, x2, y2)
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True, help="e.g. dataset/kitti/training/image_02")
    ap.add_argument("--label-dir", required=True, help="e.g. dataset/kitti/training/label_02")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    seq_dirs = sorted(glob.glob(os.path.join(args.img_dir, "*")))
    assert seq_dirs, f"No sequences found in {args.img_dir}"

    # Caption and token setup
    caption_tokens, token_spans = [], {}
    idx = 0
    for w in ORDERED_PROMPT:
        caption_tokens.append(w)
        idx += 1
        caption_tokens.append(".")
        idx += 1
        token_spans[w] = (idx - 2, idx - 2)
    caption = " ".join(caption_tokens)

    cat_name_to_id = {"car": 1, "pedestrian": 2}
    cats = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "pedestrian"},
    ]

    images, annotations = [], []
    img_id = 1
    ann_id = 1

    for seq_path in seq_dirs:
        seq = os.path.basename(seq_path)
        label_path = os.path.join(args.label_dir, f"{seq}.txt")
        if not os.path.exists(label_path):
            print(f"Warning: Missing label file for sequence {seq}")
            continue

        seq_labels = parse_label_file(label_path)
        img_files = sorted(glob.glob(os.path.join(seq_path, "*.png")))
        if not img_files:
            print(f"Warning: No images found for sequence {seq}")
            continue

        for img_path in img_files:
            stem = pathlib.Path(img_path).stem
            frame_idx = int(stem)
            images.append({
                "id": img_id,
                "file_name": f"{seq}/{os.path.basename(img_path)}",
                "height": 0,
                "width": 0,
                "caption": caption,
                "tokens_positive": [
                    [token_spans[w][0], token_spans[w][1]] for w in ORDERED_PROMPT
                ],
                "positive_categories": ORDERED_PROMPT,
            })

            if frame_idx in seq_labels:
                for (cls, x1, y1, x2, y2) in seq_labels[frame_idx]:
                    w, h = x2 - x1, y2 - y1
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "bbox": [x1, y1, w, h],
                        "area": float(w * h),
                        "iscrowd": 0,
                        "category_id": cat_name_to_id[cls],
                        "text": cls,
                        "tokens_positive": [
                            [token_spans[cls][0], token_spans[cls][1]]
                        ],
                    })
                    ann_id += 1
            img_id += 1

    data = {
        "info": {"dataset": "KITTI-Tracking", "grounding_prompt": caption},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": cats,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)

    print(f"Wrote {args.out}")
    print(f"Images: {len(images)} | Annotations: {len(annotations)}")


if __name__ == "__main__":
    main()
