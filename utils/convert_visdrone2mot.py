import os
import csv
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
# Input dataset root (VisDrone official folders)
DATA_ROOT = Path("/isis/home/hasana3/vlmtest/GroundingDINO/dataset/visdrone")
OUT_ROOT = Path("/isis/home/hasana3/vlmtest/GroundingDINO/dataset/visdrone_mot_format")

# Classes of interest (VisDrone category IDs)
# 1: pedestrian, 2: people, 3: car, 4: van, 5: truck, 6: bus
CLASSES_TO_KEEP = {1, 2, 3, 4, 5, 6}

# Split subfolders to process
SPLITS = ["train", "val", "test"]  # adjust if missing val/test locally

# Map our split names to VisDrone folder names
SPLIT_MAP = {
    "train": "VisDrone2019-MOT-train",
    "val": "VisDrone2019-MOT-val",        # optional, if available
    "test": "VisDrone2019-MOT-test-dev",  # VisDrone’s test folder
}

# ============================================================
# CONVERSION
# ============================================================

def convert_split(split_name):
    if split_name not in SPLIT_MAP:
        print(f"⚠ Unknown split name: {split_name}")
        return

    in_split = DATA_ROOT / SPLIT_MAP[split_name]
    out_split = OUT_ROOT / split_name

    ann_dir = in_split / "annotations"
    seq_dir = in_split / "sequences"

    if not ann_dir.exists():
        print(f"❌ Missing annotations for {split_name}: {ann_dir}")
        return
    if not seq_dir.exists():
        print(f"❌ Missing sequences for {split_name}: {seq_dir}")
        return

    os.makedirs(out_split, exist_ok=True)
    os.makedirs(out_split / "images", exist_ok=True)

    print(f"\n🚀 Converting {split_name.upper()} split...")
    for ann_file in sorted(ann_dir.glob("*.txt")):
        seq_name = ann_file.stem
        out_seq_dir = out_split / seq_name
        gt_dir = out_seq_dir / "gt"
        os.makedirs(gt_dir, exist_ok=True)

        out_gt_path = gt_dir / "gt.txt"
        num_kept = 0

        with open(ann_file, newline='') as f_in, open(out_gt_path, "w", newline='') as f_out:
            reader = csv.reader(f_in)
            for r in reader:
                if len(r) < 10:
                    continue
                frame, tid, x, y, w, h, score, cls, trunc, occ = map(float, r[:10])
                cls = int(cls)
                if cls not in CLASSES_TO_KEEP or w <= 0 or h <= 0:
                    continue
                f_out.write(f"{int(frame)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
                num_kept += 1

        if num_kept > 0:
            # link or copy image directory
            src = seq_dir / seq_name
            dst = out_split / "images" / seq_name
            if not dst.exists():
                try:
                    os.symlink(src.resolve(), dst)
                except OSError:
                    import shutil
                    shutil.copytree(src, dst)
            print(f"   ✓ {seq_name}: {num_kept} objects, linked images.")
        else:
            print(f"   ⚠ {seq_name}: no valid detections kept.")

for split in SPLITS:
    convert_split(split)

print("\n✅ All splits converted! Output structure:")
print(OUT_ROOT)
