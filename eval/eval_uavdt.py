#!/usr/bin/env python3
"""
UAVDT MOT evaluation using worker.py + motmetrics.

Assumes dataset structure like:

  dataset/UAV/
    M_attr/
      train/
        M0203_attr.txt
        ...
      test/
        M0203_attr.txt
        ...
    UAV-benchmark-M/
      M0203/
        000001.jpg
        ...
      ...
    UAV-benchmark-MOTD_v1.0/
      GT/
        M0203_gt.txt
        M0203_gt_ignore.txt
        M0203_gt_whole.txt
        ...

Usage example:

  python eval/eval_uavdt.py \
    --data_root dataset/UAV \
    --split test \
    --img_root UAV-benchmark-M \
    --gt_root UAV-benchmark-MOTD_v1.0/GT \
    --text_prompt "car. truck. bus." \
    --jobs 2 --devices 0,1 \
    --fp16 \
    --weights path/to/weights.pth
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

from compute_metrics import MotMetricsEvaluator

WORKER_PY = Path(__file__).resolve().parent / "worker.py"


# ----------------------------------------------------------------------
# Split handling
# ----------------------------------------------------------------------
def load_split_from_mattr(data_root: Path, split: str):
    """
    Read sequence IDs for a split ('train' or 'test') from M_attr/<split>.
    Returns clean IDs like ['M0203', 'M0701', ...].
    """
    attr_dir = data_root / "M_attr" / split
    if not attr_dir.is_dir():
        raise SystemExit(f"Missing M_attr split dir: {attr_dir}")

    seqs = []
    for f in sorted(attr_dir.glob("M*_attr.txt")):
        base = f.stem  # e.g. 'M0701_attr'
        if base.endswith("_attr"):
            seq = base[:-5]  # 'M0701'
        else:
            seq = base
        seqs.append(seq.strip())

    seqs = sorted(set(seqs))
    if not seqs:
        raise SystemExit(f"No *_attr.txt files found in {attr_dir}")

    return seqs


# ----------------------------------------------------------------------
# GT + image wiring
# ----------------------------------------------------------------------
def copy_uavdt_gt(gt_root: Path, sequences, out_gt: Path):
    """
    Copy per-sequence GT files Mxxxx_gt.txt -> out_gt/Mxxxx.txt.

    Returns the list of sequences for which GT exists.
    """
    out_gt.mkdir(parents=True, exist_ok=True)
    kept, missing = [], []

    for seq in sequences:
        src = gt_root / f"{seq}_gt.txt"
        if src.is_file():
            dst = out_gt / f"{seq}.txt"
            shutil.copy(src, dst)
            kept.append(seq)
        else:
            missing.append(seq)

    print(f"\n📋 Copied {len(kept)} GT files → {out_gt}")
    if missing:
        print("   ⚠ No GT for:", ", ".join(repr(s) for s in missing))

    return kept


def link_images(img_root: Path, sequences, temp_images: Path):
    """
    For each sequence, create a folder under temp_images/ pointing to
    UAV-benchmark-M/<seq> (symlink or copy).

    Returns the list of sequences for which image folders exist.
    """
    temp_images.mkdir(parents=True, exist_ok=True)
    print(f"\n🔗 Creating image symlinks for {len(sequences)} sequences...")

    kept, missing = [], []

    for seq in sequences:
        seq = seq.strip()
        src = img_root / seq
        dst = temp_images / seq

        if not src.is_dir():
            missing.append(seq)
            continue

        if not dst.exists():
            try:
                os.symlink(src, dst, target_is_directory=True)
            except OSError:
                shutil.copytree(src, dst)

        kept.append(seq)
        print(f"   ✓ {seq}")

    if missing:
        print("   ⚠ No image folder for:", ", ".join(repr(s) for s in missing))

    return kept


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="UAVDT MOT evaluation using worker.py + motmetrics"
    )
    ap.add_argument(
        "--data_root",
        required=True,
        help="Base UAVDT path containing M_attr, UAV-benchmark-M and GT folder.",
    )
    ap.add_argument(
        "--img_root",
        default="UAV-benchmark-M",
        help="Relative path from data_root to images root.",
    )
    ap.add_argument(
        "--gt_root",
        default="UAV-benchmark-MOTD_v1.0/GT",
        help="Relative path from data_root to GT root with Mxxxx_gt.txt.",
    )
    ap.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Which UAVDT split to evaluate (from M_attr).",
    )

    # Tracking / detection params
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.10)
    ap.add_argument("--track_thresh", type=float, default=0.30)
    ap.add_argument("--match_thresh", type=float, default=0.85)
    ap.add_argument("--track_buffer", type=int, default=80)
    ap.add_argument("--tracker", choices=["bytetrack", "clip"], default="bytetrack")
    ap.add_argument("--detector", choices=["dino", "florence2"], default="dino")
    ap.add_argument("--text_prompt", type=str, default="car. truck. bus.")
    ap.add_argument(
        "--config",
        type=str,
        default="groundingdino/config/GroundingDINO_SwinB_cfg.py",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="weights/groundingdino_swinb_cogcoor.pth",
    )
    ap.add_argument("--min_box_area", type=int, default=10)
    ap.add_argument("--frame_rate", type=int, default=25)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--save_video", action="store_true")

    # Worker dispatch
    ap.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Comma-separated GPU ids for worker.py dispatch, e.g. '0,1'.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Max concurrent worker processes (<= #devices recommended).",
    )

    # Output
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional fixed output directory. If omitted, a timestamped one is used.",
    )

    args = ap.parse_args()

    # Resolve paths
    data_root = Path(args.data_root).resolve()
    img_root = (data_root / args.img_root).resolve()
    gt_root = (data_root / args.gt_root).resolve()

    if not img_root.is_dir():
        sys.exit(f"No image root dir: {img_root}")
    if not gt_root.is_dir():
        sys.exit(f"No GT root dir: {gt_root}")

    # Determine sequences from official split
    split_seqs = load_split_from_mattr(data_root, args.split)
    print(f"Using {len(split_seqs)} sequences from M_attr/{args.split}: {split_seqs}")

    # Outputs
    if args.outdir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        run_outdir = Path("outputs") / f"uavdt_{args.split}_{ts}"
    else:
        run_outdir = Path(args.outdir)

    out_gt = run_outdir / "gt"
    out_res = run_outdir / "results"
    temp_images = run_outdir / "images"
    out_res.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"UAVDT Evaluation - {args.split.upper()}")
    print(f"Output directory: {run_outdir}")
    print("=" * 60 + "\n")

    # Filter to sequences that have GT, then to those that have images
    seqs = copy_uavdt_gt(gt_root, split_seqs, out_gt)
    seqs = link_images(img_root, seqs, temp_images)

    if not seqs:
        sys.exit("No sequences with both GT and images found.")

    # ------------------------------------------------------------------
    # Run worker.py for tracking
    # ------------------------------------------------------------------
    cmd = [
        sys.executable,
        "-u",
        str(WORKER_PY),
        "--img_folder",
        str(temp_images),
        "--all",
        "--out_dir",
        str(out_res),
        "--tracker",
        args.tracker,
        "--box_thresh",
        str(args.box_threshold),
        "--text_thresh",
        str(args.text_threshold),
        "--track_thresh",
        str(args.track_thresh),
        "--match_thresh",
        str(args.match_thresh),
        "--track_buffer",
        str(args.track_buffer),
        "--text_prompt",
        args.text_prompt,
        "--config",
        args.config,
        "--weights",
        args.weights,
        "--min_box_area",
        str(args.min_box_area),
        "--frame_rate",
        str(args.frame_rate),
        "--devices",
        args.devices,
        "--jobs",
        str(args.jobs),
        "--detector",
        args.detector,
    ]
    if args.fp16:
        cmd.append("--use_fp16")
    if args.save_video:
        cmd.append("--save_video")

    print("\n🚀 Running tracking on sequences...\n")
    print(" \\\n  ".join(cmd))
    print()

    subprocess.check_call(cmd)

    # ------------------------------------------------------------------
    # Evaluate with motmetrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 Evaluating Results")
    print("=" * 60 + "\n")

    evaluator = MotMetricsEvaluator(distth=0.5, fmt="mot15-2D")
    df = evaluator.evaluate(str(out_gt), str(out_res), verbose=True)

    if df is not None and not df.empty:
        if "AVG" in df.index:
            mota = float(df.loc["AVG"].get("mota", float("nan")))
            idf1 = float(df.loc["AVG"].get("idf1", float("nan")))
        else:
            mota = float(df["mota"].mean())
            idf1 = float(df["idf1"].mean())

        print("\n" + "=" * 60)
        print(f"📈 Summary: MOTA={mota * 100:.2f}% | IDF1={idf1 * 100:.2f}%")
        print("=" * 60 + "\n")
        print(f"OPTUNA:MOTA={mota:.6f} IDF1={idf1:.6f}")
    else:
        print("⚠ No matching GT/RES files for evaluation.")

    print(f"\n✅ Complete! Results saved to: {run_outdir}")


if __name__ == "__main__":
    main()
