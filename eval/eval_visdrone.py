#!/usr/bin/env python3
"""
VisDrone MOT Evaluation Runner
Mirrors eval_mot17.py but customized for VisDrone MOT-format datasets.

Usage:
    python eval/eval_visdrone.py \
        --data_root dataset/visdrone_mot_format \
        --split val \
        --text_prompt "pedestrian. car. van. bus. truck."
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import shutil
import pandas as pd
from compute_metrics import MotMetricsEvaluator

WORKER_PY = Path(__file__).resolve().parent / "worker.py"

DATASET_DEFAULTS = {
    'visdrone': {'text_prompt': 'pedestrian. car. van. bus. truck.', 'frame_rate': 25},
}

# ----------------------------------------------------------------------
# Utility: symlink image folders (VisDrone has seq/img1)
# ----------------------------------------------------------------------
def create_image_symlinks(data_root, split, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    split_path = os.path.join(data_root, split)
    sequences = sorted([
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ])

    print(f"\n🔗 Creating image symlinks for {len(sequences)} sequences...")
    for seq in sequences:
        src = os.path.abspath(os.path.join(split_path, seq, "img1"))
        dst = os.path.abspath(os.path.join(out_folder, seq))

        if not os.path.exists(src):
            print(f"   ✗ Missing: {src}")
            continue
        if os.path.exists(dst):
            continue

        try:
            os.symlink(src, dst, target_is_directory=True)
            print(f"   ✓ {seq}")
        except OSError:
            shutil.copytree(src, dst)
            print(f"   ✓ {seq} (copied)")

    return sequences


# ----------------------------------------------------------------------
# Utility: copy GTs into output/gt folder (flat format)
# ----------------------------------------------------------------------
def copy_visdrone_gt(data_root, split, out_folder):
    src_split = os.path.join(data_root, split)
    os.makedirs(out_folder, exist_ok=True)

    copied = 0
    for seq in sorted(os.listdir(src_split)):
        seq_gt = os.path.join(src_split, seq, "gt", "gt.txt")
        if os.path.exists(seq_gt):
            dst = os.path.join(out_folder, f"{seq}.txt")
            shutil.copy(seq_gt, dst)
            copied += 1

    print(f"\n📋 Copied {copied} ground-truth files → {out_folder}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VisDrone MOT evaluation using worker.py + motmetrics")
    ap.add_argument('--data_root', required=True, help="Path to VisDrone MOT-format dataset root")
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'], help="Dataset split")
    ap.add_argument('--box_threshold', type=float, default=0.25)
    ap.add_argument('--text_threshold', type=float, default=0.10)
    ap.add_argument('--track_thresh', type=float, default=0.30)
    ap.add_argument('--match_thresh', type=float, default=0.85)
    ap.add_argument('--track_buffer', type=int, default=80)
    ap.add_argument('--tracker', choices=['bytetrack', 'clip'], default='bytetrack')
    ap.add_argument('--detector', choices=['dino', 'florence2'], default='dino')
    ap.add_argument('--text_prompt', type=str, default=None)
    ap.add_argument('--config', type=str,
                   default="groundingdino/config/GroundingDINO_SwinB_cfg.py")
    ap.add_argument('--weights', type=str,
                   default="weights/groundingdino_swinb_cogcoor.pth")
    ap.add_argument('--min_box_area', type=int, default=10)
    ap.add_argument('--frame_rate', type=int, default=30)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--save_video', action='store_true')
    ap.add_argument('--devices', type=str, default="0")
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--outdir', type=str, default=None)

    # CLIP-specific / tracker fusion hyperparams
    ap.add_argument("--lambda_weight", type=float, default=0.25,
                    help="Weight for CLIP embedding cost in IoU+CLIP fusion (0=IoU only, 1=CLIP only)")
    ap.add_argument("--low_thresh", type=float, default=0.1,
                    help="Low detection score threshold for second-stage association")
    ap.add_argument("--text_sim_thresh", type=float, default=0.0,
                    help="Minimum CLIP text similarity for detection gating (0 disables gating)")

    # Booleans for enabling/disabling CLIP fusion in stages
    ap.add_argument("--use_clip_in_high", action="store_true",
                    help="Use CLIP fusion in high-confidence association stage")
    ap.add_argument("--use_clip_in_low", action="store_true",
                    help="Use CLIP fusion in low-confidence association stage")
    ap.add_argument("--use_clip_in_unconf", action="store_true",
                    help="Use CLIP fusion in unconfirmed stage")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Setup defaults
    # ------------------------------------------------------------------
    dataset_name = "visdrone"
    defaults = DATASET_DEFAULTS[dataset_name]
    text_prompt = args.text_prompt or defaults['text_prompt']
    frame_rate = args.frame_rate or defaults['frame_rate']

    # ------------------------------------------------------------------
    # Setup output directories
    # ------------------------------------------------------------------
    if args.outdir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        run_outdir = os.path.join("outputs", f"{dataset_name}_{args.split}_{timestamp}")
    else:
        run_outdir = args.outdir

    os.makedirs(run_outdir, exist_ok=True)
    out_gt = os.path.join(run_outdir, 'gt')
    out_res = os.path.join(run_outdir, 'results')
    temp_images = os.path.join(run_outdir, 'images')
    os.makedirs(out_gt, exist_ok=True)
    os.makedirs(out_res, exist_ok=True)
    os.makedirs(temp_images, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VisDrone Evaluation - Split: {args.split.upper()}")
    print(f"Output directory: {run_outdir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Copy ground truth and link images
    # ------------------------------------------------------------------
    copy_visdrone_gt(args.data_root, args.split, out_gt)
    sequences = create_image_symlinks(args.data_root, args.split, temp_images)

    print(f"\n⚙️  Tracking parameters:")
    print(f"   Text prompt: {text_prompt}")
    print(f"   Frame rate: {frame_rate}")
    print(f"   Box threshold: {args.box_threshold}")
    print(f"   Track threshold: {args.track_thresh}")
    print(f"   Match threshold: {args.match_thresh}")
    print(f"   Track buffer: {args.track_buffer}")
    print(f"   Detector: {args.detector}")
    print(f"   Tracker: {args.tracker}")
    print(f"   Total sequences: {len(sequences)}")

    # ------------------------------------------------------------------
    # Run worker.py for inference/tracking
    # ------------------------------------------------------------------
    cmd = [
        sys.executable, "-u", str(WORKER_PY),
        "--img_folder", temp_images,
        "--all",
        "--out_dir", out_res,
        "--tracker", args.tracker,
        "--box_thresh", str(args.box_threshold),
        "--text_thresh", str(args.text_threshold),
        "--track_thresh", str(args.track_thresh),
        "--match_thresh", str(args.match_thresh),
        "--track_buffer", str(args.track_buffer),
        "--text_prompt", text_prompt,
        "--config", args.config,
        "--weights", args.weights,
        "--min_box_area", str(args.min_box_area),
        "--frame_rate", str(frame_rate),
        "--devices", args.devices,
        "--jobs", str(args.jobs),
        "--detector", args.detector,
    ]
    if args.fp16:
        cmd.append("--use_fp16")
    if args.save_video:
        cmd.append("--save_video")
    # CLIP / fusion hyperparams → forwarded to worker.py / CLIPTracker
    cmd += [
        "--lambda_weight", str(args.lambda_weight),
        "--low_thresh", str(args.low_thresh),
        "--text_sim_thresh", str(args.text_sim_thresh),
    ]

    if args.use_clip_in_high:
        cmd.append("--use_clip_in_high")
    if args.use_clip_in_low:
        cmd.append("--use_clip_in_low")
    if args.use_clip_in_unconf:
        cmd.append("--use_clip_in_unconf")

    print(f"\n🚀 Running tracking on {len(sequences)} sequences...\n")
    print("Command:")
    print(" \\\n  ".join(cmd))
    print()

    try:
        subprocess.check_call(cmd)
        print("\n✓ Tracking complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Worker failed: return code {e.returncode}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Evaluate with motmetrics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("📊 Evaluating Results")
    print(f"{'='*60}\n")

    evaluator = MotMetricsEvaluator(distth=0.5, fmt='mot15-2D')
    df = evaluator.evaluate(out_gt, out_res, verbose=True)

    if df is not None:
        mota = float(df.loc["AVG"].get("mota", float("nan"))) if "AVG" in df.index else df["mota"].mean()
        idf1 = float(df.loc["AVG"].get("idf1", float("nan"))) if "AVG" in df.index else df["idf1"].mean()

        print(f"\n{'='*60}")
        print(f"📈 Summary: MOTA={mota*100:.2f}% | IDF1={idf1*100:.2f}%")
        print(f"{'='*60}\n")
        print(f"OPTUNA:MOTA={mota:.6f} IDF1={idf1:.6f}")
    else:
        print("⚠ No matching GT/RES files for evaluation.")

    print(f"\n✅ Complete! Results saved to: {run_outdir}")


if __name__ == '__main__':
    main()
