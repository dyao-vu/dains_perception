#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import subprocess
from datetime import datetime
import pandas as pd

from compute_metrics import MotMetricsEvaluator 


# ----------------------------
# KITTI -> MOT text conversion
# ----------------------------
def combine_gt_kitti_to_mot(label_folder: str, out_folder: str) -> None:
    os.makedirs(out_folder, exist_ok=True)
    for gt_file in glob.glob(os.path.join(label_folder, "*.txt")):
        seq = os.path.splitext(os.path.basename(gt_file))[0]
        out_path = os.path.join(out_folder, f"{seq}.txt")
        with open(gt_file) as f_in, open(out_path, "w") as f_out:
            for line in f_in:
                parts = line.split()
                if len(parts) < 10:
                    continue
                frame = int(parts[0]); tid = int(parts[1]); cls = parts[2]
                if cls not in ("Car", "Pedestrian"):
                    continue
                x1, y1, x2, y2 = map(float, parts[6:10])
                w, h = x2 - x1, y2 - y1
                f_out.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


# ----------------------------
# Run worker once for a tracker
# ----------------------------
def run_worker(images_dir: str, out_dir: str, tracker: str, args) -> None:
    worker_py = os.path.join(os.path.dirname(__file__), "worker.py")
    cmd = [
        sys.executable, "-u", worker_py,
        "--img_folder", images_dir,
        "--all",
        "--out_dir", out_dir,
        "--tracker", tracker,
        "--box_thresh", str(args.box_threshold),
        "--text_thresh", str(args.text_threshold),
        "--track_thresh", str(args.track_thresh),
        "--match_thresh", str(args.match_thresh),
        "--track_buffer", str(args.track_buffer),
        "--text_prompt", args.text_prompt,
        "--config", args.config,
        "--weights", args.weights,
        "--min_box_area", str(args.min_box_area),
        "--frame_rate", str(args.frame_rate),
    ]
    if args.fp16:
        cmd.append("--use_fp16")
    if args.devices:
        cmd += ["--devices", args.devices]
    if args.jobs is not None:
        cmd += ["--jobs", str(args.jobs)]

    # extra knobs only for the CLIP tracker
    if tracker == "clip":
        if args.lambda_weight is not None:
            cmd += ["--tracker_kv", f"lambda_weight={args.lambda_weight}"]
        if args.text_sim_thresh is not None:
            cmd += ["--tracker_kv", f"text_sim_thresh={args.text_sim_thresh}"]

    env = os.environ.copy()
    # keep logs quiet
    env["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::FutureWarning"
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env["MPLBACKEND"] = "Agg"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    print(f"[run] {tracker}: writing to {out_dir}")
    subprocess.check_call(cmd, env=env)


def main():
    p = argparse.ArgumentParser(description="Compare BYTE baseline vs CLIP-fused using worker + MotMetricsEvaluator.")
    p.add_argument("--images", required=True, help="Root of sequences (e.g., .../image_02)")
    p.add_argument("--labels", required=True, help="KITTI label_02 folder")
    p.add_argument("--outdir", default=None, help="Top-level output dir (default: outputs_compare/<timestamp>)")

    # Common model / thresholds (defaults = your tuned set)
    p.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinB_cfg.py")
    p.add_argument("--weights", default="weights/groundingdino_swinb_cogcoor.pth")
    p.add_argument("--text_prompt", default="car. pedestrian.")
    p.add_argument("--box_threshold", type=float, default=0.43)
    p.add_argument("--text_threshold", type=float, default=0.11)
    p.add_argument("--track_thresh", type=float, default=0.39)
    p.add_argument("--match_thresh", type=float, default=0.85)
    p.add_argument("--track_buffer", type=int, default=150)
    p.add_argument("--min_box_area", type=int, default=10)
    p.add_argument("--frame_rate", type=int, default=10)
    p.add_argument("--fp16", action="store_true")

    # CLIP-only extras (forwarded to worker via --tracker_kv)
    p.add_argument("--lambda_weight", type=float, default=0.23)
    p.add_argument("--text_sim_thresh", type=float, default=0.02)

    # Dispatch controls passed to worker
    p.add_argument("--devices", default="0,1", help="GPU ids string for worker (e.g., '0,1')")
    p.add_argument("--jobs", type=int, default=2, help="Max concurrent child procs inside worker")

    args = p.parse_args()

    # Output scaffolding
    if args.outdir is None:
        args.outdir = os.path.join("outputs_compare", datetime.now().strftime("%Y-%m-%d_%H%M"))
    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nAll outputs under: {args.outdir}\n")

    out_gt = os.path.join(args.outdir, "gt_mot")
    out_base = os.path.join(args.outdir, "baseline")
    out_clip = os.path.join(args.outdir, "clip_fused")
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(out_clip, exist_ok=True)

    # 1) Prepare GT in MOT format
    combine_gt_kitti_to_mot(args.labels, out_gt)

    # 2) Run worker for each tracker (same thresholds)
    run_worker(args.images, out_base, tracker="bytetrack", args=args)
    run_worker(args.images, out_clip, tracker="clip", args=args)

    # 3) Evaluate with your MotMetricsEvaluator
    evaluator = MotMetricsEvaluator(distth=0.5, fmt="mot15-2D")

    print("\n===== BASELINE (ByteTrack) =====")
    df_base = evaluator.evaluate(out_gt, out_base, verbose=True)
    print(df_base)

    print("\n===== CLIP-FUSED =====")
    df_clip = evaluator.evaluate(out_gt, out_clip, verbose=True)
    print(df_clip)

        # ---- Save individual CSVs ----
    csv_base = os.path.join(args.outdir, "baseline_metrics.csv")
    csv_clip = os.path.join(args.outdir, "clip_fused_metrics.csv")
    csv_comp = os.path.join(args.outdir, "comparison_metrics.csv")
    try:
        df_base.to_csv(csv_base)
        df_clip.to_csv(csv_clip)
    except Exception as e:
        print(f"[warn] could not save individual CSVs: {e}")

    # ---- Build consolidated comparison with deltas (CLIP - BASELINE) ----
    try:
        idx = sorted(set(df_base.index) | set(df_clip.index))
        common_cols = sorted(set(df_base.columns) & set(df_clip.columns))
        base_aligned = df_base.reindex(idx)[common_cols]
        clip_aligned = df_clip.reindex(idx)[common_cols]
        delta = clip_aligned - base_aligned

        pieces, newcols = [], []
        for c in common_cols:
            pieces += [base_aligned[[c]], clip_aligned[[c]], delta[[c]]]
            newcols += [(c, "Baseline"), (c, "CLIP-Fused"), (c, "Δ")]

        comp = pd.concat(pieces, axis=1)
        comp.columns = pd.MultiIndex.from_tuples(newcols)
        comp.to_csv(csv_comp)
        print(f"\nSaved metrics:\n- {csv_base}\n- {csv_clip}\n- {csv_comp}")
    except Exception as e:
        print(f"[warn] could not build/save comparison CSV: {e}")


    # 4) Small summary delta on key metrics (if both present)
    try:
        key_cols = [c for c in ["mota", "idf1", "precision", "recall"] if c in df_base.columns and c in df_clip.columns]
        if key_cols:
            print("\n===== Δ (CLIP - BASELINE) on key metrics =====")
            common = sorted(set(df_base.index) & set(df_clip.index))
            for name in common:
                deltas = {k: (df_clip.loc[name, k] - df_base.loc[name, k]) for k in key_cols}
                print(f"{name}: " + " | ".join(f"{k}: {deltas[k]:+.3f}" for k in key_cols))
    except Exception:
        pass

    print(f"\nDone.\nResults:\n- GT: {out_gt}\n- Baseline: {out_base}\n- CLIP-Fused: {out_clip}\n")


if __name__ == "__main__":
    main()
