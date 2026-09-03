#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from compute_metrics import MotMetricsEvaluator

WORKER_PY = Path(__file__).resolve().parent / "worker.py"

def combine_gt_local(label_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for gt_file in glob.glob(os.path.join(label_folder, "*.txt")):
        seq = os.path.splitext(os.path.basename(gt_file))[0]
        out_path = os.path.join(out_folder, f"{seq}.txt")
        with open(gt_file) as f_in, open(out_path, 'w') as f_out:
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

def parse_kv_list(kv_list):
    out = {}
    for kv in kv_list or []:
        if "=" not in kv: continue
        k, v = kv.split("=", 1)
        out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--box_threshold', type=float, default=0.43)
    ap.add_argument('--text_threshold', type=float, default=0.11)
    ap.add_argument('--track_thresh', type=float, default=0.39)
    ap.add_argument('--match_thresh', type=float, default=0.87)
    ap.add_argument('--track_buffer', type=int, default=150)
    ap.add_argument('--tracker', choices=['bytetrack', 'clip'], default='bytetrack')
    ap.add_argument('--detector', choices=['dino', 'florence2'], default='dino')
    ap.add_argument('--text_prompt', type=str, default="car. pedestrian.")
    ap.add_argument('--config', type=str, default="groundingdino/config/GroundingDINO_SwinB_cfg.py")
    ap.add_argument('--weights', type=str, default="weights/groundingdino_swinb_cogcoor.pth")
    ap.add_argument('--min_box_area', type=int, default=10)
    ap.add_argument('--frame_rate', type=int, default=10)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--tracker_kv', action='append', help="extra tracker args as key=val (repeatable)")
    ap.add_argument('--devices', type=str, default="0,1")
    ap.add_argument('--jobs', type=int, default=2)
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

    # outputs
    if args.outdir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        run_outdir = os.path.join("outputs", timestamp)
    else:
        run_outdir = args.outdir
    os.makedirs(run_outdir, exist_ok=True)
    out_gt = os.path.join(run_outdir, 'gt_local')
    out_res = os.path.join(run_outdir, 'inference_results')

    print(f"\nAll outputs for this run will be saved in: {run_outdir}\n")

    combine_gt_local(args.labels, out_gt)

    # call worker.py in eval/ by absolute path
    cmd = [
        sys.executable, "-u", str(WORKER_PY),
        "--img_folder", args.images,
        "--all",
        "--out_dir", out_res,
        "--tracker", args.tracker,
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
        "--devices", args.devices,
        "--jobs", str(args.jobs),
        "--detector", args.detector,
    ]
    if args.fp16:
        cmd.append("--use_fp16")
    
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
    
    for k, v in parse_kv_list(args.tracker_kv).items():
        cmd.extend(["--tracker_kv", f"{k}={v}"])

    subprocess.check_call(cmd)

    evaluator = MotMetricsEvaluator(distth=0.5, fmt='mot15-2D')
    df = evaluator.evaluate(out_gt, out_res, verbose=True)
    try:
        if "AVG" in df.index:
            mota = float(df.loc["AVG"].get("mota", float("nan")))
            idf1 = float(df.loc["AVG"].get("idf1", float("nan")))
        else:
            mota = float(df["mota"].mean())
            idf1 = float(df["idf1"].mean()) if "idf1" in df.columns else float("nan")
        print(f"OPTUNA:MOTA={mota:.6f} IDF1={idf1:.6f}")
    except Exception as e:
        print(f"OPTUNA_ERROR:{e}")


if __name__ == '__main__':
    main()