#!/usr/bin/env python3
import os
import cv2
import glob
import argparse
import motmetrics as mm
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import torchvision.transforms as T
from groundingdino.util.inference import load_model, predict
from tracker.byte_tracker import BYTETracker
from torch.cuda.amp import autocast
import torch.nn.functional as F
import pandas as pd
import subprocess

# === Static config ===
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "car. pedestrian."
MIN_BOX_AREA = 10
FRAME_RATE = 10

import os, sys, json, tempfile, subprocess

def launch_on_gpu(gpu_id: int, seq, img_folder, res_folder, box_thresh, text_thresh, track_thresh, match_thresh, track_buffer, use_fp16):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return subprocess.Popen(
        [sys.executable, "-u", "eval/worker.py", "--seq", seq, "--img_folder", img_folder, "--res_folder", res_folder, "--box_thresh", str(box_thresh), "--text_thresh", str(text_thresh), "--track_thresh", str(track_thresh), "--match_thresh", str(match_thresh), "--track_buffer", str(track_buffer)],
        env=env
    )

def combine_gt_local(label_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for gt_file in glob.glob(os.path.join(label_folder, "*.txt")):
        seq = os.path.splitext(os.path.basename(gt_file))[0]
        out_path = os.path.join(out_folder, f"{seq}.txt")
        with open(gt_file) as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                parts = line.split()
                frame = int(parts[0]); tid=int(parts[1]); cls=parts[2]
                if cls not in ("Car","Pedestrian"): continue
                x1,y1,x2,y2 = map(float, parts[6:10])
                w,h = x2-x1, y2-y1
                f_out.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

def run_inference_local(img_folder, res_folder, box_thresh, text_thresh, track_thresh, match_thresh, track_buffer, use_fp16=False, jobs=1):
    os.makedirs(res_folder, exist_ok=True)
    device_limit = torch.cuda.device_count()
    print(device_limit)
    job_args = []
    for seq in sorted(os.listdir(img_folder)):
        job_args.append((seq, img_folder, res_folder, box_thresh, text_thresh, track_thresh, match_thresh, track_buffer, use_fp16))
    procs = []
    for gpu, args in enumerate(job_args):
        device = gpu % device_limit
        procs.append(launch_on_gpu(device, *args))
        if len(procs) >= jobs:
            procs[0].wait()
            procs = procs[1:]
    for p in procs:
        p.wait()

def eval_all(gt_folder, res_folder):
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.txt")))
    res_files = sorted(glob.glob(os.path.join(res_folder, "*.txt")))

    all_metrics = [
        'num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr',
        'precision', 'recall', 'num_switches',
        'mostly_tracked', 'mostly_lost', 'num_fragmentations',
        'num_false_positives', 'num_misses', 'num_objects'
    ]

    mh = mm.metrics.create()
    all_summaries = []

    for gt_f, res_f in zip(gt_files, res_files):
        seq = os.path.basename(gt_f)[:-4]
        gt = mm.io.loadtxt(gt_f, fmt='mot15-2D', min_confidence=1)
        res = mm.io.loadtxt(res_f, fmt='mot15-2D')
        acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=0.5)
        summary = mh.compute(acc, metrics=all_metrics, name=seq)
        all_summaries.append(summary)
        print(f"\n===== Sequence: {seq} =====")
        print(mm.io.render_summary(
            summary,
            namemap=mm.io.motchallenge_metric_names,
            formatters=mh.formatters,
        ))

    # Manual concatenation and average row
    df_all = pd.concat(all_summaries)
    avg_row = df_all.mean(numeric_only=True)
    avg_row.name = 'AVG'
    df_all = df_all.append(avg_row)

    print("\n====== AVERAGE ACROSS SEQUENCES ======")
    print(df_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default="/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/image_02")
    parser.add_argument('--labels', default="/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/label_02")
    parser.add_argument('--box_threshold', type=float, default=0.42)
    parser.add_argument('--text_threshold', type=float, default=0.5)
    parser.add_argument('--track_thresh', type=float, default=0.41)
    parser.add_argument('--match_thresh', type=float, default=0.87)
    parser.add_argument('--track_buffer', type=int, default=200)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    # Output directory management
    if args.outdir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        base_outdir = "outputs"
        run_outdir = os.path.join(base_outdir, timestamp)
    else:
        run_outdir = args.outdir
    os.makedirs(run_outdir, exist_ok=True)
    out_gt = os.path.join(run_outdir, 'gt_local')
    out_res = os.path.join(run_outdir, 'inference_results')

    print(f"\nAll outputs for this run will be saved in: {run_outdir}\n")

    combine_gt_local(args.labels, out_gt)
    run_inference_local(
        args.images, out_res,
        args.box_threshold, args.text_threshold,
        args.track_thresh, args.match_thresh,
        args.track_buffer, args.fp16
    )
    eval_all(out_gt, out_res)
