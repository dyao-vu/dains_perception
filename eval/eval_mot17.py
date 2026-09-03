#!/usr/bin/env python3
"""
Simple MOT evaluation runner - mirrors run_kitti_2gua.py structure.
Uses existing worker.py without modification.

Usage:
    python eval/eval_mot17.py \\
        --data_root dataset/MOT17 \\
        --split train \\
        --detector_suffix DPM
"""
import os
import sys
import glob
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from compute_metrics import MotMetricsEvaluator

WORKER_PY = Path(__file__).resolve().parent / "worker.py"

# Dataset defaults
DATASET_DEFAULTS = {
    'mot17': {'text_prompt': 'person. pedestrian.', 'frame_rate': 30},
    'mot20': {'text_prompt': 'person. pedestrian.', 'frame_rate': 25},
    'mot16': {'text_prompt': 'person. pedestrian.', 'frame_rate': 30},
    'mot15': {'text_prompt': 'person. pedestrian. car.', 'frame_rate': 30},
}

def prepare_mot_gt(data_root, split, out_folder, detector_suffix=None, min_visibility=0.25):
    """
    Convert MOT ground truth to evaluation format.
    MOT GT format: frame,id,x,y,w,h,conf,class,visibility
    Output format: frame,id,x,y,w,h,1,-1,-1,-1
    """
    os.makedirs(out_folder, exist_ok=True)
    
    split_path = os.path.join(data_root, split)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split path not found: {split_path}")
    
    # Find all sequences
    sequences = []
    for seq_dir in sorted(os.listdir(split_path)):
        seq_path = os.path.join(split_path, seq_dir)
        if not os.path.isdir(seq_path):
            continue
        
        # Filter by detector suffix if specified (e.g., DPM, FRCNN, SDP)
        if detector_suffix and not seq_dir.endswith(f"-{detector_suffix}"):
            continue
        
        sequences.append(seq_dir)
    
    print(f"\n📋 Preparing ground truth for {len(sequences)} sequences...")
    
    for seq in sequences:
        gt_file = os.path.join(split_path, seq, "gt", "gt.txt")
        if not os.path.exists(gt_file):
            print(f"   ⚠ No GT file: {seq}")
            continue
        
        out_path = os.path.join(out_folder, f"{seq}.txt")
        
        with open(gt_file) as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                
                frame = int(parts[0])
                tid = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = int(parts[6])  # 0=ignore, 1=consider
                cls = int(parts[7])   # class ID
                vis = float(parts[8])  # visibility [0,1]
                
                # Filter: only consider valid detections
                if conf == 0:
                    continue
                
                # Filter: pedestrian classes only (1=Pedestrian, 2=Person on vehicle, 7=Static person)
                if cls not in (1, 2, 7):
                    continue
                
                # Filter: minimum visibility
                if vis < min_visibility:
                    continue
                
                # Filter: ignore occluders/distractors (class 8-12)
                if cls >= 8:
                    continue
                
                # Write in evaluation format
                f_out.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        
        # Count annotations
        with open(out_path) as f:
            num_lines = sum(1 for _ in f)
        print(f"   ✓ {seq}: {num_lines} annotations")

def create_image_symlinks(data_root, split, out_folder, detector_suffix=None):
    """
    Create symlinks for image folders to match worker.py expected structure.
    Worker expects: img_folder/seq_name/frame.jpg
    MOT has: data_root/split/seq_name/img1/frame.jpg
    """
    os.makedirs(out_folder, exist_ok=True)
    
    split_path = os.path.join(data_root, split)
    # Convert to absolute path for symlinks
    split_path_abs = os.path.abspath(split_path)
    sequences = []
    
    for seq_dir in sorted(os.listdir(split_path)):
        seq_path = os.path.join(split_path, seq_dir)
        if not os.path.isdir(seq_path):
            continue
        
        if detector_suffix and not seq_dir.endswith(f"-{detector_suffix}"):
            continue
        
        sequences.append(seq_dir)
    
    print(f"\n🔗 Creating image symlinks for {len(sequences)} sequences...")
    
    for seq in sequences:
        # Use absolute paths for symlinks
        src = os.path.abspath(os.path.join(split_path_abs, seq, "img1"))
        dst = os.path.abspath(os.path.join(out_folder, seq))
        
        if not os.path.exists(src):
            print(f"   ✗ Image folder not found: {src}")
            continue
        
        # Remove existing symlink/dir if it exists
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.exists(dst):
            continue
        
        try:
            os.symlink(src, dst, target_is_directory=True)
            print(f"   ✓ {seq}")
        except OSError:
            # Fallback to copy if symlink fails
            import shutil
            shutil.copytree(src, dst)
            print(f"   ✓ {seq} (copied)")
    
    return sequences

def parse_kv_list(kv_list):
    out = {}
    for kv in kv_list or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser(description="MOT evaluation using existing worker.py")
    
    # Dataset paths
    ap.add_argument('--data_root', required=True,
                   help="Path to MOT dataset root (e.g., dataset/MOT17)")
    ap.add_argument('--split', default='train', choices=['train', 'val', 'test'],
                   help="Dataset split (train, val, or test)")
    ap.add_argument('--detector_suffix', choices=['DPM', 'FRCNN', 'SDP'], default=None,
                   help="MOT17 only: filter by detector (e.g., DPM for 7 sequences)")
    
    # Model parameters (with MOT-friendly defaults)
    ap.add_argument('--box_threshold', type=float, default=0.35)
    ap.add_argument('--text_threshold', type=float, default=0.12)
    ap.add_argument('--track_thresh', type=float, default=0.45)
    ap.add_argument('--match_thresh', type=float, default=0.85)
    ap.add_argument('--track_buffer', type=int, default=30)
    ap.add_argument('--tracker', choices=['bytetrack', 'clip'], default='bytetrack')
    ap.add_argument('--detector', choices=['dino', 'florence2'], default='dino')
    ap.add_argument('--text_prompt', type=str, default="person.",
                   help="Override default text prompt")
    ap.add_argument('--config', type=str, choices=['groundingdino/config/GroundingDINO_SwinB_cfg.py', 
                                                   'groundingdino/config/GroundingDINO_SwinT_OGC.py'],
                   default="groundingdino/config/GroundingDINO_SwinB_cfg.py")
    ap.add_argument('--weights', type=str,
                   default="weights/groundingdino_swinb_cogcoor.pth")
    ap.add_argument('--min_box_area', type=int, default=10)
    ap.add_argument('--frame_rate', type=int, default=None,
                   help="Override default frame rate")
    ap.add_argument('--min_visibility', type=float, default=0.25,
                   help="Minimum visibility for GT filtering")
    
    # Performance
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--tracker_kv', action='append',
                   help="Extra tracker args as key=val (repeatable)")
    ap.add_argument('--devices', type=str, default="0,1")
    ap.add_argument('--jobs', type=int, default=2)
    
    # Output
    ap.add_argument('--outdir', type=str, default=None)
    
    args = ap.parse_args()
    
    # Detect dataset type from path
    data_root = Path(args.data_root)
    dataset_name = data_root.name.lower()  # MOT17, MOT20, etc.
    
    if dataset_name not in DATASET_DEFAULTS:
        print(f"⚠ Unknown dataset: {dataset_name}, using MOT17 defaults")
        dataset_name = 'mot17'
    
    defaults = DATASET_DEFAULTS[dataset_name]
    
    # Apply defaults
    text_prompt = args.text_prompt or defaults['text_prompt']
    frame_rate = args.frame_rate or defaults['frame_rate']
    
    # Setup output directories
    if args.outdir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        suffix = f"_{args.detector_suffix}" if args.detector_suffix else ""
        run_outdir = os.path.join("outputs", f"{dataset_name}_{args.split}{suffix}_{timestamp}")
    else:
        run_outdir = args.outdir
    
    os.makedirs(run_outdir, exist_ok=True)
    out_gt = os.path.join(run_outdir, 'gt')
    out_res = os.path.join(run_outdir, 'results')
    temp_images = os.path.join(run_outdir, 'images')
    
    print(f"\n{'='*60}")
    print(f"MOT Evaluation: {dataset_name.upper()} - {args.split}")
    if args.detector_suffix:
        print(f"Detector filter: {args.detector_suffix}")
    print(f"{'='*60}")
    print(f"Output directory: {run_outdir}\n")
    
    # Prepare ground truth (train split only)
    if args.split == 'train':
        prepare_mot_gt(
            args.data_root,
            args.split,
            out_gt,
            detector_suffix=args.detector_suffix,
            min_visibility=args.min_visibility
        )
    else:
        print("\n⚠️  Test split: Ground truth not available\n")
    
    # Create image symlinks
    sequences = create_image_symlinks(
        args.data_root,
        args.split,
        temp_images,
        detector_suffix=args.detector_suffix
    )
    
    print(f"\n⚙️  Tracking parameters:")
    print(f"   Text prompt: {text_prompt}")
    print(f"   Frame rate: {frame_rate}")
    print(f"   Box threshold: {args.box_threshold}")
    print(f"   Track threshold: {args.track_thresh}")
    print(f"   Match threshold: {args.match_thresh}")
    print(f"   Track buffer: {args.track_buffer}")
    
    # Build worker.py command (same as KITTI script)
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
    
    for k, v in parse_kv_list(args.tracker_kv).items():
        cmd.extend(["--tracker_kv", f"{k}={v}"])
    
    # Run tracking
    print(f"\n🚀 Running tracking on {len(sequences)} sequences...\n")
    print(f"Full worker command for debugging:")
    print(" \\\n  ".join(cmd))
    print()
    
    try:
        # Don't capture output - let it stream to console
        subprocess.check_call(cmd)
        print("\n✓ Tracking complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Worker failed with return code {e.returncode}")
        print("\nDebugging: Check if results were partially created:")
        print(f"  ls {out_res}/")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Evaluate (train and val splits only)
    if args.split in ['train', 'val']:
        print(f"\n{'='*60}")
        print("📊 Evaluating Results")
        print(f"{'='*60}\n")
        
        evaluator = MotMetricsEvaluator(distth=0.5, fmt='mot15-2D')
        df = evaluator.evaluate(out_gt, out_res, verbose=True)
        
        # Print summary
        try:
            if "AVG" in df.index:
                mota = float(df.loc["AVG"].get("mota", float("nan")))
                idf1 = float(df.loc["AVG"].get("idf1", float("nan")))
            else:
                mota = float(df["mota"].mean())
                idf1 = float(df["idf1"].mean()) if "idf1" in df.columns else float("nan")
            
            print(f"\n{'='*60}")
            print(f"📈 Summary: MOTA={mota*100:.2f}% | IDF1={idf1*100:.2f}%")
            print(f"{'='*60}\n")
            print(f"OPTUNA:MOTA={mota:.6f} IDF1={idf1:.6f}")
        except Exception as e:
            print(f"OPTUNA_ERROR:{e}")
    
    print(f"\n✅ Complete! Results saved to: {run_outdir}")

if __name__ == '__main__':
    main()