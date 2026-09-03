#!/usr/bin/env python3
"""
Video inference using the worker_simple pipeline:
  GroundingDINO → ByteTrack → SceneGraphBuilder
  → SceneGraphMissionFilter → ColorReIDMatcher → MOT output → annotated video.
"""
import os
import sys
import cv2
import time
import glob
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import DefaultDict, List, Optional, Tuple
from collections import defaultdict

# Add project root and eval/ so worker_simple and scene_graph are importable directly
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "eval"))
from worker_simple import Worker

# ===== Defaults =====
DEFAULT_CONFIG_PATH  = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
DEFAULT_WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
DEFAULT_TEXT_PROMPT  = "red car."

DEFAULT_BOX_THRESHOLD  = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25
DEFAULT_TRACK_THRESH   = 0.45
DEFAULT_MATCH_THRESH   = 0.80
DEFAULT_TRACK_BUFFER   = 120
DEFAULT_MIN_BOX_AREA   = 10


# ===== Frame extraction =====

def extract_frames(video_path: str, frames_dir: str) -> Tuple[int, float, int, int]:
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        cv2.imwrite(os.path.join(frames_dir, f"{n}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()

    if n == 0:
        raise RuntimeError("No frames extracted (empty video?)")
    return n, fps, W, H


# ===== MOT result reading / rendering =====

def read_mot_results(mot_txt: str) -> DefaultDict[int, List[Tuple]]:
    per_frame: DefaultDict[int, List] = defaultdict(list)
    if not os.path.isfile(mot_txt):
        return per_frame
    with open(mot_txt) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            score = float(parts[6]) if len(parts) > 6 else 1.0
            per_frame[int(float(parts[0]))].append((
                int(float(parts[1])),   # track id
                float(parts[2]),        # x
                float(parts[3]),        # y
                float(parts[4]),        # w
                float(parts[5]),        # h
                score,                  # confidence
            ))
    return per_frame


def _load_depth_model(pretrained: str = "Ruicheng/moge-2-vits-normal"):
    """Load MoGe v2 depth model onto GPU."""
    import torch
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")
    model = MoGeModel.from_pretrained(pretrained).cuda().eval()
    return model


def write_tracked_video(frames_dir: str, mot_txt: str, out_path: str,
                        fps: float, size: Tuple[int, int],
                        depth_model=None) -> None:
    import torch

    W, H = size
    per_frame = read_mot_results(mot_txt)
    frame_files = sorted(
        (Path(p) for p in glob.glob(os.path.join(frames_dir, "*.jpg"))
                         + glob.glob(os.path.join(frames_dir, "*.png"))),
        key=lambda p: int(p.stem),
    )

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    for p in frame_files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # --- depth estimation (optional, skipped on empty frames) ---
        frame_tracks = per_frame.get(int(p.stem), [])
        points_np: Optional[np.ndarray] = None
        if depth_model is not None and frame_tracks:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).float() / 255.0
            img_t = img_t.permute(2, 0, 1).cuda()   # [3, H, W]
            with torch.no_grad():
                result = depth_model.infer(img_t, resolution_level=7, use_fp16=True)
            # result['points'] is [H_out, W_out, 3] in camera space (metres)
            points_np = result["points"].cpu().numpy()

        for (tid, x, y, w, h, score) in frame_tracks:
            xi, yi, wi, hi = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (xi, yi), (xi + wi, yi + hi), (0, 255, 0), 2)
            # ID on the top-left
            cv2.putText(img, f"ID:{tid}", (xi, max(0, yi - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Confidence on the top-right of the bbox
            score_text = f"{score:.2f}"
            (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(img, score_text, (xi + wi - tw, max(0, yi - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Distance at bbox centre (bottom-left of bbox)
            if points_np is not None:
                H_out, W_out = points_np.shape[:2]
                cx = x + w / 2
                cy = y + h / 2
                px = int(max(0, min(cx / W * W_out, W_out - 1)))
                py = int(max(0, min(cy / H * H_out, H_out - 1)))
                dist = float(np.linalg.norm(points_np[py, px]))
                dist_text = f"{dist:.1f}m"
                cv2.putText(img, dist_text, (xi, min(H - 4, yi + hi + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        writer.write(img)

    writer.release()


# ===== Main =====

def main():
    ap = argparse.ArgumentParser(
        description="Run worker_simple (DINO + ByteTrack + SceneGraph + MissionFilter + ColorReID) on a video."
    )

    # I/O
    ap.add_argument("--video",    required=True,  help="Input video path")
    ap.add_argument("--output",   required=True,  help="Output annotated mp4 path")
    ap.add_argument("--workdir",  default=None,   help="Temp working dir (default: <output>_work/)")
    ap.add_argument("--keep-frames", action="store_true", help="Keep extracted frames after run")

    # Model
    ap.add_argument("--config",  default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH)

    # Prompt & detection
    ap.add_argument("--text-prompt",    default=DEFAULT_TEXT_PROMPT)
    ap.add_argument("--box-threshold",  type=float, default=DEFAULT_BOX_THRESHOLD)
    ap.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    ap.add_argument("--fp16",           action="store_true")

    # Scale-aware detection
    ap.add_argument("--small-box-area-thresh", type=int, default=5000,
                    help="Area (px²) below which lower detection thresholds apply")
    ap.add_argument("--no-scale-aware",  action="store_true",
                    help="Disable scale-aware thresholding")

    # ByteTrack
    ap.add_argument("--track-threshold", type=float, default=DEFAULT_TRACK_THRESH)
    ap.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESH)
    ap.add_argument("--track-buffer",    type=int,   default=DEFAULT_TRACK_BUFFER)
    ap.add_argument("--min-box-area",    type=int,   default=DEFAULT_MIN_BOX_AREA)

    # Mission filter
    ap.add_argument("--no-mission-filter",    action="store_true",
                    help="Disable SceneGraphMissionFilter")
    ap.add_argument("--mission-filter-hard",  action="store_true",
                    help="Use hard (all-or-nothing) filter mode instead of soft scoring")
    ap.add_argument("--mission-filter-thresh", type=float, default=0.10,
                    help="Soft-mode minimum score to keep a track (default 0.10)")

    # Color re-ID
    ap.add_argument("--no-color-reid",      action="store_true",
                    help="Disable ColorReIDMatcher (track ID recovery after FOV changes)")
    ap.add_argument("--reid-max-lost-frames", type=int, default=25,
                    help="Frames a lost track stays in re-ID graveyard (default 25)")

    # Depth estimation
    ap.add_argument("--depth", action="store_true",
                    help="Enable MoGe-2 monocular depth — displays camera-to-object distance on each bbox")
    ap.add_argument("--depth-model", default="Ruicheng/moge-2-vits-normal",
                    help="MoGe v2 pretrained model ID (default: moge-2-vits-normal)")

    args = ap.parse_args()

    video_path  = os.path.abspath(args.video)
    output_path = os.path.abspath(args.output)

    # Working directory
    work_root = os.path.abspath(args.workdir) if args.workdir \
                else os.path.splitext(output_path)[0] + "_work"
    seq       = "0000"
    seq_dir   = os.path.join(work_root, "frames", seq)
    mot_dir   = os.path.join(work_root, "mot")
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(mot_dir, exist_ok=True)
    mot_file  = os.path.join(mot_dir, f"{seq}.txt")

    # 1) Extract frames
    t0 = time.time()
    num_frames, fps, W, H = extract_frames(video_path, seq_dir)
    print(f"[extract] {num_frames} frames at {fps:.2f} FPS, {W}x{H}  "
          f"({time.time()-t0:.2f}s)")

    # 2) Build worker
    worker = Worker(
        config_path=args.config,
        weights_path=args.weights,
        text_prompt=args.text_prompt,
        box_thresh=args.box_threshold,
        text_thresh=args.text_threshold,
        use_fp16=args.fp16,
        frame_rate=int(round(fps)),
        min_box_area=args.min_box_area,
        # Scale-aware detection
        use_scale_aware_thresh=not args.no_scale_aware,
        small_box_area_thresh=args.small_box_area_thresh,
        # ByteTrack
        tracker_kwargs=dict(
            track_thresh=args.track_threshold,
            match_thresh=args.match_threshold,
            track_buffer=args.track_buffer,
        ),
        # Mission filter
        use_mission_filter=not args.no_mission_filter,
        mission_filter_hard=args.mission_filter_hard,
        mission_filter_thresh=args.mission_filter_thresh,
        # Color re-ID
        use_color_reid=not args.no_color_reid,
        reid_max_lost_frames=args.reid_max_lost_frames,
    )

    # 3) Run pipeline
    t1 = time.time()
    worker.process_sequence(
        seq=seq,
        img_folder=os.path.join(work_root, "frames"),
        gt_folder="",           # no GT needed for inference
        out_path=mot_file,
        sort_frames=True,
        enable_scene_graph=True,
    )
    print(f"[worker]  {num_frames} frames in {time.time()-t1:.2f}s "
          f"({num_frames / max(1e-6, time.time()-t1):.1f} FPS)")

    # 4) Load depth model (optional)
    depth_model = None
    if args.depth:
        print(f"[depth]   loading MoGe-2 model '{args.depth_model}' ...")
        t_dm = time.time()
        depth_model = _load_depth_model(args.depth_model)
        print(f"[depth]   model ready  ({time.time()-t_dm:.2f}s)")

    # 5) Render annotated video
    t2 = time.time()
    write_tracked_video(seq_dir, mot_file, output_path, fps=fps, size=(W, H),
                        depth_model=depth_model)
    print(f"[render]  wrote {output_path}  ({time.time()-t2:.2f}s)")

    # 6) Cleanup
    if not args.keep_frames:
        shutil.rmtree(work_root, ignore_errors=True)

    print(f"[done]    total {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
