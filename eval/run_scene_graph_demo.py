#!/usr/bin/env python3
"""
Quick demo: run scene graph builder on a CARLA scenario.
Prints per-frame graph and saves JSONL + per-frame PNG visualizations.

Usage:
    python eval/run_scene_graph_demo.py
    python eval/run_scene_graph_demo.py --prompt "car." --max_frames 60
    python eval/run_scene_graph_demo.py --scenario scenario_003 --prompt "car." --max_frames 100
"""
import argparse, os, sys, json

# Ensure we run from GroundingDINO root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--scenario",   default="scenario_001")
ap.add_argument("--prompt",     default=None, help="Override prompt (reads gt.json if omitted)")
ap.add_argument("--max_frames", type=int, default=30)
ap.add_argument("--outdir",     default="/tmp/sg_demo")
ap.add_argument("--no_viz",     action="store_true", help="Skip PNG visualization")
demo_args = ap.parse_args()

from worker_clean import Worker, parse_kv_list

CARLA_BASE  = "/isis/home/hasana3/carla_sim/dataset/carla_follow_scenarios"
SCENARIO    = os.path.join(CARLA_BASE, demo_args.scenario)
IMAGES_DIR  = os.path.join(SCENARIO, "images")
GT_JSON     = os.path.join(SCENARIO, "gt.json")
OUTDIR      = demo_args.outdir
OUT_MOT     = os.path.join(OUTDIR, f"{demo_args.scenario}.txt")

os.makedirs(OUTDIR, exist_ok=True)

# Read prompt from gt.json (unless overridden)
with open(GT_JSON) as f:
    gt_data = json.load(f)
prompt = demo_args.prompt if demo_args.prompt else gt_data["meta"]["prompt"]
print(f"Scenario: {demo_args.scenario} | Prompt: '{prompt}'")

# Build a minimal images dir that Worker can read
# Worker expects: img_folder/<seq>/*.png
# CARLA images are already flat in IMAGES_DIR, just symlink
import tempfile, shutil
tmp_img_root = os.path.join(OUTDIR, "images")
os.makedirs(tmp_img_root, exist_ok=True)
link = os.path.join(tmp_img_root, demo_args.scenario)
if not os.path.exists(link):
    os.symlink(os.path.abspath(IMAGES_DIR), link)

# Dummy gt folder Worker needs
dummy_gt = os.path.join(OUTDIR, "_gt_stub")
os.makedirs(os.path.join(dummy_gt, "gt"), exist_ok=True)

# ---- Monkey-patch process_sequence to stop after N frames ----
MAX_FRAMES = demo_args.max_frames

import worker_clean as wc
_orig = wc.Worker.process_sequence

def _patched_process_sequence(self, *, seq, img_folder, gt_folder, out_path,
                               sort_frames=True, video_out_path=None,
                               enable_scene_graph=False):
    import os, cv2
    from scene_graph import SceneGraphBuilder

    seq_path = os.path.join(img_folder, seq)
    frame_files = sorted(
        [f for f in os.listdir(seq_path) if os.path.isfile(os.path.join(seq_path, f))],
        key=wc.parse_frame_id
    )[:MAX_FRAMES]

    sg_builder = SceneGraphBuilder(text_prompt=self.text_prompt)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dummy_gt_root = gt_folder

    with open(out_path, "w") as f_res:
        for idx, frame_name in enumerate(frame_files):
            frame_id = wc.parse_frame_id(frame_name)
            img = cv2.imread(os.path.join(seq_path, frame_name))
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]

            import torch
            from torch.cuda.amp import autocast
            tensor = self.preprocess_frame(img)
            dets = self.predict_detections(img, tensor, orig_h, orig_w)

            if self.referring_filter is not None:
                dets = self.referring_filter.filter(img, dets)

            if self.tracker_type in ("clip", "smartclip"):
                tracks = self.update_tracker_clip(dets, img, orig_h, orig_w)
            else:
                tracks = self.update_tracker(dets, orig_h, orig_w)

            if self.color_gate is not None:
                tracks = self.color_gate.update(tracks, img)

            # Build scene graph
            fg = sg_builder.update(frame_id, tracks, orig_h, orig_w, frame_bgr=img)

            # MOT output
            for t in tracks:
                x, y, w, h = t.tlwh
                if w * h > self.min_box_area:
                    self._write_mot_line(f_res, frame_id, t.track_id,
                                         float(x), float(y), float(w), float(h))

            print(f"\n=== Frame {frame_id} | tracks={fg['num_tracks']} edges={len(fg['edges'])} ===")
            for n in fg["nodes"]:
                print(f"  Node T{n['track_id']:>2}: region={n['region']:<15} "
                      f"size={n['size']:<7} color={n.get('color','?'):<8} "
                      f"conf={n['confidence']:.2f}  motion={n['motion']}")
            for e in fg["edges"]:
                print(f"  Edge T{e['source']}→T{e['target']}: {e['relations']}")

    sg_path = out_path.replace(".txt", "_scene_graphs.jsonl")
    sg_builder.save_jsonl(sg_path)
    summary = sg_builder.get_summary()
    print(f"\n[Summary] frames={summary['total_frames']} "
          f"avg_nodes={summary['avg_nodes_per_frame']} "
          f"avg_edges={summary['avg_edges_per_frame']} "
          f"unique_tracks={summary['total_unique_tracks']}")

    # Pretty-print the last frame as JSON for inspection
    print("\n=== Last frame (JSON) ===")
    print(json.dumps(sg_builder.frames[-1], indent=2))
    return sg_builder

wc.Worker.process_sequence = _patched_process_sequence

# ---- Build worker ----
# Disable color/spatial filters when prompt has no color/spatial keywords
# (e.g. plain "car." prompt) so we get multi-object tracking with edges
prompt_dot = prompt if prompt.endswith(".") else prompt + "."

tracker_kwargs = dict(
    track_thresh=0.45,
    track_buffer=120,
    match_thresh=0.80,
)

worker = Worker(
    tracker_type="bytetrack",
    tracker_kwargs=tracker_kwargs,
    box_thresh=0.40,
    text_thresh=0.80,
    use_fp16=True,
    text_prompt=prompt_dot,
    detector="dino",
    frame_rate=10,
    dataset_type="mot",
    referring_mode="threshold",
    referring_thresh=0.25,
    use_spatial_filter=True,
    use_color_filter=True,
    use_scale_aware_thresh=True,
)

sg = worker.process_sequence(
    seq=demo_args.scenario,
    img_folder=tmp_img_root,
    gt_folder=dummy_gt,
    out_path=OUT_MOT,
    enable_scene_graph=True,
)

jsonl_path = OUT_MOT.replace(".txt", "_scene_graphs.jsonl")
print(f"\nScene graph JSONL: {jsonl_path}")

# ---- Visualize ----
if not demo_args.no_viz:
    viz_dir = os.path.join(OUTDIR, "viz_" + demo_args.scenario)
    os.system(
        f"conda run -n dino_real python {SCRIPT_DIR}/visualize_scene_graph.py "
        f"--jsonl {jsonl_path} --images {IMAGES_DIR} --out {viz_dir}"
    )
    print(f"Visualizations: {viz_dir}/")
