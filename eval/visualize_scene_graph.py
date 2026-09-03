#!/usr/bin/env python3
"""
Visualize scene graphs from a JSONL file produced by SceneGraphBuilder.

For each frame in the JSONL:
  - Draws bounding boxes coloured by track ID
  - Draws directed arrows between nodes that share an edge
  - Annotates each arrow with the relation labels
  - Saves as PNG in --out directory

Usage:
    python eval/visualize_scene_graph.py \
        --jsonl /tmp/sg_demo/scenario_001_scene_graphs.jsonl \
        --images /isis/home/hasana3/carla_sim/dataset/carla_follow_scenarios/scenario_001/images \
        --out   /tmp/sg_demo/viz_scenario_001

    # Only render specific frames (0-indexed positions in the JSONL):
    python eval/visualize_scene_graph.py ... --frames 0 5 10 15
"""

import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np


# ── colour palette (one per track ID) ───────────────────────────────────────
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
]

def _track_color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


# ── image loader ─────────────────────────────────────────────────────────────
def _load_image(images_dir: str, frame_id: int):
    """Try common zero-padded filename patterns."""
    for fmt in [f"{frame_id:06d}.png", f"{frame_id:06d}.jpg",
                f"{frame_id:08d}.png", f"{frame_id:08d}.jpg",
                f"{frame_id}.png",     f"{frame_id}.jpg"]:
        p = os.path.join(images_dir, fmt)
        if os.path.isfile(p):
            img = cv2.imread(p)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


# ── single-frame render ──────────────────────────────────────────────────────
def render_frame(frame_graph: dict, images_dir: str, out_dir: str):
    frame_id  = frame_graph["frame_id"]
    nodes     = frame_graph["nodes"]
    edges     = frame_graph["edges"]
    prompt    = frame_graph.get("prompt", "")

    img = _load_image(images_dir, frame_id)
    if img is None:
        print(f"  [viz] Frame {frame_id}: image not found, skipping.")
        return

    H, W = img.shape[:2]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(20, 9),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    ax_img, ax_info = axes

    # ── left panel: image with overlays ─────────────────────────────────────
    ax_img.imshow(img)
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)          # image coords (y down)
    ax_img.axis("off")
    ax_img.set_title(
        f"Frame {frame_id}  |  prompt: \"{prompt}\"  |  "
        f"{len(nodes)} nodes  {len(edges)} edges",
        fontsize=11, pad=6,
    )

    # bbox centres for drawing edges
    centers = {}

    for node in nodes:
        tid = node["track_id"]
        x, y, w, h = node["bbox_tlwh"]
        color = _track_color(tid)
        cx, cy = x + w / 2, y + h / 2
        centers[tid] = (cx, cy)

        # bounding box rectangle
        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=2.5, edgecolor=color, facecolor="none",
        )
        ax_img.add_patch(rect)

        # node label (inside box, top-left corner)
        label_lines = [
            f"T{tid}",
            node.get("color", "?"),
            node["size"],
            node["motion"],
            f"conf={node['confidence']:.2f}",
        ]
        label = "\n".join(label_lines)
        ax_img.text(
            x + 3, y + 4, label,
            fontsize=6.5, color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.75),
        )

    # draw directed edges as arrows between bbox centres
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src not in centers or tgt not in centers:
            continue

        x1, y1 = centers[src]
        x2, y2 = centers[tgt]

        ax_img.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color="yellow",
                lw=1.8,
                mutation_scale=14,
                connectionstyle="arc3,rad=0.08",
            ),
            zorder=5,
        )

        # relation label at arrow midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        rel_str = "\n".join(edge["relations"][:3])   # at most 3 lines
        ax_img.text(
            mx, my, rel_str,
            fontsize=6, color="yellow", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55),
            zorder=6,
        )

    # ── right panel: node attribute table ───────────────────────────────────
    ax_info.axis("off")
    ax_info.set_title("Node attributes", fontsize=10)

    rows = []
    for node in nodes:
        tid = node["track_id"]
        rows.append([
            f"T{tid}",
            node["region"],
            node["size"],
            node.get("color", "—"),
            node["motion"],
            f"{node['confidence']:.2f}",
            str(node["tracklet_len"]),
        ])

    if rows:
        col_labels = ["ID", "region", "size", "color", "motion", "conf", "age"]
        table = ax_info.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.6)

        # colour the ID cells to match bbox colours
        for i, node in enumerate(nodes, start=1):
            cell = table[i, 0]
            cell.set_facecolor(_track_color(node["track_id"]))
            cell.set_text_props(color="white", weight="bold")

    if edges:
        edge_lines = []
        for e in edges:
            rels = ", ".join(e["relations"])
            edge_lines.append(f"T{e['source']}→T{e['target']}: {rels}")
        edge_text = "\n".join(edge_lines)
        ax_info.text(
            0.05, 0.02, edge_text,
            transform=ax_info.transAxes,
            fontsize=7.5, va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(out_dir, f"frame_{frame_id:06d}.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",   required=True, help="Scene graph JSONL file")
    ap.add_argument("--images",  required=True, help="Directory with frame images")
    ap.add_argument("--out",     required=True, help="Output directory for PNGs")
    ap.add_argument("--frames",  nargs="*", type=int, default=None,
                    help="0-indexed frame positions to render (default: all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.jsonl) as f:
        frame_graphs = [json.loads(line) for line in f if line.strip()]

    if args.frames is not None:
        frame_graphs = [frame_graphs[i] for i in args.frames if i < len(frame_graphs)]

    print(f"Rendering {len(frame_graphs)} frames → {args.out}/")
    saved = []
    for fg in frame_graphs:
        p = render_frame(fg, args.images, args.out)
        if p:
            saved.append(p)
            print(f"  saved {os.path.basename(p)}"
                  f"  (nodes={fg['num_tracks']}, edges={len(fg['edges'])})")

    print(f"\nDone. {len(saved)} PNGs in {args.out}/")


if __name__ == "__main__":
    main()
