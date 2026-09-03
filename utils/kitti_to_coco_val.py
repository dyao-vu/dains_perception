#!/usr/bin/env python3
import os
import json
import argparse

def kitti_to_coco(img_dir, label_dir, out_json):
    categories = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "pedestrian"}
    ]
    images, annotations = [], []
    ann_id = 0
    img_id = 0

    for seq_folder in sorted(os.listdir(img_dir)):
        seq_path = os.path.join(img_dir, seq_folder)
        if not os.path.isdir(seq_path):
            continue

        label_path = os.path.join(label_dir, f"{seq_folder}.txt")
        if not os.path.exists(label_path):
            print(f"[WARN] Missing label file for {seq_folder}, skipping")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        frame_to_objs = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            frame = int(parts[0])
            cls = parts[2].lower()
            if cls not in ["car", "pedestrian", "cyclist"]:
                continue
            if cls == "cyclist":
                cls = "pedestrian"

            x1, y1, x2, y2 = map(float, parts[6:10])
            w, h = x2 - x1, y2 - y1
            ann = {
                "id": ann_id,
                "image_id": frame,
                "category_id": 1 if cls == "car" else 2,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            }
            frame_to_objs.setdefault(frame, []).append(ann)
            ann_id += 1

        # Add image entries
        for frame in sorted(frame_to_objs.keys()):
            img_name = f"{seq_folder}/{frame:06d}.png"
            images.append({
                "id": img_id,
                "file_name": img_name,
                "width": 0,
                "height": 0
            })
            for ann in frame_to_objs[frame]:
                ann["image_id"] = img_id
                annotations.append(ann)
            img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"Wrote {out_json}")
    print(f"Images: {len(images)}, Annotations: {len(annotations)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    kitti_to_coco(args.img_dir, args.label_dir, args.out)
