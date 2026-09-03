#!/usr/bin/env python3
"""
Convert UA-DETRAC XML annotations to MOT15-style gt.txt files.

Expected input structure (what you currently have):
    <data_root>/
        test/
            gt/
                MVI_XXXX.xml
            MVI_XXXX/
                img1/
                    000001.jpg, ...

Output structure (what eval_uadetrac.py + worker.py expect):
    <data_root>/
        test/
            MVI_XXXX/
                img1/      # already exists
                gt/
                    gt.txt # created by this script
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_single_xml(xml_path: Path, split_dir: Path):
    """
    Parse one MVI_xxxx.xml and write MOT-format gt.txt under:
        <split_dir>/MVI_xxxx/gt/gt.txt
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()  # <sequence>

    seq_name = root.attrib.get("name", xml_path.stem)
    # e.g., "MVI_39031"
    seq_dir = split_dir / seq_name

    # Where we will write MOT GT file
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    out_path = gt_dir / "gt.txt"

    rows = []

    # UA-DETRAC XML structure:
    # <sequence>
    #   <frame num="1" ...>
    #       <target_list>
    #           <target id="1">
    #               <box left="..." top="..." width="..." height="..."/>
    #               <attribute vehicle_type="car" .../>
    #           </target>
    #       </target_list>
    #   </frame>
    # </sequence>
    for frame in root.findall("frame"):
        frame_id = int(frame.attrib["num"])

        tlist = frame.find("target_list")
        if tlist is None:
            continue

        for target in tlist.findall("target"):
            tid = int(target.attrib["id"])

            box = target.find("box")
            if box is None:
                continue

            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])

            if width <= 0 or height <= 0:
                continue

            # If you ever want class-based filtering (car/van/bus/others),
            # you can read it here:
            # attr = target.find("attribute")
            # vehicle_type = attr.attrib.get("vehicle_type", "") if attr is not None else ""

            # MOT15/16-style:
            # frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
            conf = 1
            rows.append((frame_id, tid, left, top, width, height, conf, -1, -1, -1))

    # Sort by frame, then id for cleanliness
    rows.sort(key=lambda r: (r[0], r[1]))

    with out_path.open("w") as f:
        for r in rows:
            f.write(
                f"{int(r[0])},{int(r[1])},"
                f"{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},"
                f"{r[6]},{r[7]},{r[8]},{r[9]}\n"
            )

    print(f"✓ {xml_path.name} → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        required=True,
        help="Path to DETRAC root (where 'test' folder lives, e.g. dataset/DETRAC)",
    )
    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Which split to convert (default: test)",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split
    xml_dir = split_dir / "gt"  # where your MVI_*.xml currently are

    if not xml_dir.is_dir():
        raise SystemExit(f"XML folder not found: {xml_dir}")

    xml_files = sorted(xml_dir.glob("MVI_*.xml"))
    if not xml_files:
        raise SystemExit(f"No MVI_*.xml files found in {xml_dir}")

    print(f"Converting {len(xml_files)} sequences from XML → MOT gt.txt ...\n")

    for xml_path in xml_files:
        convert_single_xml(xml_path, split_dir)

    print("\n✅ Done. You can now run:")
    print(f"  python eval/eval_uadetrac.py --data_root {data_root} --split {args.split}")


if __name__ == "__main__":
    main()
