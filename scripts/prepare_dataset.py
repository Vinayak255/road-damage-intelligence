"""
Dataset preparation script for RDD2022.

Usage:
    python scripts/prepare_dataset.py --source path/to/rdd2022 --output datasets/rdd2022

Responsibilities:
    - Validate image/annotation pairs
    - Convert XML annotations to YOLO format if needed
    - Create train/val/test splits
    - Generate dataset.yaml for Ultralytics

Does NOT modify the original dataset.
"""

import argparse
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# RDD2022 class mapping
CLASS_MAP = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3,  # Pothole
}


def parse_xml_annotation(xml_path: str) -> list[dict]:
    """Parse a Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    annotations = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue

        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        # Convert to YOLO format (center_x, center_y, w, h) normalized
        cx = ((xmin + xmax) / 2) / width
        cy = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        annotations.append({
            "class_id": CLASS_MAP[name],
            "class_name": name,
            "cx": cx, "cy": cy, "w": w, "h": h,
        })

    return annotations


def convert_to_yolo(annotations: list[dict]) -> str:
    """Convert annotations to YOLO label format."""
    lines = []
    for ann in annotations:
        lines.append(
            f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} "
            f"{ann['w']:.6f} {ann['h']:.6f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RDD2022 dataset for YOLO training"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to downloaded RDD2022 dataset"
    )
    parser.add_argument(
        "--output", type=str, default="datasets/rdd2022",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split reproducibility"
    )
    parser.add_argument(
        "--countries", type=str, nargs="+",
        default=["Japan"],
        help="Countries to include (default: Japan)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    source = Path(args.source)
    output = Path(args.output)

    print("=" * 60)
    print("Road Damage Intelligence System — Dataset Preparation")
    print("=" * 60)
    print(f"Source:     {source}")
    print(f"Output:     {output}")
    print(f"Val split:  {args.val_split}")
    print(f"Countries:  {args.countries}")
    print("=" * 60)

    if not source.exists():
        print(f"ERROR: Source directory not found: {source}")
        sys.exit(1)

    # Create output structure
    for split in ["train", "val"]:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    # Find image-annotation pairs
    all_pairs = []
    for country in args.countries:
        images_dir = source / country / "train" / "images"
        annots_dir = source / country / "train" / "annotations" / "xmls"

        if not images_dir.exists():
            print(f"WARNING: Images dir not found: {images_dir}")
            continue

        for img_path in sorted(images_dir.glob("*.*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            xml_path = annots_dir / f"{img_path.stem}.xml"
            if xml_path.exists():
                all_pairs.append((img_path, xml_path))

    print(f"\nFound {len(all_pairs)} image-annotation pairs")

    if not all_pairs:
        print("No pairs found. Check the source directory structure.")
        print("Expected: <country>/train/images/ and <country>/train/annotations/xmls/")
        sys.exit(1)

    # Shuffle and split
    random.shuffle(all_pairs)
    val_count = int(len(all_pairs) * args.val_split)
    val_pairs = all_pairs[:val_count]
    train_pairs = all_pairs[val_count:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Process pairs
    stats = {"total": 0, "skipped": 0, "annotations": 0}

    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, xml_path in pairs:
            annotations = parse_xml_annotation(str(xml_path))
            if not annotations:
                stats["skipped"] += 1
                continue

            # Copy image
            dst_img = output / split / "images" / img_path.name
            shutil.copy2(img_path, dst_img)

            # Write YOLO label
            label_text = convert_to_yolo(annotations)
            dst_label = output / split / "labels" / f"{img_path.stem}.txt"
            dst_label.write_text(label_text)

            stats["total"] += 1
            stats["annotations"] += len(annotations)

    # Generate dataset.yaml
    yaml_content = f"""# RDD2022 Dataset Configuration
# Generated by prepare_dataset.py

path: {output.resolve()}
train: train/images
val: val/images

nc: 4
names:
  0: D00
  1: D10
  2: D20
  3: D40

# Class descriptions:
# D00: Longitudinal Crack
# D10: Transverse Crack
# D20: Alligator Crack
# D40: Pothole
"""

    yaml_path = output / "data.yaml"
    yaml_path.write_text(yaml_content)

    print(f"\n{'=' * 60}")
    print(f"Dataset preparation complete!")
    print(f"  Images processed: {stats['total']}")
    print(f"  Skipped (no annotations): {stats['skipped']}")
    print(f"  Total annotations: {stats['annotations']}")
    print(f"  Dataset YAML: {yaml_path}")
    print(f"\nTo train:")
    print(f"  python scripts/train.py --data {yaml_path} --model yolov8n.pt --epochs 50")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
