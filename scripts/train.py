"""
Training script for the road damage YOLO model.

Usage:
    python scripts/train.py --data path/to/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640

This script wraps the Ultralytics training pipeline for convenience.
Training is OPTIONAL — the application is inference-first.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model for road damage detection"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML file (e.g., datasets/rdd2022/data.yaml)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Base model to fine-tune (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Training image size (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device: '', 'cpu', '0', '0,1'. Empty for auto-detect."
    )
    parser.add_argument(
        "--project", type=str, default="runs/train",
        help="Project directory for saving results"
    )
    parser.add_argument(
        "--name", type=str, default="road_damage",
        help="Experiment name"
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early stopping patience (default: 20)"
    )

    args = parser.parse_args()

    from ultralytics import YOLO

    print("=" * 60)
    print("Road Damage Intelligence System — Model Training")
    print("=" * 60)
    print(f"Base model:  {args.model}")
    print(f"Dataset:     {args.data}")
    print(f"Epochs:      {args.epochs}")
    print(f"Image size:  {args.imgsz}")
    print(f"Batch size:  {args.batch}")
    print(f"Device:      {args.device or 'auto'}")
    print("=" * 60)

    # Load base model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved to: {args.project}/{args.name}/weights/best.pt")
    print(f"\nTo use this model, copy best.pt to models/best.pt:")
    print(f"  copy {args.project}\\{args.name}\\weights\\best.pt models\\best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
