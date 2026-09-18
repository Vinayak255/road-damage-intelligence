"""
Model evaluation script.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model models/best.pt --data datasets/rdd2022/data.yaml

Evaluates the trained model and saves metrics to reports/.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO model for road damage detection"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model file (default: from .env / models/best.pt)"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to dataset YAML (default: model-embedded)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Evaluation image size"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for evaluation"
    )

    args = parser.parse_args()

    from app.evaluation.evaluator import ModelEvaluator
    from app.utils.logger import setup_logging

    setup_logging("INFO")

    print("=" * 60)
    print("Road Damage Intelligence System — Model Evaluation")
    print("=" * 60)

    evaluator = ModelEvaluator()

    try:
        results = evaluator.evaluate(
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            device=args.device,
        )

        m = results["metrics"]
        print(f"\nPrecision:       {m['precision']:.4f}")
        print(f"Recall:          {m['recall']:.4f}")
        print(f"mAP@0.5:         {m['map50']:.4f}")
        print(f"mAP@0.5:0.95:    {m['map50_95']:.4f}")
        print(f"\nEvaluation time: {results['evaluation_time_seconds']:.2f}s")
        print(f"Results saved to: reports/evaluation_results.json")
        print("=" * 60)

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        print("Make sure a trained model exists at the specified path.")
        sys.exit(1)


if __name__ == "__main__":
    main()
