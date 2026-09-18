"""
Model evaluation module using Ultralytics validation pipeline.

Evaluates trained YOLO models on a validation dataset and reports
standard object detection metrics. Never fabricates metrics.
"""

import json
import time
from pathlib import Path

from app.config.settings import get_settings
from app.utils.exceptions import ModelNotFoundError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates a YOLO model and reports detection metrics.

    Metrics:
        - Precision
        - Recall
        - mAP@0.5
        - mAP@0.5:0.95
        - Inference time
    """

    def __init__(self):
        self.settings = get_settings()
        self.results_path = (
            self.settings.reports_dir / "evaluation_results.json"
        )

    def evaluate(
        self,
        model_path: str | None = None,
        data_yaml: str | None = None,
        imgsz: int = 640,
        device: str | None = None,
    ) -> dict:
        """
        Run model evaluation on a validation dataset.

        Args:
            model_path: Path to the model file.
            data_yaml: Path to dataset YAML configuration.
            imgsz: Inference image size.
            device: Device for evaluation.

        Returns:
            Dictionary with evaluation metrics.

        Raises:
            ModelNotFoundError: If model file doesn't exist.
            RuntimeError: If evaluation fails.
        """
        mp = model_path or self.settings.model_path
        if not Path(mp).exists():
            raise ModelNotFoundError(mp)

        dev = device or self.settings.get_resolved_device()

        from ultralytics import YOLO

        logger.info(f"Starting evaluation: model={mp}, data={data_yaml}")
        model = YOLO(mp)

        start_time = time.time()

        val_args = {
            "imgsz": imgsz,
            "device": dev,
            "verbose": False,
        }
        if data_yaml:
            val_args["data"] = data_yaml

        metrics = model.val(**val_args)
        eval_time = time.time() - start_time

        results = {
            "model_path": mp,
            "dataset": data_yaml or "model-embedded",
            "image_size": imgsz,
            "device": dev,
            "evaluation_time_seconds": round(eval_time, 2),
            "metrics": {
                "precision": round(float(metrics.box.mp), 4),
                "recall": round(float(metrics.box.mr), 4),
                "map50": round(float(metrics.box.map50), 4),
                "map50_95": round(float(metrics.box.map), 4),
            },
            "per_class": {},
            "status": "completed",
        }

        # Per-class metrics if available
        class_names = metrics.names if hasattr(metrics, 'names') else {}
        if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
            for i, map_val in enumerate(metrics.box.maps):
                cls_name = class_names.get(i, f"class_{i}")
                results["per_class"][cls_name] = {
                    "map50_95": round(float(map_val), 4),
                }

        # Save results
        self._save_results(results)

        logger.info(
            f"Evaluation complete: P={results['metrics']['precision']:.4f}, "
            f"R={results['metrics']['recall']:.4f}, "
            f"mAP50={results['metrics']['map50']:.4f}, "
            f"mAP50-95={results['metrics']['map50_95']:.4f}"
        )

        return results

    def get_latest_results(self) -> dict:
        """
        Load the most recent evaluation results from disk.

        Returns:
            Evaluation results dict, or a status message if
            evaluation has not been run.
        """
        if not self.results_path.exists():
            return {
                "status": "not_executed",
                "message": "Evaluation not yet executed. "
                          "Run scripts/evaluate_model.py or use the "
                          "evaluation API endpoint.",
            }

        with open(self.results_path, "r") as f:
            return json.load(f)

    def _save_results(self, results: dict) -> None:
        """Save evaluation results to disk."""
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {self.results_path}")
