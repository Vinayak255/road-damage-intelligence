"""
Road Damage Detector — YOLO-based object detection abstraction.

This module wraps the Ultralytics YOLO model for road damage detection.
It handles model loading, validation, inference, and conversion of
raw model output into application-level Detection objects.

IMPORTANT:
- A generic COCO-pretrained YOLO model does NOT detect road damage classes.
- This application requires a YOLO model trained on RDD2022 or equivalent
  road damage dataset with classes: D00, D10, D20, D40.
- If no model is found, the application reports clearly and does not
  silently fall back to an unrelated model.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config.settings import get_settings
from app.utils.exceptions import (
    InferenceError,
    ModelLoadError,
    ModelNotFoundError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """
    Represents a single damage detection from model inference.

    This is the application-level detection object, decoupled from
    the specific model framework output format.
    """
    class_id: int
    class_code: str          # e.g., "D40"
    class_name: str          # e.g., "Pothole"
    confidence: float        # 0.0 to 1.0
    x1: float                # bounding box top-left x
    y1: float                # bounding box top-left y
    x2: float                # bounding box bottom-right x
    y2: float                # bounding box bottom-right y
    track_id: int | None = None
    severity: str = ""       # filled in by SeverityEngine

    @property
    def bbox_width(self) -> float:
        return self.x2 - self.x1

    @property
    def bbox_height(self) -> float:
        return self.y2 - self.y1

    @property
    def bbox_area(self) -> float:
        return self.bbox_width * self.bbox_height

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_code": self.class_code,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "track_id": self.track_id,
            "severity": self.severity,
        }


@dataclass
class InferenceResult:
    """Complete result of a single inference call."""
    detections: list[Detection] = field(default_factory=list)
    inference_time: float = 0.0  # seconds
    image_width: int = 0
    image_height: int = 0

    @property
    def count(self) -> int:
        return len(self.detections)


class RoadDamageDetector:
    """
    YOLO-based road damage detector.

    Wraps Ultralytics YOLO for model loading, inference, and
    result conversion. Supports both image and frame inference.
    """

    def __init__(self):
        self.model = None
        self.model_path: str = ""
        self.model_loaded: bool = False
        self.settings = get_settings()
        self.device: str = self.settings.get_resolved_device()

    def load_model(self, model_path: str | None = None) -> None:
        """
        Load the YOLO model from the specified path.

        Args:
            model_path: Path to the .pt model file.
                       Defaults to settings.model_path.

        Raises:
            ModelNotFoundError: If the model file does not exist.
            ModelLoadError: If the model fails to load.
        """
        self.model_path = model_path or self.settings.model_path

        if not Path(self.model_path).exists():
            self.model_loaded = False
            raise ModelNotFoundError(self.model_path)

        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from: {self.model_path}")
            logger.info(f"Device: {self.device}")
            self.model = YOLO(self.model_path)
            self.model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            self.model_loaded = False
            raise ModelLoadError(str(e))

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self.model_loaded and self.model is not None

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> InferenceResult:
        """
        Run object detection on a single image.

        Args:
            image: Input image as NumPy array (BGR format).
            confidence_threshold: Override default confidence threshold.
            iou_threshold: Override default IoU threshold.

        Returns:
            InferenceResult containing detections and metadata.

        Raises:
            ModelNotFoundError: If model is not loaded.
            InferenceError: If inference fails.
        """
        if not self.is_loaded():
            raise ModelNotFoundError(self.model_path)

        conf = confidence_threshold or self.settings.confidence_threshold
        iou = iou_threshold or self.settings.iou_threshold

        try:
            start_time = time.time()
            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=self.settings.image_size,
                device=self.device,
                verbose=False,
            )
            inference_time = time.time() - start_time

            detections = self._parse_results(results)
            h, w = image.shape[:2]

            return InferenceResult(
                detections=detections,
                inference_time=inference_time,
                image_width=w,
                image_height=h,
            )

        except (ModelNotFoundError, InferenceError):
            raise
        except Exception as e:
            raise InferenceError(str(e))

    def detect_with_tracking(
        self,
        image: np.ndarray,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        tracker: str = "bytetrack.yaml",
        persist: bool = True,
    ) -> InferenceResult:
        """
        Run detection with object tracking on a single frame.

        Uses Ultralytics built-in tracking (ByteTrack by default).

        Args:
            image: Input frame as NumPy array (BGR).
            confidence_threshold: Override default confidence threshold.
            iou_threshold: Override default IoU threshold.
            tracker: Tracker configuration filename.
            persist: Whether to persist tracks across frames.

        Returns:
            InferenceResult with track IDs assigned.
        """
        if not self.is_loaded():
            raise ModelNotFoundError(self.model_path)

        conf = confidence_threshold or self.settings.confidence_threshold
        iou = iou_threshold or self.settings.iou_threshold

        try:
            start_time = time.time()
            results = self.model.track(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=self.settings.image_size,
                device=self.device,
                tracker=tracker,
                persist=persist,
                verbose=False,
            )
            inference_time = time.time() - start_time

            detections = self._parse_results(results, use_tracking=True)
            h, w = image.shape[:2]

            return InferenceResult(
                detections=detections,
                inference_time=inference_time,
                image_width=w,
                image_height=h,
            )

        except (ModelNotFoundError, InferenceError):
            raise
        except Exception as e:
            raise InferenceError(str(e))

    def _parse_results(
        self,
        results,
        use_tracking: bool = False,
    ) -> list[Detection]:
        """
        Parse Ultralytics results into Detection objects.

        Maps model class indices to RDD2022 damage codes using
        the configured class mapping.
        """
        detections = []
        settings = self.settings

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()

                class_code = settings.get_class_code(cls_id)
                class_name = settings.get_damage_class_name(class_code)

                track_id = None
                if use_tracking and boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                detection = Detection(
                    class_id=cls_id,
                    class_code=class_code,
                    class_name=class_name,
                    confidence=conf,
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    track_id=track_id,
                )
                detections.append(detection)

        return detections

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        info = {
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "confidence_threshold": self.settings.confidence_threshold,
            "iou_threshold": self.settings.iou_threshold,
            "image_size": self.settings.image_size,
            "class_mapping": self.settings.class_mapping,
            "class_names": self.settings.class_names,
        }

        if self.model_loaded and self.model is not None:
            try:
                info["model_type"] = self.model.type
            except Exception:
                info["model_type"] = "unknown"

        return info
