"""Tests for the RoadDamageDetector output structure."""

import pytest
from app.cv.detector import Detection, InferenceResult, RoadDamageDetector
from app.utils.exceptions import ModelNotFoundError


class TestDetection:
    def test_bbox_properties(self):
        det = Detection(
            class_id=0, class_code="D00", class_name="Longitudinal Crack",
            confidence=0.9, x1=100, y1=200, x2=300, y2=400,
        )
        assert det.bbox_width == 200
        assert det.bbox_height == 200
        assert det.bbox_area == 40000

    def test_to_dict(self):
        det = Detection(
            class_id=3, class_code="D40", class_name="Pothole",
            confidence=0.912345, x1=10.123, y1=20.456, x2=50.789, y2=60.012,
            severity="High", track_id=5,
        )
        d = det.to_dict()
        assert d["class_code"] == "D40"
        assert d["class_name"] == "Pothole"
        assert d["confidence"] == 0.9123
        assert d["severity"] == "High"
        assert d["track_id"] == 5

    def test_default_severity_empty(self):
        det = Detection(
            class_id=0, class_code="D00", class_name="Test",
            confidence=0.5, x1=0, y1=0, x2=10, y2=10,
        )
        assert det.severity == ""
        assert det.track_id is None


class TestInferenceResult:
    def test_count(self):
        result = InferenceResult(
            detections=[
                Detection(0, "D00", "Test", 0.9, 0, 0, 10, 10),
                Detection(1, "D10", "Test2", 0.8, 0, 0, 10, 10),
            ],
            inference_time=0.05,
        )
        assert result.count == 2

    def test_empty(self):
        result = InferenceResult()
        assert result.count == 0
        assert result.inference_time == 0.0


class TestRoadDamageDetector:
    def test_model_not_found(self):
        detector = RoadDamageDetector()
        with pytest.raises(ModelNotFoundError):
            detector.load_model("nonexistent_model.pt")

    def test_is_loaded_initially_false(self):
        detector = RoadDamageDetector()
        assert detector.is_loaded() is False

    def test_get_model_info(self):
        detector = RoadDamageDetector()
        info = detector.get_model_info()
        assert "model_loaded" in info
        assert info["model_loaded"] is False
        assert "class_mapping" in info
        assert "confidence_threshold" in info
