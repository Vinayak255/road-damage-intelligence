"""Tests for the SeverityEngine."""

from app.analysis.severity import SeverityEngine
from app.cv.detector import Detection


class TestSeverityEngine:
    def _make_det(self, x1, y1, x2, y2, code="D00", conf=0.9):
        return Detection(
            class_id=0, class_code=code, class_name="Test",
            confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2,
        )

    def test_small_region_is_low(self):
        engine = SeverityEngine()
        det = self._make_det(0, 0, 10, 10)  # 100px area
        result = engine.estimate(det, image_width=1000, image_height=1000)
        assert result.level == "Low"
        assert result.area_ratio < engine.low_threshold

    def test_large_region_is_high(self):
        engine = SeverityEngine()
        det = self._make_det(0, 0, 500, 500)  # 250000px area
        result = engine.estimate(det, image_width=1000, image_height=1000)
        assert result.level == "High"

    def test_medium_region(self):
        engine = SeverityEngine()
        # Area ratio = 0.09 with D40 weight 1.2 -> 0.108 weighted
        det = self._make_det(0, 0, 300, 300, code="D40", conf=0.9)
        result = engine.estimate(det, image_width=1000, image_height=1000)
        assert result.level in ("Medium", "High")  # Depends on weights

    def test_pothole_higher_weight(self):
        engine = SeverityEngine()
        # Same box, different class
        det_crack = self._make_det(0, 0, 100, 100, code="D00", conf=0.9)
        det_pothole = self._make_det(0, 0, 100, 100, code="D40", conf=0.9)
        r_crack = engine.estimate(det_crack, 1000, 1000)
        r_pothole = engine.estimate(det_pothole, 1000, 1000)
        assert r_pothole.score >= r_crack.score

    def test_confidence_affects_score(self):
        engine = SeverityEngine()
        det_low_conf = self._make_det(0, 0, 100, 100, conf=0.3)
        det_high_conf = self._make_det(0, 0, 100, 100, conf=0.99)
        r_low = engine.estimate(det_low_conf, 1000, 1000)
        r_high = engine.estimate(det_high_conf, 1000, 1000)
        assert r_high.score >= r_low.score

    def test_zero_image_area(self):
        engine = SeverityEngine()
        det = self._make_det(0, 0, 10, 10)
        result = engine.estimate(det, image_width=0, image_height=0)
        assert result.level == "Low"
        assert result.score == 0.0

    def test_estimate_batch(self):
        engine = SeverityEngine()
        dets = [
            self._make_det(0, 0, 10, 10),
            self._make_det(0, 0, 500, 500),
        ]
        result = engine.estimate_batch(dets, 1000, 1000)
        assert result[0].severity != ""
        assert result[1].severity != ""
        assert len(result) == 2

    def test_score_between_0_and_1(self):
        engine = SeverityEngine()
        det = self._make_det(0, 0, 900, 900)
        result = engine.estimate(det, 1000, 1000)
        assert 0.0 <= result.score <= 1.0
