"""Tests for the DamageTracker."""

from app.cv.detector import Detection
from app.cv.tracker import DamageTracker


class TestDamageTracker:
    def _make_det(self, track_id, class_code="D40", conf=0.9):
        return Detection(
            class_id=3, class_code=class_code, class_name="Pothole",
            confidence=conf, x1=10, y1=10, x2=50, y2=50,
            track_id=track_id, severity="Medium",
        )

    def test_update_creates_tracked_damage(self):
        tracker = DamageTracker(cooldown_frames=10)
        dets = [self._make_det(1), self._make_det(2)]
        tracker.update(dets, frame_number=1)
        assert tracker.get_unique_damage_count() == 2

    def test_update_same_track_increments_count(self):
        tracker = DamageTracker(cooldown_frames=10)
        tracker.update([self._make_det(1)], frame_number=1)
        tracker.update([self._make_det(1)], frame_number=2)
        tracker.update([self._make_det(1)], frame_number=3)
        assert tracker.get_unique_damage_count() == 1
        summary = tracker.get_summary()
        assert summary[0]["frame_count"] == 3

    def test_evidence_cooldown(self):
        tracker = DamageTracker(cooldown_frames=10)
        # First time: should generate
        assert tracker.should_generate_evidence(1, frame_number=1) is True
        tracker.mark_evidence_generated(1, frame_number=1)

        # Within cooldown: should not generate
        assert tracker.should_generate_evidence(1, frame_number=5) is False

        # After cooldown: should generate again
        assert tracker.should_generate_evidence(1, frame_number=11) is True

    def test_different_tracks_independent_cooldown(self):
        tracker = DamageTracker(cooldown_frames=10)
        tracker.mark_evidence_generated(1, frame_number=1)
        # Track 2 has no cooldown yet
        assert tracker.should_generate_evidence(2, frame_number=2) is True

    def test_reset(self):
        tracker = DamageTracker(cooldown_frames=10)
        tracker.update([self._make_det(1)], frame_number=1)
        tracker.mark_evidence_generated(1, frame_number=1)
        tracker.reset()
        assert tracker.get_unique_damage_count() == 0
        assert tracker.should_generate_evidence(1, frame_number=1) is True

    def test_max_confidence_tracked(self):
        tracker = DamageTracker(cooldown_frames=10)
        tracker.update([self._make_det(1, conf=0.7)], frame_number=1)
        tracker.update([self._make_det(1, conf=0.95)], frame_number=2)
        tracker.update([self._make_det(1, conf=0.8)], frame_number=3)
        summary = tracker.get_summary()
        assert summary[0]["max_confidence"] == 0.95

    def test_none_track_id_ignored(self):
        tracker = DamageTracker(cooldown_frames=10)
        det = Detection(
            class_id=0, class_code="D00", class_name="Test",
            confidence=0.5, x1=0, y1=0, x2=10, y2=10,
            track_id=None, severity="Low",
        )
        tracker.update([det], frame_number=1)
        assert tracker.get_unique_damage_count() == 0
