"""
Object tracking wrapper for video-based damage detection.

Uses Ultralytics built-in ByteTrack tracking to maintain persistent
track IDs across video frames, enabling event deduplication.

ByteTrack is chosen for its lightweight design and suitability
for straightforward multi-object tracking scenarios.
"""

from dataclasses import dataclass, field

from app.cv.detector import Detection
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedDamage:
    """Represents a tracked damage event across multiple frames."""
    track_id: int
    damage_type: str
    damage_name: str
    first_frame: int
    last_frame: int
    frame_count: int = 1
    max_confidence: float = 0.0
    last_severity: str = "Low"
    evidence_generated: bool = False

    def update(
        self,
        frame_number: int,
        confidence: float,
        severity: str,
    ) -> None:
        """Update tracking state with a new frame observation."""
        self.last_frame = frame_number
        self.frame_count += 1
        self.max_confidence = max(self.max_confidence, confidence)
        self.last_severity = severity


class DamageTracker:
    """
    Manages tracked damage events and deduplication logic.

    Works alongside the Ultralytics ByteTrack integration. The actual
    tracking (ID assignment) is done by model.track() in the detector.
    This class manages application-level event tracking and cooldown
    to avoid generating duplicate evidence for the same damage.
    """

    def __init__(self, cooldown_frames: int = 30):
        """
        Args:
            cooldown_frames: Minimum frames between evidence generation
                           for the same track ID.
        """
        self.cooldown_frames = cooldown_frames
        self.tracked_damages: dict[int, TrackedDamage] = {}
        self._evidence_cooldown: dict[int, int] = {}  # track_id -> last evidence frame

    def update(
        self,
        detections: list[Detection],
        frame_number: int,
    ) -> list[Detection]:
        """
        Update tracking state with new frame detections.

        Args:
            detections: Detections with track IDs from model.track().
            frame_number: Current frame number.

        Returns:
            The detections (unchanged, tracking state is internal).
        """
        for det in detections:
            if det.track_id is None:
                continue

            if det.track_id in self.tracked_damages:
                self.tracked_damages[det.track_id].update(
                    frame_number, det.confidence, det.severity
                )
            else:
                self.tracked_damages[det.track_id] = TrackedDamage(
                    track_id=det.track_id,
                    damage_type=det.class_code,
                    damage_name=det.class_name,
                    first_frame=frame_number,
                    last_frame=frame_number,
                    max_confidence=det.confidence,
                    last_severity=det.severity,
                )

        return detections

    def should_generate_evidence(
        self,
        track_id: int,
        frame_number: int,
    ) -> bool:
        """
        Check if evidence should be generated for a track ID.

        Uses cooldown logic to avoid generating hundreds of
        duplicate evidence images for the same damage appearing
        in consecutive frames.

        Args:
            track_id: The tracking ID to check.
            frame_number: Current frame number.

        Returns:
            True if evidence should be generated.
        """
        if track_id not in self._evidence_cooldown:
            return True

        last_evidence_frame = self._evidence_cooldown[track_id]
        return (frame_number - last_evidence_frame) >= self.cooldown_frames

    def mark_evidence_generated(
        self,
        track_id: int,
        frame_number: int,
    ) -> None:
        """Record that evidence was generated for a track at this frame."""
        self._evidence_cooldown[track_id] = frame_number
        if track_id in self.tracked_damages:
            self.tracked_damages[track_id].evidence_generated = True

    def get_unique_damage_count(self) -> int:
        """Get the number of unique tracked damages."""
        return len(self.tracked_damages)

    def get_summary(self) -> list[dict]:
        """Get a summary of all tracked damages."""
        return [
            {
                "track_id": td.track_id,
                "damage_type": td.damage_type,
                "damage_name": td.damage_name,
                "first_frame": td.first_frame,
                "last_frame": td.last_frame,
                "frame_count": td.frame_count,
                "max_confidence": round(td.max_confidence, 4),
                "severity": td.last_severity,
            }
            for td in self.tracked_damages.values()
        ]

    def reset(self) -> None:
        """Reset all tracking state (e.g., for a new video)."""
        self.tracked_damages.clear()
        self._evidence_cooldown.clear()
