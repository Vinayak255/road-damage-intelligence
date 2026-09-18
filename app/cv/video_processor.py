"""
Video processing pipeline for frame-by-frame road damage analysis.

Processes video files incrementally using OpenCV VideoCapture,
running YOLO inference with ByteTrack tracking on each frame.
Supports frame skipping, progress callbacks, and evidence generation.

Does NOT load the entire video into memory.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.analysis.severity import SeverityEngine
from app.config.settings import get_settings
from app.cv.detector import Detection, InferenceResult, RoadDamageDetector
from app.cv.tracker import DamageTracker
from app.cv.visualization import (
    create_evidence_image,
    draw_detections,
    draw_summary_overlay,
)
from app.utils.exceptions import InvalidVideoError, VideoProcessingError
from app.utils.logger import get_logger
from app.utils.validators import ensure_directory

logger = get_logger(__name__)


@dataclass
class VideoResult:
    """Complete result of a video processing session."""
    output_path: str = ""
    total_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    unique_damages: int = 0
    processing_time: float = 0.0
    average_fps: float = 0.0
    frame_detections: list[dict] = field(default_factory=list)
    tracked_damages: list[dict] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)


class VideoProcessor:
    """
    Frame-by-frame video processing pipeline.

    Integrates the detector, tracker, severity engine, and
    evidence generator for end-to-end video analysis.
    """

    def __init__(
        self,
        detector: RoadDamageDetector,
        severity_engine: SeverityEngine | None = None,
    ):
        self.detector = detector
        self.severity_engine = severity_engine or SeverityEngine()
        self.settings = get_settings()
        self.tracker = DamageTracker(
            cooldown_frames=self.settings.evidence_cooldown_frames
        )

    def process_video(
        self,
        video_path: str,
        output_path: str | None = None,
        frame_skip: int | None = None,
        max_frames: int | None = None,
        progress_callback: Callable[[int, int, float], None] | None = None,
        session_id: int | None = None,
    ) -> VideoResult:
        """
        Process a video file frame-by-frame.

        Args:
            video_path: Path to the input video file.
            output_path: Path for the annotated output video.
            frame_skip: Process every Nth frame (1 = all frames).
            max_frames: Maximum number of frames to process.
            progress_callback: Called with (frame_number, total_frames, fps).
            session_id: Optional session ID for evidence filenames.

        Returns:
            VideoResult with all processing metadata and detections.

        Raises:
            InvalidVideoError: If the video cannot be opened.
            VideoProcessingError: If processing fails.
        """
        self.tracker.reset()
        skip = frame_skip or self.settings.frame_skip
        max_f = max_frames or self.settings.max_video_frames

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Processing video: {video_path} "
            f"({total_frames} frames, {fps_in:.1f} FPS, {width}x{height})"
        )

        # Setup output video writer
        if output_path is None:
            out_dir = ensure_directory(self.settings.output_dir)
            stem = Path(video_path).stem
            output_path = str(out_dir / f"{stem}_analyzed.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
        if not writer.isOpened():
            cap.release()
            raise VideoProcessingError("Failed to create output video writer")

        # Evidence directory
        evidence_dir = ensure_directory(self.settings.evidence_dir)

        result = VideoResult(output_path=output_path, total_frames=total_frames)
        all_detections_for_db = []
        start_time = time.time()
        frame_number = 0
        processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                if frame_number > max_f:
                    logger.info(f"Reached max frame limit: {max_f}")
                    break

                # Frame skipping
                if (frame_number - 1) % skip != 0:
                    writer.write(frame)
                    continue

                processed += 1

                # Run inference with tracking
                try:
                    inf_result = self.detector.detect_with_tracking(frame)
                except Exception as e:
                    logger.warning(f"Inference failed on frame {frame_number}: {e}")
                    writer.write(frame)
                    continue

                # Estimate severity
                detections = self.severity_engine.estimate_batch(
                    inf_result.detections,
                    inf_result.image_width,
                    inf_result.image_height,
                )

                # Update tracker
                self.tracker.update(detections, frame_number)

                # Calculate timestamp
                timestamp = self._frame_to_timestamp(frame_number, fps_in)

                # Generate evidence for new/cooldown-eligible tracks
                for det in detections:
                    tid = det.track_id or 0
                    if self.tracker.should_generate_evidence(tid, frame_number):
                        evidence_img = create_evidence_image(
                            frame, det, timestamp=timestamp, session_id=session_id
                        )
                        ev_filename = (
                            f"evidence_s{session_id or 0}_f{frame_number}"
                            f"_t{tid}_{det.class_code}.jpg"
                        )
                        ev_path = str(evidence_dir / ev_filename)
                        cv2.imwrite(ev_path, evidence_img)
                        result.evidence_paths.append(ev_path)
                        self.tracker.mark_evidence_generated(tid, frame_number)

                        # Record detection for DB
                        all_detections_for_db.append({
                            "session_id": session_id,
                            "frame_number": frame_number,
                            "timestamp": timestamp,
                            "damage_type": det.class_code,
                            "damage_name": det.class_name,
                            "confidence": det.confidence,
                            "severity": det.severity,
                            "x1": det.x1,
                            "y1": det.y1,
                            "x2": det.x2,
                            "y2": det.y2,
                            "track_id": det.track_id,
                            "evidence_path": ev_path,
                        })

                result.total_detections += len(detections)

                # Annotate frame
                annotated = draw_detections(frame, detections)
                elapsed = time.time() - start_time
                current_fps = processed / elapsed if elapsed > 0 else 0
                annotated = draw_summary_overlay(
                    annotated,
                    total_detections=len(detections),
                    processing_time=inf_result.inference_time,
                    frame_number=frame_number,
                    fps=current_fps,
                )
                writer.write(annotated)

                # Progress callback
                if progress_callback and frame_number % 10 == 0:
                    progress_callback(frame_number, total_frames, current_fps)

        except Exception as e:
            raise VideoProcessingError(str(e))
        finally:
            cap.release()
            writer.release()

        elapsed = time.time() - start_time
        result.processed_frames = processed
        result.processing_time = round(elapsed, 2)
        result.average_fps = round(processed / elapsed, 2) if elapsed > 0 else 0
        result.unique_damages = self.tracker.get_unique_damage_count()
        result.tracked_damages = self.tracker.get_summary()
        result.frame_detections = all_detections_for_db

        logger.info(
            f"Video processing complete: {processed} frames processed, "
            f"{result.total_detections} total detections, "
            f"{result.unique_damages} unique damages, "
            f"{result.average_fps:.1f} FPS"
        )

        return result

    @staticmethod
    def _frame_to_timestamp(frame_number: int, fps: float) -> str:
        """Convert frame number to HH:MM:SS timestamp."""
        total_seconds = frame_number / fps if fps > 0 else 0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get basic information about a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video: {video_path}")

        info = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": 0.0,
        }
        if info["fps"] > 0:
            info["duration_seconds"] = round(
                info["total_frames"] / info["fps"], 2
            )
        cap.release()
        return info
