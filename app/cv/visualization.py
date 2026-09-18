"""
Visualization utilities for drawing detection results on images.

Uses OpenCV to draw bounding boxes, labels, confidence scores,
severity levels, and metadata overlays on images.
"""

import cv2
import numpy as np

from app.cv.detector import Detection
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Color scheme for damage types (BGR format for OpenCV)
DAMAGE_COLORS = {
    "D00": (255, 165, 0),    # Orange — Longitudinal Crack
    "D10": (0, 255, 255),    # Yellow — Transverse Crack
    "D20": (0, 0, 255),      # Red — Alligator Crack
    "D40": (0, 0, 200),      # Dark Red — Pothole
}

# Severity colors (BGR)
SEVERITY_COLORS = {
    "Low": (0, 200, 0),       # Green
    "Medium": (0, 200, 255),  # Orange
    "High": (0, 0, 255),      # Red
}

DEFAULT_COLOR = (200, 200, 200)  # Gray


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    line_thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True,
    show_severity: bool = True,
    show_track_id: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image for all detections.

    Args:
        image: Input image (BGR).
        detections: List of Detection objects.
        line_thickness: Thickness of bounding box lines.
        font_scale: Scale factor for label text.
        show_confidence: Whether to show confidence scores.
        show_severity: Whether to show severity levels.
        show_track_id: Whether to show track IDs.

    Returns:
        Annotated image with bounding boxes and labels.
    """
    annotated = image.copy()

    for det in detections:
        color = DAMAGE_COLORS.get(det.class_code, DEFAULT_COLOR)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Build label text
        label_parts = [det.class_name]
        if show_confidence:
            label_parts.append(f"{det.confidence:.1%}")
        if show_severity and det.severity:
            label_parts.append(det.severity)
        if show_track_id and det.track_id is not None:
            label_parts.append(f"ID:{det.track_id}")

        label = " | ".join(label_parts)

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_y = max(y1 - 10, label_h + 10)
        cv2.rectangle(
            annotated,
            (x1, label_y - label_h - 5),
            (x1 + label_w + 5, label_y + 5),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_summary_overlay(
    image: np.ndarray,
    total_detections: int,
    processing_time: float,
    frame_number: int | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """
    Draw a summary info bar at the top of the image.

    Args:
        image: Input image.
        total_detections: Number of detections in this frame/image.
        processing_time: Time taken for inference (seconds).
        frame_number: Optional frame number for video.
        fps: Optional processing FPS.

    Returns:
        Image with info overlay.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Semi-transparent overlay bar
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # Info text
    parts = [f"Detections: {total_detections}"]
    parts.append(f"Time: {processing_time * 1000:.0f}ms")
    if frame_number is not None:
        parts.append(f"Frame: {frame_number}")
    if fps is not None:
        parts.append(f"FPS: {fps:.1f}")

    info_text = " | ".join(parts)
    cv2.putText(
        annotated,
        info_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated


def create_evidence_image(
    image: np.ndarray,
    detection: Detection,
    timestamp: str = "",
    session_id: int | None = None,
) -> np.ndarray:
    """
    Create an annotated evidence image for a single detection.

    Includes the detection bounding box, metadata panel with
    damage type, confidence, severity, track ID, and timestamp.

    Args:
        image: Source image (BGR).
        detection: The detection to document.
        timestamp: Timestamp string (e.g., "00:01:24").
        session_id: Optional analysis session ID.

    Returns:
        Evidence image with detection highlighted and metadata panel.
    """
    # Draw the detection on the image
    annotated = draw_detections(image, [detection])

    h, w = annotated.shape[:2]

    # Create metadata panel at the bottom
    panel_height = 140
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background

    # Header bar
    cv2.rectangle(panel, (0, 0), (w, 30), (0, 100, 200), -1)
    cv2.putText(
        panel,
        "ROAD DAMAGE DETECTED",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Metadata lines
    y_start = 55
    line_gap = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color = (220, 220, 220)

    lines = [
        f"Type: {detection.class_name} ({detection.class_code})",
        f"Confidence: {detection.confidence:.1%}",
        f"Severity: {detection.severity}",
    ]
    if detection.track_id is not None:
        lines.append(f"Track ID: {detection.track_id}")
    if timestamp:
        lines.append(f"Timestamp: {timestamp}")

    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (15, y_start + i * line_gap),
            font,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    # Severity indicator
    sev_color = SEVERITY_COLORS.get(detection.severity, DEFAULT_COLOR)
    cv2.circle(panel, (w - 50, 70), 25, sev_color, -1)
    cv2.putText(
        panel,
        detection.severity[0] if detection.severity else "?",
        (w - 60, 78),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Combine image and panel
    evidence = np.vstack([annotated, panel])
    return evidence


def encode_image_to_bytes(
    image: np.ndarray,
    format: str = ".jpg",
    quality: int = 90,
) -> bytes:
    """
    Encode an image to bytes for API response or file saving.

    Args:
        image: Input image (BGR).
        format: Output format (".jpg", ".png").
        quality: JPEG quality (1-100).

    Returns:
        Encoded image bytes.
    """
    params = []
    if format.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif format.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    success, buffer = cv2.imencode(format, image, params)
    if not success:
        raise ValueError(f"Failed to encode image as {format}")
    return buffer.tobytes()
