"""
Input validation utilities for file uploads and paths.

Provides security-focused validation for uploaded files including
extension checks, size limits, and path traversal protection.
"""

import os
import re
import uuid
from pathlib import Path

from app.utils.exceptions import (
    FileTooLargeError,
    InvalidImageError,
    InvalidVideoError,
    UnsupportedFileTypeError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def validate_image_extension(filename: str, allowed: list[str]) -> str:
    """
    Validate that the file has an allowed image extension.

    Args:
        filename: Original filename.
        allowed: List of allowed extensions (e.g., [".jpg", ".png"]).

    Returns:
        The lowercase extension.

    Raises:
        UnsupportedFileTypeError: If extension is not allowed.
    """
    ext = Path(filename).suffix.lower()
    if ext not in [e.strip().lower() for e in allowed]:
        raise UnsupportedFileTypeError(ext, allowed)
    return ext


def validate_video_extension(filename: str, allowed: list[str]) -> str:
    """
    Validate that the file has an allowed video extension.

    Args:
        filename: Original filename.
        allowed: List of allowed extensions (e.g., [".mp4", ".avi"]).

    Returns:
        The lowercase extension.

    Raises:
        UnsupportedFileTypeError: If extension is not allowed.
    """
    ext = Path(filename).suffix.lower()
    if ext not in [e.strip().lower() for e in allowed]:
        raise UnsupportedFileTypeError(ext, allowed)
    return ext


def validate_file_size(file_size: int, max_size_mb: int) -> None:
    """
    Validate that the file does not exceed the maximum size.

    Args:
        file_size: File size in bytes.
        max_size_mb: Maximum allowed size in megabytes.

    Raises:
        FileTooLargeError: If file exceeds the limit.
    """
    size_mb = file_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise FileTooLargeError(size_mb, max_size_mb)


def sanitize_filename(filename: str) -> str:
    """
    Generate a safe filename to prevent path traversal and overwrites.

    Strips directory components, removes unsafe characters,
    and prepends a UUID for uniqueness.

    Args:
        filename: Original filename from upload.

    Returns:
        A sanitized, unique filename.
    """
    # Strip directory components
    basename = os.path.basename(filename)
    # Remove unsafe characters, keep alphanumeric, dots, hyphens, underscores
    safe_name = re.sub(r'[^\w\-.]', '_', basename)
    # Prepend UUID for uniqueness
    unique_name = f"{uuid.uuid4().hex[:12]}_{safe_name}"
    return unique_name


def validate_image_readable(image_path: str) -> None:
    """
    Validate that an image file can be read by OpenCV.

    Args:
        image_path: Path to the image file.

    Raises:
        InvalidImageError: If the image cannot be read.
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise InvalidImageError(
            f"Could not read image at '{image_path}'. "
            "The file may be corrupt or in an unsupported format."
        )


def validate_video_readable(video_path: str) -> None:
    """
    Validate that a video file can be opened by OpenCV.

    Args:
        video_path: Path to the video file.

    Raises:
        InvalidVideoError: If the video cannot be opened.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise InvalidVideoError(
            f"Could not open video at '{video_path}'. "
            "The file may be corrupt or in an unsupported format."
        )
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise InvalidVideoError(
            f"Could not read frames from video at '{video_path}'."
        )


def ensure_directory(directory: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory.

    Returns:
        The Path object for the directory.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
