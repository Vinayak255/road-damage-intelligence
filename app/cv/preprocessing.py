"""
Image preprocessing utilities using OpenCV.

Provides functions for loading, resizing, normalizing, and
preparing images for model inference. All operations use
OpenCV (BGR format) and NumPy.
"""

import cv2
import numpy as np

from app.utils.exceptions import InvalidImageError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    OpenCV loads images in BGR format by default.

    Args:
        image_path: Path to the image file.

    Returns:
        NumPy array of the image in BGR format.

    Raises:
        InvalidImageError: If the image cannot be read.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise InvalidImageError(
            f"Failed to load image from '{image_path}'. "
            "File may be corrupt or in an unsupported format."
        )
    logger.debug(
        f"Loaded image: {image_path}, shape={image.shape}, dtype={image.dtype}"
    )
    return image


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Load an image from raw bytes.

    Args:
        file_bytes: Raw image bytes.

    Returns:
        NumPy array of the image in BGR format.

    Raises:
        InvalidImageError: If the bytes cannot be decoded.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(
            "Failed to decode image from uploaded bytes. "
            "File may be corrupt or in an unsupported format."
        )
    return image


def resize_image(
    image: np.ndarray,
    target_size: int = 640,
    maintain_aspect: bool = True,
) -> np.ndarray:
    """
    Resize image for model inference.

    Args:
        image: Input image (BGR).
        target_size: Target dimension (square).
        maintain_aspect: Whether to maintain aspect ratio with padding.

    Returns:
        Resized image.
    """
    if maintain_aspect:
        return letterbox_resize(image, target_size)
    return cv2.resize(image, (target_size, target_size))


def letterbox_resize(
    image: np.ndarray,
    target_size: int = 640,
    color: tuple = (114, 114, 114),
) -> np.ndarray:
    """
    Resize image with letterboxing to maintain aspect ratio.

    Pads with a neutral color to fill the target square.

    Args:
        image: Input image (BGR).
        target_size: Target square dimension.
        color: Padding color (BGR).

    Returns:
        Letterboxed image.
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas and center the resized image
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    return canvas


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.

    Args:
        image: Input image (uint8, 0-255).

    Returns:
        Normalized image (float32, 0.0-1.0).
    """
    return image.astype(np.float32) / 255.0


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert image from BGR (OpenCV default) to RGB.

    Args:
        image: Input image in BGR format.

    Returns:
        Image in RGB format.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert image from RGB to BGR (OpenCV format).

    Args:
        image: Input image in RGB format.

    Returns:
        Image in BGR format.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about an image.

    Args:
        image: Input image.

    Returns:
        Dictionary with height, width, channels, dtype.
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(image.dtype),
        "total_pixels": h * w,
    }


def preprocess_for_display(
    image: np.ndarray,
    max_display_size: int = 1280,
) -> np.ndarray:
    """
    Resize image for web display if it's too large.

    Args:
        image: Input image.
        max_display_size: Maximum dimension for display.

    Returns:
        Resized image suitable for display.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_display_size:
        return image.copy()

    scale = max_display_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
