"""Tests for image preprocessing functions."""

import numpy as np
import pytest
from app.cv.preprocessing import (
    load_image_from_bytes,
    resize_image,
    letterbox_resize,
    normalize_image,
    bgr_to_rgb,
    rgb_to_bgr,
    get_image_info,
    preprocess_for_display,
)
from app.utils.exceptions import InvalidImageError


class TestResizeImage:
    def test_resize_to_target(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_image(img, target_size=320, maintain_aspect=False)
        assert resized.shape == (320, 320, 3)

    def test_letterbox_preserves_aspect(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = letterbox_resize(img, 640)
        assert result.shape == (640, 640, 3)

    def test_letterbox_square_input(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = letterbox_resize(img, 640)
        assert result.shape == (640, 640, 3)


class TestNormalizeImage:
    def test_normalize_range(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        norm = normalize_image(img)
        assert norm.dtype == np.float32
        assert np.allclose(norm, 1.0)

    def test_normalize_zeros(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        norm = normalize_image(img)
        assert np.allclose(norm, 0.0)

    def test_normalize_mid(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        norm = normalize_image(img)
        assert 0.49 < norm.mean() < 0.51


class TestColorConversion:
    def test_bgr_to_rgb_and_back(self):
        img = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Blue in BGR
        rgb = bgr_to_rgb(img)
        assert rgb[0, 0, 0] == 0    # R
        assert rgb[0, 0, 2] == 255  # B
        bgr = rgb_to_bgr(rgb)
        assert np.array_equal(img, bgr)


class TestGetImageInfo:
    def test_info_3channel(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        info = get_image_info(img)
        assert info["height"] == 480
        assert info["width"] == 640
        assert info["channels"] == 3
        assert info["total_pixels"] == 480 * 640

    def test_info_grayscale(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        info = get_image_info(img)
        assert info["channels"] == 1


class TestPreprocessForDisplay:
    def test_small_image_unchanged(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = preprocess_for_display(img, max_display_size=1280)
        assert result.shape == img.shape

    def test_large_image_resized(self):
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        result = preprocess_for_display(img, max_display_size=1280)
        assert max(result.shape[:2]) <= 1280


class TestLoadFromBytes:
    def test_invalid_bytes(self):
        with pytest.raises(InvalidImageError):
            load_image_from_bytes(b"not an image")
