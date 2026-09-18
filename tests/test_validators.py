"""Tests for input validators."""

import pytest
from app.utils.validators import (
    validate_image_extension,
    validate_video_extension,
    validate_file_size,
    sanitize_filename,
)
from app.utils.exceptions import (
    UnsupportedFileTypeError,
    FileTooLargeError,
)


class TestImageExtensionValidation:
    ALLOWED = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    def test_valid_jpg(self):
        ext = validate_image_extension("photo.jpg", self.ALLOWED)
        assert ext == ".jpg"

    def test_valid_png(self):
        ext = validate_image_extension("photo.PNG", self.ALLOWED)
        assert ext == ".png"

    def test_invalid_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_image_extension("file.exe", self.ALLOWED)

    def test_no_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_image_extension("noextension", self.ALLOWED)

    def test_valid_tiff(self):
        ext = validate_image_extension("scan.tiff", self.ALLOWED)
        assert ext == ".tiff"


class TestVideoExtensionValidation:
    ALLOWED = [".mp4", ".avi", ".mov", ".mkv"]

    def test_valid_mp4(self):
        ext = validate_video_extension("clip.mp4", self.ALLOWED)
        assert ext == ".mp4"

    def test_invalid_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_video_extension("clip.gif", self.ALLOWED)


class TestFileSizeValidation:
    def test_within_limit(self):
        # 5 MB should pass 20 MB limit
        validate_file_size(5 * 1024 * 1024, 20)

    def test_exceeds_limit(self):
        with pytest.raises(FileTooLargeError):
            validate_file_size(25 * 1024 * 1024, 20)

    def test_exact_limit(self):
        # Exactly at limit should pass (not strictly greater)
        validate_file_size(20 * 1024 * 1024, 20)

    def test_zero_size(self):
        validate_file_size(0, 20)


class TestSanitizeFilename:
    def test_basic_filename(self):
        result = sanitize_filename("photo.jpg")
        assert result.endswith("_photo.jpg")
        assert len(result) > len("photo.jpg")

    def test_strips_directory(self):
        result = sanitize_filename("../../etc/passwd")
        assert "/" not in result
        assert "\\" not in result

    def test_removes_unsafe_chars(self):
        result = sanitize_filename("file name (1) [copy].jpg")
        assert "(" not in result
        assert "[" not in result

    def test_unique_names(self):
        r1 = sanitize_filename("photo.jpg")
        r2 = sanitize_filename("photo.jpg")
        assert r1 != r2  # UUID prefix ensures uniqueness
