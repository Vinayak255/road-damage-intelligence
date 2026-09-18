"""
Custom exceptions for the Road Damage Intelligence System.

These exceptions provide meaningful error messages for different
failure scenarios and are caught by the API error handlers.
"""


class RoadDamageError(Exception):
    """Base exception for the application."""

    def __init__(self, message: str = "An error occurred"):
        self.message = message
        super().__init__(self.message)


# --- Model Errors ---

class ModelNotFoundError(RoadDamageError):
    """Raised when the configured model file does not exist."""

    def __init__(self, model_path: str = ""):
        message = (
            f"Road-damage model not found at '{model_path}'. "
            "Configure MODEL_PATH in .env or place the trained model "
            "in models/best.pt. See docs/dataset.md for training instructions."
        )
        super().__init__(message)


class ModelLoadError(RoadDamageError):
    """Raised when the model fails to load."""

    def __init__(self, detail: str = ""):
        message = f"Failed to load the road-damage model. {detail}"
        super().__init__(message)


class InferenceError(RoadDamageError):
    """Raised when model inference fails."""

    def __init__(self, detail: str = ""):
        message = f"Model inference failed. {detail}"
        super().__init__(message)


# --- File/Input Errors ---

class InvalidImageError(RoadDamageError):
    """Raised when the uploaded image is invalid or corrupt."""

    def __init__(self, detail: str = ""):
        message = f"Invalid image file. {detail}"
        super().__init__(message)


class InvalidVideoError(RoadDamageError):
    """Raised when the uploaded video is invalid or corrupt."""

    def __init__(self, detail: str = ""):
        message = f"Invalid video file. {detail}"
        super().__init__(message)


class UnsupportedFileTypeError(RoadDamageError):
    """Raised when the file extension is not supported."""

    def __init__(self, extension: str = "", allowed: list | None = None):
        allowed_str = ", ".join(allowed) if allowed else "unknown"
        message = (
            f"Unsupported file type '{extension}'. "
            f"Allowed extensions: {allowed_str}"
        )
        super().__init__(message)


class FileTooLargeError(RoadDamageError):
    """Raised when the uploaded file exceeds the size limit."""

    def __init__(self, size_mb: float = 0, max_mb: float = 0):
        message = (
            f"File size ({size_mb:.1f} MB) exceeds the maximum "
            f"allowed size ({max_mb:.1f} MB)."
        )
        super().__init__(message)


# --- Processing Errors ---

class ProcessingError(RoadDamageError):
    """Raised when image/video processing fails."""

    def __init__(self, detail: str = ""):
        message = f"Processing failed. {detail}"
        super().__init__(message)


class VideoProcessingError(RoadDamageError):
    """Raised when video processing encounters an error."""

    def __init__(self, detail: str = ""):
        message = f"Video processing failed. {detail}"
        super().__init__(message)


# --- Database Errors ---

class DatabaseError(RoadDamageError):
    """Raised when a database operation fails."""

    def __init__(self, detail: str = ""):
        message = f"Database error. {detail}"
        super().__init__(message)


# --- Configuration Errors ---

class ConfigurationError(RoadDamageError):
    """Raised when required configuration is missing or invalid."""

    def __init__(self, detail: str = ""):
        message = f"Configuration error. {detail}"
        super().__init__(message)
