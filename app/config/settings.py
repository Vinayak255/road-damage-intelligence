"""
Application configuration loaded from environment variables.

All settings are centralized here. Use `get_settings()` to access
the singleton settings instance throughout the application.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # --- Model ---
    model_path: str = ""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "auto"  # "cpu", "cuda", or "auto"

    # --- Damage Class Mapping ---
    # Maps model output index (str) -> RDD2022 code
    class_mapping: Dict[str, str] = field(default_factory=dict)
    # Maps RDD2022 code -> human-readable name
    class_names: Dict[str, str] = field(default_factory=dict)

    # --- Database ---
    database_url: str = ""

    # --- Upload Limits ---
    max_image_size_mb: int = 20
    max_video_size_mb: int = 500
    allowed_image_extensions: list = field(default_factory=list)
    allowed_video_extensions: list = field(default_factory=list)

    # --- Video Processing ---
    frame_skip: int = 1
    max_video_frames: int = 10000

    # --- Severity Thresholds (bbox area / image area) ---
    severity_low_threshold: float = 0.01
    severity_medium_threshold: float = 0.05
    severity_high_threshold: float = 0.15

    # --- Tracker ---
    tracker_type: str = "bytetrack"

    # --- Evidence ---
    evidence_dir: str = ""
    evidence_cooldown_frames: int = 30

    # --- Server ---
    host: str = "localhost"
    port: int = 8000
    debug: bool = False

    # --- Logging ---
    log_level: str = "INFO"

    # --- Paths (computed) ---
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    input_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "input")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "output")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")

    def __post_init__(self):
        """Load values from environment variables."""
        self.model_path = os.getenv(
            "MODEL_PATH",
            str(PROJECT_ROOT / "models" / "best.pt"),
        )
        self.confidence_threshold = float(
            os.getenv("CONFIDENCE_THRESHOLD", "0.25")
        )
        self.iou_threshold = float(os.getenv("IOU_THRESHOLD", "0.45"))
        self.image_size = int(os.getenv("IMAGE_SIZE", "640"))
        self.device = os.getenv("DEVICE", "auto")

        # Class mapping
        default_mapping = '{"0": "D00", "1": "D10", "2": "D20", "3": "D40"}'
        self.class_mapping = json.loads(
            os.getenv("CLASS_MAPPING", default_mapping)
        )

        default_names = (
            '{"D00": "Longitudinal Crack", "D10": "Transverse Crack", '
            '"D20": "Alligator Crack", "D40": "Pothole"}'
        )
        self.class_names = json.loads(
            os.getenv("CLASS_NAMES", default_names)
        )

        # Database
        default_db = f"sqlite:///{PROJECT_ROOT / 'data' / 'road_damage.db'}"
        self.database_url = os.getenv("DATABASE_URL", default_db)

        # Upload limits
        self.max_image_size_mb = int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))
        self.max_video_size_mb = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
        self.allowed_image_extensions = os.getenv(
            "ALLOWED_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.bmp,.tiff"
        ).split(",")
        self.allowed_video_extensions = os.getenv(
            "ALLOWED_VIDEO_EXTENSIONS", ".mp4,.avi,.mov,.mkv"
        ).split(",")

        # Video
        self.frame_skip = int(os.getenv("FRAME_SKIP", "1"))
        self.max_video_frames = int(os.getenv("MAX_VIDEO_FRAMES", "10000"))

        # Severity
        self.severity_low_threshold = float(
            os.getenv("SEVERITY_LOW_THRESHOLD", "0.01")
        )
        self.severity_medium_threshold = float(
            os.getenv("SEVERITY_MEDIUM_THRESHOLD", "0.05")
        )
        self.severity_high_threshold = float(
            os.getenv("SEVERITY_HIGH_THRESHOLD", "0.15")
        )

        # Tracker
        self.tracker_type = os.getenv("TRACKER_TYPE", "bytetrack")

        # Evidence
        self.evidence_dir = os.getenv(
            "EVIDENCE_DIR", str(PROJECT_ROOT / "data" / "evidence")
        )
        self.evidence_cooldown_frames = int(
            os.getenv("EVIDENCE_COOLDOWN_FRAMES", "30")
        )

        # Server
        self.host = os.getenv("HOST", "localhost")
        self.port = int(os.getenv("PORT", "8000"))
        self.debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

    def get_resolved_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def get_damage_class_name(self, class_code: str) -> str:
        """Get human-readable name for a damage class code."""
        return self.class_names.get(class_code, class_code)

    def get_class_code(self, model_index: int) -> str:
        """Map model output index to RDD2022 class code."""
        return self.class_mapping.get(str(model_index), f"Unknown-{model_index}")


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
