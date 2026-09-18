"""
Pydantic schemas for API request/response validation.

Defines structured response models for all API endpoints.
"""

from datetime import datetime
from pydantic import BaseModel, Field


# =====================================================================
# Detection Schemas
# =====================================================================

class DetectionResponse(BaseModel):
    """Single damage detection in API response."""
    class_id: int
    class_code: str
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    severity: str
    track_id: int | None = None


class ImageAnalysisResponse(BaseModel):
    """Response for image analysis endpoint."""
    session_id: int
    filename: str
    total_detections: int
    processing_time: float
    image_width: int
    image_height: int
    detections: list[DetectionResponse]
    annotated_image_url: str
    original_image_url: str
    evidence_paths: list[str] = []


class VideoAnalysisResponse(BaseModel):
    """Response for video analysis endpoint."""
    session_id: int
    filename: str
    total_frames: int
    processed_frames: int
    total_detections: int
    unique_damages: int
    processing_time: float
    average_fps: float
    output_video_url: str
    tracked_damages: list[dict] = []
    evidence_paths: list[str] = []


# =====================================================================
# Session Schemas
# =====================================================================

class SessionResponse(BaseModel):
    """Analysis session in API response."""
    id: int
    source_type: str
    filename: str
    started_at: datetime | None
    completed_at: datetime | None
    total_frames: int
    processed_frames: int
    processing_time: float
    average_fps: float
    total_detections: int
    status: str
    output_path: str | None = None

    class Config:
        from_attributes = True


class DetectionRecordResponse(BaseModel):
    """Detection record from database."""
    id: int
    session_id: int
    frame_number: int
    timestamp: str | None
    damage_type: str
    damage_name: str | None
    confidence: float
    severity: str
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None
    evidence_path: str | None

    class Config:
        from_attributes = True


# =====================================================================
# Analytics Schemas
# =====================================================================

class AnalyticsResponse(BaseModel):
    """Analytics dashboard data."""
    total_sessions: int
    completed_sessions: int
    total_detections: int
    average_confidence: float
    average_fps: float
    average_processing_time: float
    damage_type_distribution: list[dict]
    severity_distribution: list[dict]
    detections_per_session: list[dict]
    confidence_values: list[float]
    charts: dict = {}


# =====================================================================
# Model Schemas
# =====================================================================

class ModelStatusResponse(BaseModel):
    """Model status information."""
    model_path: str
    model_loaded: bool
    device: str
    confidence_threshold: float
    iou_threshold: float
    image_size: int
    class_mapping: dict
    class_names: dict
    model_type: str = "unknown"


class EvaluationResponse(BaseModel):
    """Model evaluation results."""
    status: str
    message: str = ""
    model_path: str = ""
    dataset: str = ""
    image_size: int = 0
    device: str = ""
    evaluation_time_seconds: float = 0.0
    metrics: dict = {}
    per_class: dict = {}


# =====================================================================
# General Schemas
# =====================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    database_connected: bool = False
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    status_code: int
