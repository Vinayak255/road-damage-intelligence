"""
FastAPI route definitions for all API endpoints.

Handles image/video upload, analysis, history, analytics,
model status, and evaluation endpoints.
"""

import os
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.analysis.severity import SeverityEngine
from app.analysis.statistics import (
    generate_confidence_chart,
    generate_damage_type_chart,
    generate_detections_per_session_chart,
    generate_performance_chart,
    generate_severity_chart,
)
from app.api.schemas import (
    AnalyticsResponse,
    DetectionRecordResponse,
    DetectionResponse,
    ErrorResponse,
    EvaluationResponse,
    HealthResponse,
    ImageAnalysisResponse,
    ModelStatusResponse,
    SessionResponse,
    VideoAnalysisResponse,
)
from app.config.settings import get_settings
from app.cv.detector import RoadDamageDetector
from app.cv.video_processor import VideoProcessor
from app.cv.visualization import (
    create_evidence_image,
    draw_detections,
    draw_summary_overlay,
)
from app.database.database import get_db, get_session_factory
from app.database.models import AnalysisSession, DamageDetection
from app.database.repository import (
    add_detection,
    complete_session,
    create_session,
    fail_session,
    get_all_detections,
    get_all_sessions,
    get_analytics_summary,
    get_detections_by_session,
    get_session,
)
from app.evaluation.evaluator import ModelEvaluator
from app.utils.exceptions import (
    FileTooLargeError,
    ModelNotFoundError,
    RoadDamageError,
    UnsupportedFileTypeError,
)
from app.utils.logger import get_logger
from app.utils.validators import (
    ensure_directory,
    sanitize_filename,
    validate_file_size,
    validate_image_extension,
    validate_video_extension,
)

logger = get_logger(__name__)

router = APIRouter()

# Global detector instance (loaded once at startup)
_detector: RoadDamageDetector | None = None
_severity_engine: SeverityEngine | None = None
_evaluator: ModelEvaluator | None = None


def get_detector() -> RoadDamageDetector:
    """Get the global detector instance."""
    global _detector
    if _detector is None:
        _detector = RoadDamageDetector()
    return _detector


def get_severity_engine() -> SeverityEngine:
    """Get the global severity engine instance."""
    global _severity_engine
    if _severity_engine is None:
        _severity_engine = SeverityEngine()
    return _severity_engine


def get_evaluator() -> ModelEvaluator:
    """Get the global evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = ModelEvaluator()
    return _evaluator


def init_detector() -> bool:
    """Initialize the detector at startup. Returns True if model loaded."""
    detector = get_detector()
    try:
        detector.load_model()
        logger.info("Road damage model loaded successfully at startup")
        return True
    except ModelNotFoundError as e:
        logger.warning(f"Model not available: {e.message}")
        return False
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


# =====================================================================
# Health & Status
# =====================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Application health check endpoint."""
    detector = get_detector()
    db_ok = True
    try:
        SessionLocal = get_session_factory()
        db = SessionLocal()
        db.execute(AnalysisSession.__table__.select().limit(1))
        db.close()
    except Exception:
        db_ok = False

    return HealthResponse(
        status="healthy",
        model_loaded=detector.is_loaded(),
        database_connected=db_ok,
        version="1.0.0",
    )


@router.get("/api/model/status", response_model=ModelStatusResponse)
async def model_status():
    """Get current model configuration and status."""
    detector = get_detector()
    info = detector.get_model_info()
    return ModelStatusResponse(**info)


# =====================================================================
# Image Analysis
# =====================================================================

@router.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload and analyze a single image for road damage.

    Returns detection results with bounding boxes, confidence,
    damage classification, and severity estimation.
    """
    settings = get_settings()
    detector = get_detector()

    if not detector.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Road-damage model not loaded. Configure MODEL_PATH "
                   "or place the trained model in models/best.pt.",
        )

    # Validate file
    try:
        validate_image_extension(file.filename or "", settings.allowed_image_extensions)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=e.message)

    # Read file content
    content = await file.read()

    try:
        validate_file_size(len(content), settings.max_image_size_mb)
    except FileTooLargeError as e:
        raise HTTPException(status_code=400, detail=e.message)

    # Save uploaded file
    input_dir = ensure_directory(settings.input_dir)
    safe_name = sanitize_filename(file.filename or "upload.jpg")
    input_path = str(input_dir / safe_name)
    with open(input_path, "wb") as f:
        f.write(content)

    # Create DB session
    SessionLocal = get_session_factory()
    db = SessionLocal()

    try:
        db_session = create_session(db, "image", file.filename or "unknown")

        # Load and process image
        import numpy as np
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            fail_session(db, db_session.id, "Failed to decode image")
            raise HTTPException(status_code=400, detail="Could not decode image file")

        # Run inference
        start_time = time.time()
        result = detector.detect(image)
        processing_time = time.time() - start_time

        # Estimate severity
        severity_engine = get_severity_engine()
        detections = severity_engine.estimate_batch(
            result.detections, result.image_width, result.image_height
        )

        # Draw annotated image
        annotated = draw_detections(image, detections)
        annotated = draw_summary_overlay(
            annotated,
            total_detections=len(detections),
            processing_time=result.inference_time,
        )

        # Save annotated image
        output_dir = ensure_directory(settings.output_dir)
        annotated_name = f"annotated_{safe_name}"
        annotated_path = str(output_dir / annotated_name)
        cv2.imwrite(annotated_path, annotated)

        # Generate evidence images
        evidence_dir = ensure_directory(settings.evidence_dir)
        evidence_paths = []
        for i, det in enumerate(detections):
            ev_img = create_evidence_image(image, det, session_id=db_session.id)
            ev_name = f"evidence_s{db_session.id}_d{i}_{det.class_code}.jpg"
            ev_path = str(evidence_dir / ev_name)
            cv2.imwrite(ev_path, ev_img)
            evidence_paths.append(ev_path)

            # Save detection to DB
            add_detection(
                db,
                session_id=db_session.id,
                frame_number=0,
                timestamp=None,
                damage_type=det.class_code,
                damage_name=det.class_name,
                confidence=det.confidence,
                severity=det.severity,
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                track_id=det.track_id,
                evidence_path=ev_path,
            )

        # Complete session
        complete_session(
            db,
            db_session.id,
            processing_time=processing_time,
            processed_frames=1,
            total_detections=len(detections),
            average_fps=round(1 / processing_time, 2) if processing_time > 0 else 0,
            output_path=annotated_path,
        )

        # Build response
        detection_responses = [
            DetectionResponse(
                class_id=d.class_id,
                class_code=d.class_code,
                class_name=d.class_name,
                confidence=round(d.confidence, 4),
                x1=round(d.x1, 1), y1=round(d.y1, 1),
                x2=round(d.x2, 1), y2=round(d.y2, 1),
                severity=d.severity,
                track_id=d.track_id,
            )
            for d in detections
        ]

        return ImageAnalysisResponse(
            session_id=db_session.id,
            filename=file.filename or "unknown",
            total_detections=len(detections),
            processing_time=round(processing_time, 4),
            image_width=result.image_width,
            image_height=result.image_height,
            detections=detection_responses,
            annotated_image_url=f"/data/output/{annotated_name}",
            original_image_url=f"/data/input/{safe_name}",
            evidence_paths=[f"/data/evidence/{Path(p).name}" for p in evidence_paths],
        )

    except HTTPException:
        raise
    except RoadDamageError as e:
        fail_session(db, db_session.id, e.message)
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        try:
            fail_session(db, db_session.id, str(e))
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Image analysis failed")
    finally:
        db.close()


# =====================================================================
# Video Analysis
# =====================================================================

@router.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload and analyze a video for road damage with object tracking.
    """
    settings = get_settings()
    detector = get_detector()

    if not detector.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Road-damage model not loaded. Configure MODEL_PATH "
                   "or place the trained model in models/best.pt.",
        )

    # Validate file
    try:
        validate_video_extension(file.filename or "", settings.allowed_video_extensions)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=e.message)

    # Read and save uploaded file
    content = await file.read()

    try:
        validate_file_size(len(content), settings.max_video_size_mb)
    except FileTooLargeError as e:
        raise HTTPException(status_code=400, detail=e.message)

    input_dir = ensure_directory(settings.input_dir)
    safe_name = sanitize_filename(file.filename or "upload.mp4")
    input_path = str(input_dir / safe_name)
    with open(input_path, "wb") as f:
        f.write(content)

    # Create DB session
    SessionLocal = get_session_factory()
    db = SessionLocal()

    try:
        # Get video info
        video_info = VideoProcessor.get_video_info(input_path)
        db_session = create_session(
            db, "video", file.filename or "unknown",
            total_frames=video_info["total_frames"],
        )

        # Process video
        severity_engine = get_severity_engine()
        processor = VideoProcessor(detector, severity_engine)

        video_result = processor.process_video(
            video_path=input_path,
            session_id=db_session.id,
        )

        # Save detections to DB (deduplicated by tracker)
        from app.database.repository import add_detections_bulk
        if video_result.frame_detections:
            add_detections_bulk(db, video_result.frame_detections)

        # Complete session
        complete_session(
            db,
            db_session.id,
            processing_time=video_result.processing_time,
            processed_frames=video_result.processed_frames,
            total_detections=len(video_result.frame_detections),
            average_fps=video_result.average_fps,
            output_path=video_result.output_path,
        )

        output_name = Path(video_result.output_path).name

        return VideoAnalysisResponse(
            session_id=db_session.id,
            filename=file.filename or "unknown",
            total_frames=video_result.total_frames,
            processed_frames=video_result.processed_frames,
            total_detections=len(video_result.frame_detections),
            unique_damages=video_result.unique_damages,
            processing_time=video_result.processing_time,
            average_fps=video_result.average_fps,
            output_video_url=f"/data/output/{output_name}",
            tracked_damages=video_result.tracked_damages,
            evidence_paths=[
                f"/data/evidence/{Path(p).name}"
                for p in video_result.evidence_paths
            ],
        )

    except HTTPException:
        raise
    except RoadDamageError as e:
        fail_session(db, db_session.id, e.message)
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        logger.error(f"Video analysis failed: {e}", exc_info=True)
        try:
            fail_session(db, db_session.id, str(e))
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Video analysis failed")
    finally:
        db.close()


# =====================================================================
# History
# =====================================================================

@router.get("/api/analyses")
async def list_analyses(limit: int = 100, offset: int = 0):
    """Get all analysis sessions."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        sessions = get_all_sessions(db, limit=limit, offset=offset)
        return [
            SessionResponse.model_validate(s) for s in sessions
        ]
    finally:
        db.close()


@router.get("/api/analyses/{session_id}")
async def get_analysis(session_id: int):
    """Get a specific analysis session with its detections."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        session = get_session(db, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        detections = get_detections_by_session(db, session_id)
        return {
            "session": SessionResponse.model_validate(session),
            "detections": [
                DetectionRecordResponse.model_validate(d) for d in detections
            ],
        }
    finally:
        db.close()


@router.get("/api/detections")
async def list_detections(limit: int = 500, offset: int = 0):
    """Get all damage detection records."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        detections = get_all_detections(db, limit=limit, offset=offset)
        return [
            DetectionRecordResponse.model_validate(d) for d in detections
        ]
    finally:
        db.close()


# =====================================================================
# Analytics
# =====================================================================

@router.get("/api/analytics")
async def get_analytics():
    """
    Get analytics data generated from real database records.
    Returns summary statistics and Plotly chart configurations.
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        summary = get_analytics_summary(db)

        # Generate charts
        charts = {}
        chart = generate_damage_type_chart(summary["damage_type_distribution"])
        if chart:
            charts["damage_type"] = chart

        chart = generate_severity_chart(summary["severity_distribution"])
        if chart:
            charts["severity"] = chart

        chart = generate_confidence_chart(summary["confidence_values"])
        if chart:
            charts["confidence"] = chart

        chart = generate_detections_per_session_chart(
            summary["detections_per_session"]
        )
        if chart:
            charts["detections_per_session"] = chart

        chart = generate_performance_chart(summary["detections_per_session"])
        if chart:
            charts["performance"] = chart

        return AnalyticsResponse(
            total_sessions=summary["total_sessions"],
            completed_sessions=summary["completed_sessions"],
            total_detections=summary["total_detections"],
            average_confidence=summary["average_confidence"],
            average_fps=summary["average_fps"],
            average_processing_time=summary["average_processing_time"],
            damage_type_distribution=summary["damage_type_distribution"],
            severity_distribution=summary["severity_distribution"],
            detections_per_session=summary["detections_per_session"],
            confidence_values=summary["confidence_values"],
            charts=charts,
        )
    finally:
        db.close()


# =====================================================================
# Evaluation
# =====================================================================

@router.get("/api/evaluation")
async def get_evaluation():
    """Get the latest model evaluation results."""
    evaluator = get_evaluator()
    results = evaluator.get_latest_results()
    return EvaluationResponse(**results)
