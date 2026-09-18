"""
Data access layer (repository pattern) for database operations.

All database queries are centralized here to keep the API routes
and business logic free from raw SQL or ORM queries.
"""

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.database.models import AnalysisSession, DamageDetection
from app.utils.logger import get_logger

logger = get_logger(__name__)


# =====================================================================
# Analysis Sessions
# =====================================================================

def create_session(
    db: Session,
    source_type: str,
    filename: str,
    total_frames: int = 1,
) -> AnalysisSession:
    """Create a new analysis session record."""
    session = AnalysisSession(
        source_type=source_type,
        filename=filename,
        total_frames=total_frames,
        started_at=datetime.utcnow(),
        status="processing",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info(f"Created analysis session {session.id} for {filename}")
    return session


def update_session(
    db: Session,
    session_id: int,
    **kwargs: Any,
) -> AnalysisSession | None:
    """Update fields on an existing analysis session."""
    session = db.query(AnalysisSession).filter(
        AnalysisSession.id == session_id
    ).first()
    if session is None:
        logger.warning(f"Session {session_id} not found for update")
        return None

    for key, value in kwargs.items():
        if hasattr(session, key):
            setattr(session, key, value)

    db.commit()
    db.refresh(session)
    return session


def complete_session(
    db: Session,
    session_id: int,
    processing_time: float,
    processed_frames: int,
    total_detections: int,
    average_fps: float = 0.0,
    output_path: str | None = None,
) -> AnalysisSession | None:
    """Mark a session as completed with final metrics."""
    return update_session(
        db,
        session_id,
        completed_at=datetime.utcnow(),
        processing_time=processing_time,
        processed_frames=processed_frames,
        total_detections=total_detections,
        average_fps=average_fps,
        output_path=output_path,
        status="completed",
    )


def fail_session(
    db: Session,
    session_id: int,
    error_message: str,
) -> AnalysisSession | None:
    """Mark a session as failed."""
    return update_session(
        db,
        session_id,
        completed_at=datetime.utcnow(),
        status="failed",
        error_message=error_message,
    )


def get_session(db: Session, session_id: int) -> AnalysisSession | None:
    """Get a single analysis session by ID."""
    return db.query(AnalysisSession).filter(
        AnalysisSession.id == session_id
    ).first()


def get_all_sessions(
    db: Session,
    limit: int = 100,
    offset: int = 0,
) -> list[AnalysisSession]:
    """Get all analysis sessions, ordered by most recent first."""
    return (
        db.query(AnalysisSession)
        .order_by(AnalysisSession.started_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_session_count(db: Session) -> int:
    """Get total number of analysis sessions."""
    return db.query(AnalysisSession).count()


# =====================================================================
# Damage Detections
# =====================================================================

def add_detection(
    db: Session,
    session_id: int,
    frame_number: int,
    timestamp: str | None,
    damage_type: str,
    damage_name: str,
    confidence: float,
    severity: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    track_id: int | None = None,
    evidence_path: str | None = None,
) -> DamageDetection:
    """Add a damage detection record to a session."""
    detection = DamageDetection(
        session_id=session_id,
        frame_number=frame_number,
        timestamp=timestamp,
        damage_type=damage_type,
        damage_name=damage_name,
        confidence=confidence,
        severity=severity,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        track_id=track_id,
        evidence_path=evidence_path,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def add_detections_bulk(
    db: Session,
    detections: list[dict],
) -> int:
    """Add multiple detection records efficiently."""
    objects = [DamageDetection(**d) for d in detections]
    db.add_all(objects)
    db.commit()
    return len(objects)


def get_detections_by_session(
    db: Session,
    session_id: int,
) -> list[DamageDetection]:
    """Get all detections for a specific session."""
    return (
        db.query(DamageDetection)
        .filter(DamageDetection.session_id == session_id)
        .order_by(DamageDetection.frame_number, DamageDetection.id)
        .all()
    )


def get_all_detections(
    db: Session,
    limit: int = 500,
    offset: int = 0,
) -> list[DamageDetection]:
    """Get all detections across all sessions."""
    return (
        db.query(DamageDetection)
        .order_by(DamageDetection.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_detection_count(db: Session) -> int:
    """Get total number of damage detections."""
    return db.query(DamageDetection).count()


# =====================================================================
# Analytics Queries
# =====================================================================

def get_analytics_summary(db: Session) -> dict:
    """
    Generate analytics summary from database data.

    Returns a dictionary with aggregate statistics for the
    analytics dashboard.
    """
    from sqlalchemy import func

    total_sessions = db.query(AnalysisSession).count()
    completed_sessions = (
        db.query(AnalysisSession)
        .filter(AnalysisSession.status == "completed")
        .count()
    )
    total_detections = db.query(DamageDetection).count()

    # Damage type distribution
    type_distribution = (
        db.query(
            DamageDetection.damage_type,
            DamageDetection.damage_name,
            func.count(DamageDetection.id).label("count"),
        )
        .group_by(DamageDetection.damage_type, DamageDetection.damage_name)
        .all()
    )

    # Severity distribution
    severity_distribution = (
        db.query(
            DamageDetection.severity,
            func.count(DamageDetection.id).label("count"),
        )
        .group_by(DamageDetection.severity)
        .all()
    )

    # Average confidence
    avg_confidence_result = (
        db.query(func.avg(DamageDetection.confidence)).scalar()
    )
    avg_confidence = float(avg_confidence_result) if avg_confidence_result else 0.0

    # Processing performance
    avg_fps_result = (
        db.query(func.avg(AnalysisSession.average_fps))
        .filter(AnalysisSession.status == "completed")
        .scalar()
    )
    avg_fps = float(avg_fps_result) if avg_fps_result else 0.0

    avg_time_result = (
        db.query(func.avg(AnalysisSession.processing_time))
        .filter(AnalysisSession.status == "completed")
        .scalar()
    )
    avg_processing_time = float(avg_time_result) if avg_time_result else 0.0

    # Detections per session
    detections_per_session = (
        db.query(
            AnalysisSession.id,
            AnalysisSession.filename,
            AnalysisSession.total_detections,
            AnalysisSession.processing_time,
        )
        .filter(AnalysisSession.status == "completed")
        .order_by(AnalysisSession.started_at.desc())
        .limit(50)
        .all()
    )

    # Confidence distribution (binned)
    confidence_values = (
        db.query(DamageDetection.confidence).all()
    )

    return {
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
        "total_detections": total_detections,
        "damage_type_distribution": [
            {
                "type": row.damage_type,
                "name": row.damage_name,
                "count": row.count,
            }
            for row in type_distribution
        ],
        "severity_distribution": [
            {"severity": row.severity, "count": row.count}
            for row in severity_distribution
        ],
        "average_confidence": round(avg_confidence, 4),
        "average_fps": round(avg_fps, 2),
        "average_processing_time": round(avg_processing_time, 2),
        "detections_per_session": [
            {
                "session_id": row.id,
                "filename": row.filename,
                "detections": row.total_detections,
                "processing_time": row.processing_time,
            }
            for row in detections_per_session
        ],
        "confidence_values": [row.confidence for row in confidence_values],
    }
