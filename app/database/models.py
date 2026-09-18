"""
SQLAlchemy ORM models for the Road Damage Intelligence System.

Tables:
    - analysis_sessions: Records of each image/video analysis run.
    - damage_detections: Individual damage detections with bounding boxes.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class AnalysisSession(Base):
    """
    Records a single analysis session (image or video).

    Each upload and analysis creates one session, which can
    contain multiple damage detections.
    """

    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String(10), nullable=False)  # "image" or "video"
    filename = Column(String(255), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    total_frames = Column(Integer, default=1)
    processed_frames = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)  # seconds
    average_fps = Column(Float, default=0.0)
    total_detections = Column(Integer, default=0)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    output_path = Column(String(500), nullable=True)

    # Relationship
    detections = relationship(
        "DamageDetection",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisSession(id={self.id}, type={self.source_type}, "
            f"file={self.filename}, detections={self.total_detections})>"
        )


class DamageDetection(Base):
    """
    Records a single damage detection within an analysis session.

    Contains bounding box coordinates, classification, confidence,
    severity estimation, and optional tracking information.
    """

    __tablename__ = "damage_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        Integer,
        ForeignKey("analysis_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    frame_number = Column(Integer, default=0)
    timestamp = Column(String(20), nullable=True)  # video timestamp "HH:MM:SS"
    damage_type = Column(String(50), nullable=False)  # e.g., "D40"
    damage_name = Column(String(100), nullable=True)  # e.g., "Pothole"
    confidence = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)  # "Low", "Medium", "High"
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=True)
    evidence_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("AnalysisSession", back_populates="detections")

    def __repr__(self) -> str:
        return (
            f"<DamageDetection(id={self.id}, type={self.damage_type}, "
            f"conf={self.confidence:.2f}, severity={self.severity})>"
        )
