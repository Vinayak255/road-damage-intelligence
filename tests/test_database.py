"""Tests for database models and repository functions."""

from datetime import datetime

from app.database.models import AnalysisSession, DamageDetection
from app.database.repository import (
    create_session,
    complete_session,
    fail_session,
    get_session,
    get_all_sessions,
    add_detection,
    get_detections_by_session,
    get_detection_count,
    get_analytics_summary,
)


class TestAnalysisSessionModel:
    def test_create_session(self, db_session):
        session = create_session(db_session, "image", "test.jpg")
        assert session.id is not None
        assert session.source_type == "image"
        assert session.filename == "test.jpg"
        assert session.status == "processing"

    def test_complete_session(self, db_session):
        session = create_session(db_session, "image", "test.jpg")
        updated = complete_session(
            db_session, session.id,
            processing_time=1.5, processed_frames=1,
            total_detections=3, average_fps=0.67,
        )
        assert updated.status == "completed"
        assert updated.processing_time == 1.5
        assert updated.total_detections == 3
        assert updated.completed_at is not None

    def test_fail_session(self, db_session):
        session = create_session(db_session, "video", "test.mp4")
        updated = fail_session(db_session, session.id, "Model not found")
        assert updated.status == "failed"
        assert updated.error_message == "Model not found"

    def test_get_session(self, db_session):
        session = create_session(db_session, "image", "test.jpg")
        fetched = get_session(db_session, session.id)
        assert fetched is not None
        assert fetched.id == session.id

    def test_get_nonexistent_session(self, db_session):
        fetched = get_session(db_session, 99999)
        assert fetched is None

    def test_get_all_sessions_ordering(self, db_session):
        create_session(db_session, "image", "first.jpg")
        create_session(db_session, "image", "second.jpg")
        sessions = get_all_sessions(db_session)
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0].filename == "second.jpg"


class TestDamageDetectionModel:
    def test_add_detection(self, db_session):
        session = create_session(db_session, "image", "test.jpg")
        det = add_detection(
            db_session, session_id=session.id,
            frame_number=0, timestamp=None,
            damage_type="D40", damage_name="Pothole",
            confidence=0.91, severity="High",
            x1=100, y1=200, x2=300, y2=400,
            track_id=None, evidence_path="data/evidence/test.jpg",
        )
        assert det.id is not None
        assert det.damage_type == "D40"
        assert det.confidence == 0.91
        assert det.severity == "High"

    def test_get_detections_by_session(self, db_session):
        session = create_session(db_session, "image", "test.jpg")
        add_detection(
            db_session, session.id, 0, None,
            "D00", "Longitudinal Crack", 0.8, "Low",
            10, 20, 30, 40,
        )
        add_detection(
            db_session, session.id, 0, None,
            "D40", "Pothole", 0.95, "High",
            50, 60, 150, 160,
        )
        dets = get_detections_by_session(db_session, session.id)
        assert len(dets) == 2

    def test_detection_count(self, db_session):
        session = create_session(db_session, "image", "test.jpg")
        add_detection(
            db_session, session.id, 0, None,
            "D00", "Test", 0.8, "Low", 0, 0, 10, 10,
        )
        count = get_detection_count(db_session)
        assert count == 1

    def test_foreign_key_relationship(self, db_session):
        session = create_session(db_session, "image", "test.jpg")
        add_detection(
            db_session, session.id, 0, None,
            "D40", "Pothole", 0.9, "Medium", 0, 0, 10, 10,
        )
        fetched = get_session(db_session, session.id)
        assert len(fetched.detections) == 1
        assert fetched.detections[0].damage_type == "D40"


class TestAnalytics:
    def test_empty_analytics(self, db_session):
        summary = get_analytics_summary(db_session)
        assert summary["total_sessions"] == 0
        assert summary["total_detections"] == 0
        assert summary["average_confidence"] == 0.0

    def test_analytics_with_data(self, db_session):
        session = create_session(db_session, "image", "test.jpg")
        complete_session(
            db_session, session.id,
            processing_time=1.0, processed_frames=1,
            total_detections=2, average_fps=1.0,
        )
        add_detection(
            db_session, session.id, 0, None,
            "D40", "Pothole", 0.9, "High", 0, 0, 100, 100,
        )
        add_detection(
            db_session, session.id, 0, None,
            "D00", "Longitudinal Crack", 0.8, "Low", 0, 0, 10, 10,
        )

        summary = get_analytics_summary(db_session)
        assert summary["total_sessions"] == 1
        assert summary["total_detections"] == 2
        assert summary["average_confidence"] == 0.85
        assert len(summary["damage_type_distribution"]) == 2
        assert len(summary["severity_distribution"]) == 2
