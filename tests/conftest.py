"""
Pytest configuration and shared fixtures.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables before importing app modules
os.environ["DATABASE_URL"] = "sqlite:///./test_road_damage.db"
os.environ["MODEL_PATH"] = "models/test_model.pt"
os.environ["EVIDENCE_DIR"] = "data/test_evidence"
os.environ["DEVICE"] = "cpu"


@pytest.fixture
def settings():
    """Provide test settings."""
    from app.config.settings import Settings, reset_settings
    reset_settings()
    s = Settings()
    yield s
    reset_settings()


@pytest.fixture
def db_session():
    """Provide an in-memory database session for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sample_detections():
    """Provide sample Detection objects for testing."""
    from app.cv.detector import Detection
    return [
        Detection(
            class_id=0, class_code="D00",
            class_name="Longitudinal Crack",
            confidence=0.85, x1=100, y1=200, x2=300, y2=400,
        ),
        Detection(
            class_id=3, class_code="D40",
            class_name="Pothole",
            confidence=0.92, x1=50, y1=50, x2=250, y2=250,
        ),
    ]
