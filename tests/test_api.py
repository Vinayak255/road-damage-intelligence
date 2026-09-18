"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "database_connected" in data

    def test_health_has_version(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["version"] == "1.0.0"


class TestModelStatusEndpoint:
    def test_model_status_returns_200(self, client):
        response = client.get("/api/model/status")
        assert response.status_code == 200
        data = response.json()
        assert "model_path" in data
        assert "model_loaded" in data
        assert "class_mapping" in data
        assert "class_names" in data

    def test_model_status_has_config(self, client):
        response = client.get("/api/model/status")
        data = response.json()
        assert "confidence_threshold" in data
        assert "iou_threshold" in data
        assert "image_size" in data


class TestAnalysesEndpoint:
    def test_list_analyses_returns_list(self, client):
        response = client.get("/api/analyses")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestDetectionsEndpoint:
    def test_list_detections_returns_list(self, client):
        response = client.get("/api/detections")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestAnalyticsEndpoint:
    def test_analytics_returns_data(self, client):
        response = client.get("/api/analytics")
        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data
        assert "total_detections" in data
        assert "charts" in data


class TestEvaluationEndpoint:
    def test_evaluation_returns_status(self, client):
        response = client.get("/api/evaluation")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestImageAnalysisValidation:
    def test_no_file_returns_422(self, client):
        response = client.post("/api/analyze/image")
        assert response.status_code == 422

    def test_wrong_extension_returns_400(self, client, monkeypatch):
        # Mock the detector to pretend it's loaded to get past the 503 check
        monkeypatch.setattr("app.api.routes.RoadDamageDetector.is_loaded", lambda self: True)
        response = client.post(
            "/api/analyze/image",
            files={"file": ("test.exe", b"fake content", "application/octet-stream")},
        )
        assert response.status_code == 400


class TestVideoAnalysisValidation:
    def test_no_file_returns_422(self, client):
        response = client.post("/api/analyze/video")
        assert response.status_code == 422

    def test_wrong_extension_returns_400(self, client, monkeypatch):
        # Mock the detector to pretend it's loaded to get past the 503 check
        monkeypatch.setattr("app.api.routes.RoadDamageDetector.is_loaded", lambda self: True)
        response = client.post(
            "/api/analyze/video",
            files={"file": ("test.txt", b"fake content", "text/plain")},
        )
        assert response.status_code == 400


class TestFrontendPages:
    def test_home_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Road Damage Intelligence System" in response.text

    def test_image_analysis_page(self, client):
        response = client.get("/image-analysis")
        assert response.status_code == 200

    def test_video_analysis_page(self, client):
        response = client.get("/video-analysis")
        assert response.status_code == 200

    def test_history_page(self, client):
        response = client.get("/history")
        assert response.status_code == 200

    def test_analytics_page(self, client):
        response = client.get("/analytics")
        assert response.status_code == 200

    def test_model_page(self, client):
        response = client.get("/model")
        assert response.status_code == 200

    def test_evaluation_page(self, client):
        response = client.get("/evaluation")
        assert response.status_code == 200
