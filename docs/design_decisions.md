# Design Decisions

## Key Technical Decisions

### 1. FastAPI as Backend Framework

| | |
|--|--|
| **Decision** | Use FastAPI for the REST API |
| **Reason** | Async support, automatic OpenAPI docs, Pydantic validation, type hints, high performance |
| **Alternative** | Flask, Django |
| **Trade-off** | FastAPI has a smaller ecosystem than Django, but for an API-first application it's more appropriate |

### 2. SQLite as Database

| | |
|--|--|
| **Decision** | Use SQLite for data persistence |
| **Reason** | Zero-configuration, file-based, no external server required, perfect for single-user desktop/demo applications |
| **Alternative** | PostgreSQL, MySQL |
| **Trade-off** | Limited concurrent write performance, but sufficient for this use case |

### 3. YOLO for Object Detection

| | |
|--|--|
| **Decision** | Use Ultralytics YOLOv8 for damage detection |
| **Reason** | Single-pass detection, excellent speed-accuracy trade-off, built-in tracking, easy fine-tuning |
| **Alternative** | Faster R-CNN, SSD, EfficientDet, DETR |
| **Trade-off** | May not achieve highest possible accuracy vs. two-stage detectors, but significantly faster |

### 4. OpenCV for Image Processing

| | |
|--|--|
| **Decision** | Use OpenCV for all image/video I/O and preprocessing |
| **Reason** | Industry standard, comprehensive API, efficient C++ backend, wide format support |
| **Alternative** | Pillow, scikit-image |
| **Trade-off** | BGR default color space (vs RGB), larger dependency, but unmatched video processing capabilities |

### 5. ByteTrack for Object Tracking

| | |
|--|--|
| **Decision** | Use ByteTrack via Ultralytics integration |
| **Reason** | Lightweight, no separate model required, good performance on detection-based tracking |
| **Alternative** | DeepSORT, StrongSORT, OCSORT |
| **Trade-off** | No appearance-based re-identification (may produce ID switches), but simpler and faster |

### 6. Modular Architecture

| | |
|--|--|
| **Decision** | Separate code into distinct modules (API, CV, analysis, database, utils) |
| **Reason** | Maintainability, testability, clear separation of concerns, easier to explain in viva |
| **Alternative** | Monolithic single-file application |
| **Trade-off** | More files to manage, but each module is focused and understandable |

### 7. Configuration-Based Model Loading

| | |
|--|--|
| **Decision** | Load model path and class mapping from environment variables |
| **Reason** | Allows swapping models without code changes, supports different trained models |
| **Alternative** | Hardcoded model path |
| **Trade-off** | Requires .env setup, but enables flexibility |

### 8. Visual Severity Instead of Physical Severity

| | |
|--|--|
| **Decision** | Estimate severity from image-space bounding box area, not physical dimensions |
| **Reason** | Physical size estimation requires camera calibration and depth information, which is unavailable from single images |
| **Alternative** | Depth estimation, stereo vision, LiDAR fusion |
| **Trade-off** | Severity is approximate and relative to image framing, but transparent and honest about limitations |

### 9. Repository Pattern for Database Access

| | |
|--|--|
| **Decision** | Centralize all database queries in repository.py |
| **Reason** | Keeps API routes clean, enables testing with mock DB, single source of truth for queries |
| **Alternative** | Inline ORM queries in route handlers |
| **Trade-off** | Extra layer of indirection, but significantly improved maintainability |

### 10. Inference-First Design

| | |
|--|--|
| **Decision** | Application is designed for inference, not training |
| **Reason** | Training is a one-time operation; the application's primary purpose is detection |
| **Alternative** | Include training in the main application |
| **Trade-off** | Users must train separately, but application startup is fast and focused |
