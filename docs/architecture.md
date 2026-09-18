# Architecture

## Road Damage Intelligence System — System Architecture

### High-Level Architecture

```
                        USER
                         |
                         v
                  WEB FRONTEND
              (HTML / CSS / JS)
                         |
                         v
                    FASTAPI
                  (REST API)
                         |
              +----------+----------+
              |                     |
              v                     v
         IMAGE SERVICE         VIDEO SERVICE
              |                     |
              +----------+----------+
                         |
                         v
                  PREPROCESSING
                   (OpenCV)
                         |
                         v
                  YOLO DETECTOR
                 (Ultralytics)
                         |
                +--------+--------+
                |                 |
                v                 v
           DAMAGE ANALYSIS     TRACKER
          (Severity Engine)   (ByteTrack)
                |                 |
                +--------+--------+
                         |
                         v
                 SEVERITY ENGINE
                         |
                +--------+--------+
                |                 |
                v                 v
           EVIDENCE ENGINE     DATABASE
          (Annotated Images)   (SQLite)
                                  |
                                  v
                             ANALYTICS
                             (Plotly)
```

### Module Architecture

```
app/
├── main.py              ← FastAPI application entry point
├── api/                 ← REST API layer
│   ├── routes.py        ← Endpoint definitions
│   └── schemas.py       ← Pydantic request/response models
├── cv/                  ← Computer Vision layer
│   ├── detector.py      ← YOLO model abstraction
│   ├── preprocessing.py ← Image preprocessing (OpenCV)
│   ├── video_processor.py ← Frame-by-frame video pipeline
│   ├── tracker.py       ← Object tracking (ByteTrack)
│   └── visualization.py ← Bounding box drawing, evidence images
├── analysis/            ← Business logic layer
│   ├── severity.py      ← Heuristic severity estimation
│   └── statistics.py    ← Analytics chart generation (Plotly)
├── database/            ← Data access layer
│   ├── database.py      ← Engine & session management
│   ├── models.py        ← ORM models (SQLAlchemy)
│   └── repository.py    ← CRUD operations
├── evaluation/          ← Model evaluation
│   └── evaluator.py     ← Precision, Recall, mAP
├── config/              ← Configuration
│   └── settings.py      ← Environment-based settings
└── utils/               ← Shared utilities
    ├── logger.py        ← Centralized logging
    ├── validators.py    ← Input validation
    └── exceptions.py    ← Custom exceptions
```

### Data Flow

#### Image Analysis Pipeline

```
Upload Image → Validate → Read (OpenCV) → Preprocess
    → YOLO Inference → Parse Detections → Estimate Severity
    → Draw Bounding Boxes → Generate Evidence → Save to DB → Return Results
```

#### Video Analysis Pipeline

```
Upload Video → Validate → Open (VideoCapture)
    → For each frame:
        → Preprocess → YOLO Inference with Tracking
        → Parse Detections + Track IDs → Estimate Severity
        → Deduplication Check → Generate Evidence (if new)
        → Annotate Frame → Write to Output Video
    → Save session to DB → Return Results
```

### Key Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Configuration-Driven**: All thresholds, paths, and mappings are configurable
3. **Model Abstraction**: The detector wraps Ultralytics YOLO behind a clean interface
4. **Repository Pattern**: Database access is centralized, not scattered in routes
5. **Graceful Degradation**: Application starts even if model is missing
