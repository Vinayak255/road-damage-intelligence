"""
FastAPI application entry point.

Creates the FastAPI app, mounts static files, includes API routes,
serves frontend templates, and handles startup/shutdown events.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __project__, __version__
from app.api.routes import init_detector, router
from app.config.settings import get_settings
from app.database.database import init_db
from app.utils.exceptions import RoadDamageError
from app.utils.logger import get_logger, setup_logging
from app.utils.validators import ensure_directory

logger = get_logger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"
DATA_DIR = PROJECT_ROOT / "data"

# Create FastAPI app
app = FastAPI(
    title=__project__,
    version=__version__,
    description=(
        "A Computer Vision application for automated road damage "
        "detection, classification, severity estimation, and analytics."
    ),
)


# =====================================================================
# Startup & Shutdown
# =====================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize logging, database, model, and directories on startup."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(f"Starting {__project__} v{__version__}")

    # Ensure data directories exist
    ensure_directory(DATA_DIR / "input")
    ensure_directory(DATA_DIR / "output")
    ensure_directory(DATA_DIR / "evidence")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Load model (non-fatal if missing)
    model_loaded = init_detector()
    if model_loaded:
        logger.info("Road damage model loaded and ready")
    else:
        logger.warning(
            "Road damage model not available. Image/video analysis will "
            "return an error until a trained model is configured."
        )

    logger.info(f"Server ready on {settings.host}:{settings.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Road Damage Intelligence System")


# =====================================================================
# Error Handlers
# =====================================================================

@app.exception_handler(RoadDamageError)
async def road_damage_error_handler(request: Request, exc: RoadDamageError):
    """Handle custom application errors."""
    return JSONResponse(
        status_code=400,
        content={"error": type(exc).__name__, "detail": exc.message},
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle unexpected errors without exposing stack traces."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Please check the logs.",
        },
    )


# =====================================================================
# Mount Static Files & Data
# =====================================================================

# Static assets (CSS, JS)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Data files (uploaded images, output, evidence)
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# Include API routes
app.include_router(router)

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =====================================================================
# Frontend Pages
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/image-analysis", response_class=HTMLResponse)
async def image_analysis_page(request: Request):
    """Image analysis page."""
    return templates.TemplateResponse("image_analysis.html", {"request": request})


@app.get("/video-analysis", response_class=HTMLResponse)
async def video_analysis_page(request: Request):
    """Video analysis page."""
    return templates.TemplateResponse("video_analysis.html", {"request": request})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """Analysis history page."""
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics dashboard page."""
    return templates.TemplateResponse("analytics.html", {"request": request})


@app.get("/model", response_class=HTMLResponse)
async def model_page(request: Request):
    """Model status page."""
    return templates.TemplateResponse("model.html", {"request": request})


@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation_page(request: Request):
    """Model evaluation page."""
    return templates.TemplateResponse("evaluation.html", {"request": request})
