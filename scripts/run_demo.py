"""
Demo script — starts the application server.

Usage:
    python scripts/run_demo.py

This provides a quick way to start the application for demonstration.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    import uvicorn
    from app.config.settings import get_settings

    settings = get_settings()

    print("=" * 60)
    print("Road Damage Intelligence System — Demo")
    print("=" * 60)
    print(f"Frontend:  http://localhost:{settings.port}")
    print(f"API Docs:  http://localhost:{settings.port}/docs")
    print(f"Health:    http://localhost:{settings.port}/health")
    print("=" * 60)
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
