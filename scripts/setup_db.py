"""
Standalone script to initialize the database.

Usage:
    python scripts/setup_db.py

This creates all tables defined in the ORM models.
Safe to run multiple times.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.database.database import init_db
from app.utils.logger import setup_logging, get_logger


def main():
    setup_logging("INFO")
    logger = get_logger(__name__)

    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialization complete.")


if __name__ == "__main__":
    main()
