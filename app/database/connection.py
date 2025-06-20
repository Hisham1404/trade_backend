from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator

from app.core.config import settings

# Create synchronous engine for basic operations
# For SQLite, we need to use check_same_thread=False
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    connect_args=connect_args
)

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Dependency to get database session
def get_db() -> Generator:
    """Get database session for sync operations"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create database tables
def create_db_and_tables_sync():
    """Create database tables synchronously"""
    Base.metadata.create_all(bind=engine)

# For the lifespan context manager in main.py
async def create_db_and_tables():
    """Create database tables for application startup"""
    # Import all models so they're registered with Base
    from app.models import User, Portfolio, Asset, NewsItem, Source, Alert
    
    # For now, we'll use sync creation since we're using SQLite
    Base.metadata.create_all(bind=engine) 