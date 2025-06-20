from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import logging

from app.database.connection import create_db_and_tables
from app.core.config import settings
from app.routers import auth, users, portfolio, assets, news, alerts, scrapers, monitoring, adk, sentiment, feeds, market_impact, websockets
from app.services import setup_logging
from app.services.monitoring_service import monitoring_service
from app.services.adk_service import get_adk_service, shutdown_adk_service
from app.services.sentiment_service import get_sentiment_service, shutdown_sentiment_service

# Load environment variables
load_dotenv()

# Set up structured logging first
setup_logging()
logger = logging.getLogger(__name__)

# Create the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Trading Intelligence Agent...")
    await create_db_and_tables()
    # Start background scraping every 30 minutes
    from app.scrapers.manager import scraper_manager
    scraper_manager.start_background_scraping(interval_minutes=30, min_check_interval=30)
    # Start monitoring service
    await monitoring_service.start_monitoring(
        health_check_interval=60,  # Check health every minute
        alert_check_interval=30    # Check alerts every 30 seconds
    )
    # Initialize ADK service
    try:
        adk_service = await get_adk_service()
        logger.info("ADK service initialized successfully")
    except Exception as e:
        logger.warning(f"ADK service initialization failed: {str(e)} - ADK features will be disabled")
    
    # Initialize sentiment analysis service
    try:
        sentiment_service = await get_sentiment_service()
        logger.info("Sentiment analysis service initialized successfully")
    except Exception as e:
        logger.warning(f"Sentiment service initialization failed: {str(e)} - Sentiment analysis will be disabled")
    logger.info("All services started successfully")
    yield
    # Shutdown
    logger.info("Shutting down Trading Intelligence Agent...")
    # Stop background scraping gracefully
    await scraper_manager.stop_background_scraping()
    # Stop monitoring service
    await monitoring_service.stop_monitoring()
    # Shutdown ADK service
    try:
        await shutdown_adk_service()
        logger.info("ADK service shut down successfully")
    except Exception as e:
        logger.warning(f"ADK service shutdown warning: {str(e)}")
    
    # Shutdown sentiment service
    try:
        await shutdown_sentiment_service()
        logger.info("Sentiment service shut down successfully")
    except Exception as e:
        logger.warning(f"Sentiment service shutdown warning: {str(e)}")
    logger.info("All services stopped successfully")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered trading intelligence platform with portfolio management and real-time market insights",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Depends(api_key_header)):
    """Validate API key"""
    # This will be implemented with proper database lookup
    if api_key != "temporary-api-key":  # Replace with proper validation
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

# Root endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Trading Intelligence Agent API",
        "version": settings.APP_VERSION,
        "status": "running",
        "health_check": "/monitoring/health",
        "metrics": "/monitoring/metrics",
        "docs": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Trading Intelligence Agent",
        "version": settings.APP_VERSION
    }

# Protected endpoint example
@app.get("/api/v1/protected")
async def protected_route(api_key: str = Depends(get_api_key)):
    return {"message": "This is a protected endpoint", "api_key": api_key}

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(assets.router, prefix="/api/v1/assets", tags=["assets"])
app.include_router(news.router, prefix="/api/v1/news", tags=["news"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(scrapers.router, prefix="/api/v1", tags=["scrapers"])
app.include_router(adk.router, prefix="/api/v1", tags=["adk"])
app.include_router(sentiment.router, prefix="/api/v1", tags=["sentiment"])
app.include_router(feeds.router, prefix="/api/v1/feeds", tags=["feeds"])
app.include_router(portfolio.router, prefix="/api/v1/portfolios", tags=["portfolios"])
app.include_router(market_impact.router, prefix="/api/v1/market-impact", tags=["market-impact"])
app.include_router(websockets.router)
app.include_router(monitoring.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    ) 