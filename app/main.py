from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import logging

from app.database.connection import create_db_and_tables
from app.core.config import settings
from app.routers import auth, users, portfolio, assets, news, alerts, monitoring, adk, sentiment, feeds, market_impact, websockets, push_notifications, delivery_tracking, acknowledgments, analysis_endpoints
from app.routers import celery_monitoring
from app.services import setup_logging
from app.services.monitoring_service import monitoring_service
from app.services.adk_service import get_adk_service, shutdown_adk_service
from app.services.sentiment_service import get_sentiment_service, shutdown_sentiment_service
from app.services.alerting_service import get_alerting_service, shutdown_alerting_service

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
    
    # Initialize alerting service
    try:
        alerting_service = await get_alerting_service()
        logger.info("Alerting service initialized successfully")
    except Exception as e:
        logger.warning(f"Alerting service initialization failed: {str(e)} - Alert generation will be disabled")
    
    # Start push notification service
    try:
        from app.services.push_notification_service import get_push_notification_service
        push_service = await get_push_notification_service()
        logger.info("Push notification service initialized")
    except ImportError:
        logger.warning("Push notification service not available")
    except Exception as e:
        logger.error(f"Failed to initialize push notification service: {e}")
    
    # Initialize delivery tracking service
    try:
        from app.services.delivery_tracking_service import get_delivery_tracking_service
        delivery_service = get_delivery_tracking_service()
        logger.info("Delivery tracking service initialized")
    except ImportError:
        logger.warning("Delivery tracking service not available")
    except Exception as e:
        logger.error(f"Failed to initialize delivery tracking service: {e}")
    
    # Initialize acknowledgment service and start timeout processing
    try:
        from app.services.acknowledgment_service import get_acknowledgment_service
        acknowledgment_service = get_acknowledgment_service()
        
        # Start background task for processing timeouts
        import asyncio
        async def process_acknowledgment_timeouts():
            while True:
                try:
                    processed_count = await acknowledgment_service.process_timeouts()
                    if processed_count > 0:
                        logger.info(f"Processed {processed_count} acknowledgment timeouts")
                    await asyncio.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error processing acknowledgment timeouts: {e}")
                    await asyncio.sleep(30)  # Shorter retry interval on error
        
        # Start the timeout processor in the background
        asyncio.create_task(process_acknowledgment_timeouts())
        logger.info("Acknowledgment service initialized with timeout processing")
    except ImportError:
        logger.warning("Acknowledgment service not available")
    except Exception as e:
        logger.error(f"Failed to initialize acknowledgment service: {e}")
    
    logger.info("All services started successfully")
    yield
    # Shutdown
    logger.info("Shutting down Trading Intelligence Agent...")
    
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
    
    # Shutdown alerting service
    try:
        await shutdown_alerting_service()
        logger.info("Alerting service shut down successfully")
    except Exception as e:
        logger.warning(f"Alerting service shutdown warning: {str(e)}")
    
    # Close push notification service
    try:
        from app.services.push_notification_service import get_push_notification_service
        push_service = await get_push_notification_service()
        if push_service:
            await push_service.close()
            logger.info("Push notification service closed")
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Error closing push notification service: {e}")
    
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
app.include_router(assets.router, prefix="/api/v1/assets", tags=["assets"])
app.include_router(news.router, prefix="/api/v1/news", tags=["news"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(adk.router, prefix="/api/v1", tags=["adk"])
app.include_router(sentiment.router, prefix="/api/v1", tags=["sentiment"])
app.include_router(feeds.router, prefix="/api/v1/feeds", tags=["feeds"])
app.include_router(portfolio.router, prefix="/api/v1/portfolios", tags=["portfolios"])
app.include_router(market_impact.router, prefix="/api/v1/market-impact", tags=["market-impact"])
app.include_router(websockets.router)
app.include_router(monitoring.router)
app.include_router(push_notifications.router)
app.include_router(delivery_tracking.router)
app.include_router(acknowledgments.router)
app.include_router(celery_monitoring.router)
app.include_router(analysis_endpoints.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    ) 