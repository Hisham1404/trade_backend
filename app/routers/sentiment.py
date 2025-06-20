"""
Sentiment Analysis Router
REST API endpoints for sentiment analysis functionality
Provides comprehensive sentiment analysis capabilities for trading system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.sentiment_service import get_sentiment_service, SentimentService
from ..analysis.sentiment_engine import SentimentResult, BatchSentimentResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


# Request/Response Models
class SentimentAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for sentiment", min_length=1)
    asset_symbol: Optional[str] = Field(None, description="Asset symbol related to the text")
    source: Optional[str] = Field("api", description="Source of the text")
    preprocess: Optional[bool] = Field(True, description="Whether to preprocess the text")
    model_key: Optional[str] = Field(None, description="Specific model to use (financial, general, news)")


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    asset_symbols: Optional[List[str]] = Field(None, description="Asset symbols for each text")
    source: Optional[str] = Field("batch_api", description="Source of the texts")
    preprocess: Optional[bool] = Field(True, description="Whether to preprocess the texts")
    model_key: Optional[str] = Field(None, description="Specific model to use")


class NewsAnalysisRequest(BaseModel):
    articles: List[Dict[str, Any]] = Field(..., description="News articles to analyze", min_items=1, max_items=50)


class QueueAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text to queue for analysis")
    asset_symbol: Optional[str] = Field(None, description="Asset symbol related to the text")
    source: Optional[str] = Field("queue_api", description="Source of the text")
    priority: Optional[int] = Field(1, description="Priority level (1-5)", ge=1, le=5)


class SentimentResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str
    timestamp: str


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    sentiment_service: SentimentService = Depends(get_sentiment_service)
):
    """
    Analyze sentiment of a single text
    
    Returns detailed sentiment analysis including:
    - Sentiment label (positive/negative/neutral)
    - Confidence score
    - Intensity level
    - Normalized score (-1 to 1)
    - Processing metrics
    """
    try:
        result = await sentiment_service.analyze_text(
            text=request.text,
            asset_symbol=request.asset_symbol,
            source=request.source
        )
        
        return SentimentResponse(
            success=True,
            data=result.to_dict(),
            message="Sentiment analysis completed successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


@router.post("/analyze/batch", response_model=SentimentResponse)
async def analyze_batch_sentiment(
    request: BatchSentimentRequest,
    sentiment_service: SentimentService = Depends(get_sentiment_service)
):
    """
    Analyze sentiment for multiple texts in batch
    
    Efficiently processes multiple texts and returns:
    - Individual sentiment results
    - Batch statistics
    - Sentiment distribution
    - Processing metrics
    """
    try:
        # Validate asset_symbols length if provided
        if request.asset_symbols and len(request.asset_symbols) != len(request.texts):
            raise HTTPException(
                status_code=400,
                detail="Number of asset symbols must match number of texts"
            )
        
        result = await sentiment_service.analyze_batch(
            texts=request.texts,
            asset_symbols=request.asset_symbols,
            source=request.source
        )
        
        return SentimentResponse(
            success=True,
            data=result.to_dict(),
            message=f"Batch sentiment analysis completed for {len(request.texts)} texts",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch sentiment analysis failed: {str(e)}"
        )


@router.post("/analyze/news", response_model=SentimentResponse)
async def analyze_news_sentiment(
    request: NewsAnalysisRequest,
    sentiment_service: SentimentService = Depends(get_sentiment_service)
):
    """
    Analyze sentiment impact of news articles
    
    Specialized analysis for news content including:
    - Overall sentiment impact
    - Asset-specific sentiment
    - Market impact assessment
    - Sentiment distribution across articles
    """
    try:
        result = await sentiment_service.analyze_news_impact(request.articles)
        
        return SentimentResponse(
            success=True,
            data=result,
            message=f"News sentiment analysis completed for {len(request.articles)} articles",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in news sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"News sentiment analysis failed: {str(e)}"
        )


@router.post("/queue", response_model=SentimentResponse)
async def queue_sentiment_analysis(
    request: QueueAnalysisRequest,
    background_tasks: BackgroundTasks,
    sentiment_service: SentimentService = Depends(get_sentiment_service)
):
    """
    Queue text for background sentiment analysis
    
    Adds text to processing queue for asynchronous analysis.
    Useful for high-volume processing scenarios.
    """
    try:
        await sentiment_service.queue_for_analysis(
            text=request.text,
            asset_symbol=request.asset_symbol,
            source=request.source,
            priority=request.priority
        )
        
        return SentimentResponse(
            success=True,
            data={
                "queued": True,
                "priority": request.priority,
                "queue_size": sentiment_service.processing_queue.qsize()
            },
            message="Text queued for sentiment analysis",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error queuing sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue sentiment analysis: {str(e)}"
        )


@router.get("/asset/{asset_symbol}", response_model=SentimentResponse)
async def get_asset_sentiment(
    asset_symbol: str,
    sentiment_service: SentimentService = Depends(get_sentiment_service)
):
    """
    Get current sentiment data for a specific asset
    
    Returns:
    - Latest sentiment analysis
    - Sentiment trend information
    - Historical sentiment summary
    """
    try:
        sentiment_data = sentiment_service.get_asset_sentiment(asset_symbol.upper())
        
        if not sentiment_data:
            return SentimentResponse(
                success=True,
                data={},
                message=f"No sentiment data available for {asset_symbol}",
                timestamp=datetime.now().isoformat()
            )
        
        return SentimentResponse(
            success=True,
            data=sentiment_data,
            message=f"Sentiment data retrieved for {asset_symbol}",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving asset sentiment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve asset sentiment: {str(e)}"
        )


@router.get("/market", response_model=SentimentResponse)
async def get_market_sentiment(
    sentiment_service: SentimentService = Depends(get_sentiment_service)
):
    """
    Get overall market sentiment summary
    
    Returns:
    - Overall market sentiment
    - Sentiment distribution
    - Average sentiment score
    - Market sentiment trends
    """
    try:
        market_data = sentiment_service.get_market_sentiment()
        
        return SentimentResponse(
            success=True,
            data=market_data,
            message="Market sentiment data retrieved successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving market sentiment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve market sentiment: {str(e)}"
        )


@router.get("/status", response_model=SentimentResponse)
async def get_sentiment_service_status(
    sentiment_service: SentimentService = Depends(get_sentiment_service)
):
    """
    Get sentiment service status and configuration
    
    Returns:
    - Service initialization status
    - Available models information
    - Processing queue status
    - Performance metrics
    """
    try:
        status_data = sentiment_service.get_service_status()
        
        return SentimentResponse(
            success=True,
            data=status_data,
            message="Sentiment service status retrieved successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving service status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve service status: {str(e)}"
        )


@router.get("/models", response_model=SentimentResponse)
async def get_available_models(
    sentiment_service: SentimentService = Depends(get_sentiment_service)
):
    """
    Get information about available sentiment analysis models
    
    Returns:
    - Available model configurations
    - Current active model
    - Model capabilities and descriptions
    """
    try:
        if not sentiment_service.engine:
            raise HTTPException(
                status_code=503,
                detail="Sentiment engine not initialized"
            )
        
        model_info = sentiment_service.engine.get_model_info()
        
        return SentimentResponse(
            success=True,
            data=model_info,
            message="Model information retrieved successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving model information: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model information: {str(e)}"
        )


@router.post("/models/switch", response_model=SentimentResponse)
async def switch_sentiment_model(
    model_key: str = Query(..., description="Model to switch to (financial, general, news)"),
    sentiment_service: SentimentService = Depends(get_sentiment_service)
):
    """
    Switch the active sentiment analysis model
    
    Allows switching between different specialized models:
    - financial: FinBERT for financial text
    - general: General sentiment analysis
    - news: Financial news sentiment analysis
    """
    try:
        if not sentiment_service.engine:
            raise HTTPException(
                status_code=503,
                detail="Sentiment engine not initialized"
            )
        
        success = sentiment_service.engine.set_current_model(model_key)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_key}' not available"
            )
        
        return SentimentResponse(
            success=True,
            data={
                "previous_model": sentiment_service.engine.current_model,
                "new_model": model_key
            },
            message=f"Successfully switched to model: {model_key}",
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch model: {str(e)}"
        )


@router.get("/health")
async def sentiment_health_check():
    """
    Health check endpoint for sentiment analysis service
    """
    try:
        sentiment_service = await get_sentiment_service()
        
        if not sentiment_service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Sentiment service not initialized"
            )
        
        return {
            "status": "healthy",
            "service": "sentiment_analysis",
            "timestamp": datetime.now().isoformat(),
            "initialized": sentiment_service.is_initialized,
            "queue_size": sentiment_service.processing_queue.qsize()
        }
        
    except Exception as e:
        logger.error(f"Sentiment health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Sentiment service unhealthy: {str(e)}"
        ) 