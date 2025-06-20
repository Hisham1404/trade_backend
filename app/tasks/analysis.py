"""
Analysis Tasks for Background Processing
Handles news analysis, sentiment processing, and market correlation
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from app.tasks.celery_app import celery_app
from app.database.connection import get_db
from app.models.news import NewsItem
from app.services.sentiment_service import get_sentiment_service
from app.services.logging_service import get_logger

logger = get_logger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def analyze_news_item(self, news_item_id: int):
    """Analyze a single news item for sentiment and market impact"""
    try:
        db = next(get_db())
        logger.info(f"Starting analysis for news item ID: {news_item_id}")
        
        # Get the news item
        news_item = db.query(NewsItem).filter(NewsItem.id == news_item_id).first()
        
        if not news_item:
            logger.warning(f"News item ID {news_item_id} not found")
            return {
                'news_item_id': news_item_id,
                'status': 'not_found',
                'timestamp': datetime.now().isoformat()
            }
        
        # Skip if already analyzed
        if news_item.sentiment_score is not None:
            logger.info(f"News item {news_item_id} already analyzed")
            return {
                'news_item_id': news_item_id,
                'status': 'already_analyzed',
                'sentiment_score': news_item.sentiment_score,
                'timestamp': datetime.now().isoformat()
            }
        
        # Get sentiment service and analyze
        sentiment_service = get_sentiment_service()
        
        # Analyze the news item
        analysis_result = sentiment_service.analyze_text_sync(
            text=f"{news_item.title} {news_item.content or ''}",
            context={'source': news_item.source.name if news_item.source else None}
        )
        
        if analysis_result and 'sentiment_score' in analysis_result:
            # Update the news item with sentiment data
            news_item.sentiment_score = analysis_result['sentiment_score']
            news_item.sentiment_label = analysis_result.get('sentiment_label')
            news_item.confidence_score = analysis_result.get('confidence')
            news_item.analyzed_at = datetime.now()
            
            db.commit()
            db.close()
            
            result = {
                'news_item_id': news_item_id,
                'status': 'success',
                'sentiment_score': news_item.sentiment_score,
                'sentiment_label': news_item.sentiment_label,
                'confidence_score': news_item.confidence_score,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully analyzed news item {news_item_id}: sentiment={news_item.sentiment_score}")
            return result
        else:
            db.close()
            logger.error(f"Failed to get analysis result for news item {news_item_id}")
            return {
                'news_item_id': news_item_id,
                'status': 'analysis_failed',
                'error': 'No sentiment score returned',
                'timestamp': datetime.now().isoformat()
            }
        
    except Exception as exc:
        logger.error(f"Analysis task failed for news item {news_item_id}: {exc}")
        raise self.retry(exc=exc, countdown=min(30 * (2 ** self.request.retries), 300))

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def process_unanalyzed_news(self):
    """Process all unanalyzed news items from the last 24 hours"""
    try:
        db = next(get_db())
        logger.info("Starting batch analysis of unanalyzed news items")
        
        # Get unanalyzed news items from the last 24 hours
        cutoff_time = datetime.now() - timedelta(days=1)
        unanalyzed_items = db.query(NewsItem).filter(
            NewsItem.sentiment_score.is_(None),
            NewsItem.scraped_at >= cutoff_time
        ).all()
        
        if not unanalyzed_items:
            logger.info("No unanalyzed news items found")
            return {
                'timestamp': datetime.now().isoformat(),
                'unanalyzed_count': 0,
                'status': 'no_items'
            }
        
        # Queue individual analysis tasks
        queued_tasks = []
        for news_item in unanalyzed_items:
            task = analyze_news_item.delay(news_item.id)
            queued_tasks.append({
                'news_item_id': news_item.id,
                'task_id': task.id
            })
        
        db.close()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'unanalyzed_count': len(unanalyzed_items),
            'queued_tasks': len(queued_tasks),
            'status': 'tasks_queued'
        }
        
        logger.info(f"Queued {len(queued_tasks)} analysis tasks for unanalyzed news items")
        return result
        
    except Exception as exc:
        logger.error(f"Batch analysis task failed: {exc}")
        raise self.retry(exc=exc, countdown=min(60 * (2 ** self.request.retries), 300))

@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def run_sentiment_analysis(self, text: str, context: Optional[Dict[str, Any]] = None):
    """Run sentiment analysis on arbitrary text"""
    try:
        logger.info("Running sentiment analysis task")
        
        sentiment_service = get_sentiment_service()
        
        # Analyze the text
        analysis_result = sentiment_service.analyze_text_sync(text=text, context=context or {})
        
        if analysis_result:
            result = {
                'text_length': len(text),
                'sentiment_score': analysis_result.get('sentiment_score'),
                'sentiment_label': analysis_result.get('sentiment_label'),
                'confidence_score': analysis_result.get('confidence'),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"Sentiment analysis completed: {result['sentiment_label']} ({result['sentiment_score']})")
            return result
        else:
            logger.error("Failed to get sentiment analysis result")
            return {
                'text_length': len(text),
                'status': 'analysis_failed',
                'error': 'No analysis result returned',
                'timestamp': datetime.now().isoformat()
            }
        
    except Exception as exc:
        logger.error(f"Sentiment analysis task failed: {exc}")
        raise self.retry(exc=exc, countdown=min(120 * (2 ** self.request.retries), 600))

@celery_app.task(bind=True, max_retries=2, default_retry_delay=180)
def update_market_correlation(self):
    """Update market correlation analysis based on recent news sentiment"""
    try:
        logger.info("Starting market correlation update task")
        
        # Get recent analyzed news items (last 6 hours)
        cutoff_time = datetime.now() - timedelta(hours=6)
        db = next(get_db())
        
        recent_news = db.query(NewsItem).filter(
            NewsItem.sentiment_score.is_not(None),
            NewsItem.analyzed_at >= cutoff_time
        ).all()
        
        if not recent_news:
            logger.info("No recent analyzed news found for correlation update")
            return {
                'timestamp': datetime.now().isoformat(),
                'news_count': 0,
                'status': 'no_data'
            }
        
        # Calculate sentiment aggregates
        total_sentiment = sum(item.sentiment_score for item in recent_news)
        avg_sentiment = total_sentiment / len(recent_news)
        
        positive_count = sum(1 for item in recent_news if item.sentiment_score > 0.1)
        negative_count = sum(1 for item in recent_news if item.sentiment_score < -0.1)
        neutral_count = len(recent_news) - positive_count - negative_count
        
        db.close()
        
        correlation_data = {
            'timestamp': datetime.now().isoformat(),
            'news_count': len(recent_news),
            'average_sentiment': avg_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_distribution': {
                'positive_percentage': (positive_count / len(recent_news)) * 100,
                'negative_percentage': (negative_count / len(recent_news)) * 100,
                'neutral_percentage': (neutral_count / len(recent_news)) * 100
            },
            'status': 'completed'
        }
        
        logger.info(f"Market correlation updated: avg_sentiment={avg_sentiment:.3f}, news_count={len(recent_news)}")
        return correlation_data
        
    except Exception as exc:
        logger.error(f"Market correlation update task failed: {exc}")
        raise self.retry(exc=exc, countdown=min(180 * (2 ** self.request.retries), 900))
