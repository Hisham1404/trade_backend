"""
Sentiment Analysis Service
Service layer for real-time sentiment analysis integration with trading system
Provides sentiment tracking, analysis aggregation, and integration with scrapers
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from ..analysis.sentiment_engine import (
    get_sentiment_engine, 
    SentimentEngine, 
    SentimentResult, 
    BatchSentimentResult,
    SentimentLabel
)
from ..models.asset import Asset
from ..models.alert import Alert

logger = logging.getLogger(__name__)


class SentimentTracker:
    """Real-time sentiment tracking for assets and market"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.sentiment_history = defaultdict(lambda: deque(maxlen=max_history))
        self.asset_sentiment = defaultdict(dict)
        self.market_sentiment = deque(maxlen=max_history)
        
    def add_sentiment(self, asset_symbol: str, sentiment: SentimentResult):
        """Add sentiment result for an asset"""
        self.sentiment_history[asset_symbol].append(sentiment)
        self.asset_sentiment[asset_symbol] = {
            'latest_sentiment': sentiment.to_dict(),
            'last_updated': sentiment.timestamp,
            'sentiment_trend': self._calculate_trend(asset_symbol)
        }
        
        # Update market sentiment (using normalized scores)
        self.market_sentiment.append({
            'timestamp': sentiment.timestamp,
            'score': sentiment.normalized_score,
            'asset': asset_symbol
        })
    
    def _calculate_trend(self, asset_symbol: str) -> Dict[str, Any]:
        """Calculate sentiment trend for an asset"""
        history = list(self.sentiment_history[asset_symbol])
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'change': 0.0}
        
        # Calculate recent trend (last 10 data points)
        recent_scores = [h.normalized_score for h in history[-10:]]
        if len(recent_scores) >= 2:
            trend_change = recent_scores[-1] - recent_scores[0]
            if trend_change > 0.1:
                trend = 'improving'
            elif trend_change < -0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
            trend_change = 0.0
        
        return {
            'trend': trend,
            'change': trend_change,
            'data_points': len(recent_scores)
        }
    
    def get_asset_sentiment(self, asset_symbol: str) -> Optional[Dict[str, Any]]:
        """Get current sentiment data for an asset"""
        return self.asset_sentiment.get(asset_symbol)
    
    def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """Get overall market sentiment summary"""
        if not self.market_sentiment:
            return {
                'overall_sentiment': 'neutral',
                'average_score': 0.0,
                'sentiment_distribution': {},
                'total_data_points': 0,
                'last_updated': None
            }
        
        # Calculate overall metrics
        scores = [s['score'] for s in self.market_sentiment]
        average_score = sum(scores) / len(scores)
        
        # Determine overall sentiment
        if average_score > 0.1:
            overall_sentiment = 'positive'
        elif average_score < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Count distribution (simplified)
        positive_count = sum(1 for s in scores if s > 0.1)
        negative_count = sum(1 for s in scores if s < -0.1)
        neutral_count = len(scores) - positive_count - negative_count
        
        sentiment_distribution = {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count
        }
        
        return {
            'overall_sentiment': overall_sentiment,
            'average_score': average_score,
            'sentiment_distribution': sentiment_distribution,
            'total_data_points': len(scores),
            'last_updated': self.market_sentiment[-1]['timestamp'] if self.market_sentiment else None
        }


class SentimentService:
    """Main sentiment analysis service"""
    
    def __init__(self):
        self.engine: Optional[SentimentEngine] = None
        self.tracker = SentimentTracker()
        self.is_initialized = False
        self.processing_queue = asyncio.Queue()
        self.is_processing = False
        
        # Configuration
        self.batch_size = 10
        self.processing_interval = 5  # seconds
        
    async def initialize(self) -> bool:
        """Initialize the sentiment service"""
        try:
            self.engine = await get_sentiment_engine()
            self.is_initialized = True
            logger.info("Sentiment service initialized successfully")
            
            # Start background processing
            asyncio.create_task(self._process_sentiment_queue())
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize sentiment service: {str(e)}")
            return False
    
    async def analyze_text(
        self, 
        text: str, 
        asset_symbol: str = None,
        source: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> SentimentResult:
        """Analyze sentiment of a single text"""
        if not self.is_initialized:
            raise RuntimeError("Sentiment service not initialized")
        
        try:
            # Choose appropriate model based on content
            model_key = 'financial' if self.engine.preprocessor.is_financial_content(text) else 'general'
            
            # Analyze sentiment
            result = await self.engine.analyze_sentiment(text, model_key=model_key)
            
            # Add metadata
            if metadata:
                result.metadata = metadata
            result.source = source
            
            # Update tracking if asset is specified
            if asset_symbol:
                self.tracker.add_sentiment(asset_symbol, result)
                
                # Check for significant sentiment events
                await self._check_sentiment_alerts(asset_symbol, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            raise
    
    async def analyze_batch(
        self, 
        texts: List[str], 
        asset_symbols: List[str] = None,
        source: str = "batch",
        metadata: Dict[str, Any] = None
    ) -> BatchSentimentResult:
        """Analyze sentiment for multiple texts"""
        if not self.is_initialized:
            raise RuntimeError("Sentiment service not initialized")
        
        try:
            # Determine if content is financial
            model_key = 'financial' if any(
                self.engine.preprocessor.is_financial_content(text) for text in texts
            ) else 'general'
            
            # Analyze batch
            batch_result = await self.engine.analyze_batch(texts, model_key=model_key)
            
            # Update tracking for individual results
            if asset_symbols:
                for i, result in enumerate(batch_result.results):
                    if i < len(asset_symbols) and asset_symbols[i]:
                        result.source = source
                        if metadata:
                            result.metadata = metadata
                        self.tracker.add_sentiment(asset_symbols[i], result)
                        await self._check_sentiment_alerts(asset_symbols[i], result)
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Error analyzing batch sentiment: {str(e)}")
            raise
    
    async def queue_for_analysis(
        self, 
        text: str, 
        asset_symbol: str = None,
        source: str = "queue",
        priority: int = 1
    ):
        """Queue text for background sentiment analysis"""
        await self.processing_queue.put({
            'text': text,
            'asset_symbol': asset_symbol,
            'source': source,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _process_sentiment_queue(self):
        """Background task to process queued sentiment analysis"""
        logger.info("Started sentiment processing queue")
        
        while True:
            try:
                if self.processing_queue.empty():
                    await asyncio.sleep(self.processing_interval)
                    continue
                
                # Collect batch of items
                batch = []
                while len(batch) < self.batch_size and not self.processing_queue.empty():
                    try:
                        item = await asyncio.wait_for(
                            self.processing_queue.get(), 
                            timeout=0.1
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._process_batch(batch)
                
                await asyncio.sleep(1)  # Brief pause between batches
                
            except Exception as e:
                logger.error(f"Error in sentiment processing queue: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of queued sentiment analysis"""
        try:
            texts = [item['text'] for item in batch]
            asset_symbols = [item.get('asset_symbol') for item in batch]
            
            # Analyze batch
            result = await self.analyze_batch(
                texts, 
                asset_symbols, 
                source="background_queue"
            )
            
            logger.debug(f"Processed sentiment batch of {len(batch)} items")
            
        except Exception as e:
            logger.error(f"Error processing sentiment batch: {str(e)}")
    
    async def _check_sentiment_alerts(self, asset_symbol: str, sentiment: SentimentResult):
        """Check for sentiment-based alerts"""
        try:
            # Define alert thresholds
            extreme_positive_threshold = 0.8
            extreme_negative_threshold = -0.8
            
            # Check for extreme sentiment
            if sentiment.normalized_score >= extreme_positive_threshold:
                await self._create_sentiment_alert(
                    asset_symbol, 
                    "Extremely Positive Sentiment",
                    f"Very positive sentiment detected for {asset_symbol}: {sentiment.normalized_score:.2f}",
                    "high"
                )
            elif sentiment.normalized_score <= extreme_negative_threshold:
                await self._create_sentiment_alert(
                    asset_symbol,
                    "Extremely Negative Sentiment", 
                    f"Very negative sentiment detected for {asset_symbol}: {sentiment.normalized_score:.2f}",
                    "high"
                )
            
            # Check sentiment trend changes
            asset_data = self.tracker.get_asset_sentiment(asset_symbol)
            if asset_data and 'sentiment_trend' in asset_data:
                trend = asset_data['sentiment_trend']
                if abs(trend.get('change', 0)) > 0.5:  # Significant trend change
                    await self._create_sentiment_alert(
                        asset_symbol,
                        "Sentiment Trend Change",
                        f"Significant sentiment trend change for {asset_symbol}: {trend['trend']}",
                        "medium"
                    )
                    
        except Exception as e:
            logger.error(f"Error checking sentiment alerts: {str(e)}")
    
    async def _create_sentiment_alert(
        self, 
        asset_symbol: str, 
        title: str, 
        message: str, 
        priority: str
    ):
        """Create a sentiment-based alert"""
        try:
            # This would integrate with your existing alert system
            alert_data = {
                'type': 'sentiment',
                'asset_symbol': asset_symbol,
                'title': title,
                'message': message,
                'priority': priority,
                'timestamp': datetime.now().isoformat(),
                'source': 'sentiment_service'
            }
            
            logger.info(f"Sentiment alert: {title} for {asset_symbol}")
            # Here you would create the actual Alert model instance
            # and save it to the database
            
        except Exception as e:
            logger.error(f"Error creating sentiment alert: {str(e)}")
    
    def get_asset_sentiment(self, asset_symbol: str) -> Optional[Dict[str, Any]]:
        """Get current sentiment data for an asset"""
        return self.tracker.get_asset_sentiment(asset_symbol)
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment summary"""
        return self.tracker.get_market_sentiment_summary()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get sentiment service status"""
        engine_info = self.engine.get_model_info() if self.engine else {}
        
        return {
            'initialized': self.is_initialized,
            'engine_info': engine_info,
            'queue_size': self.processing_queue.qsize(),
            'is_processing': self.is_processing,
            'tracked_assets': len(self.tracker.asset_sentiment),
            'market_data_points': len(self.tracker.market_sentiment)
        }
    
    async def analyze_news_impact(
        self, 
        news_articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment impact of news articles"""
        try:
            if not news_articles:
                return {
                    'total_articles': 0,
                    'sentiment_summary': {},
                    'asset_impact': {},
                    'overall_impact': 'neutral'
                }
            
            texts = []
            asset_mentions = defaultdict(list)
            
            # Extract text and identify asset mentions
            for article in news_articles:
                text = article.get('content', '') or article.get('title', '')
                if text:
                    texts.append(text)
                    
                    # Look for asset mentions (basic implementation)
                    # This would be enhanced with NER or asset symbol detection
                    for asset in ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'GOOGL']:  # Example assets
                        if asset.lower() in text.lower():
                            asset_mentions[asset].append(len(texts) - 1)  # Index of the text
            
            # Analyze sentiment
            batch_result = await self.analyze_batch(texts, source="news_analysis")
            
            # Calculate asset-specific impact
            asset_impact = {}
            for asset, indices in asset_mentions.items():
                asset_sentiments = [batch_result.results[i] for i in indices]
                if asset_sentiments:
                    avg_score = sum(s.normalized_score for s in asset_sentiments) / len(asset_sentiments)
                    asset_impact[asset] = {
                        'average_sentiment': avg_score,
                        'article_count': len(asset_sentiments),
                        'sentiment_distribution': {
                            'positive': sum(1 for s in asset_sentiments if s.normalized_score > 0.1),
                            'negative': sum(1 for s in asset_sentiments if s.normalized_score < -0.1),
                            'neutral': sum(1 for s in asset_sentiments if -0.1 <= s.normalized_score <= 0.1)
                        }
                    }
            
            # Determine overall impact
            overall_score = batch_result.average_score
            if overall_score > 0.2:
                overall_impact = 'positive'
            elif overall_score < -0.2:
                overall_impact = 'negative'
            else:
                overall_impact = 'neutral'
            
            return {
                'total_articles': len(news_articles),
                'sentiment_summary': batch_result.to_dict(),
                'asset_impact': asset_impact,
                'overall_impact': overall_impact,
                'processing_time': batch_result.processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news impact: {str(e)}")
            raise


# Global sentiment service instance
_sentiment_service: Optional[SentimentService] = None


async def get_sentiment_service() -> SentimentService:
    """Get or create the global sentiment service instance"""
    global _sentiment_service
    
    if _sentiment_service is None:
        _sentiment_service = SentimentService()
        await _sentiment_service.initialize()
    
    return _sentiment_service


async def shutdown_sentiment_service():
    """Shutdown the global sentiment service"""
    global _sentiment_service
    _sentiment_service = None 