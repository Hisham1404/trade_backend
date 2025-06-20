"""
Analysis Module
Advanced analysis capabilities for trading system including sentiment analysis
"""

from .sentiment_engine import (
    get_sentiment_engine,
    shutdown_sentiment_engine,
    SentimentEngine,
    SentimentResult,
    BatchSentimentResult,
    SentimentLabel,
    SentimentIntensity
)

__all__ = [
    'get_sentiment_engine',
    'shutdown_sentiment_engine',
    'SentimentEngine',
    'SentimentResult',
    'BatchSentimentResult',
    'SentimentLabel',
    'SentimentIntensity'
] 