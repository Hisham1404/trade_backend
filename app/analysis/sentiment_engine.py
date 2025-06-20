"""
Sentiment Analysis Engine
Advanced sentiment analysis for financial text using Hugging Face Transformers
Supports multiple models including FinBERT for financial-specific sentiment analysis
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import os

import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Pipeline
)
import numpy as np

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Standardized sentiment labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    BEARISH = "bearish"


class SentimentIntensity(Enum):
    """Sentiment intensity levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class SentimentResult:
    """Structured sentiment analysis result"""
    text: str
    label: str
    score: float
    confidence: float
    intensity: str
    normalized_score: float  # -1 to 1 scale
    processing_time: float
    model_used: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class BatchSentimentResult:
    """Results for batch sentiment analysis"""
    results: List[SentimentResult]
    total_texts: int
    processing_time: float
    average_score: float
    sentiment_distribution: Dict[str, int]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class TextPreprocessor:
    """Advanced text preprocessing for financial content"""
    
    def __init__(self):
        # Financial keywords and patterns
        self.financial_patterns = {
            'currency': r'\$[0-9,]+(?:\.[0-9]{2})?',
            'percentage': r'[+-]?[0-9]+(?:\.[0-9]+)?%',
            'ticker': r'\b[A-Z]{1,5}\b',
            'market_terms': [
                'bull', 'bear', 'rally', 'crash', 'surge', 'plunge',
                'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline'
            ]
        }
        
        # Noise patterns to clean
        self.noise_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
            r'@[A-Za-z0-9_]+',  # Mentions
            r'#[A-Za-z0-9_]+',  # Hashtags (optional to keep)
            r'\[.*?\]',  # Bracketed content
            r'\{.*?\}',  # Braced content
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for sentiment analysis"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove URLs, mentions, etc.
        for pattern in self.noise_patterns:
            text = re.sub(pattern, ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text.strip()
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text"""
        entities = {
            'currencies': re.findall(self.financial_patterns['currency'], text),
            'percentages': re.findall(self.financial_patterns['percentage'], text),
            'tickers': re.findall(self.financial_patterns['ticker'], text),
            'market_terms': []
        }
        
        # Find market terms
        text_lower = text.lower()
        for term in self.financial_patterns['market_terms']:
            if term in text_lower:
                entities['market_terms'].append(term)
        
        return entities
    
    def is_financial_content(self, text: str) -> bool:
        """Determine if text contains financial content"""
        entities = self.extract_financial_entities(text)
        return any(entities.values())


class SentimentEngine:
    """Advanced sentiment analysis engine with multiple model support"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.current_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Sentiment engine initialized on device: {self.device}")
        
        # Model configurations
        self.model_configs = {
            'general': {
                'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'description': 'General sentiment analysis model'
            },
            'financial': {
                'model_name': 'ProsusAI/finbert',
                'description': 'FinBERT - Financial sentiment analysis'
            },
            'news': {
                'model_name': 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                'description': 'Financial news sentiment analysis'
            }
        }
        
        # Sentiment thresholds
        self.thresholds = {
            'very_low': 0.1,
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8,
            'very_high': 0.95
        }
    
    async def initialize_models(self, models: List[str] = None) -> bool:
        """Initialize sentiment analysis models"""
        if models is None:
            models = ['general', 'financial']
        
        try:
            for model_key in models:
                if model_key not in self.model_configs:
                    logger.warning(f"Unknown model key: {model_key}")
                    continue
                
                config = self.model_configs[model_key]
                logger.info(f"Loading {config['description']}...")
                
                # Try to load the model
                try:
                    self.models[model_key] = pipeline(
                        "sentiment-analysis",
                        model=config['model_name'],
                        device=0 if self.device.type == "cuda" else -1,
                        return_all_scores=True
                    )
                    logger.info(f"✅ Successfully loaded {model_key} model")
                
                except Exception as e:
                    logger.warning(f"Failed to load {model_key} model: {str(e)}")
                    # Fallback to default model
                    if model_key == 'general':
                        self.models[model_key] = pipeline(
                            "sentiment-analysis",
                            device=0 if self.device.type == "cuda" else -1,
                            return_all_scores=True
                        )
                        logger.info(f"✅ Loaded default model for {model_key}")
            
            # Set current model to financial if available, otherwise general
            if 'financial' in self.models:
                self.current_model = 'financial'
            elif 'general' in self.models:
                self.current_model = 'general'
            else:
                logger.error("No models successfully loaded")
                return False
            
            logger.info(f"Sentiment engine ready with {len(self.models)} models")
            logger.info(f"Current model: {self.current_model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment models: {str(e)}")
            return False
    
    def _classify_intensity(self, score: float) -> str:
        """Classify sentiment intensity based on confidence score"""
        if score >= self.thresholds['very_high']:
            return SentimentIntensity.VERY_HIGH.value
        elif score >= self.thresholds['high']:
            return SentimentIntensity.HIGH.value
        elif score >= self.thresholds['moderate']:
            return SentimentIntensity.MODERATE.value
        elif score >= self.thresholds['low']:
            return SentimentIntensity.LOW.value
        else:
            return SentimentIntensity.VERY_LOW.value
    
    def _normalize_score(self, label: str, score: float) -> float:
        """Normalize sentiment score to -1 (negative) to 1 (positive) scale"""
        if label.lower() in ['positive', 'bullish']:
            return score
        elif label.lower() in ['negative', 'bearish']:
            return -score
        else:  # neutral
            return 0.0
    
    def _standardize_label(self, label: str) -> str:
        """Standardize sentiment labels across different models"""
        label_lower = label.lower()
        
        # Map various labels to standard labels
        if label_lower in ['positive', 'pos', 'bullish', 'buy']:
            return SentimentLabel.POSITIVE.value
        elif label_lower in ['negative', 'neg', 'bearish', 'sell']:
            return SentimentLabel.NEGATIVE.value
        elif label_lower in ['neutral', 'neu', 'hold']:
            return SentimentLabel.NEUTRAL.value
        else:
            return label_lower
    
    async def analyze_sentiment(
        self, 
        text: str, 
        model_key: str = None,
        preprocess: bool = True
    ) -> SentimentResult:
        """Analyze sentiment of a single text"""
        start_time = datetime.now()
        
        # Choose model
        model_key = model_key or self.current_model
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not available")
        
        # Preprocess text
        original_text = text
        if preprocess:
            text = self.preprocessor.clean_text(text)
        
        if not text:
            return SentimentResult(
                text=original_text,
                label=SentimentLabel.NEUTRAL.value,
                score=0.0,
                confidence=0.0,
                intensity=SentimentIntensity.VERY_LOW.value,
                normalized_score=0.0,
                processing_time=0.0,
                model_used=model_key,
                timestamp=datetime.now().isoformat()
            )
        
        try:
            # Run sentiment analysis
            model = self.models[model_key]
            results = model(text)
            
            # Extract best result
            if isinstance(results[0], list):
                # Model returns all scores
                best_result = max(results[0], key=lambda x: x['score'])
            else:
                # Model returns single result
                best_result = results[0]
            
            # Standardize results
            label = self._standardize_label(best_result['label'])
            score = float(best_result['score'])
            intensity = self._classify_intensity(score)
            normalized_score = self._normalize_score(label, score)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SentimentResult(
                text=original_text,
                label=label,
                score=score,
                confidence=score,
                intensity=intensity,
                normalized_score=normalized_score,
                processing_time=processing_time,
                model_used=model_key,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Return neutral result on error
            return SentimentResult(
                text=original_text,
                label=SentimentLabel.NEUTRAL.value,
                score=0.0,
                confidence=0.0,
                intensity=SentimentIntensity.VERY_LOW.value,
                normalized_score=0.0,
                processing_time=processing_time,
                model_used=model_key,
                timestamp=datetime.now().isoformat()
            )
    
    async def analyze_batch(
        self, 
        texts: List[str], 
        model_key: str = None,
        preprocess: bool = True
    ) -> BatchSentimentResult:
        """Analyze sentiment for multiple texts"""
        start_time = datetime.now()
        
        if not texts:
            return BatchSentimentResult(
                results=[],
                total_texts=0,
                processing_time=0.0,
                average_score=0.0,
                sentiment_distribution={},
                timestamp=datetime.now().isoformat()
            )
        
        # Analyze each text
        results = []
        for text in texts:
            result = await self.analyze_sentiment(text, model_key, preprocess)
            results.append(result)
        
        # Calculate statistics
        total_texts = len(results)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        scores = [r.normalized_score for r in results]
        average_score = np.mean(scores) if scores else 0.0
        
        # Count sentiment distribution
        sentiment_distribution = {}
        for result in results:
            label = result.label
            sentiment_distribution[label] = sentiment_distribution.get(label, 0) + 1
        
        return BatchSentimentResult(
            results=results,
            total_texts=total_texts,
            processing_time=processing_time,
            average_score=float(average_score),
            sentiment_distribution=sentiment_distribution,
            timestamp=datetime.now().isoformat()
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'available_models': list(self.models.keys()),
            'current_model': self.current_model,
            'model_configs': self.model_configs,
            'device': str(self.device),
            'thresholds': self.thresholds
        }
    
    def set_current_model(self, model_key: str) -> bool:
        """Set the current model for sentiment analysis"""
        if model_key in self.models:
            self.current_model = model_key
            logger.info(f"Current model set to: {model_key}")
            return True
        else:
            logger.error(f"Model {model_key} not available")
            return False


# Global sentiment engine instance
_sentiment_engine: Optional[SentimentEngine] = None


async def get_sentiment_engine() -> SentimentEngine:
    """Get or create the global sentiment engine instance"""
    global _sentiment_engine
    
    if _sentiment_engine is None:
        _sentiment_engine = SentimentEngine()
        success = await _sentiment_engine.initialize_models()
        if not success:
            logger.error("Failed to initialize sentiment engine")
            raise RuntimeError("Sentiment engine initialization failed")
    
    return _sentiment_engine


def shutdown_sentiment_engine():
    """Shutdown the global sentiment engine"""
    global _sentiment_engine
    _sentiment_engine = None 