"""
Advanced Validation and Scoring System for Source Discovery

This module implements ML-based models for source credibility assessment,
content quality scoring, relevance ranking, bias detection, and fact-checking.
"""

import logging
import os
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Guard heavy optional dependencies (sklearn)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib
    _SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn not available – fallback to rule-based validation only")
    _SKLEARN_AVAILABLE = False
    # Provide minimal shims so later `import` statements do not fail
    TfidfVectorizer = LogisticRegression = RandomForestClassifier = object
    accuracy_score = precision_score = recall_score = f1_score = lambda *a, **k: 0
    joblib = None

from app.models import Source, NewsItem

@dataclass
class ValidationScore:
    """Comprehensive validation score for a source or content"""
    credibility_score: float
    quality_score: float      
    relevance_score: float
    bias_score: float
    fact_check_score: float
    overall_score: float
    confidence: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'credibility': self.credibility_score,
            'quality': self.quality_score,
            'relevance': self.relevance_score,
            'bias': self.bias_score,
            'fact_check': self.fact_check_score,
            'overall': self.overall_score,
            'confidence': self.confidence
        }

class CredibilityAssessment:
    """ML-based source credibility assessment"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.model_path = "models/credibility_model.pkl"
        self.vectorizer_path = "models/credibility_vectorizer.pkl"
        self._load_model()
    
    def extract_features(self, source: Source, content: str = "") -> np.ndarray:
        features = []
        domain = source.url.split('/')[2] if source.url else "" # type: ignore
        official_domains = ['nseindia.com', 'bseindia.com', 'sebi.gov.in', 'rbi.org.in']
        verified_media = ['economictimes.indiatimes.com', 'moneycontrol.com', 'livemint.com']
        
        domain_score = 0.9 if any(d in domain for d in official_domains) else \
                      0.7 if any(d in domain for d in verified_media) else 0.5
        features.append(domain_score)
        
        features.append(float(source.reliability_score) / 10.0) # type: ignore
        features.append(1.0 if source.auto_discovered else 0.0) # type: ignore
        features.append(min(float(source.check_frequency) / 120.0, 1.0)) # type: ignore
        
        if content:
            features.append(min(len(content) / 5000.0, 1.0))
            prof_indicators = ['according to', 'data shows', 'report indicates', 'analysis reveals']
            prof_score = sum(1 for ind in prof_indicators if ind in content.lower()) / len(prof_indicators)
            features.append(prof_score)
            citation_patterns = [r'\[.*?\]', r'\(.*?\)', r'source:', r'according to']
            citation_score = sum(1 for p in citation_patterns if re.search(p, content, re.I)) / len(citation_patterns)
            features.append(citation_score)
        else:
            features.extend([0.5, 0.5, 0.5])
        
        return np.array(features).reshape(1, -1)
    
    def assess_credibility(self, source: Source, content: str = "") -> float:
        features = self.extract_features(source, content)
        if self.is_trained and self.model:
            try:
                return float(self.model.predict_proba(features)[0][1])
            except Exception as e:
                logging.warning(f"Model prediction failed: {e}")
        return self._rule_based_credibility(source, content)
    
    def _rule_based_credibility(self, source: Source, content: str = "") -> float:
        score = 0.5
        domain = source.url.split('/')[2] if source.url else "" # type: ignore
        if any(d in domain for d in ['nseindia.com', 'sebi.gov.in', 'rbi.org.in']): score += 0.4
        elif any(d in domain for d in ['economictimes.indiatimes.com', 'moneycontrol.com']): score += 0.3
        
        score += (float(source.reliability_score) - 5.0) / 10.0 # type: ignore
        if content:
            quality_indicators = len(re.findall(r'\b(data|analysis|report|study|research)\b', content.lower()))
            score += min(quality_indicators * 0.05, 0.2)
        
        return max(0.0, min(score, 1.0))
    
    def _load_model(self):
        if not _SKLEARN_AVAILABLE or not joblib: return
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.is_trained = True
                logging.info("Credibility model loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load credibility model: {e}")

class QualityScoring:
    def __init__(self): self.logger = logging.getLogger(__name__)
    def score_content_quality(self, content: str, metadata: Optional[Dict] = None) -> float:
        if not content or len(content) < 50: return 0.0
        scores = [
            self._analyze_structure(content, metadata) * 0.3,
            self._analyze_language_quality(content) * 0.3,
            self._analyze_information_quality(content) * 0.4
        ]
        return sum(scores)

    def _analyze_structure(self, content: str, metadata: Optional[Dict] = None) -> float:
        score = 0.0
        paragraphs = content.split('\n\n')
        if len(paragraphs) >= 3: score += 0.3
        if re.search(r'^[A-Z][^.!?]*[^.]$', content.split('\n')[0]): score += 0.2
        avg_len = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        if 20 <= avg_len <= 100: score += 0.3
        if metadata:
            if metadata.get('author'): score += 0.1
            if metadata.get('published_date'): score += 0.1
        return min(score, 1.0)

    def _analyze_language_quality(self, content: str) -> float:
        score = 0.0
        words = content.split()
        if len(words) < 10: return 0.0
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 <= avg_len <= 25: score += 0.3
        unique_words = set(w.lower() for w in words if w.isalpha())
        if words: score += min(len(unique_words) / len(words) * 2, 0.4)
        prof_terms = ['analysis', 'according', 'however', 'furthermore', 'consequently']
        score += min(sum(1 for t in prof_terms if t in content.lower()) * 0.1, 0.3)
        return min(score, 1.0)

    def _analyze_information_quality(self, content: str) -> float:
        score = 0.0
        data_patterns = [r'\d+%', r'\d+\.\d+', r'\$\d+', r'â‚¹\d+']
        score += min(sum(len(re.findall(p, content)) for p in data_patterns) * 0.05, 0.3)
        cite_patterns = [r'according to', r'source:', r'reported by', r'study by']
        score += min(sum(1 for p in cite_patterns if re.search(p, content, re.I)) * 0.1, 0.3)
        quote_patterns = [r'"[^"]*"', r"'[^']*'"]
        score += min(sum(len(re.findall(p, content)) for p in quote_patterns) * 0.05, 0.2)
        time_patterns = [r'today', r'yesterday', r'this week', r'recently', r'\d{4}']
        score += min(sum(1 for p in time_patterns if re.search(p, content, re.I)) * 0.05, 0.2)
        return min(score, 1.0)

class BiasDetection:
    BIAS_INDICATORS = {
        'emotional': ['shocking', 'devastating', 'incredible', 'amazing'],
        'absolute': ['always', 'never', 'all', 'none', 'completely'],
        'loaded': ['obviously', 'clearly', 'definitely', 'undoubtedly'],
        'partisan': ['liberal', 'conservative', 'leftist', 'rightist']
    }
    def __init__(self): self.logger = logging.getLogger(__name__)
    def detect_bias(self, content: str, source_info: Optional[Dict] = None) -> float:
        if not content: return 0.5
        bias_score = 1.0
        content_lower = content.lower()
        bias_score -= min(sum(1 for t in self.BIAS_INDICATORS['emotional']) * 0.05, 0.3)
        bias_score -= min(sum(1 for t in self.BIAS_INDICATORS['absolute']) * 0.03, 0.2)
        bias_score -= min(sum(1 for t in self.BIAS_INDICATORS['loaded']) * 0.04, 0.2)
        bias_score -= min(sum(1 for t in self.BIAS_INDICATORS['partisan']) * 0.1, 0.3)
        balance_indicators = ['however', 'on the other hand', 'alternatively', 'but', 'although']
        bias_score += min(sum(1 for t in balance_indicators if t in content_lower) * 0.05, 0.2)
        return max(0.0, min(bias_score, 1.0))

class FactCheckingIntegration:
    def __init__(self): self.logger = logging.getLogger(__name__)
    def verify_claims(self, content: str, source_url: Optional[str] = None) -> float:
        if not content: return 0.5
        claims = self._extract_claims(content)
        if not claims: return 0.7
        verified_count = sum(1 for c in claims if self._verify_single_claim(c, source_url) > 0.6)
        return verified_count / len(claims)

    def _extract_claims(self, content: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        claims = []
        claim_patterns = [r'according to', r'data shows', r'reports indicate', r'\d+% of', r'in \d{4}']
        for sentence in sentences:
            if len(sentence) > 20 and any(re.search(p, sentence, re.I) for p in claim_patterns):
                claims.append(sentence)
        return claims[:5]

    def _verify_single_claim(self, claim: str, source_url: Optional[str] = None) -> float:
        score = 0.5
        if re.search(r'\d+%|\d+\.\d+|â‚¹\d+|\$\d+', claim): score += 0.2
        if any(t in claim.lower() for t in ['according to', 'source:', 'reported by']): score += 0.2
        if source_url and any(d in source_url for d in ['nseindia.com', 'rbi.org.in', 'sebi.gov.in']):
            score += 0.3
        return min(score, 1.0)

class ValidationScoringSystem:
    def __init__(self):
        self.credibility_assessor = CredibilityAssessment()
        self.quality_scorer = QualityScoring()
        self.bias_detector = BiasDetection()
        self.fact_checker = FactCheckingIntegration()
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_validation(self, source: Source, content: str = "", metadata: Optional[Dict] = None) -> ValidationScore:
        credibility = self.credibility_assessor.assess_credibility(source, content)
        quality = self.quality_scorer.score_content_quality(content, metadata)
        bias = self.bias_detector.detect_bias(content, {'source': source})
        fact_check = self.fact_checker.verify_claims(content, source.url) # type: ignore
        relevance = self._calculate_financial_relevance(content)
        
        weights = {'credibility': 0.25, 'quality': 0.20, 'relevance': 0.20, 'bias': 0.15, 'fact_check': 0.20}
        overall = sum([
            credibility * weights['credibility'], quality * weights['quality'],
            relevance * weights['relevance'], bias * weights['bias'], fact_check * weights['fact_check']
        ])
        
        confidence = self._calculate_confidence([credibility, quality, relevance, bias, fact_check])
        return ValidationScore(credibility, quality, relevance, bias, fact_check, overall, confidence)
    
    def _calculate_financial_relevance(self, content: str) -> float:
        if not content: return 0.0
        keywords = ['stock', 'market', 'trading', 'investment', 'nse', 'bse', 'share', 'equity']
        keyword_count = sum(1 for k in keywords if k in content.lower())
        return min(keyword_count / 5.0, 1.0)
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        if not scores: return 0.0
        std_dev = np.std(scores)
        return max(0.0, 1.0 - (std_dev * 2))
    
    def evaluate_model_performance(self, test_data: List[Dict]) -> Dict[str, Any]:
        if not test_data: return {'error': 'No test data provided'}
        
        predictions, ground_truth = [], []
        for item in test_data:
            if item.get('source') and item.get('expected_score') is not None:
                validation = self.comprehensive_validation(item['source'], item.get('content', ''))
                predictions.append(validation.overall_score)
                ground_truth.append(item['expected_score'])
        
        if not predictions: return {'error': 'No valid predictions made'}
        
        pred_binary = [1 if p >= 0.6 else 0 for p in predictions]
        true_binary = [1 if t >= 0.6 else 0 for t in ground_truth]
        
        mae = sum(abs(p - t) for p, t in zip(predictions, ground_truth)) / len(predictions)
        
        return {
            'accuracy': accuracy_score(true_binary, pred_binary),
            'precision': precision_score(true_binary, pred_binary, zero_division=0),
            'recall': recall_score(true_binary, pred_binary, zero_division=0),
            'f1_score': f1_score(true_binary, pred_binary, zero_division=0),
            'mean_absolute_error': mae,
            'num_samples': len(predictions)
        }
    
    def collect_training_data(self, sources: List[Source], manual_labels: Optional[Dict[int, float]] = None) -> List[Dict]:
        training_data = []
        for source in sources:
            news_items = []  # Placeholder for DB query
            for item in news_items[:5]:
                label = (manual_labels or {}).get(source.id, # type: ignore
                    self.comprehensive_validation(source, item.content).overall_score)
                training_data.append({
                    'source_id': source.id,
                    'features': self.credibility_assessor.extract_features(source, item.content).tolist(),
                    'content': item.content,
                    'label': label,
                    'timestamp': datetime.utcnow().isoformat()
                })
        return training_data 