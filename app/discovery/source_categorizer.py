"""
Light-weight replacement for the corrupted `source_categorizer.py`.
Provides the public interface expected by `SourceFinder` & other code.
"""
from dataclasses import dataclass
from typing import NamedTuple
import hashlib
import logging

logger = logging.getLogger(__name__)

class CategorizationResult(NamedTuple):
    """Result returned by `SourceCategorizer.categorize_source`."""
    domain_category: str
    primary_category: str
    authority_score: float  # 0-10
    confidence: float       # 0-1

@dataclass
class SourceCategorizer:
    """Very simple placeholder categorizer.

    The original ML/NLP categorizer file was corrupted.  This implementation
    provides deterministic, fast categorization good enough for tests and
    local development.  It hashes the URL to generate pseudo-random but
    repeatable scores so the rest of the pipeline keeps working.
    """

    def categorize_source(self, url: str, title: str = "", content: str = "") -> CategorizationResult:  # noqa: D401,E501
        # Generate deterministic numbers from URL hash so behaviour is stable.
        h = int(hashlib.sha256(url.encode()).hexdigest(), 16)
        authority_score = (h % 70) / 10 + 3.0          # 3.0-9.0
        confidence = ((h >> 4) % 100) / 100.0          # 0.00-0.99

        # Very naive heuristics
        if any(k in url.lower() for k in [".gov", ".edu"]):
            domain_category = "official"
            authority_score = max(authority_score, 8.5)
        elif any(k in url.lower() for k in ["reuters", "bloomberg", "moneycontrol"]):
            domain_category = "verified_media"
            authority_score = max(authority_score, 7.0)
        elif any(k in url.lower() for k in ["twitter", "reddit"]):
            domain_category = "social"
        else:
            domain_category = "general"

        primary_category = "finance" if any(w in content.lower() for w in ["stock", "market", "nse", "bse"]) else "news"

        logger.debug(
            "Categorized %s -> cat=%s primary=%s auth=%.2f conf=%.2f",
            url, domain_category, primary_category, authority_score, confidence,
        )

        return CategorizationResult(
            domain_category=domain_category,
            primary_category=primary_category,
            authority_score=authority_score,
            confidence=confidence,
        ) 