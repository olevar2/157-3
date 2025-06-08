# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Sentiment Analyzer for Platform3
Provides sentiment analysis capabilities for financial markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

# Try to import advanced NLP components with fallbacks for basic functionality
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

from engines.indicator_base import IndicatorBase


class SentimentConfig:
    """Configuration class for sentiment analysis"""
    def __init__(self):
        self.api_key = None
        self.data_sources = ['twitter', 'reddit', 'news']
        self.update_interval = 60  # seconds
        self.sentiment_threshold = 0.5
        

class SentimentSource(Enum):
    """Sources of sentiment data"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ANALYST_REPORT = "analyst_report"
    EARNINGS_CALL = "earnings_call"
    FORUM = "forum"
    OTHER = "other"

class SentimentType(Enum):
    """Types of sentiment"""
    EXTREME_BEARISH = "extreme_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    BULLISH = "bullish"
    EXTREME_BULLISH = "extreme_bullish"

@dataclass
class SentimentData:
    """Container for sentiment analysis data"""
    text: str
    source: SentimentSource
    timestamp: datetime
    score: float = 0.0  # -1.0 (extremely negative) to 1.0 (extremely positive)
    polarity: float = 0.0  # -1.0 to 1.0
    subjectivity: float = 0.0  # 0.0 to 1.0 (higher means more subjective)
    magnitude: float = 0.0  # Overall strength of sentiment
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_sentiment_type(self) -> SentimentType:
        """Convert numerical score to sentiment type"""
        if self.score >= 0.7:
            return SentimentType.EXTREME_BULLISH
        elif self.score >= 0.2:
            return SentimentType.BULLISH
        elif self.score <= -0.7:
            return SentimentType.EXTREME_BEARISH
        elif self.score <= -0.2:
            return SentimentType.BEARISH
        elif abs(self.score) < 0.1:
            return SentimentType.NEUTRAL
        else:
            return SentimentType.MIXED
            
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'text': (self.text[:100] + '...') if len(self.text) > 100 else self.text,
            'source': self.source.value,
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            'score': self.score,
            'polarity': self.polarity,
            'subjectivity': self.subjectivity,
            'magnitude': self.magnitude,
            'sentiment_type': self.get_sentiment_type().value,
            'entities': self.entities[:10] if len(self.entities) > 10 else self.entities,
            'keywords': self.keywords[:10] if len(self.keywords) > 10 else self.keywords,
            'metadata': self.metadata
        }

@dataclass
class SentimentScore:
    """Sentiment score data structure"""
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    sentiment_type: SentimentType
    source: SentimentSource
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate score ranges"""
        self.score = max(-1.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish"""
        return self.score > 0.1
    
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish"""
        return self.score < -0.1
    
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral"""
        return abs(self.score) <= 0.1

class SentimentAnalyzer:
    """
    Platform3 Sentiment Analyzer
    
    Provides sentiment analysis for financial market text data using
    multiple algorithms and sources.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.logger = logging.getLogger('SentimentAnalyzer')
        self._initialize_analyzers()
        
    def _initialize_analyzers(self):
        """Initialize available sentiment analyzers"""
        self.analyzers = {}
        
        # Initialize NLTK Vader analyzer if available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                try:
                    nltk.download('vader_lexicon', quiet=True)
                except Exception as e:
                    self.logger.warning(f"Failed to download NLTK vader_lexicon: {str(e)}")
            
            try:
                self.analyzers['vader'] = SentimentIntensityAnalyzer()
                self.logger.info("NLTK Vader sentiment analyzer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NLTK Vader: {str(e)}")
        else:
            self.logger.info("NLTK not available, Vader sentiment analyzer disabled")
            
        # Initialize TextBlob if available
        if TEXTBLOB_AVAILABLE:
            self.analyzers['textblob'] = True
            self.logger.info("TextBlob sentiment analyzer initialized")
        else:
            self.logger.info("TextBlob not available, TextBlob sentiment analyzer disabled")
            
        # If no analyzers available, use basic fallback
        if not self.analyzers:
            self.logger.warning("No advanced sentiment analyzers available, using basic fallback")
        
    def analyze_text(self, text: str, source: SentimentSource, 
                    timestamp: Optional[datetime] = None) -> SentimentData:
        """
        Analyze sentiment of given text
        
        Args:
            text: Text to analyze
            source: Source of the text
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            SentimentData: Analysis result
        """
        result = SentimentData(
            text=text,
            source=source,
            timestamp=timestamp or datetime.now()
        )        
        # Run all available analyzers and combine results
        if 'vader' in self.analyzers:
            try:
                vader_scores = self.analyzers['vader'].polarity_scores(text)
                result.score = vader_scores['compound']
                result.polarity = vader_scores['compound']
                result.metadata['vader_scores'] = {
                    'neg': vader_scores['neg'],
                    'neu': vader_scores['neu'],
                    'pos': vader_scores['pos']
                }
                result.magnitude = abs(result.score)
            except Exception as e:
                self.logger.error(f"Vader sentiment analysis failed: {str(e)}")
                
        elif 'textblob' in self.analyzers:
            try:
                blob = TextBlob(text)
                result.polarity = blob.sentiment.polarity
                result.subjectivity = blob.sentiment.subjectivity
                result.score = result.polarity
                result.magnitude = abs(result.polarity)
                
                # Extract keywords (noun phrases)
                result.keywords = [str(phrase) for phrase in blob.noun_phrases]
            except Exception as e:
                self.logger.error(f"TextBlob sentiment analysis failed: {str(e)}")
                
        else:
            # Basic fallback using simple keyword matching
            result = self._basic_sentiment_analysis(text, result)
            
        return result
    
    def _basic_sentiment_analysis(self, text: str, result: SentimentData) -> SentimentData:
        """
        Basic fallback sentiment analysis using keyword matching
        
        Args:
            text: Text to analyze
            result: Existing result to update
            
        Returns:
            SentimentData: Updated result
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Simple dictionaries of positive and negative terms
        positive_terms = {
            'bull', 'bullish', 'buy', 'strong', 'growth', 'profit', 'gain', 'rise', 'up',
            'increase', 'positive', 'good', 'great', 'excellent', 'outperform', 'beat',
            'exceed', 'rally', 'surge', 'momentum', 'upgrade', 'target'
        }
        
        negative_terms = {
            'bear', 'bearish', 'sell', 'weak', 'decline', 'loss', 'fall', 'down',
            'decrease', 'negative', 'bad', 'poor', 'terrible', 'underperform', 'miss',
            'below', 'crash', 'plunge', 'downgrade', 'risk', 'concern'
        }
        
        # Count occurrences
        positive_count = sum(1 for term in positive_terms if term in text_lower)
        negative_count = sum(1 for term in negative_terms if term in text_lower)
        
        # Calculate basic sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            result.score = (positive_count - negative_count) / total_sentiment_words
        else:
            result.score = 0.0
            
        result.polarity = result.score
        result.magnitude = abs(result.score)
        result.subjectivity = min(1.0, total_sentiment_words / len(text_lower.split()) * 2)
        
        # Store basic metadata
        result.metadata['basic_analysis'] = {
            'positive_terms_found': positive_count,
            'negative_terms_found': negative_count,
            'total_sentiment_words': total_sentiment_words
        }
        
        return result

    def analyze_batch(self, texts: List[str], source: SentimentSource, 
                     timestamps: Optional[List[datetime]] = None) -> List[SentimentData]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            source: Source of the texts
            timestamps: Optional list of timestamps
            
        Returns:
            List[SentimentData]: Analysis results
        """
        results = []
        for i, text in enumerate(texts):
            timestamp = timestamps[i] if timestamps and i < len(timestamps) else None
            try:
                result = self.analyze_text(text, source, timestamp)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to analyze text {i}: {str(e)}")
                # Add empty result to maintain list consistency
                results.append(SentimentData(
                    text=text,
                    source=source,
                    timestamp=timestamp or datetime.now()
                ))
        return results

    def get_analyzer_status(self) -> Dict[str, bool]:
        """
        Get status of available analyzers
        
        Returns:
            Dict[str, bool]: Analyzer availability status
        """
        return {
            'vader': 'vader' in self.analyzers,
            'textblob': 'textblob' in self.analyzers,
            'basic_fallback': True
        }