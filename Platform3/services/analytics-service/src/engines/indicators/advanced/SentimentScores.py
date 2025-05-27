"""
Market Sentiment Scores Module

This module provides advanced market sentiment analysis for forex trading,
integrating multiple sentiment sources and providing real-time sentiment scoring.
Optimized for scalping (M1-M5), day trading (M15-H1), and swing trading (H4) strategies.

Features:
- Multi-source sentiment aggregation
- Real-time sentiment scoring
- Sentiment momentum analysis
- Market sentiment regime detection
- Session-based sentiment patterns
- Sentiment-price correlation analysis

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentLevel(Enum):
    """Sentiment level classification"""
    EXTREMELY_BEARISH = "extremely_bearish"
    BEARISH = "bearish"
    SLIGHTLY_BEARISH = "slightly_bearish"
    NEUTRAL = "neutral"
    SLIGHTLY_BULLISH = "slightly_bullish"
    BULLISH = "bullish"
    EXTREMELY_BULLISH = "extremely_bullish"

class SentimentSource(Enum):
    """Sentiment data sources"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ECONOMIC_CALENDAR = "economic_calendar"
    ANALYST_REPORTS = "analyst_reports"
    MARKET_DATA = "market_data"
    COT_REPORTS = "cot_reports"

@dataclass
class SentimentData:
    """Container for sentiment data point"""
    source: SentimentSource
    timestamp: datetime
    raw_score: float
    normalized_score: float
    confidence: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SentimentResults:
    """Container for sentiment analysis results"""
    overall_sentiment: float
    sentiment_level: SentimentLevel
    source_breakdown: Dict[str, float]
    momentum: float
    volatility: float
    confidence: float
    regime: str
    session_sentiment: Dict[str, float]
    correlation_with_price: float
    sentiment_divergence: float

class SentimentScores:
    """
    Advanced Market Sentiment Analysis
    
    Provides comprehensive sentiment scoring from multiple sources with
    real-time analysis and trading signal generation.
    """
    
    def __init__(self, 
                 lookback_periods: int = 50,
                 source_weights: Optional[Dict[str, float]] = None,
                 decay_factor: float = 0.95):
        """
        Initialize Sentiment Scores analyzer
        
        Args:
            lookback_periods: Number of periods for sentiment analysis
            source_weights: Weights for different sentiment sources
            decay_factor: Time decay factor for historical sentiment
        """
        self.lookback_periods = lookback_periods
        self.decay_factor = decay_factor
        
        # Default source weights
        self.source_weights = source_weights or {
            SentimentSource.NEWS.value: 0.25,
            SentimentSource.SOCIAL_MEDIA.value: 0.15,
            SentimentSource.ECONOMIC_CALENDAR.value: 0.20,
            SentimentSource.ANALYST_REPORTS.value: 0.20,
            SentimentSource.MARKET_DATA.value: 0.15,
            SentimentSource.COT_REPORTS.value: 0.05
        }
        
        # Sentiment history
        self.sentiment_history: List[SentimentData] = []
        self.price_history: List[float] = []
        self.timestamp_history: List[datetime] = []
        
        # Forex-specific keywords for sentiment analysis
        self.bullish_keywords = [
            'bullish', 'buy', 'long', 'positive', 'optimistic', 'strong', 'rally',
            'uptrend', 'breakout', 'support', 'momentum', 'growth', 'recovery',
            'dovish', 'stimulus', 'easing', 'accommodation'
        ]
        
        self.bearish_keywords = [
            'bearish', 'sell', 'short', 'negative', 'pessimistic', 'weak', 'decline',
            'downtrend', 'breakdown', 'resistance', 'correction', 'recession',
            'hawkish', 'tightening', 'tapering', 'restrictive'
        ]
        
        logger.info(f"SentimentScores initialized with {lookback_periods} periods")
    
    def _normalize_sentiment_score(self, raw_score: float, source: SentimentSource) -> float:
        """
        Normalize sentiment score to -1 to 1 range
        
        Args:
            raw_score: Raw sentiment score
            source: Sentiment source
            
        Returns:
            Normalized sentiment score (-1 to 1)
        """
        # Different normalization based on source
        if source == SentimentSource.NEWS:
            # News scores typically range from -1 to 1
            return max(-1.0, min(1.0, raw_score))
        elif source == SentimentSource.SOCIAL_MEDIA:
            # Social media scores might be 0-100
            if raw_score > 10:  # Assume 0-100 scale
                return (raw_score - 50) / 50
            else:  # Assume -1 to 1 scale
                return max(-1.0, min(1.0, raw_score))
        elif source == SentimentSource.ECONOMIC_CALENDAR:
            # Economic impact scores (typically 1-3)
            if raw_score > 2:
                return (raw_score - 2) / 1  # Convert 1-3 to -1 to 1
            else:
                return max(-1.0, min(1.0, raw_score))
        else:
            # Default normalization
            return max(-1.0, min(1.0, raw_score))
    
    def _analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment from text using keyword matching
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if not text:
            return 0.0, 0.0
        
        text_lower = text.lower()
        
        # Count bullish and bearish keywords
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        
        total_keywords = bullish_count + bearish_count
        
        if total_keywords == 0:
            return 0.0, 0.0
        
        # Calculate sentiment score
        sentiment_score = (bullish_count - bearish_count) / total_keywords
        
        # Calculate confidence based on keyword density
        word_count = len(text_lower.split())
        keyword_density = total_keywords / word_count if word_count > 0 else 0
        confidence = min(1.0, keyword_density * 10)  # Scale confidence
        
        return sentiment_score, confidence
    
    def _calculate_sentiment_momentum(self, recent_sentiments: List[float]) -> float:
        """
        Calculate sentiment momentum (rate of change)
        
        Args:
            recent_sentiments: List of recent sentiment scores
            
        Returns:
            Sentiment momentum value
        """
        if len(recent_sentiments) < 2:
            return 0.0
        
        # Calculate momentum as weighted average of recent changes
        changes = np.diff(recent_sentiments)
        
        if len(changes) == 0:
            return 0.0
        
        # Apply exponential weights (more recent = higher weight)
        weights = np.array([self.decay_factor ** i for i in range(len(changes))])
        weights = weights[::-1] / np.sum(weights[::-1])
        
        momentum = np.sum(changes * weights)
        return momentum
    
    def _calculate_sentiment_volatility(self, sentiments: List[float]) -> float:
        """
        Calculate sentiment volatility (standard deviation)
        
        Args:
            sentiments: List of sentiment scores
            
        Returns:
            Sentiment volatility value
        """
        if len(sentiments) < 2:
            return 0.0
        
        return np.std(sentiments)
    
    def _classify_sentiment_regime(self, sentiment: float, volatility: float) -> str:
        """
        Classify sentiment regime based on level and volatility
        
        Args:
            sentiment: Current sentiment score
            volatility: Sentiment volatility
            
        Returns:
            Sentiment regime classification
        """
        if volatility > 0.5:
            return "volatile_sentiment"
        elif sentiment > 0.3:
            return "bullish_consensus"
        elif sentiment < -0.3:
            return "bearish_consensus"
        elif abs(sentiment) < 0.1:
            return "neutral_sentiment"
        else:
            return "mixed_sentiment"
    
    def _calculate_session_sentiment(self, timestamp: datetime) -> str:
        """
        Determine trading session for sentiment analysis
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Trading session name
        """
        utc_hour = timestamp.hour
        
        if 22 <= utc_hour or utc_hour < 8:
            return "asian"
        elif 8 <= utc_hour < 16:
            return "london"
        elif 16 <= utc_hour < 22:
            return "ny"
        else:
            return "overlap"
    
    def _calculate_price_correlation(self, sentiments: List[float], prices: List[float]) -> float:
        """
        Calculate correlation between sentiment and price movements
        
        Args:
            sentiments: List of sentiment scores
            prices: List of price values
            
        Returns:
            Correlation coefficient
        """
        if len(sentiments) < 2 or len(prices) < 2 or len(sentiments) != len(prices):
            return 0.0
        
        try:
            correlation = np.corrcoef(sentiments, prices)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def add_sentiment_data(self, 
                          source: SentimentSource,
                          raw_score: float,
                          timestamp: Optional[datetime] = None,
                          confidence: float = 1.0,
                          text: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add sentiment data point
        
        Args:
            source: Sentiment data source
            raw_score: Raw sentiment score
            timestamp: Data timestamp (uses current time if None)
            confidence: Confidence in the sentiment score
            text: Optional text for analysis
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Normalize sentiment score
        normalized_score = self._normalize_sentiment_score(raw_score, source)
        
        # Analyze text if provided
        if text:
            text_sentiment, text_confidence = self._analyze_text_sentiment(text)
            # Combine scores with text analysis
            normalized_score = (normalized_score + text_sentiment) / 2
            confidence = (confidence + text_confidence) / 2
        
        sentiment_data = SentimentData(
            source=source,
            timestamp=timestamp,
            raw_score=raw_score,
            normalized_score=normalized_score,
            confidence=confidence,
            text=text,
            metadata=metadata
        )
        
        self.sentiment_history.append(sentiment_data)
        
        # Maintain history size
        if len(self.sentiment_history) > self.lookback_periods * 2:
            self.sentiment_history = self.sentiment_history[-self.lookback_periods * 2:]
        
        logger.debug(f"Added sentiment data: {source.value} = {normalized_score:.3f}")
    
    def add_price_data(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Add price data for correlation analysis
        
        Args:
            price: Price value
            timestamp: Data timestamp
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.price_history.append(price)
        self.timestamp_history.append(timestamp)
        
        # Maintain history size
        if len(self.price_history) > self.lookback_periods * 2:
            self.price_history = self.price_history[-self.lookback_periods * 2:]
            self.timestamp_history = self.timestamp_history[-self.lookback_periods * 2:]
    
    def calculate_sentiment_scores(self, 
                                 time_window: Optional[timedelta] = None) -> SentimentResults:
        """
        Calculate comprehensive sentiment scores
        
        Args:
            time_window: Time window for analysis (uses all data if None)
            
        Returns:
            SentimentResults object with analysis
        """
        try:
            if not self.sentiment_history:
                logger.warning("No sentiment data available")
                return self._create_empty_results()
            
            current_time = datetime.utcnow()
            
            # Filter data by time window if specified
            if time_window:
                cutoff_time = current_time - time_window
                relevant_data = [d for d in self.sentiment_history if d.timestamp >= cutoff_time]
            else:
                relevant_data = self.sentiment_history[-self.lookback_periods:]
            
            if not relevant_data:
                return self._create_empty_results()
            
            # Calculate source breakdown
            source_breakdown = {}
            source_totals = {}
            
            for data in relevant_data:
                source_name = data.source.value
                weight = self.source_weights.get(source_name, 0.1)
                
                if source_name not in source_breakdown:
                    source_breakdown[source_name] = 0.0
                    source_totals[source_name] = 0.0
                
                source_breakdown[source_name] += data.normalized_score * data.confidence * weight
                source_totals[source_name] += data.confidence * weight
            
            # Normalize source breakdown
            for source_name in source_breakdown:
                if source_totals[source_name] > 0:
                    source_breakdown[source_name] /= source_totals[source_name]
            
            # Calculate overall sentiment
            total_weight = 0.0
            weighted_sentiment = 0.0
            
            for source_name, sentiment in source_breakdown.items():
                weight = self.source_weights.get(source_name, 0.1)
                weighted_sentiment += sentiment * weight
                total_weight += weight
            
            overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
            
            # Classify sentiment level
            if overall_sentiment <= -0.6:
                sentiment_level = SentimentLevel.EXTREMELY_BEARISH
            elif overall_sentiment <= -0.3:
                sentiment_level = SentimentLevel.BEARISH
            elif overall_sentiment <= -0.1:
                sentiment_level = SentimentLevel.SLIGHTLY_BEARISH
            elif overall_sentiment <= 0.1:
                sentiment_level = SentimentLevel.NEUTRAL
            elif overall_sentiment <= 0.3:
                sentiment_level = SentimentLevel.SLIGHTLY_BULLISH
            elif overall_sentiment <= 0.6:
                sentiment_level = SentimentLevel.BULLISH
            else:
                sentiment_level = SentimentLevel.EXTREMELY_BULLISH
            
            # Calculate sentiment momentum
            recent_scores = [d.normalized_score for d in relevant_data[-10:]]
            momentum = self._calculate_sentiment_momentum(recent_scores)
            
            # Calculate sentiment volatility
            all_scores = [d.normalized_score for d in relevant_data]
            volatility = self._calculate_sentiment_volatility(all_scores)
            
            # Calculate confidence
            confidences = [d.confidence for d in relevant_data]
            confidence = np.mean(confidences) if confidences else 0.0
            
            # Classify regime
            regime = self._classify_sentiment_regime(overall_sentiment, volatility)
            
            # Calculate session-based sentiment
            session_sentiment = {}
            for session in ["asian", "london", "ny", "overlap"]:
                session_data = []
                for data in relevant_data:
                    if self._calculate_session_sentiment(data.timestamp) == session:
                        session_data.append(data.normalized_score)
                
                session_sentiment[session] = np.mean(session_data) if session_data else 0.0
            
            # Calculate price correlation
            if len(self.price_history) >= len(all_scores):
                recent_prices = self.price_history[-len(all_scores):]
                correlation_with_price = self._calculate_price_correlation(all_scores, recent_prices)
            else:
                correlation_with_price = 0.0
            
            # Calculate sentiment divergence (difference between current and momentum)
            sentiment_divergence = overall_sentiment - momentum
            
            result = SentimentResults(
                overall_sentiment=overall_sentiment,
                sentiment_level=sentiment_level,
                source_breakdown=source_breakdown,
                momentum=momentum,
                volatility=volatility,
                confidence=confidence,
                regime=regime,
                session_sentiment=session_sentiment,
                correlation_with_price=correlation_with_price,
                sentiment_divergence=sentiment_divergence
            )
            
            logger.info(f"Sentiment analysis complete: {sentiment_level.value} "
                       f"({overall_sentiment:.3f}), momentum={momentum:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating sentiment scores: {str(e)}")
            raise
    
    def _create_empty_results(self) -> SentimentResults:
        """Create empty sentiment results for error cases"""
        return SentimentResults(
            overall_sentiment=0.0,
            sentiment_level=SentimentLevel.NEUTRAL,
            source_breakdown={},
            momentum=0.0,
            volatility=0.0,
            confidence=0.0,
            regime="neutral_sentiment",
            session_sentiment={},
            correlation_with_price=0.0,
            sentiment_divergence=0.0
        )
    
    def get_trading_signals(self, results: SentimentResults) -> Dict[str, Any]:
        """
        Generate trading signals based on sentiment analysis
        
        Args:
            results: SentimentResults from analysis
            
        Returns:
            Dictionary with trading signals and recommendations
        """
        signals = {
            "sentiment_direction": "bullish" if results.overall_sentiment > 0 else "bearish",
            "sentiment_strength": abs(results.overall_sentiment),
            "sentiment_level": results.sentiment_level.value,
            "momentum_direction": "positive" if results.momentum > 0 else "negative",
            "regime": results.regime,
            "confidence": results.confidence
        }
        
        # Sentiment-based trading recommendations
        if results.sentiment_level in [SentimentLevel.EXTREMELY_BULLISH, SentimentLevel.BULLISH]:
            signals["trading_bias"] = "long"
            signals["strategy_preference"] = "trend_following"
        elif results.sentiment_level in [SentimentLevel.EXTREMELY_BEARISH, SentimentLevel.BEARISH]:
            signals["trading_bias"] = "short"
            signals["strategy_preference"] = "trend_following"
        else:
            signals["trading_bias"] = "neutral"
            signals["strategy_preference"] = "range_trading"
        
        # Momentum-based signals
        if abs(results.momentum) > 0.2:
            signals["momentum_signal"] = "strong"
            signals["timeframe_preference"] = "M15-H1"
        elif abs(results.momentum) > 0.1:
            signals["momentum_signal"] = "moderate"
            signals["timeframe_preference"] = "H1-H4"
        else:
            signals["momentum_signal"] = "weak"
            signals["timeframe_preference"] = "H4-D1"
        
        # Volatility-based signals
        if results.volatility > 0.5:
            signals["volatility_regime"] = "high"
            signals["risk_adjustment"] = "increase_stops"
        elif results.volatility < 0.2:
            signals["volatility_regime"] = "low"
            signals["risk_adjustment"] = "tighten_stops"
        else:
            signals["volatility_regime"] = "normal"
            signals["risk_adjustment"] = "standard"
        
        # Divergence signals
        if abs(results.sentiment_divergence) > 0.3:
            signals["divergence_warning"] = True
            signals["divergence_type"] = "bearish" if results.sentiment_divergence > 0 else "bullish"
        else:
            signals["divergence_warning"] = False
        
        return signals
