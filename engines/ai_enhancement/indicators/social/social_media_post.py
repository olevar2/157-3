"""
Social Media Post Indicator

This indicator analyzes social media posts and extracts sentiment, engagement, and influence metrics
that can be used for market sentiment analysis and social trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
import json

# For direct script testing
try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator, IndicatorConfig
except ImportError:
    import sys
    import os
    # Add the project root to the path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    sys.path.insert(0, project_root)
    try:
        from engines.ai_enhancement.indicators.base_indicator import BaseIndicator, IndicatorConfig
    except ImportError:
        # Fallback for direct testing
        class IndicatorConfig:
            def __init__(self):
                pass
            def __post_init__(self):
                pass
        
        class BaseIndicator:
            def __init__(self, config):
                self.config = config
            def _handle_error(self, msg):
                print(f"Error: {msg}")
            def reset(self):
                pass


@dataclass
class SocialMediaConfig(IndicatorConfig):
    """Configuration for Social Media Post indicator."""
    
    sentiment_window: int = 100
    engagement_threshold: float = 0.1  # Minimum engagement rate for consideration
    influence_weight: float = 0.3  # Weight for follower count impact
    time_decay_hours: float = 24.0  # Time decay for post relevance
    bullish_keywords: List[str] = None
    bearish_keywords: List[str] = None
    platform_weights: Dict[str, float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.sentiment_window < 10:
            raise ValueError("Sentiment window must be at least 10")
        if not 0.0 <= self.engagement_threshold <= 1.0:
            raise ValueError("Engagement threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.influence_weight <= 1.0:
            raise ValueError("Influence weight must be between 0.0 and 1.0")
        
        # Default keyword lists
        if self.bullish_keywords is None:
            self.bullish_keywords = [
                'bullish', 'buy', 'long', 'bull', 'pump', 'moon', 'rocket',
                'green', 'profit', 'gain', 'up', 'rise', 'surge', 'breakout',
                'support', 'strong', 'bounce', 'rally', 'uptrend', 'momentum'
            ]
        
        if self.bearish_keywords is None:
            self.bearish_keywords = [
                'bearish', 'sell', 'short', 'bear', 'dump', 'crash', 'fall',
                'red', 'loss', 'down', 'drop', 'decline', 'breakdown',
                'resistance', 'weak', 'correction', 'downtrend', 'panic'
            ]
        
        # Default platform weights
        if self.platform_weights is None:
            self.platform_weights = {
                'twitter': 1.0,
                'reddit': 0.8,
                'telegram': 0.6,
                'discord': 0.5,
                'facebook': 0.4,
                'instagram': 0.3,
                'other': 0.2
            }


@dataclass
class SocialMediaPost:
    """Data structure for a social media post."""
    
    content: str
    platform: str
    timestamp: datetime
    author_id: str
    author_followers: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    hashtags: List[str] = None
    mentions: List[str] = None
    symbols: List[str] = None  # Financial symbols mentioned
    
    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []
        if self.mentions is None:
            self.mentions = []
        if self.symbols is None:
            self.symbols = []


class SocialMediaPostIndicator(BaseIndicator):
    """
    Social Media Post Indicator
    
    This indicator analyzes social media posts to extract sentiment and engagement
    metrics that can be used for market sentiment analysis. It processes text content,
    engagement metrics, and influence factors to generate trading signals.
    
    The indicator provides:
    1. Sentiment analysis of post content
    2. Engagement rate calculation
    3. Influence score based on author metrics
    4. Time-weighted relevance scoring
    5. Platform-specific weighting
    6. Aggregated sentiment signals
    
    Features:
    - Real-time sentiment scoring
    - Keyword-based sentiment classification
    - Engagement rate normalization
    - Follower influence weighting
    - Time decay for post relevance
    - Platform-specific adjustments
    - Trend detection and momentum
    
    Interpretation:
    - Positive values: Bullish sentiment dominant
    - Negative values: Bearish sentiment dominant
    - Values near zero: Neutral sentiment
    - Higher absolute values: Stronger sentiment conviction
    """
    
    def __init__(self, config: Optional[SocialMediaConfig] = None):
        """Initialize Social Media Post indicator."""
        if config is None:
            config = SocialMediaConfig()
        
        super().__init__(config)
        self.config: SocialMediaConfig = config
        
        # Internal state
        self._posts_buffer = []
        self._sentiment_scores = []
        self._engagement_scores = []
        self._influence_scores = []
        
        # Results storage
        self.sentiment_signal = []
        self.engagement_momentum = []
        self.influence_factor = []
        self.post_count = []
        self.trend_strength = []
        
        # Sentiment tracking
        self._bullish_count = 0
        self._bearish_count = 0
        self._neutral_count = 0
        
    def _analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment of post content."""
        content_lower = content.lower()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zA-Z]+\b', content_lower)
        
        if not words:
            return 0.0
        
        bullish_score = 0
        bearish_score = 0
        
        # Count keyword matches
        for word in words:
            if word in self.config.bullish_keywords:
                bullish_score += 1
            elif word in self.config.bearish_keywords:
                bearish_score += 1
        
        # Calculate sentiment score
        total_sentiment_words = bullish_score + bearish_score
        if total_sentiment_words == 0:
            return 0.0
        
        # Normalize to [-1, 1] range
        sentiment = (bullish_score - bearish_score) / len(words)
        
        # Apply sigmoid-like transformation for better distribution
        sentiment = np.tanh(sentiment * 10)
        
        return sentiment
    
    def _calculate_engagement_rate(self, post: SocialMediaPost) -> float:
        """Calculate engagement rate for a post."""
        if post.author_followers <= 0:
            return 0.0
        
        total_engagement = post.likes + post.shares + post.comments
        engagement_rate = total_engagement / post.author_followers
        
        # Cap at 1.0 (100% engagement rate)
        return min(engagement_rate, 1.0)
    
    def _calculate_influence_score(self, post: SocialMediaPost) -> float:
        """Calculate influence score based on author metrics."""
        # Log transformation for follower count
        follower_score = np.log(max(post.author_followers, 1)) / np.log(1000000)  # Normalize to 1M followers
        follower_score = min(follower_score, 1.0)
        
        # Engagement component
        engagement_rate = self._calculate_engagement_rate(post)
        
        # Platform weight
        platform_weight = self.config.platform_weights.get(post.platform.lower(), 0.2)
        
        # Combined influence score
        influence = (
            follower_score * self.config.influence_weight +
            engagement_rate * (1 - self.config.influence_weight)
        ) * platform_weight
        
        return influence
    
    def _calculate_time_decay(self, post: SocialMediaPost, current_time: datetime) -> float:
        """Calculate time decay factor for post relevance."""
        time_diff = (current_time - post.timestamp).total_seconds() / 3600  # Hours
        
        if time_diff < 0:
            return 1.0  # Future posts get full weight
        
        # Exponential decay
        decay_factor = np.exp(-time_diff / self.config.time_decay_hours)
        
        return decay_factor
    
    def _extract_symbols(self, content: str) -> List[str]:
        """Extract financial symbols from post content."""
        # Look for common symbol patterns
        patterns = [
            r'\$([A-Z]{1,5})',  # $AAPL format
            r'\b([A-Z]{2,5})\b',  # TSLA format
            r'#([A-Z]{1,5})',  # #BTC format
        ]
        
        symbols = []
        for pattern in patterns:
            matches = re.findall(pattern, content.upper())
            symbols.extend(matches)
        
        # Remove duplicates and common words
        common_words = {'THE', 'AND', 'OR', 'BUT', 'FOR', 'WITH', 'TO', 'FROM', 'BY', 'AT', 'ON', 'IN'}
        symbols = list(set([s for s in symbols if s not in common_words and len(s) >= 2]))
        
        return symbols
    
    def add_post(self, post: SocialMediaPost, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Add a social media post for analysis."""
        try:
            if current_time is None:
                current_time = datetime.now()
            
            # Analyze the post
            sentiment = self._analyze_sentiment(post.content)
            engagement_rate = self._calculate_engagement_rate(post)
            influence = self._calculate_influence_score(post)
            time_weight = self._calculate_time_decay(post, current_time)
            
            # Extract symbols
            post.symbols = self._extract_symbols(post.content)
            
            # Store post data
            post_data = {
                'post': post,
                'sentiment': sentiment,
                'engagement_rate': engagement_rate,
                'influence': influence,
                'time_weight': time_weight,
                'timestamp': current_time
            }
            
            self._posts_buffer.append(post_data)
            
            # Maintain buffer size
            if len(self._posts_buffer) > self.config.sentiment_window * 2:
                self._posts_buffer = self._posts_buffer[-self.config.sentiment_window * 2:]
            
            # Update sentiment counts
            if sentiment > 0.1:
                self._bullish_count += 1
            elif sentiment < -0.1:
                self._bearish_count += 1
            else:
                self._neutral_count += 1
            
            # Calculate aggregated metrics
            return self._calculate_aggregate_metrics(current_time)
            
        except Exception as e:
            self._handle_error(f"Error adding social media post: {e}")
            return {
                'sentiment_signal': 0.0,
                'engagement_momentum': 0.0,
                'influence_factor': 0.0,
                'post_count': 0,
                'trend_strength': 0.0
            }
    
    def _calculate_aggregate_metrics(self, current_time: datetime) -> Dict[str, Any]:
        """Calculate aggregated sentiment and engagement metrics."""
        if not self._posts_buffer:
            return {
                'sentiment_signal': 0.0,
                'engagement_momentum': 0.0,
                'influence_factor': 0.0,
                'post_count': 0,
                'trend_strength': 0.0
            }
        
        # Filter posts within time window
        cutoff_time = current_time - timedelta(hours=self.config.time_decay_hours)
        recent_posts = [
            p for p in self._posts_buffer 
            if p['post'].timestamp >= cutoff_time
        ]
        
        if not recent_posts:
            return {
                'sentiment_signal': 0.0,
                'engagement_momentum': 0.0,
                'influence_factor': 0.0,
                'post_count': 0,
                'trend_strength': 0.0
            }
        
        # Calculate weighted averages
        total_weight = 0
        weighted_sentiment = 0
        weighted_engagement = 0
        weighted_influence = 0
        
        for post_data in recent_posts:
            weight = (
                post_data['time_weight'] * 
                post_data['influence'] * 
                max(post_data['engagement_rate'], self.config.engagement_threshold)
            )
            
            total_weight += weight
            weighted_sentiment += post_data['sentiment'] * weight
            weighted_engagement += post_data['engagement_rate'] * weight
            weighted_influence += post_data['influence'] * weight
        
        if total_weight > 0:
            sentiment_signal = weighted_sentiment / total_weight
            engagement_momentum = weighted_engagement / total_weight
            influence_factor = weighted_influence / total_weight
        else:
            sentiment_signal = 0.0
            engagement_momentum = 0.0
            influence_factor = 0.0
        
        # Calculate trend strength
        if len(recent_posts) >= 5:
            recent_sentiments = [p['sentiment'] for p in recent_posts[-10:]]
            trend_strength = np.std(recent_sentiments) * abs(sentiment_signal)
        else:
            trend_strength = 0.0
        
        # Store results
        self.sentiment_signal.append(sentiment_signal)
        self.engagement_momentum.append(engagement_momentum)
        self.influence_factor.append(influence_factor)
        self.post_count.append(len(recent_posts))
        self.trend_strength.append(trend_strength)
        
        # Maintain result buffer sizes
        max_history = 1000
        if len(self.sentiment_signal) > max_history:
            self.sentiment_signal = self.sentiment_signal[-max_history:]
            self.engagement_momentum = self.engagement_momentum[-max_history:]
            self.influence_factor = self.influence_factor[-max_history:]
            self.post_count = self.post_count[-max_history:]
            self.trend_strength = self.trend_strength[-max_history:]
        
        return {
            'sentiment_signal': sentiment_signal,
            'engagement_momentum': engagement_momentum,
            'influence_factor': influence_factor,
            'post_count': len(recent_posts),
            'trend_strength': trend_strength,
            'bullish_ratio': self._bullish_count / max(self._bullish_count + self._bearish_count + self._neutral_count, 1),
            'bearish_ratio': self._bearish_count / max(self._bullish_count + self._bearish_count + self._neutral_count, 1),
            'sentiment_strength': self._interpret_sentiment(sentiment_signal),
            'engagement_level': self._interpret_engagement(engagement_momentum)
        }
    
    def update(self, post_data: Union[SocialMediaPost, Dict], 
               timestamp: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Update indicator with new social media post data."""
        try:
            # Convert dict to SocialMediaPost if needed
            if isinstance(post_data, dict):
                post = SocialMediaPost(**post_data)
            else:
                post = post_data
            
            # Use provided timestamp or current time
            if timestamp is not None:
                if isinstance(timestamp, (int, float)):
                    current_time = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, datetime):
                    current_time = timestamp
                else:
                    current_time = datetime.now()
            else:
                current_time = datetime.now()
            
            return self.add_post(post, current_time)
            
        except Exception as e:
            self._handle_error(f"Error in Social Media Post update: {e}")
            return {
                'sentiment_signal': 0.0,
                'engagement_momentum': 0.0,
                'influence_factor': 0.0,
                'post_count': 0,
                'trend_strength': 0.0
            }
    
    def _interpret_sentiment(self, sentiment: float) -> str:
        """Interpret sentiment signal value."""
        if sentiment > 0.5:
            return "Very Bullish"
        elif sentiment > 0.2:
            return "Bullish"
        elif sentiment > -0.2:
            return "Neutral"
        elif sentiment > -0.5:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def _interpret_engagement(self, engagement: float) -> str:
        """Interpret engagement momentum value."""
        if engagement > 0.1:
            return "Very High"
        elif engagement > 0.05:
            return "High"
        elif engagement > 0.02:
            return "Medium"
        elif engagement > 0.01:
            return "Low"
        else:
            return "Very Low"
    
    def get_sentiment_data(self) -> Dict[str, List]:
        """Get all sentiment-related data."""
        return {
            'sentiment_signal': self.sentiment_signal.copy(),
            'engagement_momentum': self.engagement_momentum.copy(),
            'influence_factor': self.influence_factor.copy(),
            'post_count': self.post_count.copy(),
            'trend_strength': self.trend_strength.copy()
        }
    
    def get_recent_posts(self, hours: float = 24.0) -> List[SocialMediaPost]:
        """Get recent posts within specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_posts = [
            p['post'] for p in self._posts_buffer 
            if p['post'].timestamp >= cutoff_time
        ]
        return recent_posts
    
    def get_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment analysis for a specific symbol."""
        symbol_posts = [
            p for p in self._posts_buffer 
            if symbol.upper() in [s.upper() for s in p['post'].symbols]
        ]
        
        if not symbol_posts:
            return {
                'symbol': symbol,
                'post_count': 0,
                'sentiment': 0.0,
                'engagement': 0.0,
                'influence': 0.0
            }
        
        # Calculate symbol-specific metrics
        total_weight = 0
        weighted_sentiment = 0
        weighted_engagement = 0
        weighted_influence = 0
        
        for post_data in symbol_posts:
            weight = post_data['influence'] * post_data['time_weight']
            total_weight += weight
            weighted_sentiment += post_data['sentiment'] * weight
            weighted_engagement += post_data['engagement_rate'] * weight
            weighted_influence += post_data['influence'] * weight
        
        if total_weight > 0:
            return {
                'symbol': symbol,
                'post_count': len(symbol_posts),
                'sentiment': weighted_sentiment / total_weight,
                'engagement': weighted_engagement / total_weight,
                'influence': weighted_influence / total_weight
            }
        else:
            return {
                'symbol': symbol,
                'post_count': len(symbol_posts),
                'sentiment': 0.0,
                'engagement': 0.0,
                'influence': 0.0
            }
    
    def reset(self):
        """Reset indicator state."""
        super().reset()
        self._posts_buffer.clear()
        self._sentiment_scores.clear()
        self._engagement_scores.clear()
        self._influence_scores.clear()
        self.sentiment_signal.clear()
        self.engagement_momentum.clear()
        self.influence_factor.clear()
        self.post_count.clear()
        self.trend_strength.clear()
        self._bullish_count = 0
        self._bearish_count = 0
        self._neutral_count = 0


# Example usage and testing
if __name__ == "__main__":
    print("Testing Social Media Post Indicator")
    print("=" * 50)
    
    # Create indicator
    config = SocialMediaConfig(
        sentiment_window=50,
        engagement_threshold=0.05,
        influence_weight=0.4,
        time_decay_hours=12.0
    )
    
    indicator = SocialMediaPostIndicator(config)
    
    # Generate test posts
    import random
    
    test_posts = [
        {
            'content': "TSLA is going to the moon! 🚀 This stock is absolutely bullish right now. Great breakout pattern!",
            'platform': 'twitter',
            'author_id': 'trader123',
            'author_followers': 10000,
            'likes': 150,
            'shares': 25,
            'comments': 30,
            'symbols': ['TSLA']
        },
        {
            'content': "Market looks bearish today. I'm short on AAPL and expecting a big drop. Resistance holding strong.",
            'platform': 'reddit',
            'author_id': 'bearish_trader',
            'author_followers': 5000,
            'likes': 80,
            'shares': 10,
            'comments': 45,
            'symbols': ['AAPL']
        },
        {
            'content': "Bitcoin pump incoming! $BTC breaking out of consolidation. Very bullish setup here! 💎🙌",
            'platform': 'twitter',
            'author_id': 'crypto_guru',
            'author_followers': 50000,
            'likes': 500,
            'shares': 100,
            'comments': 80,
            'symbols': ['BTC']
        },
        {
            'content': "SPY looks neutral to me. Waiting for a clear direction before making any moves.",
            'platform': 'telegram',
            'author_id': 'patient_trader',
            'author_followers': 2000,
            'likes': 20,
            'shares': 5,
            'comments': 10,
            'symbols': ['SPY']
        },
        {
            'content': "NVDA earnings coming up. Could see a big move either way. High volatility expected.",
            'platform': 'discord',
            'author_id': 'tech_analyst',
            'author_followers': 8000,
            'likes': 120,
            'shares': 15,
            'comments': 25,
            'symbols': ['NVDA']
        }
    ]
    
    base_time = datetime.now() - timedelta(hours=2)
    
    print(f"Processing {len(test_posts)} social media posts...")
    
    # Process posts with different timestamps
    results = []
    for i, post_data in enumerate(test_posts):
        # Add timestamp
        post_timestamp = base_time + timedelta(minutes=i * 30)
        post_data['timestamp'] = post_timestamp
        
        # Create post object
        post = SocialMediaPost(**post_data)
        
        # Update indicator
        current_time = base_time + timedelta(minutes=(i + 1) * 30)
        result = indicator.add_post(post, current_time)
        results.append(result)
        
        print(f"\nPost {i + 1}: {post.platform} by {post.author_id}")
        # Handle Unicode characters safely for Windows
        content_preview = post.content[:60].encode('ascii', 'ignore').decode('ascii')
        print(f"  Content: {content_preview}...")
        print(f"  Symbols: {post.symbols}")
        print(f"  Sentiment Signal: {result['sentiment_signal']:.3f}")
        print(f"  Engagement Momentum: {result['engagement_momentum']:.3f}")
        print(f"  Influence Factor: {result['influence_factor']:.3f}")
        print(f"  Post Count: {result['post_count']}")
        print(f"  Trend Strength: {result['trend_strength']:.3f}")
        print(f"  Sentiment Strength: {result['sentiment_strength']}")
        print(f"  Engagement Level: {result['engagement_level']}")
    
    # Get aggregated data
    sentiment_data = indicator.get_sentiment_data()
    
    print("\nFinal Analysis:")
    print("-" * 30)
    
    if len(sentiment_data['sentiment_signal']) > 0:
        latest_sentiment = sentiment_data['sentiment_signal'][-1]
        latest_engagement = sentiment_data['engagement_momentum'][-1]
        latest_influence = sentiment_data['influence_factor'][-1]
        latest_count = sentiment_data['post_count'][-1]
        latest_trend = sentiment_data['trend_strength'][-1]
        
        print(f"Latest Sentiment Signal: {latest_sentiment:.3f}")
        print(f"Latest Engagement Momentum: {latest_engagement:.3f}")
        print(f"Latest Influence Factor: {latest_influence:.3f}")
        print(f"Latest Post Count: {latest_count}")
        print(f"Latest Trend Strength: {latest_trend:.3f}")
        
        # Overall sentiment interpretation
        if latest_sentiment > 0.2:
            overall_sentiment = "Bullish"
        elif latest_sentiment < -0.2:
            overall_sentiment = "Bearish"
        else:
            overall_sentiment = "Neutral"
        
        print(f"\nOverall Market Sentiment: {overall_sentiment}")
        
        # Symbol-specific analysis
        print(f"\nSymbol-Specific Analysis:")
        symbols = ['TSLA', 'AAPL', 'BTC', 'SPY', 'NVDA']
        for symbol in symbols:
            symbol_data = indicator.get_symbol_sentiment(symbol)
            if symbol_data['post_count'] > 0:
                print(f"  {symbol}: Sentiment={symbol_data['sentiment']:.3f}, "
                      f"Posts={symbol_data['post_count']}, "
                      f"Engagement={symbol_data['engagement']:.3f}")
    
    print("\nSocial Media Post Indicator test completed successfully!")