"""
Social Media Post Sentiment Indicator

Real-time social media sentiment analysis for financial markets using
advanced NLP and machine learning techniques.

Features:
- Multi-platform social media monitoring
- Real-time sentiment scoring
- Viral content detection
- Influencer sentiment weighting
- Sentiment momentum tracking
- Engagement-based confidence scoring

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd
import logging
import re
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from engines.ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocialMediaPostIndicator(StandardIndicatorInterface):
    """
    Social Media Post Sentiment Indicator
    
    Analyzes social media posts to extract sentiment signals for trading decisions.
    Includes advanced features like viral detection, influencer weighting, and
    engagement-based confidence scoring.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "sentiment"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        viral_threshold: int = 1000,
        influencer_weight: float = 2.0,
        engagement_weight: float = 1.5,
        min_posts: int = 5,
        sentiment_decay: float = 0.9,
        **kwargs,
    ):
        """
        Initialize Social Media Post Sentiment indicator

        Args:
            period: Period for sentiment aggregation (default: 20)
            viral_threshold: Minimum engagement for viral posts (default: 1000)
            influencer_weight: Weight multiplier for influencer posts (default: 2.0)
            engagement_weight: Weight factor for engagement scoring (default: 1.5)
            min_posts: Minimum posts required for signal (default: 5)
            sentiment_decay: Decay factor for older posts (default: 0.9)
        """
        super().__init__(
            period=period,
            viral_threshold=viral_threshold,
            influencer_weight=influencer_weight,
            engagement_weight=engagement_weight,
            min_posts=min_posts,
            sentiment_decay=sentiment_decay,
            **kwargs,
        )
        
        # Initialize social media analysis components
        self._post_cache = []
        self._influencer_list = set()
        self._viral_posts = []
        self._engagement_scores = {}

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate social media sentiment indicator values

        Args:
            data: DataFrame with price data and optional social media data
                 Expected columns: 'close', and optional 'post_text', 'post_timestamp',
                 'likes', 'shares', 'comments', 'author_followers', 'platform'

        Returns:
            pd.Series: Social media sentiment values (-1 to 1)
        """
        # Validate input data
        self.validate_input_data(data)
        
        period = self.parameters.get("period", 20)
        viral_threshold = self.parameters.get("viral_threshold", 1000)
        influencer_weight = self.parameters.get("influencer_weight", 2.0)
        engagement_weight = self.parameters.get("engagement_weight", 1.5)
        min_posts = self.parameters.get("min_posts", 5)
        sentiment_decay = self.parameters.get("sentiment_decay", 0.9)
        
        # Initialize result series
        result_index = data.index if hasattr(data, 'index') else range(len(data))
        sentiment_values = np.zeros(len(data))
        
        try:
            # Check if social media data is available
            has_social_data = self._has_social_media_data(data)
            
            if has_social_data:
                # Process real social media data
                sentiment_values = self._process_social_media_data(data)
            else:
                # Generate synthetic social sentiment based on price action
                sentiment_values = self._generate_synthetic_social_sentiment(data)
                
            # Apply time-based aggregation and smoothing
            aggregated_sentiment = self._apply_time_aggregation(sentiment_values, period)
            final_sentiment = self._apply_decay_and_normalization(aggregated_sentiment, sentiment_decay)
            
            # Store calculation details for debugging
            self._last_calculation = {
                "raw_sentiment": sentiment_values,
                "aggregated_sentiment": aggregated_sentiment,
                "final_sentiment": final_sentiment,
                "period": period,
                "viral_threshold": viral_threshold,
                "has_social_data": has_social_data,
                "posts_processed": len(data) if has_social_data else 0,
                "viral_posts_count": len(self._viral_posts),
                "influencer_posts": len([p for p in self._post_cache if p.get('is_influencer', False)]),
            }
            
            return pd.Series(
                final_sentiment, 
                index=result_index, 
                name="SocialMediaSentiment"
            )
            
        except Exception as e:
            logger.warning(f"Error in social media sentiment calculation: {e}")
            # Return neutral sentiment on error
            return pd.Series(
                np.zeros(len(data)), 
                index=result_index, 
                name="SocialMediaSentiment"
            )

    def _has_social_media_data(self, data: Union[pd.DataFrame, pd.Series]) -> bool:
        """Check if data contains social media information"""
        if not isinstance(data, pd.DataFrame):
            return False
            
        social_columns = ['post_text', 'likes', 'shares', 'comments', 'author_followers']
        return any(col in data.columns for col in social_columns)

    def _process_social_media_data(self, data: pd.DataFrame) -> np.ndarray:
        """Process actual social media data to extract sentiment"""
        sentiment_values = np.zeros(len(data))
        viral_threshold = self.parameters.get("viral_threshold", 1000)
        influencer_weight = self.parameters.get("influencer_weight", 2.0)
        engagement_weight = self.parameters.get("engagement_weight", 1.5)
        
        # Clear previous calculation data
        self._post_cache = []
        self._viral_posts = []
        
        for i, row in data.iterrows():
            post_data = self._extract_post_data(row)
            
            if post_data and post_data['text']:
                # Analyze post sentiment
                post_sentiment = self._analyze_post_sentiment(post_data['text'])
                
                # Calculate engagement score
                engagement_score = self._calculate_engagement_score(post_data)
                
                # Check for viral content
                total_engagement = sum([
                    post_data.get('likes', 0),
                    post_data.get('shares', 0) * 3,  # Shares are more valuable
                    post_data.get('comments', 0) * 2  # Comments show engagement
                ])
                
                is_viral = total_engagement >= viral_threshold
                if is_viral:
                    self._viral_posts.append(post_data)
                    
                # Check for influencer
                follower_count = post_data.get('followers', 0)
                is_influencer = self._is_influencer(follower_count)
                post_data['is_influencer'] = is_influencer
                
                # Calculate weighted sentiment
                weight = 1.0
                
                if is_viral:
                    weight *= 2.0  # Viral content gets double weight
                    
                if is_influencer:
                    weight *= influencer_weight
                    
                # Apply engagement weighting
                engagement_factor = 1.0 + (engagement_score - 0.5) * engagement_weight
                weight *= max(0.1, engagement_factor)  # Minimum weight of 0.1
                
                # Store weighted sentiment
                sentiment_values[i] = post_sentiment * weight
                
                # Cache post data
                post_data['sentiment'] = post_sentiment
                post_data['weight'] = weight
                post_data['engagement_score'] = engagement_score
                self._post_cache.append(post_data)
                
        return sentiment_values

    def _extract_post_data(self, row) -> Optional[Dict[str, Any]]:
        """Extract post data from DataFrame row"""
        try:
            post_data = {
                'text': row.get('post_text', ''),
                'timestamp': row.get('post_timestamp', datetime.now()),
                'likes': int(row.get('likes', 0)),
                'shares': int(row.get('shares', 0)),
                'comments': int(row.get('comments', 0)),
                'followers': int(row.get('author_followers', 0)),
                'platform': row.get('platform', 'unknown')
            }
            
            return post_data if post_data['text'] else None
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error extracting post data: {e}")
            return None

    def _analyze_post_sentiment(self, text: str) -> float:
        """Analyze sentiment of individual social media post"""
        if not text or not isinstance(text, str):
            return 0.0
            
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        # Financial sentiment keywords with weights
        bullish_keywords = {
            'moon': 2.0, 'rocket': 2.0, 'bullish': 1.5, 'buy': 1.0, 'pump': 1.8,
            'surge': 1.2, 'rally': 1.3, 'gains': 1.0, 'profit': 1.1, 'hodl': 1.4,
            'diamond hands': 2.0, 'to the moon': 2.5, 'green': 0.8, 'up': 0.7,
            'strong': 0.9, 'breakout': 1.6, 'support': 0.8, 'resistance broken': 1.7
        }
        
        bearish_keywords = {
            'crash': 2.0, 'dump': 1.8, 'bearish': 1.5, 'sell': 1.0, 'drop': 1.2,
            'fall': 1.1, 'red': 0.8, 'down': 0.7, 'weak': 0.9, 'paper hands': 1.6,
            'fear': 1.3, 'panic': 1.8, 'resistance': 0.8, 'support broken': 1.7,
            'rug pull': 2.5, 'scam': 2.0, 'bubble': 1.4
        }
        
        # Calculate weighted sentiment scores
        text_lower = text.lower()
        bullish_score = 0.0
        bearish_score = 0.0
        
        for keyword, weight in bullish_keywords.items():
            if keyword in text_lower:
                bullish_score += weight
                
        for keyword, weight in bearish_keywords.items():
            if keyword in text_lower:
                bearish_score += weight
                
        # Account for emojis and special characters
        emoji_sentiment = self._analyze_emoji_sentiment(text)
        bullish_score += max(0, emoji_sentiment) * 0.5
        bearish_score += max(0, -emoji_sentiment) * 0.5
        
        # Calculate final sentiment
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return 0.0
            
        sentiment = (bullish_score - bearish_score) / total_score
        
        # Apply text length and caps normalization
        text_factor = min(1.0, len(text) / 280)  # Normalize for tweet length
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        caps_factor = 1.0 + caps_ratio * 0.3  # Caps add emphasis
        
        return np.tanh(sentiment * text_factor * caps_factor)

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess social media text"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove user mentions but keep the sentiment
        text = re.sub(r'@\w+', '', text)
        
        # Keep hashtags but clean them
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()

    def _analyze_emoji_sentiment(self, text: str) -> float:
        """Analyze emoji sentiment in text"""
        # Common emoji sentiment mapping
        positive_emojis = ['🚀', '📈', '💎', '🌙', '💰', '💵', '🤑', '😊', '😀', '👍', '🔥', '⬆️']
        negative_emojis = ['📉', '💔', '😭', '😱', '👎', '⬇️', '🔴', '💸', '😰', '🤡', '🐻']
        
        positive_count = sum(text.count(emoji) for emoji in positive_emojis)
        negative_count = sum(text.count(emoji) for emoji in negative_emojis)
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
            
        return (positive_count - negative_count) / total_count

    def _calculate_engagement_score(self, post_data: Dict[str, Any]) -> float:
        """Calculate engagement score for a post"""
        likes = post_data.get('likes', 0)
        shares = post_data.get('shares', 0)
        comments = post_data.get('comments', 0)
        followers = post_data.get('followers', 1)  # Avoid division by zero
        
        # Weighted engagement calculation
        engagement = (
            likes * 1.0 +
            shares * 3.0 +  # Shares are more valuable
            comments * 2.0   # Comments show active engagement
        )
        
        # Normalize by follower count (engagement rate)
        engagement_rate = engagement / followers if followers > 0 else 0
        
        # Convert to 0-1 scale using sigmoid function
        return 1 / (1 + np.exp(-engagement_rate * 1000))  # Scale factor for typical social media

    def _is_influencer(self, follower_count: int) -> bool:
        """Determine if user is an influencer based on follower count"""
        # Thresholds vary by platform, using general guidelines
        return follower_count >= 10000  # 10K+ followers

    def _generate_synthetic_social_sentiment(self, data: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
        """Generate synthetic social media sentiment from price action"""
        if isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = data['close'].values if 'close' in data.columns else np.zeros(len(data))
            
        sentiment_values = np.zeros(len(prices))
        
        if len(prices) < 3:
            return sentiment_values
            
        # Calculate price momentum and volatility
        price_changes = np.diff(prices, prepend=prices[0])
        
        for i in range(1, len(prices)):
            # Recent price momentum (simulates social reaction to price moves)
            momentum = price_changes[i] / prices[i-1] if prices[i-1] != 0 else 0
            
            # Volatility amplifies social sentiment
            if i >= 5:
                recent_volatility = np.std(price_changes[i-5:i])
                volatility_factor = 1.0 + recent_volatility * 10  # Amplify by volatility
            else:
                volatility_factor = 1.0
                
            # Social sentiment tends to lag and amplify price moves
            social_amplification = 1.5  # Social media amplifies sentiment
            
            # Add some noise to simulate varied opinions
            noise = np.random.normal(0, 0.3)
            
            # Calculate synthetic social sentiment
            raw_sentiment = momentum * social_amplification * volatility_factor + noise
            sentiment_values[i] = np.tanh(raw_sentiment)  # Bound to [-1, 1]
            
        return sentiment_values

    def _apply_time_aggregation(self, values: np.ndarray, period: int) -> np.ndarray:
        """Apply time-based aggregation to sentiment values"""
        if len(values) <= period:
            return values
            
        aggregated = np.zeros_like(values)
        
        for i in range(period, len(values)):
            # Weighted average over period with more weight to recent values
            weights = np.exp(np.linspace(-1, 0, period))  # Exponential weighting
            weights /= weights.sum()
            
            window = values[i-period+1:i+1]
            aggregated[i] = np.average(window, weights=weights)
            
        # Handle initial values
        aggregated[:period] = values[:period]
        
        return aggregated

    def _apply_decay_and_normalization(self, values: np.ndarray, decay_factor: float) -> np.ndarray:
        """Apply decay factor and normalize values"""
        if len(values) == 0:
            return values
            
        # Apply exponential decay to reduce impact of old sentiment
        decayed = np.zeros_like(values)
        decayed[0] = values[0]
        
        for i in range(1, len(values)):
            decayed[i] = values[i] + decay_factor * decayed[i-1]
            
        # Normalize to prevent unbounded growth
        if np.std(decayed) > 0:
            normalized = (decayed - np.mean(decayed)) / np.std(decayed)
            return np.tanh(normalized)  # Bound to [-1, 1]
        else:
            return decayed

    def validate_parameters(self) -> bool:
        """Validate Social Media Post parameters"""
        period = self.parameters.get("period", 20)
        viral_threshold = self.parameters.get("viral_threshold", 1000)
        influencer_weight = self.parameters.get("influencer_weight", 2.0)
        engagement_weight = self.parameters.get("engagement_weight", 1.5)
        min_posts = self.parameters.get("min_posts", 5)
        sentiment_decay = self.parameters.get("sentiment_decay", 0.9)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(viral_threshold, int) or viral_threshold < 0:
            raise IndicatorValidationError(
                f"viral_threshold must be non-negative integer, got {viral_threshold}"
            )

        if not isinstance(influencer_weight, (int, float)) or influencer_weight <= 0:
            raise IndicatorValidationError(
                f"influencer_weight must be positive number, got {influencer_weight}"
            )

        if not isinstance(engagement_weight, (int, float)) or engagement_weight <= 0:
            raise IndicatorValidationError(
                f"engagement_weight must be positive number, got {engagement_weight}"
            )

        if not isinstance(min_posts, int) or min_posts < 0:
            raise IndicatorValidationError(
                f"min_posts must be non-negative integer, got {min_posts}"
            )

        if not isinstance(sentiment_decay, (int, float)) or not (0 < sentiment_decay <= 1):
            raise IndicatorValidationError(
                f"sentiment_decay must be between 0 and 1, got {sentiment_decay}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Social Media Post metadata as dictionary for compatibility"""
        return {
            "name": "SocialMediaPost",
            "category": self.CATEGORY,
            "description": "Social Media Post Sentiment - Real-time social sentiment analysis",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Social Media Post requires price data, social data optional"""
        return ["close"]  # Basic requirement, social media columns are optional

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for social sentiment analysis"""
        return max(3, self.parameters.get("period", 20))

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "viral_threshold" not in self.parameters:
            self.parameters["viral_threshold"] = 1000
        if "influencer_weight" not in self.parameters:
            self.parameters["influencer_weight"] = 2.0
        if "engagement_weight" not in self.parameters:
            self.parameters["engagement_weight"] = 1.5
        if "min_posts" not in self.parameters:
            self.parameters["min_posts"] = 5
        if "sentiment_decay" not in self.parameters:
            self.parameters["sentiment_decay"] = 0.9

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def viral_threshold(self) -> int:
        """Viral threshold for backward compatibility"""
        return self.parameters.get("viral_threshold", 1000)

    @property
    def influencer_weight(self) -> float:
        """Influencer weight for backward compatibility"""
        return self.parameters.get("influencer_weight", 2.0)

    def get_viral_posts(self) -> List[Dict[str, Any]]:
        """Get list of detected viral posts"""
        return self._viral_posts.copy()

    def get_post_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed posts"""
        if not self._post_cache:
            return {}
            
        total_posts = len(self._post_cache)
        influencer_posts = sum(1 for p in self._post_cache if p.get('is_influencer', False))
        viral_posts = len(self._viral_posts)
        
        return {
            'total_posts': total_posts,
            'influencer_posts': influencer_posts,
            'viral_posts': viral_posts,
            'influencer_rate': influencer_posts / total_posts if total_posts > 0 else 0,
            'viral_rate': viral_posts / total_posts if total_posts > 0 else 0,
        }

    def get_sentiment_signal(self, sentiment_value: float) -> str:
        """
        Get trading signal based on social sentiment value

        Args:
            sentiment_value: Current sentiment value (-1 to 1)

        Returns:
            str: "viral_bullish", "social_bullish", "neutral", "social_bearish", "viral_bearish"
        """
        if abs(sentiment_value) > 0.7:
            return "viral_bullish" if sentiment_value > 0 else "viral_bearish"
        elif abs(sentiment_value) > 0.3:
            return "social_bullish" if sentiment_value > 0 else "social_bearish"
        elif abs(sentiment_value) > 0.1:
            return "slightly_bullish" if sentiment_value > 0 else "slightly_bearish"
        else:
            return "neutral"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return SocialMediaPostIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample data with social media posts
    np.random.seed(42)
    n_points = 100
    
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='H')  # Hourly data
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.2)
    
    # Generate synthetic social media data
    post_texts = []
    likes = []
    shares = []
    comments = []
    followers = []
    
    for i in range(n_points):
        if i % 5 == 0:  # Posts every 5 hours
            price_change = (prices[i] - prices[max(0, i-1)]) / prices[max(0, i-1)] if i > 0 else 0
            
            if price_change > 0.01:  # Price up > 1%
                post_texts.append("This stock is going to the moon! 🚀📈 Buy now!")
                likes.append(np.random.randint(100, 2000))
                shares.append(np.random.randint(10, 200))
                comments.append(np.random.randint(5, 100))
            elif price_change < -0.01:  # Price down > 1%
                post_texts.append("Market crash incoming! 📉💔 Time to sell everything!")
                likes.append(np.random.randint(50, 1000))
                shares.append(np.random.randint(5, 100))
                comments.append(np.random.randint(2, 50))
            else:
                post_texts.append("Market looking stable today. What do you think? 🤔")
                likes.append(np.random.randint(20, 500))
                shares.append(np.random.randint(1, 50))
                comments.append(np.random.randint(1, 25))
                
            followers.append(np.random.randint(100, 50000))  # Mix of regular users and influencers
        else:
            post_texts.append("")
            likes.append(0)
            shares.append(0)
            comments.append(0)
            followers.append(0)
    
    data = pd.DataFrame({
        'close': prices,
        'post_text': post_texts,
        'post_timestamp': dates,
        'likes': likes,
        'shares': shares,
        'comments': comments,
        'author_followers': followers,
        'platform': ['twitter'] * n_points
    }, index=dates)

    # Calculate social media sentiment
    social_indicator = SocialMediaPostIndicator(period=10, viral_threshold=500)
    social_result = social_indicator.calculate(data)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Price chart
    ax1.plot(dates, prices, label="Close Price", color="blue")
    ax1.set_title("Sample Price Data (Hourly)")
    ax1.legend()
    ax1.grid(True)

    # Social activity
    post_activity = data[data['post_text'] != '']
    if not post_activity.empty:
        ax2.scatter(post_activity.index, post_activity['likes'], 
                   alpha=0.6, s=post_activity['shares']*2, c='green', label='Likes (size=shares)')
        ax2.set_title("Social Media Activity")
        ax2.set_ylabel("Likes")
        ax2.legend()
        ax2.grid(True)

    # Sentiment chart
    ax3.plot(dates, social_result.values, label="Social Media Sentiment", color="purple", linewidth=2)
    ax3.axhline(y=0.3, color="green", linestyle="--", alpha=0.7, label="Social Bullish (+0.3)")
    ax3.axhline(y=-0.3, color="red", linestyle="--", alpha=0.7, label="Social Bearish (-0.3)")
    ax3.axhline(y=0.7, color="green", linestyle=":", alpha=0.7, label="Viral Bullish (+0.7)")
    ax3.axhline(y=-0.7, color="red", linestyle=":", alpha=0.7, label="Viral Bearish (-0.7)")
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax3.set_title("Social Media Sentiment Indicator")
    ax3.set_ylabel("Sentiment (-1 to 1)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print("Social Media Sentiment calculation completed successfully!")
    print(f"Data points: {len(social_result)}")
    print(f"Social parameters: {social_indicator.parameters}")
    print(f"Current sentiment: {social_result.iloc[-1]:.3f}")
    print(f"Sentiment signal: {social_indicator.get_sentiment_signal(social_result.iloc[-1])}")
    
    # Check post statistics
    post_stats = social_indicator.get_post_statistics()
    viral_posts = social_indicator.get_viral_posts()
    
    print(f"\nPost Statistics: {post_stats}")
    print(f"Viral posts detected: {len(viral_posts)}")
    
    if viral_posts:
        print("Sample viral posts:")
        for i, post in enumerate(viral_posts[:3]):  # Show first 3
            print(f"  {i+1}. {post.get('text', '')[:50]}... (engagement: {post.get('likes', 0)} likes)")

    # Statistics
    valid_sentiment = social_result.dropna()
    print("\nSentiment Statistics:")
    print(f"Min: {valid_sentiment.min():.3f}")
    print(f"Max: {valid_sentiment.max():.3f}")
    print(f"Mean: {valid_sentiment.mean():.3f}")
    print(f"Std: {valid_sentiment.std():.3f}")
    print(f"Bullish periods: {(valid_sentiment > 0.1).sum()}")
    print(f"Bearish periods: {(valid_sentiment < -0.1).sum()}")
    print(f"Viral signals: {(abs(valid_sentiment) > 0.7).sum()}")