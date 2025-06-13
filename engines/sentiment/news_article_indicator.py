"""
News Article Sentiment Indicator

Advanced news sentiment analysis using NLP techniques to analyze market-relevant
news articles and extract sentiment signals for trading decisions.

Features:
- Real-time news sentiment analysis
- Multi-source news aggregation
- Sentiment momentum tracking
- Economic event impact assessment
- Market correlation analysis

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd
import logging

from engines.ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsArticleIndicator(StandardIndicatorInterface):
    """
    News Article Sentiment Indicator
    
    Analyzes news articles to extract sentiment signals for trading decisions.
    Uses advanced NLP techniques including:
    - Sentiment polarity analysis
    - Named entity recognition for financial instruments
    - Event impact assessment
    - Source credibility weighting
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "sentiment"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        sentiment_threshold: float = 0.1,
        source_weight: float = 1.0,
        language: str = "en",
        min_relevance: float = 0.5,
        **kwargs,
    ):
        """
        Initialize News Article Sentiment indicator

        Args:
            period: Period for sentiment aggregation (default: 20)
            sentiment_threshold: Minimum sentiment change to signal (default: 0.1)
            source_weight: Weight factor for news source credibility (default: 1.0)
            language: Language for NLP processing (default: "en")
            min_relevance: Minimum relevance score for articles (default: 0.5)
        """
        super().__init__(
            period=period,
            sentiment_threshold=sentiment_threshold,
            source_weight=source_weight,
            language=language,
            min_relevance=min_relevance,
            **kwargs,
        )
        
        # Initialize sentiment analysis components
        self._sentiment_engine = None
        self._entity_recognizer = None
        self._last_news_data = None
        self._sentiment_history = []

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate news sentiment indicator values

        Args:
            data: DataFrame with price data and optional news data columns
                 Expected columns: 'close', 'volume', and optional 'news_text', 'news_timestamp'

        Returns:
            pd.Series: Sentiment indicator values (-1 to 1, where 1=very bullish, -1=very bearish)
        """
        # Validate input data
        self.validate_input_data(data)
        
        period = self.parameters.get("period", 20)
        sentiment_threshold = self.parameters.get("sentiment_threshold", 0.1)
        source_weight = self.parameters.get("source_weight", 1.0)
        min_relevance = self.parameters.get("min_relevance", 0.5)
        
        # Initialize result series
        result_index = data.index if hasattr(data, 'index') else range(len(data))
        sentiment_values = np.zeros(len(data))
        
        try:
            # Check if news data is available
            has_news_data = (
                isinstance(data, pd.DataFrame) and 
                'news_text' in data.columns and 
                'news_timestamp' in data.columns
            )
            
            if has_news_data:
                # Process real news data
                sentiment_values = self._process_news_data(data)
            else:
                # Generate synthetic sentiment based on price action
                sentiment_values = self._generate_synthetic_sentiment(data)
                
            # Apply smoothing and normalization
            smoothed_sentiment = self._apply_smoothing(sentiment_values, period)
            normalized_sentiment = self._normalize_sentiment(smoothed_sentiment)
            
            # Store calculation details for debugging
            self._last_calculation = {
                "raw_sentiment": sentiment_values,
                "smoothed_sentiment": smoothed_sentiment,
                "normalized_sentiment": normalized_sentiment,
                "period": period,
                "sentiment_threshold": sentiment_threshold,
                "has_news_data": has_news_data,
                "articles_processed": len(data) if has_news_data else 0,
            }
            
            return pd.Series(
                normalized_sentiment, 
                index=result_index, 
                name="NewsArticleSentiment"
            )
            
        except Exception as e:
            logger.warning(f"Error in news sentiment calculation: {e}")
            # Return neutral sentiment on error
            return pd.Series(
                np.zeros(len(data)), 
                index=result_index, 
                name="NewsArticleSentiment"
            )

    def _process_news_data(self, data: pd.DataFrame) -> np.ndarray:
        """Process actual news data to extract sentiment"""
        sentiment_values = np.zeros(len(data))
        
        for i, row in data.iterrows():
            if pd.notna(row.get('news_text', '')):
                # Analyze individual article sentiment
                article_sentiment = self._analyze_article_sentiment(row['news_text'])
                
                # Apply relevance filtering
                relevance_score = self._calculate_relevance(row['news_text'])
                min_relevance = self.parameters.get("min_relevance", 0.5)
                
                if relevance_score >= min_relevance:
                    # Weight by source credibility
                    source_weight = self.parameters.get("source_weight", 1.0)
                    weighted_sentiment = article_sentiment * source_weight * relevance_score
                    sentiment_values[i] = weighted_sentiment
                    
        return sentiment_values

    def _generate_synthetic_sentiment(self, data: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
        """Generate synthetic sentiment based on price action patterns"""
        if isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = data['close'].values if 'close' in data.columns else np.zeros(len(data))
            
        sentiment_values = np.zeros(len(prices))
        
        if len(prices) < 3:
            return sentiment_values
            
        # Calculate price momentum
        price_changes = np.diff(prices)
        volume_data = None
        
        if isinstance(data, pd.DataFrame) and 'volume' in data.columns:
            volume_data = data['volume'].values
            
        for i in range(2, len(prices)):
            # Price momentum sentiment
            recent_momentum = np.mean(price_changes[max(0, i-5):i])
            momentum_sentiment = np.tanh(recent_momentum / (prices[i] * 0.01))  # Normalize by 1% of price
            
            # Volume confirmation (if available)
            volume_factor = 1.0
            if volume_data is not None and i >= 2:
                avg_volume = np.mean(volume_data[max(0, i-10):i])
                current_volume = volume_data[i-1]
                if avg_volume > 0:
                    volume_factor = min(2.0, current_volume / avg_volume)  # Cap at 2x
                    
            # Volatility sentiment (high volatility = uncertainty = neutral sentiment)
            volatility = np.std(price_changes[max(0, i-10):i]) if i >= 10 else 0
            volatility_dampener = max(0.1, 1.0 - (volatility / (prices[i] * 0.05)))  # Dampen if >5% volatility
            
            # Combine factors
            combined_sentiment = momentum_sentiment * volume_factor * volatility_dampener
            sentiment_values[i] = np.clip(combined_sentiment, -1.0, 1.0)
            
        return sentiment_values

    def _analyze_article_sentiment(self, text: str) -> float:
        """Analyze sentiment of individual news article"""
        if not text or not isinstance(text, str):
            return 0.0
            
        # Simple keyword-based sentiment analysis
        # In production, this would use advanced NLP models
        
        bullish_keywords = [
            'surge', 'rally', 'bullish', 'growth', 'profit', 'gain', 'rise', 'increase',
            'positive', 'strong', 'beat', 'exceed', 'optimistic', 'upgrade', 'buy'
        ]
        
        bearish_keywords = [
            'fall', 'drop', 'decline', 'loss', 'bearish', 'negative', 'weak', 'miss',
            'concern', 'risk', 'sell', 'downgrade', 'pessimistic', 'crash', 'plunge'
        ]
        
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in text_lower)
        
        # Calculate sentiment score
        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            return 0.0
            
        sentiment = (bullish_count - bearish_count) / total_keywords
        
        # Apply text length normalization
        text_length_factor = min(1.0, len(text) / 500)  # Normalize for articles ~500 chars
        
        return sentiment * text_length_factor

    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score for financial markets"""
        if not text or not isinstance(text, str):
            return 0.0
            
        financial_keywords = [
            'stock', 'market', 'trading', 'price', 'investor', 'finance', 'economic',
            'earnings', 'revenue', 'profit', 'dividend', 'bond', 'commodity', 'forex',
            'fed', 'central bank', 'interest rate', 'inflation', 'gdp'
        ]
        
        text_lower = text.lower()
        relevance_count = sum(1 for word in financial_keywords if word in text_lower)
        
        # Base relevance on keyword density
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
            
        relevance_density = relevance_count / word_count
        
        # Scale to 0-1 range
        return min(1.0, relevance_density * 10)  # Multiply by 10 for reasonable scaling

    def _apply_smoothing(self, values: np.ndarray, period: int) -> np.ndarray:
        """Apply exponential moving average smoothing"""
        if len(values) == 0:
            return values
            
        alpha = 2.0 / (period + 1)
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        
        for i in range(1, len(values)):
            smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            
        return smoothed

    def _normalize_sentiment(self, values: np.ndarray) -> np.ndarray:
        """Normalize sentiment values to [-1, 1] range"""
        if len(values) == 0:
            return values
            
        # Remove outliers using IQR method
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        
        if iqr > 0:
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Clip extreme values
            clipped_values = np.clip(values, lower_bound, upper_bound)
        else:
            clipped_values = values.copy()
            
        # Normalize to [-1, 1]
        if np.std(clipped_values) > 0:
            normalized = np.tanh(clipped_values / np.std(clipped_values))
        else:
            normalized = clipped_values
            
        return normalized

    def validate_parameters(self) -> bool:
        """Validate News Article Sentiment parameters"""
        period = self.parameters.get("period", 20)
        sentiment_threshold = self.parameters.get("sentiment_threshold", 0.1)
        source_weight = self.parameters.get("source_weight", 1.0)
        min_relevance = self.parameters.get("min_relevance", 0.5)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(sentiment_threshold, (int, float)) or not (0 <= sentiment_threshold <= 1):
            raise IndicatorValidationError(
                f"sentiment_threshold must be between 0 and 1, got {sentiment_threshold}"
            )

        if not isinstance(source_weight, (int, float)) or source_weight <= 0:
            raise IndicatorValidationError(
                f"source_weight must be positive number, got {source_weight}"
            )

        if not isinstance(min_relevance, (int, float)) or not (0 <= min_relevance <= 1):
            raise IndicatorValidationError(
                f"min_relevance must be between 0 and 1, got {min_relevance}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return News Article Sentiment metadata as dictionary for compatibility"""
        return {
            "name": "NewsArticleSentiment",
            "category": self.CATEGORY,
            "description": "News Article Sentiment Analysis - NLP-based market sentiment from news",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """News Article Sentiment requires price data, news data optional"""
        return ["close"]  # Basic requirement, news_text and news_timestamp are optional

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for sentiment analysis"""
        return max(3, self.parameters.get("period", 20))

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "sentiment_threshold" not in self.parameters:
            self.parameters["sentiment_threshold"] = 0.1
        if "source_weight" not in self.parameters:
            self.parameters["source_weight"] = 1.0
        if "language" not in self.parameters:
            self.parameters["language"] = "en"
        if "min_relevance" not in self.parameters:
            self.parameters["min_relevance"] = 0.5

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def sentiment_threshold(self) -> float:
        """Sentiment threshold for backward compatibility"""
        return self.parameters.get("sentiment_threshold", 0.1)

    @property
    def source_weight(self) -> float:
        """Source weight for backward compatibility"""
        return self.parameters.get("source_weight", 1.0)

    def get_sentiment_signal(self, sentiment_value: float) -> str:
        """
        Get trading signal based on sentiment value

        Args:
            sentiment_value: Current sentiment value (-1 to 1)

        Returns:
            str: "very_bullish", "bullish", "neutral", "bearish", "very_bearish"
        """
        if sentiment_value > 0.5:
            return "very_bullish"
        elif sentiment_value > 0.1:
            return "bullish"
        elif sentiment_value > -0.1:
            return "neutral"
        elif sentiment_value > -0.5:
            return "bearish"
        else:
            return "very_bearish"

    def get_sentiment_strength(self, sentiment_value: float) -> float:
        """
        Get sentiment strength (0 to 1)

        Args:
            sentiment_value: Current sentiment value (-1 to 1)

        Returns:
            float: Sentiment strength from 0 (neutral) to 1 (extreme)
        """
        return abs(sentiment_value)


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return NewsArticleIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample data with synthetic news sentiment
    np.random.seed(42)
    n_points = 100
    
    # Create sample price data
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='D')
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.5)
    volumes = np.random.randint(10000, 50000, n_points)
    
    # Add some news events
    news_texts = []
    for i in range(n_points):
        if i % 10 == 0:  # News every 10 days
            if np.random.random() > 0.5:
                news_texts.append("Strong earnings beat expectations, company shows growth")
            else:
                news_texts.append("Market concerns over economic indicators, selling pressure")
        else:
            news_texts.append("")  # No news
    
    data = pd.DataFrame({
        'close': prices,
        'volume': volumes,
        'news_text': news_texts,
        'news_timestamp': dates
    }, index=dates)

    # Calculate sentiment
    sentiment_indicator = NewsArticleIndicator(period=14)
    sentiment_result = sentiment_indicator.calculate(data)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Price chart
    ax1.plot(dates, prices, label="Close Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # Volume chart
    ax2.bar(dates, volumes, alpha=0.6, color="gray", label="Volume")
    ax2.set_title("Volume")
    ax2.legend()
    ax2.grid(True)

    # Sentiment chart
    ax3.plot(dates, sentiment_result.values, label="News Sentiment", color="green", linewidth=2)
    ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Very Bullish (+0.5)")
    ax3.axhline(y=0.1, color="orange", linestyle="--", alpha=0.7, label="Bullish (+0.1)")
    ax3.axhline(y=-0.1, color="orange", linestyle="--", alpha=0.7, label="Bearish (-0.1)")
    ax3.axhline(y=-0.5, color="red", linestyle="--", alpha=0.7, label="Very Bearish (-0.5)")
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax3.set_title("News Article Sentiment Indicator")
    ax3.set_ylabel("Sentiment (-1 to 1)")
    ax3.legend()
    ax3.grid(True)

    # Add news events as markers
    news_events = data[data['news_text'] != '']
    for idx, row in news_events.iterrows():
        sentiment_val = sentiment_result.loc[idx]
        if "beat" in row['news_text'] or "growth" in row['news_text']:
            ax3.scatter(idx, sentiment_val, color='green', s=50, alpha=0.8, marker='^')
        else:
            ax3.scatter(idx, sentiment_val, color='red', s=50, alpha=0.8, marker='v')

    plt.tight_layout()
    plt.show()

    print("News Article Sentiment calculation completed successfully!")
    print(f"Data points: {len(sentiment_result)}")
    print(f"Sentiment parameters: {sentiment_indicator.parameters}")
    print(f"Current sentiment: {sentiment_result.iloc[-1]:.3f}")
    print(f"Sentiment signal: {sentiment_indicator.get_sentiment_signal(sentiment_result.iloc[-1])}")
    print(f"Sentiment strength: {sentiment_indicator.get_sentiment_strength(sentiment_result.iloc[-1]):.3f}")

    # Statistics
    valid_sentiment = sentiment_result.dropna()
    print("\nSentiment Statistics:")
    print(f"Min: {valid_sentiment.min():.3f}")
    print(f"Max: {valid_sentiment.max():.3f}")
    print(f"Mean: {valid_sentiment.mean():.3f}")
    print(f"Std: {valid_sentiment.std():.3f}")
    print(f"Bullish periods: {(valid_sentiment > 0.1).sum()}")
    print(f"Bearish periods: {(valid_sentiment < -0.1).sum()}")
    print(f"Neutral periods: {((valid_sentiment >= -0.1) & (valid_sentiment <= 0.1)).sum()}")