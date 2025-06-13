"""
Sentiment Configuration Indicator

Advanced sentiment configuration and management system for multi-source
sentiment analysis in trading applications.

Features:
- Dynamic sentiment source weighting
- Real-time sentiment calibration
- Multi-timeframe sentiment aggregation
- Sentiment confidence scoring
- Source reliability tracking

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from engines.ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentConfigIndicator(StandardIndicatorInterface):
    """
    Sentiment Configuration Indicator
    
    Manages and configures multiple sentiment analysis sources to provide
    optimized sentiment signals for trading decisions.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "sentiment"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        confidence_threshold: float = 0.6,
        source_count: int = 3,
        weight_decay: float = 0.95,
        calibration_window: int = 100,
        **kwargs,
    ):
        """
        Initialize Sentiment Configuration indicator

        Args:
            period: Period for sentiment aggregation (default: 20)
            confidence_threshold: Minimum confidence for signals (default: 0.6)
            source_count: Number of sentiment sources to manage (default: 3)
            weight_decay: Decay factor for source weights (default: 0.95)
            calibration_window: Window for sentiment calibration (default: 100)
        """
        super().__init__(
            period=period,
            confidence_threshold=confidence_threshold,
            source_count=source_count,
            weight_decay=weight_decay,
            calibration_window=calibration_window,
            **kwargs,
        )
        
        # Initialize configuration components
        self._source_weights = {}
        self._source_reliability = {}
        self._calibration_data = []
        self._sentiment_history = []

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate sentiment configuration values

        Args:
            data: DataFrame with sentiment data from multiple sources
                 Expected columns: 'close', and optional sentiment columns like 
                 'news_sentiment', 'social_sentiment', 'technical_sentiment'

        Returns:
            pd.Series: Configured sentiment values with confidence weighting
        """
        # Validate input data
        self.validate_input_data(data)
        
        period = self.parameters.get("period", 20)
        confidence_threshold = self.parameters.get("confidence_threshold", 0.6)
        source_count = self.parameters.get("source_count", 3)
        weight_decay = self.parameters.get("weight_decay", 0.95)
        calibration_window = self.parameters.get("calibration_window", 100)
        
        # Initialize result series
        result_index = data.index if hasattr(data, 'index') else range(len(data))
        config_values = np.zeros(len(data))
        
        try:
            # Identify available sentiment sources
            sentiment_sources = self._identify_sentiment_sources(data)
            
            if sentiment_sources:
                # Process multi-source sentiment data
                config_values = self._process_multi_source_sentiment(
                    data, sentiment_sources, period, confidence_threshold
                )
            else:
                # Generate synthetic sentiment configuration
                config_values = self._generate_synthetic_config(data, period)
                
            # Apply dynamic weighting and calibration
            calibrated_values = self._apply_calibration(config_values, calibration_window)
            final_values = self._apply_confidence_filtering(calibrated_values, confidence_threshold)
            
            # Store calculation details for debugging
            self._last_calculation = {
                "raw_config": config_values,
                "calibrated_values": calibrated_values,
                "final_values": final_values,
                "sentiment_sources": sentiment_sources,
                "source_weights": self._source_weights.copy(),
                "source_reliability": self._source_reliability.copy(),
                "period": period,
                "confidence_threshold": confidence_threshold,
            }
            
            return pd.Series(
                final_values, 
                index=result_index, 
                name="SentimentConfig"
            )
            
        except Exception as e:
            logger.warning(f"Error in sentiment configuration calculation: {e}")
            # Return neutral configuration on error
            return pd.Series(
                np.zeros(len(data)), 
                index=result_index, 
                name="SentimentConfig"
            )

    def _identify_sentiment_sources(self, data: Union[pd.DataFrame, pd.Series]) -> List[str]:
        """Identify available sentiment data sources"""
        if not isinstance(data, pd.DataFrame):
            return []
            
        potential_sources = [
            'news_sentiment', 'social_sentiment', 'technical_sentiment',
            'economic_sentiment', 'analyst_sentiment', 'options_sentiment',
            'insider_sentiment', 'institutional_sentiment'
        ]
        
        available_sources = [source for source in potential_sources if source in data.columns]
        return available_sources

    def _process_multi_source_sentiment(
        self, 
        data: pd.DataFrame, 
        sources: List[str], 
        period: int, 
        confidence_threshold: float
    ) -> np.ndarray:
        """Process sentiment data from multiple sources"""
        config_values = np.zeros(len(data))
        
        # Initialize source weights if not set
        if not self._source_weights:
            self._initialize_source_weights(sources)
            
        weight_decay = self.parameters.get("weight_decay", 0.95)
        
        for i in range(len(data)):
            weighted_sentiment = 0.0
            total_weight = 0.0
            confidence_sum = 0.0
            
            # Process each sentiment source
            for source in sources:
                if pd.notna(data[source].iloc[i]):
                    source_value = data[source].iloc[i]
                    source_weight = self._source_weights.get(source, 1.0)
                    source_reliability = self._source_reliability.get(source, 1.0)
                    
                    # Calculate source confidence
                    source_confidence = self._calculate_source_confidence(
                        source, data, i, period
                    )
                    
                    if source_confidence >= confidence_threshold:
                        # Weight by reliability and confidence
                        effective_weight = source_weight * source_reliability * source_confidence
                        weighted_sentiment += source_value * effective_weight
                        total_weight += effective_weight
                        confidence_sum += source_confidence
                        
                        # Update source reliability
                        self._update_source_reliability(source, source_confidence)
                        
            # Calculate final configured value
            if total_weight > 0:
                config_values[i] = weighted_sentiment / total_weight
                
                # Apply confidence scaling
                avg_confidence = confidence_sum / len(sources) if sources else 0
                config_values[i] *= avg_confidence
                
            # Decay weights over time
            self._decay_source_weights(weight_decay)
            
        return config_values

    def _generate_synthetic_config(self, data: Union[pd.DataFrame, pd.Series], period: int) -> np.ndarray:
        """Generate synthetic sentiment configuration from price data"""
        if isinstance(data, pd.Series):
            prices = data.values
        else:
            prices = data['close'].values if 'close' in data.columns else np.zeros(len(data))
            
        config_values = np.zeros(len(prices))
        
        if len(prices) < period:
            return config_values
            
        for i in range(period, len(prices)):
            # Technical sentiment from price action
            price_window = prices[i-period:i]
            price_momentum = (prices[i-1] - price_window[0]) / price_window[0] if price_window[0] != 0 else 0
            
            # Volatility-adjusted sentiment
            volatility = np.std(np.diff(price_window)) / np.mean(price_window) if np.mean(price_window) != 0 else 0
            volatility_factor = max(0.1, 1.0 - volatility * 5)  # Reduce confidence with high volatility
            
            # Trend consistency sentiment
            price_changes = np.diff(price_window)
            up_days = np.sum(price_changes > 0)
            trend_consistency = abs(up_days - len(price_changes)/2) / (len(price_changes)/2)
            
            # Combined synthetic sentiment
            synthetic_sentiment = (
                price_momentum * 0.5 +  # Price momentum
                trend_consistency * np.sign(price_momentum) * 0.3 +  # Trend consistency
                np.random.normal(0, 0.1)  # Random noise
            )
            
            config_values[i] = np.tanh(synthetic_sentiment) * volatility_factor
            
        return config_values

    def _initialize_source_weights(self, sources: List[str]):
        """Initialize weights for sentiment sources"""
        # Default weights based on typical source reliability
        default_weights = {
            'news_sentiment': 1.0,
            'social_sentiment': 0.7,
            'technical_sentiment': 0.9,
            'economic_sentiment': 1.2,
            'analyst_sentiment': 0.8,
            'options_sentiment': 1.1,
            'insider_sentiment': 1.3,
            'institutional_sentiment': 1.5
        }
        
        for source in sources:
            self._source_weights[source] = default_weights.get(source, 1.0)
            self._source_reliability[source] = 1.0

    def _calculate_source_confidence(self, source: str, data: pd.DataFrame, index: int, period: int) -> float:
        """Calculate confidence score for a sentiment source"""
        if index < period:
            return 0.5  # Low confidence for insufficient data
            
        # Historical accuracy of source
        historical_window = max(0, index - period)
        source_values = data[source].iloc[historical_window:index]
        
        if len(source_values) == 0:
            return 0.5
            
        # Confidence based on consistency and non-extreme values
        consistency = 1.0 - np.std(source_values.dropna()) if len(source_values.dropna()) > 1 else 0.5
        
        # Penalize extreme values (potential outliers)
        recent_value = data[source].iloc[index] if pd.notna(data[source].iloc[index]) else 0
        extremeness_penalty = max(0, abs(recent_value) - 1.0) * 0.5
        
        confidence = max(0.1, min(1.0, consistency - extremeness_penalty))
        return confidence

    def _update_source_reliability(self, source: str, confidence: float):
        """Update reliability score for a sentiment source"""
        if source not in self._source_reliability:
            self._source_reliability[source] = confidence
        else:
            # Exponential moving average update
            alpha = 0.1
            self._source_reliability[source] = (
                alpha * confidence + (1 - alpha) * self._source_reliability[source]
            )

    def _decay_source_weights(self, decay_factor: float):
        """Apply time decay to source weights"""
        for source in self._source_weights:
            # Weights decay towards 1.0 (neutral)
            self._source_weights[source] = (
                decay_factor * self._source_weights[source] + (1 - decay_factor) * 1.0
            )

    def _apply_calibration(self, values: np.ndarray, calibration_window: int) -> np.ndarray:
        """Apply rolling calibration to sentiment values"""
        if len(values) < calibration_window:
            return values
            
        calibrated = np.zeros_like(values)
        
        for i in range(calibration_window, len(values)):
            # Get calibration window
            window = values[i-calibration_window:i]
            
            # Calculate calibration parameters
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            if window_std > 0:
                # Z-score normalization then tanh scaling
                calibrated[i] = np.tanh((values[i] - window_mean) / window_std)
            else:
                calibrated[i] = values[i]
                
        # Handle initial values
        calibrated[:calibration_window] = values[:calibration_window]
        
        return calibrated

    def _apply_confidence_filtering(self, values: np.ndarray, threshold: float) -> np.ndarray:
        """Apply confidence-based filtering to values"""
        filtered = values.copy()
        
        # Calculate rolling confidence based on consistency
        window_size = min(20, len(values) // 4)
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            consistency = 1.0 - np.std(window) if len(window) > 1 else 0.0
            
            if consistency < threshold:
                # Reduce signal strength for low confidence
                filtered[i] *= consistency / threshold
                
        return filtered

    def validate_parameters(self) -> bool:
        """Validate Sentiment Configuration parameters"""
        period = self.parameters.get("period", 20)
        confidence_threshold = self.parameters.get("confidence_threshold", 0.6)
        source_count = self.parameters.get("source_count", 3)
        weight_decay = self.parameters.get("weight_decay", 0.95)
        calibration_window = self.parameters.get("calibration_window", 100)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(confidence_threshold, (int, float)) or not (0 <= confidence_threshold <= 1):
            raise IndicatorValidationError(
                f"confidence_threshold must be between 0 and 1, got {confidence_threshold}"
            )

        if not isinstance(source_count, int) or source_count < 1:
            raise IndicatorValidationError(
                f"source_count must be positive integer, got {source_count}"
            )

        if not isinstance(weight_decay, (int, float)) or not (0 < weight_decay <= 1):
            raise IndicatorValidationError(
                f"weight_decay must be between 0 and 1, got {weight_decay}"
            )

        if not isinstance(calibration_window, int) or calibration_window < period:
            raise IndicatorValidationError(
                f"calibration_window must be >= period, got {calibration_window} vs {period}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Sentiment Configuration metadata as dictionary for compatibility"""
        return {
            "name": "SentimentConfig",
            "category": self.CATEGORY,
            "description": "Sentiment Configuration - Multi-source sentiment analysis manager",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Sentiment Configuration requires price data, sentiment sources optional"""
        return ["close"]  # Basic requirement, sentiment columns are optional

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for configuration"""
        return max(self.parameters.get("period", 20), self.parameters.get("calibration_window", 100))

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "confidence_threshold" not in self.parameters:
            self.parameters["confidence_threshold"] = 0.6
        if "source_count" not in self.parameters:
            self.parameters["source_count"] = 3
        if "weight_decay" not in self.parameters:
            self.parameters["weight_decay"] = 0.95
        if "calibration_window" not in self.parameters:
            self.parameters["calibration_window"] = 100

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def confidence_threshold(self) -> float:
        """Confidence threshold for backward compatibility"""
        return self.parameters.get("confidence_threshold", 0.6)

    @property
    def source_count(self) -> int:
        """Source count for backward compatibility"""
        return self.parameters.get("source_count", 3)

    def get_source_weights(self) -> Dict[str, float]:
        """Get current source weights"""
        return self._source_weights.copy()

    def get_source_reliability(self) -> Dict[str, float]:
        """Get current source reliability scores"""
        return self._source_reliability.copy()

    def set_source_weight(self, source: str, weight: float):
        """Manually set weight for a sentiment source"""
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"Weight must be non-negative number, got {weight}")
        self._source_weights[source] = weight

    def get_confidence_signal(self, config_value: float, confidence: float) -> str:
        """
        Get trading signal based on configuration value and confidence

        Args:
            config_value: Current configuration value
            confidence: Current confidence score

        Returns:
            str: "high_confidence_bullish", "low_confidence_bullish", etc.
        """
        confidence_threshold = self.parameters.get("confidence_threshold", 0.6)
        
        high_confidence = confidence >= confidence_threshold
        confidence_prefix = "high_confidence" if high_confidence else "low_confidence"
        
        if config_value > 0.3:
            return f"{confidence_prefix}_bullish"
        elif config_value > 0.1:
            return f"{confidence_prefix}_slightly_bullish"
        elif config_value > -0.1:
            return f"{confidence_prefix}_neutral"
        elif config_value > -0.3:
            return f"{confidence_prefix}_slightly_bearish"
        else:
            return f"{confidence_prefix}_bearish"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return SentimentConfigIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample data with multiple sentiment sources
    np.random.seed(42)
    n_points = 150
    
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='D')
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.5)
    
    # Generate synthetic sentiment sources
    news_sentiment = np.random.randn(n_points) * 0.3
    social_sentiment = np.random.randn(n_points) * 0.4 + 0.1  # Slightly positive bias
    technical_sentiment = np.tanh(np.diff(prices, prepend=prices[0]) / prices * 10)  # Price-based
    
    data = pd.DataFrame({
        'close': prices,
        'news_sentiment': news_sentiment,
        'social_sentiment': social_sentiment,
        'technical_sentiment': technical_sentiment
    }, index=dates)

    # Calculate sentiment configuration
    config_indicator = SentimentConfigIndicator(period=20, confidence_threshold=0.6)
    config_result = config_indicator.calculate(data)

    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    # Price chart
    axes[0].plot(dates, prices, label="Close Price", color="blue")
    axes[0].set_title("Sample Price Data")
    axes[0].legend()
    axes[0].grid(True)

    # Individual sentiment sources
    axes[1].plot(dates, news_sentiment, label="News Sentiment", alpha=0.7)
    axes[1].plot(dates, social_sentiment, label="Social Sentiment", alpha=0.7)
    axes[1].plot(dates, technical_sentiment, label="Technical Sentiment", alpha=0.7)
    axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1].set_title("Individual Sentiment Sources")
    axes[1].legend()
    axes[1].grid(True)

    # Configured sentiment
    axes[2].plot(dates, config_result.values, label="Configured Sentiment", color="purple", linewidth=2)
    axes[2].axhline(y=0.3, color="green", linestyle="--", alpha=0.7, label="Bullish (+0.3)")
    axes[2].axhline(y=-0.3, color="red", linestyle="--", alpha=0.7, label="Bearish (-0.3)")
    axes[2].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[2].set_title("Sentiment Configuration Indicator")
    axes[2].legend()
    axes[2].grid(True)

    # Source weights over time (if available)
    source_weights = config_indicator.get_source_weights()
    if source_weights:
        weight_data = []
        sources = list(source_weights.keys())
        for source in sources:
            weight_data.append([source_weights[source]] * n_points)  # Simplified for demo
            
        for i, source in enumerate(sources):
            axes[3].plot(dates, weight_data[i], label=f"{source.replace('_', ' ').title()} Weight", alpha=0.7)
        axes[3].set_title("Source Weights")
        axes[3].legend()
        axes[3].grid(True)
    else:
        axes[3].text(0.5, 0.5, 'No source weights available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[3].transAxes)

    plt.tight_layout()
    plt.show()

    print("Sentiment Configuration calculation completed successfully!")
    print(f"Data points: {len(config_result)}")
    print(f"Config parameters: {config_indicator.parameters}")
    print(f"Current config value: {config_result.iloc[-1]:.3f}")
    
    # Check source information
    source_weights = config_indicator.get_source_weights()
    source_reliability = config_indicator.get_source_reliability()
    
    print(f"\nSource Weights: {source_weights}")
    print(f"Source Reliability: {source_reliability}")
    
    if config_indicator._last_calculation:
        sources = config_indicator._last_calculation.get('sentiment_sources', [])
        print(f"Active sentiment sources: {sources}")

    # Statistics
    valid_config = config_result.dropna()
    print("\nConfiguration Statistics:")
    print(f"Min: {valid_config.min():.3f}")
    print(f"Max: {valid_config.max():.3f}")
    print(f"Mean: {valid_config.mean():.3f}")
    print(f"Std: {valid_config.std():.3f}")
    print(f"Bullish periods: {(valid_config > 0.1).sum()}")
    print(f"Bearish periods: {(valid_config < -0.1).sum()}")
    print(f"Neutral periods: {((valid_config >= -0.1) & (valid_config <= 0.1)).sum()}")