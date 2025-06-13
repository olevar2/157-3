"""
Price Volume Divergence Indicator

Detects divergences between price movements and volume patterns to identify
potential trend reversals and continuation signals.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata, IndicatorValidationError


@dataclass
class PriceVolumeDivergenceResult:
    """Result structure for Price Volume Divergence analysis"""
    divergence_type: str  # "bullish", "bearish", "hidden_bullish", "hidden_bearish", "none"
    strength: float  # Divergence strength (0-1)
    duration: int  # Duration of divergence in periods
    price_trend: float  # Price trend strength
    volume_trend: float  # Volume trend strength
    confirmation_level: float  # Level of confirmation
    signal_quality: str  # "strong", "moderate", "weak"


class PriceVolumeDivergence(StandardIndicatorInterface):
    """
    Price Volume Divergence Detector
    
    Identifies divergences between price and volume:
    - Bullish divergence: Price falling, volume rising
    - Bearish divergence: Price rising, volume falling  
    - Hidden divergences for trend continuation
    - Strength and quality assessment
    """
    
    CATEGORY = "volume"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, lookback_period: int = 14, divergence_threshold: float = 0.3,
                 min_duration: int = 3, volume_smoothing: int = 5, **kwargs):
        """
        Initialize Price Volume Divergence indicator
        
        Args:
            lookback_period: Period for trend analysis
            divergence_threshold: Minimum threshold for divergence detection
            min_duration: Minimum duration for valid divergence
            volume_smoothing: Period for volume smoothing
        """
        self.lookback_period = lookback_period
        self.divergence_threshold = divergence_threshold
        self.min_duration = min_duration
        self.volume_smoothing = volume_smoothing
        super().__init__(**kwargs)
    
    def _setup_defaults(self):
        """Setup default parameters"""
        if not hasattr(self, 'lookback_period'):
            self.lookback_period = 14
        if not hasattr(self, 'divergence_threshold'):
            self.divergence_threshold = 0.3
        if not hasattr(self, 'min_duration'):
            self.min_duration = 3
        if not hasattr(self, 'volume_smoothing'):
            self.volume_smoothing = 5
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        if self.lookback_period < 5:
            raise IndicatorValidationError("Lookback period must be at least 5")
        if self.divergence_threshold <= 0 or self.divergence_threshold > 1:
            raise IndicatorValidationError("Divergence threshold must be between 0 and 1")
        if self.min_duration < 2:
            raise IndicatorValidationError("Minimum duration must be at least 2")
        if self.volume_smoothing < 1:
            raise IndicatorValidationError("Volume smoothing must be at least 1")
        return True
    
    def _get_required_columns(self) -> List[str]:
        """Required data columns"""
        return ["high", "low", "close", "volume"]
    
    def _get_minimum_data_points(self) -> int:
        """Minimum data points required"""
        return self.lookback_period + self.volume_smoothing
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price volume divergence signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with divergence analysis
        """
        try:
            self.validate_input_data(data)
            
            # Calculate price and volume trends
            price_trends = self._calculate_price_trends(data)
            volume_trends = self._calculate_volume_trends(data)
            
            # Detect divergences
            results = []
            for i in range(len(data)):
                if i < self.lookback_period:
                    result = {
                        'divergence_type': 'none',
                        'strength': 0.0,
                        'duration': 0,
                        'price_trend': 0.0,
                        'volume_trend': 0.0,
                        'confirmation_level': 0.0,
                        'signal_quality': 'weak'
                    }
                else:
                    # Analyze divergence at current position
                    divergence_analysis = self._analyze_divergence(
                        i, data, price_trends, volume_trends
                    )
                    
                    result = {
                        'divergence_type': divergence_analysis.divergence_type,
                        'strength': divergence_analysis.strength,
                        'duration': divergence_analysis.duration,
                        'price_trend': divergence_analysis.price_trend,
                        'volume_trend': divergence_analysis.volume_trend,
                        'confirmation_level': divergence_analysis.confirmation_level,
                        'signal_quality': divergence_analysis.signal_quality
                    }
                
                results.append(result)
            
            result_df = pd.DataFrame(results, index=data.index)
            self._last_calculation = result_df
            
            return result_df
            
        except Exception as e:
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _calculate_price_trends(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price trend indicators"""
        # Simple price trend using linear regression slope
        price_trend = data['close'].rolling(window=self.lookback_period).apply(
            lambda x: self._calculate_trend_slope(x.values)
        )
        
        # High/Low trends for swing analysis
        high_trend = data['high'].rolling(window=self.lookback_period).apply(
            lambda x: self._calculate_trend_slope(x.values)
        )
        
        low_trend = data['low'].rolling(window=self.lookback_period).apply(
            lambda x: self._calculate_trend_slope(x.values)
        )
        
        return pd.DataFrame({
            'price_trend': price_trend,
            'high_trend': high_trend,
            'low_trend': low_trend
        }).fillna(0)
    
    def _calculate_volume_trends(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume trend indicators"""
        # Smooth volume first
        volume_smooth = data['volume'].rolling(window=self.volume_smoothing).mean()
        
        # Volume trend
        volume_trend = volume_smooth.rolling(window=self.lookback_period).apply(
            lambda x: self._calculate_trend_slope(x.values)
        )
        
        # Volume momentum
        volume_momentum = volume_smooth.pct_change(periods=self.volume_smoothing)
        
        # Volume ratio vs average
        volume_avg = volume_smooth.rolling(window=self.lookback_period).mean()
        volume_ratio = volume_smooth / volume_avg - 1
        
        return pd.DataFrame({
            'volume_trend': volume_trend,
            'volume_momentum': volume_momentum,
            'volume_ratio': volume_ratio
        }).fillna(0)
    
    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        
        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < 3:
            return 0.0
        
        x_valid = x[valid_mask]
        y_valid = values[valid_mask]
        
        # Linear regression
        slope = np.polyfit(x_valid, y_valid, 1)[0]
        
        # Normalize slope relative to value range
        value_range = np.ptp(y_valid)
        if value_range > 0:
            normalized_slope = slope * len(x_valid) / value_range
        else:
            normalized_slope = 0.0
        
        return np.clip(normalized_slope, -2.0, 2.0)
    
    def _analyze_divergence(self, index: int, data: pd.DataFrame,
                          price_trends: pd.DataFrame, volume_trends: pd.DataFrame) -> PriceVolumeDivergenceResult:
        """Analyze divergence at specific index"""
        
        current_price_trend = price_trends['price_trend'].iloc[index]
        current_volume_trend = volume_trends['volume_trend'].iloc[index]
        
        # Detect divergence type
        divergence_type = self._detect_divergence_type(
            current_price_trend, current_volume_trend
        )
        
        # Calculate divergence strength
        strength = self._calculate_divergence_strength(
            current_price_trend, current_volume_trend
        )
        
        # Calculate duration of current divergence
        duration = self._calculate_divergence_duration(
            index, price_trends, volume_trends
        )
        
        # Calculate confirmation level
        confirmation_level = self._calculate_confirmation_level(
            index, data, price_trends, volume_trends, divergence_type
        )
        
        # Determine signal quality
        signal_quality = self._determine_signal_quality(
            strength, duration, confirmation_level
        )
        
        return PriceVolumeDivergenceResult(
            divergence_type=divergence_type,
            strength=strength,
            duration=duration,
            price_trend=current_price_trend,
            volume_trend=current_volume_trend,
            confirmation_level=confirmation_level,
            signal_quality=signal_quality
        )
    
    def _detect_divergence_type(self, price_trend: float, volume_trend: float) -> str:
        """Detect type of divergence"""
        
        # Check if trends are significant enough
        if abs(price_trend) < self.divergence_threshold or abs(volume_trend) < self.divergence_threshold:
            return 'none'
        
        # Bullish divergence: Price down, Volume up
        if price_trend < -self.divergence_threshold and volume_trend > self.divergence_threshold:
            return 'bullish'
        
        # Bearish divergence: Price up, Volume down
        elif price_trend > self.divergence_threshold and volume_trend < -self.divergence_threshold:
            return 'bearish'
        
        # Hidden bullish: Both trends up but volume stronger
        elif price_trend > 0 and volume_trend > 0 and volume_trend > price_trend * 1.5:
            return 'hidden_bullish'
        
        # Hidden bearish: Both trends down but volume weaker decline
        elif price_trend < 0 and volume_trend < 0 and abs(volume_trend) < abs(price_trend) * 0.5:
            return 'hidden_bearish'
        
        return 'none'
    
    def _calculate_divergence_strength(self, price_trend: float, volume_trend: float) -> float:
        """Calculate strength of divergence"""
        if abs(price_trend) < self.divergence_threshold or abs(volume_trend) < self.divergence_threshold:
            return 0.0
        
        # Strength based on magnitude and opposition of trends
        price_magnitude = abs(price_trend)
        volume_magnitude = abs(volume_trend)
        
        # For classic divergences, trends should be opposite
        if price_trend * volume_trend < 0:  # Opposite signs
            strength = (price_magnitude + volume_magnitude) / 2
        else:  # Same direction (hidden divergences)
            strength = abs(price_magnitude - volume_magnitude) / 2
        
        return min(strength, 1.0)
    
    def _calculate_divergence_duration(self, index: int, price_trends: pd.DataFrame,
                                     volume_trends: pd.DataFrame) -> int:
        """Calculate how long the current divergence has persisted"""
        if index < self.min_duration:
            return 0
        
        current_divergence = self._detect_divergence_type(
            price_trends['price_trend'].iloc[index],
            volume_trends['volume_trend'].iloc[index]
        )
        
        if current_divergence == 'none':
            return 0
        
        # Count backwards to find duration
        duration = 1
        for i in range(index - 1, max(0, index - self.lookback_period), -1):
            past_divergence = self._detect_divergence_type(
                price_trends['price_trend'].iloc[i],
                volume_trends['volume_trend'].iloc[i]
            )
            
            if past_divergence == current_divergence:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_confirmation_level(self, index: int, data: pd.DataFrame,
                                    price_trends: pd.DataFrame, volume_trends: pd.DataFrame,
                                    divergence_type: str) -> float:
        """Calculate confirmation level for the divergence"""
        if divergence_type == 'none':
            return 0.0
        
        confirmation = 0.0
        
        # Price action confirmation
        recent_price_action = data['close'].iloc[max(0, index-3):index+1]
        if len(recent_price_action) >= 2:
            price_direction = recent_price_action.iloc[-1] - recent_price_action.iloc[0]
            
            if divergence_type in ['bullish', 'hidden_bullish'] and price_direction > 0:
                confirmation += 0.4
            elif divergence_type in ['bearish', 'hidden_bearish'] and price_direction < 0:
                confirmation += 0.4
        
        # Volume confirmation
        recent_volume = volume_trends['volume_ratio'].iloc[max(0, index-3):index+1]
        if len(recent_volume) >= 2:
            volume_change = recent_volume.iloc[-1] - recent_volume.iloc[0]
            
            if divergence_type in ['bullish', 'hidden_bullish'] and volume_change > 0:
                confirmation += 0.3
            elif divergence_type in ['bearish', 'hidden_bearish'] and volume_change < 0:
                confirmation += 0.3
        
        # Trend strength confirmation
        trend_strength = abs(price_trends['price_trend'].iloc[index]) + abs(volume_trends['volume_trend'].iloc[index])
        confirmation += min(trend_strength / 4, 0.3)
        
        return min(confirmation, 1.0)
    
    def _determine_signal_quality(self, strength: float, duration: int, 
                                confirmation_level: float) -> str:
        """Determine overall signal quality"""
        
        # Calculate quality score
        quality_score = (
            strength * 0.4 +
            min(duration / 10, 1.0) * 0.3 +
            confirmation_level * 0.3
        )
        
        if quality_score >= 0.7:
            return 'strong'
        elif quality_score >= 0.4:
            return 'moderate'
        else:
            return 'weak'
    
    def get_metadata(self) -> IndicatorMetadata:
        """Get indicator metadata"""
        return IndicatorMetadata(
            name="Price Volume Divergence",
            category=self.CATEGORY,
            description="Detects divergences between price and volume for reversal signals",
            parameters={
                "lookback_period": self.lookback_period,
                "divergence_threshold": self.divergence_threshold,
                "min_duration": self.min_duration,
                "volume_smoothing": self.volume_smoothing
            },
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )
    
    def get_display_name(self) -> str:
        """Get display name for the indicator"""
        return f"Price Volume Divergence ({self.lookback_period})"
    
    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return {
            "lookback_period": self.lookback_period,
            "divergence_threshold": self.divergence_threshold,
            "min_duration": self.min_duration,
            "volume_smoothing": self.volume_smoothing
        }