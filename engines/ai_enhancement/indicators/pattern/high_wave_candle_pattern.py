"""
High Wave Candle Pattern Indicator

The High Wave Candle Pattern is an extreme form of spinning top:
- Very small body relative to the total range
- Extremely long upper and lower shadows
- Indicates extreme market indecision
- Often appears at major reversal points
- More significant than regular spinning tops

This indicator follows the CCI (Commodity Channel Index) gold standard template.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add the parent directory to the path to import base_indicator
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_indicator import StandardIndicatorInterface


class HighWaveCandlePattern(StandardIndicatorInterface):
    """
    High Wave Candle Pattern Indicator
    
    Identifies high wave candle patterns which indicate extreme market
    indecision and potential major reversal points.
    """
    
    def __init__(self, 
                 max_body_ratio: float = 0.15,
                 min_shadow_ratio: float = 3.0,
                 min_total_range_ratio: float = 0.02,
                 trend_lookback: int = 10,
                 trend_threshold: float = 0.02):
        """
        Initialize the High Wave Candle Pattern indicator.
        
        Args:
            max_body_ratio: Maximum body to total range ratio (default: 0.15)
            min_shadow_ratio: Minimum shadow to body ratio (default: 3.0)
            min_total_range_ratio: Minimum total range to price ratio (default: 2%)
            trend_lookback: Number of periods to determine trend (default: 10)
            trend_threshold: Minimum price change % to identify trend (default: 2%)
        """
        self.max_body_ratio = max_body_ratio
        self.min_shadow_ratio = min_shadow_ratio
        self.min_total_range_ratio = min_total_range_ratio
        self.trend_lookback = trend_lookback
        self.trend_threshold = trend_threshold
        
        # Parameters dict for base class
        self.parameters = {
            'max_body_ratio': max_body_ratio,
            'min_shadow_ratio': min_shadow_ratio,
            'min_total_range_ratio': min_total_range_ratio,
            'trend_lookback': trend_lookback,
            'trend_threshold': trend_threshold
        }
        
        # Initialize result storage
        self.high_wave_signals = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="High Wave Candle Pattern",
            category="pattern",
            description="Extreme indecision pattern with tiny body and very long shadows",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=self.trend_lookback + 1
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 0.05 <= self.max_body_ratio <= 0.3:
            raise ValueError("max_body_ratio must be between 0.05 and 0.3")
        if not 1.0 <= self.min_shadow_ratio <= 10.0:
            raise ValueError("min_shadow_ratio must be between 1.0 and 10.0")
        if not 0.005 <= self.min_total_range_ratio <= 0.1:
            raise ValueError("min_total_range_ratio must be between 0.5% and 10%")
        if not 3 <= self.trend_lookback <= 50:
            raise ValueError("trend_lookback must be between 3 and 50")
        if not 0.005 <= self.trend_threshold <= 0.1:
            raise ValueError("trend_threshold must be between 0.5% and 10%")
        return True
        
    def _get_candle_metrics(self, open_price: float, high: float, 
                           low: float, close: float) -> Dict[str, float]:
        """
        Calculate candle metrics.
        
        Args:
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            
        Returns:
            Dictionary with candle metrics
        """
        body = abs(close - open_price)
        total_range = high - low
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        return {
            'body': body,
            'total_range': total_range,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'body_ratio': body / total_range if total_range > 0 else 0,
            'upper_shadow_ratio': upper_shadow / body if body > 0 else float('inf'),
            'lower_shadow_ratio': lower_shadow / body if body > 0 else float('inf'),
            'total_range_ratio': total_range / close if close > 0 else 0,
            'is_bullish': close > open_price,
            'is_bearish': close < open_price
        }
    
    def _is_high_wave_candle(self, metrics: Dict[str, float]) -> bool:
        """
        Check if candle is a high wave candle.
        
        Args:
            metrics: Candle metrics
            
        Returns:
            True if candle is a high wave candle
        """
        # Must have very small body
        if metrics['body_ratio'] > self.max_body_ratio:
            return False
        
        # Must have very long shadows (both upper and lower)
        if (metrics['upper_shadow_ratio'] < self.min_shadow_ratio or
            metrics['lower_shadow_ratio'] < self.min_shadow_ratio):
            return False
        
        # Must have meaningful total range
        if metrics['total_range_ratio'] < self.min_total_range_ratio:
            return False
        
        # Total range should be meaningful
        if metrics['total_range'] <= 0:
            return False
        
        return True
    
    def _determine_trend(self, prices: List[float], current_idx: int) -> str:
        """
        Determine the trend preceding the current candle.
        
        Args:
            prices: List of prices (typically closing prices)
            current_idx: Current index
            
        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        start_idx = max(0, current_idx - self.trend_lookback)
        
        if start_idx >= current_idx:
            return 'sideways'
        
        start_price = prices[start_idx]
        end_price = prices[current_idx - 1]  # Price before current candle
        
        if start_price == 0:
            return 'sideways'
        
        price_change = (end_price - start_price) / start_price
        
        if price_change >= self.trend_threshold:
            return 'uptrend'
        elif price_change <= -self.trend_threshold:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_pattern_strength(self, 
                                  metrics: Dict[str, float],
                                  trend: str) -> float:
        """
        Calculate the strength of the high wave candle pattern.
        
        Args:
            metrics: Candle metrics
            trend: Preceding trend
            
        Returns:
            Pattern strength score (0-100)
        """
        strength = 60  # Higher base strength due to rarity
        
        # Strength based on tiny body (smaller is better)
        body_strength = (self.max_body_ratio - metrics['body_ratio']) / self.max_body_ratio
        strength += body_strength * 25
        
        # Strength based on very long shadows
        shadow_strength = min(
            (metrics['upper_shadow_ratio'] + metrics['lower_shadow_ratio']) / 2,
            10.0  # Cap at 10.0 ratio
        )
        strength += (shadow_strength / 10.0) * 20
        
        # Strength based on shadow balance (more balanced is better)
        if metrics['upper_shadow_ratio'] > 0 and metrics['lower_shadow_ratio'] > 0:
            shadow_ratio = min(
                metrics['upper_shadow_ratio'] / metrics['lower_shadow_ratio'],
                metrics['lower_shadow_ratio'] / metrics['upper_shadow_ratio']
            )
            strength += shadow_ratio * 15  # Higher weight for balance
        
        # Strength based on total range significance
        range_strength = min(metrics['total_range_ratio'] / 0.05, 1.0)  # Normalize to 5%
        strength += range_strength * 10
        
        # Context strength based on trend (very significant at extremes)
        if trend in ['uptrend', 'downtrend']:
            strength += 15  # Very significant in trending markets
        elif trend == 'sideways':
            strength += 5   # Still meaningful in sideways markets
        
        return min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate High Wave Candle Pattern signals.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - high_wave_signals: List of high wave signals (1.0 if pattern, 0.0 otherwise)
            - pattern_strength: List of pattern strength scores (0-100)
        """
        # Convert input to DataFrame if necessary
        if isinstance(data, np.ndarray):
            if data.shape[1] >= 5:
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
            else:
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
        else:
            df = data.copy()
        
        # Initialize result lists
        self.high_wave_signals = [0.0] * len(df)
        self.pattern_strength = [0.0] * len(df)
        
        # Need enough data for trend analysis
        if len(df) <= self.trend_lookback:
            return self._get_results()
        
        # Extract closing prices for trend analysis
        close_prices = df['close'].tolist()
        
        for i in range(self.trend_lookback, len(df)):
            candle = {
                'open': df.iloc[i]['open'],
                'high': df.iloc[i]['high'],
                'low': df.iloc[i]['low'],
                'close': df.iloc[i]['close']
            }
            
            # Calculate candle metrics
            metrics = self._get_candle_metrics(
                candle['open'], candle['high'], 
                candle['low'], candle['close']
            )
            
            # Check if candle is a high wave candle
            if self._is_high_wave_candle(metrics):
                # Determine preceding trend
                trend = self._determine_trend(close_prices, i)
                
                # Calculate pattern strength
                strength = self._calculate_pattern_strength(metrics, trend)
                
                self.high_wave_signals[i] = 1.0
                self.pattern_strength[i] = strength
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'high_wave_signals': self.high_wave_signals,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on High Wave Candle patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with reversal signals and strength
        """
        results = self.calculate(data)
        
        # High wave candles are strong reversal signals
        reversal_signals = results['high_wave_signals']
        
        return {
            'reversal_signals': reversal_signals,
            'signal_strength': results['pattern_strength']
        }


def test_high_wave_candle_pattern():
    """Test the High Wave Candle Pattern indicator with sample data."""
    print("Testing High Wave Candle Pattern Indicator...")
    
    # Create sample data with high wave candle patterns
    np.random.seed(42)
    n_periods = 50
    
    # Generate base price data
    data = []
    
    # Create trend for context
    for i in range(20):
        base_price = 100 + (i * 0.15)  # Mild rising trend
        open_price = base_price + np.random.randn() * 0.05
        close_price = open_price + np.random.randn() * 0.15
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.1
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.1
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add high wave candle pattern
    data.append([103.0, 107.0, 99.0, 103.2, 1500])  # Very small body, very long shadows
    
    # Normal candles
    for i in range(15):
        base_price = 103 + np.random.randn() * 0.2
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price + np.random.randn() * 0.2
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.15
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.15
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Another high wave candle
    data.append([102.5, 105.5, 98.5, 102.4, 1600])  # Very small body, very long shadows
    
    # Add more normal candles
    for i in range(13):
        base_price = 102.5 + np.random.randn() * 0.15
        open_price = base_price + np.random.randn() * 0.08
        close_price = open_price + np.random.randn() * 0.15
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.1
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.1
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test the indicator
    indicator = HighWaveCandlePattern(
        max_body_ratio=0.2,           # Allow slightly larger bodies for test
        min_shadow_ratio=2.5,         # Lower shadow requirement for test
        min_total_range_ratio=0.015,  # Lower range requirement for test
        trend_lookback=8,             # Shorter lookback for test
        trend_threshold=0.01          # 1% trend threshold
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"High wave candle signals: {sum(results['high_wave_signals'])}")
    
    # Find and display detected patterns
    for i, (signal, strength) in enumerate(zip(
        results['high_wave_signals'], 
        results['pattern_strength']
    )):
        if signal > 0:
            print(f"High Wave Candle at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    reversal_signals = sum(signals['reversal_signals'])
    
    print(f"Reversal signals: {reversal_signals}")
    
    print("High Wave Candle Pattern test completed successfully!")
    return True


if __name__ == "__main__":
    test_high_wave_candle_pattern()