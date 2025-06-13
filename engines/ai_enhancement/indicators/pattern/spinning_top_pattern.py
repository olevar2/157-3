"""
Spinning Top Pattern Indicator

The Spinning Top Pattern is a single-candle pattern that indicates indecision:
- Small body (open and close are close together)
- Long upper and lower shadows 
- Can be bullish or bearish colored
- Signals market indecision and potential reversal
- More significant when appearing at trend extremes

This indicator follows the CCI (Commodity Channel Index) gold standard template.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add the parent directory to the path to import base_indicator
from base_indicator import StandardIndicatorInterface


class SpinningTopPattern(StandardIndicatorInterface):
    """
    Spinning Top Pattern Indicator
    
    Identifies spinning top patterns which indicate market indecision
    and potential trend reversals.
    """
    
    def __init__(self, 
                 max_body_ratio: float = 0.3,
                 min_shadow_ratio: float = 1.5,
                 trend_lookback: int = 10,
                 trend_threshold: float = 0.02):
        """
        Initialize the Spinning Top Pattern indicator.
        
        Args:
            max_body_ratio: Maximum body to total range ratio (default: 0.3)
            min_shadow_ratio: Minimum shadow to body ratio (default: 1.5)
            trend_lookback: Number of periods to determine trend (default: 10)
            trend_threshold: Minimum price change % to identify trend (default: 2%)
        """
        self.max_body_ratio = max_body_ratio
        self.min_shadow_ratio = min_shadow_ratio
        self.trend_lookback = trend_lookback
        self.trend_threshold = trend_threshold
        
        # Parameters dict for base class
        self.parameters = {
            'max_body_ratio': max_body_ratio,
            'min_shadow_ratio': min_shadow_ratio,
            'trend_lookback': trend_lookback,
            'trend_threshold': trend_threshold
        }
        
        # Initialize result storage
        self.spinning_top_signals = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Spinning Top Pattern",
            category="pattern",
            description="Single-candle indecision pattern with small body and long shadows",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=self.trend_lookback + 1
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 0.1 <= self.max_body_ratio <= 0.6:
            raise ValueError("max_body_ratio must be between 0.1 and 0.6")
        if not 0.5 <= self.min_shadow_ratio <= 5.0:
            raise ValueError("min_shadow_ratio must be between 0.5 and 5.0")
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
            'is_bullish': close > open_price,
            'is_bearish': close < open_price
        }
    
    def _is_spinning_top(self, metrics: Dict[str, float]) -> bool:
        """
        Check if candle is a spinning top.
        
        Args:
            metrics: Candle metrics
            
        Returns:
            True if candle is a spinning top
        """
        # Must have small body
        if metrics['body_ratio'] > self.max_body_ratio:
            return False
        
        # Must have long shadows (both upper and lower)
        if (metrics['upper_shadow_ratio'] < self.min_shadow_ratio or
            metrics['lower_shadow_ratio'] < self.min_shadow_ratio):
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
        Calculate the strength of the spinning top pattern.
        
        Args:
            metrics: Candle metrics
            trend: Preceding trend
            
        Returns:
            Pattern strength score (0-100)
        """
        strength = 50  # Base strength
        
        # Strength based on small body (smaller is better)
        body_strength = (self.max_body_ratio - metrics['body_ratio']) / self.max_body_ratio
        strength += body_strength * 20
        
        # Strength based on long shadows
        shadow_strength = min(
            (metrics['upper_shadow_ratio'] + metrics['lower_shadow_ratio']) / 2,
            5.0  # Cap at 5.0 ratio
        )
        strength += (shadow_strength / 5.0) * 20
        
        # Strength based on shadow balance (more balanced is better)
        if metrics['upper_shadow_ratio'] > 0 and metrics['lower_shadow_ratio'] > 0:
            shadow_ratio = min(
                metrics['upper_shadow_ratio'] / metrics['lower_shadow_ratio'],
                metrics['lower_shadow_ratio'] / metrics['upper_shadow_ratio']
            )
            strength += shadow_ratio * 10  # Max 10 points for perfect balance
        
        # Context strength based on trend (more significant at extremes)
        if trend in ['uptrend', 'downtrend']:
            strength += 10  # More significant in trending markets
        
        return min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Spinning Top Pattern signals.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - spinning_top_signals: List of spinning top signals (1.0 if pattern, 0.0 otherwise)
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
        self.spinning_top_signals = [0.0] * len(df)
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
            
            # Check if candle is a spinning top
            if self._is_spinning_top(metrics):
                # Determine preceding trend
                trend = self._determine_trend(close_prices, i)
                
                # Calculate pattern strength
                strength = self._calculate_pattern_strength(metrics, trend)
                
                self.spinning_top_signals[i] = 1.0
                self.pattern_strength[i] = strength
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'spinning_top_signals': self.spinning_top_signals,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on Spinning Top patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with neutral signals and strength
        """
        results = self.calculate(data)
        
        # Spinning tops are neutral signals indicating indecision
        neutral_signals = results['spinning_top_signals']
        
        return {
            'neutral_signals': neutral_signals,
            'signal_strength': results['pattern_strength']
        }


def test_spinning_top_pattern():
    """Test the Spinning Top Pattern indicator with sample data."""
    print("Testing Spinning Top Pattern Indicator...")
    
    # Create sample data with spinning top patterns
    np.random.seed(42)
    n_periods = 50
    
    # Generate base price data
    data = []
    
    # Create trend for context
    for i in range(20):
        base_price = 100 + (i * 0.2)  # Rising trend
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price + np.random.randn() * 0.3
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.2
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.2
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add spinning top patterns
    # Spinning top 1
    data.append([104.0, 106.5, 102.0, 104.3, 1200])  # Small body, long shadows
    
    # Normal candles
    for i in range(10):
        base_price = 104 + np.random.randn() * 0.2
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price + np.random.randn() * 0.3
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.2
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.2
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Spinning top 2
    data.append([103.8, 105.8, 101.5, 103.6, 1100])  # Small body, long shadows
    
    # Add more normal candles
    for i in range(18):
        base_price = 103.5 + np.random.randn() * 0.3
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price + np.random.randn() * 0.2
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.15
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.15
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test the indicator
    indicator = SpinningTopPattern(
        max_body_ratio=0.4,        # Allow larger bodies for test
        min_shadow_ratio=1.2,      # Lower shadow requirement for test
        trend_lookback=8,          # Shorter lookback for test
        trend_threshold=0.015      # 1.5% trend threshold
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"Spinning top signals: {sum(results['spinning_top_signals'])}")
    
    # Find and display detected patterns
    for i, (signal, strength) in enumerate(zip(
        results['spinning_top_signals'], 
        results['pattern_strength']
    )):
        if signal > 0:
            print(f"Spinning Top at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    neutral_signals = sum(signals['neutral_signals'])
    
    print(f"Neutral signals: {neutral_signals}")
    
    print("Spinning Top Pattern test completed successfully!")
    return True


if __name__ == "__main__":
    test_spinning_top_pattern()