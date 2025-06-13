"""
Belt Hold Pattern Indicator

The Belt Hold Pattern is a single-candle reversal pattern:
- Bullish Belt Hold (White Belt Hold): 
  - Opens at or near the low of the day
  - Closes near the high with a long white body
  - Little to no lower shadow
  - Appears after a downtrend
- Bearish Belt Hold (Black Belt Hold):
  - Opens at or near the high of the day  
  - Closes near the low with a long black body
  - Little to no upper shadow
  - Appears after an uptrend

This indicator follows the CCI (Commodity Channel Index) gold standard template.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add the parent directory to the path to import base_indicator
from base_indicator import StandardIndicatorInterface


class BeltHoldType(StandardIndicatorInterface):
    """
    Belt Hold Pattern Indicator
    
    Identifies bullish and bearish belt hold patterns which are strong
    single-candle reversal signals.
    """
    
    def __init__(self, 
                 min_body_ratio: float = 0.7,
                 max_shadow_ratio: float = 0.1,
                 trend_lookback: int = 10,
                 trend_threshold: float = 0.02):
        """
        Initialize the Belt Hold Pattern indicator.
        
        Args:
            min_body_ratio: Minimum body to total range ratio (default: 0.7)
            max_shadow_ratio: Maximum shadow to body ratio (default: 0.1) 
            trend_lookback: Number of periods to determine trend (default: 10)
            trend_threshold: Minimum price change % to identify trend (default: 2%)
        """
        self.min_body_ratio = min_body_ratio
        self.max_shadow_ratio = max_shadow_ratio
        self.trend_lookback = trend_lookback
        self.trend_threshold = trend_threshold
        
        # Parameters dict for base class
        self.parameters = {
            'min_body_ratio': min_body_ratio,
            'max_shadow_ratio': max_shadow_ratio,
            'trend_lookback': trend_lookback,
            'trend_threshold': trend_threshold
        }
        
        # Initialize result storage
        self.bullish_belt_hold = []
        self.bearish_belt_hold = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Belt Hold Type",
            category="pattern",
            description="Single-candle reversal pattern with strong body and minimal shadows",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=self.trend_lookback + 1
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 0.5 <= self.min_body_ratio <= 1.0:
            raise ValueError("min_body_ratio must be between 0.5 and 1.0")
        if not 0.0 <= self.max_shadow_ratio <= 0.5:
            raise ValueError("max_shadow_ratio must be between 0.0 and 0.5")
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
            'upper_shadow_ratio': upper_shadow / body if body > 0 else 0,
            'lower_shadow_ratio': lower_shadow / body if body > 0 else 0,
            'is_bullish': close > open_price,
            'is_bearish': close < open_price,
            'open_position': (open_price - low) / total_range if total_range > 0 else 0,
            'close_position': (close - low) / total_range if total_range > 0 else 0
        }
    
    def _is_bullish_belt_hold(self, metrics: Dict[str, float]) -> bool:
        """
        Check if candle is a bullish belt hold.
        
        Args:
            metrics: Candle metrics
            
        Returns:
            True if candle is a bullish belt hold
        """
        # Must be a bullish candle
        if not metrics['is_bullish']:
            return False
        
        # Must have significant body
        if metrics['body_ratio'] < self.min_body_ratio:
            return False
        
        # Must open near the low (small lower shadow)
        if metrics['lower_shadow_ratio'] > self.max_shadow_ratio:
            return False
        
        # Open should be at or near the low
        if metrics['open_position'] > 0.2:  # Open within bottom 20% of range
            return False
        
        # Close should be near the high
        if metrics['close_position'] < 0.8:  # Close within top 20% of range
            return False
        
        return True
    
    def _is_bearish_belt_hold(self, metrics: Dict[str, float]) -> bool:
        """
        Check if candle is a bearish belt hold.
        
        Args:
            metrics: Candle metrics
            
        Returns:
            True if candle is a bearish belt hold
        """
        # Must be a bearish candle
        if not metrics['is_bearish']:
            return False
        
        # Must have significant body
        if metrics['body_ratio'] < self.min_body_ratio:
            return False
        
        # Must open near the high (small upper shadow)
        if metrics['upper_shadow_ratio'] > self.max_shadow_ratio:
            return False
        
        # Open should be at or near the high
        if metrics['open_position'] < 0.8:  # Open within top 20% of range
            return False
        
        # Close should be near the low
        if metrics['close_position'] > 0.2:  # Close within bottom 20% of range
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
                                  trend: str,
                                  pattern_type: str) -> float:
        """
        Calculate the strength of the belt hold pattern.
        
        Args:
            metrics: Candle metrics
            trend: Preceding trend
            pattern_type: 'bullish' or 'bearish'
            
        Returns:
            Pattern strength score (0-100)
        """
        strength = 60  # Base strength for confirmed pattern
        
        # Strength based on body ratio
        strength += (metrics['body_ratio'] - self.min_body_ratio) * 40
        
        # Strength based on minimal shadows
        if pattern_type == 'bullish':
            shadow_bonus = max(0, (self.max_shadow_ratio - metrics['lower_shadow_ratio']) * 100)
        else:
            shadow_bonus = max(0, (self.max_shadow_ratio - metrics['upper_shadow_ratio']) * 100)
        
        strength += min(shadow_bonus, 15)
        
        # Strength based on open/close positions
        if pattern_type == 'bullish':
            position_bonus = (1 - metrics['open_position']) * 10  # Bonus for opening near low
            position_bonus += metrics['close_position'] * 10      # Bonus for closing near high
        else:
            position_bonus = metrics['open_position'] * 10        # Bonus for opening near high
            position_bonus += (1 - metrics['close_position']) * 10 # Bonus for closing near low
        
        strength += min(position_bonus, 15)
        
        # Context strength based on trend
        if ((pattern_type == 'bullish' and trend == 'downtrend') or
            (pattern_type == 'bearish' and trend == 'uptrend')):
            strength += 10  # Perfect reversal context
        
        return min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Belt Hold Pattern signals.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - bullish_belt_hold: List of bullish belt hold signals (1.0 if pattern, 0.0 otherwise)
            - bearish_belt_hold: List of bearish belt hold signals (1.0 if pattern, 0.0 otherwise)
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
        self.bullish_belt_hold = [0.0] * len(df)
        self.bearish_belt_hold = [0.0] * len(df)
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
            
            # Determine preceding trend
            trend = self._determine_trend(close_prices, i)
            
            # Check for bullish belt hold
            if self._is_bullish_belt_hold(metrics) and trend == 'downtrend':
                self.bullish_belt_hold[i] = 1.0
                strength = self._calculate_pattern_strength(metrics, trend, 'bullish')
                self.pattern_strength[i] = strength
            
            # Check for bearish belt hold
            elif self._is_bearish_belt_hold(metrics) and trend == 'uptrend':
                self.bearish_belt_hold[i] = 1.0
                strength = self._calculate_pattern_strength(metrics, trend, 'bearish')
                self.pattern_strength[i] = strength
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'bullish_belt_hold': self.bullish_belt_hold,
            'bearish_belt_hold': self.bearish_belt_hold,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on Belt Hold patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with buy/sell signals and strength
        """
        results = self.calculate(data)
        
        buy_signals = results['bullish_belt_hold']
        sell_signals = results['bearish_belt_hold']
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_strength': results['pattern_strength']
        }


def test_belt_hold_type():
    """Test the Belt Hold Type indicator with sample data."""
    print("Testing Belt Hold Type Pattern Indicator...")
    
    # Create sample data with belt hold patterns
    np.random.seed(42)
    n_periods = 50
    
    # Generate base price data with clear trends
    data = []
    
    # Create downtrend for first 20 periods
    for i in range(20):
        base_price = 100 - (i * 0.4)  # Declining trend
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price - abs(np.random.randn()) * 0.2  # Bearish bias
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.1
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.1
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add bullish belt hold pattern at end of downtrend
    data.append([91.0, 94.0, 90.8, 93.5, 1200])  # Opens near low, closes near high
    
    # Create uptrend for next 20 periods
    for i in range(20):
        base_price = 93 + (i * 0.3)  # Rising trend
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price + abs(np.random.randn()) * 0.2  # Bullish bias
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.1
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.1
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add bearish belt hold pattern at end of uptrend
    data.append([99.0, 99.2, 96.0, 96.5, 1300])  # Opens near high, closes near low
    
    # Add a few more periods
    for i in range(8):
        base_price = 96 + np.random.randn() * 0.3
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price + np.random.randn() * 0.2
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.1
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.1
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test the indicator
    indicator = BeltHoldType(
        min_body_ratio=0.6,      # Allow smaller bodies for test
        max_shadow_ratio=0.15,   # Allow slightly larger shadows for test
        trend_lookback=8,        # Shorter lookback for test
        trend_threshold=0.015    # 1.5% trend threshold
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"Bullish belt hold signals: {sum(results['bullish_belt_hold'])}")
    print(f"Bearish belt hold signals: {sum(results['bearish_belt_hold'])}")
    
    # Find and display detected patterns
    for i, (bullish, bearish, strength) in enumerate(zip(
        results['bullish_belt_hold'], 
        results['bearish_belt_hold'],
        results['pattern_strength']
    )):
        if bullish > 0:
            print(f"Bullish Belt Hold at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
        if bearish > 0:
            print(f"Bearish Belt Hold at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    buy_signals = sum(signals['buy_signals'])
    sell_signals = sum(signals['sell_signals'])
    
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    print("Belt Hold Type test completed successfully!")
    return True


if __name__ == "__main__":
    test_belt_hold_type()