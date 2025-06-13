"""
Harami Pattern Indicator

The Harami Pattern is a two-candle reversal pattern:
- Bullish Harami: A large bearish candle followed by a small bullish candle 
  whose body is completely within the body of the first candle
- Bearish Harami: A large bullish candle followed by a small bearish candle 
  whose body is completely within the body of the first candle

The word "harami" means "pregnant" in Japanese, referring to the smaller candle 
being contained within the larger candle's body.

This indicator follows the CCI (Commodity Channel Index) gold standard template.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add the parent directory to the path to import base_indicator
from base_indicator import StandardIndicatorInterface


class HaramiType(StandardIndicatorInterface):
    """
    Harami Pattern Indicator
    
    Identifies bullish and bearish harami patterns which indicate
    potential trend reversals through momentum exhaustion.
    """
    
    def __init__(self, 
                 min_first_body_ratio: float = 0.6,
                 max_second_body_ratio: float = 0.5,
                 harami_ratio_threshold: float = 0.8,
                 trend_lookback: int = 10,
                 trend_threshold: float = 0.02):
        """
        Initialize the Harami Pattern indicator.
        
        Args:
            min_first_body_ratio: Minimum body ratio for first candle (default: 0.6)
            max_second_body_ratio: Maximum body ratio for second candle (default: 0.5)
            harami_ratio_threshold: Maximum ratio of second to first body (default: 0.8)
            trend_lookback: Number of periods to determine trend (default: 10)
            trend_threshold: Minimum price change % to identify trend (default: 2%)
        """
        self.min_first_body_ratio = min_first_body_ratio
        self.max_second_body_ratio = max_second_body_ratio
        self.harami_ratio_threshold = harami_ratio_threshold
        self.trend_lookback = trend_lookback
        self.trend_threshold = trend_threshold
        
        # Parameters dict for base class
        self.parameters = {
            'min_first_body_ratio': min_first_body_ratio,
            'max_second_body_ratio': max_second_body_ratio,
            'harami_ratio_threshold': harami_ratio_threshold,
            'trend_lookback': trend_lookback,
            'trend_threshold': trend_threshold
        }
        
        # Initialize result storage
        self.bullish_harami = []
        self.bearish_harami = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Harami Type",
            category="pattern",
            description="Two-candle reversal pattern with small candle inside large candle body",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=self.trend_lookback + 2
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 0.3 <= self.min_first_body_ratio <= 1.0:
            raise ValueError("min_first_body_ratio must be between 0.3 and 1.0")
        if not 0.1 <= self.max_second_body_ratio <= 0.8:
            raise ValueError("max_second_body_ratio must be between 0.1 and 0.8")
        if not 0.1 <= self.harami_ratio_threshold <= 1.0:
            raise ValueError("harami_ratio_threshold must be between 0.1 and 1.0")
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
            'is_bullish': close > open_price,
            'is_bearish': close < open_price,
            'body_high': max(open_price, close),
            'body_low': min(open_price, close)
        }
    
    def _is_harami_pattern(self, first_metrics: Dict[str, float],
                          second_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if two candles form a harami pattern.
        
        Args:
            first_metrics: Metrics of the first candle
            second_metrics: Metrics of the second candle
            
        Returns:
            Tuple of (is_harami, pattern_type) where pattern_type is 'bullish', 'bearish', or 'none'
        """
        # First candle must have significant body
        if first_metrics['body_ratio'] < self.min_first_body_ratio:
            return False, 'none'
        
        # Second candle should have smaller body
        if second_metrics['body_ratio'] > self.max_second_body_ratio:
            return False, 'none'
        
        # Check size ratio between candles
        if first_metrics['body'] > 0:
            ratio = second_metrics['body'] / first_metrics['body']
            if ratio > self.harami_ratio_threshold:
                return False, 'none'
        else:
            return False, 'none'
        
        # Check if second candle's body is completely within first candle's body
        is_inside = (second_metrics['body_high'] <= first_metrics['body_high'] and
                    second_metrics['body_low'] >= first_metrics['body_low'])
        
        if not is_inside:
            return False, 'none'
        
        # Determine pattern type based on first candle direction
        if first_metrics['is_bearish'] and second_metrics['is_bullish']:
            return True, 'bullish'
        elif first_metrics['is_bullish'] and second_metrics['is_bearish']:
            return True, 'bearish'
        elif first_metrics['is_bearish'] and second_metrics['is_bearish']:
            return True, 'bearish'  # Bearish continuation can also be harami
        elif first_metrics['is_bullish'] and second_metrics['is_bullish']:
            return True, 'bullish'  # Bullish continuation can also be harami
        
        return False, 'none'
    
    def _determine_trend(self, prices: List[float], current_idx: int) -> str:
        """
        Determine the trend preceding the current candles.
        
        Args:
            prices: List of prices (typically closing prices)
            current_idx: Current index (second candle)
            
        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        start_idx = max(0, current_idx - self.trend_lookback)
        
        if start_idx >= current_idx - 1:
            return 'sideways'
        
        start_price = prices[start_idx]
        end_price = prices[current_idx - 2]  # Price before first candle
        
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
                                  first_metrics: Dict[str, float],
                                  second_metrics: Dict[str, float],
                                  trend: str,
                                  pattern_type: str) -> float:
        """
        Calculate the strength of the harami pattern.
        
        Args:
            first_metrics: First candle metrics
            second_metrics: Second candle metrics
            trend: Preceding trend
            pattern_type: 'bullish' or 'bearish'
            
        Returns:
            Pattern strength score (0-100)
        """
        strength = 50  # Base strength
        
        # Strength based on first candle body size
        strength += (first_metrics['body_ratio'] - self.min_first_body_ratio) * 30
        
        # Strength based on size contrast (smaller second candle is better)
        if first_metrics['body'] > 0:
            size_ratio = second_metrics['body'] / first_metrics['body']
            strength += (self.harami_ratio_threshold - size_ratio) * 25
        
        # Strength based on how well the second candle is contained
        first_body_range = first_metrics['body_high'] - first_metrics['body_low']
        if first_body_range > 0:
            # Calculate how centered the second candle is within the first
            second_center = (second_metrics['body_high'] + second_metrics['body_low']) / 2
            first_center = (first_metrics['body_high'] + first_metrics['body_low']) / 2
            
            center_distance = abs(second_center - first_center)
            max_distance = first_body_range / 2
            
            if max_distance > 0:
                centering_score = (1 - center_distance / max_distance) * 10
                strength += max(0, centering_score)
        
        # Context strength based on trend
        if ((pattern_type == 'bullish' and trend == 'downtrend') or
            (pattern_type == 'bearish' and trend == 'uptrend')):
            strength += 15  # Strong reversal context
        elif trend == 'sideways':
            strength += 5   # Neutral context
        
        # Bonus for opposite color candles (classic harami)
        if ((first_metrics['is_bearish'] and second_metrics['is_bullish']) or
            (first_metrics['is_bullish'] and second_metrics['is_bearish'])):
            strength += 10
        
        return min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Harami Pattern signals.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - bullish_harami: List of bullish harami signals (1.0 if pattern, 0.0 otherwise)
            - bearish_harami: List of bearish harami signals (1.0 if pattern, 0.0 otherwise)
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
        self.bullish_harami = [0.0] * len(df)
        self.bearish_harami = [0.0] * len(df)
        self.pattern_strength = [0.0] * len(df)
        
        # Need enough data for trend analysis plus 2 candles
        if len(df) <= self.trend_lookback + 1:
            return self._get_results()
        
        # Extract closing prices for trend analysis
        close_prices = df['close'].tolist()
        
        for i in range(self.trend_lookback + 1, len(df)):
            # Get two consecutive candles
            first_candle = {
                'open': df.iloc[i-1]['open'],
                'high': df.iloc[i-1]['high'],
                'low': df.iloc[i-1]['low'],
                'close': df.iloc[i-1]['close']
            }
            
            second_candle = {
                'open': df.iloc[i]['open'],
                'high': df.iloc[i]['high'],
                'low': df.iloc[i]['low'],
                'close': df.iloc[i]['close']
            }
            
            # Calculate candle metrics
            first_metrics = self._get_candle_metrics(
                first_candle['open'], first_candle['high'], 
                first_candle['low'], first_candle['close']
            )
            
            second_metrics = self._get_candle_metrics(
                second_candle['open'], second_candle['high'],
                second_candle['low'], second_candle['close']
            )
            
            # Check for harami pattern
            is_harami, pattern_type = self._is_harami_pattern(first_metrics, second_metrics)
            
            if is_harami:
                # Determine preceding trend
                trend = self._determine_trend(close_prices, i)
                
                # Calculate pattern strength
                strength = self._calculate_pattern_strength(
                    first_metrics, second_metrics, trend, pattern_type
                )
                
                # Assign signals based on pattern type and trend context
                if pattern_type == 'bullish':
                    self.bullish_harami[i] = 1.0
                elif pattern_type == 'bearish':
                    self.bearish_harami[i] = 1.0
                
                self.pattern_strength[i] = strength
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'bullish_harami': self.bullish_harami,
            'bearish_harami': self.bearish_harami,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on Harami patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with buy/sell signals and strength
        """
        results = self.calculate(data)
        
        buy_signals = results['bullish_harami']
        sell_signals = results['bearish_harami']
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_strength': results['pattern_strength']
        }


def test_harami_type():
    """Test the Harami Type indicator with sample data."""
    print("Testing Harami Type Pattern Indicator...")
    
    # Create sample data with harami patterns
    np.random.seed(42)
    n_periods = 50
    
    # Generate base price data with trends
    data = []
    
    # Create downtrend for first 18 periods
    for i in range(18):
        base_price = 100 - (i * 0.3)  # Declining trend
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price - abs(np.random.randn()) * 0.15  # Bearish bias
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.08
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.08
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add bullish harami pattern at end of downtrend
    # Large bearish candle
    data.append([94.5, 94.8, 92.0, 92.5, 1200])
    # Small bullish candle inside
    data.append([93.0, 93.8, 92.8, 93.5, 800])
    
    # Create uptrend for next 18 periods
    for i in range(18):
        base_price = 93.5 + (i * 0.25)  # Rising trend
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price + abs(np.random.randn()) * 0.15  # Bullish bias
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.08
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.08
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add bearish harami pattern at end of uptrend
    # Large bullish candle
    data.append([98.0, 100.5, 97.8, 100.0, 1300])
    # Small bearish candle inside
    data.append([99.5, 99.8, 98.5, 98.8, 900])
    
    # Add a few more periods
    for i in range(10):
        base_price = 98.8 + np.random.randn() * 0.2
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price + np.random.randn() * 0.15
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.08
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.08
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test the indicator
    indicator = HaramiType(
        min_first_body_ratio=0.5,      # Allow smaller first bodies for test
        max_second_body_ratio=0.6,     # Allow larger second bodies for test
        harami_ratio_threshold=0.9,    # Allow larger ratios for test
        trend_lookback=8,              # Shorter lookback for test
        trend_threshold=0.015          # 1.5% trend threshold
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"Bullish harami signals: {sum(results['bullish_harami'])}")
    print(f"Bearish harami signals: {sum(results['bearish_harami'])}")
    
    # Find and display detected patterns
    for i, (bullish, bearish, strength) in enumerate(zip(
        results['bullish_harami'], 
        results['bearish_harami'],
        results['pattern_strength']
    )):
        if bullish > 0:
            print(f"Bullish Harami at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
        if bearish > 0:
            print(f"Bearish Harami at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    buy_signals = sum(signals['buy_signals'])
    sell_signals = sum(signals['sell_signals'])
    
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    print("Harami Type test completed successfully!")
    return True


if __name__ == "__main__":
    test_harami_type()