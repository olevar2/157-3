"""
Hammer Pattern Indicator

The Hammer Pattern identifies hammer and hanging man candlestick patterns:
- Hammer: Bullish reversal pattern at the end of a downtrend
  - Small body at the upper end of the trading range
  - Long lower shadow (at least 2x the body size)
  - Little to no upper shadow
- Hanging Man: Bearish reversal pattern at the end of an uptrend
  - Same shape as hammer but appears after uptrend
  - Confirmation needed for bearish signal

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


class HammerPattern(StandardIndicatorInterface):
    """
    Hammer Pattern Indicator
    
    Identifies hammer and hanging man patterns which are potential reversal signals.
    """
    
    def __init__(self, 
                 min_lower_shadow_ratio: float = 2.0,
                 max_upper_shadow_ratio: float = 0.5,
                 max_body_ratio: float = 0.3,
                 trend_lookback: int = 10,
                 trend_threshold: float = 0.02):
        """
        Initialize the Hammer Pattern indicator.
        
        Args:
            min_lower_shadow_ratio: Minimum ratio of lower shadow to body (default: 2.0)
            max_upper_shadow_ratio: Maximum ratio of upper shadow to body (default: 0.5)
            max_body_ratio: Maximum body to total range ratio (default: 0.3)
            trend_lookback: Number of periods to determine trend (default: 10)
            trend_threshold: Minimum price change % to identify trend (default: 2%)
        """
        self.min_lower_shadow_ratio = min_lower_shadow_ratio
        self.max_upper_shadow_ratio = max_upper_shadow_ratio
        self.max_body_ratio = max_body_ratio
        self.trend_lookback = trend_lookback
        self.trend_threshold = trend_threshold
        
        # Parameters dict for base class
        self.parameters = {
            'min_lower_shadow_ratio': min_lower_shadow_ratio,
            'max_upper_shadow_ratio': max_upper_shadow_ratio,
            'max_body_ratio': max_body_ratio,
            'trend_lookback': trend_lookback,
            'trend_threshold': trend_threshold
        }
        
        # Initialize result storage
        self.hammer_signals = []
        self.hanging_man_signals = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Hammer Pattern",
            category="pattern",
            description="Hammer and hanging man candlestick reversal patterns",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=self.trend_lookback + 1
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 1.0 <= self.min_lower_shadow_ratio <= 10.0:
            raise ValueError("min_lower_shadow_ratio must be between 1.0 and 10.0")
        if not 0.0 <= self.max_upper_shadow_ratio <= 2.0:
            raise ValueError("max_upper_shadow_ratio must be between 0.0 and 2.0")
        if not 0.1 <= self.max_body_ratio <= 0.8:
            raise ValueError("max_body_ratio must be between 0.1 and 0.8")
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
    
    def _is_hammer_shape(self, metrics: Dict[str, float]) -> bool:
        """
        Check if candle has hammer/hanging man shape.
        
        Args:
            metrics: Candle metrics
            
        Returns:
            True if candle has hammer shape
        """
        # Check body size (should be small)
        if metrics['body_ratio'] > self.max_body_ratio:
            return False
        
        # Check lower shadow (should be long)
        if metrics['lower_shadow_ratio'] < self.min_lower_shadow_ratio:
            return False
        
        # Check upper shadow (should be small)
        if metrics['upper_shadow_ratio'] > self.max_upper_shadow_ratio:
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
        Calculate the strength of the hammer pattern.
        
        Args:
            metrics: Candle metrics
            trend: Preceding trend
            pattern_type: 'hammer' or 'hanging_man'
            
        Returns:
            Pattern strength score (0-100)
        """
        strength = 50  # Base strength
        
        # Strength based on lower shadow length
        shadow_strength = min(metrics['lower_shadow_ratio'] * 10, 25)
        strength += shadow_strength
        
        # Strength based on small body
        body_strength = (1 - metrics['body_ratio']) * 15
        strength += body_strength
        
        # Strength based on small upper shadow
        if metrics['upper_shadow_ratio'] < 0.1:
            strength += 10
        elif metrics['upper_shadow_ratio'] < 0.3:
            strength += 5
        
        # Pattern context strength
        if pattern_type == 'hammer' and trend == 'downtrend':
            strength += 15  # Strong bullish reversal context
        elif pattern_type == 'hanging_man' and trend == 'uptrend':
            strength += 10  # Bearish reversal context (needs confirmation)
        
        return min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Hammer Pattern signals.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - hammer_signals: List of hammer signals (1.0 if pattern, 0.0 otherwise)
            - hanging_man_signals: List of hanging man signals (1.0 if pattern, 0.0 otherwise)
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
        self.hammer_signals = [0.0] * len(df)
        self.hanging_man_signals = [0.0] * len(df)
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
            
            # Check if candle has hammer shape
            if not self._is_hammer_shape(metrics):
                continue
            
            # Determine preceding trend
            trend = self._determine_trend(close_prices, i)
            
            # Classify pattern based on trend context
            if trend == 'downtrend':
                # Hammer pattern (bullish reversal)
                self.hammer_signals[i] = 1.0
                strength = self._calculate_pattern_strength(metrics, trend, 'hammer')
                self.pattern_strength[i] = strength
                
            elif trend == 'uptrend':
                # Hanging man pattern (bearish reversal)
                self.hanging_man_signals[i] = 1.0
                strength = self._calculate_pattern_strength(metrics, trend, 'hanging_man')
                self.pattern_strength[i] = strength
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'hammer_signals': self.hammer_signals,
            'hanging_man_signals': self.hanging_man_signals,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on Hammer patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with buy/sell signals and strength
        """
        results = self.calculate(data)
        
        buy_signals = results['hammer_signals']
        sell_signals = results['hanging_man_signals']
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_strength': results['pattern_strength']
        }


def test_hammer_pattern():
    """Test the Hammer Pattern indicator with sample data."""
    print("Testing Hammer Pattern Indicator...")
    
    # Create sample data with hammer patterns
    np.random.seed(42)
    n_periods = 50
    
    # Generate base price data with clear trends
    data = []
    
    # Create downtrend for first 20 periods
    for i in range(20):
        base_price = 100 - (i * 0.5)  # Declining trend
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price - abs(np.random.randn()) * 0.2  # Bearish bias
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.1
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.1
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add hammer pattern at end of downtrend
    data.append([89.0, 89.2, 86.0, 88.8, 1200])  # Hammer: small body, long lower shadow
    
    # Create uptrend for next 20 periods
    for i in range(20):
        base_price = 89 + (i * 0.4)  # Rising trend
        open_price = base_price + np.random.randn() * 0.1
        close_price = open_price + abs(np.random.randn()) * 0.2  # Bullish bias
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.1
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.1
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add hanging man pattern at end of uptrend
    data.append([97.0, 97.2, 94.0, 96.8, 1300])  # Hanging man: small body, long lower shadow
    
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
    indicator = HammerPattern(
        min_lower_shadow_ratio=1.5,  # Slightly lower for test
        max_upper_shadow_ratio=0.8,  # Slightly higher for test
        max_body_ratio=0.4,          # Allow larger bodies for test
        trend_lookback=8,            # Shorter lookback for test
        trend_threshold=0.015        # 1.5% trend threshold
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"Hammer signals: {sum(results['hammer_signals'])}")
    print(f"Hanging man signals: {sum(results['hanging_man_signals'])}")
    
    # Find and display detected patterns
    for i, (hammer, hanging_man, strength) in enumerate(zip(
        results['hammer_signals'], 
        results['hanging_man_signals'],
        results['pattern_strength']
    )):
        if hammer > 0:
            print(f"Hammer at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
        if hanging_man > 0:
            print(f"Hanging Man at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    buy_signals = sum(signals['buy_signals'])
    sell_signals = sum(signals['sell_signals'])
    
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    print("Hammer Pattern test completed successfully!")
    return True


if __name__ == "__main__":
    test_hammer_pattern()