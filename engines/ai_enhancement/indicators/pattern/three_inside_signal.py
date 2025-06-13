"""
Three Inside Signal Pattern Indicator

The Three Inside Signal is a three-candle reversal pattern that consists of:
- Three Inside Up (Bullish): 
  1. Large black candle
  2. Small white candle inside the body of the first candle (harami)
  3. White candle that closes above the high of the first candle
- Three Inside Down (Bearish):
  1. Large white candle
  2. Small black candle inside the body of the first candle (harami)
  3. Black candle that closes below the low of the first candle

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


class ThreeInsideSignal(StandardIndicatorInterface):
    """
    Three Inside Signal Pattern Indicator
    
    A three-candle reversal pattern that confirms trend changes through
    an initial harami pattern followed by a confirmation candle.
    """
    
    def __init__(self, 
                 min_body_ratio: float = 0.6,
                 harami_ratio: float = 0.8,
                 volume_confirmation: bool = True):
        """
        Initialize the Three Inside Signal indicator.
        
        Args:
            min_body_ratio: Minimum body to total range ratio for significant candles (default: 0.6)
            harami_ratio: Maximum ratio for harami candle body to first candle body (default: 0.8)
            volume_confirmation: Whether to require volume confirmation (default: True)
        """
        self.min_body_ratio = min_body_ratio
        self.harami_ratio = harami_ratio
        self.volume_confirmation = volume_confirmation
        
        # Parameters dict for base class
        self.parameters = {
            'min_body_ratio': min_body_ratio,
            'harami_ratio': harami_ratio,
            'volume_confirmation': volume_confirmation
        }
        
        # Initialize result storage
        self.three_inside_up = []
        self.three_inside_down = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Three Inside Signal",
            category="pattern",
            description="Three-candle reversal pattern with harami confirmation",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=3
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 0.1 <= self.min_body_ratio <= 1.0:
            raise ValueError("min_body_ratio must be between 0.1 and 1.0")
        if not 0.1 <= self.harami_ratio <= 1.0:
            raise ValueError("harami_ratio must be between 0.1 and 1.0")
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
            'is_bearish': close < open_price
        }
    
    def _is_significant_candle(self, metrics: Dict[str, float]) -> bool:
        """Check if candle has significant body."""
        return metrics['body_ratio'] >= self.min_body_ratio
    
    def _is_harami_inside(self, first_metrics: Dict[str, float],
                         first_open: float, first_close: float,
                         second_open: float, second_close: float) -> bool:
        """
        Check if second candle is inside first candle (harami pattern).
        
        Args:
            first_metrics: Metrics of the first candle
            first_open: First candle open
            first_close: First candle close
            second_open: Second candle open
            second_close: Second candle close
            
        Returns:
            True if second candle is inside first candle
        """
        first_high_body = max(first_open, first_close)
        first_low_body = min(first_open, first_close)
        
        second_high_body = max(second_open, second_close)
        second_low_body = min(second_open, second_close)
        
        # Check if second candle is completely inside first candle's body
        is_inside = (second_high_body <= first_high_body and 
                    second_low_body >= first_low_body)
        
        # Check harami ratio
        second_body = abs(second_close - second_open)
        first_body = first_metrics['body']
        
        if first_body > 0:
            ratio_ok = (second_body / first_body) <= self.harami_ratio
        else:
            ratio_ok = False
        
        return is_inside and ratio_ok
    
    def _calculate_pattern_strength(self, 
                                  candles: List[Dict],
                                  pattern_type: str) -> float:
        """
        Calculate the strength of the three inside pattern.
        
        Args:
            candles: List of three candle data dictionaries
            pattern_type: 'up' or 'down'
            
        Returns:
            Pattern strength score (0-100)
        """
        strength = 50  # Base strength
        
        # Strength based on first candle body size
        first_body_ratio = candles[0]['body_ratio']
        strength += min(first_body_ratio * 30, 20)  # Max 20 points
        
        # Strength based on harami ratio
        second_body = abs(candles[1]['close'] - candles[1]['open'])
        first_body = abs(candles[0]['close'] - candles[0]['open'])
        if first_body > 0:
            harami_ratio = second_body / first_body
            strength += (1 - harami_ratio) * 15  # Max 15 points for smaller harami
        
        # Strength based on third candle confirmation
        third_body_ratio = candles[2]['body_ratio']
        strength += min(third_body_ratio * 20, 15)  # Max 15 points
        
        # Volume confirmation
        if self.volume_confirmation:
            if all('volume' in candle for candle in candles):
                # Check if third candle has higher volume
                if candles[2]['volume'] > candles[0]['volume']:
                    strength += 10
        
        return min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Three Inside Signal patterns.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - three_inside_up: List of bullish three inside signals (1.0 if pattern, 0.0 otherwise)
            - three_inside_down: List of bearish three inside signals (1.0 if pattern, 0.0 otherwise)
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
        self.three_inside_up = [0.0] * len(df)
        self.three_inside_down = [0.0] * len(df)
        self.pattern_strength = [0.0] * len(df)
        
        # Need at least 3 candles for the pattern
        if len(df) < 3:
            return self._get_results()
        
        for i in range(2, len(df)):
            # Get three consecutive candles
            candles = []
            for j in range(3):
                idx = i - 2 + j
                candle = {
                    'open': df.iloc[idx]['open'],
                    'high': df.iloc[idx]['high'],
                    'low': df.iloc[idx]['low'],
                    'close': df.iloc[idx]['close']
                }
                if 'volume' in df.columns:
                    candle['volume'] = df.iloc[idx]['volume']
                
                # Add metrics
                metrics = self._get_candle_metrics(
                    candle['open'], candle['high'], 
                    candle['low'], candle['close']
                )
                candle.update(metrics)
                candles.append(candle)
            
            first, second, third = candles
            
            # Check for Three Inside Up pattern
            if (first['is_bearish'] and self._is_significant_candle(first) and
                second['is_bullish'] and 
                self._is_harami_inside(first, first['open'], first['close'], 
                                     second['open'], second['close']) and
                third['is_bullish'] and 
                third['close'] > first['high']):
                
                self.three_inside_up[i] = 1.0
                self.pattern_strength[i] = self._calculate_pattern_strength(candles, 'up')
            
            # Check for Three Inside Down pattern
            elif (first['is_bullish'] and self._is_significant_candle(first) and
                  second['is_bearish'] and 
                  self._is_harami_inside(first, first['open'], first['close'], 
                                       second['open'], second['close']) and
                  third['is_bearish'] and 
                  third['close'] < first['low']):
                
                self.three_inside_down[i] = 1.0
                self.pattern_strength[i] = self._calculate_pattern_strength(candles, 'down')
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'three_inside_up': self.three_inside_up,
            'three_inside_down': self.three_inside_down,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on Three Inside patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with buy/sell signals and strength
        """
        results = self.calculate(data)
        
        buy_signals = results['three_inside_up']
        sell_signals = results['three_inside_down']
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_strength': results['pattern_strength']
        }


def test_three_inside_signal():
    """Test the Three Inside Signal indicator with sample data."""
    print("Testing Three Inside Signal Pattern Indicator...")
    
    # Create sample data with three inside patterns
    np.random.seed(42)
    n_periods = 100
    
    # Generate base price data
    prices = 100 + np.cumsum(np.random.randn(n_periods) * 0.02)
    
    data = []
    for i in range(n_periods):
        # Normal candle
        open_price = prices[i]
        close_price = open_price + np.random.randn() * 0.5
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.2
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.2
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add a three inside up pattern at indices 50-52
    if n_periods > 52:
        # Large bearish candle
        data[50][0] = 102.0  # open
        data[50][1] = 102.2  # high
        data[50][2] = 98.0   # low
        data[50][3] = 98.5   # close
        
        # Small bullish harami inside
        data[51][0] = 99.0   # open
        data[51][1] = 100.5  # high
        data[51][2] = 98.8   # low
        data[51][3] = 100.0  # close
        
        # Bullish confirmation above first candle's high
        data[52][0] = 100.2  # open
        data[52][1] = 103.5  # high
        data[52][2] = 100.0  # low
        data[52][3] = 103.0  # close
    
    # Add a three inside down pattern at indices 70-72
    if n_periods > 72:
        # Large bullish candle
        data[70][0] = 105.0  # open
        data[70][1] = 109.0  # high
        data[70][2] = 104.8  # low
        data[70][3] = 108.5  # close
        
        # Small bearish harami inside
        data[71][0] = 108.0  # open
        data[71][1] = 108.2  # high
        data[71][2] = 106.5  # low
        data[71][3] = 107.0  # close
        
        # Bearish confirmation below first candle's low
        data[72][0] = 106.8  # open
        data[72][1] = 107.0  # high
        data[72][2] = 103.0  # low
        data[72][3] = 103.5  # close
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test the indicator
    indicator = ThreeInsideSignal(
        min_body_ratio=0.5,      # Allow smaller bodies for test
        harami_ratio=0.9,        # Allow larger harami for test
        volume_confirmation=True
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"Three Inside Up signals: {sum(results['three_inside_up'])}")
    print(f"Three Inside Down signals: {sum(results['three_inside_down'])}")
    
    # Find and display detected patterns
    for i, (up, down, strength) in enumerate(zip(
        results['three_inside_up'], 
        results['three_inside_down'],
        results['pattern_strength']
    )):
        if up > 0:
            print(f"Three Inside Up at index {i}, strength: {strength:.1f}")
        if down > 0:
            print(f"Three Inside Down at index {i}, strength: {strength:.1f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    buy_signals = sum(signals['buy_signals'])
    sell_signals = sum(signals['sell_signals'])
    
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    print("Three Inside Signal test completed successfully!")
    return True


if __name__ == "__main__":
    test_three_inside_signal()