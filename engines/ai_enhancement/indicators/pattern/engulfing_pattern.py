"""
Engulfing Pattern Indicator

The Engulfing Pattern is a two-candle reversal pattern:
- Bullish Engulfing: A small bearish candle followed by a larger bullish candle 
  that completely engulfs the previous candle's body
- Bearish Engulfing: A small bullish candle followed by a larger bearish candle 
  that completely engulfs the previous candle's body

This pattern indicates a potential trend reversal with strong momentum.

This indicator follows the CCI (Commodity Channel Index) gold standard template.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add the parent directory to the path to import base_indicator
from base_indicator import StandardIndicatorInterface


class EngulfingPattern(StandardIndicatorInterface):
    """
    Engulfing Pattern Indicator
    
    Identifies bullish and bearish engulfing patterns where the second candle
    completely engulfs the body of the first candle.
    """
    
    def __init__(self, 
                 min_body_ratio: float = 0.3,
                 engulfing_ratio: float = 1.1,
                 volume_confirmation: bool = True):
        """
        Initialize the Engulfing Pattern indicator.
        
        Args:
            min_body_ratio: Minimum body to total range ratio for significant candles (default: 0.3)
            engulfing_ratio: Minimum ratio of engulfing candle body to engulfed candle body (default: 1.1)
            volume_confirmation: Whether to require volume confirmation (default: True)
        """
        self.min_body_ratio = min_body_ratio
        self.engulfing_ratio = engulfing_ratio
        self.volume_confirmation = volume_confirmation
        
        # Parameters dict for base class
        self.parameters = {
            'min_body_ratio': min_body_ratio,
            'engulfing_ratio': engulfing_ratio,
            'volume_confirmation': volume_confirmation
        }
        
        # Initialize result storage
        self.bullish_engulfing = []
        self.bearish_engulfing = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Engulfing Pattern",
            category="pattern",
            description="Two-candle reversal pattern where second candle engulfs first",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=2
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 0.1 <= self.min_body_ratio <= 1.0:
            raise ValueError("min_body_ratio must be between 0.1 and 1.0")
        if not 1.0 <= self.engulfing_ratio <= 5.0:
            raise ValueError("engulfing_ratio must be between 1.0 and 5.0")
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
    
    def _is_engulfing(self, first_metrics: Dict[str, float],
                     second_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if second candle engulfs the first candle.
        
        Args:
            first_metrics: Metrics of the first candle
            second_metrics: Metrics of the second candle
            
        Returns:
            Tuple of (is_engulfing, pattern_type) where pattern_type is 'bullish', 'bearish', or 'none'
        """
        # Check if both candles have significant bodies
        if (first_metrics['body_ratio'] < self.min_body_ratio or 
            second_metrics['body_ratio'] < self.min_body_ratio):
            return False, 'none'
        
        # Check engulfing ratio
        if first_metrics['body'] > 0:
            ratio = second_metrics['body'] / first_metrics['body']
            if ratio < self.engulfing_ratio:
                return False, 'none'
        else:
            return False, 'none'
        
        # Check if second candle's body completely engulfs first candle's body
        engulfs_body = (second_metrics['body_high'] > first_metrics['body_high'] and
                       second_metrics['body_low'] < first_metrics['body_low'])
        
        if not engulfs_body:
            return False, 'none'
        
        # Determine pattern type
        if first_metrics['is_bearish'] and second_metrics['is_bullish']:
            return True, 'bullish'
        elif first_metrics['is_bullish'] and second_metrics['is_bearish']:
            return True, 'bearish'
        
        return False, 'none'
    
    def _calculate_pattern_strength(self, 
                                  first_candle: Dict,
                                  second_candle: Dict,
                                  pattern_type: str) -> float:
        """
        Calculate the strength of the engulfing pattern.
        
        Args:
            first_candle: First candle data
            second_candle: Second candle data
            pattern_type: 'bullish' or 'bearish'
            
        Returns:
            Pattern strength score (0-100)
        """
        strength = 60  # Base strength for confirmed pattern
        
        # Strength based on engulfing ratio
        if first_candle['body'] > 0:
            engulf_ratio = second_candle['body'] / first_candle['body']
            strength += min((engulf_ratio - 1) * 20, 20)  # Max 20 points
        
        # Strength based on second candle body ratio
        strength += second_candle['body_ratio'] * 10  # Max 10 points
        
        # Strength based on range engulfing (not just body)
        second_range = second_candle['total_range']
        first_range = first_candle['total_range']
        if first_range > 0:
            range_ratio = second_range / first_range
            strength += min(range_ratio * 5, 10)  # Max 10 points
        
        # Volume confirmation
        if self.volume_confirmation:
            if ('volume' in first_candle and 'volume' in second_candle and
                second_candle['volume'] > first_candle['volume']):
                strength += 10
        
        return min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Engulfing Pattern signals.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - bullish_engulfing: List of bullish engulfing signals (1.0 if pattern, 0.0 otherwise)
            - bearish_engulfing: List of bearish engulfing signals (1.0 if pattern, 0.0 otherwise)
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
        self.bullish_engulfing = [0.0] * len(df)
        self.bearish_engulfing = [0.0] * len(df)
        self.pattern_strength = [0.0] * len(df)
        
        # Need at least 2 candles for the pattern
        if len(df) < 2:
            return self._get_results()
        
        for i in range(1, len(df)):
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
            
            # Add volume if available
            if 'volume' in df.columns:
                first_candle['volume'] = df.iloc[i-1]['volume']
                second_candle['volume'] = df.iloc[i]['volume']
            
            # Calculate candle metrics
            first_metrics = self._get_candle_metrics(
                first_candle['open'], first_candle['high'], 
                first_candle['low'], first_candle['close']
            )
            
            second_metrics = self._get_candle_metrics(
                second_candle['open'], second_candle['high'],
                second_candle['low'], second_candle['close']
            )
            
            # Add metrics to candle data
            first_candle.update(first_metrics)
            second_candle.update(second_metrics)
            
            # Check for engulfing pattern
            is_engulfing, pattern_type = self._is_engulfing(first_metrics, second_metrics)
            
            if is_engulfing:
                strength = self._calculate_pattern_strength(
                    first_candle, second_candle, pattern_type
                )
                
                if pattern_type == 'bullish':
                    self.bullish_engulfing[i] = 1.0
                elif pattern_type == 'bearish':
                    self.bearish_engulfing[i] = 1.0
                
                self.pattern_strength[i] = strength
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'bullish_engulfing': self.bullish_engulfing,
            'bearish_engulfing': self.bearish_engulfing,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on Engulfing patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with buy/sell signals and strength
        """
        results = self.calculate(data)
        
        buy_signals = results['bullish_engulfing']
        sell_signals = results['bearish_engulfing']
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_strength': results['pattern_strength']
        }


def test_engulfing_pattern():
    """Test the Engulfing Pattern indicator with sample data."""
    print("Testing Engulfing Pattern Indicator...")
    
    # Create sample data with engulfing patterns
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
    
    # Add a bullish engulfing pattern at indices 50-51
    if n_periods > 51:
        # Small bearish candle
        data[50][0] = 100.0  # open
        data[50][1] = 100.2  # high
        data[50][2] = 98.5   # low
        data[50][3] = 99.0   # close
        
        # Large bullish engulfing candle
        data[51][0] = 98.0   # open (below previous close)
        data[51][1] = 102.0  # high
        data[51][2] = 97.8   # low
        data[51][3] = 101.5  # close (above previous open)
        data[51][4] = 1500   # Higher volume
    
    # Add a bearish engulfing pattern at indices 70-71
    if n_periods > 71:
        # Small bullish candle
        data[70][0] = 105.0  # open
        data[70][1] = 106.5  # high
        data[70][2] = 104.8  # low
        data[70][3] = 106.0  # close
        
        # Large bearish engulfing candle
        data[71][0] = 107.0  # open (above previous close)
        data[71][1] = 107.2  # high
        data[71][2] = 103.0  # low
        data[71][3] = 103.5  # close (below previous open)
        data[71][4] = 1600   # Higher volume
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test the indicator
    indicator = EngulfingPattern(
        min_body_ratio=0.2,      # Allow smaller bodies for test
        engulfing_ratio=1.05,    # Smaller engulfing ratio for test
        volume_confirmation=True
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"Bullish engulfing signals: {sum(results['bullish_engulfing'])}")
    print(f"Bearish engulfing signals: {sum(results['bearish_engulfing'])}")
    
    # Find and display detected patterns
    for i, (bull, bear, strength) in enumerate(zip(
        results['bullish_engulfing'], 
        results['bearish_engulfing'],
        results['pattern_strength']
    )):
        if bull > 0:
            print(f"Bullish Engulfing at index {i}, strength: {strength:.1f}")
        if bear > 0:
            print(f"Bearish Engulfing at index {i}, strength: {strength:.1f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    buy_signals = sum(signals['buy_signals'])
    sell_signals = sum(signals['sell_signals'])
    
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    print("Engulfing Pattern test completed successfully!")
    return True


if __name__ == "__main__":
    test_engulfing_pattern()