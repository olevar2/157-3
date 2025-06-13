"""
Kicking Signal Pattern Indicator

The Kicking Signal is a powerful two-candle reversal pattern characterized by:
- Two consecutive marubozu candles (candles with little to no shadows)
- Gap between the two candles (opening price of second candle is beyond the range of first candle)
- Bullish Kicking: Black marubozu followed by white marubozu with gap up
- Bearish Kicking: White marubozu followed by black marubozu with gap down

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


class KickingSignal(StandardIndicatorInterface):
    """
    Kicking Signal Pattern Indicator
    
    A two-candle reversal pattern that signals strong momentum changes.
    The pattern requires two consecutive marubozu candles with a gap between them.
    """
    
    def __init__(self, 
                 marubozu_threshold: float = 0.1,
                 gap_threshold: float = 0.001,
                 volume_confirmation: bool = True):
        """
        Initialize the Kicking Signal indicator.
        
        Args:
            marubozu_threshold: Maximum shadow ratio to body ratio for marubozu identification (default: 0.1)
            gap_threshold: Minimum gap percentage between candles (default: 0.1%)
            volume_confirmation: Whether to require volume confirmation (default: True)
        """
        self.marubozu_threshold = marubozu_threshold
        self.gap_threshold = gap_threshold
        self.volume_confirmation = volume_confirmation
        
        # Parameters dict for base class
        self.parameters = {
            'marubozu_threshold': marubozu_threshold,
            'gap_threshold': gap_threshold,
            'volume_confirmation': volume_confirmation
        }
        
        # Initialize result storage
        self.bullish_kicking = []
        self.bearish_kicking = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Kicking Signal",
            category="pattern",
            description="Two-candle reversal pattern with marubozu candles and gaps",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=2
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 0 < self.marubozu_threshold <= 1.0:
            raise ValueError("marubozu_threshold must be between 0 and 1.0")
        if not 0 <= self.gap_threshold <= 0.1:
            raise ValueError("gap_threshold must be between 0 and 0.1 (10%)")
        return True
        
    def _is_marubozu(self, open_price: float, high: float, low: float, close: float) -> Tuple[bool, str]:
        """
        Check if a candle is a marubozu (little to no shadows).
        
        Args:
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            
        Returns:
            Tuple of (is_marubozu, type) where type is 'white', 'black', or 'none'
        """
        if high == low:  # Doji-like candle
            return False, 'none'
            
        body = abs(close - open_price)
        if body == 0:  # Perfect doji
            return False, 'none'
            
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        # Calculate shadow to body ratios
        upper_ratio = upper_shadow / body if body > 0 else float('inf')
        lower_ratio = lower_shadow / body if body > 0 else float('inf')
        
        # Check if both shadows are small relative to body
        is_marubozu = (upper_ratio <= self.marubozu_threshold and 
                      lower_ratio <= self.marubozu_threshold)
        
        if is_marubozu:
            candle_type = 'white' if close > open_price else 'black'
            return True, candle_type
        
        return False, 'none'
    
    def _check_gap(self, prev_high: float, prev_low: float, 
                   curr_open: float, prev_close: float) -> Tuple[bool, str]:
        """
        Check if there's a significant gap between candles.
        
        Args:
            prev_high: Previous candle's high
            prev_low: Previous candle's low  
            curr_open: Current candle's open
            prev_close: Previous candle's close
            
        Returns:
            Tuple of (has_gap, gap_type) where gap_type is 'up', 'down', or 'none'
        """
        # Calculate gap percentage
        gap_up_pct = (curr_open - prev_high) / prev_close if prev_close > 0 else 0
        gap_down_pct = (prev_low - curr_open) / prev_close if prev_close > 0 else 0
        
        if gap_up_pct >= self.gap_threshold:
            return True, 'up'
        elif gap_down_pct >= self.gap_threshold:
            return True, 'down'
        
        return False, 'none'
    
    def _calculate_pattern_strength(self, 
                                  prev_candle: Dict,
                                  curr_candle: Dict,
                                  gap_size: float) -> float:
        """
        Calculate the strength of the kicking pattern.
        
        Args:
            prev_candle: Previous candle data
            curr_candle: Current candle data
            gap_size: Size of the gap between candles
            
        Returns:
            Pattern strength score (0-100)
        """
        strength = 50  # Base strength
        
        # Add strength based on gap size
        strength += min(gap_size * 1000, 20)  # Max 20 points for gap
        
        # Add strength based on body sizes
        prev_body_pct = abs(prev_candle['close'] - prev_candle['open']) / prev_candle['open']
        curr_body_pct = abs(curr_candle['close'] - curr_candle['open']) / curr_candle['open']
        
        strength += min(prev_body_pct * 500, 15)  # Max 15 points for prev body
        strength += min(curr_body_pct * 500, 15)  # Max 15 points for curr body
        
        # Volume confirmation
        if self.volume_confirmation and 'volume' in curr_candle and 'volume' in prev_candle:
            if curr_candle['volume'] > prev_candle['volume']:
                strength += 10
        
        return min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Kicking Signal patterns.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - bullish_kicking: List of bullish kicking signals (1.0 if pattern, 0.0 otherwise)
            - bearish_kicking: List of bearish kicking signals (1.0 if pattern, 0.0 otherwise)
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
        self.bullish_kicking = [0.0] * len(df)
        self.bearish_kicking = [0.0] * len(df)
        self.pattern_strength = [0.0] * len(df)
        
        # Need at least 2 candles for the pattern
        if len(df) < 2:
            return self._get_results()
        
        for i in range(1, len(df)):
            prev_candle = {
                'open': df.iloc[i-1]['open'],
                'high': df.iloc[i-1]['high'],
                'low': df.iloc[i-1]['low'],
                'close': df.iloc[i-1]['close']
            }
            
            curr_candle = {
                'open': df.iloc[i]['open'],
                'high': df.iloc[i]['high'],
                'low': df.iloc[i]['low'],
                'close': df.iloc[i]['close']
            }
            
            # Add volume if available
            if 'volume' in df.columns:
                prev_candle['volume'] = df.iloc[i-1]['volume']
                curr_candle['volume'] = df.iloc[i]['volume']
            
            # Check if both candles are marubozu
            prev_is_marubozu, prev_type = self._is_marubozu(
                prev_candle['open'], prev_candle['high'], 
                prev_candle['low'], prev_candle['close']
            )
            
            curr_is_marubozu, curr_type = self._is_marubozu(
                curr_candle['open'], curr_candle['high'],
                curr_candle['low'], curr_candle['close']
            )
            
            if not (prev_is_marubozu and curr_is_marubozu):
                continue
            
            # Check for gap between candles
            has_gap, gap_type = self._check_gap(
                prev_candle['high'], prev_candle['low'],
                curr_candle['open'], prev_candle['close']
            )
            
            if not has_gap:
                continue
            
            # Identify kicking patterns
            gap_size = 0
            if gap_type == 'up':
                gap_size = (curr_candle['open'] - prev_candle['high']) / prev_candle['close']
            elif gap_type == 'down':
                gap_size = (prev_candle['low'] - curr_candle['open']) / prev_candle['close']
            
            # Bullish Kicking: Black marubozu followed by white marubozu with gap up
            if (prev_type == 'black' and curr_type == 'white' and gap_type == 'up'):
                self.bullish_kicking[i] = 1.0
                self.pattern_strength[i] = self._calculate_pattern_strength(
                    prev_candle, curr_candle, gap_size
                )
            
            # Bearish Kicking: White marubozu followed by black marubozu with gap down
            elif (prev_type == 'white' and curr_type == 'black' and gap_type == 'down'):
                self.bearish_kicking[i] = 1.0
                self.pattern_strength[i] = self._calculate_pattern_strength(
                    prev_candle, curr_candle, gap_size
                )
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'bullish_kicking': self.bullish_kicking,
            'bearish_kicking': self.bearish_kicking,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on Kicking patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with buy/sell signals and strength
        """
        results = self.calculate(data)
        
        buy_signals = results['bullish_kicking']
        sell_signals = results['bearish_kicking']
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_strength': results['pattern_strength']
        }


def test_kicking_signal():
    """Test the Kicking Signal indicator with sample data."""
    print("Testing Kicking Signal Pattern Indicator...")
    
    # Create sample data with kicking patterns
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
    
    # Add a bullish kicking pattern at index 50
    if n_periods > 52:
        # Black marubozu
        data[50][0] = 100.0  # open
        data[50][1] = 100.1  # high (small shadow)
        data[50][2] = 98.0   # low
        data[50][3] = 98.1   # close (small shadow)
        
        # Gap up white marubozu
        data[51][0] = 101.0  # open (gap up)
        data[51][1] = 103.0  # high
        data[51][2] = 100.9  # low (small shadow)
        data[51][3] = 102.9  # close (small shadow)
    
    # Add a bearish kicking pattern at index 70
    if n_periods > 72:
        # White marubozu
        data[70][0] = 105.0  # open
        data[70][1] = 107.0  # high
        data[70][2] = 104.9  # low (small shadow)
        data[70][3] = 106.9  # close (small shadow)
        
        # Gap down black marubozu
        data[71][0] = 103.0  # open (gap down)
        data[71][1] = 103.1  # high (small shadow)
        data[71][2] = 101.0  # low
        data[71][3] = 101.1  # close (small shadow)
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test the indicator
    indicator = KickingSignal(
        marubozu_threshold=0.2,  # Allow slightly larger shadows for test
        gap_threshold=0.005,     # 0.5% minimum gap
        volume_confirmation=True
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"Bullish kicking signals: {sum(results['bullish_kicking'])}")
    print(f"Bearish kicking signals: {sum(results['bearish_kicking'])}")
    
    # Find and display detected patterns
    for i, (bull, bear, strength) in enumerate(zip(
        results['bullish_kicking'], 
        results['bearish_kicking'],
        results['pattern_strength']
    )):
        if bull > 0:
            print(f"Bullish Kicking at index {i}, strength: {strength:.1f}")
        if bear > 0:
            print(f"Bearish Kicking at index {i}, strength: {strength:.1f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    buy_signals = sum(signals['buy_signals'])
    sell_signals = sum(signals['sell_signals'])
    
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    print("Kicking Signal test completed successfully!")
    return True


if __name__ == "__main__":
    test_kicking_signal()