"""
Abandoned Baby Signal Pattern Indicator

The Abandoned Baby is a three-candle reversal pattern that consists of:
- Bullish Abandoned Baby:
  1. Large bearish candle in a downtrend
  2. Doji that gaps down (opening below the low of the first candle)
  3. Bullish candle that gaps up (opening above the high of the doji)
- Bearish Abandoned Baby:
  1. Large bullish candle in an uptrend
  2. Doji that gaps up (opening above the high of the first candle)
  3. Bearish candle that gaps down (opening below the low of the doji)

The doji in the middle represents indecision and the gaps indicate strong momentum shifts.

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


class AbandonedBabySignal(StandardIndicatorInterface):
    """
    Abandoned Baby Signal Pattern Indicator
    
    Identifies bullish and bearish abandoned baby patterns which are
    rare but very strong reversal signals.
    """
    
    def __init__(self, 
                 min_first_body_ratio: float = 0.6,
                 doji_body_threshold: float = 0.1,
                 min_gap_ratio: float = 0.001,
                 trend_lookback: int = 10,
                 trend_threshold: float = 0.02):
        """
        Initialize the Abandoned Baby Signal indicator.
        
        Args:
            min_first_body_ratio: Minimum body ratio for first candle (default: 0.6)
            doji_body_threshold: Maximum body ratio for doji identification (default: 0.1)
            min_gap_ratio: Minimum gap size as ratio to price (default: 0.1%)
            trend_lookback: Number of periods to determine trend (default: 10)
            trend_threshold: Minimum price change % to identify trend (default: 2%)
        """
        self.min_first_body_ratio = min_first_body_ratio
        self.doji_body_threshold = doji_body_threshold
        self.min_gap_ratio = min_gap_ratio
        self.trend_lookback = trend_lookback
        self.trend_threshold = trend_threshold
        
        # Parameters dict for base class
        self.parameters = {
            'min_first_body_ratio': min_first_body_ratio,
            'doji_body_threshold': doji_body_threshold,
            'min_gap_ratio': min_gap_ratio,
            'trend_lookback': trend_lookback,
            'trend_threshold': trend_threshold
        }
        
        # Initialize result storage
        self.bullish_abandoned_baby = []
        self.bearish_abandoned_baby = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Abandoned Baby Signal",
            category="pattern",
            description="Three-candle reversal pattern with gapped doji in middle",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=self.trend_lookback + 3
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 0.3 <= self.min_first_body_ratio <= 1.0:
            raise ValueError("min_first_body_ratio must be between 0.3 and 1.0")
        if not 0.01 <= self.doji_body_threshold <= 0.3:
            raise ValueError("doji_body_threshold must be between 0.01 and 0.3")
        if not 0.0001 <= self.min_gap_ratio <= 0.05:
            raise ValueError("min_gap_ratio must be between 0.01% and 5%")
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
            'is_doji': body / total_range <= self.doji_body_threshold if total_range > 0 else False
        }
    
    def _has_gap_down(self, high1: float, open2: float, close1: float) -> bool:
        """Check if there's a gap down between two candles."""
        gap_size = (high1 - open2) / close1 if close1 > 0 else 0
        return open2 < high1 and gap_size >= self.min_gap_ratio
    
    def _has_gap_up(self, low1: float, open2: float, close1: float) -> bool:
        """Check if there's a gap up between two candles."""
        gap_size = (open2 - low1) / close1 if close1 > 0 else 0
        return open2 > low1 and gap_size >= self.min_gap_ratio
    
    def _determine_trend(self, prices: List[float], current_idx: int) -> str:
        """
        Determine the trend preceding the current candles.
        
        Args:
            prices: List of prices (typically closing prices)
            current_idx: Current index (third candle)
            
        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        start_idx = max(0, current_idx - self.trend_lookback)
        
        if start_idx >= current_idx - 2:
            return 'sideways'
        
        start_price = prices[start_idx]
        end_price = prices[current_idx - 3]  # Price before first candle
        
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
                                  candles: List[Dict],
                                  trend: str,
                                  pattern_type: str) -> float:
        """
        Calculate the strength of the abandoned baby pattern.
        
        Args:
            candles: List of three candle data dictionaries
            trend: Preceding trend
            pattern_type: 'bullish' or 'bearish'
            
        Returns:
            Pattern strength score (0-100)
        """
        first, doji, third = candles
        
        strength = 70  # High base strength due to rarity
        
        # Strength based on first candle body size
        strength += (first['body_ratio'] - self.min_first_body_ratio) * 20
        
        # Strength based on doji quality (smaller body is better)
        doji_quality = (self.doji_body_threshold - doji['body_ratio']) / self.doji_body_threshold
        strength += doji_quality * 10
        
        # Strength based on third candle body size
        strength += min(third['body_ratio'] * 15, 10)
        
        # Strength based on gap sizes
        if pattern_type == 'bullish':
            # Gap down from first to doji
            gap1_size = (first['low'] - doji['open']) / first['close'] if first['close'] > 0 else 0
            # Gap up from doji to third
            gap2_size = (third['open'] - doji['high']) / doji['close'] if doji['close'] > 0 else 0
        else:
            # Gap up from first to doji
            gap1_size = (doji['open'] - first['high']) / first['close'] if first['close'] > 0 else 0
            # Gap down from doji to third
            gap2_size = (doji['low'] - third['open']) / doji['close'] if doji['close'] > 0 else 0
        
        gap_strength = min((gap1_size + gap2_size) * 500, 15)
        strength += gap_strength
        
        # Perfect trend context
        if ((pattern_type == 'bullish' and trend == 'downtrend') or
            (pattern_type == 'bearish' and trend == 'uptrend')):
            strength += 15
        
        return min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Abandoned Baby Signal patterns.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - bullish_abandoned_baby: List of bullish signals (1.0 if pattern, 0.0 otherwise)
            - bearish_abandoned_baby: List of bearish signals (1.0 if pattern, 0.0 otherwise)
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
        self.bullish_abandoned_baby = [0.0] * len(df)
        self.bearish_abandoned_baby = [0.0] * len(df)
        self.pattern_strength = [0.0] * len(df)
        
        # Need enough data for trend analysis plus 3 candles
        if len(df) <= self.trend_lookback + 2:
            return self._get_results()
        
        # Extract closing prices for trend analysis
        close_prices = df['close'].tolist()
        
        for i in range(self.trend_lookback + 2, len(df)):
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
                
                # Add metrics
                metrics = self._get_candle_metrics(
                    candle['open'], candle['high'], 
                    candle['low'], candle['close']
                )
                candle.update(metrics)
                candles.append(candle)
            
            first, doji, third = candles
            
            # Check for bullish abandoned baby
            if (first['is_bearish'] and 
                first['body_ratio'] >= self.min_first_body_ratio and
                doji['is_doji'] and
                self._has_gap_down(first['low'], doji['open'], first['close']) and
                third['is_bullish'] and
                self._has_gap_up(doji['high'], third['open'], doji['close'])):
                
                trend = self._determine_trend(close_prices, i)
                if trend == 'downtrend':
                    self.bullish_abandoned_baby[i] = 1.0
                    strength = self._calculate_pattern_strength(candles, trend, 'bullish')
                    self.pattern_strength[i] = strength
            
            # Check for bearish abandoned baby
            elif (first['is_bullish'] and 
                  first['body_ratio'] >= self.min_first_body_ratio and
                  doji['is_doji'] and
                  self._has_gap_up(first['high'], doji['open'], first['close']) and
                  third['is_bearish'] and
                  self._has_gap_down(doji['low'], third['open'], doji['close'])):
                
                trend = self._determine_trend(close_prices, i)
                if trend == 'uptrend':
                    self.bearish_abandoned_baby[i] = 1.0
                    strength = self._calculate_pattern_strength(candles, trend, 'bearish')
                    self.pattern_strength[i] = strength
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'bullish_abandoned_baby': self.bullish_abandoned_baby,
            'bearish_abandoned_baby': self.bearish_abandoned_baby,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on Abandoned Baby patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with buy/sell signals and strength
        """
        results = self.calculate(data)
        
        buy_signals = results['bullish_abandoned_baby']
        sell_signals = results['bearish_abandoned_baby']
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_strength': results['pattern_strength']
        }


def test_abandoned_baby_signal():
    """Test the Abandoned Baby Signal indicator with sample data."""
    print("Testing Abandoned Baby Signal Pattern Indicator...")
    
    # Create sample data with abandoned baby patterns
    np.random.seed(42)
    n_periods = 60
    
    # Generate base price data with trends
    data = []
    
    # Create downtrend for first 25 periods
    for i in range(25):
        base_price = 100 - (i * 0.2)  # Declining trend
        open_price = base_price + np.random.randn() * 0.05
        close_price = open_price - abs(np.random.randn()) * 0.1  # Bearish bias
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.05
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.05
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add bullish abandoned baby pattern
    # Large bearish candle
    data.append([95.2, 95.3, 93.0, 93.5, 1200])
    # Doji with gap down
    data.append([92.8, 93.1, 92.5, 92.9, 800])  # Gap down from 93.0
    # Bullish candle with gap up
    data.append([93.3, 95.0, 93.2, 94.5, 1100])  # Gap up from 93.1
    
    # Create uptrend for next 25 periods
    for i in range(25):
        base_price = 94.5 + (i * 0.15)  # Rising trend
        open_price = base_price + np.random.randn() * 0.05
        close_price = open_price + abs(np.random.randn()) * 0.1  # Bullish bias
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.05
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.05
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    # Add bearish abandoned baby pattern
    # Large bullish candle
    data.append([98.0, 100.0, 97.8, 99.5, 1300])
    # Doji with gap up
    data.append([100.2, 100.5, 100.0, 100.3, 900])  # Gap up from 100.0
    # Bearish candle with gap down
    data.append([99.8, 100.0, 98.0, 98.5, 1200])  # Gap down from 100.0
    
    # Add a few more periods
    for i in range(7):
        base_price = 98.5 + np.random.randn() * 0.2
        open_price = base_price + np.random.randn() * 0.05
        close_price = open_price + np.random.randn() * 0.1
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.05
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.05
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test the indicator
    indicator = AbandonedBabySignal(
        min_first_body_ratio=0.5,      # Allow smaller first bodies for test
        doji_body_threshold=0.15,      # Allow larger doji bodies for test
        min_gap_ratio=0.0005,          # Smaller minimum gap for test
        trend_lookback=8,              # Shorter lookback for test
        trend_threshold=0.01           # 1% trend threshold
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"Bullish abandoned baby signals: {sum(results['bullish_abandoned_baby'])}")
    print(f"Bearish abandoned baby signals: {sum(results['bearish_abandoned_baby'])}")
    
    # Find and display detected patterns
    for i, (bullish, bearish, strength) in enumerate(zip(
        results['bullish_abandoned_baby'], 
        results['bearish_abandoned_baby'],
        results['pattern_strength']
    )):
        if bullish > 0:
            print(f"Bullish Abandoned Baby at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
        if bearish > 0:
            print(f"Bearish Abandoned Baby at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    buy_signals = sum(signals['buy_signals'])
    sell_signals = sum(signals['sell_signals'])
    
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    print("Abandoned Baby Signal test completed successfully!")
    return True


if __name__ == "__main__":
    test_abandoned_baby_signal()