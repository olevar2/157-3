"""
Doji Pattern Indicator

The Doji Pattern identifies various types of doji candlestick patterns:
- Standard Doji: Open and close are very close, indicating indecision
- Long-Legged Doji: Standard doji with long upper and lower shadows
- Dragonfly Doji: Open and close at the high, long lower shadow, no upper shadow
- Gravestone Doji: Open and close at the low, long upper shadow, no lower shadow

Doji patterns typically signal potential reversal or indecision in the market.

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


class DojiPattern(StandardIndicatorInterface):
    """
    Doji Pattern Indicator
    
    Identifies different types of doji patterns which indicate market indecision
    and potential trend reversals.
    """
    
    def __init__(self, 
                 doji_threshold: float = 0.1,
                 long_shadow_ratio: float = 2.0,
                 min_total_range: float = 0.001):
        """
        Initialize the Doji Pattern indicator.
        
        Args:
            doji_threshold: Maximum body to total range ratio for doji identification (default: 0.1)
            long_shadow_ratio: Minimum shadow to body ratio for long-legged doji (default: 2.0)
            min_total_range: Minimum total range relative to price for valid doji (default: 0.1%)
        """
        self.doji_threshold = doji_threshold
        self.long_shadow_ratio = long_shadow_ratio
        self.min_total_range = min_total_range
        
        # Parameters dict for base class
        self.parameters = {
            'doji_threshold': doji_threshold,
            'long_shadow_ratio': long_shadow_ratio,
            'min_total_range': min_total_range
        }
        
        # Initialize result storage
        self.standard_doji = []
        self.long_legged_doji = []
        self.dragonfly_doji = []
        self.gravestone_doji = []
        self.pattern_strength = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Doji Pattern",
            category="pattern",
            description="Various doji candlestick patterns indicating market indecision",
            parameters=self.parameters,
            input_requirements=['open', 'high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=1
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        if not 0.01 <= self.doji_threshold <= 0.5:
            raise ValueError("doji_threshold must be between 0.01 and 0.5")
        if not 1.0 <= self.long_shadow_ratio <= 10.0:
            raise ValueError("long_shadow_ratio must be between 1.0 and 10.0")
        if not 0.0001 <= self.min_total_range <= 0.01:
            raise ValueError("min_total_range must be between 0.01% and 1%")
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
        
        # Calculate relative to price for validation
        price_ref = (open_price + close) / 2 if (open_price + close) > 0 else 1
        total_range_pct = total_range / price_ref
        
        return {
            'body': body,
            'total_range': total_range,
            'total_range_pct': total_range_pct,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'body_ratio': body / total_range if total_range > 0 else 0,
            'upper_shadow_ratio': upper_shadow / body if body > 0 else float('inf'),
            'lower_shadow_ratio': lower_shadow / body if body > 0 else float('inf'),
            'is_bullish': close > open_price,
            'is_bearish': close < open_price
        }
    
    def _classify_doji(self, metrics: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify the type of doji pattern.
        
        Args:
            metrics: Candle metrics
            
        Returns:
            Tuple of (doji_type, strength) where doji_type is one of:
            'standard', 'long_legged', 'dragonfly', 'gravestone', 'none'
        """
        # Check if it's a valid doji
        if metrics['body_ratio'] > self.doji_threshold:
            return 'none', 0.0
        
        # Check minimum range requirement
        if metrics['total_range_pct'] < self.min_total_range:
            return 'none', 0.0
        
        # Base strength for any doji
        strength = 60
        
        # Perfect doji bonus
        if metrics['body'] == 0:
            strength += 10
        else:
            # Better strength for smaller bodies
            strength += (1 - metrics['body_ratio'] / self.doji_threshold) * 10
        
        # Classify doji type based on shadows
        upper_long = metrics['upper_shadow_ratio'] >= self.long_shadow_ratio
        lower_long = metrics['lower_shadow_ratio'] >= self.long_shadow_ratio
        
        # Check for specialized doji types
        if metrics['upper_shadow'] < metrics['body'] and lower_long:
            # Dragonfly doji: long lower shadow, minimal upper shadow
            doji_type = 'dragonfly'
            strength += 15  # Stronger reversal signal
            
        elif metrics['lower_shadow'] < metrics['body'] and upper_long:
            # Gravestone doji: long upper shadow, minimal lower shadow
            doji_type = 'gravestone'
            strength += 15  # Stronger reversal signal
            
        elif upper_long and lower_long:
            # Long-legged doji: both shadows are long
            doji_type = 'long_legged'
            strength += 10  # Indicates high volatility and indecision
            
        else:
            # Standard doji
            doji_type = 'standard'
            strength += 5
        
        return doji_type, min(strength, 100)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Doji Pattern signals.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - standard_doji: List of standard doji signals (1.0 if pattern, 0.0 otherwise)
            - long_legged_doji: List of long-legged doji signals
            - dragonfly_doji: List of dragonfly doji signals
            - gravestone_doji: List of gravestone doji signals
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
        self.standard_doji = [0.0] * len(df)
        self.long_legged_doji = [0.0] * len(df)
        self.dragonfly_doji = [0.0] * len(df)
        self.gravestone_doji = [0.0] * len(df)
        self.pattern_strength = [0.0] * len(df)
        
        for i in range(len(df)):
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
            
            # Classify doji type
            doji_type, strength = self._classify_doji(metrics)
            
            if doji_type != 'none':
                self.pattern_strength[i] = strength
                
                if doji_type == 'standard':
                    self.standard_doji[i] = 1.0
                elif doji_type == 'long_legged':
                    self.long_legged_doji[i] = 1.0
                elif doji_type == 'dragonfly':
                    self.dragonfly_doji[i] = 1.0
                elif doji_type == 'gravestone':
                    self.gravestone_doji[i] = 1.0
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        return {
            'standard_doji': self.standard_doji,
            'long_legged_doji': self.long_legged_doji,
            'dragonfly_doji': self.dragonfly_doji,
            'gravestone_doji': self.gravestone_doji,
            'pattern_strength': self.pattern_strength
        }
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on Doji patterns.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with indecision/reversal signals and strength
        """
        results = self.calculate(data)
        
        # Combine all doji types into general indecision signal
        indecision_signals = []
        reversal_signals = []
        
        for i in range(len(results['standard_doji'])):
            # Any doji indicates indecision
            indecision = max(
                results['standard_doji'][i],
                results['long_legged_doji'][i],
                results['dragonfly_doji'][i],
                results['gravestone_doji'][i]
            )
            indecision_signals.append(indecision)
            
            # Dragonfly and gravestone are stronger reversal signals
            reversal = max(
                results['dragonfly_doji'][i],
                results['gravestone_doji'][i]
            )
            reversal_signals.append(reversal)
        
        return {
            'indecision_signals': indecision_signals,
            'reversal_signals': reversal_signals,
            'signal_strength': results['pattern_strength']
        }


def test_doji_pattern():
    """Test the Doji Pattern indicator with sample data."""
    print("Testing Doji Pattern Indicator...")
    
    # Create sample data with different doji patterns
    data = []
    
    # Normal candles
    for i in range(20):
        open_price = 100 + i * 0.1
        close_price = open_price + np.random.randn() * 0.5
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.3
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.3
        volume = 1000
        data.append([open_price, high, low, close_price, volume])
    
    # Add standard doji at index 20
    data.append([102.0, 102.3, 101.7, 102.02, 1100])  # Small body, modest shadows
    
    # Add long-legged doji at index 21
    data.append([103.0, 104.5, 101.5, 103.05, 1200])  # Small body, long both shadows
    
    # Add dragonfly doji at index 22
    data.append([104.0, 104.1, 102.0, 104.05, 1150])  # Small body at top, long lower shadow
    
    # Add gravestone doji at index 23
    data.append([105.0, 107.0, 104.95, 105.02, 1250])  # Small body at bottom, long upper shadow
    
    # Add more normal candles
    for i in range(10):
        open_price = 105 + i * 0.1
        close_price = open_price + np.random.randn() * 0.5
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.3
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.3
        volume = 1000
        data.append([open_price, high, low, close_price, volume])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test the indicator
    indicator = DojiPattern(
        doji_threshold=0.15,      # Allow slightly larger bodies for test
        long_shadow_ratio=1.5,    # Lower threshold for test
        min_total_range=0.0005    # 0.05% minimum range
    )
    
    results = indicator.calculate(df)
    
    print(f"Data points: {len(df)}")
    print(f"Standard doji signals: {sum(results['standard_doji'])}")
    print(f"Long-legged doji signals: {sum(results['long_legged_doji'])}")
    print(f"Dragonfly doji signals: {sum(results['dragonfly_doji'])}")
    print(f"Gravestone doji signals: {sum(results['gravestone_doji'])}")
    
    # Find and display detected patterns
    for i in range(len(df)):
        strength = results['pattern_strength'][i]
        if strength > 0:
            pattern_types = []
            if results['standard_doji'][i] > 0:
                pattern_types.append('Standard')
            if results['long_legged_doji'][i] > 0:
                pattern_types.append('Long-Legged')
            if results['dragonfly_doji'][i] > 0:
                pattern_types.append('Dragonfly')
            if results['gravestone_doji'][i] > 0:
                pattern_types.append('Gravestone')
            
            if pattern_types:
                print(f"{', '.join(pattern_types)} Doji at index {i}, strength: {strength:.1f}, price: {df.iloc[i]['close']:.2f}")
    
    # Test signals
    signals = indicator.get_signals(df)
    indecision_signals = sum(signals['indecision_signals'])
    reversal_signals = sum(signals['reversal_signals'])
    
    print(f"Indecision signals: {indecision_signals}")
    print(f"Reversal signals: {reversal_signals}")
    
    print("Doji Pattern test completed successfully!")
    return True


if __name__ == "__main__":
    test_doji_pattern()