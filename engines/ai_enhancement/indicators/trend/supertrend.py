"""
SuperTrend Indicator
====================

SuperTrend is a trend-following indicator that provides dynamic support and resistance levels.
It uses the Average True Range (ATR) to calculate bands around a moving average and provides
clear buy/sell signals when price crosses above or below the trend line.

The indicator oscillates between support and resistance levels:
- When price is above SuperTrend: Bullish trend (Green line)
- When price is below SuperTrend: Bearish trend (Red line)

Author: Platform3 AI Enhancement Engine
Created: 2024
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any
import warnings

# For standalone testing
import sys
import os

try:
    from base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        def __init__(self):
            pass

class SuperTrend(BaseIndicator):
    """
    SuperTrend Indicator
    
    A trend-following indicator that uses ATR-based bands to determine trend direction.
    Provides dynamic support/resistance levels and clear trend signals.
    """
    
    def __init__(self, 
                 period: int = 10,
                 multiplier: float = 3.0):
        """
        Initialize SuperTrend indicator.
        
        Parameters:
        -----------
        period : int, default=10
            Period for ATR calculation
        multiplier : float, default=3.0
            Multiplier for ATR bands
        """
        super().__init__()
        self.period = period
        self.multiplier = multiplier
        
        if self.period < 1:
            raise ValueError("Period must be positive")
        if self.multiplier <= 0:
            raise ValueError("Multiplier must be positive")
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate True Range.
        
        Parameters:
        -----------
        high, low, close : np.ndarray
            OHLC price arrays
            
        Returns:
        --------
        np.ndarray
            True Range values
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first value
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate Average True Range using Wilder's smoothing.
        
        Parameters:
        -----------
        high, low, close : np.ndarray
            OHLC price arrays
            
        Returns:
        --------
        np.ndarray
            ATR values
        """
        tr = self._calculate_true_range(high, low, close)
        
        # Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1.0 / self.period
        atr = np.zeros_like(tr)
        
        # Initialize first ATR as simple average of first 'period' TR values
        if len(tr) >= self.period:
            atr[self.period - 1] = np.mean(tr[:self.period])
            
            # Apply Wilder's smoothing for subsequent values
            for i in range(self.period, len(tr)):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        
        # Fill earlier values with NaN
        atr[:self.period - 1] = np.nan
        
        return atr
    
    def calculate(self, 
                  high: Union[pd.Series, np.ndarray],
                  low: Union[pd.Series, np.ndarray], 
                  close: Union[pd.Series, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate SuperTrend indicator.
        
        Parameters:
        -----------
        high, low, close : pd.Series or np.ndarray
            OHLC price data
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'supertrend': SuperTrend line values
            - 'trend': Trend direction (1 for up, -1 for down)
            - 'upper_band': Upper band values
            - 'lower_band': Lower band values
            - 'signals': Buy/sell signals (1 for buy, -1 for sell, 0 for hold)
        """
        # Convert to numpy arrays
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        
        n = len(close)
        
        if n < self.period:
            return {
                'supertrend': np.full(n, np.nan),
                'trend': np.full(n, np.nan),
                'upper_band': np.full(n, np.nan),
                'lower_band': np.full(n, np.nan),
                'signals': np.full(n, 0)
            }
        
        # Calculate ATR
        atr = self._calculate_atr(high, low, close)
        
        # Calculate median price (HL2)
        median_price = (high + low) / 2
        
        # Calculate basic bands
        upper_band = median_price + (self.multiplier * atr)
        lower_band = median_price - (self.multiplier * atr)
        
        # Initialize final bands
        final_upper_band = np.full(n, np.nan)
        final_lower_band = np.full(n, np.nan)
        
        # Calculate final bands with persistence logic
        for i in range(1, n):
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                # Upper band logic
                if np.isnan(final_upper_band[i-1]):
                    final_upper_band[i] = upper_band[i]
                else:
                    if upper_band[i] < final_upper_band[i-1] or close[i-1] > final_upper_band[i-1]:
                        final_upper_band[i] = upper_band[i]
                    else:
                        final_upper_band[i] = final_upper_band[i-1]
                
                # Lower band logic
                if np.isnan(final_lower_band[i-1]):
                    final_lower_band[i] = lower_band[i]
                else:
                    if lower_band[i] > final_lower_band[i-1] or close[i-1] < final_lower_band[i-1]:
                        final_lower_band[i] = lower_band[i]
                    else:
                        final_lower_band[i] = final_lower_band[i-1]
        
        # Set first values
        final_upper_band[0] = upper_band[0] if not np.isnan(upper_band[0]) else np.nan
        final_lower_band[0] = lower_band[0] if not np.isnan(lower_band[0]) else np.nan
        
        # Calculate SuperTrend and trend direction
        supertrend = np.full(n, np.nan)
        trend = np.full(n, np.nan)
        
        for i in range(n):
            if not np.isnan(final_upper_band[i]) and not np.isnan(final_lower_band[i]):
                if i == 0:
                    # Initialize trend
                    if close[i] <= final_upper_band[i]:
                        supertrend[i] = final_upper_band[i]
                        trend[i] = -1  # Downtrend
                    else:
                        supertrend[i] = final_lower_band[i]
                        trend[i] = 1   # Uptrend
                else:
                    # Continue trend logic
                    prev_trend = trend[i-1] if not np.isnan(trend[i-1]) else 1
                    
                    if prev_trend == 1:  # Previous uptrend
                        if close[i] > final_lower_band[i]:
                            supertrend[i] = final_lower_band[i]
                            trend[i] = 1
                        else:
                            supertrend[i] = final_upper_band[i]
                            trend[i] = -1
                    else:  # Previous downtrend
                        if close[i] < final_upper_band[i]:
                            supertrend[i] = final_upper_band[i]
                            trend[i] = -1
                        else:
                            supertrend[i] = final_lower_band[i]
                            trend[i] = 1
        
        # Generate signals
        signals = np.zeros(n)
        for i in range(1, n):
            if not np.isnan(trend[i]) and not np.isnan(trend[i-1]):
                if trend[i-1] == -1 and trend[i] == 1:
                    signals[i] = 1   # Buy signal
                elif trend[i-1] == 1 and trend[i] == -1:
                    signals[i] = -1  # Sell signal
        
        return {
            'supertrend': supertrend,
            'trend': trend,
            'upper_band': final_upper_band,
            'lower_band': final_lower_band,
            'signals': signals
        }
    
    def get_signals(self, 
                    high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    close: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Get trading signals from SuperTrend.
        
        Parameters:
        -----------
        high, low, close : pd.Series or np.ndarray
            OHLC price data
            
        Returns:
        --------
        dict
            Dictionary with current signal information
        """
        result = self.calculate(high, low, close)
        
        if len(result['signals']) == 0:
            return {
                'current_trend': 'Unknown',
                'signal': 'Hold',
                'supertrend_value': np.nan,
                'strength': 0.0
            }
        
        current_trend = result['trend'][-1]
        current_signal = result['signals'][-1]
        current_supertrend = result['supertrend'][-1]
        current_close = close[-1] if hasattr(close, '__getitem__') else close
        
        # Determine signal
        if current_signal == 1:
            signal = 'Buy'
        elif current_signal == -1:
            signal = 'Sell'
        else:
            signal = 'Hold'
        
        # Determine trend
        if current_trend == 1:
            trend_name = 'Uptrend'
        elif current_trend == -1:
            trend_name = 'Downtrend'
        else:
            trend_name = 'Unknown'
        
        # Calculate signal strength based on distance from SuperTrend
        if not np.isnan(current_supertrend) and not np.isnan(current_close):
            distance = abs(current_close - current_supertrend) / current_supertrend
            strength = min(1.0, distance * 10)  # Normalize to 0-1
        else:
            strength = 0.0
        
        return {
            'current_trend': trend_name,
            'signal': signal,
            'supertrend_value': current_supertrend,
            'strength': strength,
            'distance_ratio': distance if not np.isnan(current_supertrend) else np.nan
        }

def test_supertrend():
    """Test the SuperTrend indicator with sample data."""
    print("Testing SuperTrend Indicator")
    print("=" * 50)
    
    # Create sample OHLC data
    np.random.seed(42)
    n_points = 100
    
    # Generate trending price data
    base_price = 100
    trend = np.linspace(0, 20, n_points)  # Upward trend
    noise = np.random.randn(n_points) * 2
    
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n_points)) * 1.5
    low = close - np.abs(np.random.randn(n_points)) * 1.5
    
    # Ensure high >= close >= low
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    
    # Test with different parameters
    test_configs = [
        {"period": 10, "multiplier": 3.0},
        {"period": 14, "multiplier": 2.0},
        {"period": 7, "multiplier": 1.5}
    ]
    
    for config in test_configs:
        print(f"\nTesting SuperTrend with period={config['period']}, multiplier={config['multiplier']}:")
        print("-" * 60)
        
        # Initialize indicator
        supertrend = SuperTrend(period=config['period'], multiplier=config['multiplier'])
        
        # Calculate SuperTrend
        result = supertrend.calculate(high, low, close)
        
        # Get current signals
        signals = supertrend.get_signals(high, low, close)
        
        # Count signals
        buy_signals = np.sum(result['signals'] == 1)
        sell_signals = np.sum(result['signals'] == -1)
        
        # Calculate trend persistence
        trend_values = result['trend'][~np.isnan(result['trend'])]
        if len(trend_values) > 0:
            uptrend_periods = np.sum(trend_values == 1)
            downtrend_periods = np.sum(trend_values == -1)
            trend_persistence = max(uptrend_periods, downtrend_periods) / len(trend_values)
        else:
            trend_persistence = 0
        
        print(f"SuperTrend Results:")
        print(f"  Current Trend: {signals['current_trend']}")
        print(f"  Current Signal: {signals['signal']}")
        print(f"  SuperTrend Value: {signals['supertrend_value']:.2f}")
        print(f"  Signal Strength: {signals['strength']:.2f}")
        print(f"  Total Buy Signals: {buy_signals}")
        print(f"  Total Sell Signals: {sell_signals}")
        print(f"  Trend Persistence: {trend_persistence:.2f}")
        
        # Show last few values
        valid_indices = ~np.isnan(result['supertrend'])
        if np.any(valid_indices):
            last_5_indices = np.where(valid_indices)[0][-5:]
            print(f"  Last 5 SuperTrend values:")
            for idx in last_5_indices:
                trend_str = "UP" if result['trend'][idx] == 1 else "DOWN" if result['trend'][idx] == -1 else "NA"
                signal_str = "BUY" if result['signals'][idx] == 1 else "SELL" if result['signals'][idx] == -1 else ""
                print(f"    [{idx}] ST: {result['supertrend'][idx]:.2f}, Trend: {trend_str}, Close: {close[idx]:.2f} {signal_str}")
    
    # Test edge cases
    print(f"\nTesting Edge Cases:")
    print("-" * 30)
    
    # Test with insufficient data
    small_data = close[:5]
    small_high = high[:5]
    small_low = low[:5]
    
    supertrend_small = SuperTrend(period=10, multiplier=3.0)
    result_small = supertrend_small.calculate(small_high, small_low, small_data)
    
    print(f"Insufficient data test:")
    print(f"  All NaN SuperTrend: {np.all(np.isnan(result_small['supertrend']))}")
    print(f"  All zero signals: {np.all(result_small['signals'] == 0)}")
    
    # Test with flat data
    flat_close = np.full(50, 100.0)
    flat_high = np.full(50, 101.0)
    flat_low = np.full(50, 99.0)
    
    supertrend_flat = SuperTrend(period=10, multiplier=3.0)
    result_flat = supertrend_flat.calculate(flat_high, flat_low, flat_close)
    
    valid_flat = ~np.isnan(result_flat['supertrend'])
    if np.any(valid_flat):
        flat_variance = np.var(result_flat['supertrend'][valid_flat])
        print(f"Flat data test:")
        print(f"  SuperTrend variance: {flat_variance:.6f} (should be low)")
    
    print(f"\nTest completed successfully!")

if __name__ == "__main__":
    test_supertrend()