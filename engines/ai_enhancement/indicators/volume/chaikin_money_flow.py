"""
Chaikin Money Flow (CMF) Indicator
===================================

The Chaikin Money Flow (CMF) indicator measures the amount of Money Flow Volume 
over a specific period. It combines price and volume to assess the strength of 
buying and selling pressure.

CMF oscillates between -1 and +1:
- Values above 0: Buying pressure (accumulation)
- Values below 0: Selling pressure (distribution)
- Values near +1/-1: Strong buying/selling pressure

Formula:
1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
2. Money Flow Volume = Money Flow Multiplier × Volume
3. CMF = Sum(Money Flow Volume, n) / Sum(Volume, n)

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        def __init__(self):
            pass

class ChaikinMoneyFlow(BaseIndicator):
    """
    Chaikin Money Flow (CMF) Indicator
    
    Measures the flow of money into and out of a security by combining price and volume.
    Useful for confirming trends and identifying potential reversals.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize Chaikin Money Flow indicator.
        
        Parameters:
        -----------
        period : int, default=20
            Number of periods for CMF calculation
        """
        super().__init__()
        self.period = period
        
        if self.period < 1:
            raise ValueError("Period must be positive")
    
    def _calculate_money_flow_multiplier(self, 
                                       high: np.ndarray, 
                                       low: np.ndarray, 
                                       close: np.ndarray) -> np.ndarray:
        """
        Calculate Money Flow Multiplier.
        
        Parameters:
        -----------
        high, low, close : np.ndarray
            Price arrays
            
        Returns:
        --------
        np.ndarray
            Money Flow Multiplier values
        """
        # Handle zero range case
        price_range = high - low
        
        # Money Flow Multiplier formula
        # ((Close - Low) - (High - Close)) / (High - Low)
        # Simplified: (2 * Close - High - Low) / (High - Low)
        
        multiplier = np.zeros_like(close)
        
        # Only calculate where price range is not zero
        non_zero_range = price_range != 0
        
        if np.any(non_zero_range):
            multiplier[non_zero_range] = (
                (2 * close[non_zero_range] - high[non_zero_range] - low[non_zero_range]) / 
                price_range[non_zero_range]
            )
        
        # For zero range (high == low), multiplier is 0
        # This happens in gaps or when there's no price movement
        
        # Ensure multiplier is within [-1, 1] range
        multiplier = np.clip(multiplier, -1, 1)
        
        return multiplier
    
    def calculate(self, 
                  high: Union[pd.Series, np.ndarray],
                  low: Union[pd.Series, np.ndarray],
                  close: Union[pd.Series, np.ndarray],
                  volume: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Calculate Chaikin Money Flow.
        
        Parameters:
        -----------
        high, low, close, volume : pd.Series or np.ndarray
            OHLCV price and volume data
            
        Returns:
        --------
        np.ndarray
            CMF values
        """
        # Convert to numpy arrays
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        if isinstance(volume, pd.Series):
            volume = volume.values
        
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        
        n = len(close)
        
        if n < self.period:
            return np.full(n, np.nan)
        
        # Calculate Money Flow Multiplier
        mf_multiplier = self._calculate_money_flow_multiplier(high, low, close)
        
        # Calculate Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Calculate CMF using rolling window
        cmf = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            start_idx = i - self.period + 1
            end_idx = i + 1
            
            # Sum of Money Flow Volume over period
            sum_mf_volume = np.sum(mf_volume[start_idx:end_idx])
            
            # Sum of Volume over period
            sum_volume = np.sum(volume[start_idx:end_idx])
            
            # Calculate CMF
            if sum_volume != 0:
                cmf[i] = sum_mf_volume / sum_volume
            else:
                cmf[i] = 0.0
        
        return cmf
    
    def get_signals(self, 
                    high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    close: Union[pd.Series, np.ndarray],
                    volume: Union[pd.Series, np.ndarray],
                    threshold_strong: float = 0.25,
                    threshold_weak: float = 0.05) -> Dict[str, Any]:
        """
        Generate trading signals from CMF.
        
        Parameters:
        -----------
        high, low, close, volume : pd.Series or np.ndarray
            OHLCV data
        threshold_strong : float, default=0.25
            Threshold for strong signals
        threshold_weak : float, default=0.05
            Threshold for weak signals
            
        Returns:
        --------
        dict
            Signal information
        """
        cmf_values = self.calculate(high, low, close, volume)
        
        if len(cmf_values) == 0 or np.all(np.isnan(cmf_values)):
            return {
                'signal': 'Hold',
                'strength': 'Neutral',
                'cmf_value': np.nan,
                'pressure': 'Unknown'
            }
        
        current_cmf = cmf_values[-1]
        
        # Determine signal based on thresholds
        if np.isnan(current_cmf):
            signal = 'Hold'
            strength = 'Neutral'
            pressure = 'Unknown'
        elif current_cmf > threshold_strong:
            signal = 'Buy'
            strength = 'Strong'
            pressure = 'Strong Buying'
        elif current_cmf > threshold_weak:
            signal = 'Buy'
            strength = 'Weak'
            pressure = 'Buying'
        elif current_cmf < -threshold_strong:
            signal = 'Sell'
            strength = 'Strong'
            pressure = 'Strong Selling'
        elif current_cmf < -threshold_weak:
            signal = 'Sell'
            strength = 'Weak'
            pressure = 'Selling'
        else:
            signal = 'Hold'
            strength = 'Neutral'
            pressure = 'Balanced'
        
        return {
            'signal': signal,
            'strength': strength,
            'cmf_value': current_cmf,
            'pressure': pressure
        }
    
    def analyze_divergence(self, 
                          high: Union[pd.Series, np.ndarray],
                          low: Union[pd.Series, np.ndarray],
                          close: Union[pd.Series, np.ndarray],
                          volume: Union[pd.Series, np.ndarray],
                          lookback: int = 20) -> Dict[str, Any]:
        """
        Analyze price-CMF divergence patterns.
        
        Parameters:
        -----------
        high, low, close, volume : pd.Series or np.ndarray
            OHLCV data
        lookback : int, default=20
            Periods to look back for divergence analysis
            
        Returns:
        --------
        dict
            Divergence analysis results
        """
        cmf_values = self.calculate(high, low, close, volume)
        
        if isinstance(close, pd.Series):
            close = close.values
        close = np.asarray(close, dtype=float)
        
        if len(cmf_values) < lookback or np.all(np.isnan(cmf_values[-lookback:])):
            return {
                'divergence_type': 'None',
                'strength': 0.0,
                'description': 'Insufficient data for divergence analysis'
            }
        
        # Get recent data
        recent_close = close[-lookback:]
        recent_cmf = cmf_values[-lookback:]
        
        # Remove NaN values
        valid_mask = ~np.isnan(recent_cmf)
        if np.sum(valid_mask) < lookback // 2:
            return {
                'divergence_type': 'None',
                'strength': 0.0,
                'description': 'Insufficient valid data for divergence analysis'
            }
        
        recent_close = recent_close[valid_mask]
        recent_cmf = recent_cmf[valid_mask]
        
        # Calculate price and CMF trends
        price_trend = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
        cmf_trend = np.polyfit(range(len(recent_cmf)), recent_cmf, 1)[0]
        
        # Determine divergence
        price_direction = 'up' if price_trend > 0 else 'down'
        cmf_direction = 'up' if cmf_trend > 0 else 'down'
        
        if price_direction != cmf_direction:
            if price_direction == 'up' and cmf_direction == 'down':
                divergence_type = 'Bearish'
                description = 'Price rising but CMF falling - potential reversal down'
            else:
                divergence_type = 'Bullish'
                description = 'Price falling but CMF rising - potential reversal up'
            
            # Calculate divergence strength
            strength = min(1.0, abs(price_trend - cmf_trend) / max(abs(price_trend), abs(cmf_trend), 0.001))
        else:
            divergence_type = 'None'
            description = 'Price and CMF moving in same direction'
            strength = 0.0
        
        return {
            'divergence_type': divergence_type,
            'strength': strength,
            'description': description,
            'price_trend': price_trend,
            'cmf_trend': cmf_trend
        }

def test_chaikin_money_flow():
    """Test the Chaikin Money Flow indicator with sample data."""
    print("Testing Chaikin Money Flow Indicator")
    print("=" * 50)
    
    # Create sample OHLCV data
    np.random.seed(42)
    n_points = 100
    
    # Generate base price with trend
    base_price = 100
    trend = np.linspace(0, 10, n_points)
    noise = np.random.randn(n_points) * 2
    
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n_points)) * 1.5
    low = close - np.abs(np.random.randn(n_points)) * 1.5
    volume = np.random.randint(1000, 10000, n_points)
    
    # Ensure high >= close >= low
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    
    # Test with different periods
    test_periods = [10, 20, 30]
    
    for period in test_periods:
        print(f"\nTesting CMF with period={period}:")
        print("-" * 40)
        
        # Initialize indicator
        cmf = ChaikinMoneyFlow(period=period)
        
        # Calculate CMF
        cmf_values = cmf.calculate(high, low, close, volume)
        
        # Get signals
        signals = cmf.get_signals(high, low, close, volume)
        
        # Get divergence analysis
        divergence = cmf.analyze_divergence(high, low, close, volume)
        
        # Calculate statistics
        valid_cmf = cmf_values[~np.isnan(cmf_values)]
        
        if len(valid_cmf) > 0:
            cmf_mean = np.mean(valid_cmf)
            cmf_std = np.std(valid_cmf)
            cmf_min = np.min(valid_cmf)
            cmf_max = np.max(valid_cmf)
            
            # Count periods above/below zero
            above_zero = np.sum(valid_cmf > 0)
            below_zero = np.sum(valid_cmf < 0)
            
            print(f"CMF Statistics:")
            print(f"  Current Value: {signals['cmf_value']:.4f}")
            print(f"  Signal: {signals['signal']} ({signals['strength']})")
            print(f"  Pressure: {signals['pressure']}")
            print(f"  Mean: {cmf_mean:.4f}")
            print(f"  Std Dev: {cmf_std:.4f}")
            print(f"  Range: [{cmf_min:.4f}, {cmf_max:.4f}]")
            print(f"  Above Zero: {above_zero}/{len(valid_cmf)} ({above_zero/len(valid_cmf)*100:.1f}%)")
            print(f"  Below Zero: {below_zero}/{len(valid_cmf)} ({below_zero/len(valid_cmf)*100:.1f}%)")
            
            print(f"Divergence Analysis:")
            print(f"  Type: {divergence['divergence_type']}")
            print(f"  Strength: {divergence['strength']:.2f}")
            print(f"  Description: {divergence['description']}")
            
            # Show last few values
            print(f"  Last 5 CMF values: {valid_cmf[-5:]}")
        else:
            print(f"  No valid CMF values calculated")
    
    # Test edge cases
    print(f"\nTesting Edge Cases:")
    print("-" * 30)
    
    # Test with flat prices (high == low == close)
    flat_price = np.full(50, 100.0)
    flat_volume = np.full(50, 1000)
    
    cmf_flat = ChaikinMoneyFlow(period=20)
    cmf_flat_values = cmf_flat.calculate(flat_price, flat_price, flat_price, flat_volume)
    
    valid_flat = cmf_flat_values[~np.isnan(cmf_flat_values)]
    print(f"Flat price test:")
    print(f"  All CMF values zero: {np.allclose(valid_flat, 0.0) if len(valid_flat) > 0 else 'No valid values'}")
    
    # Test with zero volume
    zero_volume = np.zeros(50)
    cmf_zero_vol = ChaikinMoneyFlow(period=20)
    cmf_zero_vol_values = cmf_zero_vol.calculate(high[:50], low[:50], close[:50], zero_volume)
    
    valid_zero_vol = cmf_zero_vol_values[~np.isnan(cmf_zero_vol_values)]
    print(f"Zero volume test:")
    print(f"  All CMF values zero: {np.allclose(valid_zero_vol, 0.0) if len(valid_zero_vol) > 0 else 'No valid values'}")
    
    # Test with insufficient data
    cmf_small = ChaikinMoneyFlow(period=50)
    cmf_small_values = cmf_small.calculate(high[:10], low[:10], close[:10], volume[:10])
    
    print(f"Insufficient data test:")
    print(f"  All NaN: {np.all(np.isnan(cmf_small_values))}")
    
    print(f"\nTest completed successfully!")

if __name__ == "__main__":
    test_chaikin_money_flow()