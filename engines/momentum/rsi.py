"""
Relative Strength Index (RSI)
============================

The RSI is a momentum oscillator that measures the speed and magnitude of 
recent price changes to evaluate overbought or oversold conditions.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import IndicatorBase


class RelativeStrengthIndex(IndicatorBase):
    """
    Relative Strength Index (RSI) indicator.
    
    RSI oscillates between 0 and 100, with values above 70 typically indicating
    overbought conditions and values below 30 indicating oversold conditions.
    """
    
    def __init__(self, 
                 period: int = 14,
                 overbought_level: float = 70.0,
                 oversold_level: float = 30.0):
        """
        Initialize RSI indicator.
        
        Args:
            period: Period for RSI calculation
            overbought_level: RSI level indicating overbought condition
            oversold_level: RSI level indicating oversold condition
        """
        super().__init__()
        
        self.period = period
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        
        # Validation
        if period <= 0:
            raise ValueError("period must be positive")
        if not 0 <= oversold_level < overbought_level <= 100:
            raise ValueError("Invalid overbought/oversold levels")
    
    def _wilder_smoothing(self, values: np.ndarray, period: int) -> np.ndarray:
        """
        Apply Wilder's smoothing method (modified EMA).
        
        Args:
            values: Array of values to smooth
            period: Smoothing period
            
        Returns:
            Smoothed values array
        """
        alpha = 1.0 / period
        smoothed = np.full_like(values, np.nan)
        
        # First value is simple average
        first_valid = period - 1
        if len(values) > first_valid:
            smoothed[first_valid] = np.mean(values[:period])
            
            # Apply Wilder's smoothing for subsequent values
            for i in range(first_valid + 1, len(values)):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        
        return smoothed
    
    def calculate(self, 
                 data: pd.DataFrame,
                 price_column: str = 'close') -> pd.DataFrame:
        """
        Calculate RSI indicator.
        
        Args:
            data: DataFrame with price data
            price_column: Column name for price data
            
        Returns:
            DataFrame with RSI values and related metrics
        """
        if len(data) < self.period + 1:
            raise ValueError(f"Insufficient data. Need at least {self.period + 1} rows")
        
        prices = data[price_column].values
        
        # Calculate price changes
        price_changes = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        # Apply Wilder's smoothing
        avg_gains = self._wilder_smoothing(gains, self.period)
        avg_losses = self._wilder_smoothing(losses, self.period)
        
        # Calculate RS and RSI
        rs = np.where(avg_losses != 0, avg_gains / avg_losses, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = np.where(avg_losses == 0, 100, rsi)
        rsi = np.where(avg_gains == 0, 0, rsi)
        
        # Pad with NaN for the first value (no price change for first bar)
        rsi = np.concatenate([np.array([np.nan]), rsi])
        avg_gains = np.concatenate([np.array([np.nan]), avg_gains])
        avg_losses = np.concatenate([np.array([np.nan]), avg_losses])
        rs = np.concatenate([np.array([np.nan]), rs])
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'rsi': rsi,
            'avg_gains': avg_gains,
            'avg_losses': avg_losses,
            'rs': rs,
            'price': prices
        })
        
        # Calculate additional metrics
        result_df['rsi_momentum'] = result_df['rsi'].diff()
        result_df['rsi_acceleration'] = result_df['rsi_momentum'].diff()
        
        # RSI levels
        result_df['is_overbought'] = (result_df['rsi'] > self.overbought_level).astype(int)
        result_df['is_oversold'] = (result_df['rsi'] < self.oversold_level).astype(int)
        result_df['is_neutral'] = (
            (result_df['rsi'] >= self.oversold_level) & 
            (result_df['rsi'] <= self.overbought_level)
        ).astype(int)
        
        # Distance from extreme levels
        result_df['distance_to_overbought'] = self.overbought_level - result_df['rsi']
        result_df['distance_to_oversold'] = result_df['rsi'] - self.oversold_level
        
        return result_df
    
    def get_signals(self, 
                   indicator_data: pd.DataFrame,
                   divergence_lookback: int = 5) -> pd.DataFrame:
        """
        Generate trading signals based on RSI.
        
        Args:
            indicator_data: DataFrame from calculate() method
            divergence_lookback: Lookback period for divergence detection
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=indicator_data.index)
        
        # Basic overbought/oversold signals
        signals['rsi_oversold_signal'] = (
            (indicator_data['rsi'] < self.oversold_level) &
            (indicator_data['rsi'].shift(1) >= self.oversold_level)
        ).astype(int)
        
        signals['rsi_overbought_signal'] = (
            (indicator_data['rsi'] > self.overbought_level) &
            (indicator_data['rsi'].shift(1) <= self.overbought_level)
        ).astype(int)
        
        # RSI crossing 50 (midline)
        signals['rsi_bullish_crossover'] = (
            (indicator_data['rsi'] > 50) &
            (indicator_data['rsi'].shift(1) <= 50)
        ).astype(int)
        
        signals['rsi_bearish_crossover'] = (
            (indicator_data['rsi'] < 50) &
            (indicator_data['rsi'].shift(1) >= 50)
        ).astype(int)
        
        # RSI momentum signals
        signals['rsi_momentum_up'] = (
            indicator_data['rsi_momentum'] > 0
        ).astype(int)
        
        signals['rsi_momentum_down'] = (
            indicator_data['rsi_momentum'] < 0
        ).astype(int)
        
        # Extreme RSI signals
        signals['rsi_extreme_oversold'] = (
            indicator_data['rsi'] < 20
        ).astype(int)
        
        signals['rsi_extreme_overbought'] = (
            indicator_data['rsi'] > 80
        ).astype(int)
        
        # RSI acceleration signals
        signals['rsi_accelerating_up'] = (
            (indicator_data['rsi_acceleration'] > 0) &
            (indicator_data['rsi_momentum'] > 0)
        ).astype(int)
        
        signals['rsi_accelerating_down'] = (
            (indicator_data['rsi_acceleration'] < 0) &
            (indicator_data['rsi_momentum'] < 0)
        ).astype(int)
        
        # Divergence detection (simplified)
        if divergence_lookback > 0 and len(indicator_data) > divergence_lookback * 2:
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_min_idx = indicator_data['price'].rolling(divergence_lookback).apply(
                lambda x: x.argmin(), raw=False
            )
            rsi_min_idx = indicator_data['rsi'].rolling(divergence_lookback).apply(
                lambda x: x.argmin(), raw=False
            )
            
            signals['bullish_divergence'] = (
                (price_min_idx == divergence_lookback - 1) &
                (rsi_min_idx < divergence_lookback - 1) &
                (indicator_data['rsi'] < 50)
            ).astype(int)
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            price_max_idx = indicator_data['price'].rolling(divergence_lookback).apply(
                lambda x: x.argmax(), raw=False
            )
            rsi_max_idx = indicator_data['rsi'].rolling(divergence_lookback).apply(
                lambda x: x.argmax(), raw=False
            )
            
            signals['bearish_divergence'] = (
                (price_max_idx == divergence_lookback - 1) &
                (rsi_max_idx < divergence_lookback - 1) &
                (indicator_data['rsi'] > 50)
            ).astype(int)
        else:
            signals['bullish_divergence'] = 0
            signals['bearish_divergence'] = 0
        
        return signals
    
    def get_interpretation(self, latest_values: Dict) -> str:
        """
        Provide interpretation of current RSI state.
        
        Args:
            latest_values: Dictionary with latest indicator values
            
        Returns:
            String interpretation
        """
        rsi = latest_values.get('rsi', 50)
        momentum = latest_values.get('rsi_momentum', 0)
        acceleration = latest_values.get('rsi_acceleration', 0)
        
        # RSI level interpretation
        if rsi > 80:
            level_desc = "extremely overbought"
        elif rsi > self.overbought_level:
            level_desc = "overbought"
        elif rsi > 60:
            level_desc = "moderately bullish"
        elif rsi > 50:
            level_desc = "slightly bullish"
        elif rsi > 40:
            level_desc = "slightly bearish"
        elif rsi > self.oversold_level:
            level_desc = "moderately bearish"
        elif rsi > 20:
            level_desc = "oversold"
        else:
            level_desc = "extremely oversold"
        
        # Momentum interpretation
        if momentum > 2:
            momentum_desc = "strong upward momentum"
        elif momentum > 0:
            momentum_desc = "upward momentum"
        elif momentum > -2:
            momentum_desc = "downward momentum"
        else:
            momentum_desc = "strong downward momentum"
        
        # Acceleration interpretation
        if acceleration > 0:
            accel_desc = "accelerating"
        elif acceleration < 0:
            accel_desc = "decelerating"
        else:
            accel_desc = "steady"
        
        return f"RSI at {rsi:.1f} indicates {level_desc} conditions. " \
               f"Shows {momentum_desc} that is {accel_desc}."


def create_rsi_indicator(period: int = 14,
                        overbought_level: float = 70.0,
                        oversold_level: float = 30.0) -> RelativeStrengthIndex:
    """
    Factory function to create RSI indicator.
    
    Args:
        period: Period for RSI calculation
        overbought_level: RSI level indicating overbought condition
        oversold_level: RSI level indicating oversold condition
        
    Returns:
        Configured RelativeStrengthIndex instance
    """
    return RelativeStrengthIndex(
        period=period,
        overbought_level=overbought_level,
        oversold_level=oversold_level
    )
