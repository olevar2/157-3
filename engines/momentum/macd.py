"""
Moving Average Convergence Divergence (MACD)
============================================

MACD is a trend-following momentum indicator that shows the relationship 
between two moving averages of a security's price.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import IndicatorBase


class MovingAverageConvergenceDivergence(IndicatorBase):
    """
    Moving Average Convergence Divergence (MACD) indicator.
    
    MACD consists of:
    - MACD Line: Difference between fast EMA and slow EMA
    - Signal Line: EMA of MACD line
    - Histogram: Difference between MACD line and signal line
    """
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: Period for fast EMA
            slow_period: Period for slow EMA
            signal_period: Period for signal line EMA
        """
        super().__init__()
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Validation
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("All periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: Input data array
            period: EMA period
            
        Returns:
            EMA values array
        """
        alpha = 2.0 / (period + 1)
        ema = np.full_like(data, np.nan)
        
        # Initialize with first valid value
        for i in range(len(data)):
            if not np.isnan(data[i]):
                ema[i] = data[i]
                break
        
        # Calculate EMA
        for i in range(1, len(data)):
            if not np.isnan(data[i]) and not np.isnan(ema[i-1]):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def calculate(self, 
                 data: pd.DataFrame,
                 price_column: str = 'close') -> pd.DataFrame:
        """
        Calculate MACD indicator.
        
        Args:
            data: DataFrame with price data
            price_column: Column name for price data
            
        Returns:
            DataFrame with MACD components and related metrics
        """
        if len(data) < self.slow_period + self.signal_period:
            raise ValueError(f"Insufficient data. Need at least {self.slow_period + self.signal_period} rows")
        
        prices = data[price_column].values
        
        # Calculate fast and slow EMAs
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        signal_line = self._calculate_ema(macd_line, self.signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'price': prices
        })
        
        # Calculate additional metrics
        result_df['macd_momentum'] = result_df['macd'].diff()
        result_df['signal_momentum'] = result_df['signal'].diff()
        result_df['histogram_momentum'] = result_df['histogram'].diff()
        
        # MACD position relative to signal
        result_df['macd_above_signal'] = (result_df['macd'] > result_df['signal']).astype(int)
        result_df['macd_below_signal'] = (result_df['macd'] < result_df['signal']).astype(int)
        
        # Zero line crossings
        result_df['macd_above_zero'] = (result_df['macd'] > 0).astype(int)
        result_df['macd_below_zero'] = (result_df['macd'] < 0).astype(int)
        
        # Distance measurements
        result_df['macd_signal_distance'] = np.abs(result_df['macd'] - result_df['signal'])
        result_df['macd_zero_distance'] = np.abs(result_df['macd'])
        
        # Trend strength
        result_df['trend_strength'] = np.abs(result_df['histogram'])
        
        return result_df
    
    def get_signals(self, 
                   indicator_data: pd.DataFrame,
                   histogram_threshold: float = 0.0) -> pd.DataFrame:
        """
        Generate trading signals based on MACD.
        
        Args:
            indicator_data: DataFrame from calculate() method
            histogram_threshold: Minimum histogram value for signal generation
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=indicator_data.index)
        
        # MACD line crossing signal line
        signals['macd_bullish_crossover'] = (
            (indicator_data['macd'] > indicator_data['signal']) &
            (indicator_data['macd'].shift(1) <= indicator_data['signal'].shift(1))
        ).astype(int)
        
        signals['macd_bearish_crossover'] = (
            (indicator_data['macd'] < indicator_data['signal']) &
            (indicator_data['macd'].shift(1) >= indicator_data['signal'].shift(1))
        ).astype(int)
        
        # MACD line crossing zero line
        signals['macd_bullish_zero_cross'] = (
            (indicator_data['macd'] > 0) &
            (indicator_data['macd'].shift(1) <= 0)
        ).astype(int)
        
        signals['macd_bearish_zero_cross'] = (
            (indicator_data['macd'] < 0) &
            (indicator_data['macd'].shift(1) >= 0)
        ).astype(int)
        
        # Histogram signals
        signals['histogram_bullish'] = (
            (indicator_data['histogram'] > histogram_threshold) &
            (indicator_data['histogram_momentum'] > 0)
        ).astype(int)
        
        signals['histogram_bearish'] = (
            (indicator_data['histogram'] < -histogram_threshold) &
            (indicator_data['histogram_momentum'] < 0)
        ).astype(int)
        
        # Histogram peak/trough signals
        signals['histogram_peak'] = (
            (indicator_data['histogram_momentum'] < 0) &
            (indicator_data['histogram_momentum'].shift(1) >= 0) &
            (indicator_data['histogram'] > 0)
        ).astype(int)
        
        signals['histogram_trough'] = (
            (indicator_data['histogram_momentum'] > 0) &
            (indicator_data['histogram_momentum'].shift(1) <= 0) &
            (indicator_data['histogram'] < 0)
        ).astype(int)
        
        # Momentum signals
        signals['macd_momentum_up'] = (
            indicator_data['macd_momentum'] > 0
        ).astype(int)
        
        signals['macd_momentum_down'] = (
            indicator_data['macd_momentum'] < 0
        ).astype(int)
        
        signals['signal_momentum_up'] = (
            indicator_data['signal_momentum'] > 0
        ).astype(int)
        
        signals['signal_momentum_down'] = (
            indicator_data['signal_momentum'] < 0
        ).astype(int)
        
        # Trend confirmation signals
        signals['strong_bullish_trend'] = (
            (indicator_data['macd'] > indicator_data['signal']) &
            (indicator_data['macd'] > 0) &
            (indicator_data['histogram_momentum'] > 0)
        ).astype(int)
        
        signals['strong_bearish_trend'] = (
            (indicator_data['macd'] < indicator_data['signal']) &
            (indicator_data['macd'] < 0) &
            (indicator_data['histogram_momentum'] < 0)
        ).astype(int)
        
        # Divergence warning signals (simplified)
        signals['weakening_bullish'] = (
            (indicator_data['macd'] > indicator_data['signal']) &
            (indicator_data['histogram_momentum'] < 0) &
            (indicator_data['histogram'] > 0)
        ).astype(int)
        
        signals['weakening_bearish'] = (
            (indicator_data['macd'] < indicator_data['signal']) &
            (indicator_data['histogram_momentum'] > 0) &
            (indicator_data['histogram'] < 0)
        ).astype(int)
        
        return signals
    
    def detect_divergence(self, 
                         price_data: np.ndarray,
                         macd_data: np.ndarray,
                         lookback: int = 10) -> Dict[str, bool]:
        """
        Detect bullish and bearish divergences.
        
        Args:
            price_data: Price data array
            macd_data: MACD line data array
            lookback: Lookback period for divergence detection
            
        Returns:
            Dictionary with divergence detection results
        """
        if len(price_data) < lookback * 2 or len(macd_data) < lookback * 2:
            return {'bullish_divergence': False, 'bearish_divergence': False}
        
        # Find recent highs and lows
        recent_price = price_data[-lookback:]
        recent_macd = macd_data[-lookback:]
        prev_price = price_data[-lookback*2:-lookback]
        prev_macd = macd_data[-lookback*2:-lookback]
        
        # Find peaks and troughs
        recent_price_high = np.max(recent_price)
        recent_macd_high = np.max(recent_macd)
        prev_price_high = np.max(prev_price)
        prev_macd_high = np.max(prev_macd)
        
        recent_price_low = np.min(recent_price)
        recent_macd_low = np.min(recent_macd)
        prev_price_low = np.min(prev_price)
        prev_macd_low = np.min(prev_macd)
        
        # Bearish divergence: higher highs in price, lower highs in MACD
        bearish_divergence = (
            recent_price_high > prev_price_high and
            recent_macd_high < prev_macd_high
        )
        
        # Bullish divergence: lower lows in price, higher lows in MACD
        bullish_divergence = (
            recent_price_low < prev_price_low and
            recent_macd_low > prev_macd_low
        )
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    def get_interpretation(self, latest_values: Dict) -> str:
        """
        Provide interpretation of current MACD state.
        
        Args:
            latest_values: Dictionary with latest indicator values
            
        Returns:
            String interpretation
        """
        macd = latest_values.get('macd', 0)
        signal = latest_values.get('signal', 0)
        histogram = latest_values.get('histogram', 0)
        macd_momentum = latest_values.get('macd_momentum', 0)
        histogram_momentum = latest_values.get('histogram_momentum', 0)
        
        # Primary trend
        if macd > signal:
            primary_trend = "bullish"
        elif macd < signal:
            primary_trend = "bearish"
        else:
            primary_trend = "neutral"
        
        # Zero line position
        if macd > 0:
            zero_position = "above zero (uptrend)"
        elif macd < 0:
            zero_position = "below zero (downtrend)"
        else:
            zero_position = "at zero (neutral)"
        
        # Momentum assessment
        if macd_momentum > 0:
            momentum_desc = "accelerating"
        elif macd_momentum < 0:
            momentum_desc = "decelerating"
        else:
            momentum_desc = "steady"
        
        # Histogram interpretation
        if histogram > 0 and histogram_momentum > 0:
            hist_desc = "strengthening bullish momentum"
        elif histogram > 0 and histogram_momentum < 0:
            hist_desc = "weakening bullish momentum"
        elif histogram < 0 and histogram_momentum < 0:
            hist_desc = "strengthening bearish momentum"
        elif histogram < 0 and histogram_momentum > 0:
            hist_desc = "weakening bearish momentum"
        else:
            hist_desc = "neutral momentum"
        
        return f"MACD shows {primary_trend} bias with line {zero_position}. " \
               f"Momentum is {momentum_desc} with {hist_desc}. " \
               f"MACD: {macd:.4f}, Signal: {signal:.4f}, Histogram: {histogram:.4f}."


def create_macd_indicator(fast_period: int = 12,
                         slow_period: int = 26,
                         signal_period: int = 9) -> MovingAverageConvergenceDivergence:
    """
    Factory function to create MACD indicator.
    
    Args:
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line EMA
        
    Returns:
        Configured MovingAverageConvergenceDivergence instance
    """
    return MovingAverageConvergenceDivergence(
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period
    )
