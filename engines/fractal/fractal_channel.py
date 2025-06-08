"""
Fractal Channel Indicator
=========================

Dynamic support/resistance channel using fractal analysis.
Creates adaptive channels based on fractal highs and lows.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from ..indicator_base import IndicatorBase


class FractalChannelIndicator(IndicatorBase):
    """
    Fractal Channel Indicator for dynamic support/resistance levels.
    
    This indicator identifies fractal patterns in highs and lows to create
    adaptive channel boundaries that evolve with market structure.
    """
    
    def __init__(self, 
                 period: int = 5,
                 channel_lookback: int = 20,
                 min_fractal_strength: float = 0.5,
                 channel_deviation: float = 0.02):
        """
        Initialize Fractal Channel Indicator.
        
        Args:
            period: Period for fractal identification
            channel_lookback: Lookback period for channel calculation
            min_fractal_strength: Minimum strength for valid fractals
            channel_deviation: Maximum allowed channel deviation
        """
        super().__init__()
        
        self.period = period
        self.channel_lookback = channel_lookback
        self.min_fractal_strength = min_fractal_strength
        self.channel_deviation = channel_deviation
        
        # Validation
        if period <= 0:
            raise ValueError("period must be positive")
        if channel_lookback <= period:
            raise ValueError("channel_lookback must be greater than period")
        if not 0 < min_fractal_strength <= 1:
            raise ValueError("min_fractal_strength must be between 0 and 1")
    
    def _identify_fractal_highs(self, 
                               high: np.ndarray, 
                               low: np.ndarray) -> List[Tuple[int, float, float]]:
        """
        Identify fractal high points.
        
        Args:
            high: High prices array
            low: Low prices array
            
        Returns:
            List of tuples (index, high_value, strength)
        """
        fractal_highs = []
        n = len(high)
        
        for i in range(self.period, n - self.period):
            # Check if current high is a fractal high
            is_fractal = True
            current_high = high[i]
            
            # Check left side
            for j in range(i - self.period, i):
                if high[j] >= current_high:
                    is_fractal = False
                    break
            
            if not is_fractal:
                continue
                
            # Check right side
            for j in range(i + 1, i + self.period + 1):
                if high[j] >= current_high:
                    is_fractal = False
                    break
            
            if is_fractal:
                # Calculate fractal strength
                left_min = np.min(low[i - self.period:i])
                right_min = np.min(low[i + 1:i + self.period + 1])
                range_strength = (current_high - max(left_min, right_min)) / current_high
                
                if range_strength >= self.min_fractal_strength:
                    fractal_highs.append((i, current_high, range_strength))
        
        return fractal_highs
    
    def _identify_fractal_lows(self, 
                              high: np.ndarray, 
                              low: np.ndarray) -> List[Tuple[int, float, float]]:
        """
        Identify fractal low points.
        
        Args:
            high: High prices array
            low: Low prices array
            
        Returns:
            List of tuples (index, low_value, strength)
        """
        fractal_lows = []
        n = len(low)
        
        for i in range(self.period, n - self.period):
            # Check if current low is a fractal low
            is_fractal = True
            current_low = low[i]
            
            # Check left side
            for j in range(i - self.period, i):
                if low[j] <= current_low:
                    is_fractal = False
                    break
            
            if not is_fractal:
                continue
                
            # Check right side
            for j in range(i + 1, i + self.period + 1):
                if low[j] <= current_low:
                    is_fractal = False
                    break
            
            if is_fractal:
                # Calculate fractal strength
                left_max = np.max(high[i - self.period:i])
                right_max = np.max(high[i + 1:i + self.period + 1])
                range_strength = (min(left_max, right_max) - current_low) / current_low
                
                if range_strength >= self.min_fractal_strength:
                    fractal_lows.append((i, current_low, range_strength))
        
        return fractal_lows
    
    def _calculate_channel_levels(self, 
                                 fractal_highs: List[Tuple[int, float, float]],
                                 fractal_lows: List[Tuple[int, float, float]],
                                 current_index: int) -> Tuple[float, float, float]:
        """
        Calculate channel support and resistance levels.
        
        Args:
            fractal_highs: List of fractal high points
            fractal_lows: List of fractal low points
            current_index: Current bar index
            
        Returns:
            Tuple of (resistance, support, mid_line)
        """
        # Filter fractals within lookback period
        recent_highs = [
            (idx, val, strength) for idx, val, strength in fractal_highs
            if current_index - self.channel_lookback <= idx <= current_index
        ]
        
        recent_lows = [
            (idx, val, strength) for idx, val, strength in fractal_lows
            if current_index - self.channel_lookback <= idx <= current_index
        ]
        
        # Calculate weighted averages based on recency and strength
        if recent_highs:
            weights_high = []
            values_high = []
            
            for idx, val, strength in recent_highs:
                # Weight by recency and strength
                recency_weight = (idx - (current_index - self.channel_lookback)) / self.channel_lookback
                total_weight = recency_weight * strength
                weights_high.append(total_weight)
                values_high.append(val)
            
            resistance = np.average(values_high, weights=weights_high)
        else:
            resistance = np.nan
        
        if recent_lows:
            weights_low = []
            values_low = []
            
            for idx, val, strength in recent_lows:
                # Weight by recency and strength
                recency_weight = (idx - (current_index - self.channel_lookback)) / self.channel_lookback
                total_weight = recency_weight * strength
                weights_low.append(total_weight)
                values_low.append(val)
            
            support = np.average(values_low, weights=weights_low)
        else:
            support = np.nan
        
        # Calculate middle line
        if not np.isnan(resistance) and not np.isnan(support):
            mid_line = (resistance + support) / 2
        else:
            mid_line = np.nan
        
        return resistance, support, mid_line
    
    def _calculate_channel_width(self, resistance: float, support: float) -> float:
        """
        Calculate normalized channel width.
        
        Args:
            resistance: Resistance level
            support: Support level
            
        Returns:
            Normalized channel width
        """
        if np.isnan(resistance) or np.isnan(support) or support == 0:
            return np.nan
        
        return (resistance - support) / support
    
    def calculate(self, 
                 data: pd.DataFrame,
                 high_column: str = 'high',
                 low_column: str = 'low',
                 close_column: str = 'close') -> pd.DataFrame:
        """
        Calculate Fractal Channel Indicator.
        
        Args:
            data: DataFrame with OHLC data
            high_column: Column name for high prices
            low_column: Column name for low prices
            close_column: Column name for close prices
            
        Returns:
            DataFrame with channel levels and fractal points
        """
        if len(data) < self.channel_lookback + self.period:
            raise ValueError(f"Insufficient data. Need at least {self.channel_lookback + self.period} rows")
        
        high = data[high_column].values
        low = data[low_column].values
        close = data[close_column].values
        
        # Identify all fractal points
        fractal_highs = self._identify_fractal_highs(high, low)
        fractal_lows = self._identify_fractal_lows(high, low)
        
        # Initialize result arrays
        n = len(data)
        resistance_levels = np.full(n, np.nan)
        support_levels = np.full(n, np.nan)
        mid_lines = np.full(n, np.nan)
        channel_widths = np.full(n, np.nan)
        is_fractal_high = np.zeros(n, dtype=bool)
        is_fractal_low = np.zeros(n, dtype=bool)
        fractal_strengths = np.full(n, np.nan)
        
        # Mark fractal points
        for idx, val, strength in fractal_highs:
            is_fractal_high[idx] = True
            fractal_strengths[idx] = strength
            
        for idx, val, strength in fractal_lows:
            is_fractal_low[idx] = True
            if np.isnan(fractal_strengths[idx]):
                fractal_strengths[idx] = strength
            else:
                fractal_strengths[idx] = max(fractal_strengths[idx], strength)
        
        # Calculate channel levels for each bar
        for i in range(self.channel_lookback + self.period, n):
            resistance, support, mid_line = self._calculate_channel_levels(
                fractal_highs, fractal_lows, i
            )
            
            resistance_levels[i] = resistance
            support_levels[i] = support
            mid_lines[i] = mid_line
            channel_widths[i] = self._calculate_channel_width(resistance, support)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'resistance': resistance_levels,
            'support': support_levels,
            'mid_line': mid_lines,
            'channel_width': channel_widths,
            'is_fractal_high': is_fractal_high,
            'is_fractal_low': is_fractal_low,
            'fractal_strength': fractal_strengths,
            'close': close
        })
        
        # Calculate additional metrics
        result_df['price_position'] = np.where(
            ~np.isnan(result_df['mid_line']),
            (result_df['close'] - result_df['mid_line']) / result_df['mid_line'],
            np.nan
        )
        
        result_df['resistance_distance'] = np.where(
            ~np.isnan(result_df['resistance']),
            (result_df['resistance'] - result_df['close']) / result_df['close'],
            np.nan
        )
        
        result_df['support_distance'] = np.where(
            ~np.isnan(result_df['support']),
            (result_df['close'] - result_df['support']) / result_df['support'],
            np.nan
        )
        
        return result_df
    
    def get_signals(self, 
                   indicator_data: pd.DataFrame,
                   breakout_threshold: float = 0.01) -> pd.DataFrame:
        """
        Generate trading signals based on fractal channel.
        
        Args:
            indicator_data: DataFrame from calculate() method
            breakout_threshold: Threshold for breakout detection
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=indicator_data.index)
        
        # Resistance breakout
        signals['resistance_breakout'] = (
            (indicator_data['close'] > indicator_data['resistance']) &
            (indicator_data['close'].shift(1) <= indicator_data['resistance'].shift(1)) &
            (indicator_data['resistance_distance'] < -breakout_threshold)
        ).astype(int)
        
        # Support breakdown
        signals['support_breakdown'] = (
            (indicator_data['close'] < indicator_data['support']) &
            (indicator_data['close'].shift(1) >= indicator_data['support'].shift(1)) &
            (indicator_data['support_distance'] < -breakout_threshold)
        ).astype(int)
        
        # Channel position signals
        signals['upper_channel'] = (
            indicator_data['price_position'] > 0.7
        ).astype(int)
        
        signals['lower_channel'] = (
            indicator_data['price_position'] < -0.7
        ).astype(int)
        
        # Fractal signals
        signals['fractal_high_signal'] = indicator_data['is_fractal_high'].astype(int)
        signals['fractal_low_signal'] = indicator_data['is_fractal_low'].astype(int)
        
        # Channel width signals
        signals['narrow_channel'] = (
            indicator_data['channel_width'] < 0.05
        ).astype(int)
        
        signals['wide_channel'] = (
            indicator_data['channel_width'] > 0.15
        ).astype(int)
        
        return signals
    
    def get_interpretation(self, latest_values: Dict) -> str:
        """
        Provide interpretation of current fractal channel state.
        
        Args:
            latest_values: Dictionary with latest indicator values
            
        Returns:
            String interpretation
        """
        resistance = latest_values.get('resistance', np.nan)
        support = latest_values.get('support', np.nan)
        close = latest_values.get('close', 0)
        price_position = latest_values.get('price_position', 0)
        channel_width = latest_values.get('channel_width', np.nan)
        is_fractal_high = latest_values.get('is_fractal_high', False)
        is_fractal_low = latest_values.get('is_fractal_low', False)
        
        # Channel state
        if np.isnan(resistance) or np.isnan(support):
            return "Insufficient fractal data for channel analysis."
        
        # Channel width interpretation
        if channel_width < 0.03:
            width_desc = "very narrow"
        elif channel_width < 0.08:
            width_desc = "narrow"
        elif channel_width < 0.15:
            width_desc = "normal"
        else:
            width_desc = "wide"
        
        # Price position
        if price_position > 0.8:
            position_desc = "near resistance"
        elif price_position > 0.3:
            position_desc = "upper channel"
        elif price_position > -0.3:
            position_desc = "mid-channel"
        elif price_position > -0.8:
            position_desc = "lower channel"
        else:
            position_desc = "near support"
        
        # Fractal events
        fractal_event = ""
        if is_fractal_high:
            fractal_event = " Fractal high formed."
        elif is_fractal_low:
            fractal_event = " Fractal low formed."
        
        return f"Channel is {width_desc} (width: {channel_width:.1%}). " \
               f"Price at {close:.4f} is in {position_desc} zone. " \
               f"Resistance: {resistance:.4f}, Support: {support:.4f}.{fractal_event}"


def create_fractal_channel_indicator(period: int = 5,
                                    channel_lookback: int = 20,
                                    min_fractal_strength: float = 0.5,
                                    channel_deviation: float = 0.02) -> FractalChannelIndicator:
    """
    Factory function to create Fractal Channel Indicator.
    
    Args:
        period: Period for fractal identification
        channel_lookback: Lookback period for channel calculation
        min_fractal_strength: Minimum strength for valid fractals
        channel_deviation: Maximum allowed channel deviation
        
    Returns:
        Configured FractalChannelIndicator instance
    """
    return FractalChannelIndicator(
        period=period,
        channel_lookback=channel_lookback,
        min_fractal_strength=min_fractal_strength,
        channel_deviation=channel_deviation
    )
