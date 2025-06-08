# -*- coding: utf-8 -*-
"""
Standard Deviation Channels Indicator

Standard Deviation Channels use price channels based on standard deviation
to identify potential support and resistance levels.

Key Features:
- Upper and lower channel calculation
- Dynamic volatility-based channels
- Trend channel analysis
- Breakout detection

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SDChannelSignal(Enum):
    """Standard Deviation Channel signal types"""
    UPPER_BREAKOUT = "upper_breakout"
    LOWER_BREAKOUT = "lower_breakout"
    UPPER_TOUCH = "upper_touch"
    LOWER_TOUCH = "lower_touch"
    MIDDLE_CROSS_UP = "middle_cross_up"
    MIDDLE_CROSS_DOWN = "middle_cross_down"
    RANGE_BOUND = "range_bound"
    NEUTRAL = "neutral"

@dataclass
class SDChannelResult:
    """Standard Deviation Channel analysis result"""
    upper_channel: float
    middle_line: float
    lower_channel: float
    signal: SDChannelSignal
    strength: float
    channel_width: float
    price_position: float  # Percentage position within channel

class StandardDeviationChannels:
    """
    Standard Deviation Channels Indicator
    
    Creates price channels based on linear regression and standard deviation
    to identify potential support and resistance levels.
    """
    
    def __init__(self, period: int = 20, std_dev_factor: float = 2.0):
        """
        Initialize Standard Deviation Channels
        
        Args:
            period: Period for calculation
            std_dev_factor: Standard deviation multiplier
        """
        self.period = period
        self.std_dev_factor = std_dev_factor
        
        logger.info(f"✅ Standard Deviation Channels initialized (period={period}, std_dev={std_dev_factor})")

    def calculate_channels(self, prices: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Standard Deviation Channels
        
        Args:
            prices: Price data
            
        Returns:
            Tuple of (upper_channel, middle_line, lower_channel)
        """
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < self.period:
                return (np.full(len(prices), np.nan), 
                       np.full(len(prices), np.nan), 
                       np.full(len(prices), np.nan))
            
            upper_channel = np.full(len(prices), np.nan)
            middle_line = np.full(len(prices), np.nan)
            lower_channel = np.full(len(prices), np.nan)
            
            for i in range(self.period - 1, len(prices)):
                # Get price window
                price_window = prices[i - self.period + 1:i + 1]
                x = np.arange(len(price_window))
                
                # Linear regression for middle line
                coeffs = np.polyfit(x, price_window, 1)
                middle_value = coeffs[0] * (len(price_window) - 1) + coeffs[1]
                
                # Calculate standard deviation of residuals
                regression_line = coeffs[0] * x + coeffs[1]
                residuals = price_window - regression_line
                std_dev = np.std(residuals)
                
                # Calculate channels
                upper_channel[i] = middle_value + (self.std_dev_factor * std_dev)
                middle_line[i] = middle_value
                lower_channel[i] = middle_value - (self.std_dev_factor * std_dev)
            
            return upper_channel, middle_line, lower_channel
            
        except Exception as e:
            logger.error(f"Error calculating Standard Deviation Channels: {e}")
            return (np.full(len(prices), np.nan), 
                   np.full(len(prices), np.nan), 
                   np.full(len(prices), np.nan))

    def analyze(self, prices: Union[np.ndarray, pd.Series]) -> SDChannelResult:
        """
        Analyze Standard Deviation Channels
        
        Args:
            prices: Price data
            
        Returns:
            SDChannelResult with analysis
        """
        try:
            upper_channel, middle_line, lower_channel = self.calculate_channels(prices)
            
            if np.isnan(upper_channel[-1]):
                return SDChannelResult(np.nan, np.nan, np.nan, SDChannelSignal.NEUTRAL, 
                                     0.0, np.nan, np.nan)
            
            current_price = prices[-1]
            upper = upper_channel[-1]
            middle = middle_line[-1]
            lower = lower_channel[-1]
            
            # Calculate channel metrics
            channel_width = (upper - lower) / middle * 100
            price_position = (current_price - lower) / (upper - lower) * 100
            
            # Determine signal
            signal = self._determine_signal(prices, upper_channel, middle_line, lower_channel)
            
            # Calculate signal strength
            strength = self._calculate_strength(current_price, upper, middle, lower, price_position)
            
            return SDChannelResult(
                upper_channel=upper,
                middle_line=middle,
                lower_channel=lower,
                signal=signal,
                strength=strength,
                channel_width=channel_width,
                price_position=price_position
            )
            
        except Exception as e:
            logger.error(f"Error in Standard Deviation Channel analysis: {e}")
            return SDChannelResult(np.nan, np.nan, np.nan, SDChannelSignal.NEUTRAL, 
                                 0.0, np.nan, np.nan)

    def _determine_signal(self, prices: np.ndarray, upper: np.ndarray, 
                         middle: np.ndarray, lower: np.ndarray) -> SDChannelSignal:
        """Determine signal based on price position"""
        try:
            if len(prices) < 2:
                return SDChannelSignal.NEUTRAL
            
            current_price = prices[-1]
            prev_price = prices[-2]
            current_upper = upper[-1]
            current_middle = middle[-1]
            current_lower = lower[-1]
            
            # Check for breakouts
            if current_price > current_upper and prev_price <= upper[-2]:
                return SDChannelSignal.UPPER_BREAKOUT
            elif current_price < current_lower and prev_price >= lower[-2]:
                return SDChannelSignal.LOWER_BREAKOUT
            
            # Check for touches
            upper_distance = abs(current_price - current_upper) / current_upper
            lower_distance = abs(current_price - current_lower) / current_lower
            
            if upper_distance < 0.005:  # Very close to upper channel
                return SDChannelSignal.UPPER_TOUCH
            elif lower_distance < 0.005:  # Very close to lower channel
                return SDChannelSignal.LOWER_TOUCH
            
            # Check for middle line crosses
            if len(prices) >= 2 and len(middle) >= 2:
                if prev_price <= middle[-2] and current_price > current_middle:
                    return SDChannelSignal.MIDDLE_CROSS_UP
                elif prev_price >= middle[-2] and current_price < current_middle:
                    return SDChannelSignal.MIDDLE_CROSS_DOWN
            
            # Range bound if within channels
            if current_lower < current_price < current_upper:
                return SDChannelSignal.RANGE_BOUND
            
            return SDChannelSignal.NEUTRAL
            
        except Exception:
            return SDChannelSignal.NEUTRAL

    def _calculate_strength(self, price: float, upper: float, middle: float, 
                          lower: float, price_position: float) -> float:
        """Calculate signal strength"""
        try:
            if np.isnan(price_position):
                return 0.0
            
            # Strength based on distance from extremes
            if price_position > 95:  # Near upper channel
                strength = (price_position - 95) / 5
            elif price_position < 5:  # Near lower channel
                strength = (5 - price_position) / 5
            elif 45 < price_position < 55:  # Near middle
                strength = 1.0 - abs(price_position - 50) / 5
            else:
                strength = 0.5
            
            return max(0.0, min(1.0, strength))
            
        except Exception:
            return 0.0

    def get_support_resistance_levels(self, prices: Union[np.ndarray, pd.Series]) -> Dict:
        """Get support and resistance levels from channels"""
        try:
            upper_channel, middle_line, lower_channel = self.calculate_channels(prices)
            
            # Get recent levels (last 10 periods)
            recent_upper = upper_channel[-10:][~np.isnan(upper_channel[-10:])]
            recent_middle = middle_line[-10:][~np.isnan(middle_line[-10:])]
            recent_lower = lower_channel[-10:][~np.isnan(lower_channel[-10:])]
            
            return {
                'resistance_levels': recent_upper.tolist() if len(recent_upper) > 0 else [],
                'support_levels': recent_lower.tolist() if len(recent_lower) > 0 else [],
                'middle_levels': recent_middle.tolist() if len(recent_middle) > 0 else [],
                'current_upper': upper_channel[-1] if not np.isnan(upper_channel[-1]) else None,
                'current_middle': middle_line[-1] if not np.isnan(middle_line[-1]) else None,
                'current_lower': lower_channel[-1] if not np.isnan(lower_channel[-1]) else None
            }
            
        except Exception as e:
            logger.error(f"Error getting support/resistance levels: {e}")
            return {'resistance_levels': [], 'support_levels': [], 'middle_levels': []}
