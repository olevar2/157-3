"""
Fractal Channel Indicator

Williams Fractal-based dynamic channel analysis indicator for identifying 
support and resistance levels using fractal geometry principles.

Formula:
- Detects Williams Fractals (price reversals with N-period confirmation)
- Constructs weighted channels from fractal high/low points
- Provides breakout detection and channel strength analysis

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

# Import the base indicator interface
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
)


@dataclass
class FractalPoint:
    """Represents a fractal point in price data"""
    index: int
    price: float
    fractal_type: str  # 'high' or 'low'
    strength: float = 1.0


@dataclass
class FractalChannelResult:
    """Result structure for Fractal Channel analysis"""
    upper_channel: float
    lower_channel: float
    middle_channel: float
    channel_width: float
    fractal_high: Optional[float] = None
    fractal_low: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    breakout_signal: Optional[str] = None  # 'bullish', 'bearish', or None


class FractalChannelIndicator(StandardIndicatorInterface):
    """
    Fractal Channel Indicator

    Williams Fractal-based dynamic channel analysis for identifying key support 
    and resistance levels, with breakout detection and channel strength analysis.

    Key Features:
    - Williams Fractal detection with configurable periods
    - Weighted dynamic channel construction  
    - Support/Resistance level identification
    - Breakout signal detection with threshold control
    - Channel strength and width analysis

    Mathematical Approach:
    Uses chaos theory principles to identify fractal reversal points in price data,
    then constructs statistical channels based on weighted averages of recent 
    fractal levels, providing robust support/resistance identification.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "fractal"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        fractal_period: int = 2,
        channel_lookback: int = 20,
        min_fractals: int = 2,
        breakout_threshold: float = 0.02,
        **kwargs,
    ):
        """
        Initialize Fractal Channel Indicator

        Args:
            period: Main analysis period (default: 20)
            fractal_period: Williams fractal detection period (default: 2)
            channel_lookback: Number of fractals to include in channel (default: 20)
            min_fractals: Minimum fractals required for channel (default: 2)
            breakout_threshold: Breakout detection threshold as ratio (default: 0.02)
            **kwargs: Additional parameters
        """
        # Validate critical parameters before calling super()
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        if fractal_period <= 0:
            raise ValueError(f"fractal_period must be positive, got {fractal_period}")
        
        # REQUIRED: Call parent constructor with all parameters
        super().__init__(
            period=period,
            fractal_period=fractal_period,
            channel_lookback=channel_lookback,
            min_fractals=min_fractals,
            breakout_threshold=breakout_threshold,
            **kwargs,
        )

        # Internal state
        self.fractal_highs: List[FractalPoint] = []
        self.fractal_lows: List[FractalPoint] = []

    def detect_williams_fractals(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> tuple[List[FractalPoint], List[FractalPoint]]:
        """Detect Williams Fractals in price data."""
        fractal_highs = []
        fractal_lows = []
        period = self.parameters.get("fractal_period", 2)

        if len(highs) < 2 * period + 1:
            return fractal_highs, fractal_lows

        for i in range(period, len(highs) - period):
            # Check for fractal high
            current_high = highs[i]
            is_fractal_high = True

            for j in range(1, period + 1):
                if highs[i - j] >= current_high or highs[i + j] >= current_high:
                    is_fractal_high = False
                    break

            if is_fractal_high:
                nearby_highs = np.concatenate(
                    [highs[i - period : i], highs[i + 1 : i + period + 1]]
                )
                strength = (current_high - np.max(nearby_highs)) / current_high
                fractal_highs.append(FractalPoint(i, current_high, "high", strength))

            # Check for fractal low
            current_low = lows[i]
            is_fractal_low = True

            for j in range(1, period + 1):
                if lows[i - j] <= current_low or lows[i + j] <= current_low:
                    is_fractal_low = False
                    break

            if is_fractal_low:
                nearby_lows = np.concatenate(
                    [lows[i - period : i], lows[i + 1 : i + period + 1]]
                )
                strength = (np.min(nearby_lows) - current_low) / current_low
                fractal_lows.append(FractalPoint(i, current_low, "low", strength))

        return fractal_highs, fractal_lows

    def construct_channel(
        self, fractal_highs: List[FractalPoint], fractal_lows: List[FractalPoint]
    ) -> Optional[FractalChannelResult]:
        """Construct fractal channel from detected fractal points."""
        min_fractals = self.parameters.get("min_fractals", 2)
        channel_lookback = self.parameters.get("channel_lookback", 20)
        
        if len(fractal_highs) < min_fractals or len(fractal_lows) < min_fractals:
            return None

        recent_highs = (
            fractal_highs[-channel_lookback:]
            if len(fractal_highs) >= channel_lookback
            else fractal_highs
        )
        recent_lows = (
            fractal_lows[-channel_lookback:]
            if len(fractal_lows) >= channel_lookback
            else fractal_lows
        )

        if recent_highs:
            high_prices = [fp.price for fp in recent_highs]
            high_weights = [fp.strength + 1.0 for fp in recent_highs]
            upper_channel = np.average(high_prices, weights=high_weights)
            resistance_level = max(high_prices)
        else:
            upper_channel = None
            resistance_level = None

        if recent_lows:
            low_prices = [fp.price for fp in recent_lows]
            low_weights = [fp.strength + 1.0 for fp in recent_lows]
            lower_channel = np.average(low_prices, weights=low_weights)
            support_level = min(low_prices)
        else:
            lower_channel = None
            support_level = None

        if upper_channel is None or lower_channel is None:
            return None

        middle_channel = (upper_channel + lower_channel) / 2
        channel_width = upper_channel - lower_channel

        return FractalChannelResult(
            upper_channel=upper_channel,
            lower_channel=lower_channel,
            middle_channel=middle_channel,
            channel_width=channel_width,
            fractal_high=resistance_level,
            fractal_low=support_level,
            support_level=support_level,
            resistance_level=resistance_level,
            breakout_signal=None,
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fractal Channel analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Fractal channel analysis results
        """
        try:
            # Validate input data first
            self.validate_input_data(data)
            
            # Extract OHLC data
            highs = data["high"].values
            lows = data["low"].values
            closes = data["close"].values
            
            period = self.parameters.get("period", 20)
            breakout_threshold = self.parameters.get("breakout_threshold", 0.02)

            if len(highs) < period:
                raise IndicatorValidationError(f"Insufficient data: need {period}, got {len(highs)}")

            # Detect fractals
            fractal_highs, fractal_lows = self.detect_williams_fractals(highs, lows)
            self.fractal_highs.extend(fractal_highs)
            self.fractal_lows.extend(fractal_lows)

            # Construct channel
            channel_result = self.construct_channel(self.fractal_highs, self.fractal_lows)

            # Initialize result arrays
            result_length = len(data)
            upper_channel = np.full(result_length, np.nan)
            lower_channel = np.full(result_length, np.nan)
            middle_channel = np.full(result_length, np.nan)
            channel_width = np.full(result_length, np.nan)
            breakout_signal = np.full(result_length, 0.0)  # 0=neutral, 1=bullish, -1=bearish

            if channel_result is not None:
                # Fill in the last values (for real-time usage)
                last_idx = result_length - 1
                upper_channel[last_idx] = channel_result.upper_channel
                lower_channel[last_idx] = channel_result.lower_channel
                middle_channel[last_idx] = channel_result.middle_channel
                channel_width[last_idx] = channel_result.channel_width

                # Check for breakout
                current_price = closes[-1]
                if current_price > channel_result.upper_channel * (1 + breakout_threshold):
                    breakout_signal[last_idx] = 1.0  # Bullish breakout
                elif current_price < channel_result.lower_channel * (1 - breakout_threshold):
                    breakout_signal[last_idx] = -1.0  # Bearish breakout

            # Create result DataFrame
            result_df = pd.DataFrame({
                "upper_channel": upper_channel,
                "lower_channel": lower_channel,
                "middle_channel": middle_channel,
                "channel_width": channel_width,
                "breakout_signal": breakout_signal,
            }, index=data.index)

            # Store calculation details for debugging
            self._last_calculation = {
                "fractal_highs_count": len(fractal_highs),
                "fractal_lows_count": len(fractal_lows),
                "channel_constructed": channel_result is not None,
                "parameters_used": self.parameters
            }

            return result_df

        except Exception as e:
            raise IndicatorValidationError(f"Error in FractalChannelIndicator calculation: {e}")

    def validate_parameters(self) -> bool:
        """
        Validate indicator parameters for correctness and trading suitability.
        
        Returns:
            bool: True if parameters are valid
            
        Raises:
            IndicatorValidationError: If parameters are invalid
        """
        period = self.parameters.get("period", 20)
        fractal_period = self.parameters.get("fractal_period", 2)
        channel_lookback = self.parameters.get("channel_lookback", 20)
        min_fractals = self.parameters.get("min_fractals", 2)
        breakout_threshold = self.parameters.get("breakout_threshold", 0.02)
        
        # Validate parameter ranges
        if not isinstance(period, int) or period <= 0:
            raise IndicatorValidationError(f"period must be positive integer, got {period}")
        if not isinstance(fractal_period, int) or fractal_period <= 0:
            raise IndicatorValidationError(f"fractal_period must be positive integer, got {fractal_period}")
        if not isinstance(channel_lookback, int) or channel_lookback <= 0:
            raise IndicatorValidationError(f"channel_lookback must be positive integer, got {channel_lookback}")
        if not isinstance(min_fractals, int) or min_fractals < 1:
            raise IndicatorValidationError(f"min_fractals must be positive integer, got {min_fractals}")
        if not isinstance(breakout_threshold, (int, float)) or breakout_threshold < 0:
            raise IndicatorValidationError(f"breakout_threshold must be non-negative number, got {breakout_threshold}")
            
        # Validate logical relationships
        if fractal_period >= period:
            raise IndicatorValidationError(f"fractal_period ({fractal_period}) must be less than period ({period})")
        if min_fractals > channel_lookback:
            raise IndicatorValidationError(f"min_fractals ({min_fractals}) cannot exceed channel_lookback ({channel_lookback})")
            
        return True

    def get_metadata(self) -> IndicatorMetadata:
        """
        Return comprehensive metadata about the indicator.
        
        Returns:
            IndicatorMetadata: Complete indicator specification
        """
        return IndicatorMetadata(
            name="FractalChannelIndicator",
            category=self.CATEGORY,
            description="Williams Fractal-based dynamic channel analysis for support/resistance identification",
            parameters=self.parameters,
            input_requirements=["high", "low", "close"],
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            trading_grade=True,
            performance_tier="standard",
            min_data_points=self._get_minimum_data_points(),
            max_lookback_period=self.parameters.get("period", 20) + self.parameters.get("channel_lookback", 20)
        )

    def _get_required_columns(self) -> List[str]:
        """Get list of required data columns for this indicator."""
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Get minimum number of data points required for calculation."""
        period = self.parameters.get("period", 20)
        fractal_period = self.parameters.get("fractal_period", 2)
        return max(period, 2 * fractal_period + 1)

    # Backward compatibility properties
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def fractal_period(self) -> int:
        """Fractal period for backward compatibility"""
        return self.parameters.get("fractal_period", 2)

    @property
    def channel_lookback(self) -> int:
        """Channel lookback for backward compatibility"""
        return self.parameters.get("channel_lookback", 20)

    @property
    def min_fractals(self) -> int:
        """Min fractals for backward compatibility"""
        return self.parameters.get("min_fractals", 2)

    @property
    def breakout_threshold(self) -> float:
        """Breakout threshold for backward compatibility"""
        return self.parameters.get("breakout_threshold", 0.02)


def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return FractalChannelIndicator
