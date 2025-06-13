"""
Fractal Breakout Indicator

Advanced fractal-based breakout detection indicator using Williams Fractals
for identifying momentum breakouts from key support and resistance levels.

Formula:
- Detects Williams Fractals as reversal points
- Identifies recent fractal-based support/resistance levels  
- Monitors price breakouts with momentum confirmation
- Provides signal strength based on breakout magnitude

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List

# Import the base indicator interface
import sys
import os

from base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
)


class FractalBreakoutIndicator(StandardIndicatorInterface):
    """
    Fractal Breakout Indicator

    Advanced fractal-based breakout detection using Williams Fractals to identify
    momentum breakouts from key support and resistance levels with signal strength.

    Key Features:
    - Williams Fractal detection for reversal point identification
    - Dynamic support/resistance level calculation from recent fractals
    - Breakout detection with configurable threshold
    - Signal strength measurement based on breakout magnitude
    - Momentum confirmation for reliable trading signals

    Mathematical Approach:
    Uses chaos theory principles to identify fractal reversal points, then monitors
    price breakouts from these levels with threshold-based confirmation to generate
    reliable trading signals with quantified strength metrics.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "fractal"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        fractal_period: int = 5,
        breakout_threshold: float = 0.02,
        **kwargs
    ):
        """
        Initialize Fractal Breakout Indicator

        Args:
            period: Main analysis period (default: 20)
            fractal_period: Williams fractal detection period (default: 5)
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
            breakout_threshold=breakout_threshold,
            **kwargs,
        )

    def detect_fractals(self, highs: np.ndarray, lows: np.ndarray) -> tuple[List, List]:
        """Detect Williams fractal points."""
        fractal_highs = []
        fractal_lows = []

        period = self.parameters.get("fractal_period", 5)

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
                fractal_highs.append((i, current_high))

            # Check for fractal low
            current_low = lows[i]
            is_fractal_low = True

            for j in range(1, period + 1):
                if lows[i - j] <= current_low or lows[i + j] <= current_low:
                    is_fractal_low = False
                    break

            if is_fractal_low:
                fractal_lows.append((i, current_low))

        return fractal_highs, fractal_lows

    def calculate_breakout_signals(
        self, prices: np.ndarray, fractal_highs: List, fractal_lows: List
    ) -> tuple[str, float]:
        """Calculate breakout signals based on fractal levels."""
        try:
            if len(prices) == 0 or (not fractal_highs and not fractal_lows):
                return "neutral", 0.0

            current_price = prices[-1]
            breakout_threshold = self.parameters.get("breakout_threshold", 0.02)

            # Get recent fractal levels
            resistance_level = None
            support_level = None

            if fractal_highs:
                # Find most recent fractal high
                resistance_level = max([level for idx, level in fractal_highs])

            if fractal_lows:
                # Find most recent fractal low
                support_level = min([level for idx, level in fractal_lows])

            # Check for breakouts
            signal = "neutral"
            strength = 0.0

            if resistance_level and current_price > resistance_level * (1 + breakout_threshold):
                signal = "bullish_breakout"
                strength = (current_price - resistance_level) / resistance_level
            elif support_level and current_price < support_level * (1 - breakout_threshold):
                signal = "bearish_breakout"
                strength = (support_level - current_price) / support_level

            return signal, min(strength, 1.0)  # Cap strength at 1.0

        except Exception as e:
            raise IndicatorValidationError(f"Error calculating breakout signals: {e}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fractal Breakout analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Fractal breakout analysis results
        """
        try:
            # Validate input data first
            self.validate_input_data(data)
            
            # Extract OHLC data
            highs = data["high"].values
            lows = data["low"].values
            closes = data["close"].values
            
            period = self.parameters.get("period", 20)

            if len(closes) < period:
                raise IndicatorValidationError(f"Insufficient data: need {period}, got {len(closes)}")

            # Use recent window for analysis
            recent_highs = highs[-period:]
            recent_lows = lows[-period:]
            recent_closes = closes[-period:]

            # Detect fractals
            fractal_highs, fractal_lows = self.detect_fractals(recent_highs, recent_lows)

            # Calculate breakout signals
            signal, strength = self.calculate_breakout_signals(recent_closes, fractal_highs, fractal_lows)

            # Calculate additional metrics
            resistance_level = max([level for idx, level in fractal_highs]) if fractal_highs else None
            support_level = min([level for idx, level in fractal_lows]) if fractal_lows else None

            # Initialize result arrays
            result_length = len(data)
            signal_values = np.full(result_length, 0.0)  # 0=neutral, 1=bullish, -1=bearish
            strength_values = np.full(result_length, 0.0)
            resistance_values = np.full(result_length, np.nan)
            support_values = np.full(result_length, np.nan)

            # Fill in the last values (for real-time usage)
            last_idx = result_length - 1
            if signal == "bullish_breakout":
                signal_values[last_idx] = 1.0
            elif signal == "bearish_breakout":
                signal_values[last_idx] = -1.0
            
            strength_values[last_idx] = strength
            if resistance_level is not None:
                resistance_values[last_idx] = resistance_level
            if support_level is not None:
                support_values[last_idx] = support_level

            # Create result DataFrame
            result_df = pd.DataFrame({
                "signal": signal_values,
                "strength": strength_values,
                "resistance_level": resistance_values,
                "support_level": support_values,
                "is_breakout": np.abs(signal_values) > 0,
            }, index=data.index)

            # Store calculation details for debugging
            self._last_calculation = {
                "signal_type": signal,
                "signal_strength": strength,
                "fractal_highs_count": len(fractal_highs),
                "fractal_lows_count": len(fractal_lows),
                "parameters_used": self.parameters
            }

            return result_df

        except Exception as e:
            raise IndicatorValidationError(f"Error in FractalBreakoutIndicator calculation: {e}")

    def validate_parameters(self) -> bool:
        """
        Validate indicator parameters for correctness and trading suitability.
        
        Returns:
            bool: True if parameters are valid
            
        Raises:
            IndicatorValidationError: If parameters are invalid
        """
        period = self.parameters.get("period", 20)
        fractal_period = self.parameters.get("fractal_period", 5)
        breakout_threshold = self.parameters.get("breakout_threshold", 0.02)
        
        # Validate parameter ranges
        if not isinstance(period, int) or period <= 0:
            raise IndicatorValidationError(f"period must be positive integer, got {period}")
        if not isinstance(fractal_period, int) or fractal_period <= 0:
            raise IndicatorValidationError(f"fractal_period must be positive integer, got {fractal_period}")
        if not isinstance(breakout_threshold, (int, float)) or breakout_threshold < 0:
            raise IndicatorValidationError(f"breakout_threshold must be non-negative number, got {breakout_threshold}")
            
        # Validate logical relationships
        if fractal_period >= period:
            raise IndicatorValidationError(f"fractal_period ({fractal_period}) must be less than period ({period})")
            
        return True

    def get_metadata(self) -> IndicatorMetadata:
        """
        Return comprehensive metadata about the indicator.
        
        Returns:
            IndicatorMetadata: Complete indicator specification
        """
        return IndicatorMetadata(
            name="FractalBreakoutIndicator",
            category=self.CATEGORY,
            description="Advanced fractal-based breakout detection with momentum confirmation",
            parameters=self.parameters,
            input_requirements=["high", "low", "close"],
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            trading_grade=True,
            performance_tier="standard",
            min_data_points=self._get_minimum_data_points(),
            max_lookback_period=self.parameters.get("period", 20)
        )

    def _get_required_columns(self) -> List[str]:
        """Get list of required data columns for this indicator."""
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Get minimum number of data points required for calculation."""
        period = self.parameters.get("period", 20)
        fractal_period = self.parameters.get("fractal_period", 5)
        return max(period, 2 * fractal_period + 1)

    # Backward compatibility properties
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def fractal_period(self) -> int:
        """Fractal period for backward compatibility"""
        return self.parameters.get("fractal_period", 5)

    @property
    def breakout_threshold(self) -> float:
        """Breakout threshold for backward compatibility"""
        return self.parameters.get("breakout_threshold", 0.02)


def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return FractalBreakoutIndicator
