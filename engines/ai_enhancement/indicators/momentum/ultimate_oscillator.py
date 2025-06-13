"""
Ultimate Oscillator Indicator
Trading-grade implementation for Platform3

The Ultimate Oscillator is a momentum oscillator developed by Larry Williams
that uses three different time periods to reduce false signals and provide
more reliable overbought/oversold conditions.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Try absolute import first, fall back to relative
try:
    from engines.ai_enhancement.indicators.base_indicator import (
        IndicatorMetadata,
        IndicatorValidationError,
        StandardIndicatorInterface,
    )
except ImportError:
    from ..base_indicator import (
        IndicatorMetadata,
        IndicatorValidationError,
        StandardIndicatorInterface,
    )

logger = logging.getLogger(__name__)


class UltimateOscillatorIndicator(StandardIndicatorInterface):
    """
    Ultimate Oscillator - Multi-timeframe Momentum Oscillator

    Formula:
    1. True Low = min(Low, Previous Close)
    2. Buying Pressure = Close - True Low
    3. True Range = max(High, Previous Close) - True Low
    4. Average 7-period = sum(Buying Pressure, 7) / sum(True Range, 7)
    5. Average 14-period = sum(Buying Pressure, 14) / sum(True Range, 14)
    6. Average 28-period = sum(Buying Pressure, 28) / sum(True Range, 28)
    7. UO = 100 * [(4 * Average 7) + (2 * Average 14) + Average 28] / (4 + 2 + 1)

    The Ultimate Oscillator oscillates between 0 and 100:
    - Values above 70 indicate overbought conditions
    - Values below 30 indicate oversold conditions
    - Divergences with price provide reversal signals
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(
        self,
        short_period: int = 7,
        medium_period: int = 14,
        long_period: int = 28,
        **kwargs,
    ):
        """
        Initialize Ultimate Oscillator

        Args:
            short_period: Short timeframe period (default: 7)
            medium_period: Medium timeframe period (default: 14)
            long_period: Long timeframe period (default: 28)
        """
        super().__init__(
            short_period=short_period,
            medium_period=medium_period,
            long_period=long_period,
            **kwargs,
        )

    def validate_parameters(self) -> bool:
        """Validate Ultimate Oscillator parameters"""
        short_period = self.parameters.get("short_period", 7)
        medium_period = self.parameters.get("medium_period", 14)
        long_period = self.parameters.get("long_period", 28)

        if not isinstance(short_period, int) or short_period < 1:
            raise IndicatorValidationError(
                f"short_period must be positive integer, got {short_period}"
            )

        if not isinstance(medium_period, int) or medium_period < 1:
            raise IndicatorValidationError(
                f"medium_period must be positive integer, got {medium_period}"
            )

        if not isinstance(long_period, int) or long_period < 1:
            raise IndicatorValidationError(
                f"long_period must be positive integer, got {long_period}"
            )

        if not (short_period < medium_period < long_period):
            raise IndicatorValidationError(
                f"periods must be in ascending order: short ({short_period}) < medium ({medium_period}) < long ({long_period})"
            )

        if long_period > 1000:
            raise IndicatorValidationError("periods too large, maximum 1000")

        return True

    def _get_required_columns(self) -> List[str]:
        """Ultimate Oscillator requires OHLC data"""
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        long_period = self.parameters.get("long_period", 28)
        return long_period + 1  # Need one extra for previous close

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Ultimate Oscillator

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            pd.Series: Ultimate Oscillator values
        """
        # Validate input data
        self.validate_input_data(data)

        short_period = self.parameters.get("short_period", 7)
        medium_period = self.parameters.get("medium_period", 14)
        long_period = self.parameters.get("long_period", 28)

        highs = data["high"]
        lows = data["low"]
        closes = data["close"]

        # Calculate previous close
        prev_closes = closes.shift(1)

        # Calculate True Low and True High
        true_lows = pd.concat([lows, prev_closes], axis=1).min(axis=1)
        true_highs = pd.concat([highs, prev_closes], axis=1).max(axis=1)

        # Calculate Buying Pressure and True Range
        buying_pressure = closes - true_lows
        true_range = true_highs - true_lows

        # Calculate averages for each period
        def calculate_average(bp_series, tr_series, period):
            bp_sum = bp_series.rolling(window=period).sum()
            tr_sum = tr_series.rolling(window=period).sum()
            # Handle division by zero
            tr_sum_safe = tr_sum.replace(0, np.nan)
            return bp_sum / tr_sum_safe

        avg_short = calculate_average(buying_pressure, true_range, short_period)
        avg_medium = calculate_average(buying_pressure, true_range, medium_period)
        avg_long = calculate_average(buying_pressure, true_range, long_period)

        # Calculate Ultimate Oscillator
        # UO = 100 * [(4 * avg_short) + (2 * avg_medium) + avg_long] / (4 + 2 + 1)
        numerator = (4 * avg_short) + (2 * avg_medium) + avg_long
        ultimate_oscillator = (numerator / 7) * 100

        # Store calculation details for analysis
        self._last_calculation = {
            "buying_pressure": buying_pressure,
            "true_range": true_range,
            "true_lows": true_lows,
            "true_highs": true_highs,
            "avg_short": avg_short,
            "avg_medium": avg_medium,
            "avg_long": avg_long,
            "short_period": short_period,
            "medium_period": medium_period,
            "long_period": long_period,
        }

        return ultimate_oscillator

    def analyze_result(self, result: pd.Series) -> Dict[str, Any]:
        """
        Analyze Ultimate Oscillator results and generate trading signals

        Args:
            result: Ultimate Oscillator values from calculate()

        Returns:
            Dict containing analysis and signals
        """
        if result is None or len(result) == 0:
            return {"error": "No Ultimate Oscillator data available for analysis"}

        # Get the last valid value
        current_uo = result.dropna().iloc[-1] if not result.dropna().empty else None

        if current_uo is None:
            return {"error": "No valid Ultimate Oscillator values available"}

        # Get recent values for trend analysis
        recent_values = result.dropna().tail(5).tolist()

        # Determine market conditions
        overbought = current_uo > 70
        oversold = current_uo < 30

        # Generate signals based on UO level and trend
        if overbought:
            signal = "sell"
            signal_strength = min(
                100, (current_uo - 70) * 3.33
            )  # Scale 70-100 to 0-100
        elif oversold:
            signal = "buy"
            signal_strength = min(100, (30 - current_uo) * 3.33)  # Scale 0-30 to 100-0
        elif current_uo > 50:
            signal = "bullish"
            signal_strength = (current_uo - 50) * 2  # Scale 50-100 to 0-100
        else:
            signal = "bearish"
            signal_strength = (50 - current_uo) * 2  # Scale 0-50 to 100-0

        # Determine trend from recent values
        if len(recent_values) >= 3:
            if recent_values[-1] > recent_values[-2] > recent_values[-3]:
                trend = "bullish"
            elif recent_values[-1] < recent_values[-2] < recent_values[-3]:
                trend = "bearish"
            else:
                trend = "sideways"
        else:
            trend = "sideways"

        # Check for divergence signals (simplified)
        divergence_signal = None
        if len(recent_values) >= 4:
            # Simple momentum divergence check
            recent_momentum = [
                recent_values[i] - recent_values[i - 1]
                for i in range(1, len(recent_values))
            ]
            if len(recent_momentum) >= 2:
                if recent_momentum[-1] > 0 and recent_momentum[-2] < 0 and oversold:
                    divergence_signal = "bullish_divergence"
                elif recent_momentum[-1] < 0 and recent_momentum[-2] > 0 and overbought:
                    divergence_signal = "bearish_divergence"

        # Calculate confidence based on signal strength and trend consistency
        trend_consistency = 0
        if len(recent_values) >= 2:
            changes = [
                recent_values[i] - recent_values[i - 1]
                for i in range(1, len(recent_values))
            ]
            if changes:
                same_direction = sum(
                    1
                    for i in range(1, len(changes))
                    if (changes[i] > 0) == (changes[i - 1] > 0)
                )
                trend_consistency = (
                    float(same_direction) / (len(changes) - 1)
                    if len(changes) > 1
                    else 0.5
                )

        # Factor in extreme conditions for confidence
        extreme_boost = 0.3 if (overbought or oversold) else 0
        divergence_boost = 0.2 if divergence_signal else 0

        confidence = int(
            (
                signal_strength / 100 * 0.5
                + trend_consistency * 0.3
                + extreme_boost
                + divergence_boost
            )
            * 100
        )

        return {
            "value": float(current_uo),
            "signal": signal,
            "trend": trend,
            "strength": float(signal_strength),
            "confidence": confidence,
            "overbought": bool(overbought),
            "oversold": bool(oversold),
            "divergence_signal": divergence_signal,
            "recent_values": [float(x) for x in recent_values],
            "short_period": self.parameters.get("short_period", 7),
            "medium_period": self.parameters.get("medium_period", 14),
            "long_period": self.parameters.get("long_period", 28),
        }

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata"""
        return IndicatorMetadata(
            name="Ultimate Oscillator",
            category=self.CATEGORY,
            description="Multi-timeframe momentum oscillator with reduced false signals",
            parameters={
                "short_period": {
                    "type": "int",
                    "default": 7,
                    "description": "Short timeframe period",
                },
                "medium_period": {
                    "type": "int",
                    "default": 14,
                    "description": "Medium timeframe period",
                },
                "long_period": {
                    "type": "int",
                    "default": 28,
                    "description": "Long timeframe period",
                },
            },
            input_requirements=["high", "low", "close"],
            output_type="pd.Series",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self.parameters.get("long_period", 28) + 1,
        )
