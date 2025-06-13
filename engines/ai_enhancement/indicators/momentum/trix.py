"""
TRIX Indicator
Trading-grade implementation for Platform3

TRIX is a momentum oscillator that shows the percentage rate of change
of a triple exponentially smoothed moving average. It's designed to filter
out market noise and reveal the underlying trend.
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


class TRIXIndicator(StandardIndicatorInterface):
    """
    TRIX - Triple Exponential Average Momentum Oscillator

    Formula:
    1. First EMA = EMA(Close, period)
    2. Second EMA = EMA(First EMA, period)
    3. Third EMA = EMA(Second EMA, period)
    4. TRIX = (Third EMA today - Third EMA yesterday) / Third EMA yesterday * 10000

    TRIX oscillates around zero:
    - Positive values indicate upward momentum
    - Negative values indicate downward momentum
    - Zero line crossovers generate trading signals
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(self, period: int = 14, signal_period: int = 9, **kwargs):
        """
        Initialize TRIX

        Args:
            period: Period for triple EMA calculation (default: 14)
            signal_period: Period for signal line (default: 9)
        """
        super().__init__(period=period, signal_period=signal_period, **kwargs)

    def validate_parameters(self) -> bool:
        """Validate TRIX parameters"""
        period = self.parameters.get("period", 14)
        signal_period = self.parameters.get("signal_period", 9)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if not isinstance(signal_period, int) or signal_period < 1:
            raise IndicatorValidationError(
                f"signal_period must be positive integer, got {signal_period}"
            )

        if period > 1000 or signal_period > 1000:
            raise IndicatorValidationError("periods too large, maximum 1000")

        return True

    def _get_required_columns(self) -> List[str]:
        """TRIX requires close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        period = self.parameters.get("period", 14)
        signal_period = self.parameters.get("signal_period", 9)
        # Need enough data for triple smoothing plus signal line
        return (period * 3) + signal_period

    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate TRIX

        Args:
            data: DataFrame with 'close' column

        Returns:
            pd.DataFrame: TRIX values with columns ['trix', 'signal']
        """
        # Validate input data
        self.validate_input_data(data)

        period = self.parameters.get("period", 14)
        signal_period = self.parameters.get("signal_period", 9)

        closes = data["close"]

        # Calculate triple exponential smoothing
        first_ema = self._calculate_ema(closes, period)
        second_ema = self._calculate_ema(first_ema, period)
        third_ema = self._calculate_ema(second_ema, period)

        # Calculate TRIX (percentage rate of change of third EMA)
        trix = third_ema.pct_change() * 10000

        # Calculate signal line (EMA of TRIX)
        signal_line = self._calculate_ema(trix.dropna(), signal_period)

        # Align signal line with TRIX index
        signal_aligned = pd.Series(index=trix.index, dtype=float)
        signal_aligned.loc[signal_line.index] = signal_line

        # Create result DataFrame
        result = pd.DataFrame(
            {"trix": trix, "signal": signal_aligned}, index=data.index
        )

        # Store calculation details for analysis
        self._last_calculation = {
            "first_ema": first_ema,
            "second_ema": second_ema,
            "third_ema": third_ema,
            "period": period,
            "signal_period": signal_period,
        }

        return result

    def analyze_result(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze TRIX results and generate trading signals

        Args:
            result: TRIX DataFrame from calculate()

        Returns:
            Dict containing analysis and signals
        """
        if result is None or len(result) == 0:
            return {"error": "No TRIX data available for analysis"}

        # Get the last valid values
        last_valid_data = result.dropna().tail(1)
        if len(last_valid_data) == 0:
            return {"error": "No valid TRIX values available"}

        current_trix = last_valid_data["trix"].iloc[0]
        current_signal = last_valid_data["signal"].iloc[0]

        # Get recent values for trend analysis
        recent_data = result.dropna().tail(5)
        recent_trix = recent_data["trix"].tolist()
        recent_signal = recent_data["signal"].tolist()

        # Determine signal based on TRIX analysis
        trix_above_zero = current_trix > 0
        trix_above_signal = current_trix > current_signal

        # Generate primary signal
        if trix_above_zero and trix_above_signal:
            signal = "bullish"
        elif not trix_above_zero and not trix_above_signal:
            signal = "bearish"
        else:
            signal = "neutral"

        # Check for zero line crossovers
        zero_crossover = None
        if len(recent_trix) >= 2:
            # Bullish zero crossover: TRIX crosses above zero
            if recent_trix[-2] <= 0 and current_trix > 0:
                zero_crossover = "bullish_zero_cross"
            # Bearish zero crossover: TRIX crosses below zero
            elif recent_trix[-2] >= 0 and current_trix < 0:
                zero_crossover = "bearish_zero_cross"

        # Check for signal line crossovers
        signal_crossover = None
        if len(recent_trix) >= 2 and len(recent_signal) >= 2:
            # Bullish crossover: TRIX crosses above signal line
            if recent_trix[-2] <= recent_signal[-2] and current_trix > current_signal:
                signal_crossover = "bullish_signal_cross"
            # Bearish crossover: TRIX crosses below signal line
            elif recent_trix[-2] >= recent_signal[-2] and current_trix < current_signal:
                signal_crossover = "bearish_signal_cross"

        # Determine trend from TRIX direction
        if len(recent_trix) >= 3:
            if recent_trix[-1] > recent_trix[-2] > recent_trix[-3]:
                trend = "bullish"
            elif recent_trix[-1] < recent_trix[-2] < recent_trix[-3]:
                trend = "bearish"
            else:
                trend = "sideways"
        else:
            trend = "sideways"

        # Calculate signal strength based on TRIX magnitude
        trix_magnitude = abs(current_trix)
        if len(recent_trix) >= 5:
            avg_magnitude = np.mean([abs(x) for x in recent_trix])
            if avg_magnitude > 0:
                strength = min(100, (trix_magnitude / avg_magnitude) * 50)
            else:
                strength = 50
        else:
            strength = 50

        # Boost strength for significant crossovers
        if zero_crossover and signal == "bullish":
            strength = min(100, strength * 1.5)
        elif zero_crossover and signal == "bearish":
            strength = min(100, strength * 1.5)

        # Calculate confidence based on signal consistency and crossovers
        confidence = 50
        if len(recent_trix) >= 3:
            trend_consistency = 0.0
            for i in range(1, len(recent_trix)):
                if (recent_trix[i] > recent_trix[i - 1] and signal == "bullish") or (
                    recent_trix[i] < recent_trix[i - 1] and signal == "bearish"
                ):
                    trend_consistency += 1
            trend_consistency /= len(recent_trix) - 1

            # Factor in crossover signals
            crossover_boost = 0.3 if (zero_crossover or signal_crossover) else 0
            confidence = int((trend_consistency * 0.7 + crossover_boost) * 100)

        return {
            "trix": float(current_trix),
            "signal_line": float(current_signal),
            "signal": signal,
            "trend": trend,
            "strength": float(strength),
            "confidence": confidence,
            "zero_crossover": zero_crossover,
            "signal_crossover": signal_crossover,
            "trix_above_zero": bool(trix_above_zero),
            "trix_above_signal": bool(trix_above_signal),
            "recent_trix": [float(x) for x in recent_trix],
            "recent_signal": [float(x) for x in recent_signal],
            "period": self.parameters.get("period", 14),
            "signal_period": self.parameters.get("signal_period", 9),
        }

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata"""
        return IndicatorMetadata(
            name="TRIX",
            category=self.CATEGORY,
            description="Triple exponential average momentum oscillator for trend analysis",
            parameters={
                "period": {
                    "type": "int",
                    "default": 14,
                    "description": "Period for triple EMA calculation",
                },
                "signal_period": {
                    "type": "int",
                    "default": 9,
                    "description": "Period for signal line",
                },
            },
            input_requirements=["close"],
            output_type="pd.DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=(self.parameters.get("period", 14) * 3)
            + self.parameters.get("signal_period", 9),
        )
