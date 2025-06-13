"""
True Strength Index (TSI) Indicator
Trading-grade implementation for Platform3

The True Strength Index is a momentum oscillator that uses double-smoothed
price changes to reduce market noise and provide clearer trend signals.
It oscillates around zero with overbought/oversold levels.
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


class TrueStrengthIndexIndicator(StandardIndicatorInterface):
    """
    True Strength Index (TSI) - Double Smoothed Momentum Oscillator

    Formula:
    1. Price Change = Close - Previous Close
    2. First Smoothing:
       - PC_S1 = EMA(Price Change, long_period)
       - APC_S1 = EMA(|Price Change|, long_period)
    3. Second Smoothing:
       - PC_S2 = EMA(PC_S1, short_period)
       - APC_S2 = EMA(APC_S1, short_period)
    4. TSI = (PC_S2 / APC_S2) * 100

    TSI oscillates between -100 and +100:
    - Values above +25 often indicate overbought conditions
    - Values below -25 often indicate oversold conditions
    - Zero line crossovers generate trading signals
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(
        self,
        long_period: int = 25,
        short_period: int = 13,
        signal_period: int = 7,
        **kwargs,
    ):
        """
        Initialize True Strength Index

        Args:
            long_period: Period for first smoothing (default: 25)
            short_period: Period for second smoothing (default: 13)
            signal_period: Period for signal line (default: 7)
        """
        super().__init__(
            long_period=long_period,
            short_period=short_period,
            signal_period=signal_period,
            **kwargs,
        )

    def validate_parameters(self) -> bool:
        """Validate TSI parameters"""
        long_period = self.parameters.get("long_period", 25)
        short_period = self.parameters.get("short_period", 13)
        signal_period = self.parameters.get("signal_period", 7)

        if not isinstance(long_period, int) or long_period < 1:
            raise IndicatorValidationError(
                f"long_period must be positive integer, got {long_period}"
            )

        if not isinstance(short_period, int) or short_period < 1:
            raise IndicatorValidationError(
                f"short_period must be positive integer, got {short_period}"
            )

        if not isinstance(signal_period, int) or signal_period < 1:
            raise IndicatorValidationError(
                f"signal_period must be positive integer, got {signal_period}"
            )

        if short_period >= long_period:
            raise IndicatorValidationError(
                f"short_period ({short_period}) must be less than long_period ({long_period})"
            )

        if max(long_period, short_period, signal_period) > 1000:
            raise IndicatorValidationError("periods too large, maximum 1000")

        return True

    def _get_required_columns(self) -> List[str]:
        """TSI requires close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        long_period = self.parameters.get("long_period", 25)
        short_period = self.parameters.get("short_period", 13)
        signal_period = self.parameters.get("signal_period", 7)
        return long_period + short_period + signal_period

    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate True Strength Index

        Args:
            data: DataFrame with 'close' column

        Returns:
            pd.DataFrame: TSI values with columns ['tsi', 'signal']
        """
        # Validate input data
        self.validate_input_data(data)

        long_period = self.parameters.get("long_period", 25)
        short_period = self.parameters.get("short_period", 13)
        signal_period = self.parameters.get("signal_period", 7)

        closes = data["close"]

        # Calculate price changes
        price_changes = closes.diff()

        # Calculate absolute price changes
        abs_price_changes = price_changes.abs()

        # First smoothing
        pc_s1 = self._calculate_ema(price_changes, long_period)
        apc_s1 = self._calculate_ema(abs_price_changes, long_period)

        # Second smoothing
        pc_s2 = self._calculate_ema(pc_s1, short_period)
        apc_s2 = self._calculate_ema(apc_s1, short_period)

        # Calculate TSI
        # Handle division by zero by replacing zero values with NaN
        apc_s2_safe = apc_s2.replace(0, np.nan)
        tsi = (pc_s2 / apc_s2_safe) * 100

        # Calculate signal line (EMA of TSI)
        signal_line = self._calculate_ema(tsi.dropna(), signal_period)

        # Align signal line with TSI index
        signal_aligned = pd.Series(index=tsi.index, dtype=float)
        signal_aligned.loc[signal_line.index] = signal_line

        # Create result DataFrame
        result = pd.DataFrame({"tsi": tsi, "signal": signal_aligned}, index=data.index)

        # Store calculation details for analysis
        self._last_calculation = {
            "price_changes": price_changes,
            "abs_price_changes": abs_price_changes,
            "pc_s1": pc_s1,
            "apc_s1": apc_s1,
            "pc_s2": pc_s2,
            "apc_s2": apc_s2,
            "long_period": long_period,
            "short_period": short_period,
            "signal_period": signal_period,
        }

        return result

    def analyze_result(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze TSI results and generate trading signals

        Args:
            result: TSI DataFrame from calculate()

        Returns:
            Dict containing analysis and signals
        """
        if result is None or len(result) == 0:
            return {"error": "No TSI data available for analysis"}

        # Get the last valid values
        last_valid_data = result.dropna().tail(1)
        if len(last_valid_data) == 0:
            return {"error": "No valid TSI values available"}

        current_tsi = last_valid_data["tsi"].iloc[0]
        current_signal = last_valid_data["signal"].iloc[0]

        # Get recent values for trend analysis
        recent_data = result.dropna().tail(5)
        recent_tsi = recent_data["tsi"].tolist()
        recent_signal = recent_data["signal"].tolist()

        # Determine market conditions
        overbought = current_tsi > 25
        oversold = current_tsi < -25
        tsi_above_zero = current_tsi > 0
        tsi_above_signal = current_tsi > current_signal

        # Generate primary signal
        if overbought:
            signal = "sell"
        elif oversold:
            signal = "buy"
        elif tsi_above_signal and tsi_above_zero:
            signal = "bullish"
        elif not tsi_above_signal and not tsi_above_zero:
            signal = "bearish"
        else:
            signal = "neutral"

        # Check for zero line crossovers
        zero_crossover = None
        if len(recent_tsi) >= 2:
            # Bullish zero crossover: TSI crosses above zero
            if recent_tsi[-2] <= 0 and current_tsi > 0:
                zero_crossover = "bullish_zero_cross"
            # Bearish zero crossover: TSI crosses below zero
            elif recent_tsi[-2] >= 0 and current_tsi < 0:
                zero_crossover = "bearish_zero_cross"

        # Check for signal line crossovers
        signal_crossover = None
        if len(recent_tsi) >= 2 and len(recent_signal) >= 2:
            # Bullish crossover: TSI crosses above signal line
            if recent_tsi[-2] <= recent_signal[-2] and current_tsi > current_signal:
                signal_crossover = "bullish_signal_cross"
            # Bearish crossover: TSI crosses below signal line
            elif recent_tsi[-2] >= recent_signal[-2] and current_tsi < current_signal:
                signal_crossover = "bearish_signal_cross"

        # Determine trend from TSI direction
        if len(recent_tsi) >= 3:
            if recent_tsi[-1] > recent_tsi[-2] > recent_tsi[-3]:
                trend = "bullish"
            elif recent_tsi[-1] < recent_tsi[-2] < recent_tsi[-3]:
                trend = "bearish"
            else:
                trend = "sideways"
        else:
            trend = "sideways"

        # Calculate signal strength based on TSI magnitude and position
        tsi_magnitude = abs(current_tsi)

        # Base strength on how extreme the TSI value is
        if overbought or oversold:
            strength = min(100, tsi_magnitude * 2)  # Scale extreme values
        else:
            strength = min(100, tsi_magnitude * 1.5)  # Scale normal values

        # Boost strength for significant crossovers
        if zero_crossover:
            strength = min(100, strength * 1.3)
        elif signal_crossover:
            strength = min(100, strength * 1.2)

        # Calculate confidence based on signal consistency and crossovers
        confidence = 50
        if len(recent_tsi) >= 3:
            trend_consistency = 0.0
            for i in range(1, len(recent_tsi)):
                if (
                    recent_tsi[i] > recent_tsi[i - 1] and signal in ["bullish", "buy"]
                ) or (
                    recent_tsi[i] < recent_tsi[i - 1] and signal in ["bearish", "sell"]
                ):
                    trend_consistency += 1
            trend_consistency /= len(recent_tsi) - 1

            # Factor in crossover signals and extreme conditions
            crossover_boost = 0.3 if (zero_crossover or signal_crossover) else 0
            extreme_boost = 0.2 if (overbought or oversold) else 0
            confidence = int(
                (trend_consistency * 0.5 + crossover_boost + extreme_boost) * 100
            )

        return {
            "tsi": float(current_tsi),
            "signal_line": float(current_signal),
            "signal": signal,
            "trend": trend,
            "strength": float(strength),
            "confidence": confidence,
            "overbought": bool(overbought),
            "oversold": bool(oversold),
            "zero_crossover": zero_crossover,
            "signal_crossover": signal_crossover,
            "tsi_above_zero": bool(tsi_above_zero),
            "tsi_above_signal": bool(tsi_above_signal),
            "recent_tsi": [float(x) for x in recent_tsi],
            "recent_signal": [float(x) for x in recent_signal],
            "long_period": self.parameters.get("long_period", 25),
            "short_period": self.parameters.get("short_period", 13),
            "signal_period": self.parameters.get("signal_period", 7),
        }

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata"""
        return IndicatorMetadata(
            name="True Strength Index",
            category=self.CATEGORY,
            description="Double-smoothed momentum oscillator with reduced noise",
            parameters={
                "long_period": {
                    "type": "int",
                    "default": 25,
                    "description": "Period for first smoothing",
                },
                "short_period": {
                    "type": "int",
                    "default": 13,
                    "description": "Period for second smoothing",
                },
                "signal_period": {
                    "type": "int",
                    "default": 7,
                    "description": "Period for signal line",
                },
            },
            input_requirements=["close"],
            output_type="pd.DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self.parameters.get("long_period", 25)
            + self.parameters.get("short_period", 13)
            + self.parameters.get("signal_period", 7),
        )
