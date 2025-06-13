"""
Average Directional Index (ADX) Indicator

The ADX is a technical analysis indicator used to measure the strength or weakness
of a trend. ADX is non-directional - it quantifies trend strength without regard
to trend direction.

Components:
- ADX: Average Directional Index (trend strength)
- +DI: Positive Directional Indicator (upward movement strength)
- -DI: Negative Directional Indicator (downward movement strength)

Formula:
1. True Range (TR) = max(high - low, abs(high - close_prev), abs(low - close_prev))
2. +DM = high - high_prev (if positive, else 0)
3. -DM = low_prev - low (if positive, else 0)
4. +DI = 100 * smoothed(+DM) / smoothed(TR)
5. -DI = 100 * smoothed(-DM) / smoothed(TR)
6. DX = 100 * abs(+DI - -DI) / (+DI + -DI)
7. ADX = smoothed(DX)

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

# Import the base indicator interface
import sys
import os

from base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
)


class ADXIndicator(StandardIndicatorInterface):
    """
    Average Directional Index (ADX) Indicator

    Measures the strength of a trend without regard to direction.
    Returns ADX along with +DI and -DI components.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 14,
        smoothing_period: int = 14,
        **kwargs,
    ):
        """
        Initialize ADX indicator

        Args:
            period: Period for DI calculations (default: 14)
            smoothing_period: Period for ADX smoothing (default: 14)
        """
        super().__init__(
            period=period,
            smoothing_period=smoothing_period,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate ADX indicator

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            pd.DataFrame: DataFrame with ADX, +DI, -DI columns
        """
        # Validate input data
        if isinstance(data, pd.Series):
            raise IndicatorValidationError("ADX requires OHLC data, not single Series")

        if not isinstance(data, pd.DataFrame):
            raise IndicatorValidationError("Data must be DataFrame")

        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in data.columns:
                raise IndicatorValidationError(f"DataFrame must contain '{col}' column")

        self.validate_input_data(data)

        period = self.parameters.get("period", 14)
        smoothing_period = self.parameters.get("smoothing_period", 14)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate True Range (TR)
        high_low = high - low
        high_close_prev = abs(high - close.shift(1))
        low_close_prev = abs(low - close.shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # Calculate Directional Movement (+DM and -DM)
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low

        plus_dm = pd.Series(
            np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0),
            index=data.index,
        )

        minus_dm = pd.Series(
            np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0),
            index=data.index,
        )

        # Smooth TR, +DM, and -DM using Wilder's smoothing
        smoothed_tr = self._wilders_smoothing(tr, period)
        smoothed_plus_dm = self._wilders_smoothing(plus_dm, period)
        smoothed_minus_dm = self._wilders_smoothing(minus_dm, period)

        # Calculate Directional Indicators (+DI and -DI)
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr

        # Calculate Directional Index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        # Calculate ADX (smoothed DX)
        adx = self._wilders_smoothing(dx, smoothing_period)

        # Create result DataFrame
        result = pd.DataFrame(
            {"ADX": adx, "PLUS_DI": plus_di, "MINUS_DI": minus_di}, index=data.index
        )

        # Store calculation details for analysis
        self._last_calculation = {
            "tr": tr,
            "plus_dm": plus_dm,
            "minus_dm": minus_dm,
            "smoothed_tr": smoothed_tr,
            "smoothed_plus_dm": smoothed_plus_dm,
            "smoothed_minus_dm": smoothed_minus_dm,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "dx": dx,
            "adx": adx,
        }

        return result

    def _wilders_smoothing(self, series: pd.Series, period: int) -> pd.Series:
        """
        Apply Wilder's smoothing (modified exponential moving average)

        Args:
            series: Series to smooth
            period: Smoothing period

        Returns:
            pd.Series: Smoothed series
        """
        # First value is simple average of first 'period' values
        result = pd.Series(index=series.index, dtype=float)
        result[:] = np.nan

        # Calculate first smoothed value
        for i in range(period - 1, len(series)):
            if i == period - 1:
                # First calculation: simple average
                result.iloc[i] = series.iloc[i - period + 1 : i + 1].mean()
            else:
                # Subsequent calculations: Wilder's smoothing
                # smoothed = (previous_smoothed * (period - 1) + current_value) / period
                result.iloc[i] = (
                    result.iloc[i - 1] * (period - 1) + series.iloc[i]
                ) / period

        return result

    def validate_parameters(self) -> bool:
        """Validate ADX parameters"""
        period = self.parameters.get("period", 14)
        smoothing_period = self.parameters.get("smoothing_period", 14)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(smoothing_period, int) or smoothing_period < 1:
            raise IndicatorValidationError(
                f"smoothing_period must be positive integer, got {smoothing_period}"
            )

        if smoothing_period > 1000:
            raise IndicatorValidationError(
                f"smoothing_period too large, maximum 1000, got {smoothing_period}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return ADX metadata as dictionary for compatibility"""
        return {
            "name": "ADX",
            "category": self.CATEGORY,
            "description": "Average Directional Index - Measures trend strength without regard to direction",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
            "output_columns": ["ADX", "PLUS_DI", "MINUS_DI"],
        }

    def _get_required_columns(self) -> List[str]:
        """ADX requires high, low, close prices"""
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for ADX calculation"""
        period = self.parameters.get("period", 14)
        smoothing_period = self.parameters.get("smoothing_period", 14)
        return period + smoothing_period + 1

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 14
        if "smoothing_period" not in self.parameters:
            self.parameters["smoothing_period"] = 14

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 14)

    @property
    def smoothing_period(self) -> int:
        """Smoothing period for backward compatibility"""
        return self.parameters.get("smoothing_period", 14)

    def get_trend_strength(self, adx_value: float) -> str:
        """
        Get trend strength classification based on ADX value

        Args:
            adx_value: Current ADX value

        Returns:
            str: "weak", "moderate", "strong", or "very_strong"
        """
        if adx_value < 20:
            return "weak"
        elif adx_value < 40:
            return "moderate"
        elif adx_value < 60:
            return "strong"
        else:
            return "very_strong"

    def get_trend_direction(self, plus_di: float, minus_di: float) -> str:
        """
        Get trend direction based on DI values

        Args:
            plus_di: Current +DI value
            minus_di: Current -DI value

        Returns:
            str: "bullish", "bearish", or "neutral"
        """
        if plus_di > minus_di:
            return "bullish"
        elif minus_di > plus_di:
            return "bearish"
        else:
            return "neutral"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return ADXIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLC data
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate realistic OHLC data
    close_prices = [base_price]
    for _ in range(n_points - 1):
        change = np.random.randn() * 0.5
        close_prices.append(close_prices[-1] + change)

    # Create OHLC from close prices
    data = pd.DataFrame(
        {
            "close": close_prices,
            "high": [c + abs(np.random.randn() * 0.3) for c in close_prices],
            "low": [c - abs(np.random.randn() * 0.3) for c in close_prices],
        }
    )

    # Calculate ADX
    adx = ADXIndicator()
    result = adx.calculate(data)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Price chart
    ax1.plot(data["close"].values, label="Close Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # ADX chart
    ax2.plot(result["ADX"].values, label="ADX", color="purple", linewidth=2)
    ax2.axhline(y=20, color="red", linestyle="--", alpha=0.7, label="Weak threshold")
    ax2.axhline(
        y=40, color="orange", linestyle="--", alpha=0.7, label="Strong threshold"
    )
    ax2.set_title("ADX - Trend Strength")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)

    # DI chart
    ax3.plot(result["PLUS_DI"].values, label="+DI", color="green")
    ax3.plot(result["MINUS_DI"].values, label="-DI", color="red")
    ax3.set_title("Directional Indicators")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"ADX calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"ADX parameters: {adx.parameters}")
    print(f"Current ADX: {result['ADX'].iloc[-1]:.2f}")
    print(f"Current +DI: {result['PLUS_DI'].iloc[-1]:.2f}")
    print(f"Current -DI: {result['MINUS_DI'].iloc[-1]:.2f}")
    print(f"Trend strength: {adx.get_trend_strength(result['ADX'].iloc[-1])}")
    print(
        f"Trend direction: {adx.get_trend_direction(result['PLUS_DI'].iloc[-1], result['MINUS_DI'].iloc[-1])}"
    )
