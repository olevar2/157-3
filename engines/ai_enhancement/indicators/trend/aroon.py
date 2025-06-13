"""
Aroon Indicator

The Aroon indicator is a technical analysis tool used to identify trend changes
and the strength of the trend. It consists of two components: Aroon Up and Aroon Down.

Components:
- Aroon Up: ((period - periods since highest high) / period) * 100
- Aroon Down: ((period - periods since lowest low) / period) * 100
- Aroon Oscillator: Aroon Up - Aroon Down (optional)

Interpretation:
- Aroon Up near 100: Recent new highs (bullish)
- Aroon Down near 100: Recent new lows (bearish)
- Both near 50: Sideways/consolidating market
- Aroon Up > Aroon Down: Uptrend
- Aroon Down > Aroon Up: Downtrend

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


class AroonIndicator(StandardIndicatorInterface):
    """
    Aroon Indicator

    Identifies trend changes and measures trend strength using the concept
    of time elapsed since the last high or low within a given period.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 14,
        include_oscillator: bool = True,
        **kwargs,
    ):
        """
        Initialize Aroon indicator

        Args:
            period: Period for Aroon calculation (default: 14)
            include_oscillator: Whether to include Aroon Oscillator (default: True)
        """
        super().__init__(
            period=period,
            include_oscillator=include_oscillator,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Aroon indicator

        Args:
            data: DataFrame with 'high' and 'low' columns, or Series of prices

        Returns:
            pd.DataFrame: DataFrame with AROON_UP, AROON_DOWN, and optionally AROON_OSC columns
        """  # Handle input data
        if isinstance(data, pd.Series):
            # If Series provided, use it for both high and low
            high = low = data
        elif isinstance(data, pd.DataFrame):
            if "high" in data.columns and "low" in data.columns:
                high = data["high"]
                low = data["low"]
                self.validate_input_data(data)
            elif "close" in data.columns:
                # If only close is available, use close for both high and low
                high = low = data["close"]
                # Create a temporary DataFrame for validation with required columns
                temp_data = pd.DataFrame(
                    {"high": high, "low": low, "close": data["close"]}
                )
                self.validate_input_data(temp_data)
            else:
                raise IndicatorValidationError(
                    "DataFrame must contain 'high' and 'low' columns, or 'close' column"
                )
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        period = self.parameters.get("period", 14)
        include_oscillator = self.parameters.get("include_oscillator", True)

        # Calculate periods since highest high and lowest low
        aroon_up = pd.Series(index=high.index, dtype=float)
        aroon_down = pd.Series(index=low.index, dtype=float)

        # Initialize with NaN
        aroon_up[:] = np.nan
        aroon_down[:] = np.nan

        # Calculate Aroon for each period
        for i in range(period - 1, len(high)):
            # Get the window for this calculation
            start_idx = i - period + 1
            end_idx = i + 1

            high_window = high.iloc[start_idx:end_idx]
            low_window = low.iloc[start_idx:end_idx]

            # Find the position of highest high and lowest low within the window
            highest_high_pos = high_window.idxmax()
            lowest_low_pos = low_window.idxmin()

            # Calculate periods since the highest high and lowest low
            # The position is relative to the start of our data, so we need to convert
            # to relative position within the window
            highest_high_idx = high_window.index.get_loc(highest_high_pos)
            lowest_low_idx = low_window.index.get_loc(lowest_low_pos)

            periods_since_high = period - 1 - highest_high_idx
            periods_since_low = period - 1 - lowest_low_idx

            # Calculate Aroon values
            aroon_up.iloc[i] = ((period - periods_since_high) / period) * 100
            aroon_down.iloc[i] = ((period - periods_since_low) / period) * 100

        # Create result DataFrame
        result_dict = {"AROON_UP": aroon_up, "AROON_DOWN": aroon_down}

        # Add Aroon Oscillator if requested
        if include_oscillator:
            aroon_osc = aroon_up - aroon_down
            result_dict["AROON_OSC"] = aroon_osc

        result = pd.DataFrame(result_dict, index=high.index)

        # Store calculation details for analysis
        self._last_calculation = {
            "high": high,
            "low": low,
            "aroon_up": aroon_up,
            "aroon_down": aroon_down,
            "period": period,
        }

        if include_oscillator:
            self._last_calculation["aroon_osc"] = result_dict["AROON_OSC"]

        return result

    def validate_parameters(self) -> bool:
        """Validate Aroon parameters"""
        period = self.parameters.get("period", 14)
        include_oscillator = self.parameters.get("include_oscillator", True)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(include_oscillator, bool):
            raise IndicatorValidationError(
                f"include_oscillator must be boolean, got {include_oscillator}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Aroon metadata as dictionary for compatibility"""
        include_oscillator = self.parameters.get("include_oscillator", True)

        output_columns = ["AROON_UP", "AROON_DOWN"]
        if include_oscillator:
            output_columns.append("AROON_OSC")

        return {
            "name": "Aroon",
            "category": self.CATEGORY,
            "description": "Aroon Indicator - Identifies trend changes and measures trend strength",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
            "output_columns": output_columns,
        }

    def _get_required_columns(self) -> List[str]:
        """Aroon can use high/low or just close prices"""
        return ["high", "low"]  # Preferred, but can work with just close

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for Aroon calculation"""
        return self.parameters.get("period", 14)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 14
        if "include_oscillator" not in self.parameters:
            self.parameters["include_oscillator"] = True

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 14)

    @property
    def include_oscillator(self) -> bool:
        """Include oscillator for backward compatibility"""
        return self.parameters.get("include_oscillator", True)

    def get_trend_signal(self, aroon_up: float, aroon_down: float) -> str:
        """
        Get trend signal based on Aroon values

        Args:
            aroon_up: Current Aroon Up value
            aroon_down: Current Aroon Down value

        Returns:
            str: "bullish", "bearish", "consolidating", or "neutral"
        """
        # Strong signals
        if aroon_up > 80 and aroon_down < 20:
            return "bullish"
        elif aroon_down > 80 and aroon_up < 20:
            return "bearish"
        # Moderate signals
        elif aroon_up > aroon_down and aroon_up > 50:
            return "bullish"
        elif aroon_down > aroon_up and aroon_down > 50:
            return "bearish"
        # Consolidation
        elif (
            abs(aroon_up - aroon_down) < 20
            and (40 <= aroon_up <= 60)
            and (40 <= aroon_down <= 60)
        ):
            return "consolidating"
        else:
            return "neutral"

    def get_oscillator_signal(self, aroon_osc: float) -> str:
        """
        Get signal based on Aroon Oscillator value

        Args:
            aroon_osc: Current Aroon Oscillator value

        Returns:
            str: "bullish", "bearish", or "neutral"
        """
        if aroon_osc > 20:
            return "bullish"
        elif aroon_osc < -20:
            return "bearish"
        else:
            return "neutral"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return AroonIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLC data
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate realistic OHLC data with trends
    close_prices = [base_price]
    for i in range(n_points - 1):
        # Add some trending behavior
        trend = np.sin(i / 20) * 0.1
        change = np.random.randn() * 0.5 + trend
        close_prices.append(close_prices[-1] + change)

    # Create OHLC from close prices
    data = pd.DataFrame(
        {
            "close": close_prices,
            "high": [c + abs(np.random.randn() * 0.3) for c in close_prices],
            "low": [c - abs(np.random.randn() * 0.3) for c in close_prices],
        }
    )

    # Calculate Aroon
    aroon = AroonIndicator()
    result = aroon.calculate(data)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Price chart
    ax1.plot(data["close"].values, label="Close Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # Aroon Up/Down chart
    ax2.plot(result["AROON_UP"].values, label="Aroon Up", color="green", linewidth=2)
    ax2.plot(result["AROON_DOWN"].values, label="Aroon Down", color="red", linewidth=2)
    ax2.axhline(y=80, color="orange", linestyle="--", alpha=0.7, label="Strong level")
    ax2.axhline(y=20, color="orange", linestyle="--", alpha=0.7)
    ax2.axhline(y=50, color="gray", linestyle="-", alpha=0.3)
    ax2.set_title("Aroon Up/Down")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)

    # Aroon Oscillator chart
    if "AROON_OSC" in result.columns:
        ax3.plot(
            result["AROON_OSC"].values,
            label="Aroon Oscillator",
            color="purple",
            linewidth=2,
        )
        ax3.axhline(
            y=20, color="green", linestyle="--", alpha=0.7, label="Bullish threshold"
        )
        ax3.axhline(
            y=-20, color="red", linestyle="--", alpha=0.7, label="Bearish threshold"
        )
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax3.set_title("Aroon Oscillator")
        ax3.set_ylim(-100, 100)
        ax3.legend()
        ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Aroon calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"Aroon parameters: {aroon.parameters}")
    print(f"Current Aroon Up: {result['AROON_UP'].iloc[-1]:.2f}")
    print(f"Current Aroon Down: {result['AROON_DOWN'].iloc[-1]:.2f}")
    if "AROON_OSC" in result.columns:
        print(f"Current Aroon Oscillator: {result['AROON_OSC'].iloc[-1]:.2f}")
    print(
        f"Trend signal: {aroon.get_trend_signal(result['AROON_UP'].iloc[-1], result['AROON_DOWN'].iloc[-1])}"
    )
    if "AROON_OSC" in result.columns:
        print(
            f"Oscillator signal: {aroon.get_oscillator_signal(result['AROON_OSC'].iloc[-1])}"
        )
