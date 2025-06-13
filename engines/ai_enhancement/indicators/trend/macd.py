"""
Moving Average Convergence Divergence (MACD) Indicator

The MACD is one of the most popular trend-following momentum indicators that shows
the relationship between two moving averages of a security's price.

Formula:
- MACD Line = EMA(close, fast) - EMA(close, slow)
- Signal Line = EMA(MACD Line, signal)
- Histogram = MACD Line - Signal Line

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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
)
from typing import List


@dataclass
class MACDResult:
    """MACD calculation result containing all three components"""

    macd_line: np.ndarray
    signal_line: np.ndarray
    histogram: np.ndarray
    timestamps: Optional[np.ndarray] = None


class MACDIndicator(StandardIndicatorInterface):
    """
    Moving Average Convergence Divergence (MACD) Indicator

    A trend-following momentum indicator that shows the relationship between
    two moving averages of a security's price."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **kwargs,
    ):
        """
        Initialize MACD indicator

        Args:
            fast_period: Period for fast EMA (default: 12)
            slow_period: Period for slow EMA (default: 26)
            signal_period: Period for signal line EMA (default: 9)
        """
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            **kwargs,
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicator

        Args:
            data: Price data (close prices). Can be DataFrame, Series, or numpy array

        Returns:
            MACDResult: Object containing MACD line, signal line, and histogram
        """
        # Convert input to pandas Series for easier handling
        if isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                prices = data["close"]
                timestamps = data.index if hasattr(data, "index") else None
            else:
                raise ValueError("DataFrame must contain 'close' column")
        elif isinstance(data, pd.Series):
            prices = data
            timestamps = data.index if hasattr(data, "index") else None
        elif isinstance(data, np.ndarray):
            prices = pd.Series(data)
            timestamps = None
        else:
            raise ValueError(
                "Data must be DataFrame, Series, or numpy array"
            )  # Get parameters
        fast_period = self.parameters.get("fast_period", 12)
        slow_period = self.parameters.get("slow_period", 26)
        signal_period = self.parameters.get("signal_period", 9)

        # Validate data
        if len(prices) < slow_period + signal_period:
            raise ValueError(
                f"Insufficient data. Need at least {slow_period + signal_period} points"
            )

        # Remove NaN values
        prices = prices.dropna()
        if len(prices) == 0:
            raise ValueError("No valid price data after removing NaN values")

        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, fast_period)
        slow_ema = self._calculate_ema(prices, slow_period)

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD line)
        signal_line = self._calculate_ema(macd_line, signal_period)

        # Calculate histogram
        histogram = macd_line - signal_line

        return MACDResult(
            macd_line=macd_line.values,
            signal_line=signal_line.values,
            histogram=histogram.values,
            timestamps=timestamps,
        )

    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average using pandas ewm method

        Args:
            data: Price series
            period: EMA period

        Returns:
            EMA series
        """  # Use pandas ewm with span parameter for standard EMA calculation
        return data.ewm(span=period, adjust=False).mean()

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return {
            "fast_period": self.parameters.get("fast_period", 12),
            "slow_period": self.parameters.get("slow_period", 26),
            "signal_period": self.parameters.get("signal_period", 9),
        }

    def set_parameters(self, **kwargs) -> None:
        """Update parameters"""
        if "fast_period" in kwargs:
            self.parameters["fast_period"] = kwargs["fast_period"]
        if "slow_period" in kwargs:
            self.parameters["slow_period"] = kwargs["slow_period"]
        if "signal_period" in kwargs:
            self.parameters["signal_period"] = kwargs["signal_period"]

        # Validate new parameters
        fast_period = self.parameters.get("fast_period", 12)
        slow_period = self.parameters.get("slow_period", 26)
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")

    def validate_data(self, data: Any) -> bool:
        """Validate input data format"""
        try:
            if isinstance(data, pd.DataFrame):
                return "close" in data.columns and len(data) > 0
            elif isinstance(data, (pd.Series, np.ndarray)):
                return len(data) > 0
            return False
        except:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Return MACD metadata as dictionary for compatibility"""
        return {
            "name": "MACD",
            "category": "trend",
            "description": f"Moving Average Convergence Divergence with periods {self.parameters.get('fast_period', 12)}, {self.parameters.get('slow_period', 26)}, {self.parameters.get('signal_period', 9)}",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "dataframe",
            "output_names": ["macd_line", "signal_line", "histogram"],
            "version": "1.0.0",
            "author": "Platform3",
            "min_data_points": self._get_minimum_data_points(),
        }

    def validate_parameters(self) -> bool:
        """Validate MACD parameters"""
        fast_period = self.parameters.get("fast_period", 12)
        slow_period = self.parameters.get("slow_period", 26)
        signal_period = self.parameters.get("signal_period", 9)

        if not isinstance(fast_period, int) or fast_period < 1:
            raise ValueError(f"fast_period must be positive integer, got {fast_period}")

        if not isinstance(slow_period, int) or slow_period < 1:
            raise ValueError(f"slow_period must be positive integer, got {slow_period}")

        if not isinstance(signal_period, int) or signal_period < 1:
            raise ValueError(
                f"signal_period must be positive integer, got {signal_period}"
            )

        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )

        if slow_period > 1000:  # Reasonable upper limit
            raise ValueError(f"slow_period too large, maximum 1000, got {slow_period}")

        return True

    def _get_required_columns(self) -> List[str]:
        """MACD requires close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        slow_period = self.parameters.get("slow_period", 26)
        signal_period = self.parameters.get("signal_period", 9)
        return slow_period + signal_period

    def _setup_defaults(self):
        """Setup default parameters"""
        if "fast_period" not in self.parameters:
            self.parameters["fast_period"] = 12
        if "slow_period" not in self.parameters:
            self.parameters["slow_period"] = 26
        if "signal_period" not in self.parameters:
            self.parameters["signal_period"] = 9

    @property
    def fast_period(self) -> int:
        """Fast period for backward compatibility"""
        return self.parameters.get("fast_period", 12)

    @property
    def slow_period(self) -> int:
        """Slow period for backward compatibility"""
        return self.parameters.get("slow_period", 26)

    @property
    def signal_period(self) -> int:
        """Signal period for backward compatibility"""
        return self.parameters.get("signal_period", 9)


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return MACDIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample data
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5))

    # Calculate MACD
    macd = MACDIndicator()
    result = macd.calculate(prices)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Price chart
    ax1.plot(prices.values, label="Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # MACD chart
    ax2.plot(result.macd_line, label="MACD Line", color="blue")
    ax2.plot(result.signal_line, label="Signal Line", color="red")
    ax2.bar(
        range(len(result.histogram)), result.histogram, label="Histogram", alpha=0.3
    )
    ax2.set_title("MACD Indicator")
    ax2.legend()
    ax2.grid(True)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"MACD calculation completed successfully!")
    print(f"Data points: {len(result.macd_line)}")
    print(f"MACD parameters: {macd.get_parameters()}")
