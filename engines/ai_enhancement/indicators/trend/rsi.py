"""
Relative Strength Index (RSI) Indicator

The RSI is a momentum oscillator that measures the speed and change of price movements.
It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions.

Formula:
- RS = Average Gain / Average Loss
- RSI = 100 - (100 / (1 + RS))
- Average Gain = Sum of Gains over period / period
- Average Loss = Sum of Losses over period / period

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


class RSIIndicator(StandardIndicatorInterface):
    """
    Relative Strength Index (RSI) Indicator

    A momentum oscillator that measures the speed and change of price movements.
    RSI oscillates between 0 and 100 and is typically used to identify overbought
    or oversold conditions in a traded instrument.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        **kwargs,
    ):
        """
        Initialize RSI indicator

        Args:
            period: Period for RSI calculation (default: 14)
            overbought: Overbought threshold (default: 70.0)
            oversold: Oversold threshold (default: 30.0)
        """
        super().__init__(
            period=period,
            overbought=overbought,
            oversold=oversold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate RSI indicator

        Args:
            data: DataFrame with 'close' column or Series of close prices

        Returns:
            pd.Series: RSI values (0-100) with same index as input data
        """
        # Convert input to appropriate format for validation
        if isinstance(data, pd.Series):
            # Convert Series to DataFrame for validation
            df_data = pd.DataFrame({"close": data})
            self.validate_input_data(df_data)
            prices = data
        elif isinstance(data, pd.DataFrame):
            if "close" not in data.columns:
                raise IndicatorValidationError("DataFrame must contain 'close' column")
            self.validate_input_data(data)
            prices = data["close"]
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        period = self.parameters.get("period", 14)

        # Calculate price changes
        price_changes = prices.diff()

        # Separate gains and losses
        gains = price_changes.where(price_changes > 0, 0)
        losses = -price_changes.where(price_changes < 0, 0)

        # Calculate initial average gain and loss using simple moving average
        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()

        # Calculate RSI using Wilder's smoothing for subsequent values
        rs = np.zeros_like(prices.values, dtype=float)
        rsi = np.zeros_like(prices.values, dtype=float)

        # Fill with NaN initially
        rs[:] = np.nan
        rsi[:] = np.nan

        # Calculate first valid RSI using simple averages
        for i in range(period, len(prices)):
            if i == period:
                # First calculation uses simple average
                avg_g = avg_gain.iloc[i]
                avg_l = avg_loss.iloc[i]
            else:
                # Subsequent calculations use Wilder's smoothing
                # avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
                avg_g = (rs[i - 1] * avg_l * (period - 1) + gains.iloc[i]) / period
                avg_l = (avg_l * (period - 1) + losses.iloc[i]) / period

            if avg_l != 0:
                rs[i] = avg_g / avg_l
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100

        # Store calculation details for analysis
        self._last_calculation = {
            "price_changes": price_changes,
            "gains": gains,
            "losses": losses,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "rs": rs,
            "rsi": rsi,
        }

        return pd.Series(rsi, index=prices.index, name="RSI")

    def validate_parameters(self) -> bool:
        """Validate RSI parameters"""
        period = self.parameters.get("period", 14)
        overbought = self.parameters.get("overbought", 70.0)
        oversold = self.parameters.get("oversold", 30.0)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(overbought, (int, float)) or overbought <= oversold:
            raise IndicatorValidationError(
                f"overbought must be numeric and > oversold, got {overbought}"
            )

        if not isinstance(oversold, (int, float)) or oversold < 0:
            raise IndicatorValidationError(
                f"oversold must be numeric and >= 0, got {oversold}"
            )

        if overbought > 100 or oversold > 100:
            raise IndicatorValidationError(
                "overbought and oversold thresholds must be <= 100"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return RSI metadata as dictionary for compatibility"""
        return {
            "name": "RSI",
            "category": self.CATEGORY,
            "description": "Relative Strength Index - Momentum oscillator measuring speed and change of price movements",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """RSI requires close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for RSI calculation"""
        return self.parameters.get("period", 14) + 1

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 14
        if "overbought" not in self.parameters:
            self.parameters["overbought"] = 70.0
        if "oversold" not in self.parameters:
            self.parameters["oversold"] = 30.0

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 14)

    @property
    def overbought(self) -> float:
        """Overbought threshold for backward compatibility"""
        return self.parameters.get("overbought", 70.0)

    @property
    def oversold(self) -> float:
        """Oversold threshold for backward compatibility"""
        return self.parameters.get("oversold", 30.0)

    def get_signal(self, rsi_value: float) -> str:
        """
        Get trading signal based on RSI value

        Args:
            rsi_value: Current RSI value

        Returns:
            str: "overbought", "oversold", or "neutral"
        """
        if rsi_value >= self.overbought:
            return "overbought"
        elif rsi_value <= self.oversold:
            return "oversold"
        else:
            return "neutral"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return RSIIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample data
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5))

    # Calculate RSI
    rsi = RSIIndicator()
    result = rsi.calculate(prices)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Price chart
    ax1.plot(prices.values, label="Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # RSI chart
    ax2.plot(result.values, label="RSI", color="purple")
    ax2.axhline(y=70, color="red", linestyle="--", alpha=0.7, label="Overbought")
    ax2.axhline(y=30, color="green", linestyle="--", alpha=0.7, label="Oversold")
    ax2.axhline(y=50, color="black", linestyle="-", alpha=0.3)
    ax2.set_title("RSI Indicator")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"RSI calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"RSI parameters: {rsi.parameters}")
    print(f"Current RSI: {result.iloc[-1]:.2f}")
    print(f"Signal: {rsi.get_signal(result.iloc[-1])}")
