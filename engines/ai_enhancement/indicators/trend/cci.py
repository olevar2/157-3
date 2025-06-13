"""
Commodity Channel Index (CCI) Indicator

The CCI is a momentum-based oscillator used to help determine when an investment
vehicle has been overbought or oversold. It was originally developed for commodities
but is now used across various markets.

Formula:
1. Typical Price (TP) = (High + Low + Close) / 3
2. Simple Moving Average of TP over n periods
3. Mean Deviation = Average of |TP - SMA(TP)| over n periods
4. CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)

The constant 0.015 is used to ensure approximately 70-80% of CCI values fall
between -100 and +100.

Interpretation:
- CCI > +100: Overbought condition
- CCI < -100: Oversold condition
- CCI between -100 and +100: Normal trading range

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import os

# Import the base indicator interface
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class CCIIndicator(StandardIndicatorInterface):
    """
    Commodity Channel Index (CCI) Indicator

    A momentum-based oscillator that measures the current price level
    relative to an average price level over a given period of time.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "trend"  # CCI can be considered trend/momentum
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        constant: float = 0.015,
        overbought: float = 100.0,
        oversold: float = -100.0,
        **kwargs,
    ):
        """
        Initialize CCI indicator

        Args:
            period: Period for CCI calculation (default: 20)
            constant: CCI scaling constant (default: 0.015)
            overbought: Overbought threshold (default: 100.0)
            oversold: Oversold threshold (default: -100.0)
        """
        super().__init__(
            period=period,
            constant=constant,
            overbought=overbought,
            oversold=oversold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate CCI indicator

        Args:
            data: DataFrame with 'high', 'low', 'close' columns, or Series of prices

        Returns:
            pd.Series: CCI values with same index as input data
        """
        # Handle input data
        if isinstance(data, pd.Series):
            # If Series provided, use it as typical price
            typical_price = data
        elif isinstance(data, pd.DataFrame):
            if (
                "high" in data.columns
                and "low" in data.columns
                and "close" in data.columns
            ):
                # Calculate typical price from OHLC data
                typical_price = (data["high"] + data["low"] + data["close"]) / 3
                self.validate_input_data(data)
            elif "close" in data.columns:
                # If only close is available, use close as typical price
                typical_price = data["close"]
                # Create a temporary DataFrame for validation
                temp_data = pd.DataFrame(
                    {
                        "high": typical_price,
                        "low": typical_price,
                        "close": typical_price,
                    }
                )
                self.validate_input_data(temp_data)
            else:
                raise IndicatorValidationError(
                    "DataFrame must contain 'high', 'low', 'close' columns, or 'close' column"
                )
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        period = self.parameters.get("period", 20)
        constant = self.parameters.get(
            "constant", 0.015
        )  # Calculate Simple Moving Average of Typical Price
        sma_tp = typical_price.rolling(window=period, min_periods=period).mean()

        # Calculate Mean Deviation
        # For each position where we have SMA, calculate mean deviation using the same window
        mean_deviation = pd.Series(index=typical_price.index, dtype=float)

        # Calculate mean deviation for each position where we have enough data
        for i in range(period - 1, len(typical_price)):
            # Get the window of typical prices used for this SMA calculation
            tp_window = typical_price.iloc[i - period + 1 : i + 1]
            sma_value = sma_tp.iloc[i]

            if not pd.isna(sma_value):
                # Calculate mean deviation for this window
                deviation_window = abs(tp_window - sma_value)
                mean_deviation.iloc[i] = deviation_window.mean()
            else:
                mean_deviation.iloc[i] = np.nan

        # Calculate CCI
        # CCI = (TP - SMA(TP)) / (constant * Mean Deviation)
        cci = (typical_price - sma_tp) / (
            constant * mean_deviation
        )  # Store calculation details for analysis
        self._last_calculation = {
            "typical_price": typical_price,
            "sma_tp": sma_tp,
            "mean_deviation": mean_deviation,
            "cci": cci,
            "period": period,
            "constant": constant,
        }

        return pd.Series(cci, index=typical_price.index, name="CCI")

    def validate_parameters(self) -> bool:
        """Validate CCI parameters"""
        period = self.parameters.get("period", 20)
        constant = self.parameters.get("constant", 0.015)
        overbought = self.parameters.get("overbought", 100.0)
        oversold = self.parameters.get("oversold", -100.0)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(constant, (int, float)) or constant <= 0:
            raise IndicatorValidationError(
                f"constant must be positive number, got {constant}"
            )

        if not isinstance(overbought, (int, float)):
            raise IndicatorValidationError(
                f"overbought must be numeric, got {overbought}"
            )

        if not isinstance(oversold, (int, float)):
            raise IndicatorValidationError(f"oversold must be numeric, got {oversold}")

        if overbought <= oversold:
            raise IndicatorValidationError(
                f"overbought must be > oversold, got overbought={overbought}, oversold={oversold}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return CCI metadata as dictionary for compatibility"""
        return {
            "name": "CCI",
            "category": self.CATEGORY,
            "description": "Commodity Channel Index - Momentum oscillator measuring price deviation from average",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """CCI can use HLC or just close prices"""
        return ["high", "low", "close"]  # Preferred, but can work with just close

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for CCI calculation"""
        return self.parameters.get("period", 20)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "constant" not in self.parameters:
            self.parameters["constant"] = 0.015
        if "overbought" not in self.parameters:
            self.parameters["overbought"] = 100.0
        if "oversold" not in self.parameters:
            self.parameters["oversold"] = -100.0

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def constant(self) -> float:
        """Constant for backward compatibility"""
        return self.parameters.get("constant", 0.015)

    @property
    def overbought(self) -> float:
        """Overbought threshold for backward compatibility"""
        return self.parameters.get("overbought", 100.0)

    @property
    def oversold(self) -> float:
        """Oversold threshold for backward compatibility"""
        return self.parameters.get("oversold", -100.0)

    def get_signal(self, cci_value: float) -> str:
        """
        Get trading signal based on CCI value

        Args:
            cci_value: Current CCI value

        Returns:
            str: "overbought", "oversold", "bullish", "bearish", or "neutral"
        """
        overbought = self.overbought
        oversold = self.oversold

        if cci_value > overbought:
            return "overbought"
        elif cci_value < oversold:
            return "oversold"
        elif cci_value > 0:
            return "bullish"
        elif cci_value < 0:
            return "bearish"
        else:
            return "neutral"

    def get_extreme_signal(self, cci_value: float) -> str:
        """
        Get extreme signal based on CCI value (beyond normal thresholds)

        Args:
            cci_value: Current CCI value

        Returns:
            str: "extremely_overbought", "extremely_oversold", or "normal"
        """
        if cci_value > 200:
            return "extremely_overbought"
        elif cci_value < -200:
            return "extremely_oversold"
        else:
            return "normal"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return CCIIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLC data
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate realistic OHLC data with some volatility
    close_prices = [base_price]
    for i in range(n_points - 1):
        # Add some cyclical behavior for CCI to detect
        cycle = np.sin(i / 20) * 2
        change = np.random.randn() * 0.8 + cycle * 0.3
        close_prices.append(close_prices[-1] + change)

    # Create OHLC from close prices
    data = pd.DataFrame(
        {
            "close": close_prices,
            "high": [c + abs(np.random.randn() * 0.4) for c in close_prices],
            "low": [c - abs(np.random.randn() * 0.4) for c in close_prices],
        }
    )

    # Calculate CCI
    cci = CCIIndicator()
    result = cci.calculate(data)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Price chart
    ax1.plot(data["close"].values, label="Close Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # CCI chart
    ax2.plot(result.values, label="CCI", color="purple", linewidth=2)
    ax2.axhline(
        y=100, color="red", linestyle="--", alpha=0.7, label="Overbought (+100)"
    )
    ax2.axhline(
        y=-100, color="green", linestyle="--", alpha=0.7, label="Oversold (-100)"
    )
    ax2.axhline(
        y=200,
        color="red",
        linestyle=":",
        alpha=0.7,
        label="Extremely Overbought (+200)",
    )
    ax2.axhline(
        y=-200,
        color="green",
        linestyle=":",
        alpha=0.7,
        label="Extremely Oversold (-200)",
    )
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.set_title("CCI Indicator")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print("CCI calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"CCI parameters: {cci.parameters}")
    print(f"Current CCI: {result.iloc[-1]:.2f}")
    print(f"Signal: {cci.get_signal(result.iloc[-1])}")
    print(f"Extreme signal: {cci.get_extreme_signal(result.iloc[-1])}")

    # Statistics
    valid_cci = result.dropna()
    print("\nCCI Statistics:")
    print(f"Min: {valid_cci.min():.2f}")
    print(f"Max: {valid_cci.max():.2f}")
    print(f"Mean: {valid_cci.mean():.2f}")
    print(f"Std: {valid_cci.std():.2f}")
    print(f"Values > +100: {(valid_cci > 100).sum()}")
    print(f"Values < -100: {(valid_cci < -100).sum()}")
    print(
        f"% in normal range: {((valid_cci >= -100) & (valid_cci <= 100)).mean() * 100:.1f}%"
    )
