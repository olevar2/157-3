"""
Stochastic Oscillator - Momentum Oscillator Indicator

The Stochastic Oscillator is a momentum indicator that compares the current
closing price to the price range over a specified period. It generates values
between 0 and 100, providing overbought/oversold signals.

Formula:
%K = ((Close - LowestLow) / (HighestHigh - LowestLow)) * 100
%D = SMA(%K, d_period)

Where:
- LowestLow = Lowest low over k_period
- HighestHigh = Highest high over k_period
- %K can be smoothed using SMA
- %D is the signal line (SMA of %K)

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base_indicator import IndicatorValidationError, StandardIndicatorInterface


class StochasticIndicator(StandardIndicatorInterface):
    """
    Stochastic Oscillator implementation following Platform3 StandardIndicatorInterface

    The Stochastic Oscillator is a momentum indicator that shows the location of the
    current close relative to the high-low range over a set number of periods.

    Returns DataFrame with %K and %D columns (both range 0-100)
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "momentum"  # Stochastic is primarily a momentum indicator
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 1,
        overbought: float = 80.0,
        oversold: float = 20.0,
        **kwargs,
    ):
        """
        Initialize Stochastic Oscillator indicator

        Args:
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D smoothing (default: 3)
            smooth_k: Smoothing period for %K (default: 1, no smoothing)
            overbought: Overbought threshold (default: 80.0)
            oversold: Oversold threshold (default: 20.0)
        """
        super().__init__(
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
            overbought=overbought,
            oversold=oversold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator indicator

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            pd.DataFrame: DataFrame with '%K' and '%D' columns, values 0-100
        """
        # Handle input data - Stochastic requires HLC data
        if isinstance(data, pd.Series):
            raise IndicatorValidationError(
                "Stochastic Oscillator requires high, low, close data"
            )
        elif isinstance(data, pd.DataFrame):
            if not all(col in data.columns for col in ["high", "low", "close"]):
                raise IndicatorValidationError(
                    "DataFrame must contain 'high', 'low', 'close' columns"
                )
            self.validate_input_data(data)
        else:
            raise IndicatorValidationError(
                "Data must be DataFrame with high, low, close columns"
            )

        k_period = self.parameters.get("k_period", 14)
        d_period = self.parameters.get("d_period", 3)
        smooth_k = self.parameters.get("smooth_k", 1)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate %K (Fast Stochastic)
        # Find the lowest low and highest high over k_period
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()

        # Calculate raw %K
        # %K = ((Close - LowestLow) / (HighestHigh - LowestLow)) * 100
        denominator = highest_high - lowest_low
        # Handle case where high == low (no price movement)
        denominator = denominator.replace(0, np.nan)

        raw_k = ((close - lowest_low) / denominator) * 100

        # Apply smoothing to %K if requested
        if smooth_k > 1:
            percent_k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
        else:
            percent_k = raw_k

        # Calculate %D (Slow Stochastic - SMA of %K)
        percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()

        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result["%K"] = percent_k
        result["%D"] = percent_d

        # Store calculation details for analysis
        self._last_calculation = {
            "lowest_low": lowest_low,
            "highest_high": highest_high,
            "raw_k": raw_k,
            "percent_k": percent_k,
            "percent_d": percent_d,
            "k_period": k_period,
            "d_period": d_period,
            "smooth_k": smooth_k,
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate Stochastic Oscillator parameters"""
        k_period = self.parameters.get("k_period", 14)
        d_period = self.parameters.get("d_period", 3)
        smooth_k = self.parameters.get("smooth_k", 1)
        overbought = self.parameters.get("overbought", 80.0)
        oversold = self.parameters.get("oversold", 20.0)

        if not isinstance(k_period, int) or k_period < 1:
            raise IndicatorValidationError(
                f"k_period must be positive integer, got {k_period}"
            )

        if k_period > 1000:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"k_period too large, maximum 1000, got {k_period}"
            )

        if not isinstance(d_period, int) or d_period < 1:
            raise IndicatorValidationError(
                f"d_period must be positive integer, got {d_period}"
            )

        if d_period > 100:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"d_period too large, maximum 100, got {d_period}"
            )

        if not isinstance(smooth_k, int) or smooth_k < 1:
            raise IndicatorValidationError(
                f"smooth_k must be positive integer, got {smooth_k}"
            )

        if smooth_k > 100:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"smooth_k too large, maximum 100, got {smooth_k}"
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

        if oversold < 0 or overbought > 100:
            raise IndicatorValidationError(
                f"thresholds must be between 0-100, got overbought={overbought}, oversold={oversold}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Stochastic Oscillator metadata as dictionary for compatibility"""
        return {
            "name": "Stochastic",
            "category": self.CATEGORY,
            "description": "Stochastic Oscillator - Momentum oscillator comparing current close to high-low range",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "output_columns": ["%K", "%D"],
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
            "bounded_range": [0, 100],
        }

    def _get_required_columns(self) -> List[str]:
        """Stochastic requires high, low, close data"""
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for Stochastic calculation"""
        k_period = self.parameters.get("k_period", 14)
        d_period = self.parameters.get("d_period", 3)
        smooth_k = self.parameters.get("smooth_k", 1)

        # Need enough data for %K calculation, smoothing, and %D calculation
        return k_period + max(smooth_k - 1, 0) + d_period - 1

    def get_interpretation(self, k_value: float, d_value: float) -> Dict[str, Any]:
        """
        Get interpretation of Stochastic values

        Args:
            k_value: %K value (0-100)
            d_value: %D value (0-100)

        Returns:
            Dict with interpretation details
        """
        overbought = self.parameters.get("overbought", 80.0)
        oversold = self.parameters.get("oversold", 20.0)

        # Basic overbought/oversold analysis
        k_signal = "neutral"
        d_signal = "neutral"

        if k_value >= overbought:
            k_signal = "overbought"
        elif k_value <= oversold:
            k_signal = "oversold"

        if d_value >= overbought:
            d_signal = "overbought"
        elif d_value <= oversold:
            d_signal = "oversold"

        # Crossover analysis
        crossover = None
        if k_value > d_value:
            crossover = "bullish" if abs(k_value - d_value) < 5 else None
        elif k_value < d_value:
            crossover = "bearish" if abs(k_value - d_value) < 5 else None

        return {
            "k_signal": k_signal,
            "d_signal": d_signal,
            "crossover": crossover,
            "momentum": (
                "strong_bullish"
                if k_value > 80 and d_value > 80
                else (
                    "strong_bearish"
                    if k_value < 20 and d_value < 20
                    else (
                        "bullish"
                        if k_value > 50 and d_value > 50
                        else "bearish" if k_value < 50 and d_value < 50 else "neutral"
                    )
                )
            ),
        }

    def get_signals(self, result: pd.DataFrame) -> dict:
        """
        Generate trading signals based on stochastic values

        Args:
            result: DataFrame containing %K and %D columns

        Returns:
            Dictionary containing signal information
        """
        if result.empty or "%K" not in result.columns or "%D" not in result.columns:
            return {"signal": "neutral", "strength": 0.0}

        # Get the latest values (excluding NaN)
        k_values = result["%K"].dropna()
        d_values = result["%D"].dropna()

        if len(k_values) == 0 or len(d_values) == 0:
            return {"signal": "neutral", "strength": 0.0}

        latest_k = k_values.iloc[-1]
        latest_d = d_values.iloc[-1]

        # Generate signals based on overbought/oversold levels
        overbought = self.parameters.get("overbought", 80.0)
        oversold = self.parameters.get("oversold", 20.0)

        signal = "neutral"
        strength = 0.0

        if latest_k >= overbought and latest_d >= overbought:
            signal = "sell"
            strength = 0.8
        elif latest_k <= oversold and latest_d <= oversold:
            signal = "buy"
            strength = 0.8
        elif latest_k >= overbought or latest_d >= overbought:
            signal = "weak_sell"
            strength = 0.5
        elif latest_k <= oversold or latest_d <= oversold:
            signal = "weak_buy"
            strength = 0.5

        # Check for crossovers if we have enough data
        if len(k_values) > 1 and len(d_values) > 1:
            prev_k = k_values.iloc[-2]
            prev_d = d_values.iloc[-2]

            # Bullish crossover: %K crosses above %D
            if prev_k <= prev_d and latest_k > latest_d:
                signal = "buy"
                strength = max(strength, 0.7)
            # Bearish crossover: %K crosses below %D
            elif prev_k >= prev_d and latest_k < latest_d:
                signal = "sell"
                strength = max(strength, 0.7)

        return {
            "signal": signal,
            "strength": strength,
            "k_value": latest_k,
            "d_value": latest_d,
            "overbought_level": overbought,
            "oversold_level": oversold,
        }


# Export function for backward compatibility and registry discovery
def stochastic_oscillator(
    data: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator (export function)

    Args:
        data: DataFrame with high, low, close columns
        k_period: Period for %K calculation
        d_period: Period for %D calculation
        smooth_k: Smoothing period for %K

    Returns:
        DataFrame with %K and %D columns
    """
    indicator = StochasticIndicator(
        k_period=k_period, d_period=d_period, smooth_k=smooth_k, **kwargs
    )
    return indicator.calculate(data)


# Alias for common usage patterns
stochastic = stochastic_oscillator
stoch = stochastic_oscillator
