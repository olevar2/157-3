"""
Williams %R Indicator
==================== None

Williams %R is a momentum indicator that measures overbought and oversold levels.
It oscillates between 0 and -100, with readings above -20 indicating overbought
conditions and readings below -80 indicating oversold conditions.

Formula: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

Where:
- Highest High = Maximum high over the period
- Lowest Low = Minimum low over the period
- Close = Current closing price

Author: Platform3 AI System
Created: 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
import logging

from ..base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
)

logger = logging.getLogger(__name__)


class WilliamsRIndicator(StandardIndicatorInterface):
    """
    Williams %R momentum indicator implementation.

    Provides trading-grade Williams %R calculations with comprehensive
    validation and error handling.
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(self, period: int = 14, **kwargs):
        """
        Initialize Williams %R indicator.

        Args:
            period: Lookback period for calculation (default: 14)
            **kwargs: Additional parameters
        """
        self.period = period
        super().__init__(period=period, **kwargs)

    def validate_parameters(self) -> bool:
        """
        Validate Williams %R parameters.

        Returns:
            bool: True if parameters are valid

        Raises:
            IndicatorValidationError: If parameters are invalid
        """
        if not isinstance(self.period, int):
            raise IndicatorValidationError("period must be an integer")

        if self.period <= 0:
            raise IndicatorValidationError("period must be positive")

        if self.period > 1000:
            raise IndicatorValidationError("period too large (max 1000)")

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """
        Return Williams %R metadata.

        Returns:
            IndicatorMetadata: Complete indicator specification
        """
        return IndicatorMetadata(
            name="Williams %R",
            category=self.CATEGORY,
            description="Momentum oscillator measuring overbought/oversold conditions",
            parameters={"period": self.period},
            input_requirements=["high", "low", "close"],
            output_type="Series",
            version=self.VERSION,
            author=self.AUTHOR,
            trading_grade=True,
            performance_tier="fast",
            min_data_points=self.period,
            max_lookback_period=self.period,
        )

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Williams %R values.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            pd.Series: Williams %R values (0 to -100)

        Raises:
            IndicatorValidationError: If input data is invalid
        """
        # Validate input data
        self.validate_input_data(data)

        # Check required columns
        required_cols = ["high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise IndicatorValidationError(f"Missing required columns: {missing_cols}")

        # Check minimum data points
        if len(data) < self.period:
            raise IndicatorValidationError(
                f"Insufficient data: need at least {self.period} periods, got {len(data)}"
            )

        # Extract price arrays
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        # Validate data types and values
        self._validate_price_data(high, low, close)

        # Calculate Williams %R
        williams_r = self._calculate_williams_r(high, low, close)

        # Create result series with same index as input
        result = pd.Series(williams_r, index=data.index, name="Williams_R")

        # Store calculation metadata
        self._last_calculation = {
            "input_length": len(data),
            "valid_outputs": np.sum(~np.isnan(williams_r)),
            "calculation_period": self.period,
        }

        return result

    def _validate_price_data(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> None:
        """
        Validate price data arrays.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array

        Raises:
            IndicatorValidationError: If price data is invalid
        """
        # Check for non-finite values
        if not np.all(np.isfinite(high[np.isfinite(high)])):
            raise IndicatorValidationError("High prices contain invalid values")
        if not np.all(np.isfinite(low[np.isfinite(low)])):
            raise IndicatorValidationError("Low prices contain invalid values")
        if not np.all(np.isfinite(close[np.isfinite(close)])):
            raise IndicatorValidationError("Close prices contain invalid values")

        # Check for negative prices (warn only)
        if np.any(high[np.isfinite(high)] <= 0):
            logger.warning("High prices contain non-positive values")
        if np.any(low[np.isfinite(low)] <= 0):
            logger.warning("Low prices contain non-positive values")
        if np.any(close[np.isfinite(close)] <= 0):
            logger.warning("Close prices contain non-positive values")

        # Check high >= low relationship
        valid_indices = np.isfinite(high) & np.isfinite(low)
        if np.any(high[valid_indices] < low[valid_indices]):
            raise IndicatorValidationError("High prices must be >= low prices")

    def _calculate_williams_r(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> np.ndarray:
        """
        Core Williams %R calculation with trading-grade precision.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array

        Returns:
            np.ndarray: Williams %R values
        """
        length = len(close)
        williams_r = np.full(length, np.nan, dtype=np.float64)

        # Calculate rolling min/max with proper handling
        for i in range(self.period - 1, length):
            start_idx = i - self.period + 1
            end_idx = i + 1

            # Get period data
            period_high = high[start_idx:end_idx]
            period_low = low[start_idx:end_idx]
            current_close = close[i]

            # Skip if any values are NaN/infinite
            if (
                not np.isfinite(current_close)
                or not np.all(np.isfinite(period_high))
                or not np.all(np.isfinite(period_low))
            ):
                continue

            # Calculate period high and low
            highest_high = np.max(period_high)
            lowest_low = np.min(period_low)

            # Avoid division by zero
            range_value = highest_high - lowest_low
            if range_value == 0:
                # When no price movement, use neutral value
                williams_r[i] = -50.0
            else:
                # Standard Williams %R formula
                williams_r[i] = ((highest_high - current_close) / range_value) * -100.0

                # Ensure result is within expected bounds
                williams_r[i] = np.clip(williams_r[i], -100.0, 0.0)

        return williams_r

    def get_overbought_level(self) -> float:
        """
        Get the standard overbought level for Williams %R.

        Returns:
            float: Overbought threshold (-20)
        """
        return -20.0

    def get_oversold_level(self) -> float:
        """
        Get the standard oversold level for Williams %R.

        Returns:
            float: Oversold threshold (-80)
        """
        return -80.0

    def generate_signals(self, williams_r: pd.Series) -> pd.Series:
        """
        Generate trading signals based on Williams %R values.

        Args:
            williams_r: Williams %R values

        Returns:
            pd.Series: Trading signals (1=buy, -1=sell, 0=hold)
        """
        signals = pd.Series(0, index=williams_r.index, name="Williams_R_Signals")

        overbought = self.get_overbought_level()
        oversold = self.get_oversold_level()

        # Generate signals based on level crossings
        for i in range(1, len(williams_r)):
            if np.isnan(williams_r.iloc[i]) or np.isnan(williams_r.iloc[i - 1]):
                continue

            # Buy signal: crossing above oversold level
            if williams_r.iloc[i - 1] <= oversold and williams_r.iloc[i] > oversold:
                signals.iloc[i] = 1

            # Sell signal: crossing below overbought level
            elif (
                williams_r.iloc[i - 1] >= overbought and williams_r.iloc[i] < overbought
            ):
                signals.iloc[i] = -1

        return signals

    def get_calculation_info(self) -> dict:
        """
        Get information about the last calculation.

        Returns:
            dict: Calculation metadata
        """
        return self._last_calculation or {}


def create_williams_r_indicator(period: int = 14, **kwargs) -> WilliamsRIndicator:
    """
    Factory function to create Williams %R indicator.

    Args:
        period: Lookback period (default: 14)
        **kwargs: Additional parameters

    Returns:
        WilliamsRIndicator: Configured indicator instance
    """
    return WilliamsRIndicator(period=period, **kwargs)
