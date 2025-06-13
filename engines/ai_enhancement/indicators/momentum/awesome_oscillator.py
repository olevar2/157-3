"""
Awesome Oscillator (AO) Indicator
Trading-grade implementation for Platform3

The Awesome Oscillator measures momentum using the difference between
a 5-period and 34-period simple moving average of the median price.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import logging

from ..base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
)

logger = logging.getLogger(__name__)


class AwesomeOscillatorIndicator(StandardIndicatorInterface):
    """
    Awesome Oscillator (AO) - Bill Williams Indicator

    Formula: AO = SMA(median_price, 5) - SMA(median_price, 34)
    Where median_price = (high + low) / 2

    This indicator measures momentum and can help identify trend changes.
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(self, fast_period: int = 5, slow_period: int = 34, **kwargs):
        """
        Initialize Awesome Oscillator

        Args:
            fast_period: Fast SMA period (default: 5)
            slow_period: Slow SMA period (default: 34)
        """
        super().__init__(fast_period=fast_period, slow_period=slow_period, **kwargs)

    def validate_parameters(self) -> bool:
        """Validate AO parameters"""
        fast_period = self.parameters.get("fast_period", 5)
        slow_period = self.parameters.get("slow_period", 34)

        if not isinstance(fast_period, int) or fast_period < 1:
            raise IndicatorValidationError(
                f"fast_period must be positive integer, got {fast_period}"
            )

        if not isinstance(slow_period, int) or slow_period < 1:
            raise IndicatorValidationError(
                f"slow_period must be positive integer, got {slow_period}"
            )

        if fast_period >= slow_period:
            raise IndicatorValidationError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )

        return True

    def _get_required_columns(self) -> List[str]:
        """AO requires high and low prices"""
        return ["high", "low"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        return self.parameters.get("slow_period", 34)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Awesome Oscillator

        Args:
            data: DataFrame with 'high' and 'low' columns

        Returns:
            pd.Series: AO values with same index as input data
        """
        # Validate input data
        self.validate_input_data(data)

        fast_period = self.parameters.get("fast_period", 5)
        slow_period = self.parameters.get("slow_period", 34)

        # Calculate median price (typical price for AO)
        median_price = (data["high"] + data["low"]) / 2

        # Calculate SMAs
        fast_sma = median_price.rolling(
            window=fast_period, min_periods=fast_period
        ).mean()
        slow_sma = median_price.rolling(
            window=slow_period, min_periods=slow_period
        ).mean()

        # Calculate AO
        ao = fast_sma - slow_sma

        # Store calculation details for analysis
        self._last_calculation = {
            "median_price": median_price,
            "fast_sma": fast_sma,
            "slow_sma": slow_sma,
            "ao": ao,
        }

        return ao.fillna(np.nan)

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on AO

        Returns:
            DataFrame with signal columns
        """
        ao = self.calculate(data)

        signals = pd.DataFrame(index=data.index)
        signals["ao"] = ao

        # Zero line crossover signals
        signals["bullish_crossover"] = (ao > 0) & (ao.shift(1) <= 0)
        signals["bearish_crossover"] = (ao < 0) & (ao.shift(1) >= 0)

        # Twin peaks (saucer) signal - simplified version
        signals["ao_increasing"] = ao > ao.shift(1)
        signals["ao_decreasing"] = ao < ao.shift(1)

        # Overall signal
        signals["signal"] = np.where(ao > 0, 1, np.where(ao < 0, -1, 0))

        return signals

    def get_metadata(self) -> IndicatorMetadata:
        """Return AO metadata"""
        return IndicatorMetadata(
            name="Awesome Oscillator",
            category=self.CATEGORY,
            description="Measures momentum using difference of 5-period and 34-period SMA of median price",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="series",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _setup_defaults(self):
        """Setup default parameters"""
        if "fast_period" not in self.parameters:
            self.parameters["fast_period"] = 5
        if "slow_period" not in self.parameters:
            self.parameters["slow_period"] = 34
