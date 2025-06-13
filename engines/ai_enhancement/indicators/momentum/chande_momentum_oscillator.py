"""
Chande Momentum Oscillator (CMO) Indicator
Trading-grade implementation for Platform3

The Chande Momentum Oscillator measures momentum using the sum of gains
versus losses over a period, normalized to a -100 to +100 scale.
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


class ChandeMomentumOscillatorIndicator(StandardIndicatorInterface):
    """
    Chande Momentum Oscillator (CMO) - Tushar Chande

    Formula: CMO = ((Sum of Gains - Sum of Losses) / (Sum of Gains + Sum of Losses)) * 100

    The CMO oscillates between -100 and +100:
    - Values above +50 indicate overbought conditions
    - Values below -50 indicate oversold conditions
    - Zero line crossovers can signal trend changes
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(self, period: int = 14, **kwargs):
        """
        Initialize Chande Momentum Oscillator

        Args:
            period: Lookback period for calculation (default: 14)
        """
        super().__init__(period=period, **kwargs)

    def validate_parameters(self) -> bool:
        """Validate CMO parameters"""
        period = self.parameters.get("period", 14)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        return True

    def _get_required_columns(self) -> List[str]:
        """CMO requires close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        return self.parameters.get("period", 14) + 1  # +1 for price changes

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Chande Momentum Oscillator

        Args:
            data: DataFrame with 'close' column

        Returns:
            pd.Series: CMO values with same index as input data
        """
        # Validate input data
        self.validate_input_data(data)

        period = self.parameters.get("period", 14)
        closes = data["close"]

        # Calculate price changes
        price_changes = closes.diff()

        # Separate gains and losses
        gains = price_changes.where(price_changes > 0, 0)
        losses = (-price_changes).where(price_changes < 0, 0)

        # Calculate rolling sums
        sum_gains = gains.rolling(window=period, min_periods=period).sum()
        sum_losses = losses.rolling(window=period, min_periods=period).sum()

        # Calculate CMO
        total_movement = sum_gains + sum_losses

        # Handle division by zero
        cmo = np.where(
            total_movement != 0, ((sum_gains - sum_losses) / total_movement) * 100, 0
        )

        # Convert to pandas Series with proper index
        cmo_series = pd.Series(cmo, index=data.index)

        # Store calculation details for analysis
        self._last_calculation = {
            "price_changes": price_changes,
            "gains": gains,
            "losses": losses,
            "sum_gains": sum_gains,
            "sum_losses": sum_losses,
            "cmo": cmo_series,
        }

        return cmo_series.fillna(np.nan)

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on CMO

        Returns:
            DataFrame with signal columns
        """
        cmo = self.calculate(data)

        signals = pd.DataFrame(index=data.index)
        signals["cmo"] = cmo

        # Overbought/Oversold levels
        signals["overbought"] = cmo > 50
        signals["oversold"] = cmo < -50

        # Zero line crossovers
        signals["bullish_crossover"] = (cmo > 0) & (cmo.shift(1) <= 0)
        signals["bearish_crossover"] = (cmo < 0) & (cmo.shift(1) >= 0)

        # Momentum direction
        signals["momentum_increasing"] = cmo > cmo.shift(1)
        signals["momentum_decreasing"] = cmo < cmo.shift(1)

        # Overall signal based on multiple criteria
        signals["signal"] = np.where(
            cmo > 20,
            1,  # Bullish above +20
            np.where(cmo < -20, -1, 0),  # Bearish below -20
        )

        # Extreme signals
        signals["extreme_bullish"] = cmo > 50
        signals["extreme_bearish"] = cmo < -50

        return signals

    def get_divergence_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potential divergence between price and CMO

        Returns:
            DataFrame with divergence signals
        """
        cmo = self.calculate(data)
        closes = data["close"]

        signals = pd.DataFrame(index=data.index)

        # Simple divergence detection (price makes new high/low but CMO doesn't)
        lookback = min(20, len(data) // 2)  # Adaptive lookback

        # Rolling max/min for price and CMO
        price_max = closes.rolling(window=lookback).max()
        price_min = closes.rolling(window=lookback).min()
        cmo_max = cmo.rolling(window=lookback).max()
        cmo_min = cmo.rolling(window=lookback).min()

        # Current values are at peaks/troughs
        at_price_high = closes == price_max
        at_price_low = closes == price_min
        at_cmo_high = cmo == cmo_max
        at_cmo_low = cmo == cmo_min

        # Bearish divergence: price makes new high but CMO doesn't
        signals["bearish_divergence"] = at_price_high & ~at_cmo_high

        # Bullish divergence: price makes new low but CMO doesn't
        signals["bullish_divergence"] = at_price_low & ~at_cmo_low

        return signals

    def get_metadata(self) -> IndicatorMetadata:
        """Return CMO metadata"""
        return IndicatorMetadata(
            name="Chande Momentum Oscillator",
            category=self.CATEGORY,
            description=f"Momentum oscillator using {self.parameters.get('period', 14)}-period gain/loss ratio, range -100 to +100",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="series",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _setup_defaults(self):
        """Setup default parameters"""
        if "period" not in self.parameters:
            self.parameters["period"] = 14
