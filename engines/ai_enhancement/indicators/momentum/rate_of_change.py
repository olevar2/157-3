"""
Rate of Change (ROC) Indicator
Trading-grade implementation for Platform3

The Rate of Change measures the percentage change between the current price
and the price n periods ago, providing momentum analysis.
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


class RateOfChangeIndicator(StandardIndicatorInterface):
    """
    Rate of Change (ROC) - Price Momentum Indicator

    Formula: ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100

    The ROC oscillates around zero:
    - Positive values indicate upward momentum
    - Negative values indicate downward momentum
    - Greater absolute values indicate stronger momentum
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(self, period: int = 12, **kwargs):
        """
        Initialize Rate of Change

        Args:
            period: Lookback period for comparison (default: 12)
        """
        super().__init__(period=period, **kwargs)

    def validate_parameters(self) -> bool:
        """Validate ROC parameters"""
        period = self.parameters.get("period", 12)

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
        """ROC requires close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        return self.parameters.get("period", 12) + 1

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Rate of Change

        Args:
            data: DataFrame with 'close' column

        Returns:
            pd.Series: ROC values with same index as input data
        """
        # Validate input data
        self.validate_input_data(data)

        period = self.parameters.get("period", 12)
        closes = data["close"]

        # Get price n periods ago
        price_n_ago = closes.shift(period)

        # Calculate ROC
        # Handle division by zero by replacing zero prices with NaN
        price_n_ago_safe = price_n_ago.replace(0, np.nan)

        roc = ((closes - price_n_ago) / price_n_ago_safe) * 100

        # Store calculation details for analysis
        self._last_calculation = {
            "current_price": closes,
            "price_n_ago": price_n_ago,
            "roc": roc,
        }

        return roc.fillna(np.nan)

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on ROC

        Returns:
            DataFrame with signal columns
        """
        roc = self.calculate(data)

        signals = pd.DataFrame(index=data.index)
        signals["roc"] = roc

        # Zero line crossovers
        signals["bullish_crossover"] = (roc > 0) & (roc.shift(1) <= 0)
        signals["bearish_crossover"] = (roc < 0) & (roc.shift(1) >= 0)

        # Momentum strength levels (adaptive based on recent volatility)
        roc_abs = roc.abs()
        strong_threshold = roc_abs.rolling(window=50, min_periods=10).quantile(0.75)

        signals["strong_bullish"] = roc > strong_threshold
        signals["strong_bearish"] = roc < -strong_threshold

        # Momentum acceleration/deceleration
        roc_change = roc.diff()
        signals["momentum_accelerating"] = (roc > 0) & (roc_change > 0)
        signals["momentum_decelerating"] = (roc > 0) & (roc_change < 0)
        signals["negative_momentum_accelerating"] = (roc < 0) & (roc_change < 0)
        signals["negative_momentum_decelerating"] = (roc < 0) & (roc_change > 0)

        # Overall signal
        signals["signal"] = np.where(
            roc > 2,
            1,  # Bullish above +2%
            np.where(roc < -2, -1, 0),  # Bearish below -2%
        )

        # Extreme momentum signals
        extreme_threshold = roc_abs.rolling(window=50, min_periods=10).quantile(0.9)
        signals["extreme_momentum"] = roc_abs > extreme_threshold

        return signals

    def get_momentum_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify momentum regime based on ROC patterns

        Returns:
            DataFrame with regime classification
        """
        roc = self.calculate(data)

        regime = pd.DataFrame(index=data.index)

        # Calculate rolling statistics
        roc_mean = roc.rolling(window=20, min_periods=5).mean()
        roc_std = roc.rolling(window=20, min_periods=5).std()

        # Regime classification
        regime["high_momentum"] = roc.abs() > (roc_mean.abs() + 2 * roc_std)
        regime["low_momentum"] = roc.abs() < (roc_mean.abs() - roc_std)
        regime["trending_up"] = (roc > 0) & (roc_mean > 0)
        regime["trending_down"] = (roc < 0) & (roc_mean < 0)
        regime["sideways"] = (roc.abs() < roc_std) & (roc_mean.abs() < roc_std)

        # Momentum persistence (how long momentum has been in same direction)
        roc_direction = np.sign(roc)
        direction_changes = (roc_direction != roc_direction.shift(1)).cumsum()
        regime["momentum_persistence"] = (
            direction_changes.groupby(direction_changes).cumcount() + 1
        )

        return regime

    def get_metadata(self) -> IndicatorMetadata:
        """Return ROC metadata"""
        return IndicatorMetadata(
            name="Rate of Change",
            category=self.CATEGORY,
            description=f"Percentage change over {self.parameters.get('period', 12)} periods, measures momentum strength",
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
            self.parameters["period"] = 12
