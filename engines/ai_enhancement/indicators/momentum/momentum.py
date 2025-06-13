"""
Momentum Indicator
Trading-grade implementation for Platform3

The Momentum indicator measures the raw difference between the current price
and the price n periods ago, providing absolute momentum analysis.
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


class MomentumIndicator(StandardIndicatorInterface):
    """
    Momentum - Simple Price Momentum Indicator

    Formula: Momentum = Current Price - Price n periods ago

    Unlike ROC which shows percentage change, Momentum shows absolute change:
    - Positive values indicate upward momentum
    - Negative values indicate downward momentum
    - Greater absolute values indicate stronger momentum in absolute terms
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(self, period: int = 10, **kwargs):
        """
        Initialize Momentum

        Args:
            period: Lookback period for comparison (default: 10)
        """
        super().__init__(period=period, **kwargs)

    def validate_parameters(self) -> bool:
        """Validate Momentum parameters"""
        period = self.parameters.get("period", 10)

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
        """Momentum requires close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        return self.parameters.get("period", 10) + 1

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Momentum

        Args:
            data: DataFrame with 'close' column

        Returns:
            pd.Series: Momentum values with same index as input data
        """
        # Validate input data
        self.validate_input_data(data)

        period = self.parameters.get("period", 10)
        closes = data["close"]

        # Get price n periods ago
        price_n_ago = closes.shift(period)

        # Calculate Momentum (absolute difference)
        momentum = closes - price_n_ago

        # Store calculation details for analysis
        self._last_calculation = {
            "current_price": closes,
            "price_n_ago": price_n_ago,
            "momentum": momentum,
        }

        return momentum.fillna(np.nan)

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Momentum

        Returns:
            DataFrame with signal columns
        """
        momentum = self.calculate(data)

        signals = pd.DataFrame(index=data.index)
        signals["momentum"] = momentum

        # Zero line crossovers
        signals["bullish_crossover"] = (momentum > 0) & (momentum.shift(1) <= 0)
        signals["bearish_crossover"] = (momentum < 0) & (momentum.shift(1) >= 0)

        # Momentum strength (adaptive thresholds based on price volatility)
        price_volatility = data["close"].rolling(window=20, min_periods=5).std()
        strong_threshold = price_volatility * 2  # 2 standard deviations

        signals["strong_bullish"] = momentum > strong_threshold
        signals["strong_bearish"] = momentum < -strong_threshold

        # Momentum acceleration/deceleration
        momentum_change = momentum.diff()
        signals["momentum_accelerating"] = (momentum > 0) & (momentum_change > 0)
        signals["momentum_decelerating"] = (momentum > 0) & (momentum_change < 0)
        signals["negative_momentum_accelerating"] = (momentum < 0) & (
            momentum_change < 0
        )
        signals["negative_momentum_decelerating"] = (momentum < 0) & (
            momentum_change > 0
        )

        # Momentum trend (direction over time)
        momentum_trend = momentum.rolling(window=5, min_periods=3).mean()
        signals["momentum_trend_up"] = momentum_trend > momentum_trend.shift(1)
        signals["momentum_trend_down"] = momentum_trend < momentum_trend.shift(1)

        # Overall signal (based on momentum relative to volatility)
        relative_momentum = momentum / (
            price_volatility + 1e-10
        )  # Avoid division by zero
        signals["signal"] = np.where(
            relative_momentum > 1,
            1,  # Strong bullish momentum
            np.where(relative_momentum < -1, -1, 0),  # Strong bearish momentum
        )

        return signals

    def get_momentum_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze momentum characteristics over time

        Returns:
            DataFrame with momentum profile metrics
        """
        momentum = self.calculate(data)

        profile = pd.DataFrame(index=data.index)

        # Rolling momentum statistics
        window = min(20, len(data) // 2)
        profile["momentum_mean"] = momentum.rolling(window=window, min_periods=5).mean()
        profile["momentum_std"] = momentum.rolling(window=window, min_periods=5).std()
        profile["momentum_max"] = momentum.rolling(window=window, min_periods=5).max()
        profile["momentum_min"] = momentum.rolling(window=window, min_periods=5).min()

        # Momentum percentile rank
        profile["momentum_percentile"] = momentum.rolling(
            window=window, min_periods=5
        ).rank(pct=True)

        # Momentum consistency (how often momentum is in same direction)
        momentum_direction = np.sign(momentum)
        consistency_window = 10
        profile["momentum_consistency"] = momentum_direction.rolling(
            window=consistency_window, min_periods=5
        ).apply(lambda x: (x == x.iloc[-1]).mean())

        # Momentum velocity (rate of momentum change)
        profile["momentum_velocity"] = momentum.diff()
        profile["momentum_acceleration"] = profile["momentum_velocity"].diff()

        return profile

    def get_divergence_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect momentum divergence with price

        Returns:
            DataFrame with divergence signals
        """
        momentum = self.calculate(data)
        closes = data["close"]

        signals = pd.DataFrame(index=data.index)

        # Calculate recent highs and lows
        lookback = min(20, len(data) // 2)

        # Price peaks and troughs
        price_peaks = closes.rolling(window=lookback, center=True).max() == closes
        price_troughs = closes.rolling(window=lookback, center=True).min() == closes

        # Momentum peaks and troughs
        momentum_peaks = (
            momentum.rolling(window=lookback, center=True).max() == momentum
        )
        momentum_troughs = (
            momentum.rolling(window=lookback, center=True).min() == momentum
        )

        # Bearish divergence: price makes higher high but momentum doesn't
        signals["bearish_divergence"] = price_peaks & ~momentum_peaks

        # Bullish divergence: price makes lower low but momentum doesn't
        signals["bullish_divergence"] = price_troughs & ~momentum_troughs

        # Hidden divergence (continuation patterns)
        signals["hidden_bullish_divergence"] = momentum_troughs & ~price_troughs
        signals["hidden_bearish_divergence"] = momentum_peaks & ~price_peaks

        return signals

    def get_metadata(self) -> IndicatorMetadata:
        """Return Momentum metadata"""
        return IndicatorMetadata(
            name="Momentum",
            category=self.CATEGORY,
            description=f"Absolute price change over {self.parameters.get('period', 10)} periods, measures momentum strength",
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
            self.parameters["period"] = 10
