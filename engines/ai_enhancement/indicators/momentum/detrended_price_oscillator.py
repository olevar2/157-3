"""
Detrended Price Oscillator (DPO) Indicator

The Detrended Price Oscillator removes the trend from price to make it easier
to identify cycles and overbought/oversold conditions.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..base_indicator import (
    IndicatorMetadata,
    IndicatorValidationError,
    StandardIndicatorInterface,
)

logger = logging.getLogger(__name__)


class DetrendedPriceOscillatorIndicator(StandardIndicatorInterface):
    """
    Detrended Price Oscillator (DPO) - Cycle Analysis Indicator

    The DPO removes the trend from price by comparing the current price
    to a simple moving average from a previous period.

    Formula: DPO = Close - SMA(Close, period)[period/2 + 1 periods ago]

    The DPO is not a traditional oscillator as it's displaced in time
    to remove the trend component and highlight cycles.

    Parameters:
    ----------- None
    period : int, default=14
        The period for the simple moving average calculation
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(self, period: int = 14, **kwargs):
        """
        Initialize Detrended Price Oscillator

        Args:
            period: Period for SMA calculation (default: 14)
        """
        super().__init__(period=period, **kwargs)

    def validate_parameters(self) -> bool:
        """Validate DPO parameters"""
        period = self.parameters.get("period", 14)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError("period must be a positive integer")

        if period < 5:
            logger.warning("DPO period less than 5 may produce unreliable results")

        return True

    def _get_required_columns(self) -> List[str]:
        """Return required data columns"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Return minimum data points needed"""
        period = self.parameters.get("period", 14)
        displacement = period // 2 + 1
        return period + displacement

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Detrended Price Oscillator

        Args:
            data: OHLCV DataFrame with required columns

        Returns:
            pd.Series: DPO values"""
        # Validate input data
        self.validate_input_data(data)

        period = self.parameters.get("period", 14)
        close = data["close"]

        # Calculate displacement (look-back period)
        displacement = period // 2 + 1

        # Calculate Simple Moving Average
        sma = close.rolling(window=period, min_periods=period).mean()

        # Shift SMA back by displacement periods
        sma_displaced = sma.shift(displacement)

        # Calculate DPO
        dpo = close - sma_displaced

        # Store calculation details for analysis
        self._last_calculation = {
            "close": close,
            "sma": sma,
            "sma_displaced": sma_displaced,
            "dpo": dpo,
            "displacement": displacement,
        }

        return dpo.fillna(np.nan)

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on DPO

        Returns:
            DataFrame with signal columns
        """
        dpo = self.calculate(data)

        signals = pd.DataFrame(index=data.index)
        signals["dpo"] = dpo

        # Zero line crossover signals
        signals["bullish_crossover"] = (dpo > 0) & (dpo.shift(1) <= 0)
        signals["bearish_crossover"] = (dpo < 0) & (dpo.shift(1) >= 0)

        # Momentum signals
        signals["dpo_increasing"] = dpo > dpo.shift(1)
        signals["dpo_decreasing"] = dpo < dpo.shift(1)

        # Peak and trough identification for cycle analysis
        signals["local_peak"] = (dpo > dpo.shift(1)) & (dpo > dpo.shift(-1)) & (dpo > 0)
        signals["local_trough"] = (
            (dpo < dpo.shift(1)) & (dpo < dpo.shift(-1)) & (dpo < 0)
        )

        # Overall signal based on DPO position and momentum
        signals["signal"] = np.where(dpo > 0, 1, np.where(dpo < 0, -1, 0))

        # Enhance signals with momentum confirmation
        momentum_bullish = signals["dpo_increasing"] & (dpo > 0)
        momentum_bearish = signals["dpo_decreasing"] & (dpo < 0)

        signals.loc[momentum_bullish, "signal"] = 1
        signals.loc[momentum_bearish, "signal"] = -1

        return signals

    def get_cycle_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cycles using DPO

        Returns:
            Dictionary with cycle analysis results
        """
        dpo = self.calculate(data)
        signals = self.get_signals(data)

        # Find peaks and troughs
        peaks = signals[signals["local_peak"]].index
        troughs = signals[signals["local_trough"]].index

        # Calculate average cycle length
        if len(peaks) > 1:
            peak_distances = np.diff([data.index.get_loc(peak) for peak in peaks])
            avg_peak_cycle = (
                np.mean(peak_distances) if len(peak_distances) > 0 else np.nan
            )
        else:
            avg_peak_cycle = np.nan

        if len(troughs) > 1:
            trough_distances = np.diff(
                [data.index.get_loc(trough) for trough in troughs]
            )
            avg_trough_cycle = (
                np.mean(trough_distances) if len(trough_distances) > 0 else np.nan
            )
        else:
            avg_trough_cycle = np.nan

        # Current DPO characteristics
        current_dpo = dpo.iloc[-1] if len(dpo) > 0 else np.nan
        dpo_std = dpo.std()

        return {
            "current_dpo": current_dpo,
            "dpo_volatility": dpo_std,
            "num_peaks": len(peaks),
            "num_troughs": len(troughs),
            "avg_peak_cycle": avg_peak_cycle,
            "avg_trough_cycle": avg_trough_cycle,
            "recent_peaks": peaks[-3:].tolist() if len(peaks) >= 3 else peaks.tolist(),
            "recent_troughs": (
                troughs[-3:].tolist() if len(troughs) >= 3 else troughs.tolist()
            ),
            "cycle_strength": abs(current_dpo) / dpo_std if dpo_std > 0 else 0,
        }

    def get_metadata(self) -> IndicatorMetadata:
        """Return DPO metadata"""
        return IndicatorMetadata(
            name="Detrended Price Oscillator",
            category=self.CATEGORY,
            description="Removes trend from price to identify cycles and momentum patterns",
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
