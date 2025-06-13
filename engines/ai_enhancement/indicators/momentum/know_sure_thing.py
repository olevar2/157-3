"""
Know Sure Thing (KST) Indicator

The Know Sure Thing indicator is a momentum oscillator that uses four different
rate of change periods to capture different cycles in the market.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from ..base_indicator import (
    IndicatorMetadata,
    IndicatorValidationError,
    StandardIndicatorInterface,
)

logger = logging.getLogger(__name__)


class KnowSureThingIndicator(StandardIndicatorInterface):
    """
    Know Sure Thing (KST) - Martin Pring's Momentum Indicator

    The KST combines four different Rate of Change (ROC) indicators with
    different timeframes to create a comprehensive momentum oscillator.

    Formula:
    ROC1 = Rate of Change over roc1_period
    ROC2 = Rate of Change over roc2_period
    ROC3 = Rate of Change over roc3_period
    ROC4 = Rate of Change over roc4_period

    KST = (SMA(ROC1, sma1_period) * weight1) + None
          (SMA(ROC2, sma2_period) * weight2) + None
          (SMA(ROC3, sma3_period) * weight3) + None
          (SMA(ROC4, sma4_period) * weight4)

    Signal Line = SMA(KST, signal_period)

    Default Parameters (Daily):
    - ROC periods: 10, 15, 20, 30
    - SMA periods: 10, 10, 10, 15
    - Weights: 1, 2, 3, 4
    - Signal period: 9
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(
        self,
        roc1_period: int = 10,
        roc2_period: int = 15,
        roc3_period: int = 20,
        roc4_period: int = 30,
        sma1_period: int = 10,
        sma2_period: int = 10,
        sma3_period: int = 10,
        sma4_period: int = 15,
        weight1: float = 1.0,
        weight2: float = 2.0,
        weight3: float = 3.0,
        weight4: float = 4.0,
        signal_period: int = 9,
        **kwargs
    ):
        """
        Initialize Know Sure Thing Indicator

        Args:
            roc1_period: First ROC period (default: 10)
            roc2_period: Second ROC period (default: 15)
            roc3_period: Third ROC period (default: 20)
            roc4_period: Fourth ROC period (default: 30)
            sma1_period: First SMA period (default: 10)
            sma2_period: Second SMA period (default: 10)
            sma3_period: Third SMA period (default: 10)
            sma4_period: Fourth SMA period (default: 15)
            weight1: First component weight (default: 1.0)
            weight2: Second component weight (default: 2.0)
            weight3: Third component weight (default: 3.0)
            weight4: Fourth component weight (default: 4.0)
            signal_period: Signal line SMA period (default: 9)
        """
        super().__init__(
            roc1_period=roc1_period,
            roc2_period=roc2_period,
            roc3_period=roc3_period,
            roc4_period=roc4_period,
            sma1_period=sma1_period,
            sma2_period=sma2_period,
            sma3_period=sma3_period,
            sma4_period=sma4_period,
            weight1=weight1,
            weight2=weight2,
            weight3=weight3,
            weight4=weight4,
            signal_period=signal_period,
            **kwargs
        )

    def validate_parameters(self) -> bool:
        """Validate KST parameters"""
        roc_periods = [
            self.parameters.get("roc1_period", 10),
            self.parameters.get("roc2_period", 15),
            self.parameters.get("roc3_period", 20),
            self.parameters.get("roc4_period", 30),
        ]

        sma_periods = [
            self.parameters.get("sma1_period", 10),
            self.parameters.get("sma2_period", 10),
            self.parameters.get("sma3_period", 10),
            self.parameters.get("sma4_period", 15),
        ]

        weights = [
            self.parameters.get("weight1", 1.0),
            self.parameters.get("weight2", 2.0),
            self.parameters.get("weight3", 3.0),
            self.parameters.get("weight4", 4.0),
        ]

        signal_period = self.parameters.get("signal_period", 9)

        # Validate all periods are positive
        for period in roc_periods + sma_periods + [signal_period]:
            if not isinstance(period, int) or period < 1:
                raise IndicatorValidationError("All periods must be positive integers")

        # Validate weights are positive
        for weight in weights:
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise IndicatorValidationError("All weights must be positive numbers")

        # ROC periods should generally be in ascending order
        if not all(
            roc_periods[i] <= roc_periods[i + 1] for i in range(len(roc_periods) - 1)
        ):
            logger.warning(
                "ROC periods are not in ascending order - this may affect indicator interpretation"
            )

        return True

    def _get_required_columns(self) -> List[str]:
        """Return required data columns"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Return minimum data points needed"""
        roc4_period = self.parameters.get("roc4_period", 30)
        sma4_period = self.parameters.get("sma4_period", 15)
        signal_period = self.parameters.get("signal_period", 9)

        # Need enough data for the longest ROC period plus its SMA plus the signal line
        return roc4_period + sma4_period + signal_period

    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change"""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Know Sure Thing Indicator

        Args:
            data: OHLCV DataFrame with required columns

        Returns:
            pd.Series: KST values
        """
        # Get parameters
        roc1_period = self.parameters.get("roc1_period", 10)
        roc2_period = self.parameters.get("roc2_period", 15)
        roc3_period = self.parameters.get("roc3_period", 20)
        roc4_period = self.parameters.get("roc4_period", 30)

        sma1_period = self.parameters.get("sma1_period", 10)
        sma2_period = self.parameters.get("sma2_period", 10)
        sma3_period = self.parameters.get("sma3_period", 10)
        sma4_period = self.parameters.get("sma4_period", 15)

        weight1 = self.parameters.get("weight1", 1.0)
        weight2 = self.parameters.get("weight2", 2.0)
        weight3 = self.parameters.get("weight3", 3.0)
        weight4 = self.parameters.get("weight4", 4.0)

        close = data["close"]

        # Calculate Rate of Change for each period
        roc1 = self._calculate_roc(close, roc1_period)
        roc2 = self._calculate_roc(close, roc2_period)
        roc3 = self._calculate_roc(close, roc3_period)
        roc4 = self._calculate_roc(close, roc4_period)

        # Apply smoothing to each ROC
        smooth_roc1 = roc1.rolling(window=sma1_period, min_periods=sma1_period).mean()
        smooth_roc2 = roc2.rolling(window=sma2_period, min_periods=sma2_period).mean()
        smooth_roc3 = roc3.rolling(window=sma3_period, min_periods=sma3_period).mean()
        smooth_roc4 = roc4.rolling(window=sma4_period, min_periods=sma4_period).mean()

        # Calculate weighted KST
        kst = (
            smooth_roc1 * weight1
            + smooth_roc2 * weight2
            + smooth_roc3 * weight3
            + smooth_roc4 * weight4
        )

        # Store calculation details for analysis
        self._last_calculation = {
            "roc1": roc1,
            "roc2": roc2,
            "roc3": roc3,
            "roc4": roc4,
            "smooth_roc1": smooth_roc1,
            "smooth_roc2": smooth_roc2,
            "smooth_roc3": smooth_roc3,
            "smooth_roc4": smooth_roc4,
            "kst": kst,
        }

        return kst.fillna(np.nan)

    def calculate_with_signal(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate KST with signal line

        Returns:
            Dictionary with 'kst' and 'signal' series
        """
        kst = self.calculate(data)
        signal_period = self.parameters.get("signal_period", 9)

        signal_line = kst.rolling(
            window=signal_period, min_periods=signal_period
        ).mean()

        return {"kst": kst, "signal": signal_line}

    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on KST

        Returns:
            DataFrame with signal columns
        """
        kst_data = self.calculate_with_signal(data)
        kst = kst_data["kst"]
        signal_line = kst_data["signal"]

        signals = pd.DataFrame(index=data.index)
        signals["kst"] = kst
        signals["signal_line"] = signal_line

        # KST vs Signal Line crossovers
        signals["bullish_crossover"] = (kst > signal_line) & (
            kst.shift(1) <= signal_line.shift(1)
        )
        signals["bearish_crossover"] = (kst < signal_line) & (
            kst.shift(1) >= signal_line.shift(1)
        )

        # Zero line crossovers
        signals["kst_above_zero"] = kst > 0
        signals["kst_below_zero"] = kst < 0
        signals["zero_line_bullish"] = (kst > 0) & (kst.shift(1) <= 0)
        signals["zero_line_bearish"] = (kst < 0) & (kst.shift(1) >= 0)

        # Divergence detection (simplified)
        price_direction = data["close"] > data["close"].shift(5)
        kst_direction = kst > kst.shift(5)
        signals["bullish_divergence"] = (~price_direction) & kst_direction
        signals["bearish_divergence"] = price_direction & (~kst_direction)

        # Overall signal
        signals["signal"] = 0
        signals.loc[
            signals["bullish_crossover"] | signals["zero_line_bullish"], "signal"
        ] = 1
        signals.loc[
            signals["bearish_crossover"] | signals["zero_line_bearish"], "signal"
        ] = -1

        return signals

    def get_metadata(self) -> IndicatorMetadata:
        """Return KST metadata"""
        return IndicatorMetadata(
            name="Know Sure Thing",
            category=self.CATEGORY,
            description="Multi-timeframe momentum oscillator using weighted Rate of Change indicators",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="series",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _setup_defaults(self):
        """Setup default parameters"""
        defaults = {
            "roc1_period": 10,
            "roc2_period": 15,
            "roc3_period": 20,
            "roc4_period": 30,
            "sma1_period": 10,
            "sma2_period": 10,
            "sma3_period": 10,
            "sma4_period": 15,
            "weight1": 1.0,
            "weight2": 2.0,
            "weight3": 3.0,
            "weight4": 4.0,
            "signal_period": 9,
        }

        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value
