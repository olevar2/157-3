"""
Engulfing Type Indicator

Engulfing patterns are two-candlestick reversal patterns where the second
candle completely engulfs the body of the previous candle.

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from engines.ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


def extract_ohlc_data(data):
    """Helper function to extract OHLC data from various input formats"""
    if hasattr(data, "iloc"):  # DataFrame
        opens = data["open"].values if "open" in data.columns else np.zeros(len(data))
        highs = data["high"].values if "high" in data.columns else np.zeros(len(data))
        lows = data["low"].values if "low" in data.columns else np.zeros(len(data))
        closes = (
            data["close"].values if "close" in data.columns else np.zeros(len(data))
        )
    else:  # Dict or array-like
        opens = np.array(data.get("open", np.zeros(len(data))))
        highs = np.array(data.get("high", np.zeros(len(data))))
        lows = np.array(data.get("low", np.zeros(len(data))))
        closes = np.array(data.get("close", np.zeros(len(data))))

    return opens, highs, lows, closes


class EngulfingTypeIndicator(StandardIndicatorInterface):
    """
    Engulfing Type Indicator

    Two-candlestick reversal pattern where the second candle engulfs the first.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "pattern"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        **kwargs,
    ):
        """
        Initialize Engulfing Type indicator

        Args:
            period: Period for pattern detection (default: 20)
        """
        super().__init__(
            period=period,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate Engulfing pattern signals

        Args:
            data: DataFrame with 'open', 'high', 'low', 'close' columns

        Returns:
            pd.Series: Pattern signals (1=bullish engulfing, -1=bearish engulfing, 0=no pattern)
        """
        # Validate input data
        self.validate_data(data)
        
        if len(data) < 2:
            return pd.Series(np.zeros(len(data)), index=data.index if hasattr(data, 'index') else range(len(data)))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # Previous candle real body
            prev_top = max(opens[i - 1], closes[i - 1])
            prev_bottom = min(opens[i - 1], closes[i - 1])

            # Current candle real body
            curr_top = max(opens[i], closes[i])
            curr_bottom = min(opens[i], closes[i])

            # Bullish engulfing
            if (
                closes[i - 1] < opens[i - 1]  # Previous candle bearish
                and closes[i] > opens[i]  # Current candle bullish
                and curr_bottom < prev_bottom  # Current engulfs previous
                and curr_top > prev_top
            ):
                signals[i] = 1

            # Bearish engulfing
            elif (
                closes[i - 1] > opens[i - 1]  # Previous candle bullish
                and closes[i] < opens[i]  # Current candle bearish
                and curr_bottom < prev_bottom  # Current engulfs previous
                and curr_top > prev_top
            ):
                signals[i] = -1

        return pd.Series(signals, index=data.index if hasattr(data, 'index') else range(len(data)))

    def get_metadata(self) -> Dict[str, Any]:
        """Return Engulfing Type metadata as dictionary for compatibility"""
        return {
            "name": "EngulfingType",
            "category": self.CATEGORY,
            "description": "Engulfing candlestick pattern - two-candle reversal signal",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Engulfing Type requires OHLC data"""
        return ["open", "high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for pattern detection"""
        return 2


    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        try:
            period = self.parameters.get('period', 20)
            if not isinstance(period, (int, float)) or period <= 0:
                return False
            return True
        except Exception:
            return False


def get_engulfing_type_indicator(**params) -> EngulfingTypeIndicator:
    """
    Factory function to create Engulfing Type indicator
    
    Args:
        **params: Indicator parameters
        
    Returns:
        EngulfingTypeIndicator: Configured indicator instance
    """
    return EngulfingTypeIndicator(**params)


# Export for registry discovery
__all__ = ['EngulfingTypeIndicator', 'get_engulfing_type_indicator']