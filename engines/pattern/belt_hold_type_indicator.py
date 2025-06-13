"""
Belt Hold Type Indicator

Belt Hold is a single candlestick pattern that indicates potential reversal.
It's characterized by a long body with little to no shadow on one end.

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


class BeltHoldTypeIndicator(StandardIndicatorInterface):
    """
    Belt Hold Type Indicator

    Single candlestick pattern that indicates potential reversal points.
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
        Initialize Belt Hold Type indicator

        Args:
            period: Period for pattern detection (default: 20)
        """
        super().__init__(
            period=period,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate Belt Hold pattern signals

        Args:
            data: DataFrame with 'open', 'high', 'low', 'close' columns

        Returns:
            pd.Series: Pattern signals (1=bullish, -1=bearish, 0=no pattern)
        """
        # Validate input data
        self.validate_data(data)
        
        if len(data) < 1:
            return pd.Series(np.zeros(len(data)), index=data.index if hasattr(data, 'index') else range(len(data)))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]

            # Bullish belt hold (white/green belt hold)
            if (
                opens[i] == lows[i]  # Opens at low
                and closes[i] > opens[i]  # Bullish candle
                and body_size > (highs[i] - lows[i]) * 0.7  # Long body
                and upper_shadow < body_size * 0.1
            ):  # Minimal upper shadow
                signals[i] = 1

            # Bearish belt hold (black/red belt hold)
            elif (
                opens[i] == highs[i]  # Opens at high
                and closes[i] < opens[i]  # Bearish candle
                and body_size > (highs[i] - lows[i]) * 0.7  # Long body
                and lower_shadow < body_size * 0.1
            ):  # Minimal lower shadow
                signals[i] = -1

        return pd.Series(signals, index=data.index if hasattr(data, 'index') else range(len(data)))

    def get_metadata(self) -> Dict[str, Any]:
        """Return Belt Hold Type metadata as dictionary for compatibility"""
        return {
            "name": "BeltHoldType",
            "category": self.CATEGORY,
            "description": "Belt Hold candlestick pattern - single candle reversal signal",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Belt Hold Type requires OHLC data"""
        return ["open", "high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for pattern detection"""
        return 1


    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        try:
            period = self.parameters.get('period', 20)
            if not isinstance(period, (int, float)) or period <= 0:
                return False
            return True
        except Exception:
            return False


def get_belt_hold_type_indicator(**params) -> BeltHoldTypeIndicator:
    """
    Factory function to create Belt Hold Type indicator
    
    Args:
        **params: Indicator parameters
        
    Returns:
        BeltHoldTypeIndicator: Configured indicator instance
    """
    return BeltHoldTypeIndicator(**params)


# Export for registry discovery
__all__ = ['BeltHoldTypeIndicator', 'get_belt_hold_type_indicator']