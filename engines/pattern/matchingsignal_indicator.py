"""
MatchingSignal Indicator

MatchingSignal candlestick/chart pattern indicator for technical analysis.

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



class MatchingSignalIndicator(StandardIndicatorInterface):
    """
    MatchingSignal Indicator

    Pattern recognition indicator for matchingsignal patterns.
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
        Initialize MatchingSignal indicator

        Args:
            period: Period for pattern detection (default: 20)
        """
        super().__init__(
            period=period,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        calculate MatchingSignal pattern signals

        Args:
            data: DataFrame with OHLC columns or relevant data

        Returns:
            pd.Series: Pattern signals
        """
        # Validate input data
        self.validate_data(data)
        
        # Original calculation logic
        original_result = self._original_calculate(data)
        
        # Convert to pandas Series
        if isinstance(original_result, np.ndarray):
            return pd.Series(original_result, index=data.index if hasattr(data, 'index') else range(len(data)))
        else:
            return pd.Series(original_result, index=data.index if hasattr(data, 'index') else range(len(data)))

    def _original_calculate(self, data):
        """Original calculation method from consolidated file"""
        """calculate Matching pattern signals"""
        if len(data) < 2:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(1, len(data)):
            # Simple matching close pattern
            if abs(closes[i] - closes[i - 1]) <= (
                closes[i] * 0.001
            ):  # Very close prices
                signals[i] = 0.5

        return signals



    def get_metadata(self) -> Dict[str, Any]:
        """Return MatchingSignal metadata as dictionary for compatibility"""
        return {
            "name": "MatchingSignal",
            "category": self.CATEGORY,
            "description": "MatchingSignal pattern recognition indicator",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """MatchingSignal requires OHLC data"""
        return ["open", "high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for pattern detection"""
        return self.parameters.get("period", 20)


    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        try:
            period = self.parameters.get("period", 20)
            if not isinstance(period, (int, float)) or period <= 0:
                return False
            return True
        except Exception:
            return False


def get_matchingsignal_indicator(**params) -> MatchingSignalIndicator:
    """
    Factory function to create MatchingSignal indicator
    
    Args:
        **params: Indicator parameters
        
    Returns:
        MatchingSignalIndicator: Configured indicator instance
    """
    return MatchingSignalIndicator(**params)


# Export for registry discovery
__all__ = ['MatchingSignalIndicator', 'get_matchingsignal_indicator']
