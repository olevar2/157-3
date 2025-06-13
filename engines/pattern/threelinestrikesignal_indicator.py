"""
ThreeLineStrikeSignal Indicator

ThreeLineStrikeSignal candlestick/chart pattern indicator for technical analysis.

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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



class ThreeLineStrikeSignalIndicator(StandardIndicatorInterface):
    """
    ThreeLineStrikeSignal Indicator

    Pattern recognition indicator for threelinestrikesignal patterns.
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
        Initialize ThreeLineStrikeSignal indicator

        Args:
            period: Period for pattern detection (default: 20)
        """
        super().__init__(
            period=period,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        calculate ThreeLineStrikeSignal pattern signals

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
        """calculate Three Line Strike pattern signals"""
        if len(data) < 4:
            return np.zeros(len(data))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(3, len(data)):
            # Three consecutive candles same direction, fourth reverses strongly
            same_dir = (
                closes[i - 3] > opens[i - 3]
                and closes[i - 2] > opens[i - 2]
                and closes[i - 1] > opens[i - 1]
            ) or (
                closes[i - 3] < opens[i - 3]
                and closes[i - 2] < opens[i - 2]
                and closes[i - 1] < opens[i - 1]
            )

            if same_dir:
                # Fourth candle engulfs all three
                if opens[i] > max(
                    closes[i - 3], closes[i - 2], closes[i - 1]
                ) and closes[i] < min(opens[i - 3], opens[i - 2], opens[i - 1]):
                    signals[i] = -1  # Bearish strike
                elif opens[i] < min(
                    closes[i - 3], closes[i - 2], closes[i - 1]
                ) and closes[i] > max(opens[i - 3], opens[i - 2], opens[i - 1]):
                    signals[i] = 1  # Bullish strike

        return signals



    def get_metadata(self) -> Dict[str, Any]:
        """Return ThreeLineStrikeSignal metadata as dictionary for compatibility"""
        return {
            "name": "ThreeLineStrikeSignal",
            "category": self.CATEGORY,
            "description": "ThreeLineStrikeSignal pattern recognition indicator",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """ThreeLineStrikeSignal requires OHLC data"""
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


def get_threelinestrikesignal_indicator(**params) -> ThreeLineStrikeSignalIndicator:
    """
    Factory function to create ThreeLineStrikeSignal indicator
    
    Args:
        **params: Indicator parameters
        
    Returns:
        ThreeLineStrikeSignalIndicator: Configured indicator instance
    """
    return ThreeLineStrikeSignalIndicator(**params)


# Export for registry discovery
__all__ = ['ThreeLineStrikeSignalIndicator', 'get_threelinestrikesignal_indicator']
