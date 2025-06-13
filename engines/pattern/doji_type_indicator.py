"""
Doji Type Indicator

Doji is a candlestick pattern where open and close prices are nearly equal,
indicating market indecision and potential reversal points.

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


class DojiTypeIndicator(StandardIndicatorInterface):
    """
    Doji Type Indicator

    Detects various types of Doji candlestick patterns indicating market indecision.
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
        Initialize Doji Type indicator

        Args:
            period: Period for pattern detection (default: 20)
        """
        super().__init__(
            period=period,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate Doji pattern signals

        Args:
            data: DataFrame with 'open', 'high', 'low', 'close' columns

        Returns:
            pd.Series: Pattern signals (1=dragonfly, -1=gravestone, 0.5=standard, 0=no doji)
        """
        # Validate input data
        self.validate_data(data)
        
        if len(data) < 1:
            return pd.Series(np.zeros(len(data)), index=data.index if hasattr(data, 'index') else range(len(data)))

        opens, highs, lows, closes = extract_ohlc_data(data)
        signals = np.zeros(len(data))

        for i in range(len(data)):
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]

            # Doji criteria: very small body relative to total range
            if total_range > 0 and body_size <= total_range * 0.1:
                upper_shadow = highs[i] - max(opens[i], closes[i])
                lower_shadow = min(opens[i], closes[i]) - lows[i]

                # Different doji types
                if upper_shadow > total_range * 0.6:  # Dragonfly doji
                    signals[i] = 1
                elif lower_shadow > total_range * 0.6:  # Gravestone doji
                    signals[i] = -1
                else:  # Standard doji
                    signals[i] = 0.5

        return pd.Series(signals, index=data.index if hasattr(data, 'index') else range(len(data)))

    def get_metadata(self) -> Dict[str, Any]:
        """Return Doji Type metadata as dictionary for compatibility"""
        return {
            "name": "DojiType",
            "category": self.CATEGORY,
            "description": "Doji candlestick pattern - indicates market indecision and potential reversal",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Doji Type requires OHLC data"""
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


def get_doji_type_indicator(**params) -> DojiTypeIndicator:
    """
    Factory function to create Doji Type indicator
    
    Args:
        **params: Indicator parameters
        
    Returns:
        DojiTypeIndicator: Configured indicator instance
    """
    return DojiTypeIndicator(**params)


# Export for registry discovery
__all__ = ['DojiTypeIndicator', 'get_doji_type_indicator']