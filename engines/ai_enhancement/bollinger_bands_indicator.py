"""
Bollinger Bands Indicator - Fixed Version

Bollinger Bands are volatility-based envelopes around a moving average.
"""

import os
import sys
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
    IndicatorMetadata,
)


class BollingerBandsIndicator(StandardIndicatorInterface):
    """Bollinger Bands Indicator for volatility analysis"""

    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, period: int = 20, multiplier: float = 2.0, **kwargs):
        super().__init__(period=period, multiplier=multiplier, **kwargs)

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        self.validate_input_data(data)
        
        period = self.parameters.get("period", 20)
        multiplier = self.parameters.get("multiplier", 2.0)
        
        if isinstance(data, pd.Series):
            price_series = data
        else:
            price_series = data["close"]
        
        # Calculate middle band (SMA)
        middle_band = price_series.rolling(window=period).mean()
        
        # Calculate standard deviation
        std_dev = price_series.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (multiplier * std_dev)
        lower_band = middle_band - (multiplier * std_dev)
        
        # Calculate %B
        band_width = upper_band - lower_band
        percent_b = pd.Series(index=price_series.index, dtype=float)
        valid_mask = band_width > 0
        percent_b[valid_mask] = (price_series[valid_mask] - lower_band[valid_mask]) / band_width[valid_mask]
        
        result = pd.DataFrame({
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'percent_b': percent_b
        }, index=price_series.index)
        
        self._last_calculation = {"result": result}
        return result

    def validate_parameters(self) -> bool:
        period = self.parameters.get("period", 20)
        multiplier = self.parameters.get("multiplier", 2.0)
        if not isinstance(period, int) or period < 2:
            raise IndicatorValidationError(f"Invalid period: {period}")
        if not isinstance(multiplier, (int, float)) or multiplier <= 0:
            raise IndicatorValidationError(f"Invalid multiplier: {multiplier}")
        return True

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="BollingerBands", category=self.CATEGORY, description="Bollinger Bands",
            parameters=self.parameters, input_requirements=self._get_required_columns(),
            output_type="DataFrame", version=self.VERSION, author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )

    def _get_required_columns(self) -> List[str]:
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        return self.parameters.get("period", 20)

    def _setup_defaults(self):
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "multiplier" not in self.parameters:
            self.parameters["multiplier"] = 2.0

    @property
    def period(self) -> int:
        return self.parameters.get("period", 20)
    
    @property
    def minimum_periods(self) -> int:
        return self.parameters.get("period", 20)

    def get_config(self) -> Dict[str, Any]:
        return {"name": "BollingerBands", "category": self.CATEGORY, "parameters": self.parameters, "version": self.VERSION}


def get_indicator_class():
    return BollingerBandsIndicator