"""
Keltner Channel Indicator

Keltner Channels are volatility-based envelopes set above and below an exponential 
moving average. The channels use the Average True Range (ATR) to set channel distance.

Formula:
- Middle Line = EMA(close, period)
- Upper Channel = Middle Line + (multiplier * ATR)
- Lower Channel = Middle Line - (multiplier * ATR)

Author: Platform3 AI Framework
Created: 2025-06-10
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


class KeltnerChannelIndicator(StandardIndicatorInterface):
    """Keltner Channel Indicator for trend and volatility analysis"""

    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, period: int = 20, multiplier: float = 2.0, atr_period: int = 10, **kwargs):
        super().__init__(period=period, multiplier=multiplier, atr_period=atr_period, **kwargs)

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Calculate Keltner Channels"""
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError("Keltner Channels require OHLC data")

        period = self.parameters.get("period", 20)
        multiplier = self.parameters.get("multiplier", 2.0)
        atr_period = self.parameters.get("atr_period", 10)
        
        # Calculate EMA of close prices (middle line)
        close = data["close"]
        middle_line = close.ewm(span=period).mean()
        
        # Calculate ATR
        high = data["high"]
        low = data["low"]
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=atr_period).mean()
        
        # Calculate channels
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        result = pd.DataFrame({
            'upper_channel': upper_channel,
            'middle_line': middle_line,
            'lower_channel': lower_channel,
            'atr': atr
        }, index=data.index)
        
        self._last_calculation = {"result": result, "period": period, "multiplier": multiplier}
        return result

    def validate_parameters(self) -> bool:
        period = self.parameters.get("period", 20)
        multiplier = self.parameters.get("multiplier", 2.0)
        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(f"Invalid period: {period}")
        if not isinstance(multiplier, (int, float)) or multiplier <= 0:
            raise IndicatorValidationError(f"Invalid multiplier: {multiplier}")
        return True

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="KeltnerChannel", category=self.CATEGORY, description="Keltner Channels",
            parameters=self.parameters, input_requirements=self._get_required_columns(),
            output_type="DataFrame", version=self.VERSION, author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )

    def _get_required_columns(self) -> List[str]:
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        return max(self.parameters.get("period", 20), self.parameters.get("atr_period", 10))

    def _setup_defaults(self):
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "multiplier" not in self.parameters:
            self.parameters["multiplier"] = 2.0
        if "atr_period" not in self.parameters:
            self.parameters["atr_period"] = 10

    @property
    def period(self) -> int:
        return self.parameters.get("period", 20)
    
    @property
    def minimum_periods(self) -> int:
        return self.parameters.get("period", 20)

    def get_config(self) -> Dict[str, Any]:
        return {"name": "KeltnerChannel", "category": self.CATEGORY, "parameters": self.parameters, "version": self.VERSION}


def get_indicator_class():
    return KeltnerChannelIndicator