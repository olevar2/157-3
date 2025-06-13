"""
Vortex Indicator

The Vortex Indicator is used to identify the start of a new trend or continuation
of an existing trend. It consists of two oscillators that capture positive and
negative trend movement.

Formula:
1. True Range (TR) = max(high - low, |high - prev_close|, |low - prev_close|)
2. Vortex Movement: VM+ = |high - prev_low|, VM- = |low - prev_high|
3. VI+ = sum(VM+, period) / sum(TR, period)
4. VI- = sum(VM-, period) / sum(TR, period)

Interpretation:
- VI+ > VI-: Bullish trend
- VI- > VI+: Bearish trend
- VI+ crossover above VI-: Potential bullish signal
- VI- crossover above VI+: Potential bearish signal

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

from indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
    IndicatorMetadata,
)


class VortexIndicator(StandardIndicatorInterface):
    """Vortex Indicator for trend identification"""

    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, period: int = 14, **kwargs):
        super().__init__(period=period, **kwargs)

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Calculate Vortex Indicator"""
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError("Vortex Indicator requires OHLC data")

        period = self.parameters.get("period", 14)
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Vortex Movement
        vm_plus = abs(high - low.shift(1))
        vm_minus = abs(low - high.shift(1))
        
        # Calculate VI+ and VI-
        sum_tr = true_range.rolling(window=period).sum()
        sum_vm_plus = vm_plus.rolling(window=period).sum()
        sum_vm_minus = vm_minus.rolling(window=period).sum()
        
        vi_plus = sum_vm_plus / sum_tr
        vi_minus = sum_vm_minus / sum_tr
        
        result = pd.DataFrame({
            'vi_plus': vi_plus,
            'vi_minus': vi_minus,
            'vi_diff': vi_plus - vi_minus
        }, index=data.index)
        
        self._last_calculation = {"result": result, "period": period}
        return result

    def validate_parameters(self) -> bool:
        period = self.parameters.get("period", 14)
        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(f"Invalid period: {period}")
        return True

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Vortex", category=self.CATEGORY, description="Vortex Indicator for trend analysis",
            parameters=self.parameters, input_requirements=self._get_required_columns(),
            output_type="DataFrame", version=self.VERSION, author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )

    def _get_required_columns(self) -> List[str]:
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        return self.parameters.get("period", 14) + 1

    def _setup_defaults(self):
        if "period" not in self.parameters:
            self.parameters["period"] = 14

    @property
    def period(self) -> int:
        return self.parameters.get("period", 14)
    
    @property
    def minimum_periods(self) -> int:
        return self.parameters.get("period", 14) + 1

    def get_config(self) -> Dict[str, Any]:
        return {"name": "Vortex", "category": self.CATEGORY, "parameters": self.parameters, "version": self.VERSION}


def get_indicator_class():
    return VortexIndicator