"""
Directional Movement System Indicator - Fixed Version

ADX and Directional Indicators for trend strength analysis.
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


class DirectionalMovementSystemIndicator(StandardIndicatorInterface):
    """Directional Movement System (ADX, +DI, -DI)"""

    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, period: int = 14, **kwargs):
        super().__init__(period=period, **kwargs)

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Calculate ADX and DI indicators"""
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError("DM System requires OHLC data")

        period = self.parameters.get("period", 14)
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low
        
        # Simple +DM and -DM calculation
        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)
        
        up_move = high_diff
        down_move = low_diff
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)
        
        # Smooth using simple moving average
        smoothed_tr = true_range.rolling(window=period).mean()
        smoothed_plus_dm = plus_dm.rolling(window=period).mean()
        smoothed_minus_dm = minus_dm.rolling(window=period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        result = pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'dx': dx
        }, index=data.index)
        
        self._last_calculation = {"result": result}
        return result

    def validate_parameters(self) -> bool:
        period = self.parameters.get("period", 14)
        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(f"Invalid period: {period}")
        return True

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="DirectionalMovementSystem", category=self.CATEGORY,
            description="ADX and DI indicators", parameters=self.parameters,
            input_requirements=self._get_required_columns(), output_type="DataFrame",
            version=self.VERSION, author=self.AUTHOR,
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
        return {"name": "DirectionalMovementSystem", "category": self.CATEGORY, "parameters": self.parameters, "version": self.VERSION}


def get_indicator_class():
    return DirectionalMovementSystemIndicator