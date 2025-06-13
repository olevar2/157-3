"""
Chaikin Volatility

Chaikin Volatility measures the volatility based on the spread between 
high and low prices over a given period.

Formula:
Chaikin Volatility = ((EMA(High-Low) - EMA(High-Low, N periods ago)) / EMA(High-Low, N periods ago)) * 100

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

from indicators.base_indicator import (
    IndicatorValidationError, StandardIndicatorInterface, IndicatorMetadata,
)

class ChaikinVolatility(StandardIndicatorInterface):
    """Chaikin Volatility for volatility measurement"""
    
    CATEGORY: str = "volatility"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, period: int = 10, roc_period: int = 10, **kwargs):
        super().__init__(period=period, roc_period=roc_period, **kwargs)

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError("Chaikin Volatility requires DataFrame with 'high' and 'low'")
            
        period = self.parameters.get("period", 10)
        roc_period = self.parameters.get("roc_period", 10)
        
        high = data["high"]
        low = data["low"]
        
        # Calculate High-Low spread
        hl_spread = high - low
        
        # Calculate EMA of High-Low spread
        hl_ema = hl_spread.ewm(span=period).mean()
        
        # Calculate Chaikin Volatility (Rate of Change of EMA)
        chaikin_volatility = ((hl_ema - hl_ema.shift(roc_period)) / hl_ema.shift(roc_period)) * 100
        
        # Generate signals
        signals = pd.Series(index=data.index, dtype=str)
        signals[:] = "normal_volatility"
        
        volatility_threshold = chaikin_volatility.std()
        signals[chaikin_volatility > volatility_threshold] = "increasing_volatility"
        signals[chaikin_volatility < -volatility_threshold] = "decreasing_volatility"
        
        result = pd.DataFrame({
            'chaikin_volatility': chaikin_volatility,
            'hl_spread': hl_spread,
            'hl_ema': hl_ema,
            'signals': signals
        }, index=data.index)
        
        self._last_calculation = {"chaikin_volatility": chaikin_volatility, "signals": signals}
        return result

    def validate_parameters(self) -> bool:
        period = self.parameters.get("period", 10)
        roc_period = self.parameters.get("roc_period", 10)
        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(f"period must be integer >= 1, got {period}")
        if not isinstance(roc_period, int) or roc_period < 1:
            raise IndicatorValidationError(f"roc_period must be integer >= 1, got {roc_period}")
        return True

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ChaikinVolatility", category=self.CATEGORY,
            description="Chaikin Volatility - Volatility measurement indicator",
            parameters=self.parameters, input_requirements=self._get_required_columns(),
            output_type="DataFrame", version=self.VERSION, author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        return ["high", "low"]

    def _get_minimum_data_points(self) -> int:
        return max(self.parameters.get("period", 10), self.parameters.get("roc_period", 10))

    def _setup_defaults(self):
        if "period" not in self.parameters:
            self.parameters["period"] = 10
        if "roc_period" not in self.parameters:
            self.parameters["roc_period"] = 10

    @property
    def period(self) -> int:
        return self.parameters.get("period", 10)

    @property
    def minimum_periods(self) -> int:
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        return {"name": "ChaikinVolatility", "category": self.CATEGORY, 
                "parameters": self.parameters, "version": self.VERSION}

def get_indicator_class():
    return ChaikinVolatility