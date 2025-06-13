"""
Chaikin Money Flow (CMF)

Chaikin Money Flow measures buying and selling pressure over a given period
by combining price location and volume.

Formula:
Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
Money Flow Volume = Money Flow Multiplier × Volume
CMF = Sum(Money Flow Volume, N) / Sum(Volume, N)

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
    IndicatorValidationError, StandardIndicatorInterface, IndicatorMetadata,
)

class ChaikinMoneyFlowSignal(StandardIndicatorInterface):
    """Chaikin Money Flow for money flow analysis"""
    
    CATEGORY: str = "volume"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, period: int = 20, **kwargs):
        super().__init__(period=period, **kwargs)

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError("CMF requires DataFrame with HLCV data")
            
        period = self.parameters.get("period", 20)
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        
        # Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Chaikin Money Flow
        cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # Generate signals
        signals = pd.Series(index=data.index, dtype=str)
        signals[:] = "neutral"
        signals[cmf > 0.1] = "accumulation"
        signals[cmf < -0.1] = "distribution"
        
        result = pd.DataFrame({
            'cmf': cmf,
            'mf_multiplier': mf_multiplier,
            'mf_volume': mf_volume,
            'signals': signals
        }, index=data.index)
        
        self._last_calculation = {"cmf": cmf, "signals": signals}
        return result

    def validate_parameters(self) -> bool:
        period = self.parameters.get("period", 20)
        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(f"period must be integer >= 1, got {period}")
        return True

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ChaikinMoneyFlow", category=self.CATEGORY,
            description="Chaikin Money Flow - Money flow indicator",
            parameters=self.parameters, input_requirements=self._get_required_columns(),
            output_type="DataFrame", version=self.VERSION, author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        return ["high", "low", "close", "volume"]

    def _get_minimum_data_points(self) -> int:
        return self.parameters.get("period", 20)

    def _setup_defaults(self):
        if "period" not in self.parameters:
            self.parameters["period"] = 20

    @property
    def period(self) -> int:
        return self.parameters.get("period", 20)

    @property
    def minimum_periods(self) -> int:
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        return {"name": "ChaikinMoneyFlow", "category": self.CATEGORY, 
                "parameters": self.parameters, "version": self.VERSION}

def get_indicator_class():
    return ChaikinMoneyFlowSignal