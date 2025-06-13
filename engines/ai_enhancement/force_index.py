"""
Force Index

The Force Index combines price change and volume to measure the force 
behind market moves, identifying trend strength and potential reversals.

Formula:
Force Index = Volume × (Close - Previous Close)

Interpretation:
- Positive values: Buying pressure (bullish force)
- Negative values: Selling pressure (bearish force)
- High absolute values: Strong force behind the move
- Force Index divergence: Potential trend reversal

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

class ForceIndex(StandardIndicatorInterface):
    """Force Index for measuring buying/selling pressure"""
    
    CATEGORY: str = "volume"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, period: int = 13, smoothing_period: int = 20, **kwargs):
        super().__init__(period=period, smoothing_period=smoothing_period, **kwargs)

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError("Force Index requires DataFrame with 'close' and 'volume'")
            
        period = self.parameters.get("period", 13)
        smoothing_period = self.parameters.get("smoothing_period", 20)
        
        close = data["close"]
        volume = data["volume"]
        
        # Calculate raw Force Index
        force_index = volume * (close - close.shift(1))
        
        # Calculate smoothed Force Index
        force_index_ema = force_index.ewm(span=period).mean()
        force_index_sma = force_index.rolling(window=smoothing_period).mean()
        
        # Generate signals
        signals = pd.Series(index=data.index, dtype=str)
        signals[:] = "neutral"
        
        # Crossover signals
        bullish_cross = (force_index_ema > 0) & (force_index_ema.shift(1) <= 0)
        bearish_cross = (force_index_ema < 0) & (force_index_ema.shift(1) >= 0)
        
        signals[bullish_cross] = "bullish_pressure"
        signals[bearish_cross] = "bearish_pressure"
        
        result = pd.DataFrame({
            'force_index': force_index,
            'force_index_ema': force_index_ema,
            'force_index_sma': force_index_sma,
            'signals': signals
        }, index=data.index)
        
        self._last_calculation = {"force_index": force_index, "signals": signals}
        return result

    def validate_parameters(self) -> bool:
        period = self.parameters.get("period", 13)
        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(f"period must be integer >= 1, got {period}")
        return True

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ForceIndex", category=self.CATEGORY,
            description="Force Index - Volume-price momentum indicator",
            parameters=self.parameters, input_requirements=self._get_required_columns(),
            output_type="DataFrame", version=self.VERSION, author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        return ["close", "volume"]

    def _get_minimum_data_points(self) -> int:
        return self.parameters.get("period", 13)

    def _setup_defaults(self):
        if "period" not in self.parameters:
            self.parameters["period"] = 13
        if "smoothing_period" not in self.parameters:
            self.parameters["smoothing_period"] = 20

    @property
    def period(self) -> int:
        return self.parameters.get("period", 13)

    @property
    def minimum_periods(self) -> int:
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        return {"name": "ForceIndex", "category": self.CATEGORY, 
                "parameters": self.parameters, "version": self.VERSION}

def get_indicator_class():
    return ForceIndex