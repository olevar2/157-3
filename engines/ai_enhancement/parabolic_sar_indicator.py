"""
Parabolic SAR (Stop and Reverse) Indicator

The Parabolic SAR is a trend-following indicator that provides entry and exit points.
It appears as dots above or below price candles to indicate trend direction.

Formula:
SAR(i) = SAR(i-1) + AF * (EP - SAR(i-1))
Where:
- AF = Acceleration Factor (starts at 0.02, increases by 0.02 each period, max 0.20)
- EP = Extreme Point (highest high in uptrend, lowest low in downtrend)

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


class ParabolicSARIndicator(StandardIndicatorInterface):
    """Parabolic SAR Indicator for trend reversal points"""

    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.20, **kwargs):
        super().__init__(af_start=af_start, af_increment=af_increment, af_max=af_max, **kwargs)

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Calculate Parabolic SAR"""
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError("Parabolic SAR requires OHLC data")

        af_start = self.parameters.get("af_start", 0.02)
        af_increment = self.parameters.get("af_increment", 0.02)
        af_max = self.parameters.get("af_max", 0.20)
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Initialize variables
        length = len(data)
        sar = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=int)  # 1 for up, -1 for down
        af = af_start
        ep = 0.0
        
        # First SAR value
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1
        ep = high.iloc[0]
        
        for i in range(1, length):
            prev_sar = sar.iloc[i-1]
            prev_trend = trend.iloc[i-1]
            
            # Calculate new SAR
            new_sar = prev_sar + af * (ep - prev_sar)
            
            # Determine trend direction
            if prev_trend == 1:  # Uptrend
                # Check for reversal
                if low.iloc[i] <= new_sar:
                    # Trend reversal to downtrend
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep  # Use previous EP as new SAR
                    ep = low.iloc[i]  # New EP is current low
                    af = af_start  # Reset AF
                else:
                    # Continue uptrend
                    trend.iloc[i] = 1
                    sar.iloc[i] = max(new_sar, min(low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1]))
                    
                    # Update EP and AF
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_increment, af_max)
            else:  # Downtrend
                # Check for reversal
                if high.iloc[i] >= new_sar:
                    # Trend reversal to uptrend
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep  # Use previous EP as new SAR
                    ep = high.iloc[i]  # New EP is current high
                    af = af_start  # Reset AF
                else:
                    # Continue downtrend
                    trend.iloc[i] = -1
                    sar.iloc[i] = min(new_sar, max(high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1]))
                    
                    # Update EP and AF
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_increment, af_max)
        
        self._last_calculation = {"sar": sar, "trend": trend, "af_start": af_start, "af_max": af_max}
        return sar

    def validate_parameters(self) -> bool:
        af_start = self.parameters.get("af_start", 0.02)
        af_increment = self.parameters.get("af_increment", 0.02)
        af_max = self.parameters.get("af_max", 0.20)
        
        if not isinstance(af_start, (int, float)) or af_start <= 0:
            raise IndicatorValidationError(f"Invalid af_start: {af_start}")
        if not isinstance(af_increment, (int, float)) or af_increment <= 0:
            raise IndicatorValidationError(f"Invalid af_increment: {af_increment}")
        if not isinstance(af_max, (int, float)) or af_max <= af_start:
            raise IndicatorValidationError(f"Invalid af_max: {af_max}")
        return True

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ParabolicSAR", category=self.CATEGORY, description="Parabolic SAR Stop and Reverse",
            parameters=self.parameters, input_requirements=self._get_required_columns(),
            output_type="Series", version=self.VERSION, author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )

    def _get_required_columns(self) -> List[str]:
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        return 3

    def _setup_defaults(self):
        if "af_start" not in self.parameters:
            self.parameters["af_start"] = 0.02
        if "af_increment" not in self.parameters:
            self.parameters["af_increment"] = 0.02
        if "af_max" not in self.parameters:
            self.parameters["af_max"] = 0.20

    @property
    def minimum_periods(self) -> int:
        return 3

    def get_config(self) -> Dict[str, Any]:
        return {"name": "ParabolicSAR", "category": self.CATEGORY, "parameters": self.parameters, "version": self.VERSION}


def get_indicator_class():
    return ParabolicSARIndicator