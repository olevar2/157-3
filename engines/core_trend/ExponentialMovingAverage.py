"""
Exponential Moving Average Indicator - Auto-generated wrapper
This file provides individual indicator classes that wrap the MovingAverages functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from .SMA_EMA import MovingAverages, MAType

class ExponentialMovingAverage:
    """
    Exponential Moving Average Indicator
    Auto-generated wrapper for MovingAverages class
    """
    
    def __init__(self):
        self.ma_calculator = MovingAverages()
        self.indicator_type = MAType.EMA
    
    def calculate(self, data: Union[np.ndarray, pd.Series], period: int = 20) -> Dict[str, Any]:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price data (typically close prices)
            period: Period for calculation
            
        Returns:
            Dictionary containing Exponential Moving Average values and signals
        """
        try:
            if self.indicator_type == MAType.SMA:
                values = self.ma_calculator.calculate_sma(data, period)
            elif self.indicator_type == MAType.EMA:
                values = self.ma_calculator.calculate_ema(data, period)
            elif self.indicator_type == MAType.WMA:
                values = self.ma_calculator.calculate_wma(data, period)
            else:
                values = self.ma_calculator.calculate_sma(data, period)  # Default to SMA
            
            return {
                'values': values,
                'period': period,
                'type': self.indicator_type.value,
                'last_value': values[-1] if len(values) > 0 else None
            }
            
        except Exception as e:
            return {
                'values': np.array([]),
                'error': str(e),
                'period': period,
                'type': self.indicator_type.value
            }
