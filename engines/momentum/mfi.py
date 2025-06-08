#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Money Flow Index - Enhanced Trading Engine
Platform3 Phase 3 - Enhanced with Framework Integration
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame

class MoneyFlowIndex(IndicatorBase):
    """Enhanced Money Flow Index indicator with Platform3 framework integration"""
    
    def __init__(self, period: int = 14):
        """Initialize the Money Flow Index indicator"""
        super().__init__()
        self.period = period
        self.logger.info(f"{self.__class__.__name__} initialized with period={period}")

    def calculate(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> IndicatorResult:
        """
        Calculate Money Flow Index values
        
        Args:
            data: OHLCV market data as DataFrame or list of dicts
            
        Returns:
            IndicatorResult with MFI values and signals
        """
        try:
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            if len(data) < self.period + 1:
                return IndicatorResult(
                    timestamp=datetime.now(),
                    indicator_name="Money Flow Index",
                    indicator_type=IndicatorType.VOLUME,
                    timeframe=TimeFrame.D1,
                    value=[],
                    signal=[],
                    raw_data={"error": f"Insufficient data for MFI calculation. Need at least {self.period+1} periods."}
                )
                
            # Calculate typical price
            data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
            
            # Calculate raw money flow
            data['money_flow'] = data['typical_price'] * data['volume']
            
            # Calculate positive and negative money flow
            data['price_change'] = data['typical_price'].diff()
            data['positive_flow'] = np.where(data['price_change'] > 0, data['money_flow'], 0)
            data['negative_flow'] = np.where(data['price_change'] < 0, data['money_flow'], 0)
            
            # Calculate money flow ratio and MFI
            positive_flow_sum = data['positive_flow'].rolling(window=self.period).sum()
            negative_flow_sum = data['negative_flow'].rolling(window=self.period).sum()
            
            mfi_values = []
            signals = []
            
            for i in range(self.period, len(data)):
                if negative_flow_sum[i] == 0:
                    mfi = 100.0
                else:
                    mf_ratio = positive_flow_sum[i] / negative_flow_sum[i]
                    mfi = 100.0 - (100.0 / (1.0 + mf_ratio))
                
                mfi_values.append({
                    'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else str(data.index[i]),
                    'value': mfi
                })
                
                # Generate signals
                signal_type = "neutral"
                if mfi > 80:
                    signal_type = "overbought"
                elif mfi < 20:
                    signal_type = "oversold"
                
                signals.append({
                    'timestamp': data.index[i] if hasattr(data.index[i], 'timestamp') else str(data.index[i]),
                    'type': signal_type,
                    'strength': abs((mfi - 50) / 50)  # 0 to 1 strength based on distance from neutral
                })
            
            return IndicatorResult(
                timestamp=datetime.now(),
                indicator_name="Money Flow Index",
                indicator_type=IndicatorType.VOLUME,
                timeframe=TimeFrame.D1,
                value=mfi_values,
                signal=signals,
                raw_data={
                    'period': self.period,
                    'overbought_threshold': 80,
                    'oversold_threshold': 20
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in MFI calculation: {e}")
            return IndicatorResult(
                timestamp=datetime.now(),
                indicator_name="Money Flow Index",
                indicator_type=IndicatorType.VOLUME,
                timeframe=TimeFrame.D1,
                value=[],
                signal=[],
                raw_data={"error": str(e)}
            )

# Legacy alias for backward compatibility
class Mfi(MoneyFlowIndex):
    """Legacy Mfi class - use MoneyFlowIndex instead"""
    pass