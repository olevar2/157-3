"""
On Balance Volume (OBV)
A technical indicator that reflects cumulative buying/selling pressure by adding volume on up days and subtracting volume on down days.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Dict, Any
from datetime import datetime
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType, IndicatorConfig, IndicatorType, TimeFrame


class OnBalanceVolume(TechnicalIndicator):
    """
    On Balance Volume (OBV)
    
    The OBV is a cumulative indicator that adds volume on up days and subtracts volume on down days.
    It's used to confirm price movements by measuring buying/selling pressure.
    
    Formula:
    If close > previous close:
        OBV = previous OBV + current volume
    Else if close < previous close:
        OBV = previous OBV - current volume
    Else:
        OBV = previous OBV
    """
    
    def __init__(self, config=None):
        """
        Initialize On Balance Volume
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary (not used for OBV but accepted for consistency)
        """
        # Create proper IndicatorConfig if needed
        if config is None or not isinstance(config, IndicatorConfig):
            config_dict = config or {}
            config = IndicatorConfig(
                name="On Balance Volume",
                indicator_type=IndicatorType.VOLUME,
                timeframe=config_dict.get('timeframe', TimeFrame.M15),
                lookback_periods=20
            )
        
        super().__init__(config)
        self.name = "On Balance Volume"
        
    def calculate(self, data: List[MarketData]) -> 'IndicatorResult':
        """
        Calculate OBV for base class compatibility
        
        Parameters:
        -----------
        data : List[MarketData]
            List of market data points
            
        Returns:
        --------
        IndicatorResult
            Result with OBV values
        """
        from engines.indicator_base import IndicatorResult
        
        if not data:
            return IndicatorResult(
                timestamp=datetime.now(),
                indicator_name=self.name,
                indicator_type=IndicatorType.VOLUME,
                timeframe=self.config.timeframe,
                value=0.0
            )
        
        # Convert MarketData to pandas series
        close = pd.Series([d.close for d in data])
        volume = pd.Series([d.volume for d in data])
        
        obv = self.calculate_values(close, volume)
        
        latest_timestamp = data[-1].timestamp if data else datetime.now()
        
        return IndicatorResult(
            timestamp=latest_timestamp,
            indicator_name=self.name,
            indicator_type=IndicatorType.VOLUME,
            timeframe=self.config.timeframe,
            value=obv.iloc[-1] if len(obv) > 0 else 0.0
        )
        
    def calculate_values(self, close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Calculate On Balance Volume
        
        Parameters:
        -----------
        close : pd.Series or np.ndarray
            Close prices
        volume : pd.Series or np.ndarray
            Volume data
            
        Returns:
        --------
        pd.Series
            OBV values
        """
        # Convert to pandas Series if numpy arrays
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
            
        # Initialize OBV series
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]  # Start with first day's volume
        
        # Calculate OBV
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                # Price increased - add volume
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                # Price decreased - subtract volume
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                # Price unchanged - keep previous OBV
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def generate_signal(self, data: MarketData) -> IndicatorSignal:
        """
        Generate signals based on On Balance Volume for compatibility with indicator registry
        
        Parameters:
        -----------
        data : MarketData
            Market data containing OHLCV information
            
        Returns:
        --------
        IndicatorSignal
            Signal object with buy/sell/neutral recommendation
        """
        if not hasattr(data, 'data') or not all(key in data.data for key in ["close", "volume"]):
            return IndicatorSignal(
                timestamp=data.timestamp if hasattr(data, 'timestamp') else datetime.now(),
                indicator_name=self.name, 
                signal_type=SignalType.NEUTRAL, 
                strength=0.0,
                confidence=0.0
            )
            
        close = data.data["close"]
        volume = data.data["volume"]
        
        # Calculate OBV
        obv = self.calculate_values(close, volume)
        
        # Calculate 20-period EMA of OBV for signal generation
        obv_ema = obv.ewm(span=20, adjust=False).mean()
        
        # Determine signal based on OBV vs its EMA and trend
        last_obv = obv.iloc[-1]
        prev_obv = obv.iloc[-2] if len(obv) > 1 else last_obv
        last_obv_ema = obv_ema.iloc[-1]
        
        # Check if OBV is rising and above its EMA
        if last_obv > prev_obv and last_obv > last_obv_ema:
            signal_type = SignalType.BUY
            strength = 0.6
            confidence = 0.5
            message = f"OBV ({last_obv:.0f}) is rising and above its EMA ({last_obv_ema:.0f}), suggesting strong buying pressure"
        # Check if OBV is falling and below its EMA
        elif last_obv < prev_obv and last_obv < last_obv_ema:
            signal_type = SignalType.SELL
            strength = 0.6
            confidence = 0.5
            message = f"OBV ({last_obv:.0f}) is falling and below its EMA ({last_obv_ema:.0f}), suggesting strong selling pressure"
        else:
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            confidence = 0.3
            message = f"OBV ({last_obv:.0f}) shows no clear trend relative to its EMA ({last_obv_ema:.0f})"
        
        return IndicatorSignal(
            timestamp=data.timestamp if hasattr(data, 'timestamp') else datetime.now(),
            indicator_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            metadata={
                "obv": last_obv,
                "obv_ema": last_obv_ema,
                "message": message
            }
        )
