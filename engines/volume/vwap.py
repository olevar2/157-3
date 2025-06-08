"""
Volume Weighted Average Price (VWAP)
A trading indicator that shows the ratio of the value traded to total volume traded over a time period.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Dict, Any
from datetime import datetime
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType, IndicatorConfig, IndicatorType, TimeFrame


class VolumeWeightedAveragePrice(TechnicalIndicator):
    """
    Volume Weighted Average Price (VWAP)
    
    VWAP is calculated by finding the sum of dollars traded for each transaction
    (price multiplied by the number of shares traded) and then dividing by the 
    total shares traded. It represents the average price a security has traded 
    at throughout the day, based on both volume and price.
    
    Formula:
    VWAP = ∑(Price * Volume) / ∑(Volume)
    
    Often calculated per day and reset at market open.
    """
    
    def __init__(self, reset_period: str = 'day', config=None):
        """
        Initialize Volume Weighted Average Price
        
        Args:
            reset_period: When to reset the VWAP calculation ('day', 'week', 'month')
            config : dict, optional
                Configuration dictionary containing parameters
        """
        # Handle config parameter
        if config is not None:
            if isinstance(config, dict):
                self.reset_period = config.get('reset_period', reset_period)
            else:
                self.reset_period = reset_period
        else:
            self.reset_period = reset_period
        
        # Create proper IndicatorConfig if needed
        if config is None or not isinstance(config, IndicatorConfig):
            config_dict = config if isinstance(config, dict) else {}
            config = IndicatorConfig(
                name="Volume Weighted Average Price",
                indicator_type=IndicatorType.VOLUME,
                timeframe=config_dict.get('timeframe', TimeFrame.M15),
                lookback_periods=50,
                parameters={'reset_period': self.reset_period}
            )
        
        super().__init__(config)    
        self.name = "Volume Weighted Average Price"
        
    def calculate(self, data: List[MarketData]) -> 'IndicatorResult':
        """
        Calculate VWAP for base class compatibility
        
        Parameters:
        -----------
        data : List[MarketData]
            List of market data points
            
        Returns:
        --------
        IndicatorResult
            Result with VWAP values
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
        high = pd.Series([d.high for d in data])
        low = pd.Series([d.low for d in data]) 
        close = pd.Series([d.close for d in data])
        volume = pd.Series([d.volume for d in data])
        dates = pd.Series([d.timestamp for d in data])
        
        vwap = self.calculate_values(high, low, close, volume, dates)
        
        latest_timestamp = data[-1].timestamp if data else datetime.now()
        
        return IndicatorResult(
            timestamp=latest_timestamp,
            indicator_name=self.name,
            indicator_type=IndicatorType.VOLUME,
            timeframe=self.config.timeframe,
            value=vwap.iloc[-1] if len(vwap) > 0 else 0.0
        )
        
    def calculate_values(self, high: Union[pd.Series, np.ndarray], 
                low: Union[pd.Series, np.ndarray],
                close: Union[pd.Series, np.ndarray], 
                volume: Union[pd.Series, np.ndarray],
                dates: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate Volume Weighted Average Price
        
        Parameters:
        -----------
        high : pd.Series or np.ndarray
            High prices
        low : pd.Series or np.ndarray
            Low prices
        close : pd.Series or np.ndarray
            Close prices
        volume : pd.Series or np.ndarray
            Volume data
        dates : pd.Series, optional
            Datetime index for resetting VWAP calculation
            
        Returns:
        --------
        pd.Series
            VWAP values
        """
        # Convert to pandas Series if numpy arrays
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
            
        # Calculate typical price: (high + low + close) / 3
        typical_price = (high + low + close) / 3
        
        # Use index as dates if none provided
        if dates is None:
            if isinstance(close.index, pd.DatetimeIndex):
                dates = close.index
            else:
                # If no dates provided and index is not DatetimeIndex, 
                # we can't reset based on dates, so calculate cumulative VWAP
                cumulative_tp_vol = (typical_price * volume).cumsum()
                cumulative_vol = volume.cumsum()
                return cumulative_tp_vol / cumulative_vol
        
        # Reset VWAP calculation based on reset_period
        vwap = pd.Series(index=close.index, dtype=float)
        
        if self.reset_period == 'day':
            date_groups = dates.dt.date
        elif self.reset_period == 'week':
            date_groups = dates.dt.isocalendar().week
        elif self.reset_period == 'month':
            date_groups = dates.dt.month
        else:
            # Default to daily reset
            date_groups = dates.dt.date
        
        # Calculate VWAP for each period
        for date_value in date_groups.unique():
            mask = date_groups == date_value
            period_tp = typical_price.loc[mask]
            period_volume = volume.loc[mask]
            
            # Calculate cumulative values for the period
            cum_tp_vol = (period_tp * period_volume).cumsum()
            cum_vol = period_volume.cumsum()
            
            # Calculate VWAP
            period_vwap = cum_tp_vol / cum_vol
            vwap.loc[mask] = period_vwap
        
        return vwap
    
    def generate_signal(self, data: MarketData) -> IndicatorSignal:
        """
        Generate signals based on Volume Weighted Average Price for compatibility with indicator registry
        
        Parameters:
        -----------
        data : MarketData
            Market data containing OHLCV information
            
        Returns:
        --------
        IndicatorSignal
            Signal object with buy/sell/neutral recommendation
        """
        if not hasattr(data, 'data') or not all(key in data.data for key in ["high", "low", "close", "volume"]):
            return IndicatorSignal(
                timestamp=data.timestamp if hasattr(data, 'timestamp') else datetime.now(),
                indicator_name=self.name, 
                signal_type=SignalType.NEUTRAL, 
                strength=0.0,
                confidence=0.0
            )
            
        high = data.data["high"]
        low = data.data["low"]
        close = data.data["close"]
        volume = data.data["volume"]
        
        # Get datetime index if available
        dates = None
        if isinstance(data.data.get("datetime"), pd.Series) or isinstance(close.index, pd.DatetimeIndex):
            dates = data.data.get("datetime") if "datetime" in data.data else close.index
        
        # Calculate VWAP
        vwap = self.calculate_values(high, low, close, volume, dates)
        
        # Get the last close and VWAP values
        last_close = close.iloc[-1]
        last_vwap = vwap.iloc[-1]
        
        # Calculate distance from VWAP as percentage
        distance = ((last_close / last_vwap) - 1) * 100
        
        # Thresholds for signals (price relative to VWAP)
        upper_threshold = 1.5  # 1.5% above VWAP
        lower_threshold = -1.5  # 1.5% below VWAP
        
        if distance > upper_threshold:
            signal_type = SignalType.SELL
            strength = 0.6
            confidence = 0.5
            message = f"Price ({last_close:.4f}) is {distance:.2f}% above VWAP ({last_vwap:.4f}), potentially overbought"
        elif distance < lower_threshold:
            signal_type = SignalType.BUY
            strength = 0.6
            confidence = 0.5
            message = f"Price ({last_close:.4f}) is {distance:.2f}% below VWAP ({last_vwap:.4f}), potentially oversold"
        else:
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            confidence = 0.3
            message = f"Price ({last_close:.4f}) is close to VWAP ({last_vwap:.4f}), {distance:.2f}% difference"
        
        return IndicatorSignal(
            timestamp=data.timestamp if hasattr(data, 'timestamp') else datetime.now(),
            indicator_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            metadata={
                "vwap": last_vwap,
                "close": last_close,
                "distance_pct": distance,
                "message": message
            }
        )
