"""
Klinger Oscillator Indicator

The Klinger Oscillator (KO) is a volume-based technical indicator that attempts to
predict long-term money flow while remaining sensitive to short-term fluctuations.
It combines price and volume to measure accumulation and distribution.

Formula:
KO = EMA(VF, fast) - EMA(VF, slow)
Where VF (Volume Force) = Volume × Trend × (2 × ((dm/cm) - 1))

Author: Platform3 Trading System
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
from datetime import datetime
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType, IndicatorConfig, IndicatorType, TimeFrame


class KlingerOscillator(TechnicalIndicator):
    """
    Klinger Oscillator Implementation    
    The Klinger Oscillator is a volume-based momentum oscillator that uses the
    relationship between price movement and volume to generate trading signals.
    """
    
    def __init__(self, config: dict = None, fast_period: int = 34, slow_period: int = 55, signal_period: int = 13):
        """
        Initialize Klinger Oscillator
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary
        fast_period : int, default 34
            Period for fast EMA
        slow_period : int, default 55
            Period for slow EMA
        signal_period : int, default 13
            Period for signal line EMA
        """
        # Create proper IndicatorConfig if needed
        if config is None or not isinstance(config, IndicatorConfig):
            config_dict = config or {}
            config = IndicatorConfig(
                name=f"Klinger Oscillator ({fast_period}/{slow_period}/{signal_period})",
                indicator_type=IndicatorType.VOLUME,
                timeframe=config_dict.get('timeframe', TimeFrame.M15),
                lookback_periods=max(fast_period, slow_period, signal_period) + 20,
                parameters={
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'signal_period': signal_period
                }
            )
        
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.name = f"Klinger Oscillator ({fast_period}/{slow_period}/{signal_period})"
        
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
    
    def _calculate_volume_force(self, high: pd.Series, low: pd.Series, 
                               close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Volume Force (VF)
        
        Parameters:
        -----------
        high, low, close : pd.Series
            OHLC price data
        volume : pd.Series
            Volume data
            
        Returns:
        --------
        pd.Series
            Volume Force values
        """
        # Calculate typical price (HLC/3)
        hlc = (high + low + close) / 3
        
        # Determine trend direction
        trend = pd.Series(index=close.index, dtype=float)
        trend.iloc[0] = 1  # Initialize first value
        
        for i in range(1, len(close)):
            if hlc.iloc[i] > hlc.iloc[i-1]:
                trend.iloc[i] = 1
            elif hlc.iloc[i] < hlc.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]  # Keep previous trend
        
        # Calculate dm (daily measurement)
        dm = high - low
        
        # Calculate cm (cumulative measurement)
        cm = pd.Series(index=close.index, dtype=float)
        cm.iloc[0] = dm.iloc[0]
        
        for i in range(1, len(close)):
            if trend.iloc[i] == trend.iloc[i-1]:
                cm.iloc[i] = cm.iloc[i-1] + dm.iloc[i]
            else:
                cm.iloc[i] = dm.iloc[i-1] + dm.iloc[i]
        
        # Avoid division by zero
        cm = cm.replace(0, np.nan)
          # Calculate Volume Force
        vf = volume * trend * (2 * ((dm / cm) - 1))
        return vf.fillna(0)
    
    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate(self, data: List[MarketData]) -> 'IndicatorResult':
        """
        Calculate Klinger Oscillator for base class compatibility
        
        Parameters:
        -----------
        data : List[MarketData]
            List of market data points
            
        Returns:
        --------
        IndicatorResult
            Result with Klinger Oscillator values
        """
        from engines.indicator_base import IndicatorResult
        
        if not data:
            return IndicatorResult(
                timestamp=datetime.now(),
                indicator_name=self.name,
                indicator_type=IndicatorType.VOLUME,
                timeframe=self.config.timeframe,
                value={"klinger_oscillator": 0.0, "signal": 0.0, "histogram": 0.0}
            )
        
        # Convert MarketData to pandas series
        high = pd.Series([d.high for d in data])
        low = pd.Series([d.low for d in data]) 
        close = pd.Series([d.close for d in data])
        volume = pd.Series([d.volume for d in data])
        
        result = self.calculate_values(high, low, close, volume)
        
        latest_timestamp = data[-1].timestamp if data else datetime.now()
        
        return IndicatorResult(
            timestamp=latest_timestamp,
            indicator_name=self.name,
            indicator_type=IndicatorType.VOLUME,
            timeframe=self.config.timeframe,
            value={
                "klinger_oscillator": result['klinger_oscillator'].iloc[-1] if len(result) > 0 else 0.0,
                "signal": result['signal'].iloc[-1] if len(result) > 0 else 0.0,
                "histogram": result['histogram'].iloc[-1] if len(result) > 0 else 0.0
            }
        )
    
    def calculate_values(self, high: Union[pd.Series, np.ndarray], 
                 low: Union[pd.Series, np.ndarray],
                 close: Union[pd.Series, np.ndarray], 
                 volume: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Calculate Klinger Oscillator
        
        Parameters:
        -----------
        high, low, close : pd.Series or np.ndarray
            OHLC price data
        volume : pd.Series or np.ndarray
            Volume data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with Klinger Oscillator and signal line
        """
        # Convert to pandas Series if needed
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
        
        # Calculate Volume Force
        vf = self._calculate_volume_force(high, low, close, volume)
        
        # Calculate fast and slow EMAs of Volume Force
        fast_ema = self._ema(vf, self.fast_period)
        slow_ema = self._ema(vf, self.slow_period)
        
        # Calculate Klinger Oscillator
        ko = fast_ema - slow_ema
        
        # Calculate signal line
        signal = self._ema(ko, self.signal_period)
        
        result = pd.DataFrame(index=close.index)
        result['klinger_oscillator'] = ko
        result['signal'] = signal
        result['histogram'] = ko - signal
        
        return result
    
    def get_signals(self, high: Union[pd.Series, np.ndarray], 
                   low: Union[pd.Series, np.ndarray],
                   close: Union[pd.Series, np.ndarray], 
                   volume: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Generate trading signals based on Klinger Oscillator
        
        Parameters:
        -----------
        high, low, close : pd.Series or np.ndarray
            OHLC price data
        volume : pd.Series or np.ndarray
            Volume data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with Klinger Oscillator and signals
        """
        result = self.calculate_values(high, low, close, volume)
        
        # Signal line crossovers
        result['signal_cross_up'] = (
            (result['klinger_oscillator'] > result['signal']) & 
            (result['klinger_oscillator'].shift(1) <= result['signal'].shift(1))
        )
        result['signal_cross_down'] = (
            (result['klinger_oscillator'] < result['signal']) & 
            (result['klinger_oscillator'].shift(1) >= result['signal'].shift(1))
        )
        
        # Zero line crossovers
        result['zero_cross_up'] = (
            (result['klinger_oscillator'] > 0) & 
            (result['klinger_oscillator'].shift(1) <= 0)
        )
        result['zero_cross_down'] = (
            (result['klinger_oscillator'] < 0) & 
            (result['klinger_oscillator'].shift(1) >= 0)
        )
        
        # Trading signals        result['trade_signal'] = 0
        result.loc[result['signal_cross_up'], 'trade_signal'] = 1
        result.loc[result['signal_cross_down'], 'trade_signal'] = -1
        
        return result

    def generate_signal(self, data: MarketData) -> IndicatorSignal:
        """
        Generate signals based on Klinger Oscillator for compatibility with indicator registry
        
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
        
        result = self.get_signals(high, low, close, volume)
        
        # Get the last signal
        if result['trade_signal'].iloc[-1] > 0:
            signal_type = SignalType.BUY
            strength = 0.7
            confidence = 0.6
            message = f"Klinger Oscillator crossed above signal line. KO: {result['klinger_oscillator'].iloc[-1]:.4f}"
        elif result['trade_signal'].iloc[-1] < 0:
            signal_type = SignalType.SELL
            strength = 0.7
            confidence = 0.6
            message = f"Klinger Oscillator crossed below signal line. KO: {result['klinger_oscillator'].iloc[-1]:.4f}"
        else:
            signal_type = SignalType.NEUTRAL
            strength = 0.0
            confidence = 0.3
            message = f"No signal from Klinger Oscillator. KO: {result['klinger_oscillator'].iloc[-1]:.4f}"
        
        return IndicatorSignal(
            timestamp=data.timestamp if hasattr(data, 'timestamp') else datetime.now(),
            indicator_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            metadata={
                "klinger_oscillator": result['klinger_oscillator'].iloc[-1],
                "signal": result['signal'].iloc[-1],
                "histogram": result['histogram'].iloc[-1],
                "message": message
            }
        )
