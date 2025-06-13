"""
Commodity Channel Index (CCI)
=============================

The CCI is a momentum indicator that compares the current price to an 
average price over a specified period. It's used to identify cyclical 
trends and potential reversal points.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
        from indicator_base import IndicatorBase


class CommodityChannelIndex(IndicatorBase):
    """
    Commodity Channel Index (CCI) indicator.
    
    CCI measures the difference between a security's price change and its 
    average price change. Values above +100 may indicate the beginning 
    of a new uptrend, while values below -100 may indicate a new downtrend.
    """
    
    def __init__(self, 
                 period: int = 20,
                 constant: float = 0.015,
                 overbought_level: float = 100.0,
                 oversold_level: float = -100.0):
        """
        Initialize CCI indicator.
        
        Args:
            period: Period for CCI calculation
            constant: Constant factor (typically 0.015)
            overbought_level: CCI level indicating overbought condition
            oversold_level: CCI level indicating oversold condition
        """
        super().__init__()
        
        self.period = period
        self.constant = constant
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        
        # Validation
        if period <= 0:
            raise ValueError("period must be positive")
        if constant <= 0:
            raise ValueError("constant must be positive")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate CCI values.
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Dictionary containing CCI values and signals
        """
        try:
            # Validate input data
            required_columns = ['high', 'low', 'close']
            self._validate_data(data, required_columns)
            
            if len(data) < self.period:
                raise ValueError(f"Insufficient data: need at least {self.period} periods")
            
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate CCI
            cci = self._calculate_cci(high, low, close)
            
            # Generate signals
            signals = self._generate_signals(cci)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(cci)
            
            return {
                'cci': cci,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(cci[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            raise
    
    def _calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate CCI values."""
        # Calculate Typical Price (TP)
        typical_price = (high + low + close) / 3
        
        cci = np.full(len(close), np.nan)
        
        for i in range(self.period - 1, len(close)):
            # Calculate Simple Moving Average of TP
            tp_sma = np.mean(typical_price[i - self.period + 1:i + 1])
            
            # Calculate Mean Absolute Deviation
            mad = np.mean(np.abs(typical_price[i - self.period + 1:i + 1] - tp_sma))
            
            # Calculate CCI
            if mad != 0:
                cci[i] = (typical_price[i] - tp_sma) / (self.constant * mad)
            else:
                cci[i] = 0
        
        return cci
    
    def _generate_signals(self, cci: np.ndarray) -> np.ndarray:
        """Generate trading signals based on CCI."""
        signals = np.zeros(len(cci))
        
        for i in range(1, len(cci)):
            if np.isnan(cci[i]) or np.isnan(cci[i-1]):
                continue
            
            # Oversold bounce signal
            if (cci[i-1] <= self.oversold_level and 
                cci[i] > self.oversold_level):
                signals[i] = 1
            
            # Overbought reversal signal
            elif (cci[i-1] >= self.overbought_level and 
                  cci[i] < self.overbought_level):
                signals[i] = -1
            
            # Zero line crossovers
            elif cci[i-1] < 0 and cci[i] > 0:
                signals[i] = 0.5  # Weak buy
            elif cci[i-1] > 0 and cci[i] < 0:
                signals[i] = -0.5  # Weak sell
        
        return signals
    
    def _calculate_metrics(self, cci: np.ndarray) -> Dict:
        """Calculate additional CCI metrics."""
        valid_values = cci[~np.isnan(cci)]
        
        if len(valid_values) == 0:
            return {}
        
        # Time in different zones
        overbought_pct = np.sum(valid_values >= self.overbought_level) / len(valid_values) * 100
        oversold_pct = np.sum(valid_values <= self.oversold_level) / len(valid_values) * 100
        neutral_pct = 100 - overbought_pct - oversold_pct
        
        # Extreme readings
        extreme_high = np.max(valid_values)
        extreme_low = np.min(valid_values)
        
        # Current momentum
        recent_values = valid_values[-min(5, len(valid_values)):]
        momentum = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
        
        return {
            'current_value': cci[-1] if not np.isnan(cci[-1]) else None,
            'overbought_percentage': overbought_pct,
            'oversold_percentage': oversold_pct,
            'neutral_percentage': neutral_pct,
            'extreme_high': extreme_high,
            'extreme_low': extreme_low,
            'momentum': momentum,
            'volatility': np.std(valid_values),
            'mean_value': np.mean(valid_values)
        }
    
    def _interpret_signals(self, current_cci: float, current_signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_cci):
            return "Insufficient data for CCI calculation"
        
        if current_cci >= self.overbought_level:
            condition = "OVERBOUGHT"
        elif current_cci <= self.oversold_level:
            condition = "OVERSOLD"
        elif current_cci > 0:
            condition = "BULLISH"
        else:
            condition = "BEARISH"
        
        signal_desc = {
            1: "Strong BUY signal",
            0.5: "Weak BUY signal",
            -0.5: "Weak SELL signal",
            -1: "Strong SELL signal",
            0: "No signal"
        }.get(current_signal, "No signal")
        
        return f"CCI: {current_cci:.2f} ({condition}) - {signal_desc}"


def create_cci(period: int = 20, **kwargs) -> CommodityChannelIndex:
    """Factory function to create CCI indicator."""
    return CommodityChannelIndex(period=period, **kwargs)
