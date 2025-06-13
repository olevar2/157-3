"""
Williams %R
===========

Williams %R is a momentum indicator that measures overbought and oversold levels.
It oscillates between 0 and -100, with readings above -20 indicating overbought 
conditions and readings below -80 indicating oversold conditions.

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


class WilliamsR(IndicatorBase):
    """
    Williams %R indicator.
    
    Williams %R compares the closing price to the high-low range over 
    a specific period, typically 14 periods.
    """
    
    def __init__(self, 
                 period: int = 14,
                 overbought_level: float = -20.0,
                 oversold_level: float = -80.0):
        """
        Initialize Williams %R indicator.
        
        Args:
            period: Period for Williams %R calculation
            overbought_level: Level indicating overbought condition
            oversold_level: Level indicating oversold condition
        """
        super().__init__()
        
        self.period = period
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        
        # Validation
        if period <= 0:
            raise ValueError("period must be positive")
        if not -100 <= oversold_level < overbought_level <= 0:
            raise ValueError("Invalid overbought/oversold levels")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Williams %R values.
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Dictionary containing Williams %R values and signals
        """
        try:
            # Validate input data using parent class method
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError("Empty DataFrame provided")
                
                # Check required columns manually
                required_columns = ['high', 'low', 'close']
                missing = []
                for col in required_columns:
                    if col not in data.columns:
                        missing.append(col)
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")
            else:
                validation_result = super()._validate_data(data)
                if not validation_result:
                    raise ValueError("Invalid input data format")
            
            if len(data) < self.period:
                raise ValueError(f"Insufficient data: need at least {self.period} periods")
            
            # Convert to numpy arrays for calculation
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate Williams %R
            williams_r = self._calculate_williams_r(high, low, close)
            
            # Generate signals
            signals = self._generate_signals(williams_r)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(williams_r)
            
            return {
                'williams_r': williams_r,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(williams_r[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            raise
    
    def _calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Williams %R values."""
        williams_r = np.full(len(close), np.nan)
        
        for i in range(self.period - 1, len(close)):
            # Get the period's high and low
            period_high = np.max(high[i - self.period + 1:i + 1])
            period_low = np.min(low[i - self.period + 1:i + 1])
            
            # Calculate Williams %R
            if period_high != period_low:
                williams_r[i] = ((period_high - close[i]) / (period_high - period_low)) * -100
            else:
                williams_r[i] = -50  # Neutral when no range
        
        return williams_r
    
    def _generate_signals(self, williams_r: np.ndarray) -> np.ndarray:
        """Generate trading signals based on Williams %R."""
        signals = np.zeros(len(williams_r))
        
        for i in range(1, len(williams_r)):
            if np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]):
                continue
            
            # Oversold bounce signal
            if (williams_r[i-1] <= self.oversold_level and 
                williams_r[i] > self.oversold_level):
                signals[i] = 1
            
            # Overbought reversal signal
            elif (williams_r[i-1] >= self.overbought_level and 
                  williams_r[i] < self.overbought_level):
                signals[i] = -1
        
        return signals
    
    def _calculate_metrics(self, williams_r: np.ndarray) -> Dict:
        """Calculate additional Williams %R metrics."""
        valid_values = williams_r[~np.isnan(williams_r)]
        
        if len(valid_values) == 0:
            return {}
        
        # Time in different zones
        overbought_pct = np.sum(valid_values >= self.overbought_level) / len(valid_values) * 100
        oversold_pct = np.sum(valid_values <= self.oversold_level) / len(valid_values) * 100
        neutral_pct = 100 - overbought_pct - oversold_pct
        
        # Current momentum
        recent_values = valid_values[-min(5, len(valid_values)):]
        momentum = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
        
        return {
            'current_value': williams_r[-1] if not np.isnan(williams_r[-1]) else None,
            'overbought_percentage': overbought_pct,
            'oversold_percentage': oversold_pct,
            'neutral_percentage': neutral_pct,
            'momentum': momentum,
            'volatility': np.std(valid_values),
            'mean_value': np.mean(valid_values)
        }
    
    def _interpret_signals(self, current_wr: float, current_signal: int) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_wr):
            return "Insufficient data for Williams %R calculation"
        
        if current_wr >= self.overbought_level:
            condition = "OVERBOUGHT"
        elif current_wr <= self.oversold_level:
            condition = "OVERSOLD"
        else:
            condition = "NEUTRAL"
        
        signal_desc = {
            1: "BUY signal generated",
            -1: "SELL signal generated", 
            0: "No signal"
        }.get(current_signal, "No signal")
        
        return f"Williams %R: {current_wr:.2f} ({condition}) - {signal_desc}"


def create_williams_r(period: int = 14, **kwargs) -> WilliamsR:
    """Factory function to create Williams %R indicator."""
    return WilliamsR(period=period, **kwargs)
