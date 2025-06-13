"""
Rate of Change (ROC)
===================

ROC is a momentum indicator that measures the percentage change in price 
from one period to the next. It oscillates around zero, with positive 
values indicating upward momentum and negative values indicating downward momentum.

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


class RateOfChange(IndicatorBase):
    """
    Rate of Change (ROC) indicator.
    
    ROC calculates the percentage change in price over a specified period,
    providing insight into the momentum and velocity of price movements.
    """
    
    def __init__(self, 
                 period: int = 12,
                 use_percentage: bool = True):
        """
        Initialize ROC indicator.
        
        Args:
            period: Period for ROC calculation
            use_percentage: If True, return percentage change; if False, return ratio
        """
        super().__init__()
        
        self.period = period
        self.use_percentage = use_percentage
        
        # Validation
        if period <= 0:
            raise ValueError("period must be positive")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate ROC values.
        
        Args:
            data: DataFrame with 'close' column (can also have 'high', 'low')
            
        Returns:
            Dictionary containing ROC values and signals
        """
        try:
            # Validate input data
            required_columns = ['close']
            self._validate_data(data, required_columns)
            
            if len(data) < self.period + 1:
                raise ValueError(f"Insufficient data: need at least {self.period + 1} periods")
            
            close = data['close'].values
            
            # Calculate ROC
            roc = self._calculate_roc(close)
            
            # Generate signals
            signals = self._generate_signals(roc)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(roc)
            
            return {
                'roc': roc,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(roc[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ROC: {e}")
            raise
    
    def _calculate_roc(self, close: np.ndarray) -> np.ndarray:
        """Calculate ROC values."""
        roc = np.full(len(close), np.nan)
        
        for i in range(self.period, len(close)):
            prev_close = close[i - self.period]
            current_close = close[i]
            
            if prev_close != 0:
                if self.use_percentage:
                    roc[i] = ((current_close - prev_close) / prev_close) * 100
                else:
                    roc[i] = current_close / prev_close
            else:
                roc[i] = 0
        
        return roc
    
    def _generate_signals(self, roc: np.ndarray) -> np.ndarray:
        """Generate trading signals based on ROC."""
        signals = np.zeros(len(roc))
        
        # Calculate dynamic thresholds based on historical volatility
        valid_values = roc[~np.isnan(roc)]
        if len(valid_values) < 10:
            threshold = 5.0 if self.use_percentage else 0.05
        else:
            std_dev = np.std(valid_values)
            threshold = std_dev * 0.5
        
        for i in range(1, len(roc)):
            if np.isnan(roc[i]) or np.isnan(roc[i-1]):
                continue
            
            # Zero line crossovers
            if roc[i-1] <= 0 and roc[i] > 0:
                signals[i] = 1  # Buy signal
            elif roc[i-1] >= 0 and roc[i] < 0:
                signals[i] = -1  # Sell signal
            
            # Extreme momentum signals
            elif roc[i] > threshold and roc[i-1] <= threshold:
                signals[i] = 0.5  # Weak buy (strong momentum)
            elif roc[i] < -threshold and roc[i-1] >= -threshold:
                signals[i] = -0.5  # Weak sell (strong negative momentum)
        
        return signals
    
    def _calculate_metrics(self, roc: np.ndarray) -> Dict:
        """Calculate additional ROC metrics."""
        valid_values = roc[~np.isnan(roc)]
        
        if len(valid_values) == 0:
            return {}
        
        # Momentum analysis
        positive_pct = np.sum(valid_values > 0) / len(valid_values) * 100
        negative_pct = np.sum(valid_values < 0) / len(valid_values) * 100
        neutral_pct = np.sum(valid_values == 0) / len(valid_values) * 100
        
        # Extreme readings
        max_momentum = np.max(valid_values)
        min_momentum = np.min(valid_values)
        
        # Recent trend
        recent_values = valid_values[-min(5, len(valid_values)):]
        trend = np.mean(recent_values) if len(recent_values) > 0 else 0
        
        # Acceleration (second derivative)
        if len(valid_values) > 1:
            acceleration = np.mean(np.diff(valid_values[-min(3, len(valid_values)):]))
        else:
            acceleration = 0
        
        return {
            'current_value': roc[-1] if not np.isnan(roc[-1]) else None,
            'positive_momentum_pct': positive_pct,
            'negative_momentum_pct': negative_pct,
            'neutral_pct': neutral_pct,
            'max_momentum': max_momentum,
            'min_momentum': min_momentum,
            'recent_trend': trend,
            'acceleration': acceleration,
            'volatility': np.std(valid_values),
            'mean_value': np.mean(valid_values)
        }
    
    def _interpret_signals(self, current_roc: float, current_signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_roc):
            return "Insufficient data for ROC calculation"
        
        if current_roc > 5:
            momentum = "STRONG POSITIVE"
        elif current_roc > 0:
            momentum = "POSITIVE"
        elif current_roc < -5:
            momentum = "STRONG NEGATIVE"
        else:
            momentum = "NEGATIVE"
        
        signal_desc = {
            1: "BUY signal (zero line cross up)",
            0.5: "Momentum acceleration signal",
            -0.5: "Momentum deceleration signal",
            -1: "SELL signal (zero line cross down)",
            0: "No signal"
        }.get(current_signal, "No signal")
        
        unit = "%" if self.use_percentage else ""
        return f"ROC: {current_roc:.2f}{unit} ({momentum} momentum) - {signal_desc}"


def create_roc(period: int = 12, **kwargs) -> RateOfChange:
    """Factory function to create ROC indicator."""
    return RateOfChange(period=period, **kwargs)
