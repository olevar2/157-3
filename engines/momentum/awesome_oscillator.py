"""
Awesome Oscillator (AO)
=======================

The Awesome Oscillator is a momentum indicator that shows the difference 
between a 5-period and 34-period simple moving average of the median prices.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
        from indicator_base import IndicatorBase


class AwesomeOscillator(IndicatorBase):
    """
    Awesome Oscillator (AO) indicator.
    
    AO is calculated as the difference between a 5-period SMA and 34-period SMA 
    of the median price (high+low)/2.
    """
    
    def __init__(self, 
                 fast_period: int = 5,
                 slow_period: int = 34):
        """
        Initialize Awesome Oscillator.
        
        Args:
            fast_period: Fast SMA period (typically 5)
            slow_period: Slow SMA period (typically 34)
        """
        super().__init__()
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Validation
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("Periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Awesome Oscillator values.
        
        Args:
            data: DataFrame with 'high', 'low' columns
            
        Returns:
            Dictionary containing AO values and signals
        """
        try:
            # Validate input data
            required_columns = ['high', 'low']
            self._validate_data(data, required_columns)
            
            if len(data) < self.slow_period:
                raise ValueError(f"Insufficient data: need at least {self.slow_period} periods")
            
            high = data['high'].values
            low = data['low'].values
            
            # Calculate Awesome Oscillator
            ao = self._calculate_awesome_oscillator(high, low)
            
            # Generate signals
            signals = self._generate_signals(ao)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(ao)
            
            return {
                'awesome_oscillator': ao,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(ao[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Awesome Oscillator: {e}")
            raise
    
    def _calculate_awesome_oscillator(self, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Calculate Awesome Oscillator values."""
        # Calculate median price
        median_price = (high + low) / 2
        
        # Calculate SMAs
        fast_sma = self._sma(median_price, self.fast_period)
        slow_sma = self._sma(median_price, self.slow_period)
        
        # Calculate AO
        ao = fast_sma - slow_sma
        
        return ao
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        return sma
    
    def _generate_signals(self, ao: np.ndarray) -> np.ndarray:
        """Generate trading signals based on Awesome Oscillator."""
        signals = np.zeros(len(ao))
        
        for i in range(2, len(ao)):
            if np.isnan(ao[i]) or np.isnan(ao[i-1]) or np.isnan(ao[i-2]):
                continue
            
            # Zero line crossovers
            if ao[i-1] <= 0 and ao[i] > 0:
                signals[i] = 1  # Buy signal
            elif ao[i-1] >= 0 and ao[i] < 0:
                signals[i] = -1  # Sell signal
            
            # Saucer signal (bullish)
            elif (ao[i] > 0 and ao[i-1] > 0 and ao[i-2] > 0 and
                  ao[i] > ao[i-1] and ao[i-1] < ao[i-2]):
                signals[i] = 0.5  # Weak buy
            
            # Twin peaks signal (bearish)
            elif (ao[i] < 0 and ao[i-1] < 0 and ao[i-2] < 0 and
                  ao[i] < ao[i-1] and ao[i-1] > ao[i-2]):
                signals[i] = -0.5  # Weak sell
        
        return signals
    
    def _calculate_metrics(self, ao: np.ndarray) -> Dict:
        """Calculate additional Awesome Oscillator metrics."""
        valid_values = ao[~np.isnan(ao)]
        
        if len(valid_values) == 0:
            return {}
        
        # Momentum analysis
        positive_pct = np.sum(valid_values > 0) / len(valid_values) * 100
        negative_pct = np.sum(valid_values < 0) / len(valid_values) * 100
        
        # Recent trend
        recent_values = valid_values[-min(5, len(valid_values)):]
        trend = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
        
        # Histogram analysis
        histogram_increasing = 0
        histogram_decreasing = 0
        
        for i in range(1, len(valid_values)):
            if valid_values[i] > valid_values[i-1]:
                histogram_increasing += 1
            elif valid_values[i] < valid_values[i-1]:
                histogram_decreasing += 1
        
        return {
            'current_value': ao[-1] if not np.isnan(ao[-1]) else None,
            'positive_momentum_pct': positive_pct,
            'negative_momentum_pct': negative_pct,
            'recent_trend': trend,
            'histogram_increasing_pct': histogram_increasing / len(valid_values) * 100 if len(valid_values) > 1 else 0,
            'histogram_decreasing_pct': histogram_decreasing / len(valid_values) * 100 if len(valid_values) > 1 else 0,
            'volatility': np.std(valid_values),
            'mean_value': np.mean(valid_values),
            'max_value': np.max(valid_values),
            'min_value': np.min(valid_values)
        }
    
    def _interpret_signals(self, current_ao: float, current_signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_ao):
            return "Insufficient data for Awesome Oscillator calculation"
        
        if current_ao > 0:
            momentum = "BULLISH"
        else:
            momentum = "BEARISH"
        
        signal_desc = {
            1: "BUY signal (zero line cross up)",
            0.5: "Weak BUY signal (saucer pattern)",
            -0.5: "Weak SELL signal (twin peaks)",
            -1: "SELL signal (zero line cross down)",
            0: "No signal"
        }.get(current_signal, "No signal")
        
        return f"Awesome Oscillator: {current_ao:.4f} ({momentum}) - {signal_desc}"


def create_awesome_oscillator(fast_period: int = 5, **kwargs) -> AwesomeOscillator:
    """Factory function to create Awesome Oscillator indicator."""
    return AwesomeOscillator(fast_period=fast_period, **kwargs)
