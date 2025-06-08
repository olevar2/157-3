"""
TRIX (Triple Exponential Average) 
=================================

The TRIX indicator is a momentum oscillator that uses a triple exponential 
moving average to filter out price noise and identify trend changes.

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
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import IndicatorBase


class TRIX(IndicatorBase):
    """
    TRIX (Triple Exponential Average) indicator.
    
    TRIX is calculated by taking the rate of change (1-period percent change) 
    of a triple exponentially smoothed moving average.
    """
    
    def __init__(self, 
                 period: int = 14,
                 signal_period: int = 9):
        """
        Initialize TRIX indicator.
        
        Args:
            period: Period for triple exponential smoothing (typically 14)
            signal_period: Period for signal line EMA (typically 9)
        """
        super().__init__(name="TRIX")
        
        self.period = period
        self.signal_period = signal_period
        
        # Validation
        if period <= 0 or signal_period <= 0:
            raise ValueError("Periods must be positive")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate TRIX values.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Dictionary containing TRIX values and signals
        """
        try:
            # Validate input data
            required_columns = ['close']
            self._validate_data(data, required_columns)
            
            if len(data) < self.period * 3:
                raise ValueError(f"Insufficient data: need at least {self.period * 3} periods")
            
            close = data['close'].values
            
            # Calculate TRIX
            trix, signal = self._calculate_trix(close)
            
            # Generate signals
            signals = self._generate_signals(trix, signal)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(trix, signal)
            
            return {
                'trix': trix,
                'signal': signal,
                'histogram': trix - signal,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(trix[-1], signal[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating TRIX: {e}")
            raise
    
    def _calculate_trix(self, close: np.ndarray) -> tuple:
        """Calculate TRIX and signal line."""
        # First EMA
        ema1 = self._ema(close, self.period)
        
        # Second EMA
        ema2 = self._ema(ema1, self.period)
        
        # Third EMA
        ema3 = self._ema(ema2, self.period)
        
        # Calculate TRIX (rate of change of triple EMA)
        trix = np.full(len(close), np.nan)
        
        for i in range(1, len(ema3)):
            if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
                trix[i] = ((ema3[i] - ema3[i-1]) / ema3[i-1]) * 10000  # Convert to basis points
        
        # Calculate signal line (EMA of TRIX)
        signal = self._ema(trix, self.signal_period)
        
        return trix, signal
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.full(len(data), np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid value
        start_idx = 0
        while start_idx < len(data) and np.isnan(data[start_idx]):
            start_idx += 1
        
        if start_idx >= len(data):
            return ema
        
        ema[start_idx] = data[start_idx]
        
        for i in range(start_idx + 1, len(data)):
            if not np.isnan(data[i]):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            else:
                ema[i] = ema[i-1]
        
        return ema
    
    def _generate_signals(self, trix: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Generate trading signals based on TRIX."""
        signals = np.zeros(len(trix))
        
        for i in range(1, len(trix)):
            if np.isnan(trix[i]) or np.isnan(signal[i]) or np.isnan(trix[i-1]) or np.isnan(signal[i-1]):
                continue
            
            # Zero line crossovers
            if trix[i-1] <= 0 and trix[i] > 0:
                signals[i] = 1  # Buy signal
            elif trix[i-1] >= 0 and trix[i] < 0:
                signals[i] = -1  # Sell signal
            
            # Signal line crossovers
            elif trix[i-1] <= signal[i-1] and trix[i] > signal[i]:
                signals[i] = 0.5  # Weak buy
            elif trix[i-1] >= signal[i-1] and trix[i] < signal[i]:
                signals[i] = -0.5  # Weak sell
        
        return signals
    
    def _calculate_metrics(self, trix: np.ndarray, signal: np.ndarray) -> Dict:
        """Calculate additional TRIX metrics."""
        valid_trix = trix[~np.isnan(trix)]
        valid_signal = signal[~np.isnan(signal)]
        
        if len(valid_trix) == 0:
            return {}
        
        # Momentum analysis
        positive_pct = np.sum(valid_trix > 0) / len(valid_trix) * 100
        negative_pct = np.sum(valid_trix < 0) / len(valid_trix) * 100
        
        # Histogram analysis
        histogram = trix - signal
        valid_histogram = histogram[~np.isnan(histogram)]
        
        positive_histogram_pct = np.sum(valid_histogram > 0) / len(valid_histogram) * 100 if len(valid_histogram) > 0 else 0
        
        # Recent trend
        recent_trix = valid_trix[-min(5, len(valid_trix)):]
        trend = np.mean(np.diff(recent_trix)) if len(recent_trix) > 1 else 0
        
        # Volatility
        trix_volatility = np.std(valid_trix)
        
        return {
            'current_trix': trix[-1] if not np.isnan(trix[-1]) else None,
            'current_signal': signal[-1] if not np.isnan(signal[-1]) else None,
            'positive_momentum_pct': positive_pct,
            'negative_momentum_pct': negative_pct,
            'positive_histogram_pct': positive_histogram_pct,
            'recent_trend': trend,
            'volatility': trix_volatility,
            'mean_trix': np.mean(valid_trix),
            'max_trix': np.max(valid_trix),
            'min_trix': np.min(valid_trix),
            'signal_strength': abs(trix[-1] - signal[-1]) if not np.isnan(trix[-1]) and not np.isnan(signal[-1]) else 0
        }
    
    def _interpret_signals(self, current_trix: float, current_signal: float, current_signal_value: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_trix) or np.isnan(current_signal):
            return "Insufficient data for TRIX calculation"
        
        momentum = "BULLISH" if current_trix > 0 else "BEARISH"
        histogram = "above" if current_trix > current_signal else "below"
        
        signal_desc = {
            1: "BUY signal (zero line cross up)",
            0.5: "Weak BUY signal (signal line cross up)",
            -0.5: "Weak SELL signal (signal line cross down)",
            -1: "SELL signal (zero line cross down)",
            0: "No signal"
        }.get(current_signal_value, "No signal")
        
        return f"TRIX: {current_trix:.4f} ({momentum}, {histogram} signal line) - {signal_desc}"


def create_trix(period: int = 14, **kwargs) -> TRIX:
    """Factory function to create TRIX indicator."""
    return TRIX(period=period, **kwargs)
