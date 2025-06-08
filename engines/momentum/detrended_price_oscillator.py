"""
Detrended Price Oscillator (DPO)
================================

The Detrended Price Oscillator is designed to remove the trend from price 
to make it easier to identify cycles. It's calculated by comparing price 
to its displaced moving average.

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


class DetrendedPriceOscillator(IndicatorBase):
    """
    Detrended Price Oscillator (DPO) indicator.
    
    DPO = Close - SMA(Close, period)[displaced by (period/2)+1]
    
    The DPO removes trend to help identify cycles in price action.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize Detrended Price Oscillator.
        
        Args:
            period: Look-back period for the moving average (typically 20)
        """
        super().__init__()
        
        self.period = period
        
        # Validation
        if period <= 0:
            raise ValueError("Period must be positive")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Detrended Price Oscillator values.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Dictionary containing DPO values and signals
        """
        try:
            # Validate input data
            required_columns = ['close']
            self._validate_data(data, required_columns)
            
            if len(data) < self.period:
                raise ValueError(f"Insufficient data: need at least {self.period} periods")
            
            close = data['close'].values
            
            # Calculate Detrended Price Oscillator
            dpo = self._calculate_dpo(close)
            
            # Generate signals
            signals = self._generate_signals(dpo)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(dpo)
            
            return {
                'dpo': dpo,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(dpo[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Detrended Price Oscillator: {e}")
            raise
    
    def _calculate_dpo(self, close: np.ndarray) -> np.ndarray:
        """Calculate Detrended Price Oscillator values."""
        dpo = np.full(len(close), np.nan)
        
        # Calculate SMA
        sma = self._sma(close, self.period)
        
        # Calculate displacement
        displacement = (self.period // 2) + 1
        
        # Calculate DPO
        for i in range(self.period - 1, len(close)):
            if i >= displacement and not np.isnan(sma[i - displacement]):
                dpo[i] = close[i] - sma[i - displacement]
        
        return dpo
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        return sma
    
    def _generate_signals(self, dpo: np.ndarray) -> np.ndarray:
        """Generate trading signals based on DPO."""
        signals = np.zeros(len(dpo))
        
        for i in range(1, len(dpo)):
            if np.isnan(dpo[i]) or np.isnan(dpo[i-1]):
                continue
            
            # Zero line crossovers
            if dpo[i-1] <= 0 and dpo[i] > 0:
                signals[i] = 1  # Buy signal
            elif dpo[i-1] >= 0 and dpo[i] < 0:
                signals[i] = -1  # Sell signal
            
            # Peak/trough signals
            elif i >= 2:
                if (not np.isnan(dpo[i-2]) and 
                    dpo[i-2] < dpo[i-1] and dpo[i-1] > dpo[i] and dpo[i-1] > 0):
                    signals[i] = -0.5  # Weak sell (peak)
                elif (not np.isnan(dpo[i-2]) and 
                      dpo[i-2] > dpo[i-1] and dpo[i-1] < dpo[i] and dpo[i-1] < 0):
                    signals[i] = 0.5  # Weak buy (trough)
        
        return signals
    
    def _calculate_metrics(self, dpo: np.ndarray) -> Dict:
        """Calculate additional DPO metrics."""
        valid_values = dpo[~np.isnan(dpo)]
        
        if len(valid_values) == 0:
            return {}
        
        # Cycle analysis
        positive_pct = np.sum(valid_values > 0) / len(valid_values) * 100
        negative_pct = np.sum(valid_values < 0) / len(valid_values) * 100
        
        # Cycle peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(1, len(valid_values) - 1):
            if valid_values[i] > valid_values[i-1] and valid_values[i] > valid_values[i+1]:
                peaks.append(valid_values[i])
            elif valid_values[i] < valid_values[i-1] and valid_values[i] < valid_values[i+1]:
                troughs.append(valid_values[i])
        
        # Recent trend
        recent_values = valid_values[-min(5, len(valid_values)):]
        trend = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
        
        return {
            'current_value': dpo[-1] if not np.isnan(dpo[-1]) else None,
            'positive_cycle_pct': positive_pct,
            'negative_cycle_pct': negative_pct,
            'recent_trend': trend,
            'volatility': np.std(valid_values),
            'mean_value': np.mean(valid_values),
            'max_value': np.max(valid_values),
            'min_value': np.min(valid_values),
            'cycle_peaks_count': len(peaks),
            'cycle_troughs_count': len(troughs),
            'avg_peak_value': np.mean(peaks) if peaks else None,
            'avg_trough_value': np.mean(troughs) if troughs else None
        }
    
    def _interpret_signals(self, current_dpo: float, current_signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_dpo):
            return "Insufficient data for DPO calculation"
        
        if current_dpo > 0:
            cycle_position = "above trend (potential cycle high)"
        else:
            cycle_position = "below trend (potential cycle low)"
        
        signal_desc = {
            1: "BUY signal (zero line cross up)",
            0.5: "Weak BUY signal (cycle trough)",
            -0.5: "Weak SELL signal (cycle peak)",
            -1: "SELL signal (zero line cross down)",
            0: "No signal"
        }.get(current_signal, "No signal")
        
        return f"DPO: {current_dpo:.4f} ({cycle_position}) - {signal_desc}"


def create_detrended_price_oscillator(period: int = 20, **kwargs) -> DetrendedPriceOscillator:
    """Factory function to create Detrended Price Oscillator indicator."""
    return DetrendedPriceOscillator(period=period, **kwargs)
