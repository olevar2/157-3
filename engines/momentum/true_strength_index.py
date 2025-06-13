"""
True Strength Index (TSI)
=========================

TSI is a momentum oscillator that uses price changes smoothed by two 
exponential moving averages to filter out noise and provide clearer signals.

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


class TrueStrengthIndex(IndicatorBase):
    """
    True Strength Index (TSI) indicator.
    
    TSI is a momentum oscillator that double-smooths price changes to reduce 
    noise and provide more reliable signals than traditional momentum indicators.
    """
    
    def __init__(self, 
                 first_smoothing: int = 25,
                 second_smoothing: int = 13,
                 signal_smoothing: int = 7):
        """
        Initialize TSI indicator.
        
        Args:
            first_smoothing: First smoothing period
            second_smoothing: Second smoothing period  
            signal_smoothing: Signal line smoothing period
        """
        super().__init__()
        
        self.first_smoothing = first_smoothing
        self.second_smoothing = second_smoothing
        self.signal_smoothing = signal_smoothing
        
        # Validation
        if first_smoothing <= 0 or second_smoothing <= 0 or signal_smoothing <= 0:
            raise ValueError("All smoothing periods must be positive")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate TSI values.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Dictionary containing TSI values and signals
        """
        try:
            # Validate input data
            required_columns = ['close']
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError("Empty DataFrame provided")
                
                # Check required columns
                missing = [col for col in required_columns if col not in data.columns]
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")
            else:
                validation_result = super()._validate_data(data)
                if not validation_result:
                    raise ValueError("Invalid input data format")
            
            min_periods = self.first_smoothing + self.second_smoothing + 1
            if len(data) < min_periods:
                raise ValueError(f"Insufficient data: need at least {min_periods} periods")
            
            close = data['close'].values
            
            # Calculate TSI
            tsi, signal_line = self._calculate_tsi(close)
            
            # Calculate histogram
            histogram = tsi - signal_line
            
            # Generate signals
            signals = self._generate_signals(tsi, signal_line, histogram)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(tsi, signal_line, histogram)
            
            return {
                'tsi': tsi,
                'signal_line': signal_line,
                'histogram': histogram,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(tsi[-1], signal_line[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating TSI: {e}")
            raise
    
    def _calculate_tsi(self, close: np.ndarray) -> tuple:
        """Calculate TSI and signal line."""
        # Calculate price changes
        price_changes = np.diff(close)
        abs_price_changes = np.abs(price_changes)
        
        # First smoothing
        first_smooth_pc = self._ema(price_changes, self.first_smoothing)
        first_smooth_abs_pc = self._ema(abs_price_changes, self.first_smoothing)
        
        # Second smoothing
        second_smooth_pc = self._ema(first_smooth_pc, self.second_smoothing)
        second_smooth_abs_pc = self._ema(first_smooth_abs_pc, self.second_smoothing)
        
        # Calculate TSI
        tsi = np.full(len(close), np.nan)
        tsi[1:] = np.where(second_smooth_abs_pc != 0, 
                          100 * (second_smooth_pc / second_smooth_abs_pc), 
                          0)
        
        # Calculate signal line
        signal_line = self._ema(tsi[~np.isnan(tsi)], self.signal_smoothing)
        
        # Align signal line with TSI
        signal_aligned = np.full(len(close), np.nan)
        valid_indices = ~np.isnan(tsi)
        signal_aligned[valid_indices] = signal_line
        
        return tsi, signal_aligned
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2 / (period + 1)
        ema = np.full(len(data), np.nan)
        
        # Initialize with first valid value
        for i, val in enumerate(data):
            if not np.isnan(val):
                ema[i] = val
                break
        
        # Calculate EMA
        for i in range(1, len(data)):
            if not np.isnan(data[i]) and not np.isnan(ema[i-1]):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _generate_signals(self, tsi: np.ndarray, signal_line: np.ndarray, histogram: np.ndarray) -> np.ndarray:
        """Generate trading signals based on TSI."""
        signals = np.zeros(len(tsi))
        
        for i in range(1, len(tsi)):
            if (np.isnan(tsi[i]) or np.isnan(signal_line[i]) or 
                np.isnan(tsi[i-1]) or np.isnan(signal_line[i-1])):
                continue
            
            # TSI crosses above signal line
            if tsi[i-1] <= signal_line[i-1] and tsi[i] > signal_line[i]:
                signals[i] = 1
            
            # TSI crosses below signal line
            elif tsi[i-1] >= signal_line[i-1] and tsi[i] < signal_line[i]:
                signals[i] = -1
            
            # Zero line crossovers
            elif tsi[i-1] <= 0 and tsi[i] > 0:
                signals[i] = 0.5  # Weak buy
            elif tsi[i-1] >= 0 and tsi[i] < 0:
                signals[i] = -0.5  # Weak sell
        
        return signals
    
    def _calculate_metrics(self, tsi: np.ndarray, signal_line: np.ndarray, histogram: np.ndarray) -> Dict:
        """Calculate additional TSI metrics."""
        valid_tsi = tsi[~np.isnan(tsi)]
        valid_histogram = histogram[~np.isnan(histogram)]
        
        if len(valid_tsi) == 0:
            return {}
        
        # Momentum analysis
        positive_momentum = np.sum(valid_tsi > 0) / len(valid_tsi) * 100
        negative_momentum = np.sum(valid_tsi < 0) / len(valid_tsi) * 100
        
        # Signal line relationship
        above_signal = np.sum(tsi > signal_line) / len(valid_tsi) * 100
        
        # Histogram analysis
        if len(valid_histogram) > 0:
            histogram_trend = np.mean(np.diff(valid_histogram[-min(5, len(valid_histogram)):]))
        else:
            histogram_trend = 0
        
        return {
            'current_tsi': tsi[-1] if not np.isnan(tsi[-1]) else None,
            'current_signal': signal_line[-1] if not np.isnan(signal_line[-1]) else None,
            'current_histogram': histogram[-1] if not np.isnan(histogram[-1]) else None,
            'positive_momentum_pct': positive_momentum,
            'negative_momentum_pct': negative_momentum,
            'above_signal_pct': above_signal,
            'histogram_trend': histogram_trend,
            'tsi_volatility': np.std(valid_tsi),
            'mean_tsi': np.mean(valid_tsi)
        }
    
    def _interpret_signals(self, current_tsi: float, current_signal: float, signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_tsi):
            return "Insufficient data for TSI calculation"
        
        if current_tsi > current_signal:
            momentum = "BULLISH"
        else:
            momentum = "BEARISH"
        
        if current_tsi > 25:
            strength = "STRONG POSITIVE"
        elif current_tsi > 0:
            strength = "POSITIVE"
        elif current_tsi < -25:
            strength = "STRONG NEGATIVE"
        else:
            strength = "NEGATIVE"
        
        signal_desc = {
            1: "BUY signal (TSI crosses above signal)",
            0.5: "Weak BUY (zero line cross)",
            -0.5: "Weak SELL (zero line cross)",
            -1: "SELL signal (TSI crosses below signal)",
            0: "No signal"
        }.get(signal, "No signal")
        
        return f"TSI: {current_tsi:.2f} ({strength}, {momentum}) - {signal_desc}"


def create_tsi(first_smoothing: int = 25, **kwargs) -> TrueStrengthIndex:
    """Factory function to create TSI indicator."""
    return TrueStrengthIndex(first_smoothing=first_smoothing, **kwargs)
