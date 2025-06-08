"""
Percentage Price Oscillator (PPO)
=================================

PPO is similar to MACD but shows the percentage difference between two EMAs
rather than the absolute difference, making it useful for comparing securities
with different price levels.

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


class PercentagePriceOscillator(IndicatorBase):
    """
    Percentage Price Oscillator (PPO) indicator.
    
    PPO shows the percentage difference between two exponential moving averages,
    making it useful for comparing different securities or timeframes.
    """
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9):
        """
        Initialize PPO indicator.
        
        Args:
            fast_period: Fast EMA period (typically 12)
            slow_period: Slow EMA period (typically 26)
            signal_period: Signal line EMA period (typically 9)
        """
        super().__init__()
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Validation
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("All periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate PPO values.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Dictionary containing PPO values and signals
        """
        try:
            # Validate input data
            required_columns = ['close']
            self._validate_data(data, required_columns)
            
            min_periods = max(self.slow_period, self.signal_period) + 10
            if len(data) < min_periods:
                raise ValueError(f"Insufficient data: need at least {min_periods} periods")
            
            close = data['close'].values
            
            # Calculate PPO
            ppo, signal_line, histogram = self._calculate_ppo(close)
            
            # Generate signals
            signals = self._generate_signals(ppo, signal_line, histogram)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(ppo, signal_line, histogram)
            
            return {
                'ppo': ppo,
                'signal_line': signal_line,
                'histogram': histogram,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(ppo[-1], signal_line[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating PPO: {e}")
            raise
    
    def _calculate_ppo(self, close: np.ndarray) -> tuple:
        """Calculate PPO, signal line, and histogram."""
        # Calculate EMAs
        fast_ema = self._ema(close, self.fast_period)
        slow_ema = self._ema(close, self.slow_period)
        
        # Calculate PPO
        ppo = np.full(len(close), np.nan)
        for i in range(len(close)):
            if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]) and slow_ema[i] != 0:
                ppo[i] = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100
        
        # Calculate signal line
        signal_line = self._ema(ppo, self.signal_period)
        
        # Calculate histogram
        histogram = ppo - signal_line
        
        return ppo, signal_line, histogram
    
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
    
    def _generate_signals(self, ppo: np.ndarray, signal_line: np.ndarray, histogram: np.ndarray) -> np.ndarray:
        """Generate trading signals based on PPO."""
        signals = np.zeros(len(ppo))
        
        for i in range(1, len(ppo)):
            if (np.isnan(ppo[i]) or np.isnan(signal_line[i]) or 
                np.isnan(ppo[i-1]) or np.isnan(signal_line[i-1])):
                continue
            
            # PPO crosses above signal line
            if ppo[i-1] <= signal_line[i-1] and ppo[i] > signal_line[i]:
                signals[i] = 1
            
            # PPO crosses below signal line
            elif ppo[i-1] >= signal_line[i-1] and ppo[i] < signal_line[i]:
                signals[i] = -1
            
            # Zero line crossovers
            elif ppo[i-1] <= 0 and ppo[i] > 0:
                signals[i] = 0.5  # Weak buy
            elif ppo[i-1] >= 0 and ppo[i] < 0:
                signals[i] = -0.5  # Weak sell
        
        return signals
    
    def _calculate_metrics(self, ppo: np.ndarray, signal_line: np.ndarray, histogram: np.ndarray) -> Dict:
        """Calculate additional PPO metrics."""
        valid_ppo = ppo[~np.isnan(ppo)]
        valid_histogram = histogram[~np.isnan(histogram)]
        
        if len(valid_ppo) == 0:
            return {}
        
        # Momentum analysis
        positive_momentum = np.sum(valid_ppo > 0) / len(valid_ppo) * 100
        above_signal = np.sum(ppo > signal_line) / len(valid_ppo) * 100
        
        # Histogram trend
        if len(valid_histogram) > 1:
            histogram_trend = np.mean(np.diff(valid_histogram[-min(5, len(valid_histogram)):]))
        else:
            histogram_trend = 0
        
        # Current divergence
        current_divergence = 0
        if not np.isnan(ppo[-1]) and not np.isnan(signal_line[-1]):
            current_divergence = ppo[-1] - signal_line[-1]
        
        return {
            'current_ppo': ppo[-1] if not np.isnan(ppo[-1]) else None,
            'current_signal': signal_line[-1] if not np.isnan(signal_line[-1]) else None,
            'current_histogram': histogram[-1] if not np.isnan(histogram[-1]) else None,
            'positive_momentum_pct': positive_momentum,
            'above_signal_pct': above_signal,
            'histogram_trend': histogram_trend,
            'current_divergence': current_divergence,
            'ppo_volatility': np.std(valid_ppo),
            'mean_ppo': np.mean(valid_ppo)
        }
    
    def _interpret_signals(self, current_ppo: float, current_signal: float, signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_ppo):
            return "Insufficient data for PPO calculation"
        
        if current_ppo > current_signal:
            momentum = "BULLISH"
        else:
            momentum = "BEARISH"
        
        if current_ppo > 0:
            trend = "POSITIVE"
        else:
            trend = "NEGATIVE"
        
        signal_desc = {
            1: "BUY signal (PPO crosses above signal)",
            0.5: "Weak BUY (zero line cross)",
            -0.5: "Weak SELL (zero line cross)",
            -1: "SELL signal (PPO crosses below signal)",
            0: "No signal"
        }.get(signal, "No signal")
        
        return f"PPO: {current_ppo:.2f}% ({trend}, {momentum}) - {signal_desc}"


def create_ppo(fast_period: int = 12, **kwargs) -> PercentagePriceOscillator:
    """Factory function to create PPO indicator."""
    return PercentagePriceOscillator(fast_period=fast_period, **kwargs)
