"""
Ultimate Oscillator
===================

The Ultimate Oscillator uses weighted sums of three different time periods 
to reduce the volatility and false signals associated with single-period oscillators.

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


class UltimateOscillator(IndicatorBase):
    """
    Ultimate Oscillator indicator.
    
    The Ultimate Oscillator combines three different timeframes to provide 
    a more reliable momentum reading and reduce false signals.
    """
    
    def __init__(self, 
                 fast_period: int = 7,
                 medium_period: int = 14,
                 slow_period: int = 28,
                 fast_weight: float = 4.0,
                 medium_weight: float = 2.0,
                 slow_weight: float = 1.0,
                 overbought_level: float = 70.0,
                 oversold_level: float = 30.0):
        """
        Initialize Ultimate Oscillator.
        
        Args:
            fast_period: Fast period (typically 7)
            medium_period: Medium period (typically 14)
            slow_period: Slow period (typically 28)
            fast_weight: Weight for fast period
            medium_weight: Weight for medium period
            slow_weight: Weight for slow period
            overbought_level: Overbought threshold
            oversold_level: Oversold threshold
        """
        super().__init__()
        
        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period
        self.fast_weight = fast_weight
        self.medium_weight = medium_weight
        self.slow_weight = slow_weight
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        
        # Validation
        if not (fast_period < medium_period < slow_period):
            raise ValueError("Periods must be in ascending order: fast < medium < slow")
        if any(w <= 0 for w in [fast_weight, medium_weight, slow_weight]):
            raise ValueError("All weights must be positive")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Ultimate Oscillator values.
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Dictionary containing Ultimate Oscillator values and signals
        """
        try:
            # Validate input data
            required_columns = ['high', 'low', 'close']
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
            
            if len(data) < self.slow_period + 1:
                raise ValueError(f"Insufficient data: need at least {self.slow_period + 1} periods")
            
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calculate Ultimate Oscillator
            uo = self._calculate_ultimate_oscillator(high, low, close)
            
            # Generate signals
            signals = self._generate_signals(uo)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(uo)
            
            return {
                'ultimate_oscillator': uo,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(uo[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Ultimate Oscillator: {e}")
            raise
    
    def _calculate_ultimate_oscillator(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Ultimate Oscillator values."""
        uo = np.full(len(close), np.nan)
        
        # Calculate buying pressure (BP) and true range (TR)
        bp = close - np.minimum(low, np.roll(close, 1))
        tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
        
        # Handle first element (no previous close)
        bp[0] = close[0] - low[0]
        tr[0] = high[0] - low[0]
        
        for i in range(self.slow_period - 1, len(close)):
            # Calculate sums for each period
            bp_fast = np.sum(bp[i - self.fast_period + 1:i + 1])
            tr_fast = np.sum(tr[i - self.fast_period + 1:i + 1])
            
            bp_medium = np.sum(bp[i - self.medium_period + 1:i + 1])
            tr_medium = np.sum(tr[i - self.medium_period + 1:i + 1])
            
            bp_slow = np.sum(bp[i - self.slow_period + 1:i + 1])
            tr_slow = np.sum(tr[i - self.slow_period + 1:i + 1])
            
            # Calculate raw values for each period
            raw_fast = (bp_fast / tr_fast) * 100 if tr_fast != 0 else 0
            raw_medium = (bp_medium / tr_medium) * 100 if tr_medium != 0 else 0
            raw_slow = (bp_slow / tr_slow) * 100 if tr_slow != 0 else 0
            
            # Calculate weighted Ultimate Oscillator
            total_weight = self.fast_weight + self.medium_weight + self.slow_weight
            uo[i] = ((raw_fast * self.fast_weight + 
                     raw_medium * self.medium_weight + 
                     raw_slow * self.slow_weight) / total_weight)
        
        return uo
    
    def _generate_signals(self, uo: np.ndarray) -> np.ndarray:
        """Generate trading signals based on Ultimate Oscillator."""
        signals = np.zeros(len(uo))
        
        for i in range(1, len(uo)):
            if np.isnan(uo[i]) or np.isnan(uo[i-1]):
                continue
            
            # Oversold bounce signal
            if (uo[i-1] <= self.oversold_level and 
                uo[i] > self.oversold_level):
                signals[i] = 1
            
            # Overbought reversal signal
            elif (uo[i-1] >= self.overbought_level and 
                  uo[i] < self.overbought_level):
                signals[i] = -1
            
            # Bullish divergence (simplified)
            elif uo[i] > 50 and uo[i-1] <= 50:
                signals[i] = 0.5  # Weak buy
            
            # Bearish divergence (simplified)
            elif uo[i] < 50 and uo[i-1] >= 50:
                signals[i] = -0.5  # Weak sell
        
        return signals
    
    def _calculate_metrics(self, uo: np.ndarray) -> Dict:
        """Calculate additional Ultimate Oscillator metrics."""
        valid_values = uo[~np.isnan(uo)]
        
        if len(valid_values) == 0:
            return {}
        
        # Time in different zones
        overbought_pct = np.sum(valid_values >= self.overbought_level) / len(valid_values) * 100
        oversold_pct = np.sum(valid_values <= self.oversold_level) / len(valid_values) * 100
        neutral_pct = 100 - overbought_pct - oversold_pct
        
        # Momentum analysis
        above_50_pct = np.sum(valid_values > 50) / len(valid_values) * 100
        
        # Current momentum
        recent_values = valid_values[-min(5, len(valid_values)):]
        momentum = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
        
        return {
            'current_value': uo[-1] if not np.isnan(uo[-1]) else None,
            'overbought_percentage': overbought_pct,
            'oversold_percentage': oversold_pct,
            'neutral_percentage': neutral_pct,
            'above_50_percentage': above_50_pct,
            'momentum': momentum,
            'volatility': np.std(valid_values),
            'mean_value': np.mean(valid_values),
            'max_value': np.max(valid_values),
            'min_value': np.min(valid_values)
        }
    
    def _interpret_signals(self, current_uo: float, current_signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_uo):
            return "Insufficient data for Ultimate Oscillator calculation"
        
        if current_uo >= self.overbought_level:
            condition = "OVERBOUGHT"
        elif current_uo <= self.oversold_level:
            condition = "OVERSOLD"
        elif current_uo > 50:
            condition = "BULLISH"
        else:
            condition = "BEARISH"
        
        signal_desc = {
            1: "BUY signal (oversold bounce)",
            0.5: "Weak BUY signal",
            -0.5: "Weak SELL signal",
            -1: "SELL signal (overbought reversal)",
            0: "No signal"
        }.get(current_signal, "No signal")
        
        return f"Ultimate Oscillator: {current_uo:.2f} ({condition}) - {signal_desc}"


def create_ultimate_oscillator(fast_period: int = 7, **kwargs) -> UltimateOscillator:
    """Factory function to create Ultimate Oscillator indicator."""
    return UltimateOscillator(fast_period=fast_period, **kwargs)
