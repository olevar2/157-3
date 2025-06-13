"""
Chande Momentum Oscillator (CMO)
================================

The Chande Momentum Oscillator is a technical momentum indicator that measures 
the difference between the sum of recent gains and the sum of recent losses, 
divided by the sum of all price movement over the same period.

Formula: CMO = 100 * ((Su - Sd) / (Su + Sd))
Where Su = sum of up days, Sd = sum of down days

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


class ChandeMomentumOscillator(IndicatorBase):
    """
    Chande Momentum Oscillator (CMO) indicator.
    
    CMO oscillates between +100 and -100, with values above +50 indicating 
    overbought conditions and values below -50 indicating oversold conditions.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize Chande Momentum Oscillator.
        
        Args:
            period: Look-back period for calculation (typically 14)
        """
        super().__init__()
        
        self.period = period
        
        # Validation
        if period <= 0:
            raise ValueError("Period must be positive")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Chande Momentum Oscillator values.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Dictionary containing CMO values and signals
        """
        try:
            # Validate input data
            required_columns = ['close']
            self._validate_data(data, required_columns)
            
            if len(data) < self.period + 1:
                raise ValueError(f"Insufficient data: need at least {self.period + 1} periods")
            
            close = data['close'].values
            
            # Calculate Chande Momentum Oscillator
            cmo = self._calculate_cmo(close)
            
            # Generate signals
            signals = self._generate_signals(cmo)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(cmo)
            
            return {
                'cmo': cmo,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(cmo[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Chande Momentum Oscillator: {e}")
            raise
    
    def _calculate_cmo(self, close: np.ndarray) -> np.ndarray:
        """Calculate Chande Momentum Oscillator values."""
        cmo = np.full(len(close), np.nan)
        
        # Calculate price changes
        price_changes = np.diff(close)
        
        for i in range(self.period, len(close)):
            period_changes = price_changes[i - self.period:i]
            
            # Sum of gains and losses
            gains = period_changes[period_changes > 0]
            losses = np.abs(period_changes[period_changes < 0])
            
            sum_gains = np.sum(gains) if len(gains) > 0 else 0
            sum_losses = np.sum(losses) if len(losses) > 0 else 0
            
            # Calculate CMO
            total_movement = sum_gains + sum_losses
            if total_movement > 0:
                cmo[i] = 100 * (sum_gains - sum_losses) / total_movement
            else:
                cmo[i] = 0
        
        return cmo
    
    def _generate_signals(self, cmo: np.ndarray) -> np.ndarray:
        """Generate trading signals based on CMO."""
        signals = np.zeros(len(cmo))
        
        for i in range(1, len(cmo)):
            if np.isnan(cmo[i]) or np.isnan(cmo[i-1]):
                continue
            
            # Overbought/Oversold signals
            if cmo[i-1] <= -50 and cmo[i] > -50:
                signals[i] = 1  # Buy signal (exit oversold)
            elif cmo[i-1] >= 50 and cmo[i] < 50:
                signals[i] = -1  # Sell signal (exit overbought)
            
            # Zero line crossovers
            elif cmo[i-1] <= 0 and cmo[i] > 0:
                signals[i] = 0.5  # Weak buy signal
            elif cmo[i-1] >= 0 and cmo[i] < 0:
                signals[i] = -0.5  # Weak sell signal
            
            # Extreme readings
            elif cmo[i] > 75:
                signals[i] = -0.3  # Extreme overbought
            elif cmo[i] < -75:
                signals[i] = 0.3  # Extreme oversold
        
        return signals
    
    def _calculate_metrics(self, cmo: np.ndarray) -> Dict:
        """Calculate additional CMO metrics."""
        valid_values = cmo[~np.isnan(cmo)]
        
        if len(valid_values) == 0:
            return {}
        
        # Momentum conditions
        overbought_pct = np.sum(valid_values > 50) / len(valid_values) * 100
        oversold_pct = np.sum(valid_values < -50) / len(valid_values) * 100
        extreme_overbought_pct = np.sum(valid_values > 75) / len(valid_values) * 100
        extreme_oversold_pct = np.sum(valid_values < -75) / len(valid_values) * 100
        
        # Trend analysis
        positive_momentum_pct = np.sum(valid_values > 0) / len(valid_values) * 100
        negative_momentum_pct = np.sum(valid_values < 0) / len(valid_values) * 100
        
        # Recent trend
        recent_values = valid_values[-min(5, len(valid_values)):]
        trend = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
        
        # Oscillation analysis
        peak_count = 0
        trough_count = 0
        
        for i in range(1, len(valid_values) - 1):
            if (valid_values[i] > valid_values[i-1] and 
                valid_values[i] > valid_values[i+1] and 
                valid_values[i] > 25):
                peak_count += 1
            elif (valid_values[i] < valid_values[i-1] and 
                  valid_values[i] < valid_values[i+1] and 
                  valid_values[i] < -25):
                trough_count += 1
        
        return {
            'current_value': cmo[-1] if not np.isnan(cmo[-1]) else None,
            'overbought_pct': overbought_pct,
            'oversold_pct': oversold_pct,
            'extreme_overbought_pct': extreme_overbought_pct,
            'extreme_oversold_pct': extreme_oversold_pct,
            'positive_momentum_pct': positive_momentum_pct,
            'negative_momentum_pct': negative_momentum_pct,
            'recent_trend': trend,
            'volatility': np.std(valid_values),
            'mean_value': np.mean(valid_values),
            'max_value': np.max(valid_values),
            'min_value': np.min(valid_values),
            'peak_count': peak_count,
            'trough_count': trough_count,
            'momentum_range': np.max(valid_values) - np.min(valid_values)
        }
    
    def _interpret_signals(self, current_cmo: float, current_signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_cmo):
            return "Insufficient data for CMO calculation"
        
        if current_cmo > 75:
            momentum_desc = "EXTREME OVERBOUGHT"
        elif current_cmo > 50:
            momentum_desc = "OVERBOUGHT"
        elif current_cmo > 0:
            momentum_desc = "Bullish momentum"
        elif current_cmo > -50:
            momentum_desc = "Bearish momentum"
        elif current_cmo > -75:
            momentum_desc = "OVERSOLD"
        else:
            momentum_desc = "EXTREME OVERSOLD"
        
        signal_desc = {
            1: "BUY signal (exit oversold)",
            0.5: "Weak BUY signal (zero line cross up)",
            0.3: "Potential BUY (extreme oversold)",
            -0.3: "Potential SELL (extreme overbought)",
            -0.5: "Weak SELL signal (zero line cross down)",
            -1: "SELL signal (exit overbought)",
            0: "No signal"
        }.get(current_signal, "No signal")
        
        return f"CMO: {current_cmo:.2f} ({momentum_desc}) - {signal_desc}"


def create_chande_momentum_oscillator(period: int = 14, **kwargs) -> ChandeMomentumOscillator:
    """Factory function to create Chande Momentum Oscillator indicator."""
    return ChandeMomentumOscillator(period=period, **kwargs)
