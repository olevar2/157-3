"""
Momentum Indicator
==================

The Momentum indicator is one of the simplest technical indicators, measuring 
the rate of change in price over a specified period. It's calculated as the 
current price divided by the price n periods ago.

Formula: Momentum = (Current Price / Price n periods ago) * 100
Or: Momentum = Current Price - Price n periods ago

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


class MomentumIndicator(IndicatorBase):
    """
    Momentum Indicator.
    
    Measures the rate of change in price over a specified period. 
    Values above 100 (or 0 for difference method) indicate upward momentum,
    while values below 100 (or 0) indicate downward momentum.
    """
    
    def __init__(self, 
                 period: int = 14,
                 method: str = "ratio"):
        """
        Initialize Momentum Indicator.
        
        Args:
            period: Look-back period for momentum calculation
            method: "ratio" (price/past_price*100) or "difference" (price-past_price)
        """
        super().__init__()
        
        self.period = period
        self.method = method
        
        # Validation
        if period <= 0:
            raise ValueError("Period must be positive")
        if method not in ["ratio", "difference"]:
            raise ValueError("Method must be 'ratio' or 'difference'")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Momentum Indicator values.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Dictionary containing Momentum values and signals
        """
        try:            # Validate input data
            required_columns = ['close']
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError("Empty DataFrame provided")
                
                # Check required columns
                missing = [col for col in required_columns if col not in data.columns]
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")            else:
                validation_result = super()._validate_data(data)
                if not validation_result:
                    raise ValueError("Invalid input data format")
            
            if len(data) < self.period + 1:
                raise ValueError(f"Insufficient data: need at least {self.period + 1} periods")
            
            close = data['close'].values
            
            # Calculate Momentum
            momentum = self._calculate_momentum(close)
            
            # Generate signals
            signals = self._generate_signals(momentum)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(momentum)
            
            return {
                'momentum': momentum,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(momentum[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Momentum Indicator: {e}")
            raise
    
    def _calculate_momentum(self, close: np.ndarray) -> np.ndarray:
        """Calculate Momentum values."""
        momentum = np.full(len(close), np.nan)
        
        for i in range(self.period, len(close)):
            if self.method == "ratio":
                if close[i - self.period] != 0:
                    momentum[i] = (close[i] / close[i - self.period]) * 100
            else:  # difference
                momentum[i] = close[i] - close[i - self.period]
        
        return momentum
    
    def _generate_signals(self, momentum: np.ndarray) -> np.ndarray:
        """Generate trading signals based on Momentum."""
        signals = np.zeros(len(momentum))
        
        # Determine neutral line based on method
        neutral_line = 100 if self.method == "ratio" else 0
        
        for i in range(1, len(momentum)):
            if np.isnan(momentum[i]) or np.isnan(momentum[i-1]):
                continue
            
            # Neutral line crossovers
            if momentum[i-1] <= neutral_line and momentum[i] > neutral_line:
                signals[i] = 1  # Buy signal
            elif momentum[i-1] >= neutral_line and momentum[i] < neutral_line:
                signals[i] = -1  # Sell signal
            
            # Trend continuation signals
            elif momentum[i] > neutral_line and momentum[i] > momentum[i-1]:
                signals[i] = 0.5  # Weak buy (upward momentum increasing)
            elif momentum[i] < neutral_line and momentum[i] < momentum[i-1]:
                signals[i] = -0.5  # Weak sell (downward momentum increasing)
            
            # Momentum exhaustion signals
            elif i >= 2:
                if (momentum[i-2] < momentum[i-1] and momentum[i-1] > momentum[i] and 
                    momentum[i-1] > neutral_line):
                    signals[i] = -0.3  # Potential reversal from high
                elif (momentum[i-2] > momentum[i-1] and momentum[i-1] < momentum[i] and 
                      momentum[i-1] < neutral_line):
                    signals[i] = 0.3  # Potential reversal from low
        
        return signals
    
    def _calculate_metrics(self, momentum: np.ndarray) -> Dict:
        """Calculate additional Momentum metrics."""
        valid_values = momentum[~np.isnan(momentum)]
        
        if len(valid_values) == 0:
            return {}
        
        neutral_line = 100 if self.method == "ratio" else 0
        
        # Momentum conditions
        positive_momentum_pct = np.sum(valid_values > neutral_line) / len(valid_values) * 100
        negative_momentum_pct = np.sum(valid_values < neutral_line) / len(valid_values) * 100
        
        # Extreme momentum analysis
        if self.method == "ratio":
            strong_positive_pct = np.sum(valid_values > 110) / len(valid_values) * 100
            strong_negative_pct = np.sum(valid_values < 90) / len(valid_values) * 100
            extreme_positive_pct = np.sum(valid_values > 120) / len(valid_values) * 100
            extreme_negative_pct = np.sum(valid_values < 80) / len(valid_values) * 100
        else:
            std_dev = np.std(valid_values)
            mean_val = np.mean(valid_values)
            strong_positive_pct = np.sum(valid_values > mean_val + std_dev) / len(valid_values) * 100
            strong_negative_pct = np.sum(valid_values < mean_val - std_dev) / len(valid_values) * 100
            extreme_positive_pct = np.sum(valid_values > mean_val + 2*std_dev) / len(valid_values) * 100
            extreme_negative_pct = np.sum(valid_values < mean_val - 2*std_dev) / len(valid_values) * 100
        
        # Recent trend
        recent_values = valid_values[-min(5, len(valid_values)):]
        trend = np.mean(np.diff(recent_values)) if len(recent_values) > 1 else 0
        
        # Peak and trough analysis
        peaks = 0
        troughs = 0
        
        for i in range(1, len(valid_values) - 1):
            if (valid_values[i] > valid_values[i-1] and 
                valid_values[i] > valid_values[i+1]):
                peaks += 1
            elif (valid_values[i] < valid_values[i-1] and 
                  valid_values[i] < valid_values[i+1]):
                troughs += 1
        
        # Momentum persistence
        current_above_neutral = valid_values[-1] > neutral_line
        persistence_count = 0
        
        for i in range(len(valid_values) - 1, -1, -1):
            if (valid_values[i] > neutral_line) == current_above_neutral:
                persistence_count += 1
            else:
                break
        
        return {
            'current_value': momentum[-1] if not np.isnan(momentum[-1]) else None,
            'neutral_line': neutral_line,
            'positive_momentum_pct': positive_momentum_pct,
            'negative_momentum_pct': negative_momentum_pct,
            'strong_positive_pct': strong_positive_pct,
            'strong_negative_pct': strong_negative_pct,
            'extreme_positive_pct': extreme_positive_pct,
            'extreme_negative_pct': extreme_negative_pct,
            'recent_trend': trend,
            'volatility': np.std(valid_values),
            'mean_value': np.mean(valid_values),
            'max_value': np.max(valid_values),
            'min_value': np.min(valid_values),
            'peaks_count': peaks,
            'troughs_count': troughs,
            'momentum_persistence': persistence_count,
            'momentum_range': np.max(valid_values) - np.min(valid_values)
        }
    
    def _interpret_signals(self, current_momentum: float, current_signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_momentum):
            return "Insufficient data for Momentum calculation"
        
        neutral_line = 100 if self.method == "ratio" else 0
        
        if self.method == "ratio":
            if current_momentum > 120:
                momentum_desc = "VERY STRONG positive momentum"
            elif current_momentum > 110:
                momentum_desc = "STRONG positive momentum"
            elif current_momentum > neutral_line:
                momentum_desc = "Positive momentum"
            elif current_momentum > 90:
                momentum_desc = "Negative momentum"
            elif current_momentum > 80:
                momentum_desc = "STRONG negative momentum"
            else:
                momentum_desc = "VERY STRONG negative momentum"
        else:
            if current_momentum > 0:
                momentum_desc = f"Positive momentum (+{current_momentum:.2f})"
            else:
                momentum_desc = f"Negative momentum ({current_momentum:.2f})"
        
        signal_desc = {
            1: "BUY signal (crossed above neutral)",
            0.5: "Weak BUY signal (momentum increasing)",
            0.3: "Potential BUY (reversal from low)",
            -0.3: "Potential SELL (reversal from high)",
            -0.5: "Weak SELL signal (momentum decreasing)",
            -1: "SELL signal (crossed below neutral)",
            0: "No signal"
        }.get(current_signal, "No signal")
        
        method_label = "Ratio" if self.method == "ratio" else "Difference"
        
        return f"Momentum ({method_label}): {current_momentum:.2f} ({momentum_desc}) - {signal_desc}"


def create_momentum_indicator(period: int = 14, method: str = "ratio", **kwargs) -> MomentumIndicator:
    """Factory function to create Momentum Indicator."""
    return MomentumIndicator(period=period, method=method, **kwargs)
