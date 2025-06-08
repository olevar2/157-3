"""
Know Sure Thing (KST)
=====================

The Know Sure Thing (KST) is a momentum oscillator developed by Martin Pring 
that uses the rate-of-change (ROC) of four different time frames. It combines 
short-term, medium-term, and long-term momentum to create a more comprehensive 
momentum indicator.

Formula: KST = (ROC1 × 1) + (ROC2 × 2) + (ROC3 × 3) + (ROC4 × 4)
Then smoothed with moving averages.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple, List # Added List
from datetime import datetime

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import IndicatorBase


class KnowSureThing(IndicatorBase):
    """
    Know Sure Thing (KST) indicator.
    
    KST combines four different rate-of-change indicators with different 
    periods and weights them to create a comprehensive momentum oscillator.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, # Added config parameter
                 roc_periods: Tuple[int, int, int, int] = (10, 15, 20, 30),
                 sma_periods: Tuple[int, int, int, int] = (10, 10, 10, 15),
                 signal_period: int = 9):
        """
        Initialize Know Sure Thing indicator.
        
        Args:
            config: Configuration dictionary (optional)
            roc_periods: ROC periods for the four time frames
            sma_periods: SMA periods for smoothing each ROC
            signal_period: Period for the signal line
        """
        super().__init__(config=config) # Pass config to IndicatorBase
        
        self.roc_periods = self.config.get('roc_periods', roc_periods)
        self.sma_periods = self.config.get('sma_periods', sma_periods)
        self.signal_period = self.config.get('signal_period', signal_period)
        
        # Validation
        if len(self.roc_periods) != 4 or len(self.sma_periods) != 4:
            raise ValueError("Need exactly 4 ROC and SMA periods")
        if any(p <= 0 for p in self.roc_periods + self.sma_periods) or self.signal_period <= 0:
            raise ValueError("All periods must be positive")
      def _validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """Override or extend base validation if specific checks are needed for KST."""
        # Call super's validation first
        validation_result = super()._validate_data(data)
        if not validation_result:
             raise ValueError("Base data validation failed.")

        # KST specific validation for DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame for KnowSureThing.")

        if data.empty:
            raise ValueError("Input DataFrame is empty.")

        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Know Sure Thing values.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Dictionary containing KST values and signals
        """
        try:
            # Validate input data
            required_columns = ['close']
            self._validate_data(data, required_columns)
            
            min_required = max(self.roc_periods) + max(self.sma_periods) + self.signal_period
            if len(data) < min_required:
                raise ValueError(f"Insufficient data: need at least {min_required} periods")
            
            close = data['close'].values
            
            # Calculate Know Sure Thing
            kst, signal_line = self._calculate_kst(close)
            
            # Generate signals
            signals = self._generate_signals(kst, signal_line)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(kst, signal_line)
            
            return {
                'kst': kst,
                'kst_signal': signal_line,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(kst[-1], signal_line[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Know Sure Thing: {e}")
            raise
    
    def _calculate_kst(self, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Know Sure Thing and signal line."""
        kst = np.full(len(close), np.nan)
        
        # Calculate ROC for each time frame
        roc1 = self._calculate_roc(close, self.roc_periods[0])
        roc2 = self._calculate_roc(close, self.roc_periods[1])
        roc3 = self._calculate_roc(close, self.roc_periods[2])
        roc4 = self._calculate_roc(close, self.roc_periods[3])
        
        # Smooth each ROC with SMA
        smoothed_roc1 = self._sma(roc1, self.sma_periods[0])
        smoothed_roc2 = self._sma(roc2, self.sma_periods[1])
        smoothed_roc3 = self._sma(roc3, self.sma_periods[2])
        smoothed_roc4 = self._sma(roc4, self.sma_periods[3])
        
        # Calculate weighted KST
        weights = [1, 2, 3, 4]
        
        for i in range(len(close)):
            if (not np.isnan(smoothed_roc1[i]) and not np.isnan(smoothed_roc2[i]) and
                not np.isnan(smoothed_roc3[i]) and not np.isnan(smoothed_roc4[i])):
                
                kst[i] = (smoothed_roc1[i] * weights[0] +
                         smoothed_roc2[i] * weights[1] +
                         smoothed_roc3[i] * weights[2] +
                         smoothed_roc4[i] * weights[3])
        
        # Calculate signal line
        signal_line = self._sma(kst, self.signal_period)
        
        return kst, signal_line
    
    def _calculate_roc(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Rate of Change."""
        roc = np.full(len(data), np.nan)
        
        for i in range(period, len(data)):
            if data[i - period] != 0:
                roc[i] = ((data[i] - data[i - period]) / data[i - period]) * 100
        
        return roc
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            valid_data = data[i - period + 1:i + 1]
            if not np.any(np.isnan(valid_data)):
                sma[i] = np.mean(valid_data)
        
        return sma
    
    def _generate_signals(self, kst: np.ndarray, signal_line: np.ndarray) -> np.ndarray:
        """Generate trading signals based on KST."""
        signals = np.zeros(len(kst))
        
        for i in range(1, len(kst)):
            if (np.isnan(kst[i]) or np.isnan(kst[i-1]) or 
                np.isnan(signal_line[i]) or np.isnan(signal_line[i-1])):
                continue
            
            # Signal line crossovers
            if kst[i-1] <= signal_line[i-1] and kst[i] > signal_line[i]:
                signals[i] = 1  # Buy signal
            elif kst[i-1] >= signal_line[i-1] and kst[i] < signal_line[i]:
                signals[i] = -1  # Sell signal
            
            # Zero line crossovers
            elif kst[i-1] <= 0 and kst[i] > 0:
                signals[i] = 0.5  # Weak buy signal
            elif kst[i-1] >= 0 and kst[i] < 0:
                signals[i] = -0.5  # Weak sell signal
            
            # Divergence signals (simplified)
            elif i >= 5:
                recent_kst = kst[i-4:i+1]
                if (np.all(np.diff(recent_kst) > 0) and kst[i] > 0):
                    signals[i] = 0.3  # Momentum building
                elif (np.all(np.diff(recent_kst) < 0) and kst[i] < 0):
                    signals[i] = -0.3  # Momentum declining
        
        return signals
    
    def _calculate_metrics(self, kst: np.ndarray, signal_line: np.ndarray) -> Dict:
        """Calculate additional KST metrics."""
        valid_kst = kst[~np.isnan(kst)]
        valid_signal = signal_line[~np.isnan(signal_line)]
        
        if len(valid_kst) == 0:
            return {}
        
        # Momentum analysis
        positive_momentum_pct = np.sum(valid_kst > 0) / len(valid_kst) * 100
        negative_momentum_pct = np.sum(valid_kst < 0) / len(valid_kst) * 100
        
        # Above/below signal line
        above_signal_pct = 0
        below_signal_pct = 0
        
        if len(valid_signal) > 0:
            valid_both = len(min(valid_kst, valid_signal, key=len))
            if valid_both > 0:
                kst_subset = valid_kst[-valid_both:]
                signal_subset = valid_signal[-valid_both:]
                above_signal_pct = np.sum(kst_subset > signal_subset) / valid_both * 100
                below_signal_pct = np.sum(kst_subset < signal_subset) / valid_both * 100
        
        # Recent trend
        recent_kst = valid_kst[-min(5, len(valid_kst)):]
        trend = np.mean(np.diff(recent_kst)) if len(recent_kst) > 1 else 0
        
        # Crossover analysis
        crossover_count = 0
        for i in range(1, min(len(valid_kst), len(valid_signal))):
            if ((valid_kst[i-1] <= valid_signal[i-1] and valid_kst[i] > valid_signal[i]) or
                (valid_kst[i-1] >= valid_signal[i-1] and valid_kst[i] < valid_signal[i])):
                crossover_count += 1
        
        # Current histogram
        current_histogram = None
        if not np.isnan(kst[-1]) and not np.isnan(signal_line[-1]):
            current_histogram = kst[-1] - signal_line[-1]
        
        return {
            'current_kst': kst[-1] if not np.isnan(kst[-1]) else None,
            'current_signal': signal_line[-1] if not np.isnan(signal_line[-1]) else None,
            'current_histogram': current_histogram,
            'positive_momentum_pct': positive_momentum_pct,
            'negative_momentum_pct': negative_momentum_pct,
            'above_signal_pct': above_signal_pct,
            'below_signal_pct': below_signal_pct,
            'recent_trend': trend,
            'volatility': np.std(valid_kst),
            'mean_value': np.mean(valid_kst),
            'max_value': np.max(valid_kst),
            'min_value': np.min(valid_kst),
            'crossover_count': crossover_count,
            'momentum_strength': abs(trend) if trend != 0 else 0
        }
    
    def _interpret_signals(self, current_kst: float, current_signal: float, current_signal_val: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_kst) or np.isnan(current_signal):
            return "Insufficient data for KST calculation"
        
        # Determine momentum direction
        if current_kst > current_signal:
            momentum_desc = "BULLISH (above signal line)"
        else:
            momentum_desc = "BEARISH (below signal line)"
        
        # Overall momentum
        if current_kst > 0:
            overall = "Positive momentum"
        else:
            overall = "Negative momentum"
        
        signal_desc = {
            1: "BUY signal (KST crossed above signal line)",
            0.5: "Weak BUY signal (zero line cross up)",
            0.3: "Momentum building",
            -0.3: "Momentum declining", 
            -0.5: "Weak SELL signal (zero line cross down)",
            -1: "SELL signal (KST crossed below signal line)",
            0: "No signal"
        }.get(current_signal_val, "No signal")
        
        histogram = current_kst - current_signal
        
        return f"KST: {current_kst:.2f} vs Signal: {current_signal:.2f} (Histogram: {histogram:.2f}) - {overall}, {momentum_desc} - {signal_desc}"


def create_know_sure_thing(roc_periods: Tuple[int, int, int, int] = (10, 15, 20, 30), **kwargs) -> KnowSureThing:
    """Factory function to create Know Sure Thing indicator."""
    return KnowSureThing(roc_periods=roc_periods, **kwargs)
