"""
Fractal Efficiency Ratio (FER)
==============================

The Fractal Efficiency Ratio measures how efficiently price moves
relative to its theoretical maximum efficiency using fractal analysis.
Combines Kaufman's Efficiency Ratio with fractal dimension concepts.

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


class FractalEfficiencyRatio(IndicatorBase):
    """
    Fractal Efficiency Ratio (FER) indicator.
    
    Measures price movement efficiency by combining:
    - Kaufman's Efficiency Ratio
    - Fractal dimension analysis
    - Path length optimization
    - Market noise filtering
    """
    
    def __init__(self, 
                 period: int = 20,
                 smoothing_period: int = 5):
        """
        Initialize Fractal Efficiency Ratio.
        
        Args:
            period: Period for efficiency calculation
            smoothing_period: Period for smoothing the ratio
        """
        super().__init__()
        
        self.period = period
        self.smoothing_period = smoothing_period
        
        # Validation
        if period <= 0 or smoothing_period <= 0:
            raise ValueError("Periods must be positive")
        if smoothing_period > period:
            raise ValueError("Smoothing period cannot be greater than main period")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Fractal Efficiency Ratio values.
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Dictionary containing FER values and signals
        """
        try:
            # Validate input data
            required_columns = ['high', 'low', 'close']
            self._validate_data(data, required_columns)
            
            if len(data) < self.period:
                raise ValueError(f"Insufficient data: need at least {self.period} periods")
            
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate Fractal Efficiency Ratio
            fer = self._calculate_fractal_efficiency_ratio(close, high, low)
            
            # Smooth the ratio
            fer_smooth = self._smooth_ratio(fer)
            
            # Generate signals
            signals = self._generate_signals(fer, fer_smooth)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(fer, fer_smooth)
            
            return {
                'fractal_efficiency_ratio': fer,
                'fer_smooth': fer_smooth,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(fer[-1], fer_smooth[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Fractal Efficiency Ratio: {e}")
            raise
    
    def _calculate_fractal_efficiency_ratio(self, close: np.ndarray, 
                                          high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Calculate Fractal Efficiency Ratio values."""
        fer = np.full(len(close), np.nan)
        
        for i in range(self.period - 1, len(close)):
            # Extract period data
            period_close = close[i - self.period + 1:i + 1]
            period_high = high[i - self.period + 1:i + 1]
            period_low = low[i - self.period + 1:i + 1]
            
            # Calculate components
            net_change = abs(period_close[-1] - period_close[0])
            total_movement = self._calculate_total_movement(period_high, period_low, period_close)
            fractal_dimension = self._calculate_fractal_dimension(period_close)
            path_efficiency = self._calculate_path_efficiency(period_close)
            
            # Calculate Fractal Efficiency Ratio
            if total_movement > 0:
                # Base efficiency ratio (Kaufman style)
                base_efficiency = net_change / total_movement
                
                # Fractal adjustment (optimal fractal dimension is ~1.5)
                fractal_adjustment = 1.5 / max(fractal_dimension, 1.0)
                
                # Path efficiency component
                path_component = path_efficiency
                
                # Combined Fractal Efficiency Ratio
                fer[i] = base_efficiency * fractal_adjustment * path_component
                
                # Normalize to 0-1 range
                fer[i] = min(fer[i], 1.0)
            else:
                fer[i] = 0.0
        
        return fer
    
    def _calculate_total_movement(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate total price movement using True Range concept."""
        total_movement = 0.0
        
        for j in range(1, len(close)):
            # True Range calculation
            true_range = max(
                high[j] - low[j],
                abs(high[j] - close[j-1]),
                abs(low[j] - close[j-1])
            )
            total_movement += true_range
        
        return total_movement
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using Higuchi method."""
        if len(prices) < 4:
            return 1.5
        
        # Higuchi fractal dimension
        k_max = min(8, len(prices) // 4)
        lk = []
        
        for k in range(1, k_max + 1):
            l_mk = 0
            for m in range(k):
                ll = 0
                n = (len(prices) - m - 1) // k
                if n > 0:
                    for i in range(1, n + 1):
                        ll += abs(prices[m + i * k] - prices[m + (i - 1) * k])
                    ll = ll * (len(prices) - 1) / (n * k)
                    l_mk += ll
            
            if k > 0:
                lk.append(l_mk / k)
            else:
                lk.append(0)
        
        # Calculate fractal dimension
        if len(lk) > 1:
            k_values = np.arange(1, len(lk) + 1)
            lk = np.array(lk)
            lk = lk[lk > 0]  # Remove zeros
            k_values = k_values[:len(lk)]
            
            if len(lk) > 1:
                # Linear regression in log-log space
                log_k = np.log(k_values)
                log_lk = np.log(lk)
                slope = np.polyfit(log_k, log_lk, 1)[0]
                return 2 - slope
        
        return 1.5  # Default fractal dimension
    
    def _calculate_path_efficiency(self, prices: np.ndarray) -> float:
        """Calculate path efficiency using geometric approach."""
        if len(prices) < 2:
            return 0.5
        
        # Euclidean distance (direct path)
        time_span = len(prices) - 1
        price_change = prices[-1] - prices[0]
        direct_distance = np.sqrt(time_span**2 + price_change**2)
        
        # Actual path length
        actual_path = 0.0
        for i in range(1, len(prices)):
            time_step = 1
            price_step = prices[i] - prices[i-1]
            actual_path += np.sqrt(time_step**2 + price_step**2)
        
        # Path efficiency
        if actual_path > 0:
            return direct_distance / actual_path
        else:
            return 0.0
    
    def _smooth_ratio(self, fer: np.ndarray) -> np.ndarray:
        """Smooth the Fractal Efficiency Ratio."""
        fer_smooth = np.full(len(fer), np.nan)
        
        for i in range(self.smoothing_period - 1, len(fer)):
            if not np.isnan(fer[i]):
                # Simple moving average
                values = fer[i - self.smoothing_period + 1:i + 1]
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    fer_smooth[i] = np.mean(valid_values)
        
        return fer_smooth
    
    def _generate_signals(self, fer: np.ndarray, fer_smooth: np.ndarray) -> np.ndarray:
        """Generate trading signals based on Fractal Efficiency Ratio."""
        signals = np.zeros(len(fer))
        
        for i in range(2, len(fer)):
            if np.isnan(fer[i]) or np.isnan(fer_smooth[i]):
                continue
            
            current_fer = fer[i]
            smooth_fer = fer_smooth[i]
            prev_fer = fer[i-1]
            prev_smooth = fer_smooth[i-1]
            
            # High efficiency trending signals
            if current_fer > 0.7 and smooth_fer > 0.6:
                if current_fer > prev_fer and smooth_fer > prev_smooth:
                    signals[i] = 1  # Strong buy
                else:
                    signals[i] = 0.5  # Weak buy
            
            # Low efficiency (choppy market) signals
            elif current_fer < 0.3 and smooth_fer < 0.4:
                signals[i] = -1  # Sell (avoid choppy market)
            
            # Efficiency increasing from low levels
            elif (current_fer > prev_fer and current_fer > 0.4 and 
                  prev_fer < 0.4 and smooth_fer > prev_smooth):
                signals[i] = 0.5  # Emerging trend
            
            # Efficiency decreasing from high levels
            elif (current_fer < prev_fer and current_fer < 0.6 and 
                  prev_fer > 0.7):
                signals[i] = -0.5  # Trend weakening
        
        return signals
    
    def _calculate_metrics(self, fer: np.ndarray, fer_smooth: np.ndarray) -> Dict:
        """Calculate additional Fractal Efficiency Ratio metrics."""
        valid_fer = fer[~np.isnan(fer)]
        valid_smooth = fer_smooth[~np.isnan(fer_smooth)]
        
        if len(valid_fer) == 0:
            return {}
        
        # Efficiency distribution
        high_efficiency_pct = np.sum(valid_fer > 0.7) / len(valid_fer) * 100
        low_efficiency_pct = np.sum(valid_fer < 0.3) / len(valid_fer) * 100
        
        # Trend analysis
        if len(valid_smooth) > 5:
            recent_trend = np.mean(np.diff(valid_smooth[-5:]))
        else:
            recent_trend = 0
        
        # Efficiency persistence
        efficiency_changes = np.diff(valid_fer)
        persistence = len(efficiency_changes[efficiency_changes > 0]) / len(efficiency_changes) * 100 if len(efficiency_changes) > 0 else 50
        
        return {
            'current_fer': fer[-1] if not np.isnan(fer[-1]) else None,
            'current_fer_smooth': fer_smooth[-1] if not np.isnan(fer_smooth[-1]) else None,
            'avg_efficiency': np.mean(valid_fer),
            'efficiency_volatility': np.std(valid_fer),
            'high_efficiency_pct': high_efficiency_pct,
            'low_efficiency_pct': low_efficiency_pct,
            'medium_efficiency_pct': 100 - high_efficiency_pct - low_efficiency_pct,
            'efficiency_trend': recent_trend,
            'efficiency_persistence': persistence,
            'max_efficiency': np.max(valid_fer),
            'min_efficiency': np.min(valid_fer),
            'efficiency_range': np.max(valid_fer) - np.min(valid_fer)
        }
    
    def _interpret_signals(self, current_fer: float, current_smooth: float, current_signal: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_fer) or np.isnan(current_smooth):
            return "Insufficient data for Fractal Efficiency Ratio calculation"
        
        # Efficiency classification
        if current_fer > 0.7:
            efficiency_level = "HIGH"
        elif current_fer > 0.4:
            efficiency_level = "MEDIUM"
        else:
            efficiency_level = "LOW"
        
        # Market condition
        if current_fer > 0.6 and current_smooth > 0.5:
            market_condition = "TRENDING"
        elif current_fer < 0.4 and current_smooth < 0.4:
            market_condition = "CHOPPY"
        else:
            market_condition = "TRANSITIONAL"
        
        signal_desc = {
            1: "BUY signal (High efficiency trend)",
            0.5: "Weak BUY signal (Emerging efficiency)",
            -0.5: "Weak SELL signal (Declining efficiency)",
            -1: "SELL signal (Low efficiency/choppy)",
            0: "No signal"
        }.get(current_signal, "No signal")
        
        return (f"Fractal Efficiency Ratio: {current_fer:.3f} ({efficiency_level}) | "
                f"Smoothed: {current_smooth:.3f} | "
                f"Market: {market_condition} - {signal_desc}")


def create_fractal_efficiency_ratio(period: int = 20, **kwargs) -> FractalEfficiencyRatio:
    """Factory function to create Fractal Efficiency Ratio indicator."""
    return FractalEfficiencyRatio(period=period, **kwargs)
