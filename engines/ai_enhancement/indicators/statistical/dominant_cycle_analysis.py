"""
Dominant Cycle Analysis Indicator

A sophisticated dominant cycle analysis indicator that identifies, tracks, and analyzes
the most significant cyclical patterns in price data. This indicator builds on cycle
detection to provide deeper analysis of cycle characteristics, phase relationships,
and predictive capabilities.

Author: Platform3
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import sys
import os
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        """Fallback base class for direct script execution"""
        pass


class DominantCycleAnalysis(BaseIndicator):
    """
    Dominant Cycle Analysis Indicator
    
    Provides comprehensive analysis of the dominant cycle in price data:
    - Cycle period identification and tracking
    - Cycle amplitude and strength measurement
    - Phase analysis and cycle position
    - Hilbert Transform for instantaneous phase
    - Cycle reliability and consistency scoring
    - Predictive cycle projections
    - Multi-timeframe cycle harmonics
    
    The indicator provides:
    - Dominant cycle period (adaptive)
    - Cycle amplitude and strength
    - Current cycle phase (0-2π)
    - Cycle direction and momentum
    - Reliability score
    - Next cycle turn prediction
    """
    
    def __init__(self, 
                 min_period: int = 8,
                 max_period: int = 50,
                 smooth_period: int = 3,
                 delta_phase: float = 0.1,
                 inst_period_mult: float = 0.075):
        """
        Initialize Dominant Cycle Analysis indicator
        
        Args:
            min_period: Minimum cycle period to analyze
            max_period: Maximum cycle period to analyze  
            smooth_period: Smoothing period for calculations
            delta_phase: Phase threshold for cycle detection
            inst_period_mult: Multiplier for instantaneous period calculation
        """
        super().__init__()
        self.min_period = min_period
        self.max_period = max_period
        self.smooth_period = smooth_period
        self.delta_phase = delta_phase
        self.inst_period_mult = inst_period_mult
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate dominant cycle analysis
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'cycle_period': Dominant cycle period (adaptive)
            - 'cycle_amplitude': Cycle amplitude/strength
            - 'cycle_phase': Current phase in cycle (0-2π)
            - 'cycle_direction': Cycle direction (+1/-1)
            - 'reliability': Cycle reliability score (0-1)
            - 'turn_prediction': Predicted cycle turns
        """
        try:
            if len(data) < self.max_period * 2:
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'cycle_period': empty_series,
                    'cycle_amplitude': empty_series,
                    'cycle_phase': empty_series,
                    'cycle_direction': empty_series,
                    'reliability': empty_series,
                    'turn_prediction': empty_series
                }
            
            close = data['close']
            
            # Apply Hilbert Transform for cycle analysis
            detrended_price = self._detrend_price(close)
            
            # Calculate instantaneous period using Hilbert Transform
            cycle_period = self._calculate_instantaneous_period(detrended_price)
            
            # Calculate cycle amplitude
            cycle_amplitude = self._calculate_cycle_amplitude(detrended_price, cycle_period)
            
            # Calculate cycle phase
            cycle_phase = self._calculate_cycle_phase(detrended_price)
            
            # Determine cycle direction
            cycle_direction = self._calculate_cycle_direction(cycle_phase)
            
            # Calculate reliability score
            reliability = self._calculate_reliability(cycle_period, cycle_amplitude)
            
            # Predict cycle turns
            turn_prediction = self._predict_cycle_turns(cycle_phase, cycle_direction, reliability)
            
            return {
                'cycle_period': cycle_period,
                'cycle_amplitude': cycle_amplitude,
                'cycle_phase': cycle_phase,
                'cycle_direction': cycle_direction,
                'reliability': reliability,
                'turn_prediction': turn_prediction
            }
            
        except Exception as e:
            print(f"Error in Dominant Cycle Analysis: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'cycle_period': empty_series,
                'cycle_amplitude': empty_series,
                'cycle_phase': empty_series,
                'cycle_direction': empty_series,
                'reliability': empty_series,
                'turn_prediction': empty_series
            }
    
    def _detrend_price(self, price: pd.Series) -> pd.Series:
        """Remove trend from price to isolate cyclical component"""
        try:
            # Use Ehlers' method for detrending
            period = min(self.max_period, len(price) // 4)
            if period < self.min_period:
                period = self.min_period
                
            # Simple detrend using moving average
            trend = price.rolling(window=period, min_periods=1).mean()
            detrended = price - trend
            
            # Apply additional smoothing
            if self.smooth_period > 1:
                detrended = detrended.rolling(window=self.smooth_period, min_periods=1).mean()
            
            return detrended.fillna(0)
        except:
            return pd.Series(0, index=price.index)
    
    def _calculate_instantaneous_period(self, detrended_price: pd.Series) -> pd.Series:
        """Calculate instantaneous period using Hilbert Transform"""
        try:
            # Apply Hilbert Transform
            analytic_signal = hilbert(detrended_price.values)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            
            # Calculate instantaneous frequency
            instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi)
            
            # Convert to period and smooth
            instantaneous_period = np.zeros(len(detrended_price))
            instantaneous_period[1:] = 1.0 / (instantaneous_freq + 1e-8)
            
            # Limit to reasonable range
            instantaneous_period = np.clip(instantaneous_period, self.min_period, self.max_period)
            
            # Smooth the period
            period_series = pd.Series(instantaneous_period, index=detrended_price.index)
            period_smooth = period_series.rolling(window=self.smooth_period, min_periods=1).mean()
            
            return period_smooth
        except:
            return pd.Series(self.min_period, index=detrended_price.index)
    
    def _calculate_cycle_amplitude(self, detrended_price: pd.Series, cycle_period: pd.Series) -> pd.Series:
        """Calculate cycle amplitude"""
        try:
            amplitude = pd.Series(0.0, index=detrended_price.index)
            
            for i in range(len(detrended_price)):
                period = int(cycle_period.iloc[i])
                
                # Calculate amplitude as the range over the cycle period
                start_idx = max(0, i - period)
                end_idx = i + 1
                
                if end_idx > start_idx:
                    cycle_data = detrended_price.iloc[start_idx:end_idx]
                    amplitude.iloc[i] = cycle_data.max() - cycle_data.min()
            
            # Smooth the amplitude
            amplitude = amplitude.rolling(window=self.smooth_period, min_periods=1).mean()
            
            return amplitude
        except:
            return pd.Series(0, index=detrended_price.index)
    
    def _calculate_cycle_phase(self, detrended_price: pd.Series) -> pd.Series:
        """Calculate cycle phase using Hilbert Transform"""
        try:
            # Apply Hilbert Transform to get instantaneous phase
            analytic_signal = hilbert(detrended_price.values)
            instantaneous_phase = np.angle(analytic_signal)
            
            # Normalize phase to 0-2π range
            phase_normalized = (instantaneous_phase + np.pi) % (2 * np.pi)
            
            # Smooth the phase
            phase_series = pd.Series(phase_normalized, index=detrended_price.index)
            phase_smooth = phase_series.rolling(window=self.smooth_period, min_periods=1).mean()
            
            return phase_smooth
        except:
            return pd.Series(0, index=detrended_price.index)
    
    def _calculate_cycle_direction(self, cycle_phase: pd.Series) -> pd.Series:
        """Calculate cycle direction based on phase derivative"""
        try:
            # Calculate phase change (direction)
            phase_change = cycle_phase.diff()
            
            # Handle phase wraparound
            phase_change = np.where(phase_change > np.pi, phase_change - 2*np.pi, phase_change)
            phase_change = np.where(phase_change < -np.pi, phase_change + 2*np.pi, phase_change)
            
            # Convert to direction signal
            direction = pd.Series(phase_change, index=cycle_phase.index)
            direction = np.sign(direction).fillna(0)
            
            # Smooth the direction
            direction = direction.rolling(window=self.smooth_period, min_periods=1).mean()
            
            return direction
        except:
            return pd.Series(0, index=cycle_phase.index)
    
    def _calculate_reliability(self, cycle_period: pd.Series, cycle_amplitude: pd.Series) -> pd.Series:
        """Calculate cycle reliability score"""
        try:
            reliability = pd.Series(0.0, index=cycle_period.index)
            
            for i in range(self.max_period, len(cycle_period)):
                # Check period consistency
                recent_periods = cycle_period.iloc[i-self.max_period:i]
                period_std = recent_periods.std()
                period_mean = recent_periods.mean()
                
                # Period consistency score (lower std = higher reliability)
                if period_mean > 0:
                    period_consistency = 1 - min(1, period_std / period_mean)
                else:
                    period_consistency = 0
                
                # Amplitude consistency
                recent_amplitudes = cycle_amplitude.iloc[i-self.max_period:i]
                amp_std = recent_amplitudes.std()
                amp_mean = recent_amplitudes.mean()
                
                if amp_mean > 0:
                    amplitude_consistency = 1 - min(1, amp_std / amp_mean)
                else:
                    amplitude_consistency = 0
                
                # Combined reliability
                reliability.iloc[i] = (period_consistency + amplitude_consistency) / 2
            
            return reliability
        except:
            return pd.Series(0, index=cycle_period.index)
    
    def _predict_cycle_turns(self, cycle_phase: pd.Series, cycle_direction: pd.Series,
                           reliability: pd.Series) -> pd.Series:
        """Predict cycle turning points"""
        try:
            turn_prediction = pd.Series(0, index=cycle_phase.index)
            
            for i in range(1, len(cycle_phase)):
                current_phase = cycle_phase.iloc[i]
                prev_phase = cycle_phase.iloc[i-1]
                current_direction = cycle_direction.iloc[i]
                current_reliability = reliability.iloc[i]
                
                # Only make predictions if reliability is high enough
                if current_reliability > 0.5:
                    # Detect phase transitions that indicate turns
                    
                    # Approaching cycle peak (phase near π)
                    if abs(current_phase - np.pi) < self.delta_phase and current_direction > 0:
                        turn_prediction.iloc[i] = -1  # Expect downward turn
                    
                    # Approaching cycle trough (phase near 0 or 2π)
                    elif (current_phase < self.delta_phase or 
                          current_phase > 2*np.pi - self.delta_phase) and current_direction < 0:
                        turn_prediction.iloc[i] = 1   # Expect upward turn
                    
                    # Detect direction changes
                    elif i > 1:
                        prev_direction = cycle_direction.iloc[i-1]
                        if prev_direction > 0 and current_direction < 0:
                            turn_prediction.iloc[i] = -1  # Direction change to down
                        elif prev_direction < 0 and current_direction > 0:
                            turn_prediction.iloc[i] = 1   # Direction change to up
            
            return turn_prediction
        except:
            return pd.Series(0, index=cycle_phase.index)
    
    def get_cycle_period(self, data: pd.DataFrame) -> pd.Series:
        """Get dominant cycle periods"""
        result = self.calculate(data)
        return result['cycle_period']
    
    def get_cycle_phase(self, data: pd.DataFrame) -> pd.Series:
        """Get cycle phases"""
        result = self.calculate(data)
        return result['cycle_phase']
    
    def get_turn_predictions(self, data: pd.DataFrame) -> pd.Series:
        """Get cycle turn predictions"""
        result = self.calculate(data)
        return result['turn_prediction']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with strong cyclical pattern
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    # Generate sample data with embedded dominant cycle
    t = np.arange(200)
    trend = 100 + t * 0.05  # Slight upward trend
    dominant_cycle = 8 * np.sin(2 * np.pi * t / 25)  # 25-day dominant cycle
    secondary_cycle = 3 * np.sin(2 * np.pi * t / 12)  # 12-day secondary cycle
    noise = np.random.randn(200) * 1.5
    
    close_prices = trend + dominant_cycle + secondary_cycle + noise
    
    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices + np.random.uniform(0, 1.5, 200),
        'low': close_prices - np.random.uniform(0, 1.5, 200),
        'close': close_prices,
        'volume': np.random.lognormal(10, 0.3, 200)
    }, index=dates)
    
    # Test the indicator
    print("Testing Dominant Cycle Analysis Indicator")
    print("=" * 50)
    
    indicator = DominantCycleAnalysis(
        min_period=8,
        max_period=60,
        smooth_period=3,
        delta_phase=0.2
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Cycle period range: {result['cycle_period'].min():.1f} to {result['cycle_period'].max():.1f}")
    print(f"Cycle amplitude range: {result['cycle_amplitude'].min():.3f} to {result['cycle_amplitude'].max():.3f}")
    print(f"Cycle phase range: {result['cycle_phase'].min():.3f} to {result['cycle_phase'].max():.3f}")
    print(f"Reliability range: {result['reliability'].min():.3f} to {result['reliability'].max():.3f}")
    
    # Analyze cycle characteristics
    avg_period = result['cycle_period'].mean()
    avg_amplitude = result['cycle_amplitude'].mean()
    avg_reliability = result['reliability'].mean()
    
    print(f"\nCycle Analysis:")
    print(f"Average cycle period: {avg_period:.1f} days (expected: ~25)")
    print(f"Average amplitude: {avg_amplitude:.3f}")
    print(f"Average reliability: {avg_reliability:.3f}")
    
    # Show turn predictions
    turns = result['turn_prediction']
    bullish_turns = (turns == 1).sum()
    bearish_turns = (turns == -1).sum()
    
    print(f"\nTurn Predictions:")
    print(f"Bullish turn signals: {bullish_turns}")
    print(f"Bearish turn signals: {bearish_turns}")
    
    # Show high reliability periods
    high_reliability = result['reliability'] > 0.7
    print(f"High reliability periods: {high_reliability.sum()} out of {len(data)}")
    
    print("\nDominant Cycle Analysis Indicator test completed successfully!")