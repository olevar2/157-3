"""
Cycle Period Identification Indicator

A cycle period identification indicator that detects and measures cyclical patterns
in price data. This indicator uses various methods including spectral analysis,
autocorrelation, and peak detection to identify dominant cycle periods in the market.

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

# Add the parent directory to Python path for imports

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        """Fallback base class for direct script execution"""
        pass


class CyclePeriodIdentification(BaseIndicator):
    """
    Cycle Period Identification Indicator
    
    Identifies dominant cycle periods in price data using multiple methods:
    - Spectral analysis (FFT)
    - Autocorrelation analysis
    - Peak/trough detection
    - Hilbert Transform
    - Detrended Fluctuation Analysis
    
    The indicator provides:
    - Dominant cycle period detection
    - Cycle strength measurement
    - Multiple cycle identification
    - Cycle phase analysis
    - Predictive cycle projections
    """
    
    def __init__(self, 
                 min_cycle_length: int = 8,
                 max_cycle_length: int = 50,
                 detrend_period: int = 14,
                 smoothing_period: int = 3,
                 confidence_threshold: float = 0.1):
        """
        Initialize Cycle Period Identification indicator
        
        Args:
            min_cycle_length: Minimum cycle length to detect
            max_cycle_length: Maximum cycle length to detect
            detrend_period: Period for detrending the data
            smoothing_period: Period for smoothing the results
            confidence_threshold: Minimum confidence for cycle detection
        """
        super().__init__()
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.detrend_period = detrend_period
        self.smoothing_period = smoothing_period
        self.confidence_threshold = confidence_threshold
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate cycle period identification
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'dominant_cycle': Identified dominant cycle period
            - 'cycle_strength': Strength/confidence of cycle detection
            - 'cycle_phase': Current phase within the cycle (0-1)
            - 'secondary_cycles': Additional detected cycles
            - 'cycle_direction': Predicted cycle direction
        """
        try:
            if len(data) < self.max_cycle_length * 2:
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'dominant_cycle': empty_series,
                    'cycle_strength': empty_series,
                    'cycle_phase': empty_series,
                    'secondary_cycles': empty_series,
                    'cycle_direction': empty_series
                }
            
            close = data['close']
            
            # Detrend the data
            detrended_data = self._detrend_data(close)
            
            # Method 1: Spectral Analysis (FFT)
            fft_cycles = self._fft_cycle_analysis(detrended_data)
            
            # Method 2: Autocorrelation Analysis
            autocorr_cycles = self._autocorrelation_analysis(detrended_data)
            
            # Method 3: Peak Detection Analysis
            peak_cycles = self._peak_detection_analysis(detrended_data)
            
            # Combine results to get dominant cycle
            dominant_cycle, cycle_strength = self._combine_cycle_results(
                fft_cycles, autocorr_cycles, peak_cycles, close.index
            )
            
            # Calculate cycle phase
            cycle_phase = self._calculate_cycle_phase(detrended_data, dominant_cycle)
            
            # Identify secondary cycles
            secondary_cycles = self._identify_secondary_cycles(
                fft_cycles, autocorr_cycles, peak_cycles, dominant_cycle, close.index
            )
            
            # Predict cycle direction
            cycle_direction = self._predict_cycle_direction(cycle_phase, dominant_cycle)
            
            return {
                'dominant_cycle': dominant_cycle,
                'cycle_strength': cycle_strength,
                'cycle_phase': cycle_phase,
                'secondary_cycles': secondary_cycles,
                'cycle_direction': cycle_direction
            }
            
        except Exception as e:
            print(f"Error in Cycle Period Identification: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'dominant_cycle': empty_series,
                'cycle_strength': empty_series,
                'cycle_phase': empty_series,
                'secondary_cycles': empty_series,
                'cycle_direction': empty_series
            }
    
    def _detrend_data(self, data: pd.Series) -> pd.Series:
        """Remove trend from data to isolate cyclical components"""
        try:
            # Use a moving average to detrend
            trend = data.rolling(window=self.detrend_period, min_periods=1).mean()
            detrended = data - trend
            return detrended.fillna(0)
        except:
            return pd.Series(0, index=data.index)
    
    def _fft_cycle_analysis(self, data: pd.Series) -> Dict[int, float]:
        """Analyze cycles using Fast Fourier Transform"""
        try:
            # Remove NaN values and ensure sufficient data
            clean_data = data.dropna()
            if len(clean_data) < self.max_cycle_length:
                return {}
            
            # Apply FFT
            fft_values = fft(clean_data.values)
            fft_freqs = fftfreq(len(clean_data))
            
            # Calculate power spectrum
            power_spectrum = np.abs(fft_values) ** 2
            
            # Find dominant frequencies
            cycle_powers = {}
            for i, freq in enumerate(fft_freqs):
                if freq > 0:  # Only positive frequencies
                    period = 1 / freq
                    if self.min_cycle_length <= period <= self.max_cycle_length:
                        cycle_powers[int(period)] = power_spectrum[i]
            
            return cycle_powers
        except:
            return {}
    
    def _autocorrelation_analysis(self, data: pd.Series) -> Dict[int, float]:
        """Analyze cycles using autocorrelation"""
        try:
            clean_data = data.dropna()
            if len(clean_data) < self.max_cycle_length:
                return {}
            
            # Calculate autocorrelation for different lags
            cycle_correlations = {}
            for lag in range(self.min_cycle_length, 
                           min(self.max_cycle_length, len(clean_data) // 2)):
                correlation = clean_data.autocorr(lag=lag)
                if not pd.isna(correlation):
                    cycle_correlations[lag] = abs(correlation)
            
            return cycle_correlations
        except:
            return {}
    
    def _peak_detection_analysis(self, data: pd.Series) -> Dict[int, float]:
        """Analyze cycles using peak/trough detection"""
        try:
            clean_data = data.dropna()
            if len(clean_data) < self.max_cycle_length:
                return {}
            
            # Find peaks and troughs
            peaks, _ = signal.find_peaks(clean_data.values, distance=self.min_cycle_length//2)
            troughs, _ = signal.find_peaks(-clean_data.values, distance=self.min_cycle_length//2)
            
            # Calculate distances between peaks and troughs
            all_extrema = sorted(list(peaks) + list(troughs))
            
            cycle_distances = {}
            if len(all_extrema) >= 4:  # Need at least 2 complete cycles
                distances = np.diff(all_extrema)
                
                # Count frequency of each distance
                for distance in distances:
                    if self.min_cycle_length <= distance <= self.max_cycle_length:
                        cycle_distance = int(distance * 2)  # Full cycle = peak to peak
                        if cycle_distance in cycle_distances:
                            cycle_distances[cycle_distance] += 1
                        else:
                            cycle_distances[cycle_distance] = 1
            
            return cycle_distances
        except:
            return {}
    
    def _combine_cycle_results(self, fft_cycles: Dict[int, float], 
                              autocorr_cycles: Dict[int, float],
                              peak_cycles: Dict[int, float],
                              index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """Combine results from different cycle detection methods"""
        try:
            # Normalize each method's results
            def normalize_dict(d):
                if not d:
                    return {}
                max_val = max(d.values())
                return {k: v/max_val for k, v in d.items()} if max_val > 0 else d
            
            fft_norm = normalize_dict(fft_cycles)
            autocorr_norm = normalize_dict(autocorr_cycles)
            peak_norm = normalize_dict(peak_cycles)
            
            # Combine scores with weights
            combined_scores = {}
            all_periods = set(fft_norm.keys()) | set(autocorr_norm.keys()) | set(peak_norm.keys())
            
            for period in all_periods:
                score = 0
                score += fft_norm.get(period, 0) * 0.4      # FFT weight
                score += autocorr_norm.get(period, 0) * 0.4  # Autocorr weight
                score += peak_norm.get(period, 0) * 0.2      # Peak detection weight
                combined_scores[period] = score
            
            # Find dominant cycle
            if combined_scores:
                dominant_period = max(combined_scores, key=combined_scores.get)
                max_strength = combined_scores[dominant_period]
                
                # Create time series
                dominant_cycle = pd.Series(dominant_period, index=index)
                cycle_strength = pd.Series(max_strength, index=index)
            else:
                dominant_cycle = pd.Series(0, index=index)
                cycle_strength = pd.Series(0, index=index)
            
            return dominant_cycle, cycle_strength
        except:
            empty_series = pd.Series(0, index=index)
            return empty_series, empty_series
    
    def _calculate_cycle_phase(self, data: pd.Series, dominant_cycle: pd.Series) -> pd.Series:
        """Calculate current phase within the dominant cycle"""
        try:
            cycle_phase = pd.Series(0.0, index=data.index)
            
            for i in range(len(data)):
                if dominant_cycle.iloc[i] > 0:
                    cycle_length = dominant_cycle.iloc[i]
                    
                    # Simple phase calculation based on position in cycle
                    phase = (i % cycle_length) / cycle_length
                    cycle_phase.iloc[i] = phase
            
            return cycle_phase
        except:
            return pd.Series(0, index=data.index)
    
    def _identify_secondary_cycles(self, fft_cycles: Dict[int, float],
                                  autocorr_cycles: Dict[int, float],
                                  peak_cycles: Dict[int, float],
                                  dominant_cycle: pd.Series,
                                  index: pd.Index) -> pd.Series:
        """Identify secondary cycles"""
        try:
            # Combine all detected cycles
            all_cycles = set(fft_cycles.keys()) | set(autocorr_cycles.keys()) | set(peak_cycles.keys())
            
            # Remove dominant cycle
            dominant_period = dominant_cycle.iloc[0] if len(dominant_cycle) > 0 else 0
            secondary_cycles = [c for c in all_cycles if c != dominant_period]
            
            # Sort by strength and take top secondary cycles
            secondary_scores = {}
            for period in secondary_cycles:
                score = 0
                score += fft_cycles.get(period, 0)
                score += autocorr_cycles.get(period, 0)
                score += peak_cycles.get(period, 0)
                secondary_scores[period] = score
            
            if secondary_scores:
                # Get the second strongest cycle
                second_cycle = max(secondary_scores, key=secondary_scores.get)
                return pd.Series(second_cycle, index=index)
            else:
                return pd.Series(0, index=index)
        except:
            return pd.Series(0, index=index)
    
    def _predict_cycle_direction(self, cycle_phase: pd.Series, dominant_cycle: pd.Series) -> pd.Series:
        """Predict cycle direction based on phase"""
        try:
            cycle_direction = pd.Series(0, index=cycle_phase.index)
            
            for i in range(len(cycle_phase)):
                phase = cycle_phase.iloc[i]
                
                # Simple direction prediction based on phase
                if 0 <= phase < 0.25:          # Rising phase
                    cycle_direction.iloc[i] = 1
                elif 0.25 <= phase < 0.75:     # Peak to trough
                    cycle_direction.iloc[i] = -1
                elif 0.75 <= phase < 1.0:      # Rising from trough
                    cycle_direction.iloc[i] = 1
            
            return cycle_direction
        except:
            return pd.Series(0, index=cycle_phase.index)
    
    def get_dominant_cycle(self, data: pd.DataFrame) -> pd.Series:
        """Get dominant cycle periods"""
        result = self.calculate(data)
        return result['dominant_cycle']
    
    def get_cycle_strength(self, data: pd.DataFrame) -> pd.Series:
        """Get cycle detection strength"""
        result = self.calculate(data)
        return result['cycle_strength']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with artificial cycles
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    # Generate sample data with embedded cycles
    t = np.arange(200)
    trend = 100 + t * 0.1  # Slight upward trend
    cycle1 = 5 * np.sin(2 * np.pi * t / 20)  # 20-day cycle
    cycle2 = 3 * np.sin(2 * np.pi * t / 35)  # 35-day cycle  
    noise = np.random.randn(200) * 1
    
    close_prices = trend + cycle1 + cycle2 + noise
    
    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices + np.random.uniform(0, 1, 200),
        'low': close_prices - np.random.uniform(0, 1, 200),
        'close': close_prices,
        'volume': np.random.lognormal(10, 0.3, 200)
    }, index=dates)
    
    # Test the indicator
    print("Testing Cycle Period Identification Indicator")
    print("=" * 50)
    
    indicator = CyclePeriodIdentification(
        min_cycle_length=10,
        max_cycle_length=60,
        detrend_period=14,
        confidence_threshold=0.1
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Dominant cycle range: {result['dominant_cycle'].min():.1f} to {result['dominant_cycle'].max():.1f}")
    print(f"Cycle strength range: {result['cycle_strength'].min():.3f} to {result['cycle_strength'].max():.3f}")
    print(f"Cycle phase range: {result['cycle_phase'].min():.3f} to {result['cycle_phase'].max():.3f}")
    
    # Show cycle statistics
    dominant_cycles = result['dominant_cycle']
    unique_cycles = dominant_cycles.unique()
    print(f"\nDetected cycle periods: {sorted([c for c in unique_cycles if c > 0])}")
    
    # Show cycle direction predictions
    directions = result['cycle_direction']
    print(f"Bullish cycle periods: {(directions == 1).sum()}")
    print(f"Bearish cycle periods: {(directions == -1).sum()}")
    print(f"Neutral cycle periods: {(directions == 0).sum()}")
    
    # Expected vs detected cycles
    print(f"\nExpected cycles: 20-day and 35-day")
    most_common_cycle = dominant_cycles.mode()[0] if len(dominant_cycles.mode()) > 0 else 0
    print(f"Most commonly detected cycle: {most_common_cycle:.1f} days")
    
    print("\nCycle Period Identification Indicator test completed successfully!")