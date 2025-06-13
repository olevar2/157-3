"""
Hurst Exponent Indicator
========================

The Hurst Exponent is a measure of long-term memory in time series data.
It quantifies the tendency of a time series to regress to its long-term mean 
or cluster in a specific direction.

Values:
- H > 0.5: Persistent (trending) behavior
- H = 0.5: Random walk behavior
- H < 0.5: Anti-persistent (mean-reverting) behavior

Author: Platform3 AI Enhancement Engine
Created: 2024
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any
import warnings

# For standalone testing
import sys
import os

try:
    from base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        def __init__(self):
            pass

class HurstExponent(BaseIndicator):
    """
    Hurst Exponent Calculator
    
    The Hurst Exponent (H) measures the long-range dependence of a time series.
    It's particularly useful for:
    - Detecting trending vs mean-reverting behavior
    - Risk assessment in financial markets
    - Fractal analysis of market microstructure
    """
    
    def __init__(self, 
                 window: int = 100,
                 min_window: int = 10,
                 method: str = 'rs'):
        """
        Initialize Hurst Exponent calculator.
        
        Parameters:
        -----------
        window : int, default=100
            Number of periods for calculation
        min_window : int, default=10
            Minimum window size for calculation
        method : str, default='rs'
            Method to use: 'rs' (R/S analysis) or 'dfa' (Detrended Fluctuation Analysis)
        """
        super().__init__()
        self.window = window
        self.min_window = max(min_window, 10)
        self.method = method.lower()
        
        if self.method not in ['rs', 'dfa']:
            raise ValueError("Method must be 'rs' or 'dfa'")
    
    def _rs_analysis(self, data: np.ndarray) -> float:
        """
        Calculate Hurst Exponent using R/S Analysis (Rescaled Range).
        
        Parameters:
        -----------
        data : np.ndarray
            Price or return data
            
        Returns:
        --------
        float
            Hurst exponent value
        """
        n = len(data)
        if n < self.min_window:
            return np.nan
        
        # Convert to log returns if prices
        if np.all(data > 0):
            returns = np.diff(np.log(data))
        else:
            returns = np.diff(data)
        
        if len(returns) < self.min_window:
            return np.nan
        
        # Calculate mean return
        mean_return = np.mean(returns)
        
        # Calculate cumulative deviations
        deviations = returns - mean_return
        cumulative_deviations = np.cumsum(deviations)
        
        # Calculate range
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
        
        # Calculate standard deviation
        S = np.std(returns, ddof=1)
        
        if S == 0 or R == 0:
            return 0.5  # Random walk case
        
        # R/S ratio
        rs_ratio = R / S
        
        # Hurst exponent approximation
        if rs_ratio <= 0:
            return 0.5
        
        # Use multiple sub-periods for better estimation
        sub_periods = []
        rs_values = []
        
        min_size = max(self.min_window, len(returns) // 10)
        max_periods = min(10, len(returns) // min_size)
        
        for i in range(2, max_periods + 1):
            period_size = len(returns) // i
            if period_size < min_size:
                continue
                
            period_rs = []
            for j in range(i):
                start_idx = j * period_size
                end_idx = start_idx + period_size
                if end_idx > len(returns):
                    break
                    
                period_data = returns[start_idx:end_idx]
                if len(period_data) < min_size:
                    continue
                
                period_mean = np.mean(period_data)
                period_deviations = period_data - period_mean
                period_cumulative = np.cumsum(period_deviations)
                
                period_R = np.max(period_cumulative) - np.min(period_cumulative)
                period_S = np.std(period_data, ddof=1)
                
                if period_S > 0 and period_R > 0:
                    period_rs.append(period_R / period_S)
            
            if period_rs:
                sub_periods.append(period_size)
                rs_values.append(np.mean(period_rs))
        
        if len(sub_periods) < 2:
            # Fallback to simple calculation
            return np.log(rs_ratio) / np.log(len(returns))
        
        # Linear regression to find Hurst exponent
        log_periods = np.log(sub_periods)
        log_rs = np.log(rs_values)
        
        # Remove any infinite or NaN values
        valid_mask = np.isfinite(log_periods) & np.isfinite(log_rs)
        if np.sum(valid_mask) < 2:
            return 0.5
        
        log_periods = log_periods[valid_mask]
        log_rs = log_rs[valid_mask]
        
        # Calculate Hurst exponent as slope
        try:
            slope = np.polyfit(log_periods, log_rs, 1)[0]
            return max(0.0, min(1.0, slope))  # Constrain between 0 and 1
        except:
            return 0.5
    
    def _dfa_analysis(self, data: np.ndarray) -> float:
        """
        Calculate Hurst Exponent using Detrended Fluctuation Analysis.
        
        Parameters:
        -----------
        data : np.ndarray
            Price or return data
            
        Returns:
        --------
        float
            Hurst exponent value
        """
        n = len(data)
        if n < self.min_window:
            return np.nan
        
        # Convert to returns
        if np.all(data > 0):
            returns = np.diff(np.log(data))
        else:
            returns = np.diff(data)
        
        if len(returns) < self.min_window:
            return np.nan
        
        # Remove mean
        returns = returns - np.mean(returns)
        
        # Integrate the series
        y = np.cumsum(returns)
        
        # Define box sizes
        min_box_size = max(4, self.min_window // 4)
        max_box_size = min(len(y) // 4, 50)
        
        if min_box_size >= max_box_size:
            return 0.5
        
        box_sizes = np.logspace(np.log10(min_box_size), 
                               np.log10(max_box_size), 
                               num=min(10, max_box_size - min_box_size + 1)).astype(int)
        box_sizes = np.unique(box_sizes)
        
        fluctuations = []
        
        for box_size in box_sizes:
            # Number of boxes
            n_boxes = len(y) // box_size
            if n_boxes < 2:
                continue
            
            box_fluctuations = []
            
            for i in range(n_boxes):
                start_idx = i * box_size
                end_idx = start_idx + box_size
                
                box_data = y[start_idx:end_idx]
                x = np.arange(len(box_data))
                
                # Linear detrending
                try:
                    coeffs = np.polyfit(x, box_data, 1)
                    trend = np.polyval(coeffs, x)
                    detrended = box_data - trend
                    
                    # Calculate fluctuation
                    fluctuation = np.sqrt(np.mean(detrended**2))
                    if fluctuation > 0:
                        box_fluctuations.append(fluctuation)
                except:
                    continue
            
            if box_fluctuations:
                fluctuations.append(np.mean(box_fluctuations))
            else:
                fluctuations.append(np.nan)
        
        # Remove NaN values
        valid_mask = ~np.isnan(fluctuations)
        if np.sum(valid_mask) < 2:
            return 0.5
        
        box_sizes = box_sizes[valid_mask]
        fluctuations = np.array(fluctuations)[valid_mask]
        
        # Linear regression in log-log space
        log_box_sizes = np.log(box_sizes)
        log_fluctuations = np.log(fluctuations)
        
        try:
            slope = np.polyfit(log_box_sizes, log_fluctuations, 1)[0]
            return max(0.0, min(1.0, slope))  # Constrain between 0 and 1
        except:
            return 0.5
    
    def calculate(self, data: Union[pd.Series, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Hurst Exponent.
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            Price or return data
            
        Returns:
        --------
        float or np.ndarray
            Hurst exponent value(s)
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        data = np.asarray(data, dtype=float)
        
        if len(data) < self.min_window:
            return np.nan
        
        # For rolling calculation
        if len(data) <= self.window:
            if self.method == 'rs':
                return self._rs_analysis(data)
            else:
                return self._dfa_analysis(data)
        
        # Rolling Hurst Exponent
        result = np.full(len(data), np.nan)
        
        for i in range(self.window, len(data) + 1):
            window_data = data[i - self.window:i]
            
            if self.method == 'rs':
                result[i - 1] = self._rs_analysis(window_data)
            else:
                result[i - 1] = self._dfa_analysis(window_data)
        
        return result
    
    def calculate_single(self, data: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate single Hurst Exponent value for entire dataset.
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            Price or return data
            
        Returns:
        --------
        float
            Single Hurst exponent value
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        data = np.asarray(data, dtype=float)
        
        if self.method == 'rs':
            return self._rs_analysis(data)
        else:
            return self._dfa_analysis(data)
    
    def interpret_hurst(self, hurst_value: float) -> Dict[str, Any]:
        """
        Interpret Hurst Exponent value.
        
        Parameters:
        -----------
        hurst_value : float
            Hurst exponent value
            
        Returns:
        --------
        dict
            Interpretation of the Hurst value
        """
        if np.isnan(hurst_value):
            return {
                'interpretation': 'Insufficient data',
                'behavior': 'Unknown',
                'confidence': 0.0,
                'trading_signal': 'Neutral'
            }
        
        if hurst_value > 0.55:
            interpretation = 'Persistent/Trending'
            behavior = 'Long-term memory, trending behavior'
            confidence = min(1.0, (hurst_value - 0.5) * 2)
            trading_signal = 'Trend Following'
        elif hurst_value < 0.45:
            interpretation = 'Anti-persistent/Mean Reverting'
            behavior = 'Short-term memory, mean-reverting behavior'
            confidence = min(1.0, (0.5 - hurst_value) * 2)
            trading_signal = 'Mean Reversion'
        else:
            interpretation = 'Random Walk'
            behavior = 'No significant memory, random behavior'
            confidence = 1.0 - abs(hurst_value - 0.5) * 2
            trading_signal = 'Neutral'
        
        return {
            'hurst_value': hurst_value,
            'interpretation': interpretation,
            'behavior': behavior,
            'confidence': confidence,
            'trading_signal': trading_signal
        }

def test_hurst_exponent():
    """Test the Hurst Exponent indicator with sample data."""
    print("Testing Hurst Exponent Indicator")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_points = 200
    
    # Test 1: Random walk (should have H ≈ 0.5)
    random_walk = np.cumsum(np.random.randn(n_points))
    
    # Test 2: Trending data (should have H > 0.5)
    trend = np.arange(n_points) * 0.1 + np.random.randn(n_points) * 0.5
    
    # Test 3: Mean-reverting data (should have H < 0.5)
    mean_reverting = np.sin(np.arange(n_points) * 0.1) + np.random.randn(n_points) * 0.2
    
    # Initialize indicator
    hurst_rs = HurstExponent(window=100, method='rs')
    hurst_dfa = HurstExponent(window=100, method='dfa')
    
    # Test cases
    test_cases = [
        ("Random Walk", random_walk),
        ("Trending Data", trend), 
        ("Mean-Reverting Data", mean_reverting)
    ]
    
    for name, data in test_cases:
        print(f"\nTesting {name}:")
        print("-" * 30)
        
        # R/S Analysis
        hurst_rs_value = hurst_rs.calculate_single(data)
        interpretation_rs = hurst_rs.interpret_hurst(hurst_rs_value)
        
        print(f"R/S Analysis:")
        print(f"  Hurst Exponent: {hurst_rs_value:.4f}")
        print(f"  Interpretation: {interpretation_rs['interpretation']}")
        print(f"  Trading Signal: {interpretation_rs['trading_signal']}")
        print(f"  Confidence: {interpretation_rs['confidence']:.2f}")
        
        # DFA Analysis
        hurst_dfa_value = hurst_dfa.calculate_single(data)
        interpretation_dfa = hurst_dfa.interpret_hurst(hurst_dfa_value)
        
        print(f"DFA Analysis:")
        print(f"  Hurst Exponent: {hurst_dfa_value:.4f}")
        print(f"  Interpretation: {interpretation_dfa['interpretation']}")
        print(f"  Trading Signal: {interpretation_dfa['trading_signal']}")
        print(f"  Confidence: {interpretation_dfa['confidence']:.2f}")
    
    # Test rolling calculation
    print(f"\nTesting Rolling Calculation:")
    print("-" * 30)
    
    rolling_hurst = hurst_rs.calculate(trend)
    valid_values = rolling_hurst[~np.isnan(rolling_hurst)]
    
    if len(valid_values) > 0:
        print(f"Rolling Hurst (last 10 values): {valid_values[-10:]}")
        print(f"Mean Hurst: {np.mean(valid_values):.4f}")
        print(f"Std Hurst: {np.std(valid_values):.4f}")
    else:
        print("No valid rolling values calculated")
    
    print(f"\nTest completed successfully!")

if __name__ == "__main__":
    test_hurst_exponent()