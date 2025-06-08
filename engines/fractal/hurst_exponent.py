"""
Hurst Exponent Calculator
========================

Measures price series persistence and mean-reversion using advanced Hurst exponent calculations.
Detects long-term memory in time series and forecasts trend sustainability.

Author: Platform3 AI System
Created: June 6, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
import logging

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame


class HurstExponentCalculator(IndicatorBase):
    """
    Hurst Exponent Calculator for time series memory analysis.
    
    The Hurst exponent measures the long-term memory of a time series through
    its autocorrelations. It helps determine if price movements are:
    - Random walk (H ≈ 0.5)
    - Trending/persistent (H > 0.5)
    - Mean-reverting/anti-persistent (H < 0.5)
    
    Multiple methods implemented for robust estimation.
    """
    
    def __init__(self, 
                 min_window_size: int = 100,
                 max_window_size: int = 2000,
                 num_divisions: int = 10,
                 method: str = 'all'):
        """
        Initialize Hurst Exponent Calculator.
        
        Args:
            min_window_size: Minimum window size for analysis
            max_window_size: Maximum window size for analysis
            num_divisions: Number of window size divisions to use
            method: Calculation method ('rs', 'dma', 'all', or 'wavelet')
        """
        super().__init__()
        
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.num_divisions = num_divisions
        self.method = method
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized HurstExponentCalculator with method '{method}'")
        
        # Set up window sizes
        self._setup_window_sizes()
    
    def _setup_window_sizes(self):
        """Set up window sizes for analysis."""
        # Use logarithmic spacing for window sizes
        window_sizes = np.logspace(
            np.log10(self.min_window_size),
            np.log10(self.max_window_size),
            self.num_divisions
        ).astype(int)
        
        # Ensure unique and sorted
        self.window_sizes = np.unique(window_sizes)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Hurst exponent for price series.
        
        Args:
            data: DataFrame with price data (must contain 'close')
            
        Returns:
            Dictionary containing Hurst exponent values and interpretations
        """
        try:
            # Validate input
            self._validate_data(data, ['close'])
            
            if len(data) < self.min_window_size:
                raise ValueError(f"Insufficient data: need at least {self.min_window_size} periods")
            
            # Get price series
            prices = data['close'].values
            
            # Calculate log returns
            log_returns = np.diff(np.log(prices))
            
            # Calculate Hurst exponent using different methods
            results = {}
            
            if self.method in ['rs', 'all']:
                results['hurst_rs'] = self._calculate_rs(log_returns)
            
            if self.method in ['dma', 'all']:
                results['hurst_dma'] = self._calculate_dma(log_returns)
            
            if self.method in ['wavelet', 'all']:
                results['hurst_wavelet'] = self._calculate_wavelet(log_returns)
            
            # Calculate consensus Hurst value
            results['hurst'] = self._calculate_consensus(results)
            
            # Add market interpretation
            results.update(self._interpret_hurst(results['hurst']))
            
            # Add additional metrics
            results.update(self._calculate_additional_metrics(log_returns, results['hurst']))
            
            # Create signal
            signal = self._generate_signal(results['hurst'], results['trend_strength'])
            
            # Create properly constructed IndicatorResult
            from datetime import datetime
            return IndicatorResult(
                timestamp=datetime.now(),
                indicator_name=self.__class__.__name__,
                indicator_type=IndicatorType.FRACTAL,
                timeframe=TimeFrame.D1,
                value=results['hurst'],
                signal=signal,
                raw_data=results
            )
            
        except Exception as e:
            self.logger.error(f"Error in Hurst exponent calculation: {e}")
            raise
    
    def _calculate_rs(self, time_series: np.ndarray) -> float:
        """
        Calculate Hurst exponent using Rescaled Range (R/S) method.
        
        Args:
            time_series: Input time series data
            
        Returns:
            Hurst exponent value
        """
        # Calculate for different window sizes
        rs_values = []
        
        for window in self.window_sizes:
            if window >= len(time_series):
                break
                
            # Calculate average R/S value for this window size
            rs_window = []
            
            # Use non-overlapping windows
            n_windows = len(time_series) // window
            
            if n_windows == 0:
                continue
                
            for i in range(n_windows):
                start_idx = i * window
                end_idx = (i + 1) * window
                
                # Extract segment
                segment = time_series[start_idx:end_idx]
                
                # Calculate rescaled range
                segment_mean = np.mean(segment)
                segment_std = np.std(segment)
                
                # Cumulative deviate series
                cumulative_deviate = np.cumsum(segment - segment_mean)
                
                # Range
                segment_range = np.max(cumulative_deviate) - np.min(cumulative_deviate)
                
                # Rescaled range (avoid division by zero)
                if segment_std > 0:
                    rs = segment_range / segment_std
                    rs_window.append(rs)
            
            if rs_window:
                rs_values.append((window, np.mean(rs_window)))
        
        # Log-log regression to find Hurst exponent
        if len(rs_values) > 1:
            x_vals = np.log10([x[0] for x in rs_values])
            y_vals = np.log10([x[1] for x in rs_values])
            
            slope, _, r_value, _, _ = stats.linregress(x_vals, y_vals)
            
            # A good R/S fit should have a high r value
            if r_value > 0.9:
                return slope
        
        # Default value if calculation fails
        return 0.5
    
    def _calculate_dma(self, time_series: np.ndarray) -> float:
        """
        Calculate Hurst exponent using Detrended Moving Average (DMA) method.
        
        Args:
            time_series: Input time series data
            
        Returns:
            Hurst exponent value
        """
        # Calculate for different window sizes
        dma_values = []
        
        for window in self.window_sizes:
            if window >= len(time_series) or window < 3:
                continue
                
            # Calculate moving average
            ma = np.convolve(time_series, np.ones(window)/window, mode='valid')
            
            # Calculate fluctuation
            y = time_series[window-1:window-1+len(ma)]
            fluctuation = np.sqrt(np.mean((y - ma)**2))
            
            dma_values.append((window, fluctuation))
        
        # Log-log regression
        if len(dma_values) > 1:
            x_vals = np.log10([x[0] for x in dma_values])
            y_vals = np.log10([x[1] for x in dma_values])
            
            slope, _, r_value, _, _ = stats.linregress(x_vals, y_vals)
            
            return abs(slope)
        
        return 0.5
    
    def _calculate_wavelet(self, time_series: np.ndarray) -> float:
        """
        Calculate Hurst exponent using wavelet-based method.
        
        Args:
            time_series: Input time series data
            
        Returns:
            Hurst exponent value
        """
        # Simplified wavelet transform method
        try:
            # Determine maximum dyadic length
            dyadic_length = 2**int(np.floor(np.log2(len(time_series))))
            series = time_series[:dyadic_length]
            
            # Calculate wavelet coefficients at different scales
            scales = []
            variances = []
            
            j_max = int(np.log2(dyadic_length)) - 2
            
            for j in range(1, min(j_max, 8)):
                scale = 2**j
                scales.append(scale)
                
                # Simple Haar wavelet transform
                coeffs = []
                for i in range(0, len(series), 2*scale):
                    if i + 2*scale <= len(series):
                        # Calculate wavelet coefficient
                        c = np.mean(series[i:i+scale]) - np.mean(series[i+scale:i+2*scale])
                        coeffs.append(c)
                
                if coeffs:
                    variances.append(np.var(coeffs))
            
            if len(scales) > 1 and all(v > 0 for v in variances):
                # Log-log regression
                x_vals = np.log10(scales)
                y_vals = np.log10(variances)
                
                slope, _, _, _, _ = stats.linregress(x_vals, y_vals)
                
                # Convert wavelet slope to Hurst exponent
                h = (slope + 1) / 2
                return max(0, min(h, 1))
        except:
            pass
        
        return 0.5
    
    def _calculate_consensus(self, results: Dict) -> float:
        """
        Calculate consensus Hurst value from different methods.
        
        Args:
            results: Dictionary with Hurst values from different methods
            
        Returns:
            Consensus Hurst value
        """
        hurst_values = []
        
        if 'hurst_rs' in results:
            hurst_values.append(results['hurst_rs'])
            
        if 'hurst_dma' in results:
            hurst_values.append(results['hurst_dma'])
            
        if 'hurst_wavelet' in results:
            hurst_values.append(results['hurst_wavelet'])
            
        # Weight R/S method higher if it's available
        if 'hurst_rs' in results:
            hurst_values.append(results['hurst_rs'])
        
        if not hurst_values:
            return 0.5
            
        # Remove outliers
        if len(hurst_values) >= 3:
            mean_h = np.mean(hurst_values)
            std_h = np.std(hurst_values)
            hurst_values = [h for h in hurst_values if abs(h - mean_h) <= 1.5 * std_h]
        
        # Calculate weighted average
        return np.mean(hurst_values)
    
    def _interpret_hurst(self, hurst: float) -> Dict:
        """
        Interpret Hurst exponent value.
        
        Args:
            hurst: Calculated Hurst exponent
            
        Returns:
            Dictionary with interpretation values
        """
        # Determine market type
        if hurst > 0.6:
            market_type = "trending"
            memory_type = "persistent"
            description = "Long-term memory with trend-following behavior"
        elif hurst < 0.4:
            market_type = "mean_reverting"
            memory_type = "anti-persistent"
            description = "Short-term mean reversion behavior"
        else:
            market_type = "random_walk"
            memory_type = "efficient"
            description = "Efficient market with minimal predictability"
        
        # Calculate trend persistence
        trend_strength = abs(hurst - 0.5) * 2  # Normalize to [0, 1]
        
        # Recommended trading strategy
        if hurst > 0.65:
            strategy = "trend_following"
        elif hurst < 0.35:
            strategy = "mean_reversion"
        else:
            strategy = "market_neutral"
        
        return {
            'market_type': market_type,
            'memory_type': memory_type,
            'description': description,
            'trend_strength': trend_strength,
            'recommended_strategy': strategy
        }
    
    def _calculate_additional_metrics(self, returns: np.ndarray, hurst: float) -> Dict:
        """
        Calculate additional metrics based on Hurst exponent.
        
        Args:
            returns: Return series
            hurst: Calculated Hurst exponent
            
        Returns:
            Dictionary with additional metrics
        """
        # Fractal dimension (D = 2 - H)
        fractal_dimension = 2.0 - hurst
        
        # Forecast horizon estimate
        if hurst > 0.5:
            forecast_horizon = int(10 * (hurst - 0.5) * 20) + 1  # More persistent = longer horizon
        else:
            forecast_horizon = int((0.5 - hurst) * 10) + 1  # More mean-reverting = shorter horizon
        
        # Volatility adjustments
        volatility = np.std(returns)
        fractal_volatility = volatility * (len(returns) ** (hurst - 0.5))
        
        return {
            'fractal_dimension': fractal_dimension,
            'forecast_horizon': forecast_horizon,
            'standard_volatility': volatility,
            'fractal_volatility': fractal_volatility
        }
    
    def _generate_signal(self, hurst: float, trend_strength: float) -> float:
        """
        Generate trading signal based on Hurst exponent.
        
        Args:
            hurst: Hurst exponent value
            trend_strength: Strength of trend signal
            
        Returns:
            Signal value between -1 and 1
        """
        # Strong trend following signal
        if hurst > 0.7 and trend_strength > 0.6:
            return 1.0
            
        # Moderate trend following signal
        elif hurst > 0.6:
            return 0.5
            
        # Strong mean reversion signal
        elif hurst < 0.3 and trend_strength > 0.6:
            return -1.0
            
        # Moderate mean reversion signal
        elif hurst < 0.4:
            return -0.5
            
        # No clear signal
        else:
            return 0.0


def create_hurst_exponent_calculator(
    min_window_size: int = 100,
    max_window_size: int = 2000,
    num_divisions: int = 10,
    method: str = 'all'
) -> HurstExponentCalculator:
    """
    Factory function to create Hurst Exponent Calculator.
    
    Args:
        min_window_size: Minimum window size for analysis
        max_window_size: Maximum window size for analysis
        num_divisions: Number of window size divisions to use
        method: Calculation method ('rs', 'dma', 'all', or 'wavelet')
        
    Returns:
        Configured HurstExponentCalculator instance
    """
    return HurstExponentCalculator(
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        num_divisions=num_divisions,
        method=method
    )