"""
Multi-Fractal Detrended Fluctuation Analysis (MFDFA)
====================================================

Multi-scale market analysis using multifractal detrended fluctuation analysis.
Provides deep insights into market complexity and scaling behavior.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal
from ..indicator_base import IndicatorBase


class MultiFractalDFA(IndicatorBase):
    """
    Multi-Fractal Detrended Fluctuation Analysis (MFDFA) indicator.
    
    MFDFA analyzes the multifractal properties of financial time series,
    providing insights into market efficiency, volatility clustering,
    and scaling behavior across different time horizons.
    """
    
    def __init__(self, 
                 min_scale: int = 10,
                 max_scale: int = 100,
                 num_scales: int = 20,
                 q_orders: List[float] = None,
                 detrend_order: int = 1):
        """
        Initialize MFDFA indicator.
        
        Args:
            min_scale: Minimum scale for analysis
            max_scale: Maximum scale for analysis
            num_scales: Number of scales to analyze
            q_orders: List of q-order moments to calculate
            detrend_order: Order of polynomial detrending (1=linear, 2=quadratic)
        """
        super().__init__()
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_scales = num_scales
        self.q_orders = q_orders if q_orders else [-5, -3, -1, 0, 1, 3, 5]
        self.detrend_order = detrend_order
        
        # Validation
        if min_scale <= 0 or max_scale <= min_scale:
            raise ValueError("Invalid scale parameters")
        if num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if detrend_order < 1:
            raise ValueError("detrend_order must be >= 1")
        
        # Generate scale sequence
        self.scales = np.unique(np.logspace(
            np.log10(min_scale), 
            np.log10(max_scale), 
            num_scales
        ).astype(int))
    
    def _integrate_series(self, data: np.ndarray) -> np.ndarray:
        """
        Create integrated time series (profile).
        
        Args:
            data: Input time series
            
        Returns:
            Integrated time series
        """
        # Center the data by removing mean
        centered_data = data - np.mean(data)
        
        # Calculate cumulative sum (integration)
        return np.cumsum(centered_data)
    
    def _detrend_fluctuation(self, 
                           profile: np.ndarray, 
                           scale: int) -> np.ndarray:
        """
        Calculate detrended fluctuation for given scale.
        
        Args:
            profile: Integrated time series
            scale: Time scale for analysis
            
        Returns:
            Array of local fluctuations
        """
        n = len(profile)
        num_segments = n // scale
        fluctuations = []
        
        # Forward segments
        for i in range(num_segments):
            start_idx = i * scale
            end_idx = (i + 1) * scale
            segment = profile[start_idx:end_idx]
            
            # Polynomial detrending
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, self.detrend_order)
            trend = np.polyval(coeffs, x)
            
            # Calculate fluctuation
            fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
            fluctuations.append(fluctuation)
        
        # Backward segments (if data allows)
        if n % scale != 0:
            remaining = n % scale
            for i in range(num_segments):
                start_idx = n - (i + 1) * scale
                end_idx = n - i * scale
                segment = profile[start_idx:end_idx]
                
                x = np.arange(scale)
                coeffs = np.polyfit(x, segment, self.detrend_order)
                trend = np.polyval(coeffs, x)
                
                fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
                fluctuations.append(fluctuation)
        
        return np.array(fluctuations)
    
    def _calculate_q_order_fluctuation(self, 
                                     fluctuations: np.ndarray, 
                                     q: float) -> float:
        """
        Calculate q-order fluctuation function.
        
        Args:
            fluctuations: Array of local fluctuations
            q: Moment order
            
        Returns:
            Q-order fluctuation value
        """
        if len(fluctuations) == 0:
            return np.nan
        
        # Remove zero fluctuations to avoid issues
        valid_fluctuations = fluctuations[fluctuations > 0]
        
        if len(valid_fluctuations) == 0:
            return np.nan
        
        if q == 0:
            # Special case: geometric mean
            return np.exp(np.mean(np.log(valid_fluctuations)))
        else:
            # General case: q-th moment
            return np.power(np.mean(np.power(valid_fluctuations, q)), 1/q)
    
    def _calculate_hurst_exponents(self, 
                                 scales: np.ndarray, 
                                 fluctuations: np.ndarray) -> Dict[float, float]:
        """
        Calculate generalized Hurst exponents for all q orders.
        
        Args:
            scales: Array of time scales
            fluctuations: 2D array of fluctuations [q_index, scale_index]
            
        Returns:
            Dictionary mapping q orders to Hurst exponents
        """
        hurst_exponents = {}
        
        for i, q in enumerate(self.q_orders):
            # Get fluctuation function for this q
            fq = fluctuations[i, :]
            
            # Remove invalid values
            valid_mask = ~np.isnan(fq) & (fq > 0)
            if np.sum(valid_mask) < 3:
                hurst_exponents[q] = np.nan
                continue
            
            valid_scales = scales[valid_mask]
            valid_fq = fq[valid_mask]
            
            # Linear regression in log-log space
            try:
                log_scales = np.log(valid_scales)
                log_fq = np.log(valid_fq)
                
                # Fit line: log(F(q,s)) = H(q) * log(s) + const
                coeffs = np.polyfit(log_scales, log_fq, 1)
                hurst_exponents[q] = coeffs[0]  # Slope is Hurst exponent
            except:
                hurst_exponents[q] = np.nan
        
        return hurst_exponents
    
    def _calculate_multifractal_spectrum(self, 
                                       hurst_exponents: Dict[float, float]) -> Dict[str, np.ndarray]:
        """
        Calculate multifractal spectrum from Hurst exponents.
        
        Args:
            hurst_exponents: Dictionary of q -> H(q) mappings
            
        Returns:
            Dictionary with spectrum parameters
        """
        q_values = np.array(list(hurst_exponents.keys()))
        h_values = np.array(list(hurst_exponents.values()))
        
        # Remove NaN values
        valid_mask = ~np.isnan(h_values)
        q_values = q_values[valid_mask]
        h_values = h_values[valid_mask]
        
        if len(q_values) < 3:
            return {
                'alpha': np.array([]),
                'f_alpha': np.array([]),
                'width': np.nan,
                'asymmetry': np.nan
            }
        
        # Calculate tau(q) = q*H(q) - 1
        tau_values = q_values * h_values - 1
        
        # Calculate alpha and f(alpha) using Legendre transform
        alpha_values = np.gradient(tau_values, q_values)
        f_alpha_values = q_values * alpha_values - tau_values
        
        # Spectrum characteristics
        if len(alpha_values) > 0:
            spectrum_width = np.max(alpha_values) - np.min(alpha_values)
            
            # Asymmetry: difference between left and right sides
            alpha_0_idx = np.argmax(f_alpha_values)
            if 0 < alpha_0_idx < len(alpha_values) - 1:
                left_width = alpha_values[alpha_0_idx] - np.min(alpha_values)
                right_width = np.max(alpha_values) - alpha_values[alpha_0_idx]
                asymmetry = (right_width - left_width) / spectrum_width
            else:
                asymmetry = np.nan
        else:
            spectrum_width = np.nan
            asymmetry = np.nan
        
        return {
            'alpha': alpha_values,
            'f_alpha': f_alpha_values,
            'width': spectrum_width,
            'asymmetry': asymmetry
        }
    
    def calculate(self, 
                 data: pd.DataFrame,
                 price_column: str = 'close',
                 window_size: int = 200) -> pd.DataFrame:
        """
        Calculate MFDFA indicator.
        
        Args:
            data: DataFrame with price data
            price_column: Column name for price data
            window_size: Rolling window size for analysis
            
        Returns:
            DataFrame with multifractal analysis results
        """
        if len(data) < window_size:
            raise ValueError(f"Insufficient data. Need at least {window_size} rows")
        
        prices = data[price_column].values
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        results = []
        
        for i in range(window_size - 1, len(data)):
            # Get window of returns
            start_idx = i - window_size + 1
            window_returns = log_returns[start_idx:i]
            
            if len(window_returns) < self.min_scale * 2:
                # Insufficient data
                results.append({
                    'hurst_h2': np.nan,
                    'hurst_h0': np.nan,
                    'hurst_h_neg2': np.nan,
                    'spectrum_width': np.nan,
                    'asymmetry': np.nan,
                    'multifractality': np.nan,
                    'efficiency': np.nan
                })
                continue
            
            # Step 1: Create integrated profile
            profile = self._integrate_series(window_returns)
            
            # Step 2: Calculate fluctuation functions for all scales and q orders
            fluctuation_matrix = np.full((len(self.q_orders), len(self.scales)), np.nan)
            
            for scale_idx, scale in enumerate(self.scales):
                if scale >= len(profile) // 4:  # Skip if scale too large
                    continue
                    
                fluctuations = self._detrend_fluctuation(profile, scale)
                
                for q_idx, q in enumerate(self.q_orders):
                    fq = self._calculate_q_order_fluctuation(fluctuations, q)
                    fluctuation_matrix[q_idx, scale_idx] = fq
            
            # Step 3: Calculate Hurst exponents
            hurst_exponents = self._calculate_hurst_exponents(
                self.scales, fluctuation_matrix
            )
            
            # Step 4: Calculate multifractal spectrum
            spectrum = self._calculate_multifractal_spectrum(hurst_exponents)
            
            # Extract key metrics
            h2 = hurst_exponents.get(2, np.nan)
            h0 = hurst_exponents.get(0, np.nan)
            h_neg2 = hurst_exponents.get(-2, np.nan)
            
            # Multifractality measure
            multifractality = spectrum['width'] if not np.isnan(spectrum['width']) else 0
            
            # Market efficiency (based on H(2))
            if not np.isnan(h2):
                efficiency = abs(h2 - 0.5)  # Distance from random walk
            else:
                efficiency = np.nan
            
            results.append({
                'hurst_h2': h2,
                'hurst_h0': h0,
                'hurst_h_neg2': h_neg2,
                'spectrum_width': spectrum['width'],
                'asymmetry': spectrum['asymmetry'],
                'multifractality': multifractality,
                'efficiency': efficiency
            })
        
        # Create result DataFrame
        result_df = pd.DataFrame(results)
        
        # Pad with NaN for the initial window
        pad_rows = window_size - 1
        for col in result_df.columns:
            result_df = pd.concat([
                pd.DataFrame({col: [np.nan] * pad_rows}),
                result_df
            ], ignore_index=True)
        
        return result_df
    
    def get_signals(self, 
                   indicator_data: pd.DataFrame,
                   efficiency_threshold: float = 0.1,
                   multifractal_threshold: float = 0.3) -> pd.DataFrame:
        """
        Generate trading signals based on MFDFA analysis.
        
        Args:
            indicator_data: DataFrame from calculate() method
            efficiency_threshold: Threshold for market efficiency
            multifractal_threshold: Threshold for multifractality
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=indicator_data.index)
        
        # Market regime signals
        signals['trending_market'] = (
            (indicator_data['hurst_h2'] > 0.6) &
            (indicator_data['efficiency'] > efficiency_threshold)
        ).astype(int)
        
        signals['mean_reverting'] = (
            (indicator_data['hurst_h2'] < 0.4) &
            (indicator_data['efficiency'] > efficiency_threshold)
        ).astype(int)
        
        signals['random_walk'] = (
            (indicator_data['hurst_h2'].between(0.45, 0.55)) &
            (indicator_data['efficiency'] < efficiency_threshold)
        ).astype(int)
        
        # Multifractality signals
        signals['high_multifractality'] = (
            indicator_data['multifractality'] > multifractal_threshold
        ).astype(int)
        
        signals['low_multifractality'] = (
            indicator_data['multifractality'] < multifractal_threshold / 2
        ).astype(int)
        
        # Asymmetry signals
        signals['positive_asymmetry'] = (
            indicator_data['asymmetry'] > 0.1
        ).astype(int)
        
        signals['negative_asymmetry'] = (
            indicator_data['asymmetry'] < -0.1
        ).astype(int)
        
        return signals
    
    def get_interpretation(self, latest_values: Dict) -> str:
        """
        Provide interpretation of current MFDFA state.
        
        Args:
            latest_values: Dictionary with latest indicator values
            
        Returns:
            String interpretation
        """
        h2 = latest_values.get('hurst_h2', np.nan)
        multifractality = latest_values.get('multifractality', np.nan)
        efficiency = latest_values.get('efficiency', np.nan)
        asymmetry = latest_values.get('asymmetry', np.nan)
        
        if np.isnan(h2):
            return "Insufficient data for MFDFA analysis."
        
        # Market behavior
        if h2 > 0.6:
            behavior = "persistent/trending"
        elif h2 < 0.4:
            behavior = "anti-persistent/mean-reverting"
        else:
            behavior = "random walk"
        
        # Multifractality
        if multifractality > 0.3:
            mf_desc = "high multifractality"
        elif multifractality > 0.1:
            mf_desc = "moderate multifractality"
        else:
            mf_desc = "low multifractality"
        
        # Efficiency
        if efficiency > 0.2:
            eff_desc = "inefficient"
        elif efficiency > 0.1:
            eff_desc = "moderately efficient"
        else:
            eff_desc = "efficient"
        
        return f"Market shows {behavior} behavior (H={h2:.2f}) with {mf_desc} " \
               f"(W={multifractality:.2f}). Market is {eff_desc} " \
               f"(eff={efficiency:.2f}). Asymmetry: {asymmetry:.2f}."


def create_mfdfa_indicator(min_scale: int = 10,
                          max_scale: int = 100,
                          num_scales: int = 20,
                          q_orders: List[float] = None,
                          detrend_order: int = 1) -> MultiFractalDFA:
    """
    Factory function to create MFDFA indicator.
    
    Args:
        min_scale: Minimum scale for analysis
        max_scale: Maximum scale for analysis
        num_scales: Number of scales to analyze
        q_orders: List of q-order moments to calculate
        detrend_order: Order of polynomial detrending
        
    Returns:
        Configured MultiFractalDFA instance
    """
    return MultiFractalDFA(
        min_scale=min_scale,
        max_scale=max_scale,
        num_scales=num_scales,
        q_orders=q_orders,
        detrend_order=detrend_order
    )
