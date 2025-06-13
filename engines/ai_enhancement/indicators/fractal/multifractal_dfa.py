"""
Multi-Fractal Detrended Fluctuation Analysis (MFDFA) for Platform3

This indicator implements Multi-Fractal Detrended Fluctuation Analysis to analyze 
the multi-fractal properties of financial time series. It helps identify
long-range correlations, regime changes, and market efficiency.

Author: Platform3 Development Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import signal
    from scipy.stats import linregress
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class MultiFractalDFA:
    """
    Multi-Fractal Detrended Fluctuation Analysis
    
    MFDFA is an extension of regular DFA that analyzes the multi-fractal
    scaling properties of time series data. It provides insights into:
    
    - Multi-fractal spectrum
    - Generalized Hurst exponents
    - Singularity spectrum
    - Market efficiency measures
    - Long-range dependence
    - Fat-tail behavior
    
    The analysis reveals different fractal behaviors at different scales
    and time horizons, which is crucial for understanding market dynamics.
    """
    
    def __init__(self,
                 min_box_size: int = 10,
                 max_box_size: int = None,
                 n_boxes: int = 20,
                 q_range: Tuple[float, float] = (-5, 5),
                 n_q: int = 21,
                 polynomial_order: int = 1,
                 overlap: bool = True):
        """
        Initialize Multi-Fractal DFA
        
        Args:
            min_box_size: Minimum window size for analysis
            max_box_size: Maximum window size (None = auto)
            n_boxes: Number of box sizes to analyze
            q_range: Range of q values for multi-fractal analysis
            n_q: Number of q values to use
            polynomial_order: Order of polynomial for detrending
            overlap: Whether to use overlapping windows
        """
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.n_boxes = n_boxes
        self.q_range = q_range
        self.n_q = n_q
        self.polynomial_order = polynomial_order
        self.overlap = overlap
        
        # Generate q values
        self.q_values = np.linspace(q_range[0], q_range[1], n_q)
        
        # Storage for analysis results
        self.box_sizes = None
        self.fluctuation_functions = None
        self.hurst_exponents = None
        self.singularity_spectrum = None
        self.multifractal_spectrum = None
        
    def _create_profile(self, series: np.ndarray) -> np.ndarray:
        """
        Create cumulative profile from time series
        
        Args:
            series: Input time series
            
        Returns:
            Cumulative profile
        """
        # Remove mean and create cumulative sum
        mean_centered = series - np.mean(series)
        profile = np.cumsum(mean_centered)
        
        return profile
    
    def _detrend_fluctuation(self, profile: np.ndarray, box_size: int) -> float:
        """
        Calculate detrended fluctuation for a given box size
        
        Args:
            profile: Cumulative profile
            box_size: Window size for analysis
            
        Returns:
            RMS fluctuation
        """
        n = len(profile)
        
        if self.overlap:
            # Overlapping windows
            n_windows = n - box_size + 1
            fluctuations = []
            
            for i in range(n_windows):
                # Extract window
                window = profile[i:i + box_size]
                
                # Fit polynomial trend
                x = np.arange(len(window))
                if SCIPY_AVAILABLE:
                    trend = np.polyval(np.polyfit(x, window, self.polynomial_order), x)
                else:
                    # Simple linear trend as fallback
                    if self.polynomial_order == 1:
                        slope, intercept = np.polyfit(x, window, 1)
                        trend = slope * x + intercept
                    else:
                        trend = np.mean(window)  # Just use mean for higher orders
                
                # Calculate detrended fluctuation
                detrended = window - trend
                fluctuation = np.sqrt(np.mean(detrended**2))
                fluctuations.append(fluctuation)
            
            # Average fluctuation
            avg_fluctuation = np.mean(fluctuations)
            
        else:
            # Non-overlapping windows
            n_windows = n // box_size
            fluctuations = []
            
            for i in range(n_windows):
                start_idx = i * box_size
                end_idx = start_idx + box_size
                
                # Extract window
                window = profile[start_idx:end_idx]
                
                # Fit polynomial trend
                x = np.arange(len(window))
                if SCIPY_AVAILABLE:
                    trend = np.polyval(np.polyfit(x, window, self.polynomial_order), x)
                else:
                    # Simple linear trend as fallback
                    if self.polynomial_order == 1:
                        slope, intercept = np.polyfit(x, window, 1)
                        trend = slope * x + intercept
                    else:
                        trend = np.mean(window)
                
                # Calculate detrended fluctuation
                detrended = window - trend
                fluctuation = np.sqrt(np.mean(detrended**2))
                fluctuations.append(fluctuation)
            
            # Average fluctuation
            avg_fluctuation = np.mean(fluctuations)
        
        return avg_fluctuation
    
    def _calculate_generalized_fluctuation(self, profile: np.ndarray, 
                                         box_size: int, q: float) -> float:
        """
        Calculate generalized fluctuation function F_q(s)
        
        Args:
            profile: Cumulative profile
            box_size: Window size
            q: Moment order
            
        Returns:
            Generalized fluctuation function
        """
        n = len(profile)
        
        if self.overlap:
            # Overlapping windows
            n_windows = n - box_size + 1
            fluctuations = []
            
            for i in range(n_windows):
                # Extract window
                window = profile[i:i + box_size]
                
                # Fit polynomial trend
                x = np.arange(len(window))
                if SCIPY_AVAILABLE:
                    trend = np.polyval(np.polyfit(x, window, self.polynomial_order), x)
                else:
                    if self.polynomial_order == 1:
                        slope, intercept = np.polyfit(x, window, 1)
                        trend = slope * x + intercept
                    else:
                        trend = np.mean(window)
                
                # Calculate detrended fluctuation
                detrended = window - trend
                fluctuation = np.sqrt(np.mean(detrended**2))
                fluctuations.append(fluctuation)
            
        else:
            # Non-overlapping windows
            n_windows = n // box_size
            fluctuations = []
            
            for i in range(n_windows):
                start_idx = i * box_size
                end_idx = start_idx + box_size
                
                window = profile[start_idx:end_idx]
                x = np.arange(len(window))
                
                if SCIPY_AVAILABLE:
                    trend = np.polyval(np.polyfit(x, window, self.polynomial_order), x)
                else:
                    if self.polynomial_order == 1:
                        slope, intercept = np.polyfit(x, window, 1)
                        trend = slope * x + intercept
                    else:
                        trend = np.mean(window)
                
                detrended = window - trend
                fluctuation = np.sqrt(np.mean(detrended**2))
                fluctuations.append(fluctuation)
        
        # Calculate generalized fluctuation function
        fluctuations = np.array(fluctuations)
        fluctuations = fluctuations[fluctuations > 0]  # Remove zeros
        
        if len(fluctuations) == 0:
            return 0.0
        
        if q == 0:
            # Special case for q=0 (logarithmic average)
            f_q = np.exp(np.mean(np.log(fluctuations)))
        else:
            # General case
            f_q = np.power(np.mean(np.power(fluctuations, q)), 1.0/q)
        
        return f_q
    
    def _generate_box_sizes(self, n_data: int) -> np.ndarray:
        """
        Generate logarithmically spaced box sizes
        
        Args:
            n_data: Length of data series
            
        Returns:
            Array of box sizes
        """
        max_size = self.max_box_size or min(n_data // 4, 200)
        max_size = min(max_size, n_data // 2)
        
        # Generate logarithmically spaced sizes
        log_min = np.log10(self.min_box_size)
        log_max = np.log10(max_size)
        
        box_sizes = np.logspace(log_min, log_max, self.n_boxes, dtype=int)
        box_sizes = np.unique(box_sizes)  # Remove duplicates
        box_sizes = box_sizes[box_sizes >= self.min_box_size]
        box_sizes = box_sizes[box_sizes <= max_size]
        
        return box_sizes
    
    def analyze(self, data: Union[np.ndarray, pd.Series]) -> Dict[str, any]:
        """
        Perform Multi-Fractal DFA analysis
        
        Args:
            data: Time series data
            
        Returns:
            MFDFA analysis results
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            series = data.values
        else:
            series = np.array(data)
        
        if len(series) < self.min_box_size * 2:
            return {'error': 'Insufficient data for MFDFA analysis'}
        
        # Create cumulative profile
        profile = self._create_profile(series)
        
        # Generate box sizes
        self.box_sizes = self._generate_box_sizes(len(series))
        
        if len(self.box_sizes) < 3:
            return {'error': 'Insufficient box sizes for analysis'}
        
        # Calculate fluctuation functions for all q values
        fluctuation_matrix = np.zeros((len(self.q_values), len(self.box_sizes)))
        
        for i, q in enumerate(self.q_values):
            for j, box_size in enumerate(self.box_sizes):
                f_q = self._calculate_generalized_fluctuation(profile, box_size, q)
                fluctuation_matrix[i, j] = f_q
        
        self.fluctuation_functions = fluctuation_matrix
        
        # Calculate generalized Hurst exponents
        self.hurst_exponents = np.zeros(len(self.q_values))
        
        for i, q in enumerate(self.q_values):
            # Linear regression in log-log space
            log_sizes = np.log10(self.box_sizes)
            log_fluctuations = np.log10(fluctuation_matrix[i])
            
            # Remove infinite values
            valid_mask = np.isfinite(log_fluctuations) & (log_fluctuations > -np.inf)
            
            if np.sum(valid_mask) >= 3:
                if SCIPY_AVAILABLE:
                    slope, _, r_value, _, _ = linregress(
                        log_sizes[valid_mask], 
                        log_fluctuations[valid_mask]
                    )
                else:
                    slope, _ = np.polyfit(
                        log_sizes[valid_mask], 
                        log_fluctuations[valid_mask], 
                        1
                    )
                    r_value = np.corrcoef(
                        log_sizes[valid_mask], 
                        log_fluctuations[valid_mask]
                    )[0, 1]
                
                self.hurst_exponents[i] = slope
            else:
                self.hurst_exponents[i] = 0.5  # Default value
        
        # Calculate scaling exponent τ(q)
        tau_q = self.q_values * self.hurst_exponents - 1
        
        # Calculate singularity spectrum via Legendre transform
        alpha = np.gradient(tau_q, self.q_values)  # α = dτ/dq
        f_alpha = self.q_values * alpha - tau_q     # f(α) = qα - τ(q)
        
        self.singularity_spectrum = {
            'alpha': alpha,
            'f_alpha': f_alpha,
            'tau_q': tau_q
        }
        
        # Multi-fractal characteristics
        multifractal_spectrum = self._calculate_multifractal_characteristics(alpha, f_alpha)
        
        return {
            'box_sizes': self.box_sizes,
            'q_values': self.q_values,
            'hurst_exponents': self.hurst_exponents,
            'fluctuation_functions': self.fluctuation_functions,
            'singularity_spectrum': self.singularity_spectrum,
            'multifractal_spectrum': multifractal_spectrum,
            'is_multifractal': multifractal_spectrum['is_multifractal'],
            'multifractal_strength': multifractal_spectrum['strength']
        }
    
    def _calculate_multifractal_characteristics(self, alpha: np.ndarray, 
                                             f_alpha: np.ndarray) -> Dict[str, any]:
        """
        Calculate multi-fractal characteristics
        
        Args:
            alpha: Singularity exponents
            f_alpha: Singularity spectrum
            
        Returns:
            Multi-fractal characteristics
        """
        # Remove NaN values
        valid_mask = np.isfinite(alpha) & np.isfinite(f_alpha)
        alpha_clean = alpha[valid_mask]
        f_alpha_clean = f_alpha[valid_mask]
        
        if len(alpha_clean) < 3:
            return {
                'is_multifractal': False,
                'strength': 0.0,
                'width': 0.0,
                'asymmetry': 0.0,
                'peak_alpha': 0.5,
                'peak_f_alpha': 1.0
            }
        
        # Width of singularity spectrum
        alpha_min = np.min(alpha_clean)
        alpha_max = np.max(alpha_clean)
        width = alpha_max - alpha_min
        
        # Peak of spectrum
        peak_idx = np.argmax(f_alpha_clean)
        peak_alpha = alpha_clean[peak_idx]
        peak_f_alpha = f_alpha_clean[peak_idx]
        
        # Asymmetry
        left_width = peak_alpha - alpha_min
        right_width = alpha_max - peak_alpha
        
        if (left_width + right_width) > 0:
            asymmetry = (right_width - left_width) / (right_width + left_width)
        else:
            asymmetry = 0.0
        
        # Multi-fractal strength
        strength = width
        
        # Determine if multi-fractal
        is_multifractal = width > 0.1  # Threshold for multi-fractality
        
        return {
            'is_multifractal': is_multifractal,
            'strength': strength,
            'width': width,
            'asymmetry': asymmetry,
            'peak_alpha': peak_alpha,
            'peak_f_alpha': peak_f_alpha,
            'alpha_min': alpha_min,
            'alpha_max': alpha_max
        }
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate MFDFA signals for backtesting
        
        Args:
            data: Market data with price information
            
        Returns:
            Array of multi-fractal strength signals
        """
        if 'close' not in data.columns:
            return np.zeros(len(data))
        
        signals = np.zeros(len(data))
        window_size = max(100, self.min_box_size * 5)  # Minimum window for analysis
        
        # Calculate rolling MFDFA
        for i in range(window_size, len(data)):
            # Extract window
            window_data = data['close'].iloc[i-window_size:i]
            
            # Analyze multi-fractal properties
            result = self.analyze(window_data)
            
            if 'error' not in result:
                # Use multi-fractal strength as signal
                strength = result['multifractal_spectrum']['strength']
                signals[i] = strength
            else:
                signals[i] = 0.0
        
        return signals
    
    def get_efficiency_measure(self) -> float:
        """
        Get market efficiency measure based on Hurst exponent
        
        Returns:
            Efficiency measure (0 = random walk, 1 = most efficient)
        """
        if self.hurst_exponents is None:
            return 0.5
        
        # Use Hurst exponent for q=2 (closest to classical DFA)
        h2_idx = np.argmin(np.abs(self.q_values - 2.0))
        h2 = self.hurst_exponents[h2_idx]
        
        # Convert to efficiency measure
        # H = 0.5 (random walk) is most efficient
        # H > 0.5 or H < 0.5 indicates inefficiency
        efficiency = 1.0 - 2.0 * abs(h2 - 0.5)
        
        return max(0.0, efficiency)


# Test and example usage
if __name__ == "__main__":
    print("Testing Multi-Fractal DFA Indicator...")
    
    # Generate sample data with different fractal properties
    np.random.seed(42)
    n_points = 500
    
    # Create multi-fractal time series
    # Combine different fractal components
    t = np.arange(n_points)
    
    # Base random walk
    random_walk = np.cumsum(np.random.randn(n_points))
    
    # Add long-range dependence
    # Fractional Brownian motion approximation
    h = 0.7  # Hurst parameter
    noise = np.random.randn(n_points)
    
    # Create correlated noise (simplified fBm)
    fbm = np.zeros(n_points)
    for i in range(1, n_points):
        fbm[i] = fbm[i-1] + noise[i] + 0.3 * noise[i-1]
    
    # Add multi-fractal component
    # Scale-dependent volatility
    volatility = 1 + 0.5 * np.sin(t * 0.01) + 0.3 * np.sin(t * 0.05)
    multifractal_component = volatility * np.random.randn(n_points)
    
    # Combine components
    series = random_walk + 0.5 * fbm + 0.3 * np.cumsum(multifractal_component)
    
    # Add trend and convert to price-like data
    trend = 0.001 * t
    prices = 100 * np.exp(trend + 0.01 * series / np.std(series))
    
    # Create DataFrame
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    print(f"Generated {n_points} data points")
    print(f"Price range: {prices.min():.2f} to {prices.max():.2f}")
    
    # Initialize MFDFA
    mfdfa = MultiFractalDFA(
        min_box_size=10,
        max_box_size=100,
        n_boxes=20,
        q_range=(-3, 3),
        n_q=15
    )
    
    print(f"Initialized MFDFA with {mfdfa.n_q} q-values")
    print(f"Q-range: [{mfdfa.q_range[0]}, {mfdfa.q_range[1]}]")
    print(f"Using {'scipy' if SCIPY_AVAILABLE else 'basic'} polynomial fitting")
    
    # Perform MFDFA analysis
    print("\nPerforming MFDFA analysis...")
    result = mfdfa.analyze(data['close'])
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print("MFDFA Analysis Results:")
        print(f"  Box sizes analyzed: {len(result['box_sizes'])}")
        print(f"  Box size range: {result['box_sizes'][0]} to {result['box_sizes'][-1]}")
        
        # Hurst exponents
        print(f"\nGeneralized Hurst Exponents:")
        for i, (q, h) in enumerate(zip(result['q_values'], result['hurst_exponents'])):
            if i % 3 == 0:  # Print every 3rd value
                print(f"  H({q:.1f}) = {h:.4f}")
        
        # Multi-fractal characteristics
        mf_spec = result['multifractal_spectrum']
        print(f"\nMulti-fractal Properties:")
        print(f"  Is multi-fractal: {mf_spec['is_multifractal']}")
        print(f"  Multi-fractal strength: {mf_spec['strength']:.4f}")
        print(f"  Spectrum width: {mf_spec['width']:.4f}")
        print(f"  Asymmetry: {mf_spec['asymmetry']:.4f}")
        print(f"  Peak alpha: {mf_spec['peak_alpha']:.4f}")
        print(f"  Peak f(alpha): {mf_spec['peak_f_alpha']:.4f}")
        
        # Singularity spectrum
        sing_spec = result['singularity_spectrum']
        print(f"\nSingularity Spectrum:")
        print(f"  Alpha range: [{np.min(sing_spec['alpha']):.4f}, {np.max(sing_spec['alpha']):.4f}]")
        print(f"  f(Alpha) range: [{np.min(sing_spec['f_alpha']):.4f}, {np.max(sing_spec['f_alpha']):.4f}]")
        
        # Market efficiency
        efficiency = mfdfa.get_efficiency_measure()
        print(f"\nMarket Efficiency Measure: {efficiency:.4f}")
        print(f"Interpretation: {'High' if efficiency > 0.7 else 'Medium' if efficiency > 0.3 else 'Low'} efficiency")
    
    # Test signal calculation
    print("\nTesting signal calculation...")
    signals = mfdfa.calculate(data)
    
    print(f"Generated {len(signals)} signals")
    print(f"Signal range: [{signals.min():.6f}, {signals.max():.6f}]")
    print(f"Non-zero signals: {np.count_nonzero(signals)}")
    print(f"Average signal: {np.mean(signals[signals > 0]):.6f}")
    
    # Show last few signals
    print(f"Last 10 signals: {signals[-10:]}")
    
    print("\nMulti-Fractal DFA test completed successfully!")


# Create alias for consistent naming
MultifractalDFA = MultiFractalDFA