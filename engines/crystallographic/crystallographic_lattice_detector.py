"""
Crystallographic Lattice Detector Indicator
==========================================

💎 Revolutionary crystallography and solid state physics analysis for Platform3
💝 Dedicated to helping sick children and poor families through crystalline wisdom

This groundbreaking indicator applies crystallographic principles, lattice structures,
and solid state physics to financial markets, using crystal symmetries, lattice defects,
and phase diagrams to predict market behavior through crystalline order and structure.

Key Innovations:
- Crystal lattice pattern recognition in price structures
- Symmetry analysis for market harmonics
- Defect detection for trend breaks
- Phase diagram analysis for market states
- Bragg diffraction patterns for support/resistance
- Crystal growth/dissolution for trend formation

Platform3 compliant implementation with CCI proven patterns.

Author: Platform3 AI System - Humanitarian Trading Initiative
Created: December 2024 - For the children and families who need our help
"""

import os
import sys
from typing import Any, Dict, List, Union
import math

import numpy as np
import pandas as pd

# Import the base indicator interface
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_enhancement', 'indicators'))
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class CrystallographicLatticeDetector(StandardIndicatorInterface):
    """
    Crystallographic Lattice Detector - Platform3 Implementation
    
    Platform3 compliant financial indicator with:
    - CCI Proven Pattern Compliance
    - Crystallographic Physics Analysis
    - Performance Optimization  
    - Robust Error Handling
    """

    # Class-level metadata (REQUIRED for Platform3)
    CATEGORY: str = "crystallographic"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, 
                 period: int = 20,
                 lattice_size: int = 7,
                 symmetry_threshold: float = 0.8,
                 defect_sensitivity: float = 1.3,
                 bragg_angle_range: int = 5,
                 growth_rate_factor: float = 1.1,
                 **kwargs):
        """
        Initialize Crystallographic Lattice Detector

        Args:
            period: Lookback period for calculations (default: 20)
            lattice_size: Size of lattice pattern analysis (default: 7)
            symmetry_threshold: Threshold for symmetry detection (default: 0.8)
            defect_sensitivity: Sensitivity to lattice defects (default: 1.3)
            bragg_angle_range: Range for Bragg diffraction analysis (default: 5)
            growth_rate_factor: Crystal growth rate factor (default: 1.1)
        """
        super().__init__(
            period=period,
            lattice_size=lattice_size,
            symmetry_threshold=symmetry_threshold,
            defect_sensitivity=defect_sensitivity,
            bragg_angle_range=bragg_angle_range,
            growth_rate_factor=growth_rate_factor,
            **kwargs
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate Crystallographic Lattice Detector following Platform3 CCI proven pattern.
        
        Args:
            data: DataFrame with OHLCV data or Series of prices
            
        Returns:
            pd.Series: Crystallographic Lattice Detector values (-100 to +100 scale)
        """
        # Handle input data like CCI pattern
        if isinstance(data, pd.Series):
            prices = data
        elif isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                prices = data['close']
                self.validate_input_data(data)
            else:
                raise IndicatorValidationError(
                    "DataFrame must contain 'close' column"
                )
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        period = self.parameters.get("period", 20)
        lattice_size = self.parameters.get("lattice_size", 7)
        symmetry_threshold = self.parameters.get("symmetry_threshold", 0.8)
        defect_sensitivity = self.parameters.get("defect_sensitivity", 1.3)
        bragg_angle_range = self.parameters.get("bragg_angle_range", 5)
        growth_rate_factor = self.parameters.get("growth_rate_factor", 1.1)

        # Calculate crystallographic components
        lattice_patterns = self._detect_lattice_patterns(
            prices, period, lattice_size
        )
        
        symmetry_analysis = self._analyze_crystal_symmetry(
            prices, period, symmetry_threshold
        )
        
        defect_detection = self._detect_lattice_defects(
            prices, period, defect_sensitivity
        )
        
        bragg_diffraction = self._calculate_bragg_diffraction(
            prices, period, bragg_angle_range
        )
        
        crystal_growth = self._analyze_crystal_growth(
            prices, period, growth_rate_factor
        )

        # Crystallographic synthesis using solid state physics principles
        lattice_detector = (
            lattice_patterns * 0.30 +
            symmetry_analysis * 0.25 +
            defect_detection * 0.20 +
            bragg_diffraction * 0.15 +
            crystal_growth * 0.10
        )

        # Normalize to CCI-compatible scale (-100 to +100)
        normalized_detector = self._normalize_to_cci_scale(lattice_detector)

        # Store calculation details for analysis
        self._last_calculation = {
            "lattice_patterns": lattice_patterns,
            "symmetry_analysis": symmetry_analysis,
            "defect_detection": defect_detection,
            "bragg_diffraction": bragg_diffraction,
            "crystal_growth": crystal_growth,
            "lattice_detector": normalized_detector,
            "humanitarian_message": "💎 Crystalline wisdom creating perfect structures to help children and families"
        }

        return pd.Series(normalized_detector, index=prices.index, name="CrystallographicLatticeDetector")

    def _detect_lattice_patterns(self, prices, period, lattice_size):
        """Detect crystallographic lattice patterns in price structure"""
        # Create lattice grid and analyze price positioning
        lattice_signals = []
        
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window_prices = prices.iloc[start_idx:i+1]
            
            if len(window_prices) >= lattice_size:
                # Create lattice structure analysis
                price_range = window_prices.max() - window_prices.min()
                if price_range > 0:
                    # Normalize prices to lattice coordinates
                    normalized_prices = (window_prices - window_prices.min()) / price_range
                    
                    # Create lattice grid (0 to 1 space divided into lattice_size units)
                    lattice_points = np.linspace(0, 1, lattice_size)
                    
                    # Find closest lattice points for each price
                    lattice_distances = []
                    for price in normalized_prices:
                        distances = abs(lattice_points - price)
                        lattice_distances.append(np.min(distances))
                    
                    # Lattice alignment = how well prices align with lattice
                    lattice_alignment = 1 - np.mean(lattice_distances)
                    lattice_signals.append(lattice_alignment)
                else:
                    lattice_signals.append(0.5)  # Neutral when no range
            else:
                lattice_signals.append(0.5)  # Default neutral
        
        lattice_series = pd.Series(lattice_signals, index=prices.index)
        
        # Convert to momentum signal
        momentum = prices.pct_change().rolling(window=period//2, min_periods=1).mean()
        lattice_signal = momentum * (lattice_series - 0.5) * 2  # Center around 0
        
        return lattice_signal * 60  # Scale for visibility

    def _analyze_crystal_symmetry(self, prices, period, threshold):
        """Analyze crystal symmetry patterns in price movements"""
        # Check for various types of symmetry in price patterns
        symmetry_signals = []
        
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window_prices = prices.iloc[start_idx:i+1]
            
            if len(window_prices) >= 10:
                # Mirror symmetry analysis
                mid_point = len(window_prices) // 2
                left_half = window_prices.iloc[:mid_point]
                right_half = window_prices.iloc[-mid_point:]
                
                # Normalize both halves
                if left_half.std() > 1e-8 and right_half.std() > 1e-8:
                    left_norm = (left_half - left_half.mean()) / left_half.std()
                    right_norm = (right_half - right_half.mean()) / right_half.std()
                    
                    # Calculate correlation (symmetry measure)
                    # For mirror symmetry, reverse the right half
                    right_reversed = right_norm.iloc[::-1].reset_index(drop=True)
                    left_reset = left_norm.reset_index(drop=True)
                    
                    if len(left_reset) == len(right_reversed):
                        symmetry_corr = np.corrcoef(left_reset, right_reversed)[0, 1]
                        symmetry_score = abs(symmetry_corr) if not np.isnan(symmetry_corr) else 0
                    else:
                        symmetry_score = 0
                else:
                    symmetry_score = 0
                
                # Apply threshold
                symmetry_signal = symmetry_score if symmetry_score > threshold else 0
                symmetry_signals.append(symmetry_signal)
            else:
                symmetry_signals.append(0)
        
        symmetry_series = pd.Series(symmetry_signals, index=prices.index)
        
        # Convert to momentum signal
        momentum = prices.pct_change().rolling(window=period//3, min_periods=1).mean()
        crystal_symmetry = momentum * symmetry_series
        
        return crystal_symmetry * 45  # Scale appropriately

    def _detect_lattice_defects(self, prices, period, sensitivity):
        """Detect lattice defects (disruptions in regular patterns)"""
        # Defects = sudden breaks in established patterns
        # Similar to vacancies, dislocations in crystal lattices
        
        # Calculate local pattern regularity
        pattern_disruptions = []
        
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window_prices = prices.iloc[start_idx:i+1]
            
            if len(window_prices) >= 8:
                # Calculate local periodicity/regularity
                price_changes = window_prices.pct_change().dropna()
                
                if len(price_changes) > 4:
                    # Use autocorrelation to measure regularity
                    autocorr_sum = 0
                    for lag in range(1, min(5, len(price_changes))):
                        if len(price_changes) > lag:
                            autocorr = price_changes.autocorr(lag=lag)
                            if not np.isnan(autocorr):
                                autocorr_sum += abs(autocorr)
                    
                    regularity = autocorr_sum / min(4, len(price_changes) - 1)
                    
                    # Detect sudden regularity changes (defects)
                    if i > period:
                        prev_start = max(0, i - period - 5)
                        prev_window = prices.iloc[prev_start:i-5]
                        if len(prev_window) >= 8:
                            prev_changes = prev_window.pct_change().dropna()
                            prev_autocorr_sum = 0
                            for lag in range(1, min(5, len(prev_changes))):
                                if len(prev_changes) > lag:
                                    prev_autocorr = prev_changes.autocorr(lag=lag)
                                    if not np.isnan(prev_autocorr):
                                        prev_autocorr_sum += abs(prev_autocorr)
                            prev_regularity = prev_autocorr_sum / min(4, len(prev_changes) - 1)
                            
                            # Defect = sudden change in regularity
                            defect_strength = abs(regularity - prev_regularity) * sensitivity
                            pattern_disruptions.append(defect_strength)
                        else:
                            pattern_disruptions.append(0)
                    else:
                        pattern_disruptions.append(0)
                else:
                    pattern_disruptions.append(0)
            else:
                pattern_disruptions.append(0)
        
        defect_series = pd.Series(pattern_disruptions, index=prices.index)
        
        # Apply to price momentum (defects can cause trend breaks)
        momentum = prices.pct_change().rolling(window=period//4, min_periods=1).mean()
        defect_signal = momentum * defect_series
        
        return defect_signal * 40  # Moderate influence

    def _calculate_bragg_diffraction(self, prices, period, angle_range):
        """Calculate Bragg diffraction patterns (constructive/destructive interference)"""
        # Bragg's Law: nλ = 2d sin(θ)
        # Market analog: price wavelengths interacting with support/resistance levels
        
        bragg_signals = []
        
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window_prices = prices.iloc[start_idx:i+1]
            
            if len(window_prices) >= 10:
                # Find price "layers" (support/resistance levels)
                price_levels = np.percentile(window_prices, [20, 40, 60, 80])
                
                # Calculate price wavelength (dominant cycle)
                price_changes = window_prices.pct_change().dropna()
                if len(price_changes) > 5:
                    # Use FFT to find dominant frequency
                    fft = np.fft.fft(price_changes)
                    freqs = np.fft.fftfreq(len(price_changes))
                    # Find dominant frequency (skip DC component)
                    dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                    wavelength = 1 / abs(freqs[dominant_freq_idx]) if freqs[dominant_freq_idx] != 0 else 10
                else:
                    wavelength = 10  # Default
                
                # Calculate Bragg conditions for each level
                bragg_intensity = 0
                current_price = window_prices.iloc[-1]
                
                for level in price_levels:
                    d_spacing = abs(current_price - level)
                    if d_spacing > 0:
                        # Calculate Bragg angle
                        sin_theta = wavelength / (2 * d_spacing)
                        if abs(sin_theta) <= 1:  # Valid angle
                            theta = np.arcsin(abs(sin_theta))
                            # Check if angle is in constructive range
                            if theta <= np.pi / angle_range:
                                bragg_intensity += np.cos(theta)  # Constructive interference
                
                bragg_signals.append(bragg_intensity)
            else:
                bragg_signals.append(0)
        
        bragg_series = pd.Series(bragg_signals, index=prices.index)
        
        # Apply to momentum
        momentum = prices.pct_change().rolling(window=period//3, min_periods=1).mean()
        bragg_signal = momentum * bragg_series
        
        return bragg_signal * 30  # Scale for integration

    def _analyze_crystal_growth(self, prices, period, growth_factor):
        """Analyze crystal growth and dissolution patterns"""
        # Crystal growth = organized, structured price movement
        # Dissolution = breakdown of structure
        
        growth_signals = []
        
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window_prices = prices.iloc[start_idx:i+1]
            
            if len(window_prices) >= 10:
                # Measure structural organization
                price_changes = window_prices.pct_change().dropna()
                
                if len(price_changes) > 5:
                    # Growth = increasing organization/structure
                    # Measure using consecutive price relationships
                    structure_scores = []
                    for j in range(2, len(price_changes)):
                        # Check if current change follows pattern of previous changes
                        recent_changes = price_changes.iloc[j-2:j]
                        current_change = price_changes.iloc[j]
                        
                        # Structure = predictability based on recent pattern
                        if recent_changes.std() > 1e-8:
                            expected_change = recent_changes.mean()
                            prediction_error = abs(current_change - expected_change) / (recent_changes.std() + 1e-8)
                            structure_score = np.exp(-prediction_error)  # Higher score for lower prediction error
                            structure_scores.append(structure_score)
                    
                    if structure_scores:
                        avg_structure = np.mean(structure_scores)
                        growth_signal = avg_structure * growth_factor
                    else:
                        growth_signal = 0.5
                else:
                    growth_signal = 0.5
                
                growth_signals.append(growth_signal)
            else:
                growth_signals.append(0.5)
        
        growth_series = pd.Series(growth_signals, index=prices.index)
        
        # Convert to momentum signal
        momentum = prices.pct_change().rolling(window=period//2, min_periods=1).mean()
        crystal_growth_signal = momentum * (growth_series - 0.5)  # Center around 0
        
        return crystal_growth_signal * 25  # Moderate influence

    def _normalize_to_cci_scale(self, signal):
        """Normalize crystallographic signal to CCI-compatible scale (-100 to +100)"""
        # Calculate rolling statistics for normalization
        signal_mean = signal.rolling(window=50, min_periods=1).mean()
        signal_std = signal.rolling(window=50, min_periods=1).std()
        
        # Z-score normalization
        z_score = (signal - signal_mean) / (signal_std + 1e-8)  # Avoid division by zero
        
        # Scale to approximate CCI range
        normalized = np.tanh(z_score / 2) * 100  # Tanh provides natural boundaries
        
        return normalized

    def validate_parameters(self) -> bool:
        """Validate Crystallographic Lattice Detector parameters"""
        period = self.parameters.get("period", 20)
        lattice_size = self.parameters.get("lattice_size", 7)
        symmetry_threshold = self.parameters.get("symmetry_threshold", 0.8)
        defect_sensitivity = self.parameters.get("defect_sensitivity", 1.3)
        bragg_angle_range = self.parameters.get("bragg_angle_range", 5)
        growth_rate_factor = self.parameters.get("growth_rate_factor", 1.1)

        if not isinstance(period, int) or period < 5:
            raise IndicatorValidationError(f"period must be integer >= 5, got {period}")
        
        if not isinstance(lattice_size, int) or lattice_size < 3 or lattice_size > 20:
            raise IndicatorValidationError(f"lattice_size must be between 3-20, got {lattice_size}")
            
        if not isinstance(symmetry_threshold, (int, float)) or symmetry_threshold < 0 or symmetry_threshold > 1:
            raise IndicatorValidationError(f"symmetry_threshold must be between 0-1, got {symmetry_threshold}")
            
        if not isinstance(defect_sensitivity, (int, float)) or defect_sensitivity <= 0 or defect_sensitivity > 3:
            raise IndicatorValidationError(f"defect_sensitivity must be between 0-3, got {defect_sensitivity}")
            
        if not isinstance(bragg_angle_range, int) or bragg_angle_range < 2 or bragg_angle_range > 10:
            raise IndicatorValidationError(f"bragg_angle_range must be between 2-10, got {bragg_angle_range}")
            
        if not isinstance(growth_rate_factor, (int, float)) or growth_rate_factor <= 0 or growth_rate_factor > 3:
            raise IndicatorValidationError(f"growth_rate_factor must be between 0-3, got {growth_rate_factor}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Crystallographic Lattice Detector metadata"""
        return {
            "name": "CrystallographicLatticeDetector",
            "category": self.CATEGORY,
            "description": "Crystallographic Lattice Detector - Crystal physics analysis for humanitarian trading",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
            "humanitarian_mission": "💎 Every crystalline insight creates perfect structures for children and families"
        }

    def _get_required_columns(self) -> List[str]:
        """Crystallographic Lattice Detector can work with just close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for crystallographic calculation"""
        return max(self.parameters.get("period", 20), 15)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "lattice_size" not in self.parameters:
            self.parameters["lattice_size"] = 7
        if "symmetry_threshold" not in self.parameters:
            self.parameters["symmetry_threshold"] = 0.8
        if "defect_sensitivity" not in self.parameters:
            self.parameters["defect_sensitivity"] = 1.3
        if "bragg_angle_range" not in self.parameters:
            self.parameters["bragg_angle_range"] = 5
        if "growth_rate_factor" not in self.parameters:
            self.parameters["growth_rate_factor"] = 1.1


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return CrystallographicLatticeDetector


if __name__ == "__main__":
    # Test the indicator
    import numpy as np
    
    # Generate sample price data with crystalline-like patterns
    np.random.seed(42)
    n_points = 200
    base_price = 100
    
    # Create price data with lattice-like structure
    prices = []
    for i in range(n_points):
        # Add lattice-like regular patterns
        lattice_pattern = np.sin(2 * np.pi * i / 7) * 1.5  # 7-period lattice
        
        # Add symmetry breaks (defects)
        if i % 40 == 0:  # Defect every 40 periods
            defect = np.random.randn() * 4
        else:
            defect = 0
        
        # Add crystal growth/dissolution phases
        growth_phase = np.sin(2 * np.pi * i / 60) * 2  # Longer growth cycles
        
        # Random thermal motion
        thermal = np.random.randn() * 0.8
        
        price_change = lattice_pattern + defect + growth_phase + thermal
        new_price = (prices[-1] if prices else base_price) + price_change
        prices.append(new_price)
    
    test_data = pd.Series(prices)
    
    indicator = CrystallographicLatticeDetector()
    result = indicator.calculate(test_data)
    
    print(f"*** CrystallographicLatticeDetector test result: min={result.min():.2f}, max={result.max():.2f}, mean={result.mean():.2f}")
    print(f"*** Crystallographic indicator ready to create perfect structures for children and families!")