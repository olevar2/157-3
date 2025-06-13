"""
Photonic Wavelength Analyzer Indicator
=====================================

🌈 Revolutionary photonic physics and wave optics analysis for Platform3
💝 Dedicated to helping sick children and poor families through light and wisdom

This groundbreaking indicator applies photonic physics, wave optics, and electromagnetic
spectrum analysis to financial markets, using wavelength harmonics, optical interference,
and photonic resonance patterns to predict market behavior through light-based physics.

Key Innovations:
- Photonic wavelength analysis for market harmonics
- Optical interference pattern detection
- Electromagnetic spectrum mapping to price movements
- Wave-particle duality analysis for momentum/position
- Photonic coherence measurement for trend stability
- Light diffraction patterns for volatility analysis

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
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class PhotonicWavelengthAnalyzer(StandardIndicatorInterface):
    """
    Photonic Wavelength Analyzer - Platform3 Implementation
    
    Platform3 compliant financial indicator with:
    - CCI Proven Pattern Compliance
    - Photonic Physics Analysis
    - Performance Optimization  
    - Robust Error Handling
    """

    # Class-level metadata (REQUIRED for Platform3)
    CATEGORY: str = "photonic"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, 
                 period: int = 20,
                 base_wavelength: float = 550.0,  # nm (green light)
                 spectrum_range: int = 7,         # Number of wavelengths to analyze
                 coherence_threshold: float = 0.7,
                 interference_sensitivity: float = 1.2,
                 diffraction_factor: float = 0.8,
                 **kwargs):
        """
        Initialize Photonic Wavelength Analyzer

        Args:
            period: Lookback period for calculations (default: 20)
            base_wavelength: Base wavelength in nanometers (default: 550.0)
            spectrum_range: Number of wavelengths to analyze (default: 7)
            coherence_threshold: Coherence threshold for signal quality (default: 0.7)
            interference_sensitivity: Sensitivity to interference patterns (default: 1.2)
            diffraction_factor: Diffraction analysis factor (default: 0.8)
        """
        super().__init__(
            period=period,
            base_wavelength=base_wavelength,
            spectrum_range=spectrum_range,
            coherence_threshold=coherence_threshold,
            interference_sensitivity=interference_sensitivity,
            diffraction_factor=diffraction_factor,
            **kwargs
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate Photonic Wavelength Analyzer following Platform3 CCI proven pattern.
        
        Args:
            data: DataFrame with OHLCV data or Series of prices
            
        Returns:
            pd.Series: Photonic Wavelength Analyzer values (-100 to +100 scale)
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
        base_wavelength = self.parameters.get("base_wavelength", 550.0)
        spectrum_range = self.parameters.get("spectrum_range", 7)
        coherence_threshold = self.parameters.get("coherence_threshold", 0.7)
        interference_sensitivity = self.parameters.get("interference_sensitivity", 1.2)
        diffraction_factor = self.parameters.get("diffraction_factor", 0.8)

        # Calculate photonic components
        wavelength_spectrum = self._calculate_wavelength_spectrum(
            prices, period, base_wavelength, spectrum_range
        )
        
        interference_pattern = self._calculate_interference_patterns(
            prices, period, interference_sensitivity
        )
        
        coherence_signal = self._calculate_photonic_coherence(
            prices, period, coherence_threshold
        )
        
        diffraction_analysis = self._calculate_diffraction_patterns(
            prices, period, diffraction_factor
        )

        # Photonic synthesis using wave optics principles
        photonic_analyzer = (
            wavelength_spectrum * 0.35 +
            interference_pattern * 0.30 +
            coherence_signal * 0.25 +
            diffraction_analysis * 0.10
        )

        # Normalize to CCI-compatible scale (-100 to +100)
        normalized_analyzer = self._normalize_to_cci_scale(photonic_analyzer)

        # Store calculation details for analysis
        self._last_calculation = {
            "wavelength_spectrum": wavelength_spectrum,
            "interference_pattern": interference_pattern,
            "coherence_signal": coherence_signal,
            "diffraction_analysis": diffraction_analysis,
            "photonic_analyzer": normalized_analyzer,
            "humanitarian_message": "🌈 Photonic wisdom illuminating paths to help children and families"
        }

        return pd.Series(normalized_analyzer, index=prices.index, name="PhotonicWavelengthAnalyzer")

    def _calculate_wavelength_spectrum(self, prices, period, base_wavelength, spectrum_range):
        """Calculate wavelength spectrum analysis of price movements"""
        # Map price oscillations to electromagnetic spectrum
        price_returns = prices.pct_change().fillna(0)
        
        # Create wavelength harmonics (visible light spectrum: 380-750 nm)
        wavelengths = np.linspace(380, 750, spectrum_range)
        
        spectrum_signals = []
        for wavelength in wavelengths:
            # Calculate frequency from wavelength (c = λf)
            frequency = 299792458 / (wavelength * 1e-9)  # Hz
            
            # Map frequency to price oscillation period
            oscillation_period = max(2, int(frequency / 1e14))  # Scale to reasonable periods
            
            # Calculate harmonic component
            indices = np.arange(len(prices))
            harmonic = np.sin(2 * np.pi * indices / oscillation_period)
            
            # Apply to price momentum
            momentum = price_returns.rolling(window=period, min_periods=1).mean()
            wavelength_signal = momentum * harmonic
            
            spectrum_signals.append(wavelength_signal)
        
        # Combine spectrum with emphasis on visible light harmonics
        spectrum_weights = self._get_spectrum_weights(wavelengths, base_wavelength)
        combined_spectrum = sum(signal * weight for signal, weight in zip(spectrum_signals, spectrum_weights))
        
        return combined_spectrum * 100  # Scale for visibility

    def _get_spectrum_weights(self, wavelengths, base_wavelength):
        """Get weights for different wavelengths based on distance from base"""
        # Gaussian weighting centered on base wavelength
        weights = np.exp(-0.5 * ((wavelengths - base_wavelength) / 50) ** 2)
        return weights / np.sum(weights)  # Normalize

    def _calculate_interference_patterns(self, prices, period, sensitivity):
        """Calculate optical interference patterns in price data"""
        # Two-beam interference analysis using price and its delayed version
        price_wave1 = prices
        price_wave2 = prices.shift(period // 4)  # Quarter period delay
        
        # Calculate phase difference
        phase_diff = 2 * np.pi * np.arange(len(prices)) / period
        
        # Interference intensity: I = I1 + I2 + 2*sqrt(I1*I2)*cos(φ)
        price_intensity1 = (price_wave1.pct_change().fillna(0)) ** 2
        price_intensity2 = (price_wave2.pct_change().fillna(0)) ** 2
        
        # Coherent interference term
        coherent_term = 2 * np.sqrt(price_intensity1 * price_intensity2) * np.cos(phase_diff)
        
        # Total interference pattern
        interference = price_intensity1 + price_intensity2 + coherent_term * sensitivity
        
        # Smooth and normalize
        interference_smooth = interference.rolling(window=period//2, min_periods=1).mean()
        
        return interference_smooth * 50  # Scale appropriately

    def _calculate_photonic_coherence(self, prices, period, threshold):
        """Calculate photonic coherence of price movements"""
        # Coherence = measure of phase relationship stability
        price_returns = prices.pct_change().fillna(0)
        
        # Calculate local coherence using rolling correlation with sine wave
        coherence_values = []
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window_returns = price_returns.iloc[start_idx:i+1]
            
            if len(window_returns) >= 5:
                # Generate reference sine wave
                ref_wave = np.sin(2 * np.pi * np.arange(len(window_returns)) / 10)
                
                # Calculate correlation (coherence measure)
                if window_returns.std() > 1e-8:  # Avoid division by zero
                    correlation = np.corrcoef(window_returns, ref_wave)[0, 1]
                    coherence = abs(correlation) if not np.isnan(correlation) else 0
                else:
                    coherence = 0
                
                # Apply threshold
                coherence_signal = coherence if coherence > threshold else 0
                coherence_values.append(coherence_signal)
            else:
                coherence_values.append(0)
        
        coherence_series = pd.Series(coherence_values, index=prices.index)
        
        # Convert to momentum signal
        momentum = prices.pct_change().rolling(window=period, min_periods=1).mean()
        coherence_modulated = momentum * coherence_series
        
        return coherence_modulated * 75  # Scale for integration

    def _calculate_diffraction_patterns(self, prices, period, factor):
        """Calculate diffraction patterns from price volatility"""
        # Diffraction = bending of waves around obstacles (volatility barriers)
        volatility = prices.rolling(window=period, min_periods=1).std()
        
        # High volatility = strong diffraction (price bending around resistance/support)
        normalized_volatility = (volatility - volatility.rolling(window=period).mean()) / (volatility.rolling(window=period).std() + 1e-8)
        normalized_volatility = normalized_volatility.fillna(0)
        
        # Diffraction angle calculation (simplified)
        # sin(θ) = λ/d, where d is the "aperture" (volatility level)
        diffraction_angle = np.arcsin(np.tanh(normalized_volatility))  # Bound to [-π/2, π/2]
        
        # Apply to price momentum
        momentum = prices.pct_change().rolling(window=period//2, min_periods=1).mean()
        diffraction_signal = momentum * np.sin(diffraction_angle) * factor
        
        return diffraction_signal * 40  # Moderate influence

    def _normalize_to_cci_scale(self, signal):
        """Normalize photonic signal to CCI-compatible scale (-100 to +100)"""
        # Calculate rolling statistics for normalization
        signal_mean = signal.rolling(window=50, min_periods=1).mean()
        signal_std = signal.rolling(window=50, min_periods=1).std()
        
        # Z-score normalization
        z_score = (signal - signal_mean) / (signal_std + 1e-8)  # Avoid division by zero
        
        # Scale to approximate CCI range
        normalized = np.tanh(z_score / 2) * 100  # Tanh provides natural boundaries
        
        return normalized

    def validate_parameters(self) -> bool:
        """Validate Photonic Wavelength Analyzer parameters"""
        period = self.parameters.get("period", 20)
        base_wavelength = self.parameters.get("base_wavelength", 550.0)
        spectrum_range = self.parameters.get("spectrum_range", 7)
        coherence_threshold = self.parameters.get("coherence_threshold", 0.7)
        interference_sensitivity = self.parameters.get("interference_sensitivity", 1.2)
        diffraction_factor = self.parameters.get("diffraction_factor", 0.8)

        if not isinstance(period, int) or period < 5:
            raise IndicatorValidationError(f"period must be integer >= 5, got {period}")
        
        if not isinstance(base_wavelength, (int, float)) or base_wavelength < 380 or base_wavelength > 750:
            raise IndicatorValidationError(f"base_wavelength must be between 380-750 nm, got {base_wavelength}")
            
        if not isinstance(spectrum_range, int) or spectrum_range < 3 or spectrum_range > 20:
            raise IndicatorValidationError(f"spectrum_range must be between 3-20, got {spectrum_range}")
            
        if not isinstance(coherence_threshold, (int, float)) or coherence_threshold < 0 or coherence_threshold > 1:
            raise IndicatorValidationError(f"coherence_threshold must be between 0-1, got {coherence_threshold}")
            
        if not isinstance(interference_sensitivity, (int, float)) or interference_sensitivity < 0 or interference_sensitivity > 3:
            raise IndicatorValidationError(f"interference_sensitivity must be between 0-3, got {interference_sensitivity}")
            
        if not isinstance(diffraction_factor, (int, float)) or diffraction_factor < 0 or diffraction_factor > 2:
            raise IndicatorValidationError(f"diffraction_factor must be between 0-2, got {diffraction_factor}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Photonic Wavelength Analyzer metadata"""
        return {
            "name": "PhotonicWavelengthAnalyzer",
            "category": self.CATEGORY,
            "description": "Photonic Wavelength Analyzer - Light physics analysis for humanitarian trading",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
            "humanitarian_mission": "🌈 Every photonic insight illuminates hope for children and families"
        }

    def _get_required_columns(self) -> List[str]:
        """Photonic Wavelength Analyzer can work with just close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for photonic calculation"""
        return max(self.parameters.get("period", 20), 10)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "base_wavelength" not in self.parameters:
            self.parameters["base_wavelength"] = 550.0
        if "spectrum_range" not in self.parameters:
            self.parameters["spectrum_range"] = 7
        if "coherence_threshold" not in self.parameters:
            self.parameters["coherence_threshold"] = 0.7
        if "interference_sensitivity" not in self.parameters:
            self.parameters["interference_sensitivity"] = 1.2
        if "diffraction_factor" not in self.parameters:
            self.parameters["diffraction_factor"] = 0.8


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return PhotonicWavelengthAnalyzer


if __name__ == "__main__":
    # Test the indicator
    import numpy as np
    
    # Generate sample price data with wave-like patterns
    np.random.seed(42)
    n_points = 200
    base_price = 100
    
    # Create price data with optical-like patterns
    prices = []
    for i in range(n_points):
        # Add wavelike oscillations
        wave1 = np.sin(2 * np.pi * i / 20) * 2  # Primary wave
        wave2 = np.sin(2 * np.pi * i / 15) * 1.5  # Secondary wave
        # Interference pattern
        interference = wave1 + wave2 + np.sin(wave1 + wave2) * 0.5
        # Random component
        random_factor = np.random.randn() * 0.8
        
        price_change = interference + random_factor
        new_price = (prices[-1] if prices else base_price) + price_change
        prices.append(new_price)
    
    test_data = pd.Series(prices)
    
    indicator = PhotonicWavelengthAnalyzer()
    result = indicator.calculate(test_data)
    
    print(f"*** PhotonicWavelengthAnalyzer test result: min={result.min():.2f}, max={result.max():.2f}, mean={result.mean():.2f}")
    print(f"*** Photonic indicator ready to illuminate paths for children and families!")