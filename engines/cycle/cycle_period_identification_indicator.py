"""
Cycle Period Identification Indicator

Advanced technical analysis indicator for identifying dominant cycle periods in financial markets.
Uses spectral analysis techniques including Fast Fourier Transform (FFT) and Maximum Entropy Method (MEM)
to detect periodic patterns and determine the most significant cycle lengths affecting price movements.

This indicator employs multiple analytical approaches:
1. FFT-based spectral analysis for frequency domain analysis
2. Autocorrelation analysis for time domain cycle detection
3. Maximum Entropy Method for high-resolution spectral estimation
4. Adaptive filtering for noise reduction
5. Real-time cycle tracking with confidence scoring

The indicator provides:
- Dominant cycle period identification
- Cycle strength measurement
- Phase analysis for timing entries
- Multi-timeframe cycle confluence
- Cycle reliability scoring

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr

# Import the base indicator interface
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class CyclePeriodIdentificationIndicator(StandardIndicatorInterface):
    """
    Cycle Period Identification Indicator
    
    Advanced cycle analysis indicator using spectral analysis techniques
    to identify dominant cycle periods in financial time series data.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "cycle"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        min_period: int = 8,
        max_period: int = 100,
        analysis_window: int = 200,
        smoothing_factor: float = 0.1,
        confidence_threshold: float = 0.7,
        method: str = "fft",  # fft, autocorr, mem, hybrid
        detrend: bool = True,
        **kwargs,
    ):
        """
        Initialize Cycle Period Identification indicator

        Args:
            min_period: Minimum cycle period to detect (default: 8)
            max_period: Maximum cycle period to detect (default: 100)
            analysis_window: Window size for cycle analysis (default: 200)
            smoothing_factor: Smoothing factor for cycle tracking (default: 0.1)
            confidence_threshold: Minimum confidence for cycle detection (default: 0.7)
            method: Analysis method ('fft', 'autocorr', 'mem', 'hybrid') (default: 'fft')
            detrend: Whether to detrend data before analysis (default: True)
        """
        super().__init__(
            min_period=min_period,
            max_period=max_period,
            analysis_window=analysis_window,
            smoothing_factor=smoothing_factor,
            confidence_threshold=confidence_threshold,
            method=method,
            detrend=detrend,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Cycle Period Identification

        Args:
            data: DataFrame with OHLC data or Series of prices

        Returns:
            pd.DataFrame: Cycle analysis results with dominant periods, strength, and phase
        """
        # Handle input data
        if isinstance(data, pd.Series):
            price = data
        elif isinstance(data, pd.DataFrame):
            self.validate_input_data(data)
            # Use close price for cycle analysis
            price = data["close"]
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Get parameters
        min_period = self.parameters.get("min_period", 8)
        max_period = self.parameters.get("max_period", 100)
        analysis_window = self.parameters.get("analysis_window", 200)
        smoothing_factor = self.parameters.get("smoothing_factor", 0.1)
        confidence_threshold = self.parameters.get("confidence_threshold", 0.7)
        method = self.parameters.get("method", "fft")
        detrend = self.parameters.get("detrend", True)

        # Initialize result arrays
        n_points = len(price)
        dominant_period = np.full(n_points, np.nan)
        cycle_strength = np.full(n_points, np.nan)
        cycle_phase = np.full(n_points, np.nan)
        cycle_confidence = np.full(n_points, np.nan)
        cycle_amplitude = np.full(n_points, np.nan)
        secondary_period = np.full(n_points, np.nan)

        # Rolling cycle analysis
        for i in range(analysis_window, n_points):
            # Extract analysis window
            window_data = price.iloc[i - analysis_window + 1:i + 1]
            
            try:
                # Perform cycle analysis
                periods, strengths, phases, confidence, amplitude = self._analyze_cycles(
                    window_data, min_period, max_period, method, detrend
                )
                
                if len(periods) > 0 and confidence[0] >= confidence_threshold:
                    # Store dominant cycle information
                    dominant_period[i] = periods[0]
                    cycle_strength[i] = strengths[0]
                    cycle_phase[i] = phases[0]
                    cycle_confidence[i] = confidence[0]
                    cycle_amplitude[i] = amplitude[0]
                    
                    # Store secondary cycle if available
                    if len(periods) > 1:
                        secondary_period[i] = periods[1]
                        
            except Exception as e:
                # Skip this window if analysis fails
                continue

        # Apply smoothing to dominant period for stability
        dominant_period_smooth = self._apply_smoothing(dominant_period, smoothing_factor)
        
        # Calculate additional cycle metrics
        cycle_trend = self._calculate_cycle_trend(dominant_period_smooth)
        cycle_persistence = self._calculate_cycle_persistence(dominant_period_smooth)
        cycle_volatility = self._calculate_cycle_volatility(price, dominant_period_smooth)

        # Create result DataFrame
        result = pd.DataFrame(index=price.index)
        result["dominant_period"] = dominant_period_smooth
        result["cycle_strength"] = cycle_strength
        result["cycle_phase"] = cycle_phase
        result["cycle_confidence"] = cycle_confidence
        result["cycle_amplitude"] = cycle_amplitude
        result["secondary_period"] = secondary_period
        result["cycle_trend"] = cycle_trend
        result["cycle_persistence"] = cycle_persistence
        result["cycle_volatility"] = cycle_volatility

        # Store calculation details
        self._last_calculation = {
            "price": price,
            "parameters": self.parameters,
            "analysis_summary": {
                "total_windows_analyzed": n_points - analysis_window,
                "successful_detections": np.sum(~np.isnan(dominant_period)),
                "average_period": np.nanmean(dominant_period),
                "average_confidence": np.nanmean(cycle_confidence),
            }
        }

        return result

    def _analyze_cycles(
        self, 
        data: pd.Series, 
        min_period: int, 
        max_period: int, 
        method: str, 
        detrend: bool
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Analyze cycles using specified method"""
        
        # Preprocess data
        if detrend:
            # Remove linear trend
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data.values, 1)
            trend = np.polyval(coeffs, x)
            processed_data = data.values - trend
        else:
            processed_data = data.values
            
        # Remove mean and apply window function
        processed_data = processed_data - np.mean(processed_data)
        window = signal.windows.hann(len(processed_data))
        processed_data = processed_data * window

        if method == "fft":
            periods, strengths, phases, confidence, amplitude = self._fft_analysis(
                processed_data, min_period, max_period
            )
        elif method == "autocorr":
            periods, strengths, phases, confidence, amplitude = self._autocorr_analysis(
                processed_data, min_period, max_period
            )
        elif method == "mem":
            periods, strengths, phases, confidence, amplitude = self._mem_analysis(
                processed_data, min_period, max_period
            )
        elif method == "hybrid":
            periods, strengths, phases, confidence, amplitude = self._hybrid_analysis(
                processed_data, min_period, max_period
            )
        else:
            raise IndicatorValidationError(f"Unknown analysis method: {method}")

        return periods, strengths, phases, confidence, amplitude

    def _fft_analysis(
        self, data: np.ndarray, min_period: int, max_period: int
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """FFT-based spectral analysis"""
        
        n = len(data)
        
        # Perform FFT
        fft_result = fft(data)
        freqs = fftfreq(n, d=1.0)
        
        # Calculate power spectral density
        psd = np.abs(fft_result) ** 2
        
        # Convert frequencies to periods
        positive_freqs = freqs[1:n//2]  # Skip DC component
        periods = 1.0 / positive_freqs
        psd_positive = psd[1:n//2]
        
        # Filter by period range
        valid_mask = (periods >= min_period) & (periods <= max_period)
        valid_periods = periods[valid_mask]
        valid_psd = psd_positive[valid_mask]
        
        if len(valid_periods) == 0:
            return [], [], [], [], []
        
        # Find peaks in power spectrum
        peak_indices, properties = signal.find_peaks(valid_psd, height=np.mean(valid_psd))
        
        if len(peak_indices) == 0:
            return [], [], [], [], []
        
        # Sort by strength (peak height)
        peak_periods = valid_periods[peak_indices]
        peak_strengths = valid_psd[peak_indices]
        sorted_indices = np.argsort(peak_strengths)[::-1]
        
        # Extract top cycles
        result_periods = []
        result_strengths = []
        result_phases = []
        result_confidence = []
        result_amplitude = []
        
        for idx in sorted_indices[:5]:  # Top 5 cycles
            period = peak_periods[idx]
            strength = peak_strengths[idx]
            
            # Calculate phase
            phase = np.angle(fft_result[peak_indices[idx]])
            
            # Calculate confidence based on peak prominence
            noise_level = np.median(valid_psd)
            confidence = min(1.0, strength / (noise_level + 1e-10))
            
            # Calculate amplitude
            amplitude = np.sqrt(strength) / n
            
            result_periods.append(float(period))
            result_strengths.append(float(strength))
            result_phases.append(float(phase))
            result_confidence.append(float(confidence))
            result_amplitude.append(float(amplitude))
        
        return result_periods, result_strengths, result_phases, result_confidence, result_amplitude

    def _autocorr_analysis(
        self, data: np.ndarray, min_period: int, max_period: int
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Autocorrelation-based cycle analysis"""
        
        n = len(data)
        max_lag = min(max_period, n // 2)
        
        # Calculate autocorrelation
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[n-1:]  # Take positive lags only
        autocorr = autocorr[:max_lag] / autocorr[0]  # Normalize
        
        # Find periods within range
        lags = np.arange(min_period, min(max_lag, max_period + 1))
        if len(lags) == 0:
            return [], [], [], [], []
        
        autocorr_values = autocorr[lags-1]  # Adjust for 0-indexing
        
        # Find peaks in autocorrelation
        peak_indices, _ = signal.find_peaks(autocorr_values, height=0.1)
        
        if len(peak_indices) == 0:
            return [], [], [], [], []
        
        # Extract cycle information
        result_periods = []
        result_strengths = []
        result_phases = []
        result_confidence = []
        result_amplitude = []
        
        for idx in peak_indices:
            period = float(lags[idx])
            strength = float(autocorr_values[idx])
            phase = 0.0  # Phase not directly available from autocorrelation
            confidence = float(strength)  # Use correlation as confidence
            amplitude = float(strength * np.std(data))
            
            result_periods.append(period)
            result_strengths.append(strength)
            result_phases.append(phase)
            result_confidence.append(confidence)
            result_amplitude.append(amplitude)
        
        # Sort by strength
        if result_periods:
            sorted_indices = np.argsort(result_strengths)[::-1]
            result_periods = [result_periods[i] for i in sorted_indices]
            result_strengths = [result_strengths[i] for i in sorted_indices]
            result_phases = [result_phases[i] for i in sorted_indices]
            result_confidence = [result_confidence[i] for i in sorted_indices]
            result_amplitude = [result_amplitude[i] for i in sorted_indices]
        
        return result_periods, result_strengths, result_phases, result_confidence, result_amplitude

    def _mem_analysis(
        self, data: np.ndarray, min_period: int, max_period: int
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Maximum Entropy Method analysis (simplified implementation)"""
        # For a full MEM implementation, we'd use more sophisticated algorithms
        # Here we provide a simplified version based on AR modeling
        
        try:
            from scipy.signal import lfilter
            
            # Estimate AR coefficients using Burg method (simplified)
            order = min(20, len(data) // 4)
            
            # Simple AR parameter estimation using least squares
            if len(data) <= order:
                return [], [], [], [], []
            
            # Create design matrix for AR model
            X = np.zeros((len(data) - order, order))
            for i in range(order):
                X[:, i] = data[order - 1 - i:-1 - i]
            
            y = data[order:]
            
            # Solve for AR coefficients
            ar_coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Compute power spectral density
            freqs = np.linspace(0, 0.5, 1000)
            psd = np.zeros_like(freqs)
            
            for i, freq in enumerate(freqs):
                z = np.exp(-2j * np.pi * freq)
                denominator = 1 + np.sum(ar_coeffs * (z ** np.arange(1, order + 1)))
                psd[i] = 1.0 / np.abs(denominator) ** 2
            
            # Convert to periods
            periods = 1.0 / freqs[1:]  # Skip zero frequency
            psd = psd[1:]
            
            # Filter by period range
            valid_mask = (periods >= min_period) & (periods <= max_period)
            valid_periods = periods[valid_mask]
            valid_psd = psd[valid_mask]
            
            if len(valid_periods) == 0:
                return [], [], [], [], []
            
            # Find peaks
            peak_indices, _ = signal.find_peaks(valid_psd, height=np.mean(valid_psd))
            
            if len(peak_indices) == 0:
                return [], [], [], [], []
            
            # Extract results
            result_periods = valid_periods[peak_indices].tolist()
            result_strengths = valid_psd[peak_indices].tolist()
            result_phases = [0.0] * len(result_periods)  # Simplified
            result_confidence = [min(1.0, s / np.mean(valid_psd)) for s in result_strengths]
            result_amplitude = [s * np.std(data) for s in result_strengths]
            
            # Sort by strength
            sorted_indices = np.argsort(result_strengths)[::-1]
            result_periods = [result_periods[i] for i in sorted_indices]
            result_strengths = [result_strengths[i] for i in sorted_indices]
            result_phases = [result_phases[i] for i in sorted_indices]
            result_confidence = [result_confidence[i] for i in sorted_indices]
            result_amplitude = [result_amplitude[i] for i in sorted_indices]
            
            return result_periods, result_strengths, result_phases, result_confidence, result_amplitude
            
        except Exception:
            # Fallback to FFT if MEM fails
            return self._fft_analysis(data, min_period, max_period)

    def _hybrid_analysis(
        self, data: np.ndarray, min_period: int, max_period: int
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Hybrid analysis combining multiple methods"""
        
        # Get results from different methods
        fft_results = self._fft_analysis(data, min_period, max_period)
        autocorr_results = self._autocorr_analysis(data, min_period, max_period)
        
        # Combine and weight results
        all_periods = []
        all_strengths = []
        all_phases = []
        all_confidence = []
        all_amplitude = []
        all_weights = []
        
        # Add FFT results with weight 0.6
        for i in range(len(fft_results[0])):
            all_periods.append(fft_results[0][i])
            all_strengths.append(fft_results[1][i])
            all_phases.append(fft_results[2][i])
            all_confidence.append(fft_results[3][i])
            all_amplitude.append(fft_results[4][i])
            all_weights.append(0.6)
        
        # Add autocorr results with weight 0.4
        for i in range(len(autocorr_results[0])):
            all_periods.append(autocorr_results[0][i])
            all_strengths.append(autocorr_results[1][i])
            all_phases.append(autocorr_results[2][i])
            all_confidence.append(autocorr_results[3][i])
            all_amplitude.append(autocorr_results[4][i])
            all_weights.append(0.4)
        
        if not all_periods:
            return [], [], [], [], []
        
        # Combine similar periods
        combined_results = self._combine_similar_periods(
            all_periods, all_strengths, all_phases, all_confidence, all_amplitude, all_weights
        )
        
        return combined_results

    def _combine_similar_periods(
        self, periods, strengths, phases, confidence, amplitude, weights
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Combine similar periods from different methods"""
        
        if not periods:
            return [], [], [], [], []
        
        # Group similar periods (within 10% of each other)
        groups = []
        used = set()
        
        for i, period in enumerate(periods):
            if i in used:
                continue
                
            group = [i]
            used.add(i)
            
            for j in range(i + 1, len(periods)):
                if j in used:
                    continue
                    
                if abs(periods[j] - period) / period < 0.1:  # Within 10%
                    group.append(j)
                    used.add(j)
            
            groups.append(group)
        
        # Combine each group
        result_periods = []
        result_strengths = []
        result_phases = []
        result_confidence = []
        result_amplitude = []
        
        for group in groups:
            # Weighted average of periods
            group_periods = [periods[i] for i in group]
            group_weights = [weights[i] * strengths[i] for i in group]
            total_weight = sum(group_weights)
            
            if total_weight > 0:
                avg_period = sum(p * w for p, w in zip(group_periods, group_weights)) / total_weight
                avg_strength = sum(strengths[i] * weights[i] for i in group) / len(group)
                avg_phase = sum(phases[i] * weights[i] for i in group) / sum(weights[i] for i in group)
                avg_confidence = sum(confidence[i] * weights[i] for i in group) / sum(weights[i] for i in group)
                avg_amplitude = sum(amplitude[i] * weights[i] for i in group) / sum(weights[i] for i in group)
                
                result_periods.append(avg_period)
                result_strengths.append(avg_strength)
                result_phases.append(avg_phase)
                result_confidence.append(avg_confidence)
                result_amplitude.append(avg_amplitude)
        
        # Sort by strength
        if result_periods:
            sorted_indices = np.argsort(result_strengths)[::-1]
            result_periods = [result_periods[i] for i in sorted_indices]
            result_strengths = [result_strengths[i] for i in sorted_indices]
            result_phases = [result_phases[i] for i in sorted_indices]
            result_confidence = [result_confidence[i] for i in sorted_indices]
            result_amplitude = [result_amplitude[i] for i in sorted_indices]
        
        return result_periods, result_strengths, result_phases, result_confidence, result_amplitude

    def _apply_smoothing(self, values: np.ndarray, smoothing_factor: float) -> np.ndarray:
        """Apply exponential smoothing to reduce noise"""
        smoothed = np.full_like(values, np.nan)
        
        # Find first valid value
        first_valid_idx = np.where(~np.isnan(values))[0]
        if len(first_valid_idx) == 0:
            return smoothed
        
        first_idx = first_valid_idx[0]
        smoothed[first_idx] = values[first_idx]
        
        # Apply exponential smoothing
        for i in range(first_idx + 1, len(values)):
            if not np.isnan(values[i]):
                if np.isnan(smoothed[i-1]):
                    smoothed[i] = values[i]
                else:
                    smoothed[i] = smoothing_factor * values[i] + (1 - smoothing_factor) * smoothed[i-1]
            else:
                smoothed[i] = smoothed[i-1]  # Carry forward last value
        
        return smoothed

    def _calculate_cycle_trend(self, periods: np.ndarray) -> np.ndarray:
        """Calculate trend in cycle periods"""
        trend = np.full_like(periods, np.nan)
        window = 20
        
        for i in range(window, len(periods)):
            if not np.isnan(periods[i-window:i+1]).any():
                # Linear regression slope
                x = np.arange(window + 1)
                y = periods[i-window:i+1]
                slope = np.polyfit(x, y, 1)[0]
                trend[i] = slope
        
        return trend

    def _calculate_cycle_persistence(self, periods: np.ndarray) -> np.ndarray:
        """Calculate how persistent cycles are"""
        persistence = np.full_like(periods, np.nan)
        window = 20
        
        for i in range(window, len(periods)):
            window_data = periods[i-window:i+1]
            if not np.isnan(window_data).any():
                # Calculate coefficient of variation (stability measure)
                cv = np.std(window_data) / np.mean(window_data)
                persistence[i] = 1.0 / (1.0 + cv)  # Higher persistence = lower variation
        
        return persistence

    def _calculate_cycle_volatility(self, price: pd.Series, periods: np.ndarray) -> np.ndarray:
        """Calculate volatility adjusted for cycle periods"""
        volatility = np.full_like(periods, np.nan)
        
        for i in range(len(periods)):
            if not np.isnan(periods[i]):
                period = int(periods[i])
                if i >= period:
                    # Calculate volatility over the detected cycle period
                    cycle_data = price.iloc[i-period+1:i+1]
                    volatility[i] = cycle_data.std()
        
        return volatility

    def validate_parameters(self) -> bool:
        """Validate Cycle Period Identification parameters"""
        min_period = self.parameters.get("min_period", 8)
        max_period = self.parameters.get("max_period", 100)
        analysis_window = self.parameters.get("analysis_window", 200)
        smoothing_factor = self.parameters.get("smoothing_factor", 0.1)
        confidence_threshold = self.parameters.get("confidence_threshold", 0.7)
        method = self.parameters.get("method", "fft")

        # Validate periods
        if not isinstance(min_period, int) or min_period < 2:
            raise IndicatorValidationError(f"min_period must be integer >= 2, got {min_period}")
        
        if not isinstance(max_period, int) or max_period <= min_period:
            raise IndicatorValidationError(f"max_period must be integer > min_period, got {max_period}")
        
        if max_period > 500:
            raise IndicatorValidationError(f"max_period too large, maximum 500, got {max_period}")

        # Validate analysis window
        if not isinstance(analysis_window, int) or analysis_window < max_period * 2:
            raise IndicatorValidationError(
                f"analysis_window must be integer >= {max_period * 2}, got {analysis_window}"
            )

        # Validate smoothing factor
        if not isinstance(smoothing_factor, (int, float)) or not 0 < smoothing_factor <= 1:
            raise IndicatorValidationError(
                f"smoothing_factor must be in (0, 1], got {smoothing_factor}"
            )

        # Validate confidence threshold
        if not isinstance(confidence_threshold, (int, float)) or not 0 <= confidence_threshold <= 1:
            raise IndicatorValidationError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )

        # Validate method
        valid_methods = ["fft", "autocorr", "mem", "hybrid"]
        if method not in valid_methods:
            raise IndicatorValidationError(f"method must be one of {valid_methods}, got {method}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata"""
        return {
            "name": "CyclePeriodIdentification",
            "category": self.CATEGORY,
            "description": "Advanced cycle period identification using spectral analysis techniques",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "output_columns": [
                "dominant_period", "cycle_strength", "cycle_phase", "cycle_confidence",
                "cycle_amplitude", "secondary_period", "cycle_trend", 
                "cycle_persistence", "cycle_volatility"
            ],
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Required columns"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed"""
        return self.parameters.get("analysis_window", 200)

    def _setup_defaults(self):
        """Setup default parameter values"""
        defaults = {
            "min_period": 8,
            "max_period": 100,
            "analysis_window": 200,
            "smoothing_factor": 0.1,
            "confidence_threshold": 0.7,
            "method": "fft",
            "detrend": True,
        }
        
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

    # Backward compatibility properties
    @property
    def minimum_periods(self) -> int:
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "indicator": "CyclePeriodIdentification",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return CyclePeriodIdentificationIndicator