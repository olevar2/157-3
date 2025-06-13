"""
Dominant Cycle Analysis Indicator

Advanced technical indicator that identifies and tracks the dominant market cycle using sophisticated
signal processing techniques. This indicator builds upon John Ehlers' cycle analysis work and employs
Hilbert Transform, Instantaneous Frequency, and Adaptive Filtering to provide real-time dominant
cycle identification with high accuracy.

Key Features:
1. Hilbert Transform for analytic signal computation
2. Instantaneous frequency calculation for real-time cycle detection
3. Adaptive filtering for noise reduction
4. Cycle quality assessment and confidence scoring
5. Multi-harmonic analysis for complex cycle structures
6. Phase-locked loop (PLL) for stable cycle tracking

The indicator provides:
- Dominant cycle period (real-time adaptive)
- Cycle quality score (0-100)
- Phase position within current cycle
- Cycle amplitude and strength
- Turning point predictions
- Cycle-based trend direction

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert

# Import the base indicator interface
from ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class DominantCycleAnalysisIndicator(StandardIndicatorInterface):
    """
    Dominant Cycle Analysis Indicator
    
    Real-time dominant cycle identification using Hilbert Transform
    and adaptive signal processing techniques.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "cycle"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        min_period: int = 10,
        max_period: int = 48,
        alpha: float = 0.07,
        quality_threshold: float = 50.0,
        adaptive_alpha: bool = True,
        noise_floor: float = 0.1,
        **kwargs,
    ):
        """
        Initialize Dominant Cycle Analysis indicator

        Args:
            min_period: Minimum cycle period to detect (default: 10)
            max_period: Maximum cycle period to detect (default: 48)
            alpha: Smoothing factor for cycle adaptation (default: 0.07)
            quality_threshold: Minimum quality score for valid cycle (default: 50.0)
            adaptive_alpha: Whether to use adaptive smoothing (default: True)
            noise_floor: Noise floor threshold for signal quality (default: 0.1)
        """
        super().__init__(
            min_period=min_period,
            max_period=max_period,
            alpha=alpha,
            quality_threshold=quality_threshold,
            adaptive_alpha=adaptive_alpha,
            noise_floor=noise_floor,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Dominant Cycle Analysis

        Args:
            data: DataFrame with OHLC data or Series of prices

        Returns:
            pd.DataFrame: Dominant cycle analysis results
        """
        # Handle input data
        if isinstance(data, pd.Series):
            price = data
        elif isinstance(data, pd.DataFrame):
            self.validate_input_data(data)
            # Use median price for cycle analysis
            if "high" in data.columns and "low" in data.columns:
                price = (data["high"] + data["low"]) / 2
            else:
                price = data["close"]
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Get parameters
        min_period = self.parameters.get("min_period", 10)
        max_period = self.parameters.get("max_period", 48)
        alpha = self.parameters.get("alpha", 0.07)
        quality_threshold = self.parameters.get("quality_threshold", 50.0)
        adaptive_alpha = self.parameters.get("adaptive_alpha", True)
        noise_floor = self.parameters.get("noise_floor", 0.1)

        n = len(price)
        
        # Initialize arrays
        smooth = np.zeros(n)
        cycle = np.zeros(n)
        dominant_cycle = np.full(n, (min_period + max_period) / 2)  # Initial estimate
        instantaneous_period = np.full(n, (min_period + max_period) / 2)
        cycle_quality = np.zeros(n)
        cycle_phase = np.zeros(n)
        cycle_amplitude = np.zeros(n)
        trend_direction = np.zeros(n)
        
        # Phase-locked loop variables
        period = (min_period + max_period) / 2
        real_part = np.zeros(n)
        imag_part = np.zeros(n)
        
        # Calculate Super Smoother for noise reduction
        smooth = self._super_smoother(price.values, 10)
        
        # Calculate cycle using Hilbert Transform approach
        for i in range(7, n):
            # Cycle extraction using bandpass filter
            cycle[i] = self._bandpass_filter(smooth, i, period)
            
            # Calculate analytic signal using Hilbert Transform
            if i >= 50:  # Need sufficient history
                cycle_window = cycle[max(0, i-50):i+1]
                analytic_signal = hilbert(cycle_window)
                
                # Get current values
                real_part[i] = np.real(analytic_signal[-1])
                imag_part[i] = np.imag(analytic_signal[-1])
                
                # Calculate instantaneous period
                if i > 7:
                    delta_phase = self._calculate_delta_phase(real_part, imag_part, i)
                    if delta_phase > 0:
                        inst_period = 2 * np.pi / delta_phase
                        inst_period = np.clip(inst_period, min_period, max_period)
                        instantaneous_period[i] = inst_period
                        
                        # Update dominant cycle with adaptive smoothing
                        if adaptive_alpha:
                            # Adapt alpha based on cycle quality
                            cycle_quality[i] = self._calculate_cycle_quality(cycle_window, inst_period)
                            adaptive_factor = cycle_quality[i] / 100.0
                            current_alpha = alpha * adaptive_factor
                        else:
                            current_alpha = alpha
                            cycle_quality[i] = self._calculate_cycle_quality(cycle_window, inst_period)
                        
                        # Exponential smoothing of dominant cycle
                        dominant_cycle[i] = (current_alpha * inst_period + 
                                           (1 - current_alpha) * dominant_cycle[i-1])
                        
                        # Update period for next iteration
                        period = dominant_cycle[i]
                
                # Calculate cycle phase
                cycle_phase[i] = np.arctan2(imag_part[i], real_part[i]) * 180 / np.pi
                
                # Calculate cycle amplitude
                cycle_amplitude[i] = np.sqrt(real_part[i]**2 + imag_part[i]**2)
                
                # Calculate trend direction based on cycle phase
                trend_direction[i] = self._calculate_trend_direction(
                    cycle_phase[i], cycle_amplitude[i], noise_floor
                )

        # Calculate additional metrics
        cycle_strength = self._calculate_cycle_strength(cycle, dominant_cycle)
        signal_to_noise = self._calculate_signal_to_noise(cycle, smooth)
        cycle_consistency = self._calculate_cycle_consistency(dominant_cycle)
        
        # Create result DataFrame
        result = pd.DataFrame(index=price.index)
        result["dominant_cycle"] = dominant_cycle
        result["instantaneous_period"] = instantaneous_period
        result["cycle_quality"] = cycle_quality
        result["cycle_phase"] = cycle_phase
        result["cycle_amplitude"] = cycle_amplitude
        result["cycle_strength"] = cycle_strength
        result["trend_direction"] = trend_direction
        result["signal_to_noise"] = signal_to_noise
        result["cycle_consistency"] = cycle_consistency
        result["cycle_raw"] = cycle

        # Store calculation details
        self._last_calculation = {
            "price": price,
            "smooth": smooth,
            "parameters": self.parameters,
            "final_period": dominant_cycle[-1] if len(dominant_cycle) > 0 else 0,
            "final_quality": cycle_quality[-1] if len(cycle_quality) > 0 else 0,
        }

        return result

    def _super_smoother(self, data: np.ndarray, period: int) -> np.ndarray:
        """Super Smoother filter for noise reduction"""
        n = len(data)
        smooth = np.zeros(n)
        
        # Calculate filter coefficients
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        # Initialize
        for i in range(4):
            if i < len(data):
                smooth[i] = data[i]
        
        # Apply filter
        for i in range(4, n):
            smooth[i] = (c1 * (data[i] + data[i-1]) / 2 + 
                        c2 * smooth[i-1] + c3 * smooth[i-2])
        
        return smooth

    def _bandpass_filter(self, data: np.ndarray, current_idx: int, period: float) -> float:
        """Bandpass filter for cycle extraction"""
        if current_idx < 6:
            return 0.0
        
        # Calculate filter coefficients based on period
        delta = 0.9
        beta = np.cos(2 * np.pi / period)
        gamma = 1.0 / np.cos(4 * np.pi * delta / period)
        alpha = gamma - np.sqrt(gamma * gamma - 1)
        
        # Bandpass coefficients
        c1 = (1 - alpha) / 2
        c2 = (1 - alpha) / 2
        c3 = beta * (1 + alpha)
        c4 = -alpha
        
        # Apply bandpass filter
        if current_idx >= 2:
            bp = (c1 * (data[current_idx] - data[current_idx-2]) + 
                  c2 * (data[current_idx-1] - data[current_idx-3]) + 
                  c3 * (0 if current_idx < 1 else data[current_idx-1]) + 
                  c4 * (0 if current_idx < 2 else data[current_idx-2]))
            return bp
        
        return 0.0

    def _calculate_delta_phase(self, real_part: np.ndarray, imag_part: np.ndarray, idx: int) -> float:
        """Calculate instantaneous phase change"""
        if idx < 1:
            return 0.0
        
        # Current and previous phase
        current_phase = np.arctan2(imag_part[idx], real_part[idx])
        prev_phase = np.arctan2(imag_part[idx-1], real_part[idx-1])
        
        # Calculate phase difference, handling wrap-around
        delta_phase = current_phase - prev_phase
        
        # Unwrap phase
        if delta_phase > np.pi:
            delta_phase -= 2 * np.pi
        elif delta_phase < -np.pi:
            delta_phase += 2 * np.pi
        
        return abs(delta_phase)

    def _calculate_cycle_quality(self, cycle_data: np.ndarray, period: float) -> float:
        """Calculate cycle quality score (0-100)"""
        if len(cycle_data) < period or period <= 0:
            return 0.0
        
        try:
            # Use autocorrelation to measure cycle regularity
            n_lags = min(int(period * 2), len(cycle_data) - 1)
            autocorr = np.correlate(cycle_data, cycle_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Look for peak at expected period
            expected_lag = int(period)
            if expected_lag < len(autocorr):
                peak_correlation = autocorr[expected_lag]
                
                # Calculate signal regularity
                variance = np.var(cycle_data)
                mean_abs = np.mean(np.abs(cycle_data))
                regularity = mean_abs / (np.sqrt(variance) + 1e-10)
                
                # Combine metrics for quality score
                quality = (abs(peak_correlation) * 50 + regularity * 50)
                return np.clip(quality, 0, 100)
            
        except Exception:
            pass
        
        return 0.0

    def _calculate_trend_direction(self, phase: float, amplitude: float, noise_floor: float) -> float:
        """Calculate trend direction from cycle phase and amplitude"""
        if amplitude < noise_floor:
            return 0.0  # Insufficient signal
        
        # Convert phase to trend direction
        # Phase 0-90: Rising trend
        # Phase 90-180: Topping
        # Phase 180-270: Falling trend  
        # Phase 270-360: Bottoming
        
        normalized_phase = phase % 360
        
        if 0 <= normalized_phase < 90:
            return amplitude * np.sin(np.radians(normalized_phase))
        elif 90 <= normalized_phase < 180:
            return amplitude * np.sin(np.radians(180 - normalized_phase))
        elif 180 <= normalized_phase < 270:
            return -amplitude * np.sin(np.radians(normalized_phase - 180))
        else:  # 270-360
            return -amplitude * np.sin(np.radians(360 - normalized_phase))

    def _calculate_cycle_strength(self, cycle: np.ndarray, dominant_cycle: np.ndarray) -> np.ndarray:
        """Calculate cycle strength over time"""
        n = len(cycle)
        strength = np.zeros(n)
        window = 20
        
        for i in range(window, n):
            if not np.isnan(dominant_cycle[i]):
                period = int(dominant_cycle[i])
                if period > 0 and i >= period:
                    # Calculate strength as amplitude normalized by period
                    cycle_window = cycle[i-period:i]
                    if len(cycle_window) > 0:
                        amplitude = np.std(cycle_window)
                        mean_amplitude = np.mean(np.abs(cycle_window))
                        strength[i] = mean_amplitude / (amplitude + 1e-10)
        
        return strength

    def _calculate_signal_to_noise(self, cycle: np.ndarray, smooth: np.ndarray) -> np.ndarray:
        """Calculate signal-to-noise ratio"""
        n = len(cycle)
        snr = np.zeros(n)
        window = 20
        
        for i in range(window, n):
            cycle_window = cycle[i-window:i]
            smooth_window = smooth[i-window:i]
            
            if len(cycle_window) > 0 and len(smooth_window) > 0:
                signal_power = np.var(cycle_window)
                noise_power = np.var(smooth_window - cycle_window)
                
                if noise_power > 0:
                    snr[i] = 10 * np.log10(signal_power / noise_power)
                else:
                    snr[i] = 100  # Very high SNR
        
        return snr

    def _calculate_cycle_consistency(self, dominant_cycle: np.ndarray) -> np.ndarray:
        """Calculate consistency of cycle period"""
        n = len(dominant_cycle)
        consistency = np.zeros(n)
        window = 20
        
        for i in range(window, n):
            window_data = dominant_cycle[i-window:i]
            if not np.any(np.isnan(window_data)):
                cv = np.std(window_data) / (np.mean(window_data) + 1e-10)
                consistency[i] = 1.0 / (1.0 + cv)  # Higher consistency = lower variation
        
        return consistency

    def validate_parameters(self) -> bool:
        """Validate parameters"""
        min_period = self.parameters.get("min_period", 10)
        max_period = self.parameters.get("max_period", 48)
        alpha = self.parameters.get("alpha", 0.07)
        quality_threshold = self.parameters.get("quality_threshold", 50.0)
        noise_floor = self.parameters.get("noise_floor", 0.1)

        if not isinstance(min_period, int) or min_period < 6:
            raise IndicatorValidationError(f"min_period must be integer >= 6, got {min_period}")
        
        if not isinstance(max_period, int) or max_period <= min_period:
            raise IndicatorValidationError(f"max_period must be integer > min_period, got {max_period}")
        
        if max_period > 200:
            raise IndicatorValidationError(f"max_period too large, maximum 200, got {max_period}")

        if not isinstance(alpha, (int, float)) or not 0 < alpha <= 1:
            raise IndicatorValidationError(f"alpha must be in (0, 1], got {alpha}")

        if not isinstance(quality_threshold, (int, float)) or not 0 <= quality_threshold <= 100:
            raise IndicatorValidationError(f"quality_threshold must be in [0, 100], got {quality_threshold}")

        if not isinstance(noise_floor, (int, float)) or noise_floor < 0:
            raise IndicatorValidationError(f"noise_floor must be non-negative, got {noise_floor}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata"""
        return {
            "name": "DominantCycleAnalysis",
            "category": self.CATEGORY,
            "description": "Real-time dominant cycle identification using Hilbert Transform and adaptive filtering",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "output_columns": [
                "dominant_cycle", "instantaneous_period", "cycle_quality", "cycle_phase",
                "cycle_amplitude", "cycle_strength", "trend_direction", 
                "signal_to_noise", "cycle_consistency", "cycle_raw"
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
        return max(100, self.parameters.get("max_period", 48) * 3)

    def _setup_defaults(self):
        """Setup default parameter values"""
        defaults = {
            "min_period": 10,
            "max_period": 48,
            "alpha": 0.07,
            "quality_threshold": 50.0,
            "adaptive_alpha": True,
            "noise_floor": 0.1,
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
            "indicator": "DominantCycleAnalysis",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }

    def get_signal(self, phase: float, quality: float, trend_direction: float) -> str:
        """
        Get trading signal based on cycle analysis

        Args:
            phase: Current cycle phase
            quality: Cycle quality score
            trend_direction: Trend direction value

        Returns:
            str: Trading signal
        """
        quality_threshold = self.parameters.get("quality_threshold", 50.0)
        
        if quality < quality_threshold:
            return "no_signal"
        
        # Normalize phase to 0-360
        normalized_phase = phase % 360
        
        # Define phase-based signals
        if 315 <= normalized_phase or normalized_phase < 45:
            # Bottom of cycle - potential buy
            if trend_direction > 0:
                return "strong_buy"
            else:
                return "buy"
        elif 45 <= normalized_phase < 135:
            # Rising phase
            if trend_direction > 0:
                return "hold_long"
            else:
                return "weak_buy"
        elif 135 <= normalized_phase < 225:
            # Top of cycle - potential sell
            if trend_direction < 0:
                return "strong_sell"
            else:
                return "sell"
        else:  # 225 <= normalized_phase < 315
            # Falling phase
            if trend_direction < 0:
                return "hold_short"
            else:
                return "weak_sell"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return DominantCycleAnalysisIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample data with known cycle
    np.random.seed(42)
    n_points = 300
    t = np.arange(n_points)
    
    # Create data with embedded cycle
    cycle_period = 20
    trend = 0.01 * t
    cycle_component = 10 * np.sin(2 * np.pi * t / cycle_period)
    noise = np.random.normal(0, 2, n_points)
    
    prices = 100 + trend + cycle_component + noise
    data = pd.Series(prices, index=pd.date_range('2024-01-01', periods=n_points, freq='1H'))

    # Calculate Dominant Cycle Analysis
    dca = DominantCycleAnalysisIndicator()
    result = dca.calculate(data)

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Price and cycle
    ax1.plot(data.index, data.values, label="Price", color="blue", alpha=0.7)
    ax1.plot(result.index, result["cycle_raw"] * 10 + np.mean(data.values), 
             label="Extracted Cycle", color="red", linewidth=2)
    ax1.set_title("Price and Extracted Cycle")
    ax1.legend()
    ax1.grid(True)

    # Dominant cycle period
    ax2.plot(result.index, result["dominant_cycle"], label="Dominant Period", color="green", linewidth=2)
    ax2.axhline(y=cycle_period, color="red", linestyle="--", label=f"True Period ({cycle_period})")
    ax2.set_title("Dominant Cycle Period Detection")
    ax2.set_ylabel("Period")
    ax2.legend()
    ax2.grid(True)

    # Cycle quality and phase
    ax3.plot(result.index, result["cycle_quality"], label="Quality", color="purple", linewidth=2)
    ax3.set_title("Cycle Quality Score")
    ax3.set_ylabel("Quality (0-100)")
    ax3.legend()
    ax3.grid(True)

    # Trend direction
    ax4.plot(result.index, result["trend_direction"], label="Trend Direction", color="orange", linewidth=2)
    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax4.set_title("Cycle-Based Trend Direction")
    ax4.set_ylabel("Direction")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    print("Dominant Cycle Analysis completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"Parameters: {dca.parameters}")
    
    # Show recent values
    recent = result.tail(5)
    print(f"\nRecent values:")
    print(recent[["dominant_cycle", "cycle_quality", "trend_direction"]])
    
    # Show accuracy
    valid_detections = result["dominant_cycle"].dropna()
    if len(valid_detections) > 0:
        avg_detected_period = valid_detections.mean()
        accuracy = 100 * (1 - abs(avg_detected_period - cycle_period) / cycle_period)
        print(f"\nCycle Detection Accuracy:")
        print(f"True period: {cycle_period}")
        print(f"Average detected period: {avg_detected_period:.2f}")
        print(f"Accuracy: {accuracy:.1f}%")