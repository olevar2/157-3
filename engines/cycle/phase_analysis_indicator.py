"""
Phase Analysis Indicator

Advanced phase analysis indicator that tracks the instantaneous phase of market cycles
using Hilbert Transform and quadrature detection techniques. Phase analysis is crucial
for timing market entries and exits by determining where the market is within its
current cycle.

This indicator provides:
1. Instantaneous phase calculation using Hilbert Transform
2. Phase velocity and acceleration for cycle timing
3. Phase-based signal generation
4. Cycle turning point prediction
5. Multi-timeframe phase confluence
6. Phase momentum and strength indicators
7. Adaptive phase smoothing and noise reduction

Key Phase Interpretations:
- Phase 0°-90°: Cycle bottom to mid-rise (bullish acceleration)
- Phase 90°-180°: Mid-rise to cycle top (bullish deceleration)  
- Phase 180°-270°: Cycle top to mid-fall (bearish acceleration)
- Phase 270°-360°: Mid-fall to cycle bottom (bearish deceleration)

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.signal import hilbert

# Import the base indicator interface
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class PhaseAnalysisIndicator(StandardIndicatorInterface):
    """
    Phase Analysis Indicator
    
    Advanced cycle phase analysis using Hilbert Transform for
    precise market timing and cycle position determination.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "cycle"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        cycle_period: int = 20,
        smooth_period: int = 5,
        phase_threshold: float = 45.0,
        adaptive_smoothing: bool = True,
        velocity_window: int = 3,
        **kwargs,
    ):
        """
        Initialize Phase Analysis indicator

        Args:
            cycle_period: Expected cycle period for analysis (default: 20)
            smooth_period: Period for phase smoothing (default: 5)
            phase_threshold: Threshold for phase-based signals (default: 45.0 degrees)
            adaptive_smoothing: Whether to use adaptive smoothing (default: True)
            velocity_window: Window for phase velocity calculation (default: 3)
        """
        super().__init__(
            cycle_period=cycle_period,
            smooth_period=smooth_period,
            phase_threshold=phase_threshold,
            adaptive_smoothing=adaptive_smoothing,
            velocity_window=velocity_window,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Phase Analysis

        Args:
            data: DataFrame with OHLC data or Series of prices

        Returns:
            pd.DataFrame: Phase analysis results
        """
        # Handle input data
        if isinstance(data, pd.Series):
            price = data
        elif isinstance(data, pd.DataFrame):
            self.validate_input_data(data)
            # Use median price for phase analysis
            if "high" in data.columns and "low" in data.columns:
                price = (data["high"] + data["low"]) / 2
            else:
                price = data["close"]
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Get parameters
        cycle_period = self.parameters.get("cycle_period", 20)
        smooth_period = self.parameters.get("smooth_period", 5)
        phase_threshold = self.parameters.get("phase_threshold", 45.0)
        adaptive_smoothing = self.parameters.get("adaptive_smoothing", True)
        velocity_window = self.parameters.get("velocity_window", 3)

        # Prepare price data for phase analysis
        smoothed_price = self._smooth_price(price, cycle_period)
        
        # Calculate instantaneous phase using Hilbert Transform
        instantaneous_phase = self._calculate_instantaneous_phase(smoothed_price)
        
        # Apply smoothing to phase if enabled
        if adaptive_smoothing:
            phase_smooth = self._adaptive_phase_smoothing(instantaneous_phase, smooth_period)
        else:
            phase_smooth = instantaneous_phase.rolling(window=smooth_period, min_periods=1).mean()
        
        # Calculate phase derivatives
        phase_velocity = self._calculate_phase_velocity(phase_smooth, velocity_window)
        phase_acceleration = self._calculate_phase_acceleration(phase_velocity, velocity_window)
        
        # Calculate cycle position and timing
        cycle_position = self._calculate_cycle_position(phase_smooth)
        cycle_timing = self._calculate_cycle_timing(phase_smooth)
        
        # Generate phase-based signals
        phase_signals = self._generate_phase_signals(phase_smooth, phase_velocity, phase_threshold)
        
        # Calculate turning point predictions
        turning_points = self._predict_turning_points(phase_smooth, phase_velocity, phase_acceleration)
        
        # Calculate phase strength and momentum
        phase_strength = self._calculate_phase_strength(phase_velocity, phase_acceleration)
        phase_momentum = self._calculate_phase_momentum(phase_smooth, phase_velocity)
        
        # Calculate phase quality score
        phase_quality = self._calculate_phase_quality(instantaneous_phase, phase_smooth)

        # Create result DataFrame
        result = pd.DataFrame(index=price.index)
        result["instantaneous_phase"] = instantaneous_phase
        result["phase_smooth"] = phase_smooth
        result["phase_velocity"] = phase_velocity
        result["phase_acceleration"] = phase_acceleration
        result["cycle_position"] = cycle_position
        result["cycle_timing"] = cycle_timing
        result["phase_signals"] = phase_signals
        result["turning_points"] = turning_points
        result["phase_strength"] = phase_strength
        result["phase_momentum"] = phase_momentum
        result["phase_quality"] = phase_quality

        # Store calculation details
        self._last_calculation = {
            "price": price,
            "smoothed_price": smoothed_price,
            "parameters": self.parameters,
            "phase_stats": {
                "current_phase": phase_smooth.iloc[-1] if len(phase_smooth) > 0 else 0,
                "phase_velocity": phase_velocity.iloc[-1] if len(phase_velocity) > 0 else 0,
                "cycle_position": cycle_position.iloc[-1] if len(cycle_position) > 0 else "unknown",
            }
        }

        return result

    def _smooth_price(self, price: pd.Series, period: int) -> pd.Series:
        """Apply smoothing filter to price data"""
        # Use exponential moving average for smoothing
        alpha = 2.0 / (period + 1)
        smoothed = price.ewm(alpha=alpha, adjust=False).mean()
        return smoothed

    def _calculate_instantaneous_phase(self, price: pd.Series) -> pd.Series:
        """Calculate instantaneous phase using Hilbert Transform"""
        # Remove trend first
        detrended_price = price - price.rolling(window=50, min_periods=1).mean()
        
        # Apply Hilbert Transform
        analytic_signal = hilbert(detrended_price.fillna(0).values)
        
        # Calculate instantaneous phase
        instantaneous_phase = np.angle(analytic_signal) * 180 / np.pi
        
        # Ensure phase is in 0-360 range
        instantaneous_phase = (instantaneous_phase + 360) % 360
        
        return pd.Series(instantaneous_phase, index=price.index)

    def _adaptive_phase_smoothing(self, phase: pd.Series, period: int) -> pd.Series:
        """Apply adaptive smoothing to phase data"""
        smoothed = pd.Series(index=phase.index, dtype=float)
        smoothed.iloc[0] = phase.iloc[0] if len(phase) > 0 else 0
        
        for i in range(1, len(phase)):
            # Calculate phase difference, handling wraparound
            phase_diff = phase.iloc[i] - phase.iloc[i-1]
            
            # Handle phase wraparound (360° to 0°)
            if phase_diff > 180:
                phase_diff -= 360
            elif phase_diff < -180:
                phase_diff += 360
            
            # Adaptive smoothing factor based on phase velocity
            velocity = abs(phase_diff)
            if velocity > 90:  # High velocity, less smoothing
                alpha = 0.8
            elif velocity > 45:  # Medium velocity
                alpha = 0.5
            else:  # Low velocity, more smoothing
                alpha = 0.2
            
            # Apply smoothing
            smoothed_diff = alpha * phase_diff + (1 - alpha) * 0
            smoothed.iloc[i] = (smoothed.iloc[i-1] + smoothed_diff) % 360
        
        return smoothed

    def _calculate_phase_velocity(self, phase: pd.Series, window: int) -> pd.Series:
        """Calculate phase velocity (degrees per period)"""
        velocity = pd.Series(index=phase.index, dtype=float)
        
        for i in range(window, len(phase)):
            phase_window = phase.iloc[i-window:i+1]
            
            # Calculate average phase change per period
            total_change = 0
            for j in range(1, len(phase_window)):
                phase_diff = phase_window.iloc[j] - phase_window.iloc[j-1]
                
                # Handle wraparound
                if phase_diff > 180:
                    phase_diff -= 360
                elif phase_diff < -180:
                    phase_diff += 360
                
                total_change += phase_diff
            
            velocity.iloc[i] = total_change / window
        
        return velocity

    def _calculate_phase_acceleration(self, velocity: pd.Series, window: int) -> pd.Series:
        """Calculate phase acceleration"""
        acceleration = velocity.diff().rolling(window=window, min_periods=1).mean()
        return acceleration

    def _calculate_cycle_position(self, phase: pd.Series) -> pd.Series:
        """Calculate descriptive cycle position"""
        position = pd.Series("unknown", index=phase.index)
        
        # Define cycle positions based on phase
        bottom = (phase >= 315) | (phase < 45)
        rising = (phase >= 45) & (phase < 135)
        top = (phase >= 135) & (phase < 225)
        falling = (phase >= 225) & (phase < 315)
        
        position.loc[bottom] = "bottom"
        position.loc[rising] = "rising"
        position.loc[top] = "top"
        position.loc[falling] = "falling"
        
        return position

    def _calculate_cycle_timing(self, phase: pd.Series) -> pd.Series:
        """Calculate cycle timing as percentage (0-100%)"""
        # Convert phase to cycle completion percentage
        timing = (phase / 360) * 100
        return timing

    def _generate_phase_signals(
        self, phase: pd.Series, velocity: pd.Series, threshold: float
    ) -> pd.Series:
        """Generate trading signals based on phase analysis"""
        signals = pd.Series("neutral", index=phase.index)
        
        # Bottom signals (phase near 0° or 360° with positive velocity)
        bottom_buy = ((phase < threshold) | (phase > (360 - threshold))) & (velocity > 0)
        signals.loc[bottom_buy] = "buy"
        
        # Top signals (phase near 180° with negative velocity)
        top_sell = (abs(phase - 180) < threshold) & (velocity < 0)
        signals.loc[top_sell] = "sell"
        
        # Rising phase continuation
        rising_hold = (phase > 45) & (phase < 135) & (velocity > 0)
        signals.loc[rising_hold] = "hold_long"
        
        # Falling phase continuation
        falling_hold = (phase > 225) & (phase < 315) & (velocity < 0)
        signals.loc[falling_hold] = "hold_short"
        
        # Transition warnings
        top_transition = (phase > 135) & (phase < 180) & (velocity < 2)
        bottom_transition = (phase > 315) | ((phase < 45) & (velocity < 2))
        
        signals.loc[top_transition] = "prepare_sell"
        signals.loc[bottom_transition] = "prepare_buy"
        
        return signals

    def _predict_turning_points(
        self, phase: pd.Series, velocity: pd.Series, acceleration: pd.Series
    ) -> pd.Series:
        """Predict cycle turning points"""
        turning_points = pd.Series(0, index=phase.index)  # 0 = no turning point
        
        # Top turning point: phase near 180°, velocity slowing, negative acceleration
        top_conditions = (
            (abs(phase - 180) < 30) & 
            (velocity > 0) & 
            (velocity < 5) & 
            (acceleration < -0.5)
        )
        turning_points.loc[top_conditions] = -1  # -1 = bearish turning point
        
        # Bottom turning point: phase near 0°/360°, velocity slowing, positive acceleration
        bottom_conditions = (
            ((phase < 30) | (phase > 330)) & 
            (velocity < 0) & 
            (velocity > -5) & 
            (acceleration > 0.5)
        )
        turning_points.loc[bottom_conditions] = 1  # 1 = bullish turning point
        
        return turning_points

    def _calculate_phase_strength(self, velocity: pd.Series, acceleration: pd.Series) -> pd.Series:
        """Calculate phase strength based on velocity and acceleration consistency"""
        # High strength when velocity and acceleration are aligned
        velocity_strength = np.abs(velocity) / (np.abs(velocity).rolling(window=20, min_periods=1).max() + 1e-10)
        acceleration_strength = np.abs(acceleration) / (np.abs(acceleration).rolling(window=20, min_periods=1).max() + 1e-10)
        
        # Combine velocity and acceleration strength
        phase_strength = (velocity_strength * 0.7 + acceleration_strength * 0.3) * 100
        
        return phase_strength

    def _calculate_phase_momentum(self, phase: pd.Series, velocity: pd.Series) -> pd.Series:
        """Calculate phase momentum score"""
        # Momentum based on consistent velocity direction and magnitude
        velocity_consistency = velocity.rolling(window=10, min_periods=1).apply(
            lambda x: (np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x)) if len(x) > 0 else 0
        )
        
        velocity_magnitude = np.abs(velocity) / 10  # Normalize
        
        momentum = velocity_consistency * velocity_magnitude * 100
        return momentum

    def _calculate_phase_quality(self, raw_phase: pd.Series, smooth_phase: pd.Series) -> pd.Series:
        """Calculate phase quality score based on smoothness and consistency"""
        # Calculate difference between raw and smoothed phase
        phase_diff = np.abs(raw_phase - smooth_phase)
        
        # Handle wraparound
        phase_diff = np.minimum(phase_diff, 360 - phase_diff)
        
        # Quality score (lower difference = higher quality)
        quality = 100 - np.clip(phase_diff / 180 * 100, 0, 100)
        
        return pd.Series(quality, index=raw_phase.index)

    def validate_parameters(self) -> bool:
        """Validate parameters"""
        cycle_period = self.parameters.get("cycle_period", 20)
        smooth_period = self.parameters.get("smooth_period", 5)
        phase_threshold = self.parameters.get("phase_threshold", 45.0)
        velocity_window = self.parameters.get("velocity_window", 3)

        if not isinstance(cycle_period, int) or cycle_period < 5:
            raise IndicatorValidationError(f"cycle_period must be integer >= 5, got {cycle_period}")
        
        if cycle_period > 200:
            raise IndicatorValidationError(f"cycle_period too large, maximum 200, got {cycle_period}")

        if not isinstance(smooth_period, int) or smooth_period < 1:
            raise IndicatorValidationError(f"smooth_period must be integer >= 1, got {smooth_period}")

        if not isinstance(phase_threshold, (int, float)) or not 0 < phase_threshold <= 180:
            raise IndicatorValidationError(f"phase_threshold must be in (0, 180], got {phase_threshold}")

        if not isinstance(velocity_window, int) or velocity_window < 2:
            raise IndicatorValidationError(f"velocity_window must be integer >= 2, got {velocity_window}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata"""
        return {
            "name": "PhaseAnalysis",
            "category": self.CATEGORY,
            "description": "Advanced cycle phase analysis using Hilbert Transform for market timing",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "output_columns": [
                "instantaneous_phase", "phase_smooth", "phase_velocity", "phase_acceleration",
                "cycle_position", "cycle_timing", "phase_signals", "turning_points",
                "phase_strength", "phase_momentum", "phase_quality"
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
        return max(50, self.parameters.get("cycle_period", 20) * 3)

    def _setup_defaults(self):
        """Setup default parameter values"""
        defaults = {
            "cycle_period": 20,
            "smooth_period": 5,
            "phase_threshold": 45.0,
            "adaptive_smoothing": True,
            "velocity_window": 3,
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
            "indicator": "PhaseAnalysis",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return PhaseAnalysisIndicator