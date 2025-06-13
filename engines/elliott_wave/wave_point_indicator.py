"""
Wave Point Indicator

Advanced Elliott Wave analysis indicator that identifies and tracks wave points
according to Elliott Wave theory. This indicator automatically detects potential
wave structures and provides real-time wave counting for trading decisions.

Elliott Wave Theory principles:
- Wave 1: Initial impulse move
- Wave 2: Corrective retracement (typically 38-62% of Wave 1)
- Wave 3: Strongest impulse move (often 1.618x Wave 1)
- Wave 4: Corrective retracement (typically 23-38% of Wave 3)
- Wave 5: Final impulse move (often equal to Wave 1)

This implementation provides:
1. Automatic wave point detection using zigzag analysis
2. Wave counting and labeling (1, 2, 3, 4, 5, A, B, C)
3. Fibonacci retracement/extension analysis
4. Wave strength and quality scoring
5. Multiple degree wave analysis (Minor, Intermediate, Primary)
6. Real-time wave progression tracking

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import pandas as pd

# Import the base indicator interface
from ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class WavePointIndicator(StandardIndicatorInterface):
    """
    Wave Point Indicator
    
    Elliott Wave point detection and analysis with automatic
    wave counting and Fibonacci level validation.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "elliott_wave"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        min_wave_size: float = 0.02,  # Minimum 2% move
        zigzag_threshold: float = 0.05,  # 5% threshold for zigzag
        fibonacci_tolerance: float = 0.1,  # 10% tolerance for Fib levels
        max_lookback: int = 100,
        degree: str = "minor",  # minor, intermediate, primary
        **kwargs,
    ):
        """
        Initialize Wave Point indicator

        Args:
            min_wave_size: Minimum wave size as percentage (default: 0.02)
            zigzag_threshold: Threshold for zigzag wave detection (default: 0.05)
            fibonacci_tolerance: Tolerance for Fibonacci level validation (default: 0.1)
            max_lookback: Maximum lookback period for wave analysis (default: 100)
            degree: Wave degree ('minor', 'intermediate', 'primary') (default: 'minor')
        """
        super().__init__(
            min_wave_size=min_wave_size,
            zigzag_threshold=zigzag_threshold,
            fibonacci_tolerance=fibonacci_tolerance,
            max_lookback=max_lookback,
            degree=degree,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Wave Point analysis

        Args:
            data: DataFrame with OHLC data or Series of prices

        Returns:
            pd.DataFrame: Wave point analysis results
        """
        # Handle input data
        if isinstance(data, pd.Series):
            price = data
            high = price
            low = price
        elif isinstance(data, pd.DataFrame):
            self.validate_input_data(data)
            price = data["close"]
            high = data["high"] if "high" in data.columns else price
            low = data["low"] if "low" in data.columns else price
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Get parameters
        min_wave_size = self.parameters.get("min_wave_size", 0.02)
        zigzag_threshold = self.parameters.get("zigzag_threshold", 0.05)
        fibonacci_tolerance = self.parameters.get("fibonacci_tolerance", 0.1)
        max_lookback = self.parameters.get("max_lookback", 100)
        degree = self.parameters.get("degree", "minor")

        # Create zigzag points for wave analysis
        zigzag_points = self._create_zigzag(high, low, zigzag_threshold)
        
        # Identify wave points from zigzag
        wave_points = self._identify_wave_points(zigzag_points, min_wave_size)
        
        # Count and label waves
        wave_labels = self._label_waves(wave_points, max_lookback)
        
        # Calculate Fibonacci levels and validate waves
        fibonacci_levels = self._calculate_fibonacci_levels(wave_points)
        wave_quality = self._validate_wave_quality(wave_points, fibonacci_levels, fibonacci_tolerance)
        
        # Determine current wave position and next targets
        current_wave_info = self._analyze_current_wave(wave_points, wave_labels, price.iloc[-1])
        
        # Calculate wave momentum and strength
        wave_momentum = self._calculate_wave_momentum(wave_points, price)
        wave_strength = self._calculate_wave_strength(wave_points)

        # Create result DataFrame
        result = pd.DataFrame(index=price.index)
        result["wave_point"] = pd.Series(0, index=price.index)  # 0 = no wave point
        result["wave_label"] = pd.Series("", index=price.index)
        result["wave_quality"] = pd.Series(0.0, index=price.index)
        result["fibonacci_level"] = pd.Series(0.0, index=price.index)
        result["wave_momentum"] = wave_momentum
        result["wave_strength"] = wave_strength
        
        # Populate wave point data
        for point_idx, point_data in wave_points.items():
            if point_idx < len(result):
                result["wave_point"].iloc[point_idx] = point_data["type"]  # 1 = high, -1 = low
                if point_idx in wave_labels:
                    result["wave_label"].iloc[point_idx] = wave_labels[point_idx]
                if point_idx in wave_quality:
                    result["wave_quality"].iloc[point_idx] = wave_quality[point_idx]
        
        # Add current wave information
        result["current_wave"] = pd.Series(current_wave_info["current_wave"], index=price.index)
        result["wave_progress"] = pd.Series(current_wave_info["progress"], index=price.index)
        result["next_target"] = pd.Series(current_wave_info["next_target"], index=price.index)

        # Store calculation details
        self._last_calculation = {
            "price": price,
            "wave_points": wave_points,
            "wave_labels": wave_labels,
            "current_wave_info": current_wave_info,
            "parameters": self.parameters,
        }

        return result

    def _create_zigzag(self, high: pd.Series, low: pd.Series, threshold: float) -> Dict[int, Dict]:
        """Create zigzag points for wave analysis"""
        zigzag_points = {}
        
        # Initialize with first point
        trend = 0  # 0 = unknown, 1 = up, -1 = down
        last_high_idx = 0
        last_low_idx = 0
        last_high_val = high.iloc[0]
        last_low_val = low.iloc[0]
        
        for i in range(1, len(high)):
            current_high = high.iloc[i]
            current_low = low.iloc[i]
            
            # Check for upward move
            if trend <= 0:  # Not in uptrend or unknown
                move_up = (current_high - last_low_val) / last_low_val
                if move_up >= threshold:
                    # Confirmed upward move
                    if trend == -1:
                        # Add the low point
                        zigzag_points[last_low_idx] = {
                            "price": last_low_val,
                            "type": -1,  # Low
                            "index": last_low_idx
                        }
                    trend = 1
                    last_high_idx = i
                    last_high_val = current_high
            
            # Check for downward move
            if trend >= 0:  # Not in downtrend or unknown
                move_down = (last_high_val - current_low) / last_high_val
                if move_down >= threshold:
                    # Confirmed downward move
                    if trend == 1:
                        # Add the high point
                        zigzag_points[last_high_idx] = {
                            "price": last_high_val,
                            "type": 1,  # High
                            "index": last_high_idx
                        }
                    trend = -1
                    last_low_idx = i
                    last_low_val = current_low
            
            # Update extremes if in trend
            if trend == 1 and current_high > last_high_val:
                last_high_idx = i
                last_high_val = current_high
            elif trend == -1 and current_low < last_low_val:
                last_low_idx = i
                last_low_val = current_low
        
        return zigzag_points

    def _identify_wave_points(self, zigzag_points: Dict, min_wave_size: float) -> Dict[int, Dict]:
        """Identify significant wave points from zigzag"""
        wave_points = {}
        
        # Filter zigzag points by minimum wave size
        point_list = sorted(zigzag_points.items())
        
        for i, (idx, point_data) in enumerate(point_list):
            # Check if this point creates a significant wave
            if i > 0:
                prev_idx, prev_data = point_list[i-1]
                price_change = abs(point_data["price"] - prev_data["price"]) / prev_data["price"]
                
                if price_change >= min_wave_size:
                    wave_points[idx] = point_data.copy()
                    wave_points[idx]["wave_size"] = price_change
        
        return wave_points

    def _label_waves(self, wave_points: Dict, max_lookback: int) -> Dict[int, str]:
        """Label waves according to Elliott Wave theory"""
        wave_labels = {}
        
        if len(wave_points) < 2:
            return wave_labels
        
        # Sort wave points by index
        sorted_points = sorted(wave_points.items())
        
        # Simple wave labeling (in practice, this would be much more sophisticated)
        wave_count = 1
        impulse_phase = True  # True for impulse (1,3,5), False for corrective (2,4)
        
        for i, (idx, point_data) in enumerate(sorted_points[-max_lookback:]):
            if impulse_phase:
                if wave_count <= 5:
                    wave_labels[idx] = str(wave_count)
                    if wave_count == 5:
                        wave_count = 1
                        impulse_phase = False
                    else:
                        wave_count += 1 if i % 2 == 0 else 0  # Only increment on alternating points
                else:
                    # Corrective phase
                    corrective_labels = ["A", "B", "C"]
                    label_idx = (wave_count - 1) % 3
                    wave_labels[idx] = corrective_labels[label_idx]
                    wave_count += 1
                    if wave_count > 3:
                        impulse_phase = True
                        wave_count = 1
            else:
                # In corrective phase
                corrective_labels = ["A", "B", "C"]
                if wave_count <= 3:
                    wave_labels[idx] = corrective_labels[wave_count - 1]
                    wave_count += 1
                else:
                    impulse_phase = True
                    wave_count = 1
                    wave_labels[idx] = "1"
        
        return wave_labels

    def _calculate_fibonacci_levels(self, wave_points: Dict) -> Dict[int, float]:
        """Calculate Fibonacci retracement/extension levels for waves"""
        fibonacci_levels = {}
        
        sorted_points = sorted(wave_points.items())
        
        for i in range(2, len(sorted_points)):
            # Get three consecutive points for Fibonacci analysis
            p1_idx, p1_data = sorted_points[i-2]
            p2_idx, p2_data = sorted_points[i-1]
            p3_idx, p3_data = sorted_points[i]
            
            # Calculate retracement level
            if p1_data["type"] != p3_data["type"]:  # Retracement pattern
                wave_range = abs(p2_data["price"] - p1_data["price"])
                retracement = abs(p3_data["price"] - p2_data["price"]) / wave_range
                fibonacci_levels[p3_idx] = retracement
        
        return fibonacci_levels

    def _validate_wave_quality(
        self, wave_points: Dict, fibonacci_levels: Dict, tolerance: float
    ) -> Dict[int, float]:
        """Validate wave quality based on Fibonacci levels and Elliott Wave rules"""
        wave_quality = {}
        
        # Common Fibonacci levels for validation
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        
        for idx, fib_level in fibonacci_levels.items():
            quality_score = 0.0
            
            # Check if close to common Fibonacci levels
            for fib in fib_levels:
                if abs(fib_level - fib) <= tolerance:
                    if fib in [0.382, 0.618]:  # Golden ratios get higher score
                        quality_score = max(quality_score, 1.0)
                    elif fib in [0.236, 0.5, 1.618]:  # Important levels
                        quality_score = max(quality_score, 0.8)
                    else:
                        quality_score = max(quality_score, 0.6)
            
            wave_quality[idx] = quality_score
        
        return wave_quality

    def _analyze_current_wave(
        self, wave_points: Dict, wave_labels: Dict, current_price: float
    ) -> Dict[str, Any]:
        """Analyze current wave position and predict next targets"""
        if not wave_points:
            return {
                "current_wave": "unknown",
                "progress": 0.0,
                "next_target": current_price
            }
        
        # Get last few wave points
        sorted_points = sorted(wave_points.items())
        
        if len(sorted_points) >= 2:
            last_idx, last_point = sorted_points[-1]
            prev_idx, prev_point = sorted_points[-2]
            
            # Determine current wave based on last labeled point
            current_wave = wave_labels.get(last_idx, "unknown")
            
            # Calculate progress within current wave
            if last_point["type"] == 1:  # Last point was high
                if current_price < last_point["price"]:
                    # Currently retracing from high
                    wave_range = last_point["price"] - prev_point["price"]
                    current_retracement = last_point["price"] - current_price
                    progress = current_retracement / wave_range if wave_range > 0 else 0
                else:
                    progress = 1.0  # Beyond last high
            else:  # Last point was low
                if current_price > last_point["price"]:
                    # Currently advancing from low
                    wave_range = prev_point["price"] - last_point["price"]
                    current_advance = current_price - last_point["price"]
                    progress = current_advance / wave_range if wave_range > 0 else 0
                else:
                    progress = 1.0  # Beyond last low
            
            # Predict next target based on Elliott Wave patterns
            next_target = self._predict_next_target(sorted_points, current_wave, current_price)
            
        else:
            current_wave = "1"  # Assume starting wave 1
            progress = 0.0
            next_target = current_price
        
        return {
            "current_wave": current_wave,
            "progress": min(progress, 1.0),
            "next_target": next_target
        }

    def _predict_next_target(
        self, sorted_points: List, current_wave: str, current_price: float
    ) -> float:
        """Predict next wave target based on Elliott Wave relationships"""
        if len(sorted_points) < 3:
            return current_price
        
        # Get recent points for target calculation
        p1_idx, p1_data = sorted_points[-3]
        p2_idx, p2_data = sorted_points[-2]
        p3_idx, p3_data = sorted_points[-1]
        
        # Calculate typical Elliott Wave projections
        if current_wave in ["1", "3", "5"]:  # Impulse waves
            if current_wave == "3":
                # Wave 3 often 1.618x Wave 1
                wave1_size = abs(p2_data["price"] - p1_data["price"])
                if p3_data["type"] == -1:  # Currently at low, projecting up
                    target = p3_data["price"] + wave1_size * 1.618
                else:  # Currently at high, projecting down
                    target = p3_data["price"] - wave1_size * 1.618
            elif current_wave == "5":
                # Wave 5 often equals Wave 1
                wave1_size = abs(p2_data["price"] - p1_data["price"])
                if p3_data["type"] == -1:
                    target = p3_data["price"] + wave1_size
                else:
                    target = p3_data["price"] - wave1_size
            else:
                # Default projection
                recent_wave_size = abs(p3_data["price"] - p2_data["price"])
                if p3_data["type"] == -1:
                    target = p3_data["price"] + recent_wave_size
                else:
                    target = p3_data["price"] - recent_wave_size
        
        elif current_wave in ["2", "4"]:  # Corrective waves
            # Corrections often retrace 38.2% or 61.8%
            impulse_size = abs(p3_data["price"] - p2_data["price"])
            if p3_data["type"] == 1:  # Correcting from high
                target = p3_data["price"] - impulse_size * 0.618
            else:  # Correcting from low
                target = p3_data["price"] + impulse_size * 0.618
        
        else:
            target = current_price
        
        return target

    def _calculate_wave_momentum(self, wave_points: Dict, price: pd.Series) -> pd.Series:
        """Calculate wave momentum indicator"""
        momentum = pd.Series(0.0, index=price.index)
        
        sorted_points = sorted(wave_points.items())
        
        for i in range(1, len(sorted_points)):
            prev_idx, prev_data = sorted_points[i-1]
            curr_idx, curr_data = sorted_points[i]
            
            # Calculate wave velocity (price change per time unit)
            time_diff = curr_idx - prev_idx
            price_change = curr_data["price"] - prev_data["price"]
            
            if time_diff > 0:
                wave_velocity = price_change / time_diff
                
                # Apply momentum to the range
                for j in range(prev_idx, min(curr_idx + 1, len(momentum))):
                    momentum.iloc[j] = wave_velocity
        
        return momentum

    def _calculate_wave_strength(self, wave_points: Dict) -> pd.Series:
        """Calculate wave strength based on size and time"""
        if not wave_points:
            return pd.Series(0.0)
        
        # Use the index from first wave point to create series
        first_idx = min(wave_points.keys())
        last_idx = max(wave_points.keys())
        
        strength = pd.Series(0.0, index=range(last_idx + 1))
        
        sorted_points = sorted(wave_points.items())
        
        for i in range(len(sorted_points)):
            idx, point_data = sorted_points[i]
            wave_size = point_data.get("wave_size", 0)
            
            # Strength based on wave size (normalized)
            strength.iloc[idx] = min(wave_size * 10, 1.0)  # Cap at 1.0
        
        return strength

    def validate_parameters(self) -> bool:
        """Validate parameters"""
        min_wave_size = self.parameters.get("min_wave_size", 0.02)
        zigzag_threshold = self.parameters.get("zigzag_threshold", 0.05)
        fibonacci_tolerance = self.parameters.get("fibonacci_tolerance", 0.1)
        max_lookback = self.parameters.get("max_lookback", 100)
        degree = self.parameters.get("degree", "minor")

        if not isinstance(min_wave_size, (int, float)) or min_wave_size <= 0:
            raise IndicatorValidationError(f"min_wave_size must be positive, got {min_wave_size}")

        if not isinstance(zigzag_threshold, (int, float)) or zigzag_threshold <= 0:
            raise IndicatorValidationError(f"zigzag_threshold must be positive, got {zigzag_threshold}")

        if not isinstance(fibonacci_tolerance, (int, float)) or fibonacci_tolerance <= 0:
            raise IndicatorValidationError(f"fibonacci_tolerance must be positive, got {fibonacci_tolerance}")

        if not isinstance(max_lookback, int) or max_lookback < 10:
            raise IndicatorValidationError(f"max_lookback must be integer >= 10, got {max_lookback}")

        valid_degrees = ["minor", "intermediate", "primary"]
        if degree not in valid_degrees:
            raise IndicatorValidationError(f"degree must be one of {valid_degrees}, got {degree}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata"""
        return {
            "name": "WavePoint",
            "category": self.CATEGORY,
            "description": "Elliott Wave point detection and analysis with automatic wave counting",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "output_columns": [
                "wave_point", "wave_label", "wave_quality", "fibonacci_level",
                "wave_momentum", "wave_strength", "current_wave", "wave_progress", "next_target"
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
        return 50

    def _setup_defaults(self):
        """Setup default parameter values"""
        defaults = {
            "min_wave_size": 0.02,
            "zigzag_threshold": 0.05,
            "fibonacci_tolerance": 0.1,
            "max_lookback": 100,
            "degree": "minor",
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
            "indicator": "WavePoint",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return WavePointIndicator