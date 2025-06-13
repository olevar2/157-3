"""
Volume Oscillator

The Volume Oscillator measures the difference between fast and slow volume moving averages,
helping to identify changes in volume trends and momentum.

Formula:
Volume Oscillator = ((Short Volume MA - Long Volume MA) / Long Volume MA) * 100

Interpretation:
- Positive values: Short-term volume above long-term average (increasing volume momentum)
- Negative values: Short-term volume below long-term average (decreasing volume momentum)
- Zero line crossovers: Changes in volume momentum direction
- Extreme readings: Potential volume exhaustion or acceleration

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
    IndicatorMetadata,
)


class VolumeOscillator(StandardIndicatorInterface):
    """
    Volume Oscillator for volume momentum analysis
    
    Identifies changes in volume trends and helps confirm price movements
    through volume momentum analysis.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "volume"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        short_period: int = 5,
        long_period: int = 20,
        signal_period: int = 10,
        threshold: float = 20.0,
        **kwargs,
    ):
        """
        Initialize Volume Oscillator

        Args:
            short_period: Short-term volume MA period (default: 5)
            long_period: Long-term volume MA period (default: 20)
            signal_period: Signal line period (default: 10)
            threshold: Threshold for extreme readings (default: 20.0)
        """
        super().__init__(
            short_period=short_period,
            long_period=long_period,
            signal_period=signal_period,
            threshold=threshold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Volume Oscillator

        Args:
            data: DataFrame with 'volume' column

        Returns:
            pd.DataFrame: DataFrame with Volume Oscillator and analysis
        """
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            volume = data
        else:
            if 'volume' not in data.columns:
                raise IndicatorValidationError("Volume Oscillator requires 'volume' column")
            volume = data['volume']

        short_period = self.parameters.get("short_period", 5)
        long_period = self.parameters.get("long_period", 20)
        signal_period = self.parameters.get("signal_period", 10)
        threshold = self.parameters.get("threshold", 20.0)
        
        # Calculate volume moving averages
        volume_short_ma = volume.rolling(window=short_period, min_periods=1).mean()
        volume_long_ma = volume.rolling(window=long_period, min_periods=1).mean()
        
        # Calculate Volume Oscillator
        volume_oscillator = ((volume_short_ma - volume_long_ma) / volume_long_ma) * 100
        
        # Calculate signal line
        signal_line = volume_oscillator.rolling(window=signal_period, min_periods=1).mean()
        
        # Calculate histogram (difference between oscillator and signal)
        histogram = volume_oscillator - signal_line
        
        # Generate signals
        signals = pd.Series(index=data.index if isinstance(data, pd.DataFrame) else volume.index, dtype=str)
        signals[:] = "neutral"
        
        # Crossover signals
        oscillator_cross_up = (volume_oscillator > 0) & (volume_oscillator.shift(1) <= 0)
        oscillator_cross_down = (volume_oscillator < 0) & (volume_oscillator.shift(1) >= 0)
        
        # Extreme readings
        extreme_high = volume_oscillator > threshold
        extreme_low = volume_oscillator < -threshold
        
        signals[oscillator_cross_up] = "volume_momentum_up"
        signals[oscillator_cross_down] = "volume_momentum_down"
        signals[extreme_high] = "volume_exhaustion_high"
        signals[extreme_low] = "volume_exhaustion_low"
        
        # Create result DataFrame
        result = pd.DataFrame({
            'volume_oscillator': volume_oscillator,
            'signal_line': signal_line,
            'histogram': histogram,
            'volume_short_ma': volume_short_ma,
            'volume_long_ma': volume_long_ma,
            'signals': signals
        }, index=data.index if isinstance(data, pd.DataFrame) else volume.index)
        
        self._last_calculation = {
            "volume_oscillator": volume_oscillator,
            "final_value": volume_oscillator.iloc[-1] if len(volume_oscillator) > 0 else 0,
            "signals": signals,
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate Volume Oscillator parameters"""
        short_period = self.parameters.get("short_period", 5)
        long_period = self.parameters.get("long_period", 20)
        signal_period = self.parameters.get("signal_period", 10)
        threshold = self.parameters.get("threshold", 20.0)

        if not isinstance(short_period, int) or short_period < 1:
            raise IndicatorValidationError(f"short_period must be integer >= 1, got {short_period}")
        
        if not isinstance(long_period, int) or long_period < 1:
            raise IndicatorValidationError(f"long_period must be integer >= 1, got {long_period}")
            
        if short_period >= long_period:
            raise IndicatorValidationError(f"short_period must be < long_period")
            
        if not isinstance(signal_period, int) or signal_period < 1:
            raise IndicatorValidationError(f"signal_period must be integer >= 1, got {signal_period}")
            
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise IndicatorValidationError(f"threshold must be positive number, got {threshold}")

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """Return Volume Oscillator metadata"""
        return IndicatorMetadata(
            name="VolumeOscillator",
            category=self.CATEGORY,
            description="Volume Oscillator - Volume momentum indicator",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        """Volume Oscillator requires volume data"""
        return ["volume"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        return self.parameters.get("long_period", 20)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "short_period" not in self.parameters:
            self.parameters["short_period"] = 5
        if "long_period" not in self.parameters:
            self.parameters["long_period"] = 20
        if "signal_period" not in self.parameters:
            self.parameters["signal_period"] = 10
        if "threshold" not in self.parameters:
            self.parameters["threshold"] = 20.0

    @property
    def short_period(self) -> int:
        return self.parameters.get("short_period", 5)

    @property
    def long_period(self) -> int:
        return self.parameters.get("long_period", 20)

    @property
    def minimum_periods(self) -> int:
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "name": "VolumeOscillator",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return VolumeOscillator