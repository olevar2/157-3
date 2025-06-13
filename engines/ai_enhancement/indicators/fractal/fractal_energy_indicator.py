"""
Fractal Energy Indicator

Advanced fractal energy analysis using momentum and volatility patterns
to measure market energy states and sustainability.

Formula:
- Calculates kinetic energy from price momentum (velocity squared)
- Computes potential energy from volatility patterns
- Determines energy direction and sustainability metrics
- Provides power ratio analysis for trend strength assessment

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

# Import the base indicator interface
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
)


@dataclass
class FractalEnergyResult:
    """Result structure for Fractal Energy analysis"""
    energy_level: float
    momentum_strength: float
    energy_direction: str  # 'bullish', 'bearish', 'neutral'
    energy_sustainability: float
    power_ratio: float
    kinetic_energy: float
    potential_energy: float


class FractalEnergyIndicator(StandardIndicatorInterface):
    """
    Fractal Energy Indicator

    Advanced fractal energy analysis using momentum and volatility patterns to
    measure market energy states, direction, and sustainability for trend analysis.

    Key Features:
    - Energy-based momentum analysis using kinetic/potential energy separation
    - Energy direction and sustainability measurement
    - Power ratio analysis for trend strength assessment
    - Fractal-based energy calculations with volatility integration
    - Energy threshold analysis for trading signal generation

    Mathematical Approach:
    Applies physics-based energy concepts to financial markets using fractal geometry
    principles, measuring kinetic energy from price momentum and potential energy 
    from volatility patterns to assess market energy states and trend sustainability.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "fractal"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        energy_window: int = 14,
        momentum_period: int = 10,
        volatility_period: int = 20,
        energy_threshold: float = 0.6,
        **kwargs,
    ):
        """
        Initialize Fractal Energy Indicator

        Args:
            period: Main analysis period (default: 20)
            energy_window: Window for energy calculations (default: 14)
            momentum_period: Period for momentum analysis (default: 10)
            volatility_period: Period for volatility analysis (default: 20)
            energy_threshold: Energy threshold for signal generation (default: 0.6)
            **kwargs: Additional parameters
        """
        # Validate critical parameters before calling super()
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        if energy_window <= 0:
            raise ValueError(f"energy_window must be positive, got {energy_window}")
        
        # REQUIRED: Call parent constructor with all parameters
        super().__init__(
            period=period,
            energy_window=energy_window,
            momentum_period=momentum_period,
            volatility_period=volatility_period,
            energy_threshold=energy_threshold,
            **kwargs,
        )

    def calculate_kinetic_energy(self, prices: np.ndarray) -> float:
        """Calculate kinetic energy based on price momentum."""
        try:
            if len(prices) < 2:
                return 0.0

            # Calculate price velocity (rate of change)
            velocity = np.diff(prices)

            # Kinetic energy proportional to velocity squared
            kinetic_energy = np.mean(velocity**2)

            # Normalize by price level to make it scale-independent
            price_level = np.mean(prices)
            if price_level > 0:
                kinetic_energy = kinetic_energy / (price_level**2)

            return float(kinetic_energy)

        except Exception as e:
            raise IndicatorValidationError(f"Error calculating kinetic energy: {e}")

    def calculate_potential_energy(self, prices: np.ndarray) -> float:
        """Calculate potential energy based on volatility."""
        try:
            if len(prices) < 2:
                return 0.0

            # Calculate price volatility (standard deviation)
            volatility = np.std(prices)

            # Potential energy based on height differences (volatility)
            price_level = np.mean(prices)
            if price_level > 0:
                potential_energy = (volatility / price_level) ** 2
            else:
                potential_energy = 0.0

            return float(potential_energy)

        except Exception as e:
            raise IndicatorValidationError(f"Error calculating potential energy: {e}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fractal Energy analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Fractal energy analysis results
        """
        try:
            # Validate input data first
            self.validate_input_data(data)
            
            # Extract price data
            closes = data["close"].values
            
            period = self.parameters.get("period", 20)
            energy_window = self.parameters.get("energy_window", 14)
            momentum_period = self.parameters.get("momentum_period", 10)
            energy_threshold = self.parameters.get("energy_threshold", 0.6)

            if len(closes) < period:
                raise IndicatorValidationError(f"Insufficient data: need {period}, got {len(closes)}")

            # Initialize result arrays
            result_length = len(data)
            energy_level = np.full(result_length, np.nan)
            momentum_strength = np.full(result_length, np.nan)
            energy_direction = np.full(result_length, 0.0)  # 1=bullish, -1=bearish, 0=neutral
            energy_sustainability = np.full(result_length, np.nan)
            power_ratio = np.full(result_length, np.nan)
            kinetic_energy = np.full(result_length, np.nan)
            potential_energy = np.full(result_length, np.nan)

            # Calculate energy analysis for sliding windows
            for i in range(period - 1, result_length):
                window_data = closes[i - period + 1:i + 1]
                
                # Calculate kinetic and potential energy
                ke = self.calculate_kinetic_energy(window_data[-energy_window:])
                pe = self.calculate_potential_energy(window_data[-energy_window:])
                
                kinetic_energy[i] = ke
                potential_energy[i] = pe
                
                # Total energy level
                total_energy = ke + pe
                energy_level[i] = total_energy
                
                # Power ratio (kinetic vs potential)
                if pe > 0:
                    power_ratio[i] = ke / pe
                else:
                    power_ratio[i] = 1.0 if ke > 0 else 0.0
                
                # Momentum strength
                if len(window_data) >= momentum_period:
                    momentum_strength[i] = np.std(np.diff(window_data[-momentum_period:]))
                else:
                    momentum_strength[i] = 0.0
                
                # Energy direction
                recent_momentum = window_data[-1] - window_data[-min(momentum_period, len(window_data))]
                if recent_momentum > 0 and total_energy > energy_threshold:
                    energy_direction[i] = 1.0  # Bullish
                elif recent_momentum < 0 and total_energy > energy_threshold:
                    energy_direction[i] = -1.0  # Bearish
                else:
                    energy_direction[i] = 0.0  # Neutral
                
                # Energy sustainability (based on consistency of recent energy levels)
                if i >= energy_window:
                    recent_energies = energy_level[i-energy_window+1:i+1]
                    valid_energies = recent_energies[~np.isnan(recent_energies)]
                    if len(valid_energies) > 1:
                        sustainability = 1.0 - (np.std(valid_energies) / (np.mean(valid_energies) + 1e-10))
                        energy_sustainability[i] = max(0.0, min(1.0, sustainability))
                    else:
                        energy_sustainability[i] = 0.5
                else:
                    energy_sustainability[i] = 0.5

            # Create result DataFrame
            result_df = pd.DataFrame({
                "energy_level": energy_level,
                "momentum_strength": momentum_strength,
                "energy_direction": energy_direction,
                "energy_sustainability": energy_sustainability,
                "power_ratio": power_ratio,
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy,
            }, index=data.index)

            # Store calculation details for debugging
            self._last_calculation = {
                "final_energy_level": float(energy_level[-1]) if not np.isnan(energy_level[-1]) else None,
                "avg_sustainability": float(np.nanmean(energy_sustainability)),
                "parameters_used": self.parameters
            }

            return result_df

        except Exception as e:
            raise IndicatorValidationError(f"Error in FractalEnergyIndicator calculation: {e}")

    def validate_parameters(self) -> bool:
        """
        Validate indicator parameters for correctness and trading suitability.
        
        Returns:
            bool: True if parameters are valid
            
        Raises:
            IndicatorValidationError: If parameters are invalid
        """
        period = self.parameters.get("period", 20)
        energy_window = self.parameters.get("energy_window", 14)
        momentum_period = self.parameters.get("momentum_period", 10)
        volatility_period = self.parameters.get("volatility_period", 20)
        energy_threshold = self.parameters.get("energy_threshold", 0.6)
        
        # Validate parameter ranges
        if not isinstance(period, int) or period <= 0:
            raise IndicatorValidationError(f"period must be positive integer, got {period}")
        if not isinstance(energy_window, int) or energy_window <= 0:
            raise IndicatorValidationError(f"energy_window must be positive integer, got {energy_window}")
        if not isinstance(momentum_period, int) or momentum_period <= 0:
            raise IndicatorValidationError(f"momentum_period must be positive integer, got {momentum_period}")
        if not isinstance(energy_threshold, (int, float)) or energy_threshold < 0:
            raise IndicatorValidationError(f"energy_threshold must be non-negative number, got {energy_threshold}")
            
        return True

    def get_metadata(self) -> IndicatorMetadata:
        """
        Return comprehensive metadata about the indicator.
        
        Returns:
            IndicatorMetadata: Complete indicator specification
        """
        return IndicatorMetadata(
            name="FractalEnergyIndicator",
            category=self.CATEGORY,
            description="Advanced fractal energy analysis using momentum and volatility patterns",
            parameters=self.parameters,
            input_requirements=["close"],
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            trading_grade=True,
            performance_tier="standard",
            min_data_points=self._get_minimum_data_points(),
            max_lookback_period=self.parameters.get("period", 20)
        )

    def _get_required_columns(self) -> List[str]:
        """Get list of required data columns for this indicator."""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Get minimum number of data points required for calculation."""
        return self.parameters.get("period", 20)

    # Backward compatibility properties
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def energy_window(self) -> int:
        """Energy window for backward compatibility"""
        return self.parameters.get("energy_window", 14)


def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return FractalEnergyIndicator