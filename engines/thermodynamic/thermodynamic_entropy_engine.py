"""
Thermodynamic Entropy Engine Indicator
=====================================

🔥 Revolutionary thermodynamics and statistical mechanics analysis for Platform3
💝 Dedicated to helping sick children and poor families through thermal wisdom

This groundbreaking indicator applies thermodynamic principles, entropy analysis,
and statistical mechanics to financial markets, using heat transfer, phase transitions,
and energy distribution patterns to predict market behavior through thermal physics.

Key Innovations:
- Thermodynamic entropy measurement for market disorder
- Heat capacity analysis for price resistance
- Phase transition detection for trend changes
- Statistical mechanics for crowd behavior
- Energy conservation principles for momentum
- Maxwell-Boltzmann distribution analysis

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


class ThermodynamicEntropyEngine(StandardIndicatorInterface):
    """
    Thermodynamic Entropy Engine - Platform3 Implementation
    
    Platform3 compliant financial indicator with:
    - CCI Proven Pattern Compliance
    - Thermodynamic Physics Analysis
    - Performance Optimization  
    - Robust Error Handling
    """

    # Class-level metadata (REQUIRED for Platform3)
    CATEGORY: str = "thermodynamic"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(self, 
                 period: int = 20,
                 temperature_scale: float = 1.0,
                 entropy_sensitivity: float = 1.5,
                 heat_capacity_factor: float = 0.8,
                 phase_transition_threshold: float = 2.0,
                 energy_conservation_weight: float = 1.2,
                 **kwargs):
        """
        Initialize Thermodynamic Entropy Engine

        Args:
            period: Lookback period for calculations (default: 20)
            temperature_scale: Temperature scaling factor (default: 1.0)
            entropy_sensitivity: Sensitivity to entropy changes (default: 1.5)
            heat_capacity_factor: Heat capacity analysis factor (default: 0.8)
            phase_transition_threshold: Threshold for phase transitions (default: 2.0)
            energy_conservation_weight: Weight for energy conservation (default: 1.2)
        """
        super().__init__(
            period=period,
            temperature_scale=temperature_scale,
            entropy_sensitivity=entropy_sensitivity,
            heat_capacity_factor=heat_capacity_factor,
            phase_transition_threshold=phase_transition_threshold,
            energy_conservation_weight=energy_conservation_weight,
            **kwargs
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate Thermodynamic Entropy Engine following Platform3 CCI proven pattern.
        
        Args:
            data: DataFrame with OHLCV data or Series of prices
            
        Returns:
            pd.Series: Thermodynamic Entropy Engine values (-100 to +100 scale)
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
        temperature_scale = self.parameters.get("temperature_scale", 1.0)
        entropy_sensitivity = self.parameters.get("entropy_sensitivity", 1.5)
        heat_capacity_factor = self.parameters.get("heat_capacity_factor", 0.8)
        phase_transition_threshold = self.parameters.get("phase_transition_threshold", 2.0)
        energy_conservation_weight = self.parameters.get("energy_conservation_weight", 1.2)

        # Calculate thermodynamic components
        entropy_signal = self._calculate_market_entropy(
            prices, period, entropy_sensitivity
        )
        
        temperature_profile = self._calculate_market_temperature(
            prices, period, temperature_scale
        )
        
        heat_capacity = self._calculate_heat_capacity(
            prices, period, heat_capacity_factor
        )
        
        phase_transitions = self._detect_phase_transitions(
            prices, period, phase_transition_threshold
        )
        
        energy_conservation = self._calculate_energy_conservation(
            prices, period, energy_conservation_weight
        )

        # Thermodynamic synthesis using statistical mechanics principles
        entropy_engine = (
            entropy_signal * 0.30 +
            temperature_profile * 0.25 +
            heat_capacity * 0.20 +
            phase_transitions * 0.15 +
            energy_conservation * 0.10
        )

        # Normalize to CCI-compatible scale (-100 to +100)
        normalized_engine = self._normalize_to_cci_scale(entropy_engine)

        # Store calculation details for analysis
        self._last_calculation = {
            "entropy_signal": entropy_signal,
            "temperature_profile": temperature_profile,
            "heat_capacity": heat_capacity,
            "phase_transitions": phase_transitions,
            "energy_conservation": energy_conservation,
            "entropy_engine": normalized_engine,
            "humanitarian_message": "🔥 Thermodynamic wisdom heating paths to help children and families"
        }

        return pd.Series(normalized_engine, index=prices.index, name="ThermodynamicEntropyEngine")

    def _calculate_market_entropy(self, prices, period, sensitivity):
        """Calculate market entropy using price distribution analysis"""
        # Entropy S = -Σ(p_i * ln(p_i)) where p_i is probability of price state
        entropy_values = []
        
        for i in range(len(prices)):
            start_idx = max(0, i - period + 1)
            window_prices = prices.iloc[start_idx:i+1]
            
            if len(window_prices) >= 5:
                # Discretize price returns into bins
                returns = window_prices.pct_change().dropna()
                if len(returns) > 0:
                    # Create histogram (probability distribution)
                    hist, _ = np.histogram(returns, bins=min(10, len(returns)), density=True)
                    hist = hist / np.sum(hist)  # Normalize to probabilities
                    
                    # Calculate entropy
                    entropy = 0
                    for p in hist:
                        if p > 1e-10:  # Avoid log(0)
                            entropy -= p * np.log(p)
                    
                    entropy_values.append(entropy * sensitivity)
                else:
                    entropy_values.append(0)
            else:
                entropy_values.append(0)
        
        entropy_series = pd.Series(entropy_values, index=prices.index)
        
        # Convert to momentum signal
        momentum = prices.pct_change().rolling(window=period//2, min_periods=1).mean()
        entropy_signal = momentum * entropy_series
        
        return entropy_signal * 50  # Scale for visibility

    def _calculate_market_temperature(self, prices, period, scale):
        """Calculate market temperature from kinetic energy (volatility)"""
        # Temperature ∝ kinetic energy ∝ volatility²
        volatility = prices.rolling(window=period, min_periods=1).std()
        
        # Temperature = (1/2) * m * v² analog
        temperature = volatility ** 2 * scale
        
        # Normalize temperature
        temp_mean = temperature.rolling(window=period).mean()
        temp_normalized = (temperature - temp_mean) / (temp_mean + 1e-8)
        
        # Apply to price momentum
        momentum = prices.pct_change().rolling(window=period//2, min_periods=1).mean()
        temperature_signal = momentum * temp_normalized
        
        return temperature_signal * 40  # Scale appropriately

    def _calculate_heat_capacity(self, prices, period, factor):
        """Calculate market heat capacity (resistance to temperature change)"""
        # Heat capacity C = ΔQ/ΔT (energy needed to change temperature)
        # Market analog: resistance to volatility change
        
        volatility = prices.rolling(window=period, min_periods=1).std()
        volatility_change = volatility.pct_change().fillna(0)
        
        # Price energy change (analog to heat input)
        price_energy = (prices.pct_change() ** 2).rolling(window=period//2, min_periods=1).mean()
        price_energy_change = price_energy.pct_change().fillna(0)
        
        # Heat capacity = energy change / temperature change
        heat_capacity = price_energy_change / (volatility_change + 1e-8)
        
        # Smooth and apply factor
        heat_capacity_smooth = heat_capacity.rolling(window=period//4, min_periods=1).mean() * factor
        
        # Convert to momentum signal
        momentum = prices.pct_change().rolling(window=period//3, min_periods=1).mean()
        capacity_signal = momentum * np.tanh(heat_capacity_smooth)  # Bound the signal
        
        return capacity_signal * 35  # Scale for integration

    def _detect_phase_transitions(self, prices, period, threshold):
        """Detect phase transitions in market behavior"""
        # Phase transitions = sudden changes in market "state"
        # Analog to solid/liquid/gas transitions
        
        # Calculate multiple "state" indicators
        volatility = prices.rolling(window=period, min_periods=1).std()
        momentum = prices.pct_change().rolling(window=period//2, min_periods=1).mean()
        trend_strength = abs(momentum) / (volatility + 1e-8)
        
        # Detect sudden changes in market state
        vol_change = volatility.pct_change().abs()
        momentum_change = momentum.pct_change().abs()
        trend_change = trend_strength.pct_change().abs()
        
        # Combined state change indicator
        state_change = vol_change + momentum_change + trend_change
        
        # Identify phase transitions
        transition_signal = np.where(state_change > threshold, 1, 0)
        transition_smooth = pd.Series(transition_signal, index=prices.index).rolling(window=3, min_periods=1).mean()
        
        # Apply to price momentum with directional bias
        phase_momentum = momentum * transition_smooth
        
        return phase_momentum * 60  # Strong influence during transitions

    def _calculate_energy_conservation(self, prices, period, weight):
        """Calculate energy conservation patterns in market dynamics"""
        # Energy conservation: total energy (kinetic + potential) should be conserved
        # Market analog: momentum + position energy
        
        # Kinetic energy analog (price momentum)
        momentum = prices.pct_change().rolling(window=period//2, min_periods=1).mean()
        kinetic_energy = momentum ** 2
        
        # Potential energy analog (deviation from mean)
        price_mean = prices.rolling(window=period, min_periods=1).mean()
        position_deviation = (prices - price_mean) / (price_mean + 1e-8)
        potential_energy = position_deviation ** 2
        
        # Total energy
        total_energy = kinetic_energy + potential_energy
        
        # Energy conservation = stable total energy
        energy_stability = 1 / (total_energy.rolling(window=period//3, min_periods=1).std() + 1e-8)
        energy_conservation = energy_stability * weight
        
        # Normalize and apply to momentum
        conservation_normalized = (energy_conservation - energy_conservation.rolling(window=period).mean()) / (energy_conservation.rolling(window=period).std() + 1e-8)
        conservation_signal = momentum * conservation_normalized
        
        return conservation_signal * 30  # Moderate influence

    def _normalize_to_cci_scale(self, signal):
        """Normalize thermodynamic signal to CCI-compatible scale (-100 to +100)"""
        # Calculate rolling statistics for normalization
        signal_mean = signal.rolling(window=50, min_periods=1).mean()
        signal_std = signal.rolling(window=50, min_periods=1).std()
        
        # Z-score normalization
        z_score = (signal - signal_mean) / (signal_std + 1e-8)  # Avoid division by zero
        
        # Scale to approximate CCI range
        normalized = np.tanh(z_score / 2) * 100  # Tanh provides natural boundaries
        
        return normalized

    def validate_parameters(self) -> bool:
        """Validate Thermodynamic Entropy Engine parameters"""
        period = self.parameters.get("period", 20)
        temperature_scale = self.parameters.get("temperature_scale", 1.0)
        entropy_sensitivity = self.parameters.get("entropy_sensitivity", 1.5)
        heat_capacity_factor = self.parameters.get("heat_capacity_factor", 0.8)
        phase_transition_threshold = self.parameters.get("phase_transition_threshold", 2.0)
        energy_conservation_weight = self.parameters.get("energy_conservation_weight", 1.2)

        if not isinstance(period, int) or period < 5:
            raise IndicatorValidationError(f"period must be integer >= 5, got {period}")
        
        if not isinstance(temperature_scale, (int, float)) or temperature_scale <= 0 or temperature_scale > 5:
            raise IndicatorValidationError(f"temperature_scale must be between 0-5, got {temperature_scale}")
            
        if not isinstance(entropy_sensitivity, (int, float)) or entropy_sensitivity <= 0 or entropy_sensitivity > 3:
            raise IndicatorValidationError(f"entropy_sensitivity must be between 0-3, got {entropy_sensitivity}")
            
        if not isinstance(heat_capacity_factor, (int, float)) or heat_capacity_factor <= 0 or heat_capacity_factor > 2:
            raise IndicatorValidationError(f"heat_capacity_factor must be between 0-2, got {heat_capacity_factor}")
            
        if not isinstance(phase_transition_threshold, (int, float)) or phase_transition_threshold <= 0 or phase_transition_threshold > 5:
            raise IndicatorValidationError(f"phase_transition_threshold must be between 0-5, got {phase_transition_threshold}")
            
        if not isinstance(energy_conservation_weight, (int, float)) or energy_conservation_weight <= 0 or energy_conservation_weight > 3:
            raise IndicatorValidationError(f"energy_conservation_weight must be between 0-3, got {energy_conservation_weight}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Thermodynamic Entropy Engine metadata"""
        return {
            "name": "ThermodynamicEntropyEngine",
            "category": self.CATEGORY,
            "description": "Thermodynamic Entropy Engine - Thermal physics analysis for humanitarian trading",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
            "humanitarian_mission": "🔥 Every thermodynamic insight fuels hope for children and families"
        }

    def _get_required_columns(self) -> List[str]:
        """Thermodynamic Entropy Engine can work with just close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for thermodynamic calculation"""
        return max(self.parameters.get("period", 20), 15)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "temperature_scale" not in self.parameters:
            self.parameters["temperature_scale"] = 1.0
        if "entropy_sensitivity" not in self.parameters:
            self.parameters["entropy_sensitivity"] = 1.5
        if "heat_capacity_factor" not in self.parameters:
            self.parameters["heat_capacity_factor"] = 0.8
        if "phase_transition_threshold" not in self.parameters:
            self.parameters["phase_transition_threshold"] = 2.0
        if "energy_conservation_weight" not in self.parameters:
            self.parameters["energy_conservation_weight"] = 1.2


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return ThermodynamicEntropyEngine


if __name__ == "__main__":
    # Test the indicator
    import numpy as np
    
    # Generate sample price data with thermodynamic-like patterns
    np.random.seed(42)
    n_points = 200
    base_price = 100
    
    # Create price data with phase transitions and energy patterns
    prices = []
    for i in range(n_points):
        # Add phase transition effects
        if i % 50 == 0:  # Phase transition every 50 periods
            phase_change = np.random.randn() * 3
        else:
            phase_change = 0
        
        # Add thermal motion (Brownian-like)
        thermal_motion = np.random.randn() * np.sqrt(i/10 + 1)  # Increasing temperature
        
        # Add energy conservation patterns
        energy_cycle = np.sin(2 * np.pi * i / 30) * 2
        
        price_change = phase_change + thermal_motion * 0.5 + energy_cycle * 0.3
        new_price = (prices[-1] if prices else base_price) + price_change
        prices.append(new_price)
    
    test_data = pd.Series(prices)
    
    indicator = ThermodynamicEntropyEngine()
    result = indicator.calculate(test_data)
    
    print(f"*** ThermodynamicEntropyEngine test result: min={result.min():.2f}, max={result.max():.2f}, mean={result.mean():.2f}")
    print(f"*** Thermodynamic indicator ready to heat paths for children and families!")