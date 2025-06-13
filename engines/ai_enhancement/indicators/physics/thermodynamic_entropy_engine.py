"""
Thermodynamic Entropy Engine - Platform3 Physics Indicator
Advanced entropy analysis for financial market thermal equilibrium states
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from engines.indicator_base import IndicatorBase


@dataclass
class ThermodynamicResult:
    """Results from thermodynamic entropy analysis"""
    entropy_level: float
    temperature: float
    pressure: float
    volume: float
    energy_state: str
    phase_transition: Optional[str]
    equilibrium_distance: float
    thermal_gradient: float
    
    
class ThermodynamicEntropyEngine(IndicatorBase):
    """
    Applies thermodynamic principles to market analysis.
    
    Models market as a thermodynamic system with:
    - Price = Energy particles
    - Volume = System pressure
    - Volatility = Temperature
    - Liquidity = Molecular density
    """
    
    def __init__(self, lookback: int = 20, temp_factor: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.temp_factor = temp_factor
        self.name = "ThermodynamicEntropyEngine"
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate thermodynamic properties"""
        if len(data) < self.lookback:
            return self._empty_result()
            
        # Calculate market temperature (volatility-based)
        returns = data['close'].pct_change()
        temperature = returns.rolling(self.lookback).std() * self.temp_factor
        
        # Calculate market pressure (volume-normalized price changes)
        if 'volume' in data.columns:
            pressure = (returns.abs() * data['volume']).rolling(self.lookback).mean()
        else:
            pressure = returns.abs().rolling(self.lookback).mean()
            
        # Calculate market volume (price range expansion)
        high_low_range = (data['high'] - data['low']) / data['close']
        volume = high_low_range.rolling(self.lookback).mean()
        
        # Calculate entropy using Boltzmann formula adaptation
        prob_dist = self._calculate_price_probability_distribution(data)
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        
        # Determine energy state
        energy_state = self._classify_energy_state(temperature.iloc[-1], pressure.iloc[-1])
        
        # Check for phase transitions
        phase_transition = self._detect_phase_transition(data)
        
        # Calculate equilibrium distance
        equilibrium_distance = self._calculate_equilibrium_distance(data)
        
        result = ThermodynamicResult(
            entropy_level=entropy,
            temperature=temperature.iloc[-1],
            pressure=pressure.iloc[-1], 
            volume=volume.iloc[-1],
            energy_state=energy_state,
            phase_transition=phase_transition,
            equilibrium_distance=equilibrium_distance,
            thermal_gradient=self._calculate_thermal_gradient(temperature)
        )
        
        return {
            'result': result,
            'signal': self._generate_signal(result),
            'confidence': self._calculate_confidence(result)
        }
        
    def _calculate_price_probability_distribution(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate probability distribution of price movements"""
        returns = data['close'].pct_change().dropna()
        hist, bins = np.histogram(returns.tail(self.lookback), bins=10, density=True)
        return hist / np.sum(hist)
        
    def _classify_energy_state(self, temperature: float, pressure: float) -> str:
        """Classify current market energy state"""
        if temperature > 0.02 and pressure > 0.01:
            return "high_energy"
        elif temperature > 0.01 and pressure > 0.005:
            return "medium_energy"
        else:
            return "low_energy"
            
    def _detect_phase_transition(self, data: pd.DataFrame) -> Optional[str]:
        """Detect market phase transitions"""
        # Simple momentum-based phase detection
        momentum = data['close'].rolling(5).mean() - data['close'].rolling(20).mean()
        recent_momentum = momentum.tail(3)
        
        if all(recent_momentum > 0) and momentum.iloc[-4] <= 0:
            return "bullish_transition"
        elif all(recent_momentum < 0) and momentum.iloc[-4] >= 0:
            return "bearish_transition"
        return None
        
    def _calculate_equilibrium_distance(self, data: pd.DataFrame) -> float:
        """Calculate distance from thermal equilibrium"""
        ma_20 = data['close'].rolling(20).mean()
        current_price = data['close'].iloc[-1]
        return abs(current_price - ma_20.iloc[-1]) / ma_20.iloc[-1]
        
    def _calculate_thermal_gradient(self, temperature: pd.Series) -> float:
        """Calculate rate of temperature change"""
        return temperature.diff().iloc[-1] if len(temperature) > 1 else 0.0
        
    def _generate_signal(self, result: ThermodynamicResult) -> str:
        """Generate trading signal based on thermodynamic analysis"""
        if result.phase_transition == "bullish_transition" and result.energy_state != "low_energy":
            return "BUY"
        elif result.phase_transition == "bearish_transition" and result.energy_state == "high_energy":
            return "SELL"
        else:
            return "HOLD"
            
    def _calculate_confidence(self, result: ThermodynamicResult) -> float:
        """Calculate signal confidence"""
        confidence = 0.5
        
        if result.phase_transition:
            confidence += 0.2
        if result.energy_state == "high_energy":
            confidence += 0.1
        if result.equilibrium_distance > 0.02:
            confidence += 0.1
            
        return min(confidence, 1.0)
        
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for insufficient data"""
        return {
            'result': None,
            'signal': 'HOLD',
            'confidence': 0.0
        }