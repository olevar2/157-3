"""
Quantum Momentum Oracle - Platform3 Physics Indicator
Quantum mechanics-inspired momentum analysis with uncertainty principles
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from engines.indicator_base import IndicatorBase


@dataclass
class QuantumResult:
    """Results from quantum momentum analysis"""
    momentum_eigenvalue: float
    position_uncertainty: float
    momentum_uncertainty: float
    wave_function_collapse: bool
    quantum_tunneling_probability: float
    entanglement_strength: float
    decoherence_time: float
    superposition_state: str
    
    
class QuantumMomentumOracle(IndicatorBase):
    """
    Applies quantum mechanics principles to momentum analysis.
    
    Uses concepts like:
    - Heisenberg uncertainty principle
    - Wave function collapse
    - Quantum tunneling
    - Entanglement between assets
    """
    
    def __init__(self, lookback: int = 14, uncertainty_factor: float = 0.05):
        super().__init__()
        self.lookback = lookback
        self.uncertainty_factor = uncertainty_factor
        self.name = "QuantumMomentumOracle"
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quantum momentum properties"""
        if len(data) < self.lookback:
            return self._empty_result()
            
        # Calculate momentum eigenvalue (discrete momentum levels)
        momentum = self._calculate_momentum_eigenvalue(data)
        
        # Apply Heisenberg uncertainty principle
        position_uncertainty, momentum_uncertainty = self._calculate_uncertainties(data)
        
        # Detect wave function collapse events
        wave_collapse = self._detect_wave_function_collapse(data)
        
        # Calculate quantum tunneling probability
        tunneling_prob = self._calculate_tunneling_probability(data)
        
        # Measure entanglement with market
        entanglement = self._measure_entanglement_strength(data)
        
        # Calculate decoherence time
        decoherence_time = self._calculate_decoherence_time(data)
        
        # Determine superposition state
        superposition = self._determine_superposition_state(data)
        
        result = QuantumResult(
            momentum_eigenvalue=momentum,
            position_uncertainty=position_uncertainty,
            momentum_uncertainty=momentum_uncertainty,
            wave_function_collapse=wave_collapse,
            quantum_tunneling_probability=tunneling_prob,
            entanglement_strength=entanglement,
            decoherence_time=decoherence_time,
            superposition_state=superposition
        )
        
        return {
            'result': result,
            'signal': self._generate_signal(result),
            'confidence': self._calculate_confidence(result)
        }
        
    def _calculate_momentum_eigenvalue(self, data: pd.DataFrame) -> float:
        """Calculate discrete momentum levels using ROC"""
        roc = data['close'].pct_change(self.lookback)
        return roc.iloc[-1] * 100  # Convert to percentage
        
    def _calculate_uncertainties(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Apply Heisenberg uncertainty principle to price/momentum"""
        # Position uncertainty (price volatility)
        position_std = data['close'].rolling(self.lookback).std()
        position_uncertainty = position_std.iloc[-1] / data['close'].iloc[-1]
        
        # Momentum uncertainty (momentum volatility)
        momentum = data['close'].pct_change()
        momentum_std = momentum.rolling(self.lookback).std()
        momentum_uncertainty = momentum_std.iloc[-1]
        
        # Ensure uncertainty principle holds: Δx * Δp >= ℏ/2 (scaled)
        uncertainty_product = position_uncertainty * momentum_uncertainty
        min_uncertainty = self.uncertainty_factor / 2
        
        if uncertainty_product < min_uncertainty:
            scaling_factor = np.sqrt(min_uncertainty / uncertainty_product)
            position_uncertainty *= scaling_factor
            momentum_uncertainty *= scaling_factor
            
        return position_uncertainty, momentum_uncertainty
        
    def _detect_wave_function_collapse(self, data: pd.DataFrame) -> bool:
        """Detect sudden 'measurement' events causing price collapse"""
        # Large price gaps or volume spikes indicate 'measurement'
        price_change = abs(data['close'].pct_change().iloc[-1])
        
        if 'volume' in data.columns:
            volume_spike = (data['volume'].iloc[-1] / 
                          data['volume'].rolling(self.lookback).mean().iloc[-1])
            return price_change > 0.02 or volume_spike > 2.0
        else:
            return price_change > 0.03
            
    def _calculate_tunneling_probability(self, data: pd.DataFrame) -> float:
        """Calculate probability of breaking through resistance/support"""
        # Use bollinger bands as quantum barriers
        ma = data['close'].rolling(self.lookback).mean()
        std = data['close'].rolling(self.lookback).std()
        
        upper_barrier = ma + 2 * std
        lower_barrier = ma - 2 * std
        current_price = data['close'].iloc[-1]
        
        # Calculate barrier penetration depth
        if current_price > upper_barrier.iloc[-1]:
            barrier_depth = (current_price - upper_barrier.iloc[-1]) / std.iloc[-1]
        elif current_price < lower_barrier.iloc[-1]:
            barrier_depth = (lower_barrier.iloc[-1] - current_price) / std.iloc[-1]
        else:
            barrier_depth = 0
            
        # Quantum tunneling probability (exponential decay)
        return np.exp(-barrier_depth) if barrier_depth > 0 else 0.0
        
    def _measure_entanglement_strength(self, data: pd.DataFrame) -> float:
        """Measure quantum entanglement with overall market trend"""
        # Correlation with moving average represents entanglement
        ma = data['close'].rolling(self.lookback).mean()
        correlation = data['close'].tail(self.lookback).corr(ma.tail(self.lookback))
        return abs(correlation) if not np.isnan(correlation) else 0.0
        
    def _calculate_decoherence_time(self, data: pd.DataFrame) -> float:
        """Calculate how long quantum effects persist"""
        # Autocorrelation decay time
        returns = data['close'].pct_change().dropna()
        autocorr = [returns.autocorr(lag) for lag in range(1, min(10, len(returns)))]
        
        # Find when autocorrelation drops below threshold
        threshold = 0.1
        for i, corr in enumerate(autocorr):
            if abs(corr) < threshold:
                return i + 1
        return len(autocorr)
        
    def _determine_superposition_state(self, data: pd.DataFrame) -> str:
        """Determine if price is in quantum superposition"""
        # Price near middle of trading range suggests superposition
        high_ma = data['high'].rolling(self.lookback).mean()
        low_ma = data['low'].rolling(self.lookback).mean()
        current_price = data['close'].iloc[-1]
        
        mid_point = (high_ma.iloc[-1] + low_ma.iloc[-1]) / 2
        range_size = high_ma.iloc[-1] - low_ma.iloc[-1]
        
        if abs(current_price - mid_point) < range_size * 0.2:
            return "superposition"
        elif current_price > mid_point:
            return "bullish_eigenstate"
        else:
            return "bearish_eigenstate"
            
    def _generate_signal(self, result: QuantumResult) -> str:
        """Generate trading signal based on quantum analysis"""
        if (result.wave_function_collapse and 
            result.quantum_tunneling_probability > 0.5 and
            result.momentum_eigenvalue > 0):
            return "BUY"
        elif (result.wave_function_collapse and 
              result.quantum_tunneling_probability > 0.5 and
              result.momentum_eigenvalue < 0):
            return "SELL"
        else:
            return "HOLD"
            
    def _calculate_confidence(self, result: QuantumResult) -> float:
        """Calculate signal confidence"""
        confidence = 0.5
        
        if result.wave_function_collapse:
            confidence += 0.2
        if result.quantum_tunneling_probability > 0.3:
            confidence += 0.15
        if result.entanglement_strength > 0.7:
            confidence += 0.1
        if result.decoherence_time > 3:
            confidence += 0.05
            
        return min(confidence, 1.0)
        
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for insufficient data"""
        return {
            'result': None,
            'signal': 'HOLD', 
            'confidence': 0.0
        }