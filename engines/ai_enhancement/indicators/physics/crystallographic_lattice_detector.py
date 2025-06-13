"""
CrystallographicLatticeDetector - Platform3 Physics Indicator  
Advanced Crystallographic Lattice Detector analysis for financial markets
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional
from engines.indicator_base import IndicatorBase


@dataclass
class CrystallographicLatticeDetectorResult:
    """Results from Crystallographic Lattice Detector analysis"""
    primary_value: float
    secondary_value: float
    pattern_strength: float
    physics_confidence: float
    signal_direction: str


class CrystallographicLatticeDetector(IndicatorBase):
    """
    Crystallographic Lattice Detector physics-based indicator.
    
    Applies advanced crystallographic lattice detector principles to market analysis.
    """
    
    def __init__(self, period: int = 21, sensitivity: float = 1.0):
        super().__init__()
        self.period = period
        self.sensitivity = sensitivity
        self.name = "CrystallographicLatticeDetector"
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate crystallographic lattice detector analysis"""
        if len(data) < self.period:
            return self._empty_result()
            
        # Physics-based calculations (simplified for demo)
        prices = data['close'].values
        
        # Primary analysis
        primary_value = self._calculate_primary_metric(prices)
        
        # Secondary analysis  
        secondary_value = self._calculate_secondary_metric(data)
        
        # Pattern strength using physics principles
        pattern_strength = self._calculate_pattern_strength(data)
        
        # Physics confidence
        physics_confidence = self._calculate_physics_confidence(data)
        
        # Signal direction
        signal_direction = self._determine_signal_direction(primary_value, secondary_value)
        
        result = CrystallographicLatticeDetectorResult(
            primary_value=primary_value,
            secondary_value=secondary_value,
            pattern_strength=pattern_strength,
            physics_confidence=physics_confidence,
            signal_direction=signal_direction
        )
        
        return {
            'result': result,
            'signal': self._generate_signal(result),
            'confidence': physics_confidence
        }
        
    def _calculate_primary_metric(self, prices: np.ndarray) -> float:
        """Calculate primary physics metric"""
        # Use mathematical transforms inspired by physics
        fft_transform = np.fft.fft(prices[-self.period:])
        dominant_frequency = np.abs(fft_transform).argmax()
        return float(dominant_frequency) * self.sensitivity
        
    def _calculate_secondary_metric(self, data: pd.DataFrame) -> float:
        """Calculate secondary physics metric"""
        # Energy-like calculation
        price_changes = data['close'].pct_change().dropna()
        kinetic_energy = 0.5 * np.sum(price_changes.tail(self.period) ** 2)
        return kinetic_energy
        
    def _calculate_pattern_strength(self, data: pd.DataFrame) -> float:
        """Calculate pattern strength using physics principles"""
        # Harmonic analysis
        prices = data['close'].tail(self.period)
        detrended = prices - prices.rolling(5).mean()
        strength = np.std(detrended) / np.mean(prices)
        return min(strength * 10, 1.0)
        
    def _calculate_physics_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence using physics-based metrics"""
        # Use autocorrelation as measure of system coherence
        returns = data['close'].pct_change().dropna().tail(self.period)
        if len(returns) > 5:
            autocorr = returns.autocorr(lag=1)
            confidence = abs(autocorr) if not np.isnan(autocorr) else 0.0
        else:
            confidence = 0.0
        return min(confidence + 0.3, 1.0)
        
    def _determine_signal_direction(self, primary: float, secondary: float) -> str:
        """Determine signal direction from physics metrics"""
        if primary > secondary:
            return "bullish"
        elif primary < secondary:
            return "bearish" 
        else:
            return "neutral"
            
    def _generate_signal(self, result: CrystallographicLatticeDetectorResult) -> str:
        """Generate trading signal"""
        if (result.signal_direction == "bullish" and 
            result.pattern_strength > 0.5 and 
            result.physics_confidence > 0.6):
            return "BUY"
        elif (result.signal_direction == "bearish" and 
              result.pattern_strength > 0.5 and 
              result.physics_confidence > 0.6):
            return "SELL"
        else:
            return "HOLD"
            
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for insufficient data"""
        return {
            'result': None,
            'signal': 'HOLD',
            'confidence': 0.0
        }
