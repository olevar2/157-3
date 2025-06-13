"""
Biorhythm Market Synth - Platform3 Physics Indicator
Biological rhythm synchronization with market cycles
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from engines.indicator_base import IndicatorBase
import math


@dataclass  
class BiorhythmResult:
    """Results from biorhythm analysis"""
    physical_cycle: float
    emotional_cycle: float
    intellectual_cycle: float
    composite_biorhythm: float
    cycle_phase: str
    sync_strength: float
    critical_days: List[int]
    optimal_trading_periods: List[str]
    

class BiorhythmMarketSynth(IndicatorBase):
    """
    Synchronizes biological rhythms with market cycles.
    
    Uses traditional biorhythm periods:
    - Physical: 23 days
    - Emotional: 28 days  
    - Intellectual: 33 days
    """
    
    def __init__(self, start_date: str = None):
        super().__init__()
        self.start_date = pd.Timestamp('2020-01-01') if start_date is None else pd.Timestamp(start_date)
        self.name = "BiorhythmMarketSynth"
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate biorhythm synchronization"""
        if len(data) < 33:  # Need at least one intellectual cycle
            return self._empty_result()
            
        current_date = data.index[-1] if isinstance(data.index[0], pd.Timestamp) else pd.Timestamp.now()
        days_since_start = (current_date - self.start_date).days
        
        # Calculate biorhythm cycles
        physical = math.sin(2 * math.pi * days_since_start / 23)
        emotional = math.sin(2 * math.pi * days_since_start / 28)  
        intellectual = math.sin(2 * math.pi * days_since_start / 33)
        
        # Composite biorhythm
        composite = (physical + emotional + intellectual) / 3
        
        # Determine cycle phase
        cycle_phase = self._determine_cycle_phase(physical, emotional, intellectual)
        
        # Calculate market synchronization strength
        sync_strength = self._calculate_sync_strength(data, composite)
        
        # Find critical days (cycle crossings)
        critical_days = self._find_critical_days(days_since_start)
        
        # Determine optimal trading periods
        optimal_periods = self._find_optimal_periods(physical, emotional, intellectual)
        
        result = BiorhythmResult(
            physical_cycle=physical,
            emotional_cycle=emotional,
            intellectual_cycle=intellectual,
            composite_biorhythm=composite,
            cycle_phase=cycle_phase,
            sync_strength=sync_strength,
            critical_days=critical_days,
            optimal_trading_periods=optimal_periods
        )
        
        return {
            'result': result,
            'signal': self._generate_signal(result),
            'confidence': self._calculate_confidence(result)
        }        
    def _determine_cycle_phase(self, physical: float, emotional: float, intellectual: float) -> str:
        """Determine current biorhythm phase"""
        avg_cycle = (physical + emotional + intellectual) / 3
        
        if avg_cycle > 0.5:
            return "high_energy"
        elif avg_cycle > 0:
            return "rising_energy" 
        elif avg_cycle > -0.5:
            return "declining_energy"
        else:
            return "low_energy"
            
    def _calculate_sync_strength(self, data: pd.DataFrame, composite: float) -> float:
        """Calculate synchronization between biorhythm and market"""
        # Use momentum as proxy for market energy
        momentum = data['close'].pct_change(5).iloc[-1] if len(data) >= 5 else 0
        
        # Normalize momentum to [-1, 1] range
        momentum_normalized = np.tanh(momentum * 100)
        
        # Calculate correlation/synchronization
        sync = abs(composite * momentum_normalized)
        return min(sync, 1.0)
        
    def _find_critical_days(self, days_since_start: int) -> List[int]:
        """Find upcoming critical days (cycle crossings)"""
        critical_days = []
        
        # Check next 30 days for cycle crossings
        for day_offset in range(1, 31):
            future_day = days_since_start + day_offset
            
            # Check if any cycle crosses zero
            physical_future = math.sin(2 * math.pi * future_day / 23)
            emotional_future = math.sin(2 * math.pi * future_day / 28)
            intellectual_future = math.sin(2 * math.pi * future_day / 33)
            
            physical_current = math.sin(2 * math.pi * (future_day - 1) / 23)
            emotional_current = math.sin(2 * math.pi * (future_day - 1) / 28)
            intellectual_current = math.sin(2 * math.pi * (future_day - 1) / 33)
            
            # Check for zero crossings
            if (physical_current * physical_future < 0 or
                emotional_current * emotional_future < 0 or
                intellectual_current * intellectual_future < 0):
                critical_days.append(day_offset)
                
        return critical_days[:5]  # Return next 5 critical days
        
    def _find_optimal_periods(self, physical: float, emotional: float, intellectual: float) -> List[str]:
        """Find optimal trading periods based on biorhythm alignment"""
        periods = []
        
        # All cycles positive (high energy)
        if physical > 0 and emotional > 0 and intellectual > 0:
            periods.append("optimal_buying")
            
        # All cycles negative (low energy)  
        if physical < 0 and emotional < 0 and intellectual < 0:
            periods.append("optimal_selling")
            
        # Intellectual high, others moderate (good for analysis)
        if intellectual > 0.5 and abs(physical) < 0.5 and abs(emotional) < 0.5:
            periods.append("optimal_analysis")
            
        # Emotional high (volatile period)
        if emotional > 0.7:
            periods.append("high_volatility_expected")
            
        return periods if periods else ["neutral_period"]
        
    def _generate_signal(self, result: BiorhythmResult) -> str:
        """Generate trading signal based on biorhythm analysis"""
        if ("optimal_buying" in result.optimal_trading_periods and 
            result.sync_strength > 0.5):
            return "BUY"
        elif ("optimal_selling" in result.optimal_trading_periods and
              result.sync_strength > 0.5):
            return "SELL"
        else:
            return "HOLD"
            
    def _calculate_confidence(self, result: BiorhythmResult) -> float:
        """Calculate signal confidence"""
        confidence = 0.3  # Base confidence
        
        confidence += result.sync_strength * 0.3
        
        if result.cycle_phase in ["high_energy", "low_energy"]:
            confidence += 0.2
            
        if len(result.critical_days) == 0:  # Not near critical day
            confidence += 0.2
            
        return min(confidence, 1.0)
        
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for insufficient data"""
        return {
            'result': None,
            'signal': 'HOLD',
            'confidence': 0.0
        }