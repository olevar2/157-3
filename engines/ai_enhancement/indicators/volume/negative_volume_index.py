"""
Negative Volume Index - Platform3 Indicator
Volume indicator that focuses on price movements during low volume periods
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..base_indicator import StandardIndicatorInterface


@dataclass
class NegativeVolumeIndexConfig:
    """Configuration for NegativeVolumeIndex indicator"""
    base_value: float = 1000.0
    
    def validate(self) -> bool:
        return self.base_value > 0


class NegativeVolumeIndex(StandardIndicatorInterface):
    """
    Negative Volume Index (NVI) - Tracks price changes when volume decreases
    
    The NVI is based on the premise that uninformed investors are active 
    when volume increases, while informed investors are active when volume 
    decreases. It only changes when volume decreases from the previous period.
    
    Formula:
    - If volume today < volume yesterday:
      NVI = Previous NVI + ((Close - Previous Close) / Previous Close) * Previous NVI
    - If volume today >= volume yesterday:
      NVI = Previous NVI (unchanged)
    """
    
    def __init__(self, config: Optional[NegativeVolumeIndexConfig] = None):
        self.config = config or NegativeVolumeIndexConfig()
        self.nvi_history = []
        self.prev_close = None
        self.prev_volume = None
        
        if not self.config.validate():
            raise ValueError("Invalid NegativeVolumeIndex configuration")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Negative Volume Index"""
        if len(data) < 2:
            return {"nvi": self.config.base_value, "signal": "NEUTRAL"}
        
        close = data['close'].values
        volume = data['volume'].values
        
        nvi_values = [self.config.base_value]
        
        for i in range(1, len(close)):
            current_close = close[i]
            current_volume = volume[i]
            previous_close = close[i-1]
            previous_volume = volume[i-1]
            previous_nvi = nvi_values[i-1]
            
            if current_volume < previous_volume:
                # Volume decreased - update NVI
                price_change_pct = (current_close - previous_close) / previous_close
                new_nvi = previous_nvi + (price_change_pct * previous_nvi)
            else:
                # Volume increased or same - NVI unchanged
                new_nvi = previous_nvi
            
            nvi_values.append(new_nvi)
        
        current_nvi = nvi_values[-1]
        signal = self._generate_signal(nvi_values)
        
        return {
            "nvi": current_nvi,
            "nvi_series": nvi_values,
            "signal": signal
        }
    
    def _generate_signal(self, nvi_values: List[float]) -> str:
        """Generate trading signal based on NVI trend"""
        if len(nvi_values) < 2:
            return "NEUTRAL"
        
        current = nvi_values[-1]
        previous = nvi_values[-2]
        
        if current > previous:
            return "BUY"  # NVI increasing
        elif current < previous:
            return "SELL"  # NVI decreasing
        else:
            return "NEUTRAL"
    
    def get_config(self) -> NegativeVolumeIndexConfig:
        """Get current configuration"""
        return self.config
    
    def set_config(self, config: NegativeVolumeIndexConfig) -> None:
        """Set new configuration"""
        if config.validate():
            self.config = config
        else:
            raise ValueError("Invalid configuration")
    
    def reset(self) -> None:
        """Reset indicator state"""
        self.nvi_history.clear()
        self.prev_close = None
        self.prev_volume = None
