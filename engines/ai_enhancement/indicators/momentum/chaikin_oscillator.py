"""
Chaikin Oscillator - Platform3 Indicator
Momentum indicator that measures the momentum of the Accumulation/Distribution line using MACD formula
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..base_indicator import StandardIndicatorInterface


@dataclass
class ChaikinOscillatorConfig:
    """Configuration for ChaikinOscillator indicator"""
    fast_period: int = 3
    slow_period: int = 10
    
    def validate(self) -> bool:
        return (self.fast_period > 0 and 
                self.slow_period > 0 and 
                self.fast_period < self.slow_period)


class ChaikinOscillator(StandardIndicatorInterface):
    """
    Chaikin Oscillator - Measures momentum of Accumulation/Distribution line
    
    The Chaikin Oscillator is a momentum indicator that applies the MACD formula 
    to the Accumulation/Distribution line. It's used to gauge the momentum of 
    buying and selling pressure.
    
    Formula:
    1. Calculate Accumulation/Distribution line: ((C-L)-(H-C))/(H-L) * Volume
    2. Apply MACD formula: EMA(fast) - EMA(slow) of A/D line
    """
    
    def __init__(self, config: Optional[ChaikinOscillatorConfig] = None):
        self.config = config or ChaikinOscillatorConfig()
        self.ad_line_history = []
        self.ema_fast_history = []
        self.ema_slow_history = []
        self.oscillator_history = []
        
        if not self.config.validate():
            raise ValueError("Invalid ChaikinOscillator configuration")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Chaikin Oscillator"""
        if len(data) < self.config.slow_period:
            return {"oscillator": None, "signal": "NEUTRAL"}
        
        high = data['high'].values
        low = data['low'].values  
        close = data['close'].values
        volume = data['volume'].values
        
        # Calculate Accumulation/Distribution line
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        ad_line = clv * volume
        
        # Calculate EMAs of A/D line
        ema_fast = self._calculate_ema(ad_line, self.config.fast_period)
        ema_slow = self._calculate_ema(ad_line, self.config.slow_period)
        
        # Chaikin Oscillator = EMA_fast - EMA_slow
        oscillator = ema_fast - ema_slow
        
        # Generate signal
        signal = self._generate_signal(oscillator)
        
        return {
            "oscillator": oscillator,
            "signal": signal,
            "ad_line": ad_line,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow
        }
    
    def _calculate_ema(self, values: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = values[0]
        
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
            
        return ema
    
    def _generate_signal(self, oscillator: float) -> str:
        """Generate trading signal based on oscillator value"""
        if oscillator > 0:
            return "BUY"
        elif oscillator < 0:
            return "SELL"
        else:
            return "NEUTRAL"
    
    def get_config(self) -> ChaikinOscillatorConfig:
        """Get current configuration"""
        return self.config
    
    def set_config(self, config: ChaikinOscillatorConfig) -> None:
        """Set new configuration"""
        if config.validate():
            self.config = config
        else:
            raise ValueError("Invalid configuration")
    
    def reset(self) -> None:
        """Reset indicator state"""
        self.ad_line_history.clear()
        self.ema_fast_history.clear()
        self.ema_slow_history.clear()
        self.oscillator_history.clear()
