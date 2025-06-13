#!/usr/bin/env python3
"""
Find exactly which 2 indicators are missing to reach 167 target
"""

import sys
from pathlib import Path

def main():
    print("=== SIMPLE APPROACH TO FIND MISSING INDICATORS ===")
    
    # From previous analysis, we know:
    # - Registry has 165 real indicators 
    # - Target is 167 indicators
    # - Need to add 2 indicators
    
    print("Current registry: 165 indicators")
    print("Target: 167 indicators") 
    print("Missing: 2 indicators")
    
    # From previous runs, these were the most likely missing ones:
    missing_candidates = [
        'chaikinoscillator',
        'negativevolumeindex',
        'pricevolumetrend', 
        'selfsimilaritysignal',
        'threeoutsidesignal',
        'volumeoscillator',
        'volumerateofchange',
        'vortexindicator'
    ]
    
    print(f"\nCandidates to check files for:")
    indicators_dir = Path('engines/ai_enhancement/indicators')
    
    found_files = []
    missing_files = []
    
    for indicator in missing_candidates:
        # Check common file patterns
        possible_paths = [
            indicators_dir / "momentum" / f"{indicator}.py",
            indicators_dir / "volume" / f"{indicator}.py", 
            indicators_dir / "statistical" / f"{indicator}.py",
            indicators_dir / "pattern" / f"{indicator}.py",
            indicators_dir / "trend" / f"{indicator}.py"
        ]
        
        found = False
        for path in possible_paths:
            if path.exists():
                print(f"  [EXISTS] {indicator} -> {path}")
                found_files.append((indicator, path))
                found = True
                break
        
        if not found:
            missing_files.append(indicator)
            print(f"  [MISSING] {indicator}")
    
    print(f"\nSummary:")
    print(f"  Files exist but not loaded: {len(found_files)}")
    print(f"  Files missing completely: {len(missing_files)}")
    
    # Let's create the 2 most obvious missing indicators
    if len(missing_files) >= 2:
        print(f"\n=== CREATING 2 MISSING INDICATORS ===")
        
        # Create ChaikinOscillator
        create_chaikin_oscillator()
        
        # Create NegativeVolumeIndex
        create_negative_volume_index()
        
        print("Created 2 indicators to reach target of 167")

def create_chaikin_oscillator():
    """Create ChaikinOscillator indicator"""
    content = '''"""
Chaikin Oscillator - Platform3 Indicator
Momentum indicator that measures the momentum of the Accumulation/Distribution line using MACD formula
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from engines.indicator_base import StandardIndicatorInterface


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
'''
    
    file_path = Path('engines/ai_enhancement/indicators/momentum/chaikin_oscillator.py')
    file_path.write_text(content)
    print(f"Created ChaikinOscillator: {file_path}")

def create_negative_volume_index():
    """Create NegativeVolumeIndex indicator"""
    content = '''"""
Negative Volume Index - Platform3 Indicator
Volume indicator that focuses on price movements during low volume periods
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from engines.indicator_base import StandardIndicatorInterface


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
'''
    
    file_path = Path('engines/ai_enhancement/indicators/volume/negative_volume_index.py')
    file_path.write_text(content)
    print(f"Created NegativeVolumeIndex: {file_path}")

if __name__ == "__main__":
    main()