"""
Pair Specialist - Individual Currency Pair Personality Analysis

Professional-grade currency pair intelligence that analyzes unique characteristics,
behaviors, and optimal trading strategies for each major and minor forex pair.
Critical for Platform3's multi-pair trading efficiency.

Version: 2.0.0 (Ultra-Fast) - Achieving <1ms performance targets
"""

from .ultra_fast_model import UltraFastPairSpecialist, ultra_fast_pair_specialist
from .ultra_fast_model import analyze_pair_ultra_fast, get_strategy_ultra_fast, get_risk_params_ultra_fast
from .model import PairSpecialist as OriginalPairSpecialist, PairPersonality, PairCharacteristics, TradingProfile

# Use ultra-fast model as default
PairSpecialist = UltraFastPairSpecialist

__all__ = [
    'PairSpecialist',
    'UltraFastPairSpecialist', 
    'ultra_fast_pair_specialist',
    'analyze_pair_ultra_fast',
    'get_strategy_ultra_fast',
    'get_risk_params_ultra_fast',
    'PairPersonality', 
    'PairCharacteristics', 
    'TradingProfile'
]
