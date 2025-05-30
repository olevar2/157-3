"""
Pattern Master - Advanced Pattern Recognition and Completion

Professional-grade pattern recognition that identifies and predicts completion
of chart patterns, price action patterns, and harmonic patterns for maximum
profit generation to support humanitarian causes.

Version: 2.0.0 (Ultra-Fast) - Achieving <1ms performance targets
"""

from .ultra_fast_model import UltraFastPatternMaster, ultra_fast_pattern_master
from .ultra_fast_model import analyze_patterns_ultra_fast, get_signals_ultra_fast, find_levels_ultra_fast
from .ultra_fast_model import analyze_patterns_with_67_indicators  # Enhanced function
from .model import PatternMaster as OriginalPatternMaster, PatternAnalysis, PatternSignal, PatternType

# Use ultra-fast model as default
PatternMaster = UltraFastPatternMaster

__all__ = [
    'PatternMaster',
    'UltraFastPatternMaster',
    'ultra_fast_pattern_master',
    'analyze_patterns_ultra_fast',
    'get_signals_ultra_fast',
    'find_levels_ultra_fast',
    'PatternAnalysis', 
    'PatternSignal', 
    'PatternType'
]
