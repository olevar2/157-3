"""
Session Expert - Session-Specific Trading Optimization

Professional-grade session analysis that optimizes trading strategies for different
global market sessions (Asian, London, New York, Sydney) and session overlaps.
Critical for Platform3's 24/5 forex trading optimization.

Version: 2.0.0 (Ultra-Fast) - Achieving <1ms performance targets
"""

from .ultra_fast_model import UltraFastSessionExpert, ultra_fast_session_expert
from .ultra_fast_model import analyze_session_ultra_fast, get_session_ultra_fast
from .ultra_fast_model import analyze_session_with_67_indicators  # Enhanced function
from .model import SessionExpert as OriginalSessionExpert, SessionAnalysis, SessionOptimization, MarketSession

# Use ultra-fast model as default
SessionExpert = UltraFastSessionExpert

__all__ = [
    'SessionExpert', 
    'UltraFastSessionExpert',
    'ultra_fast_session_expert',
    'analyze_session_ultra_fast',
    'get_session_ultra_fast', 
    'get_strategies_ultra_fast',
    'SessionAnalysis', 
    'SessionOptimization', 
    'MarketSession'
]
