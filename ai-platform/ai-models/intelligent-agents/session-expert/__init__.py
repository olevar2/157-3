"""
Session Expert - Session-Specific Trading Optimization

Professional-grade session analysis that optimizes trading strategies for different
global market sessions (Asian, London, New York, Sydney) and session overlaps.
Critical for Platform3's 24/5 forex trading optimization.

Version: 2.0.0 (Ultra-Fast) - Achieving <1ms performance targets
Enhanced with Adaptive Indicator Integration
"""

from .ultra_fast_model import UltraFastSessionExpert, ultra_fast_session_expert
from .ultra_fast_model import analyze_session_ultra_fast, get_session_ultra_fast
from .ultra_fast_model import analyze_session_with_indicators  # NEW: Enhanced function
from .model import SessionExpert as OriginalSessionExpert, SessionAnalysis, SessionOptimization, MarketSession

# NEW: Session-specific indicator integration
from .session_indicator_optimizer import SessionIndicatorOptimizer
from .session_adaptive_engine import SessionAdaptiveEngine

class SessionIndicatorOptimizer:
    """Optimizes indicators for specific trading sessions"""
    
    def __init__(self):
        self.session_configs = {
            'ASIAN': {
                'preferred_indicators': ['rsi', 'bollinger_bands', 'support_resistance'],
                'volatility_adjustments': {'conservative': True, 'range_focused': True}
            },
            'LONDON': {
                'preferred_indicators': ['macd', 'ema_cross', 'momentum', 'breakout_indicators'],
                'volatility_adjustments': {'aggressive': True, 'trend_focused': True}
            },
            'NEW_YORK': {
                'preferred_indicators': ['volume_indicators', 'reversal_patterns', 'news_impact'],
                'volatility_adjustments': {'high_frequency': True, 'news_sensitive': True}
            }
        }
    
    async def optimize_for_session(self, session: str, indicators: Dict) -> Dict:
        """Optimize indicator parameters for specific session"""
        session_config = self.session_configs.get(session, {})
        
        optimized_indicators = {}
        for indicator_name, indicator_data in indicators.items():
            if indicator_name in session_config.get('preferred_indicators', []):
                # Apply session-specific optimizations
                optimized_indicators[indicator_name] = await self._apply_session_optimization(
                    indicator_data, session_config
                )
            else:
                optimized_indicators[indicator_name] = indicator_data
        
        return optimized_indicators

# Use ultra-fast model as default with indicator integration
SessionExpert = UltraFastSessionExpert

__all__ = [
    'SessionExpert', 
    'UltraFastSessionExpert',
    'ultra_fast_session_expert',
    'analyze_session_ultra_fast',
    'get_session_ultra_fast', 
    'analyze_session_with_indicators',  # NEW
    'SessionIndicatorOptimizer',        # NEW
    'SessionAdaptiveEngine',           # NEW
    'SessionAnalysis', 
    'SessionOptimization', 
    'MarketSession'
]
