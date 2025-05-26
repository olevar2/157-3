"""
Comprehensive Technical Indicators Suite

This module provides a complete suite of technical indicators organized by category
for comprehensive market analysis and trading signal generation. The indicators
are designed for high-frequency forex trading with optimized performance and
accurate signal detection.

Indicator Categories:
- Momentum: RSI, MACD, Stochastic, Williams %R
- Trend: SMA, EMA, ADX, Ichimoku
- Volatility: Bollinger Bands, ATR, Keltner Channels
- Volume: OBV, MFI, Volume Profile
- Cycle: Alligator, Hurst Exponent
- Advanced: PCA Features, Autoencoder Features

Expected Benefits:
- Complete technical analysis suite with all major indicators
- Organized indicator categories for efficient computation
- Feature Store integration for centralized indicator outputs
- Enhanced trading signal generation through comprehensive analysis
- Optimized performance for real-time trading applications
"""

# Import momentum indicators
from .momentum import (
    RSI, RSISignal, RSIResult,
    MACD, MACDSignal, MACDResult, MACDData,
    Stochastic, StochasticSignal, StochasticResult, StochasticData
)

# Import trend indicators
from .trend import (
    MovingAverages, MASignal, MAResult, MAData, MAType
)

# Note: Additional indicator imports will be added as they are implemented
# from .volatility import (...)
# from .volume import (...)
# from .cycle import (...)
# from .advanced import (...)

__all__ = [
    # Momentum indicators
    'RSI', 'RSISignal', 'RSIResult',
    'MACD', 'MACDSignal', 'MACDResult', 'MACDData',
    'Stochastic', 'StochasticSignal', 'StochasticResult', 'StochasticData',
    
    # Trend indicators
    'MovingAverages', 'MASignal', 'MAResult', 'MAData', 'MAType',
    
    # Additional indicators will be added here as implemented
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Comprehensive technical indicators suite for forex trading analysis"

# Indicator registry for dynamic access
INDICATOR_REGISTRY = {
    'momentum': {
        'RSI': RSI,
        'MACD': MACD,
        'Stochastic': Stochastic
    },
    'trend': {
        'MovingAverages': MovingAverages
    }
    # Additional categories will be added as implemented
}

def get_indicator(category: str, indicator_name: str):
    """
    Get indicator class by category and name
    
    Args:
        category: Indicator category (momentum, trend, volatility, etc.)
        indicator_name: Name of the indicator
        
    Returns:
        Indicator class or None if not found
    """
    try:
        return INDICATOR_REGISTRY.get(category, {}).get(indicator_name)
    except Exception:
        return None

def list_indicators(category: str = None) -> dict:
    """
    List available indicators by category
    
    Args:
        category: Optional category filter
        
    Returns:
        Dictionary of available indicators
    """
    if category:
        return INDICATOR_REGISTRY.get(category, {})
    return INDICATOR_REGISTRY

def get_all_momentum_indicators():
    """Get all momentum indicators"""
    return INDICATOR_REGISTRY.get('momentum', {})

def get_all_trend_indicators():
    """Get all trend indicators"""
    return INDICATOR_REGISTRY.get('trend', {})

# Utility functions for indicator analysis
def calculate_indicator_consensus(indicator_results: dict) -> dict:
    """
    Calculate consensus from multiple indicator results
    
    Args:
        indicator_results: Dictionary of indicator analysis results
        
    Returns:
        Dictionary with consensus analysis
    """
    try:
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        total_strength = 0.0
        signal_count = 0
        
        for result in indicator_results.values():
            if isinstance(result, dict):
                signal = result.get('primary_signal', 'neutral')
                strength = result.get('signal_strength', 0.0)
                
                if 'bullish' in signal or 'buy' in signal:
                    bullish_signals += 1
                elif 'bearish' in signal or 'sell' in signal:
                    bearish_signals += 1
                else:
                    neutral_signals += 1
                
                total_strength += strength
                signal_count += 1
        
        if signal_count == 0:
            return {'consensus': 'neutral', 'confidence': 0.0, 'strength': 0.0}
        
        avg_strength = total_strength / signal_count
        total_signals = bullish_signals + bearish_signals + neutral_signals
        
        if bullish_signals > bearish_signals:
            consensus = 'bullish'
            confidence = bullish_signals / total_signals
        elif bearish_signals > bullish_signals:
            consensus = 'bearish'
            confidence = bearish_signals / total_signals
        else:
            consensus = 'neutral'
            confidence = neutral_signals / total_signals
        
        return {
            'consensus': consensus,
            'confidence': confidence,
            'strength': avg_strength,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'neutral_signals': neutral_signals,
            'total_signals': total_signals
        }
        
    except Exception:
        return {'consensus': 'neutral', 'confidence': 0.0, 'strength': 0.0}

def generate_trading_recommendations(consensus_analysis: dict) -> list:
    """
    Generate trading recommendations based on consensus analysis
    
    Args:
        consensus_analysis: Result from calculate_indicator_consensus
        
    Returns:
        List of trading recommendations
    """
    try:
        recommendations = []
        consensus = consensus_analysis.get('consensus', 'neutral')
        confidence = consensus_analysis.get('confidence', 0.0)
        strength = consensus_analysis.get('strength', 0.0)
        
        if consensus == 'bullish' and confidence > 0.7 and strength > 0.6:
            recommendations.append("Strong buy signal - High confidence bullish consensus")
        elif consensus == 'bearish' and confidence > 0.7 and strength > 0.6:
            recommendations.append("Strong sell signal - High confidence bearish consensus")
        elif consensus == 'bullish' and confidence > 0.5:
            recommendations.append("Moderate buy signal - Bullish consensus detected")
        elif consensus == 'bearish' and confidence > 0.5:
            recommendations.append("Moderate sell signal - Bearish consensus detected")
        elif confidence < 0.4:
            recommendations.append("Mixed signals - Wait for clearer market direction")
        else:
            recommendations.append("Neutral market conditions - Monitor for signal development")
        
        # Additional recommendations based on strength
        if strength > 0.8:
            recommendations.append("Very strong signal strength - High probability setup")
        elif strength < 0.3:
            recommendations.append("Weak signal strength - Consider waiting for confirmation")
        
        return recommendations
        
    except Exception:
        return ["Unable to generate recommendations"]
