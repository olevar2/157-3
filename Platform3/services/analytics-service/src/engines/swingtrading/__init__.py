"""
Swing Trading Pattern Engine Package
Short-term pattern recognition for 1-5 day maximum trades (H4 focus).

This package provides comprehensive swing trading analysis including:
- Short-term Elliott Wave patterns (max 5 days)
- Quick Fibonacci retracements for H4 reversals
- Session-based support/resistance levels
- Rapid trend line analysis
- Swing high/low detection for entries

Components:
- ShortTermElliottWaves: 3-5 wave structures for quick trades
- QuickFibonacci: Fast retracements for H4 reversals
- SessionSupportResistance: Session-based levels
- RapidTrendlines: Trend line breaks and continuations
- SwingHighLowDetector: Recent swing points for entries

Expected Benefits:
- Quick Elliott wave pattern recognition (max 5-day patterns)
- Fast Fibonacci level calculations for reversals
- Session-based support/resistance levels
- Rapid trend line break signals for swing entries
"""

from .ShortTermElliottWaves import (
    ShortTermElliottWaves,
    WaveType,
    WaveDirection,
    WavePoint,
    ElliottWavePattern,
    WaveAnalysisResult
)

from .QuickFibonacci import (
    QuickFibonacci,
    FibLevel,
    TrendDirection,
    FibonacciLevel,
    FibonacciRetracement,
    FibonacciSignal
)

from .SessionSupportResistance import (
    SessionSupportResistance,
    TradingSession,
    LevelType,
    LevelStrength,
    SRLevel,
    SessionAnalysis,
    SRAnalysisResult
)

# Import placeholder classes for remaining components
# These will be implemented as separate files

class RapidTrendlines:
    """Placeholder for rapid trend line analysis"""
    pass

class SwingHighLowDetector:
    """Placeholder for swing high/low detection"""
    pass

__all__ = [
    # Main engine classes
    'ShortTermElliottWaves',
    'QuickFibonacci',
    'SessionSupportResistance',
    'RapidTrendlines',
    'SwingHighLowDetector',
    
    # Elliott Wave components
    'WaveType',
    'WaveDirection',
    'WavePoint',
    'ElliottWavePattern',
    'WaveAnalysisResult',
    
    # Fibonacci components
    'FibLevel',
    'TrendDirection',
    'FibonacciLevel',
    'FibonacciRetracement',
    'FibonacciSignal',
    
    # Support/Resistance components
    'TradingSession',
    'LevelType',
    'LevelStrength',
    'SRLevel',
    'SessionAnalysis',
    'SRAnalysisResult'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Swing Trading Pattern Engine for H4 focus with 1-5 day maximum patterns"
