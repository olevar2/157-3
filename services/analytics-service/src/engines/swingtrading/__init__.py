"""
Swing Trading Pattern Engine Package
Short-term pattern recognition for 1-5 day maximum trades (H4 focus)

This package provides comprehensive swing trading analysis including:
- Elliott wave pattern recognition (3-5 wave structures)
- Fibonacci retracement calculations for reversals
- Session-based support/resistance levels
- Rapid trend line analysis and breakouts
- Swing high/low detection for entry timing

All modules are optimized for H4 timeframe with maximum 3-5 day trade duration.
"""

from .ShortTermElliottWaves import ShortTermElliottWaves, ShortTermElliottResult
from .QuickFibonacci import QuickFibonacci, QuickFibonacciResult
from .SessionSupportResistance import SessionSupportResistance, SessionSupportResistanceResult
from .RapidTrendlines import RapidTrendlines, RapidTrendlinesResult
from .SwingHighLowDetector import SwingHighLowDetector, SwingHighLowResult

__all__ = [
    'ShortTermElliottWaves',
    'ShortTermElliottResult',
    'QuickFibonacci', 
    'QuickFibonacciResult',
    'SessionSupportResistance',
    'SessionSupportResistanceResult',
    'RapidTrendlines',
    'RapidTrendlinesResult',
    'SwingHighLowDetector',
    'SwingHighLowResult'
]

__version__ = '1.0.0'
__author__ = 'Platform3 Analytics Team'
__description__ = 'Swing Trading Pattern Engine for H4 Focus (Max 3-5 Days)'
