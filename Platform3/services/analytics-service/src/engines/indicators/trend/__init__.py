"""
Trend Indicators Module

This module provides comprehensive trend-following indicators for forex trading.
Trend indicators help identify the direction and strength of market trends,
providing crucial information for trend-following strategies.

Key Indicators:
- Moving Averages (SMA, EMA, WMA, DEMA, TEMA): Trend direction and smoothing
- ADX (Average Directional Index): Trend strength measurement
- Ichimoku Cloud: Comprehensive trend analysis system
- Parabolic SAR: Trend reversal detection
- Linear Regression: Mathematical trend analysis

Expected Benefits:
- Comprehensive trend analysis for trading decisions
- Trend direction and strength identification
- Support and resistance level detection
- Trend reversal signal generation
- Multi-timeframe trend confirmation
"""

from .SMA_EMA import (
    MovingAverages,
    MASignal,
    MAResult,
    MAData,
    MAType
)

# Note: Additional trend indicators will be imported as they are implemented
# from .ADX import (...)
# from .Ichimoku import (...)
# from .ParabolicSAR import (...)

__all__ = [
    # Moving Averages
    'MovingAverages',
    'MASignal',
    'MAResult',
    'MAData',
    'MAType',
    
    # Additional indicators will be added here as implemented
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Comprehensive trend indicators for technical analysis and trading signals"
