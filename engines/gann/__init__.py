"""
Gann Analysis Module Package
Complete Gann analysis toolkit for precise geometric price analysis.

This package provides comprehensive Gann analysis including:
- Gann angle calculations (1x1, 2x1, 3x1, 4x1, 8x1)
- Gann Square of 9 algorithm for price/time predictions
- Dynamic Gann fan analysis for support/resistance
- Time-price cycle detection and forecasting
- Pattern recognition using Gann methods

Components:
- GannAnglesCalculator: 1x1, 2x1, 3x1 angle calculations
- GannSquareOfNine: Price/time predictions using Square of 9
- GannFanAnalysis: Dynamic support/resistance levels
- GannTimePrice: Cycle analysis and time-based predictions
- GannPatternDetector: Pattern recognition using Gann methods

Expected Benefits:
- Precise geometric price analysis
- Time-based cycle predictions  
- Dynamic support/resistance levels
- Mathematical precision in forecasting
"""

from .GannAnglesCalculator import (
    GannAnglesCalculator,
    GannAngle,
    GannFanLevel,
    GannAnglesResult
)

from .GannSquareOfNine import (
    GannSquareOfNine,
    SquareOfNineResult,
    PriceTimeTarget,
    SquareLevel
)

from .GannFanAnalysis import (
    GannFanAnalysis,
    GannFanResult,
    FanLine,
    FanIntersection
)

from .GannTimePrice import (
    GannTimePrice,
    TimePriceResult,
    CycleAnalysis,
    TimeTarget
)

from .GannPatternDetector import (
    GannPatternDetector,
    GannPattern,
    PatternSignal,
    PatternResult
)

__all__ = [
    # Main classes
    'GannAnglesCalculator',
    'GannSquareOfNine', 
    'GannFanAnalysis',
    'GannTimePrice',
    'GannPatternDetector',
    
    # Angle calculator components
    'GannAngle',
    'GannFanLevel',
    'GannAnglesResult',
    
    # Square of Nine components
    'SquareOfNineResult',
    'PriceTimeTarget',
    'SquareLevel',
    
    # Fan analysis components
    'GannFanResult',
    'FanLine',
    'FanIntersection',
    
    # Time-price components
    'TimePriceResult',
    'CycleAnalysis',
    'TimeTarget',
    
    # Pattern detector components
    'GannPattern',
    'PatternSignal',
    'PatternResult'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Complete Gann analysis toolkit for precise geometric price analysis"
