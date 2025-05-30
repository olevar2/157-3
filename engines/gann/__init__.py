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

from .GannPatternDetector import (
    GannPatternDetector,
    GannPattern,
    PatternSignal,
    PatternResult
)

from .gann_square_of_nine import (
    GannSquareOfNine,
    SquareAnalysis,
    SquarePoint
)

from .gann_fan_lines import (
    GannFanLines,
    GannAngle,
    GannLine,
    GannFanAnalysis
)

from .gann_time_cycles import (
    GannTimeCycles,
    TimeCycle,
    GannTimeSignal
)

from .price_time_relationships import (
    PriceTimeRelationships,
    PriceTimeRelationship,
    GannSquareLevel
)

__all__ = [
    # Main classes
    'GannPatternDetector',
    'GannSquareOfNine', 
    'GannFanLines',
    'GannTimeCycles',
    'PriceTimeRelationships',
    
    # Pattern components
    'GannPattern',
    'PatternSignal',
    'PatternResult',
      # Square of Nine components
    'SquareAnalysis',
    'SquarePoint',
    
    # Fan analysis components
    'GannAngle',
    'GannLine',
    'GannFanAnalysis',
    
    # Time cycle components
    'TimeCycle',
    'GannTimeSignal',
    
    # Price-time components
    'PriceTimeRelationship',
    'GannSquareLevel'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Complete Gann analysis toolkit for precise geometric price analysis"
