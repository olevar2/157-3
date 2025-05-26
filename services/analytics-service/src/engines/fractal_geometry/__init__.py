"""
Fractal Geometry Analysis Module

This module provides advanced fractal geometry analysis for forex trading,
including fractal dimension calculation, Hurst exponent analysis, and
geometric pattern recognition for market structure analysis.

Key Components:
- FractalGeometryIndicator: Main analysis engine
- Fractal pattern identification (Williams, Custom, Geometric)
- Market structure analysis using fractal mathematics
- Trend persistence analysis through Hurst exponent
- Geometric ratio-based pattern recognition

Expected Benefits:
- Advanced fractal pattern recognition for market structure analysis
- Geometric price analysis using fractal dimensions
- Enhanced pattern detection through fractal mathematics
- Improved market timing through fractal geometry insights
"""

from .FractalGeometryIndicator import (
    FractalGeometryIndicator,
    FractalType,
    TrendPersistence,
    FractalPoint,
    FractalDimension,
    HurstAnalysis,
    GeometricPattern
)

__all__ = [
    # Main indicator class
    'FractalGeometryIndicator',
    
    # Enums
    'FractalType',
    'TrendPersistence',
    
    # Data classes
    'FractalPoint',
    'FractalDimension', 
    'HurstAnalysis',
    'GeometricPattern'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Advanced fractal geometry analysis for market structure and pattern recognition"
