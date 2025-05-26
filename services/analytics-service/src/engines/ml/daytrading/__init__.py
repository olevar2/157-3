"""
Day Trading AI Models Package
ML models optimized for intraday trading (M15-H1 timeframes).

This package provides comprehensive machine learning capabilities for day trading:
- Intraday momentum prediction for M15-H1 timeframes
- Session-based breakout probability prediction
- Volatility spike early warning systems
- Trend continuation probability scoring
- Ensemble methods combining all day trading models

Components:
- IntradayMomentumML: Momentum prediction for M15-H1
- SessionBreakoutML: Breakout probability prediction
- VolatilityML: Volatility spike prediction
- TrendContinuationML: Intraday trend strength
- DayTradingEnsemble: Ensemble for day trading signals
"""

from .IntradayMomentumML import (
    IntradayMomentumML,
    MomentumPrediction,
    MomentumMetrics,
    MomentumFeatures
)

from .SessionBreakoutML import (
    SessionBreakoutML,
    BreakoutPrediction,
    BreakoutMetrics,
    SessionFeatures
)

from .VolatilityML import (
    VolatilityML,
    VolatilityPrediction,
    VolatilityMetrics,
    VolatilityFeatures
)

from .TrendContinuationML import (
    TrendContinuationML,
    TrendPrediction,
    TrendMetrics,
    TrendFeatures
)

from .DayTradingEnsemble import (
    DayTradingEnsemble,
    EnsemblePrediction,
    EnsembleWeights,
    EnsembleMetrics
)

__all__ = [
    # Main classes
    'IntradayMomentumML',
    'SessionBreakoutML',
    'VolatilityML',
    'TrendContinuationML',
    'DayTradingEnsemble',
    
    # Momentum components
    'MomentumPrediction',
    'MomentumMetrics',
    'MomentumFeatures',
    
    # Breakout components
    'BreakoutPrediction',
    'BreakoutMetrics',
    'SessionFeatures',
    
    # Volatility components
    'VolatilityPrediction',
    'VolatilityMetrics',
    'VolatilityFeatures',
    
    # Trend components
    'TrendPrediction',
    'TrendMetrics',
    'TrendFeatures',
    
    # Ensemble components
    'EnsemblePrediction',
    'EnsembleWeights',
    'EnsembleMetrics'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "ML models optimized for intraday trading (M15-H1)"
