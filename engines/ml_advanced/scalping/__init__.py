"""
Scalping AI Models Package
Ultra-fast ML models for M1-M5 scalping signals.

This package provides comprehensive machine learning capabilities for scalping:
- LSTM neural networks for price prediction
- Tick direction classification
- Bid/ask spread forecasting
- Market noise filtering
- Ensemble methods combining all models

Components:
- ScalpingLSTM: LSTM for M1-M5 price prediction
- TickClassifier: Next tick direction prediction
- SpreadPredictor: Bid/ask spread forecasting
- NoiseFilter: ML-based market noise filtering
- ScalpingEnsemble: Ensemble methods for M1-M5
"""

from .ScalpingLSTM import (
    ScalpingLSTM,
    LSTMPrediction,
    LSTMTrainingMetrics,
    ScalpingFeatures
)

from .TickClassifier import (
    TickClassifier,
    TickPrediction,
    TickFeatures,
    ClassifierMetrics
)

from .SpreadPredictor import (
    SpreadPredictor,
    SpreadPrediction,
    SpreadFeatures,
    SpreadAnalysis
)

from .NoiseFilter import (
    NoiseFilter,
    FilteredSignal,
    NoiseAnalysis,
    FilterConfig
)

from .ScalpingEnsemble import (
    ScalpingEnsemble,
    EnsemblePrediction,
    EnsembleWeights,
    EnsembleMetrics
)

__all__ = [
    # Main classes
    'ScalpingLSTM',
    'TickClassifier',
    'SpreadPredictor',
    'NoiseFilter',
    'ScalpingEnsemble',
    
    # LSTM components
    'LSTMPrediction',
    'LSTMTrainingMetrics',
    'ScalpingFeatures',
    
    # Tick classifier components
    'TickPrediction',
    'TickFeatures',
    'ClassifierMetrics',
    
    # Spread predictor components
    'SpreadPrediction',
    'SpreadFeatures',
    'SpreadAnalysis',
    
    # Noise filter components
    'FilteredSignal',
    'NoiseAnalysis',
    'FilterConfig',
    
    # Ensemble components
    'EnsemblePrediction',
    'EnsembleWeights',
    'EnsembleMetrics'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Ultra-fast ML models for M1-M5 scalping signals"
