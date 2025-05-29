"""
AI Enhancement Module Initialization

This module provides AI-powered enhancements to traditional technical analysis,
including adaptive indicators, machine learning signal generation, market
microstructure analysis, and intelligent pattern recognition.

Key Components:
- AdaptiveIndicators: Self-adjusting parameter indicators
- MLSignalGenerator: Machine learning signal classification
- MarketMicrostructureAnalysis: Order flow and depth analysis
- SentimentIntegration: News/social sentiment incorporation
- MultiAssetCorrelation: Cross-market relationship analysis
- RegimeDetectionAI: AI-powered market regime identification
- PatternRecognitionAI: Deep learning pattern detection
- RiskAssessmentAI: AI-driven risk measurement
- SignalConfidenceAI: AI confidence scoring for signals

Author: Platform3 Trading System
Version: 1.0.0 - AI Enhancement Suite
"""

from .adaptive_indicators import AdaptiveIndicators
from .ml_signal_generator import MLSignalGenerator
from .market_microstructure_analysis import MarketMicrostructureAnalysis
from .sentiment_integration import SentimentIntegration
from .multi_asset_correlation import MultiAssetCorrelation
from .regime_detection_ai import RegimeDetectionAI
from .pattern_recognition_ai import PatternRecognitionAI
from .risk_assessment_ai import RiskAssessmentAI
from .signal_confidence_ai import SignalConfidenceAI

__all__ = [
    'AdaptiveIndicators',
    'MLSignalGenerator',
    'MarketMicrostructureAnalysis',
    'SentimentIntegration',
    'MultiAssetCorrelation',
    'RegimeDetectionAI',
    'PatternRecognitionAI',
    'RiskAssessmentAI',
    'SignalConfidenceAI'
]
