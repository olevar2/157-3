"""
Adaptive Learning & Self-Improvement Mechanisms for AI Models

This package provides comprehensive adaptive learning capabilities for trading AI models:
- Real-time performance feedback integration
- Continuous model adaptation based on market changes
- Self-optimizing trading strategies
- Enhanced model accuracy through continuous learning

Expected Benefits:
- Improved model accuracy through continuous learning
- Automatic adaptation to changing market conditions
- Self-optimizing trading strategies
- Real-time performance feedback integration
- Enhanced model robustness and reliability
"""

from .AdaptiveLearner import (
    AdaptiveLearner,
    LearningMode,
    AdaptationTrigger,
    ModelType,
    PerformanceMetrics,
    AdaptationEvent,
    LearningConfiguration,
    MarketRegimeDetector,
    ConceptDriftDetector,
    PerformanceAnalyzer
)

from .PerformanceFeedbackLoop import (
    PerformanceFeedbackLoop,
    FeedbackType,
    FeedbackPriority,
    ModelComponent,
    FeedbackEvent,
    PerformanceSnapshot,
    FeedbackSummary
)

__all__ = [
    # Main classes
    'AdaptiveLearner',
    'PerformanceFeedbackLoop',
    
    # Adaptive learner components
    'LearningMode',
    'AdaptationTrigger',
    'ModelType',
    'PerformanceMetrics',
    'AdaptationEvent',
    'LearningConfiguration',
    'MarketRegimeDetector',
    'ConceptDriftDetector',
    'PerformanceAnalyzer',
    
    # Feedback loop components
    'FeedbackType',
    'FeedbackPriority',
    'ModelComponent',
    'FeedbackEvent',
    'PerformanceSnapshot',
    'FeedbackSummary'
]

__version__ = "1.0.0"
__author__ = "Platform3 AI Team"
__description__ = "Adaptive learning and self-improvement mechanisms for trading AI models"
