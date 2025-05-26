"""
ML Pipelines Module

This module provides comprehensive machine learning pipelines for real-time
trading applications, including rapid learning, model adaptation, and
continuous improvement capabilities.

Key Components:
- RapidLearningPipeline: Real-time model adaptation and continuous learning
- Model ensemble management and optimization
- Concept drift detection and handling
- Feature engineering and selection
- Performance monitoring and validation

Expected Benefits:
- Real-time model adaptation for changing market conditions
- Continuous learning from live trading data
- Automated model selection and ensemble optimization
- Concept drift detection and mitigation
- Enhanced prediction accuracy through adaptive learning
"""

from .RapidLearningPipeline import (
    RapidLearningPipeline,
    LearningMode,
    DriftStatus,
    ModelStatus,
    ModelPerformance,
    LearningUpdate,
    ModelEnsemble,
    DriftDetector,
    FeatureEngineer
)

from .IndicatorPipeline import (
    IndicatorPipeline,
    IndicatorConfig,
    IndicatorResult,
    IndicatorCategory,
    NormalizationMethod
)

from .DimReductionPipeline import (
    DimReductionPipeline,
    DimReductionMethod,
    DimReductionResult,
    ComponentAnalysis
)

from .AutoencoderPipeline import (
    AutoencoderPipeline,
    AutoencoderConfig,
    AutoencoderResult,
    AnomalyDetection
)

from .SentimentPipeline import (
    SentimentPipeline,
    SentimentConfig,
    SentimentResult,
    SentimentSource
)

from .TrainingPipeline import (
    TrainingPipeline,
    TrainingConfig,
    TrainingResult,
    ModelType,
    ValidationStrategy
)

from .HyperparameterTuner import (
    HyperparameterTuner,
    TuningConfig,
    TuningResult,
    OptimizationMethod,
    SearchSpace
)

from .SHAPReportGenerator import (
    SHAPReportGenerator,
    SHAPConfig,
    SHAPResult,
    ExplanationType,
    FeatureImportance
)

__all__ = [
    # Main pipeline classes
    'RapidLearningPipeline',
    'IndicatorPipeline',
    'DimReductionPipeline',
    'AutoencoderPipeline',
    'SentimentPipeline',
    'TrainingPipeline',
    'HyperparameterTuner',
    'SHAPReportGenerator',

    # Rapid Learning components
    'LearningMode',
    'DriftStatus',
    'ModelStatus',
    'ModelPerformance',
    'LearningUpdate',
    'ModelEnsemble',
    'DriftDetector',
    'FeatureEngineer',

    # Indicator Pipeline components
    'IndicatorConfig',
    'IndicatorResult',
    'IndicatorCategory',
    'NormalizationMethod',

    # Dimensionality Reduction components
    'DimReductionMethod',
    'DimReductionResult',
    'ComponentAnalysis',

    # Autoencoder components
    'AutoencoderConfig',
    'AutoencoderResult',
    'AnomalyDetection',

    # Sentiment components
    'SentimentConfig',
    'SentimentResult',
    'SentimentSource',

    # Training components
    'TrainingConfig',
    'TrainingResult',
    'ModelType',
    'ValidationStrategy',

    # Hyperparameter Tuning components
    'TuningConfig',
    'TuningResult',
    'OptimizationMethod',
    'SearchSpace',

    # SHAP components
    'SHAPConfig',
    'SHAPResult',
    'ExplanationType',
    'FeatureImportance'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Comprehensive ML pipelines for real-time trading applications with adaptive learning capabilities"
