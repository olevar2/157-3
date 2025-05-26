"""
Swing Trading Intelligence Package
ML models for short-term swing patterns (H4 focus) with max 3-5 day holding periods.

This package provides comprehensive ML-based swing trading intelligence including:
- Short-term swing pattern recognition (1-5 days)
- Quick reversal detection and prediction
- Swing momentum analysis and forecasting
- Multi-timeframe confluence analysis (M15-H4)
- Ensemble methods combining all models for robust signals

Components:
- ShortSwingPatterns: 1-5 day pattern recognition
- QuickReversalML: Rapid reversal detection
- SwingMomentumML: Swing momentum prediction
- MultiTimeframeML: M15-H4 confluence analysis
- SwingEnsemble: Ensemble for swing signals

Expected Benefits:
- Short-term swing pattern detection (max 5 days)
- Quick reversal signal generation
- Multi-timeframe confluence validation
- Optimized entry/exit timing for swing trades
"""

from .ShortSwingPatterns import (
    ShortSwingPatterns,
    SwingPatternPrediction,
    SwingPatternMetrics,
    SwingPatternFeatures
)

from .QuickReversalML import (
    QuickReversalML,
    ReversalPrediction,
    ReversalMetrics,
    ReversalFeatures
)

from .SwingMomentumML import (
    SwingMomentumML,
    MomentumPrediction,
    MomentumMetrics,
    MomentumFeatures
)

from .MultiTimeframeML import (
    MultiTimeframeML,
    ConfluencePrediction,
    ConfluenceMetrics,
    ConfluenceFeatures
)

from .SwingEnsemble import (
    SwingEnsemble,
    EnsemblePrediction,
    EnsembleWeights,
    EnsembleMetrics
)

__all__ = [
    # Main model classes
    'ShortSwingPatterns',
    'QuickReversalML', 
    'SwingMomentumML',
    'MultiTimeframeML',
    'SwingEnsemble',
    
    # Prediction classes
    'SwingPatternPrediction',
    'ReversalPrediction',
    'MomentumPrediction',
    'ConfluencePrediction',
    'EnsemblePrediction',
    
    # Metrics classes
    'SwingPatternMetrics',
    'ReversalMetrics',
    'MomentumMetrics',
    'ConfluenceMetrics',
    'EnsembleMetrics',
    
    # Feature classes
    'SwingPatternFeatures',
    'ReversalFeatures',
    'MomentumFeatures',
    'ConfluenceFeatures',
    
    # Utility classes
    'EnsembleWeights'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "ML models for swing trading intelligence with H4 focus and 1-5 day patterns"

# Model configuration defaults
DEFAULT_CONFIG = {
    'pattern_config': {
        'sequence_length': 96,  # 96 H4 periods (16 days)
        'prediction_horizon': 120,  # 5 days in hours
        'feature_count': 30
    },
    'reversal_config': {
        'sequence_length': 48,  # 48 H4 periods (8 days)
        'prediction_horizon': 24,  # 24 hours ahead
        'feature_count': 25
    },
    'momentum_config': {
        'sequence_length': 72,  # 72 H4 periods (12 days)
        'prediction_horizon': 48,  # 48 hours ahead
        'feature_count': 28
    },
    'confluence_config': {
        'timeframes': ['M15', 'M30', 'H1', 'H4'],
        'sequence_lengths': {
            'M15': 96,   # 24 hours
            'M30': 48,   # 24 hours
            'H1': 24,    # 24 hours
            'H4': 12     # 48 hours
        },
        'feature_count_per_tf': 20
    }
}

def create_swing_ensemble(config=None):
    """
    Factory function to create a configured SwingEnsemble instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        SwingEnsemble: Configured ensemble instance
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    return SwingEnsemble(config)

def get_model_info():
    """
    Get information about all swing trading models
    
    Returns:
        dict: Model information and capabilities
    """
    return {
        'package_version': __version__,
        'models': {
            'ShortSwingPatterns': {
                'description': '1-5 day pattern recognition optimized for H4 timeframes',
                'focus': 'Short-term swing patterns',
                'timeframe': 'H4 primary',
                'duration': '1-5 days',
                'features': ['price_patterns', 'volume_patterns', 'volatility_patterns', 'momentum_patterns', 'support_resistance']
            },
            'QuickReversalML': {
                'description': 'Rapid reversal detection for swing trading',
                'focus': 'Reversal point identification',
                'timeframe': 'H4 primary',
                'duration': '4-48 hours',
                'features': ['price_action', 'momentum', 'volume', 'volatility', 'divergence']
            },
            'SwingMomentumML': {
                'description': 'Swing momentum prediction and analysis',
                'focus': 'Momentum strength and direction',
                'timeframe': 'H4 primary',
                'duration': '4-72 hours',
                'features': ['price_momentum', 'volume_momentum', 'volatility_momentum', 'technical_momentum', 'market_structure']
            },
            'MultiTimeframeML': {
                'description': 'M15-H4 confluence analysis for high-probability setups',
                'focus': 'Multi-timeframe alignment',
                'timeframes': ['M15', 'M30', 'H1', 'H4'],
                'duration': '24-72 hours',
                'features': ['cross_timeframe_analysis', 'confluence_detection', 'alignment_scoring']
            },
            'SwingEnsemble': {
                'description': 'Ensemble combining all swing models for robust signals',
                'focus': 'Comprehensive swing analysis',
                'timeframe': 'H4 primary with multi-TF support',
                'duration': '1-5 days',
                'features': ['ensemble_signals', 'consensus_analysis', 'risk_assessment', 'trade_parameters']
            }
        },
        'expected_benefits': [
            'Short-term swing pattern detection (max 5 days)',
            'Quick reversal signal generation',
            'Multi-timeframe confluence validation', 
            'Optimized entry/exit timing for swing trades',
            'Risk-adjusted position sizing',
            'High-probability setup identification'
        ]
    }
