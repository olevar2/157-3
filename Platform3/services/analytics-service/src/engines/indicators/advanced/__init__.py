"""
Advanced Indicators Module

This module provides sophisticated technical indicators for forex trading analysis,
including time-weighted volatility, PCA features, autoencoder features, and sentiment scores.
Optimized for scalping (M1-M5), day trading (M15-H1), and swing trading (H4) strategies.

Features:
- Time-weighted volatility analysis with session weighting
- Principal Component Analysis (PCA) for feature extraction
- Autoencoder-based anomaly detection and feature learning
- Multi-source market sentiment analysis
- Advanced consensus analysis across all indicators
- Real-time signal generation and regime detection

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import logging

# Import all advanced indicators
from .TimeWeightedVolatility import (
    TimeWeightedVolatility, 
    VolatilityMetrics, 
    VolatilityRegime, 
    TradingSession
)
from .PCAFeatures import (
    PCAFeatures, 
    PCAResults, 
    ComponentType
)
from .AutoencoderFeatures import (
    AutoencoderFeatures, 
    AutoencoderResults, 
    AutoencoderType, 
    AnomalyLevel
)
from .SentimentScores import (
    SentimentScores, 
    SentimentResults, 
    SentimentLevel, 
    SentimentSource,
    SentimentData
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedIndicatorSuite:
    """
    Advanced Indicator Suite
    
    Provides comprehensive advanced technical analysis combining multiple
    sophisticated indicators for forex trading strategy development.
    """
    
    def __init__(self, 
                 lookback_periods: int = 50,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize Advanced Indicator Suite
        
        Args:
            lookback_periods: Number of periods for analysis
            feature_names: Names of input features for PCA/Autoencoder
        """
        self.lookback_periods = lookback_periods
        
        # Default feature names for forex analysis
        self.feature_names = feature_names or [
            'price_change', 'volume', 'volatility', 'rsi', 'macd', 'bb_position',
            'atr', 'momentum', 'trend_strength', 'support_distance', 'resistance_distance',
            'session_volume', 'spread', 'tick_volume', 'price_velocity'
        ]
        
        # Initialize indicators
        self.volatility_analyzer = TimeWeightedVolatility(lookback_periods=lookback_periods)
        self.pca_analyzer = PCAFeatures(feature_names=self.feature_names)
        self.autoencoder_analyzer = AutoencoderFeatures(
            input_dim=len(self.feature_names),
            feature_names=self.feature_names
        )
        self.sentiment_analyzer = SentimentScores(lookback_periods=lookback_periods)
        
        logger.info(f"AdvancedIndicatorSuite initialized with {lookback_periods} periods")
    
    def analyze_comprehensive(self, 
                            prices: Union[List[float], np.ndarray],
                            features: Optional[Union[List[List[float]], np.ndarray]] = None,
                            timestamps: Optional[List[datetime]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive advanced indicator analysis
        
        Args:
            prices: Price data for volatility analysis
            features: Feature matrix for PCA/Autoencoder analysis
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary with all advanced indicator results
        """
        try:
            results = {}
            
            # Volatility Analysis
            try:
                volatility_results = self.volatility_analyzer.analyze(prices, timestamps)
                results['volatility'] = {
                    'current_volatility': volatility_results.current_volatility,
                    'weighted_volatility': volatility_results.weighted_volatility,
                    'session_volatility': volatility_results.session_volatility,
                    'regime': volatility_results.regime.value,
                    'session': volatility_results.session.value,
                    'forecast': volatility_results.forecast,
                    'confidence': volatility_results.confidence,
                    'risk_adjustment': volatility_results.risk_adjustment,
                    'percentile_rank': volatility_results.percentile_rank,
                    'z_score': volatility_results.z_score
                }
                
                # Get trading recommendations
                vol_recommendations = self.volatility_analyzer.get_trading_recommendations(volatility_results)
                results['volatility']['recommendations'] = vol_recommendations
                
            except Exception as e:
                logger.error(f"Error in volatility analysis: {str(e)}")
                results['volatility'] = {'error': str(e)}
            
            # PCA Analysis (if features provided)
            if features is not None:
                try:
                    pca_results = self.pca_analyzer.transform(features)
                    results['pca'] = {
                        'explained_variance_ratio': pca_results.explained_variance_ratio.tolist(),
                        'cumulative_variance': pca_results.cumulative_variance.tolist(),
                        'feature_importance': pca_results.feature_importance,
                        'component_labels': [label.value for label in pca_results.component_labels],
                        'reconstruction_error': pca_results.reconstruction_error,
                        'n_components_95': pca_results.n_components_95,
                        'market_regime_score': pca_results.market_regime_score,
                        'feature_rankings': pca_results.feature_rankings
                    }
                    
                    # Get trading signals
                    pca_signals = self.pca_analyzer.get_trading_signals(pca_results)
                    results['pca']['signals'] = pca_signals
                    
                    # Get top features
                    top_features = self.pca_analyzer.get_top_features(pca_results, 5)
                    results['pca']['top_features'] = top_features
                    
                except Exception as e:
                    logger.error(f"Error in PCA analysis: {str(e)}")
                    results['pca'] = {'error': str(e)}
            
            # Autoencoder Analysis (if features provided)
            if features is not None:
                try:
                    autoencoder_results = self.autoencoder_analyzer.transform(features)
                    results['autoencoder'] = {
                        'reconstruction_error': autoencoder_results.reconstruction_error,
                        'anomaly_score': autoencoder_results.anomaly_score,
                        'anomaly_level': autoencoder_results.anomaly_level.value,
                        'feature_importance': autoencoder_results.feature_importance,
                        'compression_ratio': autoencoder_results.compression_ratio,
                        'model_confidence': autoencoder_results.model_confidence,
                        'regime_classification': autoencoder_results.regime_classification
                    }
                    
                    # Get trading signals
                    autoencoder_signals = self.autoencoder_analyzer.get_trading_signals(autoencoder_results)
                    results['autoencoder']['signals'] = autoencoder_signals
                    
                except Exception as e:
                    logger.error(f"Error in autoencoder analysis: {str(e)}")
                    results['autoencoder'] = {'error': str(e)}
            
            # Sentiment Analysis
            try:
                sentiment_results = self.sentiment_analyzer.calculate_sentiment_scores()
                results['sentiment'] = {
                    'overall_sentiment': sentiment_results.overall_sentiment,
                    'sentiment_level': sentiment_results.sentiment_level.value,
                    'source_breakdown': sentiment_results.source_breakdown,
                    'momentum': sentiment_results.momentum,
                    'volatility': sentiment_results.volatility,
                    'confidence': sentiment_results.confidence,
                    'regime': sentiment_results.regime,
                    'session_sentiment': sentiment_results.session_sentiment,
                    'correlation_with_price': sentiment_results.correlation_with_price,
                    'sentiment_divergence': sentiment_results.sentiment_divergence
                }
                
                # Get trading signals
                sentiment_signals = self.sentiment_analyzer.get_trading_signals(sentiment_results)
                results['sentiment']['signals'] = sentiment_signals
                
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {str(e)}")
                results['sentiment'] = {'error': str(e)}
            
            # Generate consensus analysis
            consensus = self._generate_consensus_analysis(results)
            results['consensus'] = consensus
            
            logger.info("Comprehensive advanced indicator analysis completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    def _generate_consensus_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate consensus analysis across all advanced indicators
        
        Args:
            results: Dictionary with all indicator results
            
        Returns:
            Dictionary with consensus analysis
        """
        consensus = {
            'overall_signal': 'neutral',
            'confidence': 0.0,
            'risk_level': 'normal',
            'strategy_preference': 'mixed',
            'timeframe_preference': 'M15-H1',
            'regime_classification': 'normal',
            'anomaly_detected': False,
            'trading_recommendations': []
        }
        
        try:
            signals = []
            confidences = []
            
            # Collect signals from each indicator
            if 'volatility' in results and 'recommendations' in results['volatility']:
                vol_rec = results['volatility']['recommendations']
                if 'trading_advice' in vol_rec:
                    signals.append(vol_rec['trading_advice'])
                    confidences.append(results['volatility'].get('confidence', 0.5))
            
            if 'pca' in results and 'signals' in results['pca']:
                pca_signals = results['pca']['signals']
                if 'market_regime' in pca_signals:
                    signals.append(pca_signals['market_regime'])
                    confidences.append(0.7)  # Default confidence for PCA
            
            if 'autoencoder' in results and 'signals' in results['autoencoder']:
                ae_signals = results['autoencoder']['signals']
                if 'trading_action' in ae_signals:
                    signals.append(ae_signals['trading_action'])
                    confidences.append(results['autoencoder'].get('model_confidence', 0.5))
                
                # Check for anomalies
                if ae_signals.get('anomaly_detected', False):
                    consensus['anomaly_detected'] = True
            
            if 'sentiment' in results and 'signals' in results['sentiment']:
                sent_signals = results['sentiment']['signals']
                if 'sentiment_direction' in sent_signals:
                    signals.append(sent_signals['sentiment_direction'])
                    confidences.append(results['sentiment'].get('confidence', 0.5))
            
            # Calculate consensus
            if signals and confidences:
                # Simple majority voting with confidence weighting
                signal_weights = {}
                for signal, confidence in zip(signals, confidences):
                    if signal not in signal_weights:
                        signal_weights[signal] = 0.0
                    signal_weights[signal] += confidence
                
                # Find dominant signal
                if signal_weights:
                    dominant_signal = max(signal_weights.items(), key=lambda x: x[1])
                    consensus['overall_signal'] = dominant_signal[0]
                    consensus['confidence'] = min(1.0, dominant_signal[1] / len(signals))
            
            # Determine risk level
            risk_indicators = []
            
            if 'volatility' in results:
                vol_regime = results['volatility'].get('regime', 'normal')
                if vol_regime in ['high', 'extreme']:
                    risk_indicators.append('high_volatility')
            
            if 'autoencoder' in results:
                anomaly_level = results['autoencoder'].get('anomaly_level', 'normal')
                if anomaly_level in ['severe', 'extreme']:
                    risk_indicators.append('anomaly_detected')
            
            if 'sentiment' in results:
                sent_volatility = results['sentiment'].get('volatility', 0.0)
                if sent_volatility > 0.5:
                    risk_indicators.append('sentiment_volatility')
            
            # Set risk level based on indicators
            if len(risk_indicators) >= 2:
                consensus['risk_level'] = 'high'
            elif len(risk_indicators) == 1:
                consensus['risk_level'] = 'elevated'
            else:
                consensus['risk_level'] = 'normal'
            
            # Generate trading recommendations
            recommendations = []
            
            if consensus['anomaly_detected']:
                recommendations.append("Monitor for unusual market conditions")
            
            if consensus['risk_level'] == 'high':
                recommendations.append("Reduce position sizes and tighten stops")
            
            if consensus['confidence'] > 0.7:
                recommendations.append(f"High confidence {consensus['overall_signal']} signal")
            elif consensus['confidence'] < 0.3:
                recommendations.append("Low confidence - consider waiting for clearer signals")
            
            consensus['trading_recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error generating consensus analysis: {str(e)}")
        
        return consensus
    
    def add_sentiment_data(self, 
                          source: SentimentSource,
                          score: float,
                          timestamp: Optional[datetime] = None,
                          confidence: float = 1.0,
                          text: Optional[str] = None) -> None:
        """
        Add sentiment data to the sentiment analyzer
        
        Args:
            source: Sentiment data source
            score: Sentiment score
            timestamp: Data timestamp
            confidence: Confidence in the score
            text: Optional text for analysis
        """
        self.sentiment_analyzer.add_sentiment_data(
            source=source,
            raw_score=score,
            timestamp=timestamp,
            confidence=confidence,
            text=text
        )
    
    def add_price_data(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Add price data for sentiment correlation analysis
        
        Args:
            price: Price value
            timestamp: Data timestamp
        """
        self.sentiment_analyzer.add_price_data(price, timestamp)

# Export all classes and enums
__all__ = [
    'AdvancedIndicatorSuite',
    'TimeWeightedVolatility',
    'VolatilityMetrics',
    'VolatilityRegime',
    'TradingSession',
    'PCAFeatures',
    'PCAResults',
    'ComponentType',
    'AutoencoderFeatures',
    'AutoencoderResults',
    'AutoencoderType',
    'AnomalyLevel',
    'SentimentScores',
    'SentimentResults',
    'SentimentLevel',
    'SentimentSource',
    'SentimentData'
]
