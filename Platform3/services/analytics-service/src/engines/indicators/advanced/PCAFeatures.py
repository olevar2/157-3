"""
Principal Component Analysis (PCA) Features Module

This module provides advanced PCA-based feature extraction for forex trading,
including dimensionality reduction, feature importance analysis, and market regime detection.
Optimized for scalping (M1-M5), day trading (M15-H1), and swing trading (H4) strategies.

Features:
- Multi-timeframe PCA analysis
- Dynamic feature selection and ranking
- Market regime detection through PCA
- Real-time feature extraction
- Variance explained analysis
- Component interpretation and labeling

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """PCA component interpretation types"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    NOISE = "noise"
    MIXED = "mixed"

@dataclass
class PCAResults:
    """Container for PCA analysis results"""
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    feature_importance: Dict[str, float]
    component_labels: List[ComponentType]
    transformed_features: np.ndarray
    reconstruction_error: float
    n_components_95: int
    market_regime_score: float
    feature_rankings: Dict[str, int]

class PCAFeatures:
    """
    Advanced PCA Features Extraction
    
    Provides sophisticated principal component analysis for forex market data,
    including feature extraction, dimensionality reduction, and market analysis.
    """
    
    def __init__(self, 
                 n_components: Optional[int] = None,
                 variance_threshold: float = 0.95,
                 feature_names: Optional[List[str]] = None,
                 scaling: bool = True):
        """
        Initialize PCA Features analyzer
        
        Args:
            n_components: Number of components to extract (None for auto)
            variance_threshold: Cumulative variance threshold for component selection
            feature_names: Names of input features
            scaling: Whether to apply standard scaling
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaling = scaling
        
        # Default feature names for forex analysis
        self.feature_names = feature_names or [
            'price_change', 'volume', 'volatility', 'rsi', 'macd', 'bb_position',
            'atr', 'momentum', 'trend_strength', 'support_distance', 'resistance_distance',
            'session_volume', 'spread', 'tick_volume', 'price_velocity'
        ]
        
        # Initialize PCA pipeline
        if self.scaling:
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self.n_components))
            ])
        else:
            self.pipeline = Pipeline([
                ('pca', PCA(n_components=self.n_components))
            ])
        
        # Internal state
        self.is_fitted = False
        self.feature_history: List[np.ndarray] = []
        self.component_history: List[np.ndarray] = []
        
        logger.info(f"PCAFeatures initialized with {len(self.feature_names)} features, "
                   f"variance_threshold={variance_threshold}")
    
    def _interpret_components(self, components: np.ndarray) -> List[ComponentType]:
        """
        Interpret PCA components based on feature loadings
        
        Args:
            components: PCA component matrix
            
        Returns:
            List of component type interpretations
        """
        component_labels = []
        
        for i, component in enumerate(components):
            # Get absolute loadings for interpretation
            abs_loadings = np.abs(component)
            max_loading_idx = np.argmax(abs_loadings)
            max_feature = self.feature_names[max_loading_idx] if max_loading_idx < len(self.feature_names) else "unknown"
            
            # Interpret based on dominant feature
            if 'price' in max_feature.lower() or 'trend' in max_feature.lower():
                component_labels.append(ComponentType.TREND)
            elif 'momentum' in max_feature.lower() or 'rsi' in max_feature.lower() or 'macd' in max_feature.lower():
                component_labels.append(ComponentType.MOMENTUM)
            elif 'volatility' in max_feature.lower() or 'atr' in max_feature.lower() or 'bb' in max_feature.lower():
                component_labels.append(ComponentType.VOLATILITY)
            elif 'volume' in max_feature.lower():
                component_labels.append(ComponentType.VOLUME)
            elif abs_loadings.max() < 0.3:  # Low loadings indicate noise
                component_labels.append(ComponentType.NOISE)
            else:
                component_labels.append(ComponentType.MIXED)
        
        return component_labels
    
    def _calculate_feature_importance(self, components: np.ndarray, 
                                    explained_variance: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance based on PCA loadings and explained variance
        
        Args:
            components: PCA component matrix
            explained_variance: Explained variance ratio for each component
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_scores = {}
        
        for i, feature_name in enumerate(self.feature_names):
            if i >= components.shape[1]:
                break
                
            # Calculate weighted importance across all components
            importance = 0.0
            for j, (component, variance) in enumerate(zip(components, explained_variance)):
                if i < len(component):
                    importance += abs(component[i]) * variance
            
            importance_scores[feature_name] = importance
        
        # Normalize importance scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {k: v / total_importance for k, v in importance_scores.items()}
        
        return importance_scores
    
    def _rank_features(self, importance_scores: Dict[str, float]) -> Dict[str, int]:
        """
        Rank features by importance
        
        Args:
            importance_scores: Feature importance scores
            
        Returns:
            Dictionary of feature rankings (1 = most important)
        """
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        return {feature: rank + 1 for rank, (feature, _) in enumerate(sorted_features)}
    
    def _calculate_market_regime_score(self, transformed_features: np.ndarray) -> float:
        """
        Calculate market regime score based on PCA transformation
        
        Args:
            transformed_features: PCA-transformed features
            
        Returns:
            Market regime score (0-1, higher = more volatile/trending)
        """
        if len(transformed_features) == 0:
            return 0.5
        
        # Use first principal component variance as regime indicator
        pc1_variance = np.var(transformed_features[:, 0]) if transformed_features.shape[1] > 0 else 0.0
        
        # Normalize to 0-1 range (using historical context if available)
        if len(self.component_history) > 10:
            historical_variances = [np.var(comp[:, 0]) if comp.shape[1] > 0 else 0.0 
                                  for comp in self.component_history[-10:]]
            max_var = max(historical_variances) if historical_variances else 1.0
            regime_score = min(1.0, pc1_variance / max_var) if max_var > 0 else 0.5
        else:
            regime_score = min(1.0, pc1_variance)
        
        return regime_score
    
    def _calculate_reconstruction_error(self, original_data: np.ndarray, 
                                      transformed_data: np.ndarray,
                                      components: np.ndarray) -> float:
        """
        Calculate reconstruction error for PCA quality assessment
        
        Args:
            original_data: Original feature matrix
            transformed_data: PCA-transformed data
            components: PCA components
            
        Returns:
            Reconstruction error (lower = better)
        """
        try:
            # Reconstruct data from PCA components
            if self.scaling:
                scaler = self.pipeline.named_steps['scaler']
                pca = self.pipeline.named_steps['pca']
                
                # Inverse transform
                reconstructed_scaled = pca.inverse_transform(transformed_data)
                reconstructed = scaler.inverse_transform(reconstructed_scaled)
            else:
                pca = self.pipeline.named_steps['pca']
                reconstructed = pca.inverse_transform(transformed_data)
            
            # Calculate mean squared error
            mse = np.mean((original_data - reconstructed) ** 2)
            return mse
            
        except Exception as e:
            logger.warning(f"Error calculating reconstruction error: {str(e)}")
            return 0.0
    
    def fit(self, features: Union[List[List[float]], np.ndarray]) -> 'PCAFeatures':
        """
        Fit PCA model to feature data
        
        Args:
            features: Feature matrix (samples x features)
            
        Returns:
            Self for method chaining
        """
        try:
            features = np.array(features)
            
            if features.shape[0] < 2:
                logger.warning("Insufficient data for PCA fitting")
                return self
            
            # Ensure we have enough features
            if features.shape[1] < len(self.feature_names):
                logger.warning(f"Feature matrix has {features.shape[1]} columns, "
                             f"expected {len(self.feature_names)}")
                self.feature_names = self.feature_names[:features.shape[1]]
            
            # Fit PCA pipeline
            self.pipeline.fit(features)
            self.is_fitted = True
            
            logger.info(f"PCA fitted with {features.shape[0]} samples, {features.shape[1]} features")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting PCA: {str(e)}")
            raise
    
    def transform(self, features: Union[List[List[float]], np.ndarray]) -> PCAResults:
        """
        Transform features using fitted PCA and analyze results
        
        Args:
            features: Feature matrix to transform
            
        Returns:
            PCAResults object with comprehensive analysis
        """
        try:
            if not self.is_fitted:
                logger.warning("PCA not fitted, fitting with provided data")
                self.fit(features)
            
            features = np.array(features)
            
            if features.shape[0] == 0:
                logger.warning("Empty feature matrix provided")
                return self._create_empty_results()
            
            # Transform features
            transformed_features = self.pipeline.transform(features)
            
            # Get PCA components and explained variance
            pca = self.pipeline.named_steps['pca']
            components = pca.components_
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Find number of components for 95% variance
            n_components_95 = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            
            # Interpret components
            component_labels = self._interpret_components(components)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(components, explained_variance_ratio)
            
            # Rank features
            feature_rankings = self._rank_features(feature_importance)
            
            # Calculate market regime score
            market_regime_score = self._calculate_market_regime_score(transformed_features)
            
            # Calculate reconstruction error
            reconstruction_error = self._calculate_reconstruction_error(
                features, transformed_features, components
            )
            
            # Update history
            self.feature_history.append(features)
            self.component_history.append(transformed_features)
            
            # Maintain history size
            max_history = 100
            if len(self.feature_history) > max_history:
                self.feature_history = self.feature_history[-max_history:]
                self.component_history = self.component_history[-max_history:]
            
            result = PCAResults(
                components=components,
                explained_variance_ratio=explained_variance_ratio,
                cumulative_variance=cumulative_variance,
                feature_importance=feature_importance,
                component_labels=component_labels,
                transformed_features=transformed_features,
                reconstruction_error=reconstruction_error,
                n_components_95=n_components_95,
                market_regime_score=market_regime_score,
                feature_rankings=feature_rankings
            )
            
            logger.info(f"PCA transformation complete: {len(components)} components, "
                       f"{explained_variance_ratio[0]:.3f} variance in PC1")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in PCA transformation: {str(e)}")
            raise
    
    def _create_empty_results(self) -> PCAResults:
        """Create empty PCA results for error cases"""
        return PCAResults(
            components=np.array([]),
            explained_variance_ratio=np.array([]),
            cumulative_variance=np.array([]),
            feature_importance={},
            component_labels=[],
            transformed_features=np.array([]),
            reconstruction_error=0.0,
            n_components_95=0,
            market_regime_score=0.5,
            feature_rankings={}
        )
    
    def get_top_features(self, results: PCAResults, n_features: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N most important features
        
        Args:
            results: PCAResults from transform
            n_features: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        sorted_features = sorted(results.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        return sorted_features[:n_features]
    
    def get_trading_signals(self, results: PCAResults) -> Dict[str, Any]:
        """
        Generate trading signals based on PCA analysis
        
        Args:
            results: PCAResults from transform
            
        Returns:
            Dictionary with trading signals and recommendations
        """
        signals = {
            "market_regime": "trending" if results.market_regime_score > 0.7 else 
                           "ranging" if results.market_regime_score < 0.3 else "mixed",
            "regime_strength": results.market_regime_score,
            "primary_component": results.component_labels[0].value if results.component_labels else "unknown",
            "feature_diversity": len([f for f in results.feature_importance.values() if f > 0.1]),
            "reconstruction_quality": "good" if results.reconstruction_error < 0.1 else "poor",
            "recommended_features": self.get_top_features(results, 3)
        }
        
        # Add component-specific signals
        if results.component_labels:
            primary_component = results.component_labels[0]
            
            if primary_component == ComponentType.TREND:
                signals["strategy_preference"] = "trend_following"
                signals["timeframe_preference"] = "H1-H4"
            elif primary_component == ComponentType.MOMENTUM:
                signals["strategy_preference"] = "momentum"
                signals["timeframe_preference"] = "M15-H1"
            elif primary_component == ComponentType.VOLATILITY:
                signals["strategy_preference"] = "volatility_breakout"
                signals["timeframe_preference"] = "M5-M15"
            elif primary_component == ComponentType.VOLUME:
                signals["strategy_preference"] = "volume_analysis"
                signals["timeframe_preference"] = "M15-H1"
            else:
                signals["strategy_preference"] = "mixed"
                signals["timeframe_preference"] = "M15-H1"
        
        return signals
    
    def analyze_feature_stability(self, window_size: int = 20) -> Dict[str, float]:
        """
        Analyze feature importance stability over time
        
        Args:
            window_size: Number of recent periods to analyze
            
        Returns:
            Dictionary of feature stability scores
        """
        if len(self.feature_history) < window_size:
            return {feature: 0.5 for feature in self.feature_names}
        
        stability_scores = {}
        recent_features = self.feature_history[-window_size:]
        
        # Calculate importance for each period
        importance_history = []
        for features in recent_features:
            if len(features) > 0:
                temp_pca = PCA(n_components=min(5, features.shape[1]))
                temp_pca.fit(features)
                importance = self._calculate_feature_importance(
                    temp_pca.components_, temp_pca.explained_variance_ratio_
                )
                importance_history.append(importance)
        
        # Calculate stability as inverse of variance
        for feature in self.feature_names:
            if feature in importance_history[0] if importance_history else False:
                importances = [imp.get(feature, 0.0) for imp in importance_history]
                stability = 1.0 - np.std(importances) if len(importances) > 1 else 0.5
                stability_scores[feature] = max(0.0, min(1.0, stability))
            else:
                stability_scores[feature] = 0.0
        
        return stability_scores
