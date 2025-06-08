"""
ML Signal Generator for Platform3
Advanced machine learning signal generation with multiple model ensemble
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of ML signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class MLSignal:
    """ML generated trading signal"""
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    strength: float    # 0.0 to 1.0
    timestamp: datetime
    features_used: List[str]
    model_ensemble: Dict[str, float]  # model_name -> confidence
    metadata: Dict[str, Any]


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: str
    parameters: Dict[str, Any]
    weight: float = 1.0
    enabled: bool = True


class MLSignalGenerator:
    """
    Advanced ML Signal Generator
    
    Generates trading signals using ensemble of machine learning models:
    - Random Forest
    - Gradient Boosting
    - Neural Networks
    - Support Vector Machines
    - LSTM for time series
    """
    
    def __init__(self, models_config: Optional[List[ModelConfig]] = None):
        self.models_config = models_config or self._get_default_models()
        self.models = {}
        self.feature_importance = {}
        self.signal_history = []
        
        logger.info("MLSignalGenerator initialized")
    
    def _get_default_models(self) -> List[ModelConfig]:
        """Get default model configurations"""
        return [
            ModelConfig("random_forest", {"n_estimators": 100, "max_depth": 10}, 1.0),
            ModelConfig("gradient_boost", {"n_estimators": 100, "learning_rate": 0.1}, 1.2),
            ModelConfig("neural_network", {"hidden_layers": [64, 32], "epochs": 100}, 0.9),
            ModelConfig("svm", {"kernel": "rbf", "C": 1.0}, 0.8),
            ModelConfig("lstm", {"units": 50, "sequence_length": 20}, 1.1)
        ]
    
    def initialize_models(self):
        """Initialize all ML models"""
        for config in self.models_config:
            if config.enabled:
                try:
                    self.models[config.model_type] = self._create_model(config)
                    logger.info(f"Model {config.model_type} initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize model {config.model_type}: {e}")
    
    def _create_model(self, config: ModelConfig):
        """Create a specific ML model"""
        # Placeholder for actual model creation
        # In real implementation, this would create sklearn/tensorflow models
        return {
            "config": config,
            "trained": False,
            "last_update": datetime.now()
        }
    
    def generate_signal(
        self, 
        market_data: pd.DataFrame,
        features: Dict[str, np.ndarray],
        context: Dict[str, Any] = None
    ) -> MLSignal:
        """
        Generate ML trading signal from market data and features
        
        Args:
            market_data: Historical market data
            features: Computed technical indicators and features
            context: Additional context information
            
        Returns:
            MLSignal with prediction and confidence
        """
        try:
            # Extract features for ML models
            feature_matrix = self._prepare_features(market_data, features)
            
            # Get predictions from ensemble
            ensemble_predictions = self._get_ensemble_predictions(feature_matrix)
            
            # Combine predictions
            final_signal = self._combine_predictions(ensemble_predictions)
            
            # Create signal object
            signal = MLSignal(
                signal_type=final_signal["signal_type"],
                confidence=final_signal["confidence"],
                strength=final_signal["strength"],
                timestamp=datetime.now(),
                features_used=list(features.keys()),
                model_ensemble=ensemble_predictions,
                metadata={
                    "context": context or {},
                    "feature_importance": self.feature_importance,
                    "model_count": len(self.models)
                }
            )
            
            self.signal_history.append(signal)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            # Return neutral signal on error
            return MLSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strength=0.0,
                timestamp=datetime.now(),
                features_used=[],
                model_ensemble={},
                metadata={"error": str(e)}
            )
    
    def _prepare_features(self, market_data: pd.DataFrame, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Prepare features for ML models"""
        try:
            # Combine all features into matrix
            feature_arrays = []
            
            # Add price-based features
            if len(market_data) > 0:
                close_prices = market_data['close'].values[-20:]  # Last 20 periods
                returns = np.diff(close_prices) / close_prices[:-1]
                feature_arrays.extend([
                    close_prices[-1:],  # Current price
                    [np.mean(returns)],  # Average return
                    [np.std(returns)],   # Volatility
                ])
            
            # Add technical indicator features
            for name, values in features.items():
                if len(values) > 0:
                    feature_arrays.append([values[-1]])  # Latest value
            
            # Combine into feature matrix
            if feature_arrays:
                return np.concatenate(feature_arrays).reshape(1, -1)
            else:
                return np.array([[0.0]])  # Fallback
                
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([[0.0]])
    
    def _get_ensemble_predictions(self, feature_matrix: np.ndarray) -> Dict[str, float]:
        """Get predictions from all models in ensemble"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Placeholder prediction logic
                # In real implementation, would use actual model.predict()
                confidence = np.random.uniform(0.3, 0.9)  # Mock confidence
                predictions[model_name] = confidence
                
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {e}")
                predictions[model_name] = 0.5  # Neutral prediction
        
        return predictions
    
    def _combine_predictions(self, ensemble_predictions: Dict[str, float]) -> Dict[str, Any]:
        """Combine ensemble predictions into final signal"""
        if not ensemble_predictions:
            return {
                "signal_type": SignalType.HOLD,
                "confidence": 0.0,
                "strength": 0.0
            }
        
        # Weight predictions by model configuration
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, prediction in ensemble_predictions.items():
            config = next((c for c in self.models_config if c.model_type == model_name), None)
            weight = config.weight if config else 1.0
            
            weighted_sum += prediction * weight
            total_weight += weight
        
        # Calculate final confidence
        final_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Determine signal type based on confidence
        if final_confidence > 0.7:
            signal_type = SignalType.STRONG_BUY
            strength = min(final_confidence, 1.0)
        elif final_confidence > 0.6:
            signal_type = SignalType.BUY
            strength = final_confidence
        elif final_confidence < 0.3:
            signal_type = SignalType.STRONG_SELL
            strength = 1.0 - final_confidence
        elif final_confidence < 0.4:
            signal_type = SignalType.SELL
            strength = 1.0 - final_confidence
        else:
            signal_type = SignalType.HOLD
            strength = 1.0 - abs(0.5 - final_confidence)
        
        return {
            "signal_type": signal_type,
            "confidence": final_confidence,
            "strength": strength
        }
    
    def update_models(self, training_data: pd.DataFrame, labels: np.ndarray):
        """Update/retrain models with new data"""
        try:
            logger.info("Updating ML models with new training data")
            
            # Prepare training features
            features = self._extract_training_features(training_data)
            
            # Update each model
            for model_name, model in self.models.items():
                try:
                    # Placeholder for actual model training
                    # In real implementation: model.fit(features, labels)
                    model["trained"] = True
                    model["last_update"] = datetime.now()
                    
                    logger.info(f"Model {model_name} updated successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to update model {model_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
    
    def _extract_training_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for model training"""
        # Placeholder for feature extraction
        # Would implement proper feature engineering here
        return np.random.random((len(data), 10))  # Mock features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance across all models"""
        return self.feature_importance.copy()
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for each model"""
        performance = {}
        
        for model_name, model in self.models.items():
            performance[model_name] = {
                "trained": model.get("trained", False),
                "last_update": model.get("last_update"),
                "accuracy": np.random.uniform(0.6, 0.9),  # Mock accuracy
                "precision": np.random.uniform(0.6, 0.9),  # Mock precision
                "recall": np.random.uniform(0.6, 0.9)      # Mock recall
            }
        
        return performance
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated signals"""
        if not self.signal_history:
            return {"total_signals": 0}
        
        signal_types = [s.signal_type for s in self.signal_history]
        confidences = [s.confidence for s in self.signal_history]
        
        return {
            "total_signals": len(self.signal_history),
            "signal_distribution": {
                str(signal_type): signal_types.count(signal_type) 
                for signal_type in SignalType
            },
            "average_confidence": np.mean(confidences),
            "confidence_std": np.std(confidences),
            "last_signal_time": self.signal_history[-1].timestamp
        }


# Global instance
ml_signal_generator = MLSignalGenerator()
