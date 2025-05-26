"""
Real-Time Learning Pipeline
Continuous model adaptation and learning from live trading data

Features:
- Online learning algorithms for continuous adaptation
- Concept drift detection and handling
- Real-time model updates without full retraining
- Performance monitoring and validation
- Adaptive feature selection
- Model ensemble management
- Incremental learning capabilities
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import redis
import json
import pickle
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningMode(Enum):
    ONLINE = "online"
    BATCH = "batch"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

class DriftStatus(Enum):
    STABLE = "stable"
    WARNING = "warning"
    DRIFT_DETECTED = "drift_detected"
    ADAPTING = "adapting"

class ModelStatus(Enum):
    TRAINING = "training"
    READY = "ready"
    UPDATING = "updating"
    DEGRADED = "degraded"
    FAILED = "failed"

@dataclass
class ModelPerformance:
    accuracy: float
    mse: float
    mae: float
    prediction_count: int
    last_updated: datetime
    drift_score: float
    confidence: float

@dataclass
class LearningUpdate:
    features: np.ndarray
    target: Union[float, int]
    timestamp: datetime
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelEnsemble:
    models: Dict[str, Any]
    weights: Dict[str, float]
    performance: Dict[str, ModelPerformance]
    last_updated: datetime

class DriftDetector:
    """Detect concept drift in real-time data streams"""
    
    def __init__(self, window_size: int = 1000, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_data = []
        self.current_data = []
        self.drift_scores = []
        
    def add_sample(self, features: np.ndarray, prediction: float, actual: float):
        """Add new sample for drift detection"""
        error = abs(prediction - actual)
        self.current_data.append(error)
        
        if len(self.current_data) > self.window_size:
            self.current_data.pop(0)
            
        if len(self.reference_data) == 0 and len(self.current_data) >= self.window_size:
            self.reference_data = self.current_data.copy()
    
    def detect_drift(self) -> Tuple[DriftStatus, float]:
        """Detect if concept drift has occurred"""
        if len(self.reference_data) == 0 or len(self.current_data) < self.window_size // 2:
            return DriftStatus.STABLE, 0.0
        
        # Calculate drift score using statistical tests
        ref_mean = np.mean(self.reference_data)
        curr_mean = np.mean(self.current_data)
        ref_std = np.std(self.reference_data)
        
        if ref_std == 0:
            drift_score = 0.0
        else:
            drift_score = abs(curr_mean - ref_mean) / ref_std
        
        self.drift_scores.append(drift_score)
        
        if drift_score > self.threshold * 3:
            return DriftStatus.DRIFT_DETECTED, drift_score
        elif drift_score > self.threshold * 2:
            return DriftStatus.WARNING, drift_score
        else:
            return DriftStatus.STABLE, drift_score

class FeatureEngineer:
    """Real-time feature engineering and selection"""
    
    def __init__(self):
        self.feature_importance = {}
        self.feature_stats = {}
        self.selected_features = []
        
    def engineer_features(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Engineer features from raw market data"""
        features = []
        
        # Price-based features
        if 'price' in raw_data:
            price = raw_data['price']
            features.extend([
                price,
                np.log(price) if price > 0 else 0,
                price ** 2
            ])
        
        # Volume-based features
        if 'volume' in raw_data:
            volume = raw_data['volume']
            features.extend([
                volume,
                np.log(volume + 1),
                volume / raw_data.get('avg_volume', 1)
            ])
        
        # Technical indicators
        if 'indicators' in raw_data:
            indicators = raw_data['indicators']
            for key, value in indicators.items():
                if isinstance(value, (int, float)):
                    features.append(value)
        
        # Time-based features
        if 'timestamp' in raw_data:
            timestamp = raw_data['timestamp']
            if isinstance(timestamp, datetime):
                features.extend([
                    timestamp.hour,
                    timestamp.minute,
                    timestamp.weekday(),
                    timestamp.month
                ])
        
        return np.array(features, dtype=np.float32)
    
    def update_feature_importance(self, features: np.ndarray, target: float, model_coef: np.ndarray):
        """Update feature importance scores"""
        if len(model_coef) == len(features):
            for i, importance in enumerate(model_coef):
                feature_name = f"feature_{i}"
                if feature_name not in self.feature_importance:
                    self.feature_importance[feature_name] = []
                self.feature_importance[feature_name].append(abs(importance))
    
    def select_features(self, features: np.ndarray, top_k: int = 20) -> np.ndarray:
        """Select top-k most important features"""
        if len(self.selected_features) == 0:
            return features[:top_k] if len(features) > top_k else features
        
        return features[self.selected_features]

class RealTimeLearningPipeline:
    """
    Real-time learning pipeline for continuous model adaptation
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Learning configuration
        self.config = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'window_size': 1000,
            'drift_threshold': 0.05,
            'update_frequency': 10,  # updates per minute
            'ensemble_size': 5,
            'feature_selection_threshold': 0.01
        }
        
        # Components
        self.drift_detector = DriftDetector(
            window_size=self.config['window_size'],
            threshold=self.config['drift_threshold']
        )
        self.feature_engineer = FeatureEngineer()
        
        # Models
        self.primary_model = SGDRegressor(
            learning_rate='adaptive',
            eta0=self.config['learning_rate'],
            random_state=42
        )
        self.backup_model = SGDRegressor(
            learning_rate='constant',
            eta0=self.config['learning_rate'] * 0.5,
            random_state=43
        )
        
        # Model ensemble
        self.ensemble = ModelEnsemble(
            models={'primary': self.primary_model, 'backup': self.backup_model},
            weights={'primary': 0.7, 'backup': 0.3},
            performance={},
            last_updated=datetime.now()
        )
        
        # State tracking
        self.learning_buffer = []
        self.performance_history = []
        self.model_status = ModelStatus.TRAINING
        self.learning_mode = LearningMode.ONLINE
        self.is_initialized = False
        
        # Performance statistics
        self.stats = {
            'total_updates': 0,
            'successful_predictions': 0,
            'drift_detections': 0,
            'model_adaptations': 0,
            'average_accuracy': 0.0,
            'last_update': None
        }
        
        logger.info("RealTimeLearningPipeline initialized")

    async def initialize(self, initial_data: Optional[List[Dict]] = None):
        """Initialize the learning pipeline with initial data"""
        try:
            if initial_data:
                logger.info(f"Initializing with {len(initial_data)} samples")
                
                # Prepare initial training data
                X_init = []
                y_init = []
                
                for sample in initial_data:
                    features = self.feature_engineer.engineer_features(sample)
                    if 'target' in sample and len(features) > 0:
                        X_init.append(features)
                        y_init.append(sample['target'])
                
                if len(X_init) > 0:
                    X_init = np.array(X_init)
                    y_init = np.array(y_init)
                    
                    # Ensure all samples have the same number of features
                    min_features = min(len(x) for x in X_init)
                    X_init = np.array([x[:min_features] for x in X_init])
                    
                    # Initial training
                    self.primary_model.fit(X_init, y_init)
                    self.backup_model.fit(X_init, y_init)
                    
                    # Initialize performance tracking
                    predictions = self.primary_model.predict(X_init)
                    initial_mse = mean_squared_error(y_init, predictions)
                    
                    self.ensemble.performance['primary'] = ModelPerformance(
                        accuracy=1.0 - min(initial_mse, 1.0),
                        mse=initial_mse,
                        mae=np.mean(np.abs(y_init - predictions)),
                        prediction_count=len(y_init),
                        last_updated=datetime.now(),
                        drift_score=0.0,
                        confidence=0.8
                    )
                    
                    logger.info(f"✅ Initial training completed with MSE: {initial_mse:.4f}")
            
            self.model_status = ModelStatus.READY
            self.is_initialized = True
            
            # Start background tasks
            asyncio.create_task(self._periodic_model_update())
            asyncio.create_task(self._performance_monitoring())
            
            logger.info("✅ RealTimeLearningPipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize learning pipeline: {e}")
            self.model_status = ModelStatus.FAILED
            raise

    async def learn_online(self, update: LearningUpdate) -> bool:
        """Process a single learning update in online mode"""
        try:
            if not self.is_initialized:
                logger.warning("Pipeline not initialized, buffering update")
                self.learning_buffer.append(update)
                return False
            
            # Add to learning buffer
            self.learning_buffer.append(update)
            
            # Process if we have enough samples or it's time to update
            if (len(self.learning_buffer) >= self.config['batch_size'] or 
                self._should_update_now()):
                await self._process_learning_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in online learning: {e}")
            return False

    async def predict(self, features: Union[np.ndarray, Dict[str, Any]]) -> Tuple[float, float]:
        """Make prediction with confidence score"""
        try:
            if not self.is_initialized:
                return 0.0, 0.0
            
            # Engineer features if raw data provided
            if isinstance(features, dict):
                features = self.feature_engineer.engineer_features(features)
            
            # Ensure features have correct shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get predictions from ensemble
            predictions = {}
            confidences = {}
            
            for model_name, model in self.ensemble.models.items():
                try:
                    pred = model.predict(features)[0]
                    predictions[model_name] = pred
                    
                    # Calculate confidence based on model performance
                    if model_name in self.ensemble.performance:
                        perf = self.ensemble.performance[model_name]
                        confidences[model_name] = perf.confidence
                    else:
                        confidences[model_name] = 0.5
                        
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
                    predictions[model_name] = 0.0
                    confidences[model_name] = 0.0
            
            # Weighted ensemble prediction
            final_prediction = 0.0
            total_weight = 0.0
            
            for model_name, pred in predictions.items():
                weight = self.ensemble.weights.get(model_name, 0.0)
                confidence = confidences.get(model_name, 0.0)
                effective_weight = weight * confidence
                
                final_prediction += pred * effective_weight
                total_weight += effective_weight
            
            if total_weight > 0:
                final_prediction /= total_weight
                final_confidence = total_weight / sum(self.ensemble.weights.values())
            else:
                final_prediction = 0.0
                final_confidence = 0.0
            
            self.stats['successful_predictions'] += 1
            
            return final_prediction, final_confidence
            
        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}")
            return 0.0, 0.0

    async def update_with_feedback(self, features: Union[np.ndarray, Dict[str, Any]], 
                                 prediction: float, actual: float) -> bool:
        """Update model with prediction feedback"""
        try:
            # Engineer features if needed
            if isinstance(features, dict):
                features = self.feature_engineer.engineer_features(features)
            
            # Create learning update
            update = LearningUpdate(
                features=features,
                target=actual,
                timestamp=datetime.now(),
                weight=1.0,
                metadata={'prediction': prediction, 'error': abs(prediction - actual)}
            )
            
            # Add to drift detector
            self.drift_detector.add_sample(features, prediction, actual)
            
            # Check for concept drift
            drift_status, drift_score = self.drift_detector.detect_drift()
            
            if drift_status == DriftStatus.DRIFT_DETECTED:
                logger.warning(f"🚨 Concept drift detected! Score: {drift_score:.4f}")
                await self._handle_concept_drift()
                self.stats['drift_detections'] += 1
            
            # Process learning update
            await self.learn_online(update)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating with feedback: {e}")
            return False

    async def _process_learning_buffer(self):
        """Process accumulated learning updates"""
        try:
            if len(self.learning_buffer) == 0:
                return
            
            # Prepare batch data
            X_batch = []
            y_batch = []
            weights = []
            
            for update in self.learning_buffer:
                X_batch.append(update.features)
                y_batch.append(update.target)
                weights.append(update.weight)
            
            # Ensure consistent feature dimensions
            if len(X_batch) > 0:
                min_features = min(len(x) for x in X_batch)
                X_batch = np.array([x[:min_features] for x in X_batch])
                y_batch = np.array(y_batch)
                weights = np.array(weights)
                
                # Update models
                for model_name, model in self.ensemble.models.items():
                    try:
                        model.partial_fit(X_batch, y_batch, sample_weight=weights)
                        
                        # Update performance metrics
                        predictions = model.predict(X_batch)
                        mse = mean_squared_error(y_batch, predictions)
                        mae = np.mean(np.abs(y_batch - predictions))
                        accuracy = 1.0 - min(mse, 1.0)
                        
                        self.ensemble.performance[model_name] = ModelPerformance(
                            accuracy=accuracy,
                            mse=mse,
                            mae=mae,
                            prediction_count=len(y_batch),
                            last_updated=datetime.now(),
                            drift_score=self.drift_detector.drift_scores[-1] if self.drift_detector.drift_scores else 0.0,
                            confidence=min(accuracy + 0.2, 1.0)
                        )
                        
                    except Exception as e:
                        logger.error(f"Error updating model {model_name}: {e}")
                
                self.stats['total_updates'] += 1
                self.stats['last_update'] = datetime.now()
                
                # Clear buffer
                self.learning_buffer.clear()
                
                logger.info(f"✅ Processed batch of {len(y_batch)} learning updates")
                
        except Exception as e:
            logger.error(f"❌ Error processing learning buffer: {e}")

    async def _handle_concept_drift(self):
        """Handle detected concept drift"""
        try:
            logger.info("🔄 Handling concept drift...")
            
            # Increase learning rate temporarily
            for model in self.ensemble.models.values():
                if hasattr(model, 'eta0'):
                    model.eta0 = min(model.eta0 * 1.5, 0.1)
            
            # Reset drift detector reference
            self.drift_detector.reference_data = self.drift_detector.current_data.copy()
            
            # Adjust ensemble weights based on recent performance
            await self._rebalance_ensemble()
            
            self.stats['model_adaptations'] += 1
            
            logger.info("✅ Concept drift handling completed")
            
        except Exception as e:
            logger.error(f"❌ Error handling concept drift: {e}")

    async def _rebalance_ensemble(self):
        """Rebalance ensemble weights based on performance"""
        try:
            total_performance = 0.0
            performance_scores = {}
            
            for model_name, perf in self.ensemble.performance.items():
                score = perf.accuracy * perf.confidence
                performance_scores[model_name] = score
                total_performance += score
            
            if total_performance > 0:
                for model_name in self.ensemble.weights.keys():
                    if model_name in performance_scores:
                        self.ensemble.weights[model_name] = performance_scores[model_name] / total_performance
                    else:
                        self.ensemble.weights[model_name] = 0.1
            
            self.ensemble.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Error rebalancing ensemble: {e}")

    def _should_update_now(self) -> bool:
        """Check if model should be updated now"""
        if not self.stats['last_update']:
            return True
        
        time_since_update = datetime.now() - self.stats['last_update']
        update_interval = timedelta(minutes=1 / self.config['update_frequency'])
        
        return time_since_update >= update_interval

    async def _periodic_model_update(self):
        """Periodic model maintenance and optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update feature importance
                for model_name, model in self.ensemble.models.items():
                    if hasattr(model, 'coef_') and model.coef_ is not None:
                        # Update feature importance tracking
                        pass
                
                # Rebalance ensemble if needed
                await self._rebalance_ensemble()
                
                # Update statistics
                if self.ensemble.performance:
                    avg_accuracy = np.mean([p.accuracy for p in self.ensemble.performance.values()])
                    self.stats['average_accuracy'] = avg_accuracy
                
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")

    async def _performance_monitoring(self):
        """Monitor and log performance metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Log performance statistics
                logger.info(f"📊 Learning Stats: Updates={self.stats['total_updates']}, "
                          f"Predictions={self.stats['successful_predictions']}, "
                          f"Accuracy={self.stats['average_accuracy']:.3f}, "
                          f"Drifts={self.stats['drift_detections']}")
                
                # Store metrics in Redis
                await self._store_performance_metrics()
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")

    async def _store_performance_metrics(self):
        """Store performance metrics in Redis"""
        try:
            metrics = {
                'stats': self.stats,
                'model_status': self.model_status.value,
                'learning_mode': self.learning_mode.value,
                'ensemble_weights': self.ensemble.weights,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                'realtime_learning_metrics',
                3600,  # 1 hour TTL
                json.dumps(metrics, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")

    async def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return {
            'stats': self.stats,
            'model_status': self.model_status.value,
            'learning_mode': self.learning_mode.value,
            'ensemble_performance': {
                name: {
                    'accuracy': perf.accuracy,
                    'mse': perf.mse,
                    'confidence': perf.confidence,
                    'prediction_count': perf.prediction_count
                }
                for name, perf in self.ensemble.performance.items()
            },
            'ensemble_weights': self.ensemble.weights,
            'buffer_size': len(self.learning_buffer),
            'is_initialized': self.is_initialized
        }
