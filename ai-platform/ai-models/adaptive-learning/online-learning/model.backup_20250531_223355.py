"""
Online Learning Framework
Sophisticated online learning system for continuous adaptation of trading models
to changing market conditions with real-time learning algorithms and dynamic updates.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import pickle
import joblib
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import warnings
from abc import ABC, abstractmethod

# ML imports
try:
    import tensorflow as tf
    from sklearn.base import BaseEstimator
    from sklearn.linear_model import SGDRegressor, SGDClassifier
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
    from river import drift
    import river.metrics as metrics
    import river.preprocessing as preprocessing
    import river.linear_model as linear_model
    import river.ensemble as ensemble
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.warning("Some ML libraries not available. Using fallback implementations.")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for online adaptation"""
    INCREMENTAL = "incremental"
    BATCH_INCREMENTAL = "batch_incremental"
    ENSEMBLE_WEIGHTED = "ensemble_weighted"
    CONCEPT_DRIFT_ADAPTIVE = "concept_drift_adaptive"


class DriftDetectionMethod(Enum):
    """Drift detection algorithms"""
    ADWIN = "adwin"
    DDM = "ddm"
    EDDM = "eddm"
    PAGE_HINKLEY = "page_hinkley"
    STATISTICAL = "statistical"


@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    timestamp: datetime
    model_id: str
    accuracy: float
    loss: float
    prediction_count: int
    drift_detected: bool
    adaptation_count: int
    learning_rate: float
    confidence: float
    processing_time_ms: float


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning framework"""
    learning_mode: LearningMode = LearningMode.INCREMENTAL
    drift_detection_method: DriftDetectionMethod = DriftDetectionMethod.ADWIN
    max_ensemble_size: int = 5
    min_samples_before_drift_check: int = 100
    drift_sensitivity: float = 0.95
    adaptation_rate: float = 0.01
    performance_window_size: int = 1000
    memory_limit_mb: int = 500
    auto_save_interval_minutes: int = 30
    enable_a_b_testing: bool = True
    rollback_threshold: float = 0.1  # Performance degradation threshold


class BaseLearner(ABC):
    """Abstract base class for online learners"""
    
    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)"""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict:
        """Get model state for persistence"""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict) -> None:
        """Set model state from persistence"""
        pass


class IncrementalSGDLearner(BaseLearner):
    """Incremental learning with Stochastic Gradient Descent"""
    
    def __init__(self, task_type: str = "regression", random_state: int = 42):
        self.task_type = task_type.lower()
        self.random_state = random_state
        
        if self.task_type == "regression":
            self.model = SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                random_state=random_state,
                warm_start=True
            )
        else:
            self.model = SGDClassifier(
                learning_rate='adaptive',
                eta0=0.01,
                random_state=random_state,
                warm_start=True
            )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally train the model"""
        X_scaled = self.scaler.fit_transform(X) if not self.is_fitted else self.scaler.transform(X)
        
        if not self.is_fitted:
            if self.task_type == "classification":
                classes = np.unique(y)
                self.model.partial_fit(X_scaled, y, classes=classes)
            else:
                self.model.partial_fit(X_scaled, y)
            self.is_fitted = True
        else:
            self.model.partial_fit(X_scaled, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)"""
        if not self.is_fitted or self.task_type == "regression":
            raise ValueError("Model not fitted or not a classification task")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_state(self) -> Dict:
        """Get model state for persistence"""
        return {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'task_type': self.task_type
        }
    
    def set_state(self, state: Dict) -> None:
        """Set model state from persistence"""
        self.model = state['model']
        self.scaler = state['scaler']
        self.is_fitted = state['is_fitted']
        self.task_type = state['task_type']


class DriftDetector:
    """Concept drift detection system"""
    
    def __init__(self, method: DriftDetectionMethod, sensitivity: float = 0.95):
        self.method = method
        self.sensitivity = sensitivity
        self.detector = None
        self.initialize_detector()
        
        # Statistics for custom detection
        self.performance_history = deque(maxlen=1000)
        self.baseline_performance = None
        
    def initialize_detector(self):
        """Initialize drift detector based on method"""
        if not RIVER_AVAILABLE:
            logger.warning("River library not available. Using statistical drift detection.")
            self.method = DriftDetectionMethod.STATISTICAL
            return
            
        try:
            if self.method == DriftDetectionMethod.ADWIN:
                self.detector = drift.ADWIN(delta=1-self.sensitivity)
            elif self.method == DriftDetectionMethod.DDM:
                self.detector = drift.DDM()
            elif self.method == DriftDetectionMethod.EDDM:
                self.detector = drift.EDDM()
            elif self.method == DriftDetectionMethod.PAGE_HINKLEY:
                self.detector = drift.PageHinkley()
            else:
                self.method = DriftDetectionMethod.STATISTICAL
        except Exception as e:
            logger.warning(f"Failed to initialize {self.method} detector: {e}")
            self.method = DriftDetectionMethod.STATISTICAL
    
    def update(self, performance_metric: float) -> bool:
        """Update detector and check for drift"""
        if self.method == DriftDetectionMethod.STATISTICAL:
            return self._statistical_drift_detection(performance_metric)
        
        if self.detector is None:
            return False
            
        try:
            self.detector.update(performance_metric)
            drift_detected = self.detector.drift_detected
            
            if drift_detected:
                logger.info(f"Concept drift detected using {self.method}")
                self.detector.reset()
                
            return drift_detected
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return False
    
    def _statistical_drift_detection(self, performance_metric: float) -> bool:
        """Statistical approach to drift detection"""
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) < 50:
            return False
        
        # Set baseline on first 100 samples
        if self.baseline_performance is None and len(self.performance_history) >= 100:
            self.baseline_performance = np.mean(list(self.performance_history)[:100])
        
        if self.baseline_performance is None:
            return False
        
        # Check recent performance vs baseline
        recent_window = 30
        if len(self.performance_history) >= recent_window:
            recent_performance = np.mean(list(self.performance_history)[-recent_window:])
            performance_degradation = (self.baseline_performance - recent_performance) / self.baseline_performance
            
            # Detect significant degradation
            if performance_degradation > (1 - self.sensitivity):
                logger.info(f"Statistical drift detected. Performance degraded by {performance_degradation:.3f}")
                # Update baseline
                self.baseline_performance = recent_performance
                return True
        
        return False


class EnsembleManager:
    """Manage ensemble of online learners"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.learners: List[Tuple[BaseLearner, float, datetime]] = []  # (learner, weight, created_time)
        self.performance_tracker = defaultdict(list)
        
    def add_learner(self, learner: BaseLearner, weight: float = 1.0):
        """Add new learner to ensemble"""
        self.learners.append((learner, weight, datetime.now()))
        
        # Remove oldest if exceeding max size
        if len(self.learners) > self.max_size:
            self.learners.pop(0)
            
        self._normalize_weights()
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction using weighted voting"""
        if not self.learners:
            raise ValueError("No learners in ensemble")
        
        predictions = []
        weights = []
        
        for learner, weight, _ in self.learners:
            try:
                pred = learner.predict(X)
                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                logger.warning(f"Learner prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("All learners failed to predict")
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        return np.average(predictions, axis=0, weights=weights)
    
    def update_weights(self, performances: List[float]):
        """Update learner weights based on performance"""
        if len(performances) != len(self.learners):
            return
        
        # Convert performances to weights (higher performance = higher weight)
        weights = np.array(performances)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
        
        for i, (learner, _, created_time) in enumerate(self.learners):
            self.learners[i] = (learner, weights[i], created_time)
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        if not self.learners:
            return
        
        total_weight = sum(weight for _, weight, _ in self.learners)
        if total_weight > 0:
            self.learners = [(learner, weight/total_weight, created_time) 
                           for learner, weight, created_time in self.learners]


class OnlineLearning:
    """
    Sophisticated online learning framework for continuous adaptation of trading models
    """
    
    def __init__(self, config: OnlineLearningConfig = None):
        """Initialize online learning framework"""
        self.config = config or OnlineLearningConfig()
        
        # Core components
        self.models: Dict[str, BaseLearner] = {}
        self.ensemble_managers: Dict[str, EnsembleManager] = {}
        self.drift_detectors: Dict[str, DriftDetector] = {}
        self.performance_trackers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.performance_window_size))
        
        # State management
        self.model_versions: Dict[str, int] = defaultdict(int)
        self.adaptation_counts: Dict[str, int] = defaultdict(int)
        self.last_adaptation_time: Dict[str, datetime] = {}
        
        # Threading for background processes
        self.background_thread = None
        self.stop_background = False
        
        # A/B testing
        self.ab_test_groups: Dict[str, Dict] = {}
        
        logger.info("OnlineLearning framework initialized")
        
    def register_model(self, model_id: str, task_type: str = "regression", 
                      learner_type: str = "sgd") -> str:
        """
        Register a new model for online learning
        
        Args:
            model_id (str): Unique identifier for the model
            task_type (str): 'regression' or 'classification'
            learner_type (str): Type of learner ('sgd', 'ensemble')
            
        Returns:
            str: Confirmation message
        """
        if learner_type == "sgd":
            learner = IncrementalSGDLearner(task_type=task_type)
        else:
            learner = IncrementalSGDLearner(task_type=task_type)  # Default fallback
        
        self.models[model_id] = learner
        self.ensemble_managers[model_id] = EnsembleManager(max_size=self.config.max_ensemble_size)
        self.drift_detectors[model_id] = DriftDetector(
            method=self.config.drift_detection_method,
            sensitivity=self.config.drift_sensitivity
        )
        
        # Add initial learner to ensemble
        self.ensemble_managers[model_id].add_learner(learner)
        
        logger.info(f"Model '{model_id}' registered for online learning")
        return f"Model '{model_id}' successfully registered"
    
    def update_model(self, model_id: str, new_data: Union[Dict, pd.DataFrame, Tuple]) -> Dict:
        """
        Update model with new data using online learning
        
        Args:
            model_id (str): Model identifier
            new_data: New training data (X, y) or DataFrame
            
        Returns:
            Dict: Update results and performance metrics
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not registered")
        
        start_time = time.time()
        
        # Parse input data
        X, y = self._parse_input_data(new_data)
        
        if len(X) == 0:
            return {"status": "No data to process"}
        
        # Get current model
        model = self.models[model_id]
        ensemble = self.ensemble_managers[model_id]
        drift_detector = self.drift_detectors[model_id]
        
        # Make predictions before update (for drift detection)
        try:
            if model.is_fitted:
                predictions = model.predict(X)
                if model.task_type == "regression":
                    performance = 1.0 - mean_squared_error(y, predictions, squared=False)  # RMSE-based score
                else:
                    performance = accuracy_score(y, (predictions > 0.5).astype(int))
            else:
                performance = 0.5  # Neutral starting performance
        except Exception as e:
            logger.warning(f"Error calculating performance: {e}")
            performance = 0.5
        
        # Check for concept drift
        drift_detected = drift_detector.update(performance)
        
        # Update model incrementally
        try:
            model.partial_fit(X, y)
            update_success = True
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            update_success = False
        
        # Handle concept drift
        if drift_detected:
            self._handle_concept_drift(model_id, X, y)
        
        # Update performance tracking
        performance_data = ModelPerformance(
            timestamp=datetime.now(),
            model_id=model_id,
            accuracy=performance,
            loss=1.0 - performance,
            prediction_count=len(X),
            drift_detected=drift_detected,
            adaptation_count=self.adaptation_counts[model_id],
            learning_rate=self.config.adaptation_rate,
            confidence=min(performance * 1.2, 1.0),
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        self.performance_trackers[model_id].append(performance_data)
        
        # Update version if significant change
        if drift_detected or len(X) > 100:
            self.model_versions[model_id] += 1
        
        result = {
            "model_id": model_id,
            "samples_processed": len(X),
            "drift_detected": drift_detected,
            "performance": performance,
            "model_version": self.model_versions[model_id],
            "adaptation_count": self.adaptation_counts[model_id],
            "processing_time_ms": performance_data.processing_time_ms,
            "update_success": update_success,
            "ensemble_size": len(ensemble.learners)
        }
        
        logger.info(f"Model '{model_id}' updated: {result}")
        return result
    
    def predict(self, model_id: str, X: np.ndarray, use_ensemble: bool = True) -> Dict:
        """
        Make predictions using the online learning model
        
        Args:
            model_id (str): Model identifier
            X (np.ndarray): Input features
            use_ensemble (bool): Whether to use ensemble prediction
            
        Returns:
            Dict: Predictions and metadata
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not registered")
        
        start_time = time.time()
        
        try:
            if use_ensemble and len(self.ensemble_managers[model_id].learners) > 1:
                predictions = self.ensemble_managers[model_id].predict(X)
                prediction_method = "ensemble"
            else:
                predictions = self.models[model_id].predict(X)
                prediction_method = "single_model"
            
            # Get confidence estimates
            if self.models[model_id].task_type == "classification":
                try:
                    probabilities = self.models[model_id].predict_proba(X)
                    confidence = np.max(probabilities, axis=1)
                except:
                    confidence = np.full(len(predictions), 0.5)
            else:
                # For regression, use inverse of recent prediction variance as confidence
                recent_performance = list(self.performance_trackers[model_id])[-10:]
                if recent_performance:
                    variance = np.var([p.accuracy for p in recent_performance])
                    confidence = np.full(len(predictions), max(0.1, 1.0 - variance))
                else:
                    confidence = np.full(len(predictions), 0.5)
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                "model_id": model_id,
                "predictions": predictions.tolist(),
                "confidence": confidence.tolist(),
                "prediction_method": prediction_method,
                "model_version": self.model_versions[model_id],
                "processing_time_ms": processing_time,
                "ensemble_size": len(self.ensemble_managers[model_id].learners)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for model '{model_id}': {e}")
            return {
                "model_id": model_id,
                "error": str(e),
                "predictions": [],
                "confidence": []
            }
    
    def _parse_input_data(self, data: Union[Dict, pd.DataFrame, Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """Parse various input data formats"""
        if isinstance(data, tuple) and len(data) == 2:
            X, y = data
        elif isinstance(data, pd.DataFrame):
            if 'target' in data.columns:
                y = data['target'].values
                X = data.drop('target', axis=1).values
            else:
                # Assume last column is target
                X = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values
        elif isinstance(data, dict):
            X = np.array(data.get('features', []))
            y = np.array(data.get('targets', []))
        else:
            raise ValueError("Unsupported data format")
        
        return np.atleast_2d(X), np.atleast_1d(y)
    
    def _handle_concept_drift(self, model_id: str, X: np.ndarray, y: np.ndarray):
        """Handle detected concept drift"""
        logger.info(f"Handling concept drift for model '{model_id}'")
        
        self.adaptation_counts[model_id] += 1
        self.last_adaptation_time[model_id] = datetime.now()
        
        # Strategy: Create new learner for ensemble
        if self.config.learning_mode == LearningMode.ENSEMBLE_WEIGHTED:
            new_learner = IncrementalSGDLearner(
                task_type=self.models[model_id].task_type,
                random_state=42 + self.adaptation_counts[model_id]
            )
            
            # Train new learner on recent data
            new_learner.partial_fit(X, y)
            
            # Add to ensemble
            self.ensemble_managers[model_id].add_learner(new_learner, weight=1.0)
            
        # Update adaptation strategy based on mode
        if self.config.learning_mode == LearningMode.CONCEPT_DRIFT_ADAPTIVE:
            # Increase learning rate temporarily
            if hasattr(self.models[model_id].model, 'eta0'):
                self.models[model_id].model.eta0 *= 1.5
    
    def get_model_status(self, model_id: str = None) -> Dict:
        """Get comprehensive model status and performance metrics"""
        if model_id and model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not registered")
        
        if model_id:
            model_ids = [model_id]
        else:
            model_ids = list(self.models.keys())
        
        status = {}
        
        for mid in model_ids:
            recent_performance = list(self.performance_trackers[mid])[-10:]
            
            status[mid] = {
                "model_version": self.model_versions[mid],
                "adaptation_count": self.adaptation_counts[mid],
                "ensemble_size": len(self.ensemble_managers[mid].learners),
                "is_fitted": self.models[mid].is_fitted,
                "task_type": self.models[mid].task_type,
                "total_samples_processed": sum(p.prediction_count for p in self.performance_trackers[mid]),
                "recent_performance": {
                    "avg_accuracy": np.mean([p.accuracy for p in recent_performance]) if recent_performance else 0,
                    "avg_processing_time_ms": np.mean([p.processing_time_ms for p in recent_performance]) if recent_performance else 0,
                    "drift_detections": sum(p.drift_detected for p in recent_performance),
                },
                "last_adaptation": self.last_adaptation_time.get(mid),
                "config": {
                    "learning_mode": self.config.learning_mode.value,
                    "drift_detection_method": self.config.drift_detection_method.value,
                    "max_ensemble_size": self.config.max_ensemble_size
                }
            }
        
        return status
    
    def save_models(self, save_path: str = "online_learning_models"):
        """Save all models and state"""
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        for model_id, model in self.models.items():
            model_data = {
                'model_state': model.get_state(),
                'version': self.model_versions[model_id],
                'adaptation_count': self.adaptation_counts[model_id],
                'performance_history': list(self.performance_trackers[model_id])
            }
            
            with open(f"{save_path}/{model_id}_online_model.pkl", 'wb') as f:
                pickle.dump(model_data, f)
        
        # Save framework config
        with open(f"{save_path}/framework_config.pkl", 'wb') as f:
            pickle.dump(self.config, f)
        
        logger.info(f"Models saved to {save_path}")
    
    def load_models(self, load_path: str = "online_learning_models"):
        """Load all models and state"""
        import os
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Load path {load_path} does not exist")
        
        # Load framework config
        config_path = f"{load_path}/framework_config.pkl"
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
        
        # Load models
        for filename in os.listdir(load_path):
            if filename.endswith('_online_model.pkl'):
                model_id = filename.replace('_online_model.pkl', '')
                
                with open(f"{load_path}/{filename}", 'rb') as f:
                    model_data = pickle.load(f)
                
                # Recreate model
                learner = IncrementalSGDLearner()
                learner.set_state(model_data['model_state'])
                
                self.models[model_id] = learner
                self.model_versions[model_id] = model_data['version']
                self.adaptation_counts[model_id] = model_data['adaptation_count']
                
                # Restore performance history
                self.performance_trackers[model_id] = deque(
                    model_data['performance_history'],
                    maxlen=self.config.performance_window_size
                )
                
                # Recreate ensemble and drift detector
                self.ensemble_managers[model_id] = EnsembleManager(max_size=self.config.max_ensemble_size)
                self.ensemble_managers[model_id].add_learner(learner)
                
                self.drift_detectors[model_id] = DriftDetector(
                    method=self.config.drift_detection_method,
                    sensitivity=self.config.drift_sensitivity
                )
        
        logger.info(f"Models loaded from {load_path}")
    
    def start_background_monitoring(self):
        """Start background thread for monitoring and maintenance"""
        if self.background_thread is not None:
            return
        
        def background_worker():
            while not self.stop_background:
                try:
                    # Auto-save models periodically
                    if self.config.auto_save_interval_minutes > 0:
                        self.save_models()
                    
                    # Sleep for check interval
                    time.sleep(self.config.auto_save_interval_minutes * 60)
                    
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    time.sleep(60)  # Wait 1 minute before retry
        
        self.background_thread = threading.Thread(target=background_worker, daemon=True)
        self.background_thread.start()
        logger.info("Background monitoring started")
    
    def stop_background_monitoring(self):
        """Stop background monitoring"""
        self.stop_background = True
        if self.background_thread:
            self.background_thread.join(timeout=5)
            self.background_thread = None
        logger.info("Background monitoring stopped")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_background_monitoring()


# Example usage and testing
if __name__ == "__main__":
    # Initialize online learning framework
    config = OnlineLearningConfig(
        learning_mode=LearningMode.ENSEMBLE_WEIGHTED,
        drift_detection_method=DriftDetectionMethod.STATISTICAL,
        max_ensemble_size=3,
        drift_sensitivity=0.9
    )
    
    ol = OnlineLearning(config)
    
    # Register a trading model
    ol.register_model("forex_eur_usd", task_type="regression", learner_type="sgd")
    
    # Simulate streaming data updates
    np.random.seed(42)
    n_batches = 10
    batch_size = 50
    
    for i in range(n_batches):
        # Generate synthetic trading data
        X = np.random.randn(batch_size, 5)  # 5 features
        
        # Create target with regime change at batch 5
        if i < 5:
            y = X.sum(axis=1) + np.random.normal(0, 0.1, batch_size)  # Linear relationship
        else:
            y = -X.sum(axis=1) + np.random.normal(0, 0.1, batch_size)  # Inverted relationship (concept drift)
        
        # Update model
        result = ol.update_model("forex_eur_usd", (X, y))
        print(f"Batch {i+1}: {result}")
        
        # Make predictions
        pred_result = ol.predict("forex_eur_usd", X[:5])
        print(f"Predictions: {pred_result['predictions'][:3]}")
        print(f"Confidence: {pred_result['confidence'][:3]}")
        print("-" * 50)
    
    # Get final status
    status = ol.get_model_status("forex_eur_usd")
    print(f"\nFinal Model Status:")
    for key, value in status["forex_eur_usd"].items():
        print(f"{key}: {value}")
    
    print("\nOnline Learning Framework successfully demonstrated!")
