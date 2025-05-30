"""
Rapid Learning Pipeline for Real-Time Model Adaptation

This module implements a rapid learning pipeline that enables real-time model
adaptation and continuous learning from live trading data. It provides fast
model updates, concept drift detection, and performance-based model selection.

Key Features:
- Real-time model adaptation and updates
- Concept drift detection and handling
- Performance-based model selection
- Incremental learning capabilities
- Fast feature engineering and selection
- Automated hyperparameter optimization
- Model ensemble management

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import json
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Learning mode types"""
    BATCH = "batch"
    INCREMENTAL = "incremental"
    ONLINE = "online"
    ENSEMBLE = "ensemble"

class DriftStatus(Enum):
    """Concept drift detection status"""
    STABLE = "stable"
    WARNING = "warning"
    DRIFT_DETECTED = "drift_detected"
    SEVERE_DRIFT = "severe_drift"

class ModelStatus(Enum):
    """Model status types"""
    TRAINING = "training"
    READY = "ready"
    UPDATING = "updating"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_id: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_confidence: float
    training_samples: int
    validation_samples: int
    feature_importance: Dict[str, float]
    drift_score: float

@dataclass
class LearningUpdate:
    """Learning update result"""
    model_id: str
    timestamp: datetime
    update_type: str
    performance_before: ModelPerformance
    performance_after: ModelPerformance
    features_updated: List[str]
    samples_processed: int
    update_duration: float
    success: bool

@dataclass
class ModelEnsemble:
    """Model ensemble configuration"""
    ensemble_id: str
    models: List[str]
    weights: List[float]
    voting_method: str
    performance: ModelPerformance
    last_update: datetime

class RapidLearningPipeline:
    """
    Rapid Learning Pipeline for Real-Time Model Adaptation
    
    Provides fast model updates, concept drift detection, and continuous learning
    capabilities for real-time trading applications.
    """
    
    def __init__(self,
                 learning_mode: LearningMode = LearningMode.INCREMENTAL,
                 update_frequency: int = 100,  # samples
                 drift_threshold: float = 0.1,
                 max_models: int = 5,
                 feature_selection_k: int = 20):
        """
        Initialize Rapid Learning Pipeline
        
        Args:
            learning_mode: Type of learning approach
            update_frequency: Number of samples between updates
            drift_threshold: Threshold for drift detection
            max_models: Maximum number of models in ensemble
            feature_selection_k: Number of top features to select
        """
        self.learning_mode = learning_mode
        self.update_frequency = update_frequency
        self.drift_threshold = drift_threshold
        self.max_models = max_models
        self.feature_selection_k = feature_selection_k
        
        # Model management
        self.active_models = {}
        self.model_performance = {}
        self.model_ensembles = {}
        self.feature_selectors = {}
        self.scalers = {}
        
        # Learning state
        self.sample_buffer = []
        self.performance_history = []
        self.drift_detector = DriftDetector()
        self.feature_engineer = FeatureEngineer()
        
        # Performance tracking
        self.update_count = 0
        self.total_samples_processed = 0
        
        logger.info(f"✅ RapidLearningPipeline initialized (mode={learning_mode.value})")

    async def initialize_models(self, initial_data: pd.DataFrame, 
                              target_column: str) -> bool:
        """
        Initialize models with initial training data
        
        Args:
            initial_data: Initial training dataset
            target_column: Name of target column
            
        Returns:
            Success status
        """
        try:
            # Prepare features and target
            X = initial_data.drop(columns=[target_column])
            y = initial_data[target_column]
            
            # Feature engineering
            X_engineered = await self.feature_engineer.engineer_features(X)
            
            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=self.feature_selection_k)
            X_selected = selector.fit_transform(X_engineered, y)
            selected_features = X_engineered.columns[selector.get_support()].tolist()
            
            # Scaling
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)
            
            # Store preprocessing components
            self.feature_selectors['main'] = selector
            self.scalers['main'] = scaler
            
            # Initialize multiple models
            models_config = {
                'rf_fast': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
                'gb_fast': GradientBoostingClassifier(n_estimators=50, max_depth=6, random_state=42),
                'sgd_online': SGDClassifier(loss='log', random_state=42),
                'lr_fast': LogisticRegression(random_state=42, max_iter=100)
            }
            
            # Train initial models
            for model_id, model in models_config.items():
                try:
                    model.fit(X_scaled, y)
                    self.active_models[model_id] = {
                        'model': model,
                        'status': ModelStatus.READY,
                        'features': selected_features,
                        'created_at': datetime.now(),
                        'last_update': datetime.now(),
                        'update_count': 0
                    }
                    
                    # Calculate initial performance
                    y_pred = model.predict(X_scaled)
                    performance = self._calculate_performance(model_id, y, y_pred, X_scaled.shape[0])
                    self.model_performance[model_id] = performance
                    
                    logger.info(f"✅ Model {model_id} initialized with accuracy: {performance.accuracy:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error initializing model {model_id}: {e}")
            
            # Create initial ensemble
            if len(self.active_models) > 1:
                await self._create_ensemble('main_ensemble', list(self.active_models.keys()))
            
            return len(self.active_models) > 0
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False

    async def process_new_sample(self, sample_data: Dict, target_value: float) -> Dict:
        """
        Process new sample for learning
        
        Args:
            sample_data: New sample features
            target_value: Target value for the sample
            
        Returns:
            Processing result with predictions and updates
        """
        try:
            # Add to buffer
            sample_with_target = sample_data.copy()
            sample_with_target['target'] = target_value
            sample_with_target['timestamp'] = datetime.now()
            self.sample_buffer.append(sample_with_target)
            
            # Get predictions from active models
            predictions = await self._get_model_predictions(sample_data)
            
            # Check if update is needed
            update_needed = len(self.sample_buffer) >= self.update_frequency
            
            result = {
                'predictions': predictions,
                'ensemble_prediction': await self._get_ensemble_prediction(sample_data),
                'update_triggered': update_needed,
                'drift_status': self.drift_detector.get_current_status(),
                'buffer_size': len(self.sample_buffer)
            }
            
            # Trigger update if needed
            if update_needed:
                update_result = await self._trigger_model_update()
                result['update_result'] = update_result
            
            self.total_samples_processed += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing new sample: {e}")
            return {'error': str(e)}

    async def _get_model_predictions(self, sample_data: Dict) -> Dict:
        """Get predictions from all active models"""
        predictions = {}
        
        try:
            # Convert sample to DataFrame
            sample_df = pd.DataFrame([sample_data])
            
            # Engineer features
            sample_engineered = await self.feature_engineer.engineer_features(sample_df)
            
            # Apply feature selection and scaling
            if 'main' in self.feature_selectors:
                sample_selected = self.feature_selectors['main'].transform(sample_engineered)
                sample_scaled = self.scalers['main'].transform(sample_selected)
                
                # Get predictions from each model
                for model_id, model_info in self.active_models.items():
                    if model_info['status'] == ModelStatus.READY:
                        try:
                            model = model_info['model']
                            pred = model.predict(sample_scaled)[0]
                            pred_proba = model.predict_proba(sample_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
                            
                            predictions[model_id] = {
                                'prediction': pred,
                                'confidence': max(pred_proba),
                                'probabilities': pred_proba.tolist()
                            }
                            
                        except Exception as e:
                            logger.error(f"Error getting prediction from {model_id}: {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting model predictions: {e}")
            return {}

    async def _get_ensemble_prediction(self, sample_data: Dict) -> Dict:
        """Get ensemble prediction"""
        try:
            if 'main_ensemble' not in self.model_ensembles:
                return {}
            
            ensemble = self.model_ensembles['main_ensemble']
            predictions = await self._get_model_predictions(sample_data)
            
            if not predictions:
                return {}
            
            # Weighted voting
            weighted_pred = 0.0
            total_weight = 0.0
            confidence_scores = []
            
            for i, model_id in enumerate(ensemble.models):
                if model_id in predictions:
                    weight = ensemble.weights[i]
                    pred = predictions[model_id]['prediction']
                    confidence = predictions[model_id]['confidence']
                    
                    weighted_pred += pred * weight * confidence
                    total_weight += weight * confidence
                    confidence_scores.append(confidence)
            
            if total_weight > 0:
                final_prediction = weighted_pred / total_weight
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
                
                return {
                    'prediction': final_prediction,
                    'confidence': avg_confidence,
                    'ensemble_id': ensemble.ensemble_id,
                    'models_used': len([m for m in ensemble.models if m in predictions])
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting ensemble prediction: {e}")
            return {}

    async def _trigger_model_update(self) -> LearningUpdate:
        """Trigger model update with buffered samples"""
        try:
            start_time = datetime.now()
            
            # Convert buffer to DataFrame
            buffer_df = pd.DataFrame(self.sample_buffer)
            X = buffer_df.drop(columns=['target', 'timestamp'])
            y = buffer_df['target']
            
            # Engineer features
            X_engineered = await self.feature_engineer.engineer_features(X)
            
            # Apply preprocessing
            X_selected = self.feature_selectors['main'].transform(X_engineered)
            X_scaled = self.scalers['main'].transform(X_selected)
            
            # Detect concept drift
            drift_detected = await self.drift_detector.detect_drift(X_scaled, y)
            
            # Update models based on learning mode
            if self.learning_mode == LearningMode.INCREMENTAL:
                update_result = await self._incremental_update(X_scaled, y)
            elif self.learning_mode == LearningMode.BATCH:
                update_result = await self._batch_update(X_scaled, y)
            elif self.learning_mode == LearningMode.ONLINE:
                update_result = await self._online_update(X_scaled, y)
            else:  # ENSEMBLE
                update_result = await self._ensemble_update(X_scaled, y)
            
            # Clear buffer
            self.sample_buffer = []
            self.update_count += 1
            
            # Calculate update duration
            update_duration = (datetime.now() - start_time).total_seconds()
            
            # Create update result
            learning_update = LearningUpdate(
                model_id='pipeline',
                timestamp=datetime.now(),
                update_type=self.learning_mode.value,
                performance_before=update_result.get('performance_before'),
                performance_after=update_result.get('performance_after'),
                features_updated=update_result.get('features_updated', []),
                samples_processed=len(y),
                update_duration=update_duration,
                success=update_result.get('success', False)
            )
            
            logger.info(f"✅ Model update completed in {update_duration:.2f}s")
            
            return learning_update
            
        except Exception as e:
            logger.error(f"Error in model update: {e}")
            return LearningUpdate('pipeline', datetime.now(), 'failed', None, None, [], 0, 0.0, False)

    async def _incremental_update(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform incremental model update"""
        try:
            results = {'success': True, 'features_updated': []}
            
            for model_id, model_info in self.active_models.items():
                if model_info['status'] == ModelStatus.READY:
                    model = model_info['model']
                    
                    # Store performance before update
                    y_pred_before = model.predict(X)
                    perf_before = self._calculate_performance(model_id, y, y_pred_before, len(X))
                    
                    # Incremental update for compatible models
                    if hasattr(model, 'partial_fit'):
                        model.partial_fit(X, y)
                    else:
                        # For non-incremental models, retrain with recent data
                        model.fit(X, y)
                    
                    # Calculate performance after update
                    y_pred_after = model.predict(X)
                    perf_after = self._calculate_performance(model_id, y, y_pred_after, len(X))
                    
                    # Update model info
                    model_info['last_update'] = datetime.now()
                    model_info['update_count'] += 1
                    
                    # Store performance
                    self.model_performance[model_id] = perf_after
                    
                    results['performance_before'] = perf_before
                    results['performance_after'] = perf_after
                    
                    logger.info(f"Model {model_id} updated: {perf_before.accuracy:.3f} -> {perf_after.accuracy:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            return {'success': False}

    async def _batch_update(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform batch model update"""
        try:
            # Retrain all models with new batch
            results = {'success': True, 'features_updated': []}
            
            for model_id, model_info in self.active_models.items():
                if model_info['status'] == ModelStatus.READY:
                    model = model_info['model']
                    
                    # Store performance before
                    y_pred_before = model.predict(X)
                    perf_before = self._calculate_performance(model_id, y, y_pred_before, len(X))
                    
                    # Retrain model
                    model.fit(X, y)
                    
                    # Calculate performance after
                    y_pred_after = model.predict(X)
                    perf_after = self._calculate_performance(model_id, y, y_pred_after, len(X))
                    
                    # Update tracking
                    model_info['last_update'] = datetime.now()
                    model_info['update_count'] += 1
                    self.model_performance[model_id] = perf_after
                    
                    results['performance_before'] = perf_before
                    results['performance_after'] = perf_after
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch update: {e}")
            return {'success': False}

    async def _online_update(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform online model update"""
        try:
            # Update models one sample at a time
            results = {'success': True, 'features_updated': []}
            
            for i in range(len(X)):
                sample_X = X[i:i+1]
                sample_y = y[i:i+1]
                
                for model_id, model_info in self.active_models.items():
                    if model_info['status'] == ModelStatus.READY:
                        model = model_info['model']
                        
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(sample_X, sample_y)
            
            # Calculate final performance
            for model_id, model_info in self.active_models.items():
                if model_info['status'] == ModelStatus.READY:
                    model = model_info['model']
                    y_pred = model.predict(X)
                    perf = self._calculate_performance(model_id, y, y_pred, len(X))
                    self.model_performance[model_id] = perf
                    
                    model_info['last_update'] = datetime.now()
                    model_info['update_count'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in online update: {e}")
            return {'success': False}

    async def _ensemble_update(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform ensemble model update"""
        try:
            # Update individual models first
            await self._incremental_update(X, y)
            
            # Update ensemble weights based on recent performance
            if 'main_ensemble' in self.model_ensembles:
                ensemble = self.model_ensembles['main_ensemble']
                
                # Calculate performance for each model
                model_scores = {}
                for model_id in ensemble.models:
                    if model_id in self.model_performance:
                        perf = self.model_performance[model_id]
                        model_scores[model_id] = perf.f1_score
                
                # Update weights based on performance
                if model_scores:
                    total_score = sum(model_scores.values())
                    if total_score > 0:
                        new_weights = [model_scores.get(model_id, 0) / total_score 
                                     for model_id in ensemble.models]
                        ensemble.weights = new_weights
                        ensemble.last_update = datetime.now()
            
            return {'success': True, 'features_updated': []}
            
        except Exception as e:
            logger.error(f"Error in ensemble update: {e}")
            return {'success': False}

    async def _create_ensemble(self, ensemble_id: str, model_ids: List[str]) -> bool:
        """Create model ensemble"""
        try:
            # Equal weights initially
            weights = [1.0 / len(model_ids)] * len(model_ids)
            
            ensemble = ModelEnsemble(
                ensemble_id=ensemble_id,
                models=model_ids,
                weights=weights,
                voting_method='weighted',
                performance=None,
                last_update=datetime.now()
            )
            
            self.model_ensembles[ensemble_id] = ensemble
            logger.info(f"✅ Ensemble {ensemble_id} created with {len(model_ids)} models")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            return False

    def _calculate_performance(self, model_id: str, y_true: np.ndarray, 
                             y_pred: np.ndarray, n_samples: int) -> ModelPerformance:
        """Calculate model performance metrics"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Calculate prediction confidence (simplified)
            confidence = accuracy  # Can be enhanced with actual confidence scores
            
            # Feature importance (if available)
            feature_importance = {}
            if model_id in self.active_models:
                model = self.active_models[model_id]['model']
                if hasattr(model, 'feature_importances_'):
                    features = self.active_models[model_id]['features']
                    importances = model.feature_importances_
                    feature_importance = dict(zip(features, importances))
            
            return ModelPerformance(
                model_id=model_id,
                timestamp=datetime.now(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                prediction_confidence=confidence,
                training_samples=n_samples,
                validation_samples=0,
                feature_importance=feature_importance,
                drift_score=0.0  # Will be updated by drift detector
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return ModelPerformance(model_id, datetime.now(), 0.0, 0.0, 0.0, 0.0, 0.0, n_samples, 0, {}, 0.0)

    def get_model_status(self) -> Dict:
        """Get status of all models"""
        try:
            status = {
                'active_models': len(self.active_models),
                'ensembles': len(self.model_ensembles),
                'total_updates': self.update_count,
                'samples_processed': self.total_samples_processed,
                'buffer_size': len(self.sample_buffer),
                'models': {}
            }
            
            for model_id, model_info in self.active_models.items():
                perf = self.model_performance.get(model_id)
                status['models'][model_id] = {
                    'status': model_info['status'].value,
                    'last_update': model_info['last_update'].isoformat(),
                    'update_count': model_info['update_count'],
                    'accuracy': perf.accuracy if perf else 0.0,
                    'f1_score': perf.f1_score if perf else 0.0
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return {'error': str(e)}


class DriftDetector:
    """Concept drift detection component"""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_data = None
        self.current_status = DriftStatus.STABLE
        
    async def detect_drift(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Detect concept drift in new data"""
        try:
            if self.reference_data is None:
                self.reference_data = {'X': X, 'y': y}
                return False
            
            # Simple drift detection based on feature distribution changes
            drift_score = self._calculate_drift_score(X, self.reference_data['X'])
            
            if drift_score > self.threshold * 2:
                self.current_status = DriftStatus.SEVERE_DRIFT
                return True
            elif drift_score > self.threshold:
                self.current_status = DriftStatus.DRIFT_DETECTED
                return True
            elif drift_score > self.threshold * 0.5:
                self.current_status = DriftStatus.WARNING
                return False
            else:
                self.current_status = DriftStatus.STABLE
                return False
                
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return False
    
    def _calculate_drift_score(self, X_new: np.ndarray, X_ref: np.ndarray) -> float:
        """Calculate drift score between new and reference data"""
        try:
            # Simple statistical drift detection
            drift_scores = []
            
            for i in range(min(X_new.shape[1], X_ref.shape[1])):
                new_mean = np.mean(X_new[:, i])
                ref_mean = np.mean(X_ref[:, i])
                new_std = np.std(X_new[:, i])
                ref_std = np.std(X_ref[:, i])
                
                # Normalized difference in means and stds
                mean_diff = abs(new_mean - ref_mean) / (ref_std + 1e-8)
                std_diff = abs(new_std - ref_std) / (ref_std + 1e-8)
                
                drift_scores.append(mean_diff + std_diff)
            
            return np.mean(drift_scores) if drift_scores else 0.0
            
        except Exception:
            return 0.0
    
    def get_current_status(self) -> DriftStatus:
        """Get current drift status"""
        return self.current_status


class FeatureEngineer:
    """Feature engineering component"""
    
    def __init__(self):
        self.feature_cache = {}
        
    async def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data"""
        try:
            X_engineered = X.copy()
            
            # Add basic technical features if not present
            numeric_columns = X_engineered.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                # Rolling statistics
                if len(X_engineered) > 5:
                    X_engineered[f'{col}_ma5'] = X_engineered[col].rolling(5, min_periods=1).mean()
                    X_engineered[f'{col}_std5'] = X_engineered[col].rolling(5, min_periods=1).std().fillna(0)
                
                # Lag features
                if len(X_engineered) > 1:
                    X_engineered[f'{col}_lag1'] = X_engineered[col].shift(1).fillna(X_engineered[col].iloc[0])
            
            # Fill any remaining NaN values
            X_engineered = X_engineered.fillna(0)
            
            return X_engineered
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return X
