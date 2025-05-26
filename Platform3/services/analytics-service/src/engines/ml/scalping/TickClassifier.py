"""
Tick Direction Classifier
Ultra-fast ML classifier for next tick direction prediction in scalping

Features:
- Binary classification for next tick direction (up/down)
- Sub-millisecond prediction latency
- Real-time feature engineering from tick data
- Ensemble of lightweight classifiers
- Adaptive learning with online updates
- Confidence scoring for predictions
- Session-aware prediction adjustments
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
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, using fallback implementation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TickDirection(Enum):
    UP = 1
    DOWN = -1
    NEUTRAL = 0

class ClassifierType(Enum):
    ENSEMBLE = "ensemble"
    RANDOM_FOREST = "random_forest"
    SGD = "sgd"
    LOGISTIC = "logistic"
    NAIVE_BAYES = "naive_bayes"

@dataclass
class TickPrediction:
    symbol: str
    predicted_direction: TickDirection
    confidence: float
    probability_up: float
    probability_down: float
    features_used: List[str]
    timestamp: datetime
    model_version: str
    prediction_horizon: int = 1  # Number of ticks ahead

@dataclass
class ClassifierConfig:
    ensemble_size: int = 3
    max_features: int = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    learning_rate: float = 0.01
    regularization: float = 0.01
    confidence_threshold: float = 0.6

@dataclass
class TickFeatures:
    price_change: float
    volume_ratio: float
    bid_ask_spread: float
    tick_size: float
    momentum_1: float
    momentum_3: float
    momentum_5: float
    volatility: float
    session_factor: float
    time_factor: float

class TickClassifier:
    """
    Ultra-fast tick direction classifier for scalping
    """
    
    def __init__(self, symbol: str, redis_client: Optional[redis.Redis] = None):
        self.symbol = symbol
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Configuration
        self.config = ClassifierConfig()
        self.classifier_type = ClassifierType.ENSEMBLE
        
        # Model components
        self.ensemble_model = None
        self.individual_models = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        
        # State tracking
        self.model_version = "1.0.0"
        self.last_training_time = None
        self.prediction_count = 0
        self.is_trained = False
        
        # Performance tracking
        self.performance_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'prediction_latency': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0,
            'confidence_accuracy': 0.0
        }
        
        # Data buffers for online learning
        self.tick_buffer = []
        self.feature_buffer = []
        self.label_buffer = []
        self.max_buffer_size = 1000
        
        # Feature engineering components
        self.price_history = []
        self.volume_history = []
        self.spread_history = []
        
        # Check sklearn availability
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, using fallback implementation")
            self._init_fallback_classifier()
        
        logger.info(f"TickClassifier initialized for {symbol}")

    def _init_fallback_classifier(self):
        """Initialize fallback classifier when sklearn is not available"""
        # Simple rule-based classifier
        self.fallback_weights = {
            'price_change': 0.3,
            'momentum_1': 0.25,
            'momentum_3': 0.2,
            'volume_ratio': 0.15,
            'volatility': 0.1
        }
        self.using_fallback = True

    async def initialize(self, initial_data: Optional[pd.DataFrame] = None):
        """Initialize the classifier with optional initial training data"""
        try:
            if initial_data is not None and len(initial_data) > 0:
                logger.info(f"Initializing with {len(initial_data)} tick samples")
                await self.train(initial_data)
            else:
                # Create basic model structure
                await self._build_ensemble()
                self.is_trained = False
            
            logger.info("✅ TickClassifier initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TickClassifier: {e}")
            raise

    async def train(self, tick_data: pd.DataFrame, retrain: bool = False) -> bool:
        """Train the tick classifier on historical tick data"""
        try:
            start_time = datetime.now()
            
            logger.info(f"Training TickClassifier for {self.symbol} with {len(tick_data)} ticks")
            
            # Prepare features and labels
            X, y = await self._prepare_training_data(tick_data)
            
            if len(X) < 50:  # Minimum samples for training
                logger.warning("Insufficient tick data for training")
                return False
            
            # Split data for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            if SKLEARN_AVAILABLE and not hasattr(self, 'using_fallback'):
                # Train sklearn models
                success = await self._train_sklearn_models(X_train, y_train, X_val, y_val)
            else:
                # Train fallback model
                success = await self._train_fallback_model(X_train, y_train, X_val, y_val)
            
            if success:
                # Evaluate model
                await self._evaluate_model(X_val, y_val)
                
                self.is_trained = True
                self.last_training_time = datetime.now()
                
                training_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Training completed in {training_time:.2f}s")
                
                # Save model
                await self._save_model()
                
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False

    async def predict_next_tick(self, current_tick_data: Dict[str, Any]) -> Optional[TickPrediction]:
        """Predict next tick direction"""
        try:
            if not self.is_trained:
                logger.warning("Classifier not trained yet")
                return None
            
            start_time = datetime.now()
            
            # Extract features from current tick
            features = await self._extract_tick_features(current_tick_data)
            
            if features is None:
                return None
            
            # Make prediction
            if SKLEARN_AVAILABLE and not hasattr(self, 'using_fallback') and self.ensemble_model:
                direction, confidence, prob_up, prob_down = await self._predict_sklearn(features)
            else:
                direction, confidence, prob_up, prob_down = await self._predict_fallback(features)
            
            # Create prediction result
            prediction = TickPrediction(
                symbol=self.symbol,
                predicted_direction=direction,
                confidence=confidence,
                probability_up=prob_up,
                probability_down=prob_down,
                features_used=self.feature_names,
                timestamp=datetime.now(),
                model_version=self.model_version
            )
            
            # Update performance tracking
            prediction_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_metrics['prediction_latency'] = prediction_time
            self.performance_metrics['total_predictions'] += 1
            self.prediction_count += 1
            
            # Cache prediction
            await self._cache_prediction(prediction)
            
            logger.debug(f"Tick prediction: {direction.value} (confidence: {confidence:.3f}, "
                        f"latency: {prediction_time:.2f}ms)")
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None

    async def update_with_feedback(self, prediction: TickPrediction, actual_direction: TickDirection) -> bool:
        """Update classifier with prediction feedback"""
        try:
            # Check if prediction was correct
            prediction_correct = (prediction.predicted_direction == actual_direction)
            
            # Update performance metrics
            if prediction_correct:
                self.performance_metrics['correct_predictions'] += 1
            
            self.performance_metrics['accuracy'] = (
                self.performance_metrics['correct_predictions'] / 
                self.performance_metrics['total_predictions']
            )
            
            # Store feedback for potential retraining
            await self._store_feedback(prediction, actual_direction, prediction_correct)
            
            # Online learning update if enabled
            if len(self.tick_buffer) > 0:
                await self._online_learning_update()
            
            logger.debug(f"Feedback updated: correct={prediction_correct}, "
                        f"accuracy={self.performance_metrics['accuracy']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating with feedback: {e}")
            return False

    async def _prepare_training_data(self, tick_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with features and labels"""
        features_list = []
        labels_list = []
        
        # Sort by timestamp
        tick_data = tick_data.sort_values('timestamp')
        
        # Calculate price changes for labels
        tick_data['price_change'] = tick_data['price'].diff()
        tick_data['next_direction'] = np.where(
            tick_data['price_change'].shift(-1) > 0, 1,
            np.where(tick_data['price_change'].shift(-1) < 0, -1, 0)
        )
        
        # Extract features for each tick
        for i in range(len(tick_data) - 1):  # Exclude last tick (no next direction)
            current_tick = tick_data.iloc[i].to_dict()
            features = await self._extract_tick_features(current_tick, tick_data.iloc[:i+1])
            
            if features is not None:
                features_list.append(features)
                labels_list.append(tick_data.iloc[i]['next_direction'])
        
        if len(features_list) == 0:
            return np.array([]), np.array([])
        
        # Convert to arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Remove neutral labels for binary classification
        non_neutral_mask = y != 0
        X = X[non_neutral_mask]
        y = y[non_neutral_mask]
        
        # Scale features
        if len(X) > 0:
            X = self.scaler.fit_transform(X)
        
        return X, y

    async def _extract_tick_features(self, current_tick: Dict[str, Any], 
                                   history: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """Extract features from current tick data"""
        try:
            features = []
            
            # Basic price features
            price = current_tick.get('price', 0.0)
            volume = current_tick.get('volume', 0.0)
            bid = current_tick.get('bid', price)
            ask = current_tick.get('ask', price)
            
            # Price change from previous tick
            if len(self.price_history) > 0:
                price_change = (price - self.price_history[-1]) / self.price_history[-1]
            else:
                price_change = 0.0
            features.append(price_change)
            
            # Volume ratio
            if len(self.volume_history) > 0:
                avg_volume = np.mean(self.volume_history[-10:])
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            features.append(min(volume_ratio, 5.0))  # Cap at 5x
            
            # Bid-ask spread
            spread = (ask - bid) / price if price > 0 else 0.0
            features.append(spread)
            
            # Momentum features
            if len(self.price_history) >= 1:
                momentum_1 = (price - self.price_history[-1]) / price if price > 0 else 0.0
                features.append(momentum_1)
            else:
                features.append(0.0)
            
            if len(self.price_history) >= 3:
                momentum_3 = (price - self.price_history[-3]) / price if price > 0 else 0.0
                features.append(momentum_3)
            else:
                features.append(0.0)
            
            if len(self.price_history) >= 5:
                momentum_5 = (price - self.price_history[-5]) / price if price > 0 else 0.0
                features.append(momentum_5)
            else:
                features.append(0.0)
            
            # Volatility (rolling standard deviation)
            if len(self.price_history) >= 10:
                recent_prices = self.price_history[-10:] + [price]
                price_changes = np.diff(recent_prices)
                volatility = np.std(price_changes) / price if price > 0 else 0.0
                features.append(volatility)
            else:
                features.append(0.0)
            
            # Session factor (time-based)
            timestamp = current_tick.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            hour = timestamp.hour
            session_factor = self._get_session_factor(hour)
            features.append(session_factor)
            
            # Time factor (minute within hour)
            time_factor = timestamp.minute / 60.0
            features.append(time_factor)
            
            # Tick size (relative to recent range)
            if len(self.price_history) >= 20:
                recent_high = max(self.price_history[-20:])
                recent_low = min(self.price_history[-20:])
                if recent_high != recent_low:
                    tick_size = abs(price_change) / (recent_high - recent_low)
                else:
                    tick_size = 0.0
                features.append(tick_size)
            else:
                features.append(0.0)
            
            # Update history buffers
            self.price_history.append(price)
            self.volume_history.append(volume)
            if ask > bid:
                self.spread_history.append(spread)
            
            # Maintain buffer sizes
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-100:]
            if len(self.volume_history) > 100:
                self.volume_history = self.volume_history[-100:]
            if len(self.spread_history) > 100:
                self.spread_history = self.spread_history[-100:]
            
            # Store feature names
            if not self.feature_names:
                self.feature_names = [
                    'price_change', 'volume_ratio', 'bid_ask_spread',
                    'momentum_1', 'momentum_3', 'momentum_5',
                    'volatility', 'session_factor', 'time_factor', 'tick_size'
                ]
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting tick features: {e}")
            return None

    def _get_session_factor(self, hour: int) -> float:
        """Get session factor based on hour (UTC)"""
        # London session (8-16 UTC) - high activity
        if 8 <= hour < 16:
            return 1.0
        # New York session (13-21 UTC) - high activity
        elif 13 <= hour < 21:
            return 1.0
        # Overlap (13-16 UTC) - highest activity
        elif 13 <= hour < 16:
            return 1.2
        # Asian session (0-8 UTC) - medium activity
        elif 0 <= hour < 8:
            return 0.8
        # Quiet period
        else:
            return 0.6

    async def _build_ensemble(self):
        """Build ensemble of classifiers"""
        if not SKLEARN_AVAILABLE or hasattr(self, 'using_fallback'):
            return
        
        try:
            # Create individual classifiers
            rf_classifier = RandomForestClassifier(
                n_estimators=10,  # Small for speed
                max_depth=5,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=42
            )
            
            sgd_classifier = SGDClassifier(
                loss='log_loss',
                learning_rate='adaptive',
                eta0=self.config.learning_rate,
                alpha=self.config.regularization,
                random_state=42
            )
            
            nb_classifier = GaussianNB()
            
            # Create ensemble
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', rf_classifier),
                    ('sgd', sgd_classifier),
                    ('nb', nb_classifier)
                ],
                voting='soft'  # Use probabilities
            )
            
            # Store individual models
            self.individual_models = {
                'random_forest': rf_classifier,
                'sgd': sgd_classifier,
                'naive_bayes': nb_classifier
            }
            
            logger.info("✅ Ensemble classifier built")
            
        except Exception as e:
            logger.error(f"❌ Error building ensemble: {e}")
            raise

    async def _train_sklearn_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> bool:
        """Train sklearn ensemble models"""
        try:
            if self.ensemble_model is None:
                await self._build_ensemble()
            
            # Train ensemble
            self.ensemble_model.fit(X_train, y_train)
            
            # Validate
            val_predictions = self.ensemble_model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            
            logger.info(f"Ensemble training completed. Validation accuracy: {val_accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sklearn training failed: {e}")
            return False

    async def _train_fallback_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> bool:
        """Train fallback rule-based model"""
        try:
            # Simple rule-based training (adjust weights based on feature importance)
            if len(X_train) > 0 and len(self.feature_names) > 0:
                # Calculate simple correlations
                for i, feature_name in enumerate(self.feature_names):
                    if i < X_train.shape[1]:
                        correlation = np.corrcoef(X_train[:, i], y_train)[0, 1]
                        if not np.isnan(correlation):
                            self.fallback_weights[feature_name] = abs(correlation)
            
            logger.info("Fallback model training completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fallback training failed: {e}")
            return False

    async def _predict_sklearn(self, features: np.ndarray) -> Tuple[TickDirection, float, float, float]:
        """Make prediction using sklearn ensemble"""
        try:
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction and probabilities
            prediction = self.ensemble_model.predict(features_scaled)[0]
            probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
            
            # Map to TickDirection
            direction = TickDirection.UP if prediction > 0 else TickDirection.DOWN
            
            # Calculate confidence and probabilities
            if len(probabilities) == 2:
                prob_down, prob_up = probabilities  # Assuming classes [-1, 1]
                confidence = max(prob_up, prob_down)
            else:
                prob_up = prob_down = 0.5
                confidence = 0.5
            
            return direction, confidence, prob_up, prob_down
            
        except Exception as e:
            logger.error(f"❌ Sklearn prediction failed: {e}")
            return TickDirection.UP, 0.5, 0.5, 0.5

    async def _predict_fallback(self, features: np.ndarray) -> Tuple[TickDirection, float, float, float]:
        """Make prediction using fallback rule-based model"""
        try:
            # Simple weighted sum of features
            score = 0.0
            total_weight = 0.0
            
            for i, feature_name in enumerate(self.feature_names):
                if i < len(features) and feature_name in self.fallback_weights:
                    weight = self.fallback_weights[feature_name]
                    score += features[i] * weight
                    total_weight += abs(weight)
            
            if total_weight > 0:
                normalized_score = score / total_weight
            else:
                normalized_score = 0.0
            
            # Convert to direction and confidence
            direction = TickDirection.UP if normalized_score > 0 else TickDirection.DOWN
            confidence = min(abs(normalized_score) * 2, 1.0)  # Scale to 0-1
            
            # Simple probability calculation
            prob_up = 0.5 + normalized_score / 2
            prob_down = 1.0 - prob_up
            
            return direction, confidence, prob_up, prob_down
            
        except Exception as e:
            logger.error(f"❌ Fallback prediction failed: {e}")
            return TickDirection.UP, 0.5, 0.5, 0.5

    async def _evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate model performance"""
        try:
            if SKLEARN_AVAILABLE and not hasattr(self, 'using_fallback') and self.ensemble_model:
                predictions = self.ensemble_model.predict(X_val)
                probabilities = self.ensemble_model.predict_proba(X_val)
            else:
                # Fallback evaluation
                predictions = []
                for features in X_val:
                    direction, _, _, _ = await self._predict_fallback(features)
                    predictions.append(direction.value)
                predictions = np.array(predictions)
                probabilities = None
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(y_val, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_val, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_val, predictions, average='weighted', zero_division=0)
            
            # Update performance metrics
            self.performance_metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            logger.info(f"Model evaluation - Accuracy: {accuracy:.3f}, "
                       f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")

    async def _online_learning_update(self):
        """Perform online learning update with recent data"""
        try:
            if len(self.feature_buffer) < 10:  # Need minimum samples
                return
            
            # Get recent features and labels
            recent_X = np.array(self.feature_buffer[-10:])
            recent_y = np.array(self.label_buffer[-10:])
            
            # Scale features
            recent_X_scaled = self.scaler.transform(recent_X)
            
            # Partial fit for SGD classifier
            if (SKLEARN_AVAILABLE and not hasattr(self, 'using_fallback') and 
                hasattr(self.individual_models.get('sgd'), 'partial_fit')):
                
                sgd_model = self.individual_models['sgd']
                sgd_model.partial_fit(recent_X_scaled, recent_y)
                
                logger.debug("Online learning update completed")
            
        except Exception as e:
            logger.error(f"Error in online learning update: {e}")

    async def _save_model(self):
        """Save model to Redis"""
        try:
            model_data = {
                'symbol': self.symbol,
                'model_version': self.model_version,
                'config': self.config.__dict__,
                'feature_names': self.feature_names,
                'performance_metrics': self.performance_metrics,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'is_trained': self.is_trained
            }
            
            # Save model metadata
            await self.redis_client.setex(
                f"tick_classifier_meta:{self.symbol}",
                86400,  # 24 hours
                json.dumps(model_data, default=str)
            )
            
            # Save scaler
            if hasattr(self, 'scaler'):
                scaler_data = pickle.dumps(self.scaler).hex()
                await self.redis_client.setex(
                    f"tick_classifier_scaler:{self.symbol}",
                    86400,  # 24 hours
                    scaler_data
                )
            
            logger.info("✅ Tick classifier saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")

    async def _cache_prediction(self, prediction: TickPrediction):
        """Cache prediction in Redis"""
        try:
            prediction_data = {
                'symbol': prediction.symbol,
                'predicted_direction': prediction.predicted_direction.value,
                'confidence': prediction.confidence,
                'probability_up': prediction.probability_up,
                'probability_down': prediction.probability_down,
                'timestamp': prediction.timestamp.isoformat(),
                'model_version': prediction.model_version
            }
            
            await self.redis_client.setex(
                f"tick_prediction:{self.symbol}",
                60,  # 1 minute
                json.dumps(prediction_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching prediction: {e}")

    async def _store_feedback(self, prediction: TickPrediction, actual_direction: TickDirection,
                            prediction_correct: bool):
        """Store prediction feedback for analysis"""
        try:
            feedback_data = {
                'prediction': {
                    'predicted_direction': prediction.predicted_direction.value,
                    'confidence': prediction.confidence,
                    'timestamp': prediction.timestamp.isoformat()
                },
                'actual_direction': actual_direction.value,
                'prediction_correct': prediction_correct,
                'feedback_timestamp': datetime.now().isoformat()
            }
            
            # Store in a list for batch analysis
            await self.redis_client.lpush(
                f"tick_feedback:{self.symbol}",
                json.dumps(feedback_data)
            )
            
            # Keep only last 1000 feedback entries
            await self.redis_client.ltrim(f"tick_feedback:{self.symbol}", 0, 999)
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get classifier performance statistics"""
        return {
            'symbol': self.symbol,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'performance_metrics': self.performance_metrics,
            'prediction_count': self.prediction_count,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'feature_count': len(self.feature_names),
            'using_fallback': hasattr(self, 'using_fallback'),
            'sklearn_available': SKLEARN_AVAILABLE,
            'buffer_sizes': {
                'price_history': len(self.price_history),
                'volume_history': len(self.volume_history),
                'spread_history': len(self.spread_history)
            }
        }
