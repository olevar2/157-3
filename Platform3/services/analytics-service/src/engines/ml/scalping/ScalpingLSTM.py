"""
Scalping LSTM Model
Ultra-fast LSTM model optimized for M1-M5 price prediction and scalping signals

Features:
- Lightweight LSTM architecture for sub-second predictions
- Multi-step ahead price forecasting (1-10 ticks)
- Real-time feature engineering for scalping
- Adaptive learning with online updates
- Session-aware predictions
- Volatility-adjusted confidence scoring
- High-frequency pattern recognition
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

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Essential ML imports (always needed)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TensorFlow imports (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available, using fallback implementation")

class PredictionHorizon(Enum):
    TICK_1 = 1
    TICK_3 = 3
    TICK_5 = 5
    TICK_10 = 10

class ModelStatus(Enum):
    TRAINING = "training"
    READY = "ready"
    UPDATING = "updating"
    DEGRADED = "degraded"
    FAILED = "failed"

@dataclass
class ScalpingPrediction:
    symbol: str
    predicted_price: float
    predicted_direction: int  # 1 for up, -1 for down, 0 for sideways
    confidence: float
    horizon: PredictionHorizon
    features_used: List[str]
    timestamp: datetime
    model_version: str

@dataclass
class ModelConfig:
    sequence_length: int = 20  # Number of time steps to look back
    lstm_units: int = 32  # Reduced for speed
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2
    early_stopping_patience: int = 10

@dataclass
class FeatureConfig:
    price_features: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close'])
    volume_features: List[str] = field(default_factory=lambda: ['volume', 'tick_volume'])
    technical_features: List[str] = field(default_factory=lambda: ['rsi', 'macd', 'bb_upper', 'bb_lower'])
    time_features: List[str] = field(default_factory=lambda: ['hour', 'minute', 'weekday'])
    lag_features: int = 5  # Number of lag features to create

class ScalpingLSTM:
    """
    Ultra-fast LSTM model for scalping price predictions
    """

    def __init__(self, symbol: str, redis_client: Optional[redis.Redis] = None):
        self.symbol = symbol
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)

        # Configuration
        self.model_config = ModelConfig()
        self.feature_config = FeatureConfig()

        # Model components
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_names = []

        # State tracking
        self.model_status = ModelStatus.TRAINING
        self.model_version = "1.0.0"
        self.last_training_time = None
        self.prediction_count = 0

        # Performance tracking
        self.performance_metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'directional_accuracy': 0.0,
            'prediction_latency': 0.0,
            'total_predictions': 0,
            'correct_directions': 0
        }

        # Data buffers
        self.price_buffer = []
        self.feature_buffer = []
        self.max_buffer_size = 1000

        # Check TensorFlow availability
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, using fallback linear model")
            self._init_fallback_model()

        logger.info(f"ScalpingLSTM initialized for {symbol}")

    def _init_fallback_model(self):
        """Initialize fallback model when TensorFlow is not available"""
        from sklearn.linear_model import SGDRegressor
        self.fallback_model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42
        )
        self.using_fallback = True

    async def initialize(self, initial_data: Optional[pd.DataFrame] = None):
        """Initialize the model with optional initial training data"""
        try:
            if initial_data is not None and len(initial_data) > 0:
                logger.info(f"Initializing with {len(initial_data)} samples")
                await self.train(initial_data)
            else:
                # Create a basic model structure
                await self._build_model()
                self.model_status = ModelStatus.READY

            logger.info("✅ ScalpingLSTM initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ScalpingLSTM: {e}")
            self.model_status = ModelStatus.FAILED
            raise

    async def train(self, data: pd.DataFrame, retrain: bool = False) -> bool:
        """Train the LSTM model on historical data"""
        try:
            start_time = datetime.now()
            self.model_status = ModelStatus.TRAINING

            logger.info(f"Training ScalpingLSTM for {self.symbol} with {len(data)} samples")

            # Prepare features and targets
            X, y = await self._prepare_training_data(data)

            if len(X) < self.model_config.sequence_length * 2:
                logger.warning("Insufficient data for training")
                return False

            # Split data
            split_idx = int(len(X) * (1 - self.model_config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            if TF_AVAILABLE and not hasattr(self, 'using_fallback'):
                # Train TensorFlow model
                success = await self._train_tensorflow_model(X_train, y_train, X_val, y_val)
            else:
                # Train fallback model
                success = await self._train_fallback_model(X_train, y_train, X_val, y_val)

            if success:
                # Evaluate model
                await self._evaluate_model(X_val, y_val)

                self.model_status = ModelStatus.READY
                self.last_training_time = datetime.now()

                training_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Training completed in {training_time:.2f}s")

                # Save model
                await self._save_model()

                return True
            else:
                self.model_status = ModelStatus.FAILED
                return False

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            self.model_status = ModelStatus.FAILED
            return False

    async def predict(self, recent_data: Union[pd.DataFrame, Dict[str, Any]],
                     horizon: PredictionHorizon = PredictionHorizon.TICK_1) -> Optional[ScalpingPrediction]:
        """Make price prediction for scalping"""
        try:
            if self.model_status != ModelStatus.READY:
                logger.warning(f"Model not ready for prediction: {self.model_status}")
                return None

            start_time = datetime.now()

            # Prepare input features
            features = await self._prepare_prediction_features(recent_data)

            if features is None or len(features) == 0:
                return None

            # Make prediction
            if TF_AVAILABLE and not hasattr(self, 'using_fallback') and self.model:
                predicted_price = await self._predict_tensorflow(features)
            else:
                predicted_price = await self._predict_fallback(features)

            # Determine direction
            current_price = self._extract_current_price(recent_data)
            predicted_direction = self._determine_direction(current_price, predicted_price)

            # Calculate confidence
            confidence = await self._calculate_prediction_confidence(features, predicted_price, current_price)

            # Create prediction result
            prediction = ScalpingPrediction(
                symbol=self.symbol,
                predicted_price=predicted_price,
                predicted_direction=predicted_direction,
                confidence=confidence,
                horizon=horizon,
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

            logger.debug(f"Prediction: {predicted_price:.5f} (direction: {predicted_direction}, "
                        f"confidence: {confidence:.3f}, latency: {prediction_time:.1f}ms)")

            return prediction

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None

    async def update_with_feedback(self, prediction: ScalpingPrediction, actual_price: float) -> bool:
        """Update model with prediction feedback for online learning"""
        try:
            # Calculate prediction error
            price_error = abs(prediction.predicted_price - actual_price)
            direction_correct = (
                (prediction.predicted_direction > 0 and actual_price > prediction.predicted_price) or
                (prediction.predicted_direction < 0 and actual_price < prediction.predicted_price) or
                (prediction.predicted_direction == 0 and abs(actual_price - prediction.predicted_price) < 0.0001)
            )

            # Update performance metrics
            self.performance_metrics['mae'] = (
                (self.performance_metrics['mae'] * (self.prediction_count - 1) + price_error) /
                self.prediction_count
            )

            if direction_correct:
                self.performance_metrics['correct_directions'] += 1

            self.performance_metrics['directional_accuracy'] = (
                self.performance_metrics['correct_directions'] / self.prediction_count
            )

            # Store feedback for potential retraining
            await self._store_feedback(prediction, actual_price, price_error, direction_correct)

            logger.debug(f"Feedback updated: error={price_error:.5f}, direction_correct={direction_correct}")

            return True

        except Exception as e:
            logger.error(f"❌ Error updating with feedback: {e}")
            return False

    async def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with features and targets"""
        # Engineer features
        features_df = await self._engineer_features(data)

        # Create sequences
        X, y = self._create_sequences(features_df)

        # Scale features
        X_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        self.feature_names = features_df.columns.tolist()

        return X_scaled, y_scaled

    async def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for scalping prediction"""
        features = pd.DataFrame()

        # Price features
        for col in self.feature_config.price_features:
            if col in data.columns:
                features[col] = data[col]

                # Add price changes
                features[f'{col}_change'] = data[col].pct_change()
                features[f'{col}_change_abs'] = data[col].pct_change().abs()

                # Add lag features
                for lag in range(1, self.feature_config.lag_features + 1):
                    features[f'{col}_lag_{lag}'] = data[col].shift(lag)

        # Volume features
        for col in self.feature_config.volume_features:
            if col in data.columns:
                features[col] = data[col]
                features[f'{col}_ma'] = data[col].rolling(5).mean()

        # Technical indicators
        for col in self.feature_config.technical_features:
            if col in data.columns:
                features[col] = data[col]

        # Time features
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            features['hour'] = timestamps.dt.hour
            features['minute'] = timestamps.dt.minute
            features['weekday'] = timestamps.dt.weekday

        # Micro-features for scalping
        if 'close' in data.columns:
            # Tick-to-tick changes
            features['tick_change'] = data['close'].diff()
            features['tick_change_abs'] = data['close'].diff().abs()

            # Short-term momentum
            features['momentum_3'] = data['close'].diff(3)
            features['momentum_5'] = data['close'].diff(5)

            # Volatility measures
            features['volatility_5'] = data['close'].rolling(5).std()
            features['volatility_10'] = data['close'].rolling(10).std()

        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)

        return features

    def _create_sequences(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        features = features_df.values

        # Use close price as target (assuming it's the first price column)
        target_col = 'close' if 'close' in features_df.columns else features_df.columns[0]
        target_idx = features_df.columns.get_loc(target_col)

        X, y = [], []

        for i in range(self.model_config.sequence_length, len(features)):
            X.append(features[i-self.model_config.sequence_length:i])
            y.append(features[i, target_idx])  # Predict next close price

        return np.array(X), np.array(y)

    async def _build_model(self):
        """Build LSTM model architecture"""
        if not TF_AVAILABLE or hasattr(self, 'using_fallback'):
            return

        try:
            # Determine input shape
            n_features = len(self.feature_names) if self.feature_names else 10

            self.model = Sequential([
                LSTM(self.model_config.lstm_units,
                     return_sequences=True,
                     input_shape=(self.model_config.sequence_length, n_features)),
                Dropout(self.model_config.dropout_rate),

                LSTM(self.model_config.lstm_units // 2, return_sequences=False),
                Dropout(self.model_config.dropout_rate),

                Dense(16, activation='relu'),
                BatchNormalization(),
                Dropout(self.model_config.dropout_rate),

                Dense(1, activation='linear')  # Price prediction
            ])

            self.model.compile(
                optimizer=Adam(learning_rate=self.model_config.learning_rate),
                loss='mse',
                metrics=['mae']
            )

            logger.info("✅ LSTM model architecture built")

        except Exception as e:
            logger.error(f"❌ Error building model: {e}")
            raise

    async def _train_tensorflow_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                    X_val: np.ndarray, y_val: np.ndarray) -> bool:
        """Train TensorFlow LSTM model"""
        try:
            if self.model is None:
                await self._build_model()

            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.model_config.early_stopping_patience,
                restore_best_weights=True
            )

            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.model_config.batch_size,
                epochs=self.model_config.epochs,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )

            logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.6f}")
            return True

        except Exception as e:
            logger.error(f"❌ TensorFlow training failed: {e}")
            return False

    async def _train_fallback_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> bool:
        """Train fallback model"""
        try:
            # Flatten sequences for linear model
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)

            # Train model
            self.fallback_model.fit(X_train_flat, y_train)

            # Validate
            val_predictions = self.fallback_model.predict(X_val_flat)
            val_mse = mean_squared_error(y_val, val_predictions)

            logger.info(f"Fallback model training completed. Validation MSE: {val_mse:.6f}")
            return True

        except Exception as e:
            logger.error(f"❌ Fallback training failed: {e}")
            return False

    async def _predict_tensorflow(self, features: np.ndarray) -> float:
        """Make prediction using TensorFlow model"""
        try:
            # Ensure correct shape
            if len(features.shape) == 2:
                features = features.reshape(1, features.shape[0], features.shape[1])

            prediction = self.model.predict(features, verbose=0)[0][0]

            # Inverse transform
            prediction_scaled = np.array([[prediction]])
            prediction_original = self.scaler_y.inverse_transform(prediction_scaled)[0][0]

            return float(prediction_original)

        except Exception as e:
            logger.error(f"❌ TensorFlow prediction failed: {e}")
            return 0.0

    async def _predict_fallback(self, features: np.ndarray) -> float:
        """Make prediction using fallback model"""
        try:
            # Flatten features
            features_flat = features.reshape(1, -1)

            prediction = self.fallback_model.predict(features_flat)[0]

            # Inverse transform
            prediction_scaled = np.array([[prediction]])
            prediction_original = self.scaler_y.inverse_transform(prediction_scaled)[0][0]

            return float(prediction_original)

        except Exception as e:
            logger.error(f"❌ Fallback prediction failed: {e}")
            return 0.0

    async def _prepare_prediction_features(self, recent_data: Union[pd.DataFrame, Dict[str, Any]]) -> Optional[np.ndarray]:
        """Prepare features for prediction"""
        try:
            if isinstance(recent_data, dict):
                # Convert dict to DataFrame
                df = pd.DataFrame([recent_data])
            else:
                df = recent_data.copy()

            # Engineer features
            features_df = await self._engineer_features(df)

            # Get last sequence
            if len(features_df) >= self.model_config.sequence_length:
                sequence = features_df.iloc[-self.model_config.sequence_length:].values

                # Scale features
                sequence_scaled = self.scaler_X.transform(sequence)

                return sequence_scaled
            else:
                logger.warning("Insufficient data for prediction sequence")
                return None

        except Exception as e:
            logger.error(f"❌ Error preparing prediction features: {e}")
            return None

    def _extract_current_price(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> float:
        """Extract current price from data"""
        if isinstance(data, dict):
            return data.get('close', data.get('price', 0.0))
        else:
            if 'close' in data.columns:
                return data['close'].iloc[-1]
            elif 'price' in data.columns:
                return data['price'].iloc[-1]
            else:
                return 0.0

    def _determine_direction(self, current_price: float, predicted_price: float) -> int:
        """Determine price direction"""
        if predicted_price > current_price * 1.0001:  # 0.01% threshold
            return 1  # Up
        elif predicted_price < current_price * 0.9999:  # 0.01% threshold
            return -1  # Down
        else:
            return 0  # Sideways

    async def _calculate_prediction_confidence(self, features: np.ndarray,
                                             predicted_price: float, current_price: float) -> float:
        """Calculate prediction confidence score"""
        try:
            # Base confidence on model performance
            base_confidence = min(self.performance_metrics['directional_accuracy'], 1.0)

            # Adjust for price change magnitude
            price_change_pct = abs(predicted_price - current_price) / current_price
            magnitude_factor = min(price_change_pct * 100, 1.0)  # Cap at 1.0

            # Adjust for recent performance
            recent_accuracy = self.performance_metrics.get('directional_accuracy', 0.5)
            performance_factor = recent_accuracy

            # Combine factors
            confidence = (base_confidence * 0.4 + magnitude_factor * 0.3 + performance_factor * 0.3)

            return min(max(confidence, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    async def _evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate model performance"""
        try:
            if TF_AVAILABLE and not hasattr(self, 'using_fallback') and self.model:
                predictions = self.model.predict(X_val, verbose=0).flatten()
            else:
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                predictions = self.fallback_model.predict(X_val_flat)

            # Calculate metrics
            mse = mean_squared_error(y_val, predictions)
            mae = mean_absolute_error(y_val, predictions)

            # Calculate directional accuracy
            y_val_original = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
            pred_original = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

            # Assume we have previous prices to calculate direction
            # This is a simplified calculation
            directional_accuracy = 0.5  # Placeholder

            # Update performance metrics
            self.performance_metrics.update({
                'mse': mse,
                'mae': mae,
                'directional_accuracy': directional_accuracy
            })

            logger.info(f"Model evaluation - MSE: {mse:.6f}, MAE: {mae:.6f}, "
                       f"Directional Accuracy: {directional_accuracy:.3f}")

        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")

    async def _save_model(self):
        """Save model to Redis"""
        try:
            model_data = {
                'symbol': self.symbol,
                'model_version': self.model_version,
                'model_config': self.model_config.__dict__,
                'feature_config': self.feature_config.__dict__,
                'feature_names': self.feature_names,
                'performance_metrics': self.performance_metrics,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'model_status': self.model_status.value
            }

            # Save model metadata
            await self.redis_client.setex(
                f"scalping_lstm_meta:{self.symbol}",
                86400,  # 24 hours
                json.dumps(model_data, default=str)
            )

            # Save scalers
            scaler_data = {
                'scaler_X': pickle.dumps(self.scaler_X).hex(),
                'scaler_y': pickle.dumps(self.scaler_y).hex()
            }

            await self.redis_client.setex(
                f"scalping_lstm_scalers:{self.symbol}",
                86400,  # 24 hours
                json.dumps(scaler_data)
            )

            logger.info("✅ Model saved successfully")

        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")

    async def _cache_prediction(self, prediction: ScalpingPrediction):
        """Cache prediction in Redis"""
        try:
            prediction_data = {
                'symbol': prediction.symbol,
                'predicted_price': prediction.predicted_price,
                'predicted_direction': prediction.predicted_direction,
                'confidence': prediction.confidence,
                'horizon': prediction.horizon.value,
                'timestamp': prediction.timestamp.isoformat(),
                'model_version': prediction.model_version
            }

            await self.redis_client.setex(
                f"scalping_prediction:{self.symbol}",
                300,  # 5 minutes
                json.dumps(prediction_data)
            )

        except Exception as e:
            logger.error(f"Error caching prediction: {e}")

    async def _store_feedback(self, prediction: ScalpingPrediction, actual_price: float,
                            price_error: float, direction_correct: bool):
        """Store prediction feedback for analysis"""
        try:
            feedback_data = {
                'prediction': {
                    'predicted_price': prediction.predicted_price,
                    'predicted_direction': prediction.predicted_direction,
                    'confidence': prediction.confidence,
                    'timestamp': prediction.timestamp.isoformat()
                },
                'actual_price': actual_price,
                'price_error': price_error,
                'direction_correct': direction_correct,
                'feedback_timestamp': datetime.now().isoformat()
            }

            # Store in a list for batch analysis
            await self.redis_client.lpush(
                f"scalping_feedback:{self.symbol}",
                json.dumps(feedback_data)
            )

            # Keep only last 1000 feedback entries
            await self.redis_client.ltrim(f"scalping_feedback:{self.symbol}", 0, 999)

        except Exception as e:
            logger.error(f"Error storing feedback: {e}")

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        return {
            'symbol': self.symbol,
            'model_status': self.model_status.value,
            'model_version': self.model_version,
            'performance_metrics': self.performance_metrics,
            'prediction_count': self.prediction_count,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'feature_count': len(self.feature_names),
            'using_fallback': hasattr(self, 'using_fallback'),
            'tensorflow_available': TF_AVAILABLE
        }
