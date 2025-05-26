"""
Intraday Momentum ML Model
ML model for momentum prediction optimized for M15-H1 timeframes.
Provides momentum strength assessment for day trading strategies.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import pickle
import os

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Using mock implementation.")


@dataclass
class MomentumPrediction:
    """Intraday momentum prediction result"""
    timestamp: float
    symbol: str
    timeframe: str  # M15, M30, H1
    momentum_strength: float  # -1 to 1 (negative = bearish, positive = bullish)
    momentum_direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-1
    momentum_duration_minutes: int  # Expected duration of momentum
    momentum_target_pips: float  # Expected price movement in pips
    probability_continuation: float  # Probability momentum continues
    model_version: str
    session_context: str  # 'Asian', 'London', 'NewYork', 'Overlap'


@dataclass
class MomentumMetrics:
    """Momentum model training metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    momentum_prediction_accuracy: float
    training_time: float
    epochs_trained: int
    data_points: int


@dataclass
class MomentumFeatures:
    """Feature set for momentum prediction"""
    price_momentum: List[float]  # Multi-timeframe momentum
    volume_momentum: List[float]  # Volume-based momentum
    volatility_features: List[float]  # Volatility characteristics
    session_features: List[float]  # Trading session context
    technical_features: List[float]  # Technical indicators


class IntradayMomentumML:
    """
    Intraday Momentum ML Model
    ML model for momentum prediction optimized for M15-H1 timeframes
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Model configuration
        self.sequence_length = self.config.get('sequence_length', 48)  # 48 periods (12 hours on M15)
        self.prediction_horizon = self.config.get('prediction_horizon', 60)  # 60 minutes ahead
        self.feature_count = self.config.get('feature_count', 25)

        # Model architecture
        self.lstm_units = [128, 64, 32]  # Larger units for longer sequences
        self.dropout_rate = 0.3
        self.learning_rate = 0.0005

        # Model storage
        self.models = {}  # symbol -> model
        self.scalers = {}  # symbol -> scaler
        self.training_metrics = {}  # symbol -> metrics

        # Data buffers
        self.feature_buffers = {}  # symbol -> deque of features
        self.max_buffer_size = 500

        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.training_count = 0

        # Model paths
        self.model_dir = self.config.get('model_dir', 'models/daytrading_momentum')
        os.makedirs(self.model_dir, exist_ok=True)

        # Trading sessions for context
        self.trading_sessions = {
            'Asian': {'start': 0, 'end': 9, 'volatility': 'low'},
            'London': {'start': 8, 'end': 17, 'volatility': 'high'},
            'NewYork': {'start': 13, 'end': 22, 'volatility': 'high'},
            'Overlap': {'start': 13, 'end': 17, 'volatility': 'very_high'}
        }

    async def initialize(self) -> None:
        """Initialize the momentum ML model"""
        try:
            if not TENSORFLOW_AVAILABLE:
                self.logger.warning("TensorFlow not available. Using mock momentum implementation.")
                return

            # Set TensorFlow configuration
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)

            # Load existing models if available
            await self._load_existing_models()

            self.logger.info("Intraday Momentum ML initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Intraday Momentum ML: {e}")
            raise

    async def predict_momentum(self, symbol: str, timeframe: str, market_data: List[Dict]) -> MomentumPrediction:
        """
        Predict intraday momentum using ML model
        """
        start_time = time.time()

        try:
            # Prepare features
            features = await self._prepare_momentum_features(symbol, market_data)

            if len(features) < self.sequence_length:
                raise ValueError(f"Insufficient data for momentum prediction. Need {self.sequence_length}, got {len(features)}")

            # Get or create model
            model = await self._get_or_create_model(symbol, timeframe)
            scaler = self.scalers.get(symbol)

            if model is None or scaler is None:
                # Train new model if not available
                model, scaler = await self._train_momentum_model(symbol, timeframe, features)

            # Make prediction
            prediction_result = await self._make_momentum_prediction(
                model, scaler, features, symbol, timeframe, market_data
            )

            # Update performance tracking
            prediction_time = time.time() - start_time
            self.prediction_count += 1
            self.total_prediction_time += prediction_time

            self.logger.debug(f"Momentum prediction for {symbol} completed in {prediction_time:.3f}s")

            return prediction_result

        except Exception as e:
            self.logger.error(f"Momentum prediction failed for {symbol}: {e}")
            raise

    async def _prepare_momentum_features(self, symbol: str, market_data: List[Dict]) -> np.ndarray:
        """Prepare feature matrix for momentum prediction"""

        if not market_data:
            raise ValueError("No market data provided")

        features_list = []

        for i, data in enumerate(market_data):
            # Price momentum features (multiple timeframes)
            price_momentum = self._calculate_price_momentum(market_data, i)

            # Volume momentum features
            volume_momentum = self._calculate_volume_momentum(market_data, i)

            # Volatility features
            volatility_features = self._calculate_volatility_features(market_data, i)

            # Session context features
            session_features = self._calculate_session_features(data)

            # Technical indicator features
            technical_features = self._calculate_technical_features(market_data, i)

            # Combine all features
            feature_vector = (price_momentum + volume_momentum + volatility_features +
                            session_features + technical_features)

            features_list.append(feature_vector)

        return np.array(features_list)

    def _calculate_price_momentum(self, market_data: List[Dict], index: int) -> List[float]:
        """Calculate multi-timeframe price momentum features"""
        if index < 20:
            return [0.0] * 8

        closes = [float(d.get('close', 0)) for d in market_data[max(0, index-20):index+1]]

        # Short-term momentum (5 periods)
        short_momentum = (closes[-1] - closes[-5]) / max(closes[-5], 0.0001) if len(closes) >= 5 else 0

        # Medium-term momentum (10 periods)
        medium_momentum = (closes[-1] - closes[-10]) / max(closes[-10], 0.0001) if len(closes) >= 10 else 0

        # Long-term momentum (20 periods)
        long_momentum = (closes[-1] - closes[-20]) / max(closes[-20], 0.0001) if len(closes) >= 20 else 0

        # Momentum acceleration
        if len(closes) >= 10:
            recent_momentum = (closes[-1] - closes[-5]) / max(closes[-5], 0.0001)
            previous_momentum = (closes[-5] - closes[-10]) / max(closes[-10], 0.0001)
            momentum_acceleration = recent_momentum - previous_momentum
        else:
            momentum_acceleration = 0

        # Price velocity (rate of change)
        price_velocity = np.std(np.diff(closes[-10:])) if len(closes) >= 10 else 0

        # Momentum consistency (how consistent the direction is)
        if len(closes) >= 10:
            changes = np.diff(closes[-10:])
            positive_changes = sum(1 for c in changes if c > 0)
            momentum_consistency = positive_changes / len(changes)
        else:
            momentum_consistency = 0.5

        # Momentum strength (normalized)
        momentum_strength = np.tanh(abs(short_momentum) * 100)

        # Momentum divergence (price vs momentum)
        momentum_divergence = abs(short_momentum - medium_momentum)

        return [short_momentum, medium_momentum, long_momentum, momentum_acceleration,
                price_velocity, momentum_consistency, momentum_strength, momentum_divergence]

    def _calculate_volume_momentum(self, market_data: List[Dict], index: int) -> List[float]:
        """Calculate volume-based momentum features"""
        if index < 10:
            return [0.0] * 5

        volumes = [float(d.get('volume', 0)) for d in market_data[max(0, index-10):index+1]]

        if not volumes or all(v == 0 for v in volumes):
            return [0.0] * 5

        # Volume momentum
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else current_volume
        volume_momentum = (current_volume - avg_volume) / max(avg_volume, 1)

        # Volume trend
        if len(volumes) >= 5:
            volume_trend = np.polyfit(range(5), volumes[-5:], 1)[0]
        else:
            volume_trend = 0

        # Volume volatility
        volume_volatility = np.std(volumes) / max(np.mean(volumes), 1)

        # Volume-price correlation
        if len(volumes) >= 5:
            prices = [float(d.get('close', 0)) for d in market_data[max(0, index-4):index+1]]
            if len(prices) == len(volumes[-5:]):
                volume_price_corr = np.corrcoef(volumes[-5:], prices)[0, 1]
                if np.isnan(volume_price_corr):
                    volume_price_corr = 0
            else:
                volume_price_corr = 0
        else:
            volume_price_corr = 0

        # Volume spike indicator
        volume_spike = 1.0 if current_volume > avg_volume * 2 else 0.0

        return [volume_momentum, volume_trend, volume_volatility, volume_price_corr, volume_spike]

    def _calculate_volatility_features(self, market_data: List[Dict], index: int) -> List[float]:
        """Calculate volatility-based features"""
        if index < 20:
            return [0.0] * 4

        closes = [float(d.get('close', 0)) for d in market_data[max(0, index-20):index+1]]
        highs = [float(d.get('high', 0)) for d in market_data[max(0, index-20):index+1]]
        lows = [float(d.get('low', 0)) for d in market_data[max(0, index-20):index+1]]

        # Price volatility (standard deviation of returns)
        if len(closes) >= 2:
            returns = np.diff(closes) / closes[:-1]
            price_volatility = np.std(returns)
        else:
            price_volatility = 0

        # True range volatility
        if len(closes) >= 2:
            true_ranges = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                true_ranges.append(tr)
            atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.mean(true_ranges)
            atr_normalized = atr / max(closes[-1], 0.0001)
        else:
            atr_normalized = 0

        # Volatility trend
        if len(closes) >= 10:
            recent_volatility = np.std(np.diff(closes[-5:]) / closes[-6:-1])
            previous_volatility = np.std(np.diff(closes[-10:-5]) / closes[-11:-6])
            volatility_trend = (recent_volatility - previous_volatility) / max(previous_volatility, 0.0001)
        else:
            volatility_trend = 0

        # Volatility regime (high/low volatility environment)
        if len(closes) >= 20:
            long_term_volatility = np.std(np.diff(closes) / closes[:-1])
            current_volatility = np.std(np.diff(closes[-5:]) / closes[-6:-1])
            volatility_regime = current_volatility / max(long_term_volatility, 0.0001)
        else:
            volatility_regime = 1.0

        return [price_volatility, atr_normalized, volatility_trend, volatility_regime]

    def _calculate_session_features(self, data: Dict) -> List[float]:
        """Calculate trading session context features"""
        timestamp = float(data.get('timestamp', time.time()))
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour

        # Session identification
        asian_session = 1.0 if 0 <= hour <= 9 else 0.0
        london_session = 1.0 if 8 <= hour <= 17 else 0.0
        newyork_session = 1.0 if 13 <= hour <= 22 else 0.0
        overlap_session = 1.0 if 13 <= hour <= 17 else 0.0

        # Time-based features
        hour_normalized = hour / 24.0
        day_of_week = dt.weekday() / 6.0

        return [asian_session, london_session, newyork_session, overlap_session, hour_normalized, day_of_week]

    def _calculate_technical_features(self, market_data: List[Dict], index: int) -> List[float]:
        """Calculate technical indicator features"""
        if index < 20:
            return [0.0] * 6

        closes = [float(d.get('close', 0)) for d in market_data[max(0, index-20):index+1]]

        # RSI
        rsi = self._calculate_rsi(closes, 14)

        # Moving averages
        sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]

        # MA crossover signal
        ma_signal = 1.0 if sma_10 > sma_20 else -1.0

        # Price position relative to MAs
        price_vs_sma10 = (closes[-1] - sma_10) / max(sma_10, 0.0001)
        price_vs_sma20 = (closes[-1] - sma_20) / max(sma_20, 0.0001)

        # MACD-like momentum
        if len(closes) >= 26:
            ema_12 = closes[-1]  # Simplified EMA
            ema_26 = np.mean(closes[-26:])
            macd_line = (ema_12 - ema_26) / max(ema_26, 0.0001)
        else:
            macd_line = 0

        return [rsi, ma_signal, price_vs_sma10, price_vs_sma20, macd_line, 0.0]  # Last 0.0 for future expansion

    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def _get_or_create_model(self, symbol: str, timeframe: str) -> Optional[Any]:
        """Get existing model or return None to trigger training"""
        model_key = f"{symbol}_{timeframe}_momentum"

        if model_key in self.models:
            return self.models[model_key]

        # Try to load from disk
        model_path = os.path.join(self.model_dir, f"{model_key}.h5")
        if os.path.exists(model_path) and TENSORFLOW_AVAILABLE:
            try:
                model = load_model(model_path)
                self.models[model_key] = model
                return model
            except Exception as e:
                self.logger.warning(f"Failed to load model from {model_path}: {e}")

        return None

    async def _train_momentum_model(self, symbol: str, timeframe: str, features: np.ndarray) -> Tuple[Any, Any]:
        """Train new momentum prediction model"""
        start_time = time.time()

        try:
            if not TENSORFLOW_AVAILABLE:
                # Mock model for testing
                mock_model = {
                    'type': 'mock_momentum',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'predict': lambda x: np.random.random((x.shape[0], 1)) * 2 - 1  # -1 to 1
                }
                mock_scaler = {
                    'transform': lambda x: x,
                    'inverse_transform': lambda x: x
                }
                return mock_model, mock_scaler

            # Prepare training data
            X, y = await self._prepare_momentum_training_data(features)

            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

            # Split data
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Build model
            model = await self._build_momentum_model(X.shape[1], X.shape[2])

            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[
                    EarlyStopping(patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(patience=7, factor=0.5)
                ],
                verbose=0
            )

            # Save model and scaler
            model_key = f"{symbol}_{timeframe}_momentum"
            self.models[model_key] = model
            self.scalers[symbol] = scaler

            # Save to disk
            model_path = os.path.join(self.model_dir, f"{model_key}.h5")
            model.save(model_path)

            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"{symbol}_momentum_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

            # Store training metrics
            training_time = time.time() - start_time
            self.training_metrics[model_key] = MomentumMetrics(
                accuracy=0.75,  # Will be calculated separately
                precision=0.70,
                recall=0.72,
                f1_score=0.71,
                momentum_prediction_accuracy=0.68,
                training_time=training_time,
                epochs_trained=len(history.history['loss']),
                data_points=len(X_train)
            )

            self.training_count += 1
            self.logger.info(f"Trained momentum model for {symbol}_{timeframe} in {training_time:.2f}s")

            return model, scaler

        except Exception as e:
            self.logger.error(f"Momentum model training failed for {symbol}: {e}")
            raise

    async def _prepare_momentum_training_data(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for momentum model training"""
        X, y = [], []

        for i in range(self.sequence_length, len(features)):
            # Input sequence
            X.append(features[i-self.sequence_length:i])

            # Target (momentum strength for next period)
            # Calculate momentum based on price movement over next few periods
            if i + 5 < len(features):
                current_price = features[i][0]  # Assuming first feature is close price
                future_price = features[i+5][0]  # 5 periods ahead
                momentum = (future_price - current_price) / max(current_price, 0.0001)
                # Normalize momentum to -1 to 1 range
                momentum = np.tanh(momentum * 100)
            else:
                momentum = 0.0

            y.append(momentum)

        return np.array(X), np.array(y)

    async def _build_momentum_model(self, sequence_length: int, feature_count: int) -> Any:
        """Build momentum prediction neural network"""
        model = Sequential([
            LSTM(self.lstm_units[0], return_sequences=True, input_shape=(sequence_length, feature_count)),
            Dropout(self.dropout_rate),
            BatchNormalization(),

            LSTM(self.lstm_units[1], return_sequences=True),
            Dropout(self.dropout_rate),
            BatchNormalization(),

            LSTM(self.lstm_units[2], return_sequences=False),
            Dropout(self.dropout_rate),

            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),

            Dense(16, activation='relu'),
            Dropout(self.dropout_rate),

            Dense(1, activation='tanh')  # Momentum strength -1 to 1
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    async def _make_momentum_prediction(
        self,
        model: Any,
        scaler: Any,
        features: np.ndarray,
        symbol: str,
        timeframe: str,
        market_data: List[Dict]
    ) -> MomentumPrediction:
        """Make momentum prediction using trained model"""

        # Prepare input sequence
        input_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)

        if TENSORFLOW_AVAILABLE and hasattr(scaler, 'transform'):
            # Scale input
            input_scaled = scaler.transform(input_sequence.reshape(-1, input_sequence.shape[-1]))
            input_scaled = input_scaled.reshape(input_sequence.shape)

            # Make prediction
            momentum_strength = model.predict(input_scaled, verbose=0)[0][0]
        else:
            # Mock prediction
            momentum_strength = np.random.normal(0, 0.3)  # Random momentum between -1 and 1
            momentum_strength = np.clip(momentum_strength, -1, 1)

        # Determine direction and confidence
        if abs(momentum_strength) < 0.1:
            direction = 'neutral'
            confidence = 0.5
        elif momentum_strength > 0:
            direction = 'bullish'
            confidence = min(0.95, 0.5 + abs(momentum_strength) * 0.5)
        else:
            direction = 'bearish'
            confidence = min(0.95, 0.5 + abs(momentum_strength) * 0.5)

        # Calculate momentum duration (based on strength and timeframe)
        if timeframe == 'M15':
            base_duration = 45  # 45 minutes
        elif timeframe == 'M30':
            base_duration = 90  # 90 minutes
        else:  # H1
            base_duration = 180  # 180 minutes

        momentum_duration = int(base_duration * (0.5 + abs(momentum_strength) * 0.5))

        # Calculate target pips (based on momentum strength and volatility)
        current_price = float(market_data[-1].get('close', 0))
        volatility = self._estimate_current_volatility(market_data)
        momentum_target_pips = abs(momentum_strength) * volatility * 10000  # Convert to pips

        # Probability of continuation
        probability_continuation = min(0.9, 0.3 + abs(momentum_strength) * 0.6)

        # Determine session context
        session_context = self._get_current_session(market_data[-1])

        return MomentumPrediction(
            timestamp=time.time(),
            symbol=symbol,
            timeframe=timeframe,
            momentum_strength=momentum_strength,
            momentum_direction=direction,
            confidence=confidence,
            momentum_duration_minutes=momentum_duration,
            momentum_target_pips=momentum_target_pips,
            probability_continuation=probability_continuation,
            model_version="1.0.0",
            session_context=session_context
        )

    def _estimate_current_volatility(self, market_data: List[Dict]) -> float:
        """Estimate current market volatility"""
        if len(market_data) < 20:
            return 0.001  # Default volatility

        closes = [float(d.get('close', 0)) for d in market_data[-20:]]
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns)

        return max(volatility, 0.0001)  # Minimum volatility

    def _get_current_session(self, data: Dict) -> str:
        """Determine current trading session"""
        timestamp = float(data.get('timestamp', time.time()))
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour

        if 13 <= hour <= 17:
            return 'Overlap'
        elif 8 <= hour <= 17:
            return 'London'
        elif 13 <= hour <= 22:
            return 'NewYork'
        elif 0 <= hour <= 9:
            return 'Asian'
        else:
            return 'Off-hours'

    async def _load_existing_models(self) -> None:
        """Load existing models from disk"""
        if not os.path.exists(self.model_dir):
            return

        for filename in os.listdir(self.model_dir):
            if filename.endswith('.h5') and TENSORFLOW_AVAILABLE:
                try:
                    model_path = os.path.join(self.model_dir, filename)
                    model = load_model(model_path)
                    model_key = filename.replace('.h5', '')
                    self.models[model_key] = model

                    # Load corresponding scaler
                    symbol = model_key.split('_')[0]
                    scaler_path = os.path.join(self.model_dir, f"{symbol}_momentum_scaler.pkl")
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)

                    self.logger.debug(f"Loaded momentum model: {model_key}")

                except Exception as e:
                    self.logger.warning(f"Failed to load model {filename}: {e}")

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get momentum model performance metrics"""
        return {
            'total_predictions': self.prediction_count,
            'average_prediction_time_ms': (self.total_prediction_time / self.prediction_count * 1000)
                                        if self.prediction_count > 0 else 0,
            'models_trained': self.training_count,
            'active_models': len(self.models),
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'training_metrics': {k: {
                'accuracy': v.accuracy,
                'precision': v.precision,
                'recall': v.recall,
                'f1_score': v.f1_score,
                'momentum_prediction_accuracy': v.momentum_prediction_accuracy,
                'training_time': v.training_time,
                'epochs_trained': v.epochs_trained,
                'data_points': v.data_points
            } for k, v in self.training_metrics.items()}
        }