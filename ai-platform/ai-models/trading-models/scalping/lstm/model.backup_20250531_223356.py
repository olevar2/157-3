"""
Scalping LSTM Model
LSTM neural network for M1-M5 price prediction optimized for scalping strategies.
Provides ultra-fast price direction prediction for sub-second trading decisions.
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
class LSTMPrediction:
    """LSTM price prediction result for scalping trades"""
    timestamp: float
    symbol: str
    timeframe: str  # M1, M5
    predicted_price: float
    predicted_direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    price_change_pips: float
    probability_up: float
    probability_down: float
    model_version: str
    prediction_horizon: int  # seconds ahead
    
    # Essential scalping metrics for daily profit
    signal_strength: float  # 0-1 signal quality
    risk_reward_ratio: float  # expected profit/loss ratio
    optimal_entry_price: float  # suggested entry price
    stop_loss_price: float  # risk management stop loss
    take_profit_price: float  # profit target
    market_session: str  # 'ASIAN', 'LONDON', 'NY', 'OVERLAP'
    volatility_regime: str  # 'LOW', 'MEDIUM', 'HIGH'


@dataclass
class LSTMTrainingMetrics:
    """LSTM model training metrics"""
    loss: float
    val_loss: float
    mae: float
    val_mae: float
    accuracy: float
    training_time: float
    epochs_trained: int
    data_points: int


@dataclass
class ScalpingFeatures:
    """Feature set for scalping LSTM"""
    price_features: List[float]  # OHLC, price changes
    volume_features: List[float]  # Volume, volume changes
    spread_features: List[float]  # Bid/ask spread data
    momentum_features: List[float]  # Short-term momentum
    microstructure_features: List[float]  # Market microstructure


class ScalpingLSTM:
    """
    Scalping LSTM Model
    Ultra-fast LSTM for M1-M5 price prediction optimized for scalping
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.sequence_length = self.config.get('sequence_length', 60)  # 60 ticks/bars
        self.prediction_horizon = self.config.get('prediction_horizon', 30)  # 30 seconds ahead
        self.feature_count = self.config.get('feature_count', 20)
        
        # LSTM architecture
        self.lstm_units = [64, 32, 16]  # Multi-layer LSTM
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        
        # Model storage
        self.models = {}  # symbol -> model
        self.scalers = {}  # symbol -> scaler
        self.training_metrics = {}  # symbol -> metrics
        
        # Data buffers for real-time prediction
        self.feature_buffers = {}  # symbol -> deque of features
        self.max_buffer_size = 1000
        
        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.training_count = 0
        
        # Model paths
        self.model_dir = self.config.get('model_dir', 'models/scalping_lstm')
        os.makedirs(self.model_dir, exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the LSTM model"""
        try:
            if not TENSORFLOW_AVAILABLE:
                self.logger.warning("TensorFlow not available. Using mock LSTM implementation.")
                return
            
            # Set TensorFlow configuration for speed
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            
            # Load existing models if available
            await self._load_existing_models()
            
            self.logger.info("Scalping LSTM initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Scalping LSTM: {e}")
            raise
    
    async def predict_price(self, symbol: str, timeframe: str, market_data: List[Dict]) -> LSTMPrediction:
        """
        Predict next price movement using LSTM
        """
        start_time = time.time()
        
        try:
            # Prepare features
            features = await self._prepare_features(symbol, market_data)
            
            if len(features) < self.sequence_length:
                raise ValueError(f"Insufficient data for prediction. Need {self.sequence_length}, got {len(features)}")
            
            # Get or create model
            model = await self._get_or_create_model(symbol, timeframe)
            scaler = self.scalers.get(symbol)
            
            if model is None or scaler is None:
                # Train new model if not available
                model, scaler = await self._train_model(symbol, timeframe, features)
            
            # Make prediction
            prediction_result = await self._make_prediction(
                model, scaler, features, symbol, timeframe
            )
            
            # Update performance tracking
            prediction_time = time.time() - start_time
            self.prediction_count += 1
            self.total_prediction_time += prediction_time
            
            self.logger.debug(f"LSTM prediction for {symbol} completed in {prediction_time:.3f}s")
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"LSTM prediction failed for {symbol}: {e}")
            raise
    
    async def _prepare_features(self, symbol: str, market_data: List[Dict]) -> np.ndarray:
        """Prepare feature matrix for LSTM input"""
        
        if not market_data:
            raise ValueError("No market data provided")
        
        features_list = []
        
        for i, data in enumerate(market_data):
            # Price features
            price_features = [
                float(data.get('open', 0)),
                float(data.get('high', 0)),
                float(data.get('low', 0)),
                float(data.get('close', 0)),
                float(data.get('close', 0)) - float(data.get('open', 0)),  # Price change
            ]
            
            # Volume features
            volume_features = [
                float(data.get('volume', 0)),
                float(data.get('volume', 0)) / max(1, np.mean([d.get('volume', 1) for d in market_data[max(0, i-10):i+1]])),  # Volume ratio
            ]
            
            # Spread features
            spread_features = [
                float(data.get('spread', 0.0001)),
                float(data.get('bid', 0)),
                float(data.get('ask', 0)),
            ]
            
            # Momentum features (short-term)
            if i >= 5:
                recent_closes = [float(d.get('close', 0)) for d in market_data[i-5:i+1]]
                momentum_features = [
                    (recent_closes[-1] - recent_closes[0]) / max(recent_closes[0], 0.0001),  # 5-period momentum
                    np.std(recent_closes),  # 5-period volatility
                    (recent_closes[-1] - np.mean(recent_closes)) / max(np.std(recent_closes), 0.0001),  # Z-score
                ]
            else:
                momentum_features = [0.0, 0.0, 0.0]
            
            # Microstructure features
            microstructure_features = [
                float(data.get('tick_direction', 0)),  # Tick direction
                float(data.get('order_imbalance', 0)),  # Order book imbalance
                float(data.get('trade_intensity', 0)),  # Trade intensity
            ]
            
            # Technical indicators (fast)
            if i >= 10:
                recent_closes = [float(d.get('close', 0)) for d in market_data[i-10:i+1]]
                rsi = self._calculate_rsi(recent_closes, 10)
                sma = np.mean(recent_closes)
                tech_features = [
                    rsi,
                    (recent_closes[-1] - sma) / max(sma, 0.0001),  # Price vs SMA
                ]
            else:
                tech_features = [50.0, 0.0]  # Neutral RSI, no SMA deviation
            
            # Combine all features
            feature_vector = (price_features + volume_features + spread_features + 
                            momentum_features + microstructure_features + tech_features)
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    async def _get_or_create_model(self, symbol: str, timeframe: str) -> Optional[Any]:
        """Get existing model or return None to trigger training"""
        model_key = f"{symbol}_{timeframe}"
        
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
    
    async def _train_model(self, symbol: str, timeframe: str, features: np.ndarray) -> Tuple[Any, Any]:
        """Train new LSTM model"""
        start_time = time.time()
        
        try:
            if not TENSORFLOW_AVAILABLE:
                # Mock model for testing
                mock_model = {
                    'type': 'mock_lstm',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'predict': lambda x: np.random.random((x.shape[0], 1))
                }
                mock_scaler = {
                    'transform': lambda x: x,
                    'inverse_transform': lambda x: x
                }
                return mock_model, mock_scaler
            
            # Prepare training data
            X, y = await self._prepare_training_data(features)
            
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Split data
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = await self._build_lstm_model(X.shape[1], X.shape[2])
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(patience=5, factor=0.5)
                ],
                verbose=0
            )
            
            # Save model and scaler
            model_key = f"{symbol}_{timeframe}"
            self.models[model_key] = model
            self.scalers[symbol] = scaler
            
            # Save to disk
            model_path = os.path.join(self.model_dir, f"{model_key}.h5")
            model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Store training metrics
            training_time = time.time() - start_time
            self.training_metrics[model_key] = LSTMTrainingMetrics(
                loss=history.history['loss'][-1],
                val_loss=history.history['val_loss'][-1],
                mae=history.history.get('mae', [0])[-1],
                val_mae=history.history.get('val_mae', [0])[-1],
                accuracy=0.0,  # Will be calculated separately
                training_time=training_time,
                epochs_trained=len(history.history['loss']),
                data_points=len(X_train)
            )
            
            self.training_count += 1
            self.logger.info(f"Trained LSTM model for {symbol}_{timeframe} in {training_time:.2f}s")
            
            return model, scaler
            
        except Exception as e:
            self.logger.error(f"LSTM training failed for {symbol}: {e}")
            raise
    
    async def _prepare_training_data(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            # Input sequence
            X.append(features[i-self.sequence_length:i])
            
            # Target (next price change)
            current_price = features[i-1][3]  # Close price
            next_price = features[i][3]  # Next close price
            price_change = (next_price - current_price) / max(current_price, 0.0001)
            y.append(price_change)
        
        return np.array(X), np.array(y)
    
    async def _build_lstm_model(self, sequence_length: int, feature_count: int) -> Any:
        """Build LSTM neural network architecture"""
        model = Sequential([
            LSTM(self.lstm_units[0], return_sequences=True, input_shape=(sequence_length, feature_count)),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            LSTM(self.lstm_units[1], return_sequences=True),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            LSTM(self.lstm_units[2], return_sequences=False),
            Dropout(self.dropout_rate),
            
            Dense(16, activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(1, activation='linear')  # Price change prediction
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    async def _make_prediction(
        self, 
        model: Any, 
        scaler: Any, 
        features: np.ndarray, 
        symbol: str, 
        timeframe: str
    ) -> LSTMPrediction:
        """Make price prediction using trained model"""
        
        # Prepare input sequence
        input_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        if TENSORFLOW_AVAILABLE and hasattr(scaler, 'transform'):
            # Scale input
            input_scaled = scaler.transform(input_sequence.reshape(-1, input_sequence.shape[-1]))
            input_scaled = input_scaled.reshape(input_sequence.shape)
            
            # Make prediction
            prediction = model.predict(input_scaled, verbose=0)[0][0]
        else:
            # Mock prediction
            prediction = np.random.normal(0, 0.001)  # Small random price change
        
        # Convert to price and direction
        current_price = features[-1][3]  # Last close price
        predicted_price = current_price * (1 + prediction)
        
        # Determine direction and confidence
        if abs(prediction) < 0.0001:  # Very small change
            direction = 'sideways'
            confidence = 0.5
        elif prediction > 0:
            direction = 'up'
            confidence = min(0.9, 0.5 + abs(prediction) * 1000)
        else:
            direction = 'down'
            confidence = min(0.9, 0.5 + abs(prediction) * 1000)
        
        # Calculate probabilities
        prob_up = max(0.1, min(0.9, 0.5 + prediction * 500))
        prob_down = 1.0 - prob_up
        
        # Convert to pips (assuming 4-decimal currency pair)
        price_change_pips = prediction * current_price * 10000
        
        return LSTMPrediction(
            timestamp=time.time(),
            symbol=symbol,
            timeframe=timeframe,
            predicted_price=predicted_price,
            predicted_direction=direction,
            confidence=confidence,
            price_change_pips=price_change_pips,
            probability_up=prob_up,
            probability_down=prob_down,
            model_version="1.0.0",
            prediction_horizon=self.prediction_horizon
        )
    
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
                    scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)
                    
                    self.logger.debug(f"Loaded LSTM model: {model_key}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load model {filename}: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get LSTM performance metrics"""
        return {
            'total_predictions': self.prediction_count,
            'average_prediction_time_ms': (self.total_prediction_time / self.prediction_count * 1000) 
                                        if self.prediction_count > 0 else 0,
            'models_trained': self.training_count,
            'active_models': len(self.models),
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'training_metrics': {k: {
                'loss': v.loss,
                'val_loss': v.val_loss,
                'training_time': v.training_time,
                'epochs': v.epochs_trained,
                'data_points': v.data_points
            } for k, v in self.training_metrics.items()}
        }
