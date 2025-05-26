"""
Training Pipeline for LSTM/GRU/Transformer Model Training

This module provides comprehensive model training capabilities for various
deep learning architectures used in trading applications. It includes
LSTM, GRU, and Transformer models with advanced training features.

Key Features:
- LSTM/GRU sequence models for time series
- Transformer models for attention-based learning
- Advanced training strategies
- Model validation and evaluation
- Hyperparameter optimization integration
- Real-time training monitoring

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, callbacks
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Using mock implementations.")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of models for training."""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    BIDIRECTIONAL_LSTM = "bidirectional_lstm"
    BIDIRECTIONAL_GRU = "bidirectional_gru"
    CNN_LSTM = "cnn_lstm"
    ATTENTION_LSTM = "attention_lstm"

class ValidationStrategy(Enum):
    """Validation strategies for time series."""
    HOLDOUT = "holdout"
    TIME_SERIES_SPLIT = "time_series_split"
    WALK_FORWARD = "walk_forward"
    EXPANDING_WINDOW = "expanding_window"

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: ModelType = ModelType.LSTM
    sequence_length: int = 60
    prediction_horizon: int = 1
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    validation_strategy: ValidationStrategy = ValidationStrategy.TIME_SERIES_SPLIT
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 10
    optimizer: str = "adam"
    loss_function: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["mae", "mse"])
    use_attention: bool = False
    attention_heads: int = 8
    transformer_layers: int = 4
    cnn_filters: int = 64
    cnn_kernel_size: int = 3

@dataclass
class TrainingResult:
    """Result from model training."""
    model: Any
    training_history: Dict[str, List[float]]
    validation_scores: Dict[str, float]
    test_scores: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    model_summary: str
    training_time: float
    best_epoch: int
    final_loss: float

class TrainingPipeline:
    """
    Comprehensive Model Training Pipeline

    Provides training capabilities for various deep learning models
    with advanced features for time series forecasting.
    """

    def __init__(self, config: TrainingConfig = None):
        """
        Initialize training pipeline.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model = None
        self.scaler_X = StandardScaler() if TF_AVAILABLE else None
        self.scaler_y = StandardScaler() if TF_AVAILABLE else None
        self.is_trained = False
        self.feature_names = None

        # Set random seeds for reproducibility
        if TF_AVAILABLE:
            tf.random.set_seed(42)
        np.random.seed(42)

        logger.info(f"TrainingPipeline initialized with model type: {self.config.model_type.value}")

    async def train_model(self,
                         X: pd.DataFrame,
                         y: pd.Series,
                         validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> TrainingResult:
        """
        Train the model with the given data.

        Args:
            X: Input features
            y: Target variable
            validation_data: Optional validation data

        Returns:
            Training result with model and metrics
        """
        start_time = datetime.now()
        logger.info(f"Starting model training with {self.config.model_type.value}...")

        # Validate input
        if X.empty or y.empty:
            raise ValueError("Input data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.feature_names = X.columns.tolist()

        # Prepare data for time series
        X_sequences, y_sequences = await self._prepare_sequences(X, y)

        # Split data
        if validation_data is None:
            X_train, X_val, y_train, y_val = await self._split_data(X_sequences, y_sequences)
        else:
            X_train, y_train = X_sequences, y_sequences
            X_val_df, y_val_series = validation_data
            X_val, y_val = await self._prepare_sequences(X_val_df, y_val_series)

        # Build model
        await self._build_model(X_train.shape)

        # Train model
        history = await self._train_model(X_train, y_train, X_val, y_val)

        # Evaluate model
        val_scores = await self._evaluate_model(X_val, y_val, "validation")
        test_scores = await self._evaluate_model(X_val, y_val, "test")  # Using val as test for now

        # Calculate feature importance (if applicable)
        feature_importance = await self._calculate_feature_importance(X_train, y_train)

        # Get model summary
        model_summary = self._get_model_summary()

        training_time = (datetime.now() - start_time).total_seconds()

        # Find best epoch
        best_epoch = 0
        if history and 'val_loss' in history:
            best_epoch = np.argmin(history['val_loss']) + 1

        final_loss = history['loss'][-1] if history and 'loss' in history else 0.0

        result = TrainingResult(
            model=self.model,
            training_history=history,
            validation_scores=val_scores,
            test_scores=test_scores,
            feature_importance=feature_importance,
            model_summary=model_summary,
            training_time=training_time,
            best_epoch=best_epoch,
            final_loss=final_loss
        )

        self.is_trained = True
        logger.info(f"Model training completed in {training_time:.2f}s. "
                   f"Best epoch: {best_epoch}, Final loss: {final_loss:.6f}")

        return result

    async def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare sequences
        X_sequences, _ = await self._prepare_sequences(X, pd.Series([0] * len(X)))

        if TF_AVAILABLE and self.model:
            predictions = self.model.predict(X_sequences, verbose=0)

            # Inverse transform if scaler was used
            if self.scaler_y and hasattr(self.scaler_y, 'scale_'):
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

            return predictions
        else:
            # Mock predictions
            return np.random.randn(len(X_sequences))

    async def _prepare_sequences(self,
                                X: pd.DataFrame,
                                y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series models."""
        # Scale features
        if self.scaler_X and TF_AVAILABLE:
            if not hasattr(self.scaler_X, 'scale_'):
                X_scaled = self.scaler_X.fit_transform(X)
            else:
                X_scaled = self.scaler_X.transform(X)
        else:
            X_scaled = X.values

        # Scale targets
        if self.scaler_y and TF_AVAILABLE:
            if not hasattr(self.scaler_y, 'scale_'):
                y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
            else:
                y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1)).flatten()
        else:
            y_scaled = y.values

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(self.config.sequence_length, len(X_scaled) - self.config.prediction_horizon + 1):
            X_sequences.append(X_scaled[i - self.config.sequence_length:i])
            y_sequences.append(y_scaled[i + self.config.prediction_horizon - 1])

        return np.array(X_sequences), np.array(y_sequences)

    async def _split_data(self,
                         X: np.ndarray,
                         y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data based on validation strategy."""
        if self.config.validation_strategy == ValidationStrategy.HOLDOUT:
            # Simple train-test split
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

        elif self.config.validation_strategy == ValidationStrategy.TIME_SERIES_SPLIT:
            # Time series split (last portion for validation)
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

        else:
            # Default to holdout
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

        return X_train, X_val, y_train, y_val

    async def _build_model(self, input_shape: Tuple[int, ...]):
        """Build the model based on configuration."""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Using mock model.")
            return

        if self.config.model_type == ModelType.LSTM:
            self.model = self._build_lstm_model(input_shape)
        elif self.config.model_type == ModelType.GRU:
            self.model = self._build_gru_model(input_shape)
        elif self.config.model_type == ModelType.BIDIRECTIONAL_LSTM:
            self.model = self._build_bidirectional_lstm_model(input_shape)
        elif self.config.model_type == ModelType.BIDIRECTIONAL_GRU:
            self.model = self._build_bidirectional_gru_model(input_shape)
        elif self.config.model_type == ModelType.CNN_LSTM:
            self.model = self._build_cnn_lstm_model(input_shape)
        elif self.config.model_type == ModelType.TRANSFORMER:
            self.model = self._build_transformer_model(input_shape)
        elif self.config.model_type == ModelType.ATTENTION_LSTM:
            self.model = self._build_attention_lstm_model(input_shape)
        else:
            # Default to LSTM
            self.model = self._build_lstm_model(input_shape)

        # Compile model
        optimizer = self._get_optimizer()
        self.model.compile(
            optimizer=optimizer,
            loss=self.config.loss_function,
            metrics=self.config.metrics
        )

        logger.info(f"Built {self.config.model_type.value} model with input shape: {input_shape}")

    def _build_lstm_model(self, input_shape: Tuple[int, ...]) -> Model:
        """Build LSTM model."""
        model = keras.Sequential()

        # Add LSTM layers
        for i, units in enumerate(self.config.hidden_units):
            return_sequences = i < len(self.config.hidden_units) - 1

            if i == 0:
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    input_shape=input_shape[1:],
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                ))
            else:
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                ))

            model.add(layers.Dropout(self.config.dropout_rate))

        # Output layer
        model.add(layers.Dense(1))

        return model

    def _build_gru_model(self, input_shape: Tuple[int, ...]) -> Model:
        """Build GRU model."""
        model = keras.Sequential()

        # Add GRU layers
        for i, units in enumerate(self.config.hidden_units):
            return_sequences = i < len(self.config.hidden_units) - 1

            if i == 0:
                model.add(layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    input_shape=input_shape[1:],
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                ))
            else:
                model.add(layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.dropout_rate
                ))

            model.add(layers.Dropout(self.config.dropout_rate))

        # Output layer
        model.add(layers.Dense(1))

        return model

    def _build_bidirectional_lstm_model(self, input_shape: Tuple[int, ...]) -> Model:
        """Build Bidirectional LSTM model."""
        model = keras.Sequential()

        # Add Bidirectional LSTM layers
        for i, units in enumerate(self.config.hidden_units):
            return_sequences = i < len(self.config.hidden_units) - 1

            if i == 0:
                model.add(layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.config.dropout_rate,
                        recurrent_dropout=self.config.dropout_rate
                    ),
                    input_shape=input_shape[1:]
                ))
            else:
                model.add(layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.config.dropout_rate,
                        recurrent_dropout=self.config.dropout_rate
                    )
                ))

            model.add(layers.Dropout(self.config.dropout_rate))

        # Output layer
        model.add(layers.Dense(1))

        return model

    def _build_bidirectional_gru_model(self, input_shape: Tuple[int, ...]) -> Model:
        """Build Bidirectional GRU model."""
        model = keras.Sequential()

        # Add Bidirectional GRU layers
        for i, units in enumerate(self.config.hidden_units):
            return_sequences = i < len(self.config.hidden_units) - 1

            if i == 0:
                model.add(layers.Bidirectional(
                    layers.GRU(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.config.dropout_rate,
                        recurrent_dropout=self.config.dropout_rate
                    ),
                    input_shape=input_shape[1:]
                ))
            else:
                model.add(layers.Bidirectional(
                    layers.GRU(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.config.dropout_rate,
                        recurrent_dropout=self.config.dropout_rate
                    )
                ))

            model.add(layers.Dropout(self.config.dropout_rate))

        # Output layer
        model.add(layers.Dense(1))

        return model

    def _build_cnn_lstm_model(self, input_shape: Tuple[int, ...]) -> Model:
        """Build CNN-LSTM model."""
        model = keras.Sequential()

        # CNN layers
        model.add(layers.Conv1D(
            filters=self.config.cnn_filters,
            kernel_size=self.config.cnn_kernel_size,
            activation='relu',
            input_shape=input_shape[1:]
        ))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(self.config.dropout_rate))

        # LSTM layers
        for i, units in enumerate(self.config.hidden_units):
            return_sequences = i < len(self.config.hidden_units) - 1

            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ))

            model.add(layers.Dropout(self.config.dropout_rate))

        # Output layer
        model.add(layers.Dense(1))

        return model
