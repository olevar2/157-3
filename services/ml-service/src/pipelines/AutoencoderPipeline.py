"""
Autoencoder Pipeline for Feature Extraction and Anomaly Detection

This module provides comprehensive autoencoder-based feature extraction and
anomaly detection capabilities for trading applications. It includes various
autoencoder architectures and anomaly detection methods.

Key Features:
- Vanilla Autoencoder for feature extraction
- Variational Autoencoder (VAE) for probabilistic features
- Denoising Autoencoder for robust features
- Anomaly detection and scoring
- Real-time feature extraction
- Latent space analysis

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
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
    from tensorflow.keras import layers, Model
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Using mock implementations.")

logger = logging.getLogger(__name__)

class AutoencoderType(Enum):
    """Types of autoencoders."""
    VANILLA = "vanilla"
    VARIATIONAL = "variational"
    DENOISING = "denoising"
    SPARSE = "sparse"
    CONTRACTIVE = "contractive"

@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder pipeline."""
    autoencoder_type: AutoencoderType = AutoencoderType.VANILLA
    encoding_dim: int = 32
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    output_activation: str = "linear"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    noise_factor: float = 0.1  # For denoising autoencoder
    sparsity_regularizer: float = 1e-5  # For sparse autoencoder
    beta: float = 1.0  # For VAE
    early_stopping_patience: int = 10

@dataclass
class AnomalyDetection:
    """Anomaly detection results."""
    anomaly_scores: np.ndarray
    threshold: float
    anomalies: np.ndarray
    reconstruction_errors: np.ndarray
    percentile_threshold: float = 95.0

@dataclass
class AutoencoderResult:
    """Result from autoencoder pipeline."""
    encoded_features: pd.DataFrame
    decoded_features: pd.DataFrame
    reconstruction_error: np.ndarray
    anomaly_detection: AnomalyDetection
    model_summary: str
    training_history: Dict[str, List[float]]
    computation_time: float
    encoding_dimension: int

class AutoencoderPipeline:
    """
    Comprehensive Autoencoder Pipeline

    Provides various autoencoder architectures for feature extraction and
    anomaly detection in trading applications.
    """

    def __init__(self, config: AutoencoderConfig = None):
        """
        Initialize autoencoder pipeline.

        Args:
            config: Autoencoder configuration
        """
        self.config = config or AutoencoderConfig()
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler() if TF_AVAILABLE else None
        self.is_fitted = False
        self.feature_names = None

        # Set random seeds for reproducibility
        if TF_AVAILABLE:
            tf.random.set_seed(42)
        np.random.seed(42)

        logger.info(f"AutoencoderPipeline initialized with type: {self.config.autoencoder_type.value}")

    async def fit_transform(self,
                           X: pd.DataFrame,
                           validation_data: Optional[pd.DataFrame] = None) -> AutoencoderResult:
        """
        Fit the autoencoder model and transform data.

        Args:
            X: Input features
            validation_data: Optional validation data

        Returns:
            Autoencoder transformation result
        """
        start_time = datetime.now()
        logger.info(f"Starting autoencoder training with {self.config.autoencoder_type.value}...")

        # Validate input
        if X.empty:
            raise ValueError("Input features cannot be empty")

        self.feature_names = X.columns.tolist()

        # Prepare data
        X_processed = await self._prepare_data(X)

        # Split data if no validation data provided
        if validation_data is None:
            X_train, X_val = train_test_split(
                X_processed,
                test_size=self.config.validation_split,
                random_state=42
            ) if TF_AVAILABLE else (X_processed, X_processed[:10])
        else:
            X_train = X_processed
            X_val = await self._prepare_data(validation_data, fit_scaler=False)

        # Build and train model
        await self._build_model(X_train.shape[1])
        history = await self._train_model(X_train, X_val)

        # Transform data
        encoded_features = await self._encode_features(X_processed)
        decoded_features = await self._decode_features(encoded_features)

        # Calculate reconstruction error
        reconstruction_error = await self._calculate_reconstruction_error(X_processed, decoded_features)

        # Perform anomaly detection
        anomaly_detection = await self._detect_anomalies(reconstruction_error)

        # Get model summary
        model_summary = self._get_model_summary()

        computation_time = (datetime.now() - start_time).total_seconds()

        result = AutoencoderResult(
            encoded_features=pd.DataFrame(
                encoded_features,
                columns=[f'encoded_{i}' for i in range(encoded_features.shape[1])],
                index=X.index
            ),
            decoded_features=pd.DataFrame(
                decoded_features,
                columns=self.feature_names,
                index=X.index
            ),
            reconstruction_error=reconstruction_error,
            anomaly_detection=anomaly_detection,
            model_summary=model_summary,
            training_history=history,
            computation_time=computation_time,
            encoding_dimension=self.config.encoding_dim
        )

        self.is_fitted = True
        logger.info(f"Autoencoder training completed in {computation_time:.2f}s. "
                   f"Encoded to {self.config.encoding_dim} dimensions.")

        return result

    async def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform new data using fitted autoencoder.

        Args:
            X: Input features to transform

        Returns:
            Tuple of (encoded_features, decoded_features)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        X_processed = await self._prepare_data(X, fit_scaler=False)
        encoded_features = await self._encode_features(X_processed)
        decoded_features = await self._decode_features(encoded_features)

        encoded_df = pd.DataFrame(
            encoded_features,
            columns=[f'encoded_{i}' for i in range(encoded_features.shape[1])],
            index=X.index
        )

        decoded_df = pd.DataFrame(
            decoded_features,
            columns=self.feature_names,
            index=X.index
        )

        return encoded_df, decoded_df

    async def detect_anomalies(self, X: pd.DataFrame) -> AnomalyDetection:
        """
        Detect anomalies in new data.

        Args:
            X: Input features

        Returns:
            Anomaly detection results
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before anomaly detection")

        X_processed = await self._prepare_data(X, fit_scaler=False)
        decoded_features = await self._decode_features(await self._encode_features(X_processed))
        reconstruction_error = await self._calculate_reconstruction_error(X_processed, decoded_features)

        return await self._detect_anomalies(reconstruction_error)

    async def _prepare_data(self, X: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """Prepare data for autoencoder."""
        # Handle missing values
        X_clean = X.fillna(X.mean())

        # Scale features
        if self.scaler and TF_AVAILABLE:
            if fit_scaler:
                X_scaled = self.scaler.fit_transform(X_clean)
            else:
                X_scaled = self.scaler.transform(X_clean)
            return X_scaled

        return X_clean.values

    async def _build_model(self, input_dim: int):
        """Build the autoencoder model."""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Using mock model.")
            return

        # Input layer
        input_layer = keras.Input(shape=(input_dim,))

        if self.config.autoencoder_type == AutoencoderType.VANILLA:
            self.model, self.encoder, self.decoder = self._build_vanilla_autoencoder(input_layer, input_dim)
        elif self.config.autoencoder_type == AutoencoderType.VARIATIONAL:
            self.model, self.encoder, self.decoder = self._build_variational_autoencoder(input_layer, input_dim)
        elif self.config.autoencoder_type == AutoencoderType.DENOISING:
            self.model, self.encoder, self.decoder = self._build_denoising_autoencoder(input_layer, input_dim)
        elif self.config.autoencoder_type == AutoencoderType.SPARSE:
            self.model, self.encoder, self.decoder = self._build_sparse_autoencoder(input_layer, input_dim)
        else:
            # Default to vanilla
            self.model, self.encoder, self.decoder = self._build_vanilla_autoencoder(input_layer, input_dim)

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        logger.info(f"Built {self.config.autoencoder_type.value} autoencoder model")

    def _build_vanilla_autoencoder(self, input_layer, input_dim):
        """Build vanilla autoencoder."""
        # Encoder
        encoded = input_layer
        for hidden_dim in self.config.hidden_layers:
            encoded = layers.Dense(hidden_dim, activation=self.config.activation)(encoded)

        # Bottleneck
        encoded = layers.Dense(self.config.encoding_dim, activation=self.config.activation)(encoded)

        # Decoder
        decoded = encoded
        for hidden_dim in reversed(self.config.hidden_layers):
            decoded = layers.Dense(hidden_dim, activation=self.config.activation)(decoded)

        # Output
        decoded = layers.Dense(input_dim, activation=self.config.output_activation)(decoded)

        # Models
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)

        # Decoder model
        encoded_input = keras.Input(shape=(self.config.encoding_dim,))
        decoder_layer = encoded_input
        for i, hidden_dim in enumerate(reversed(self.config.hidden_layers)):
            decoder_layer = autoencoder.layers[-(len(self.config.hidden_layers) + 1 - i)](decoder_layer)
        decoder_layer = autoencoder.layers[-1](decoder_layer)
        decoder = Model(encoded_input, decoder_layer)

        return autoencoder, encoder, decoder

    def _build_variational_autoencoder(self, input_layer, input_dim):
        """Build variational autoencoder."""
        # Encoder
        encoded = input_layer
        for hidden_dim in self.config.hidden_layers:
            encoded = layers.Dense(hidden_dim, activation=self.config.activation)(encoded)

        # Latent space parameters
        z_mean = layers.Dense(self.config.encoding_dim)(encoded)
        z_log_var = layers.Dense(self.config.encoding_dim)(encoded)

        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling)([z_mean, z_log_var])

        # Decoder
        decoded = z
        for hidden_dim in reversed(self.config.hidden_layers):
            decoded = layers.Dense(hidden_dim, activation=self.config.activation)(decoded)
        decoded = layers.Dense(input_dim, activation=self.config.output_activation)(decoded)

        # VAE model
        vae = Model(input_layer, decoded)
        encoder = Model(input_layer, [z_mean, z_log_var, z])

        # Decoder model
        decoder_input = keras.Input(shape=(self.config.encoding_dim,))
        decoder_layer = decoder_input
        for i, hidden_dim in enumerate(reversed(self.config.hidden_layers)):
            decoder_layer = layers.Dense(hidden_dim, activation=self.config.activation)(decoder_layer)
        decoder_output = layers.Dense(input_dim, activation=self.config.output_activation)(decoder_layer)
        decoder = Model(decoder_input, decoder_output)

        # Add VAE loss
        reconstruction_loss = tf.reduce_mean(tf.square(input_layer - decoded))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        vae_loss = reconstruction_loss + self.config.beta * kl_loss
        vae.add_loss(vae_loss)

        return vae, encoder, decoder

    def _build_denoising_autoencoder(self, input_layer, input_dim):
        """Build denoising autoencoder."""
        # Add noise to input
        noise = layers.GaussianNoise(self.config.noise_factor)(input_layer)

        # Encoder
        encoded = noise
        for hidden_dim in self.config.hidden_layers:
            encoded = layers.Dense(hidden_dim, activation=self.config.activation)(encoded)
        encoded = layers.Dense(self.config.encoding_dim, activation=self.config.activation)(encoded)

        # Decoder
        decoded = encoded
        for hidden_dim in reversed(self.config.hidden_layers):
            decoded = layers.Dense(hidden_dim, activation=self.config.activation)(decoded)
        decoded = layers.Dense(input_dim, activation=self.config.output_activation)(decoded)

        # Models
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)

        # Decoder model
        encoded_input = keras.Input(shape=(self.config.encoding_dim,))
        decoder_layer = encoded_input
        for i, hidden_dim in enumerate(reversed(self.config.hidden_layers)):
            decoder_layer = layers.Dense(hidden_dim, activation=self.config.activation)(decoder_layer)
        decoder_output = layers.Dense(input_dim, activation=self.config.output_activation)(decoder_layer)
        decoder = Model(encoded_input, decoder_output)

        return autoencoder, encoder, decoder

    def _build_sparse_autoencoder(self, input_layer, input_dim):
        """Build sparse autoencoder."""
        # Encoder with sparsity regularization
        encoded = input_layer
        for hidden_dim in self.config.hidden_layers:
            encoded = layers.Dense(
                hidden_dim,
                activation=self.config.activation,
                activity_regularizer=keras.regularizers.l1(self.config.sparsity_regularizer)
            )(encoded)

        encoded = layers.Dense(
            self.config.encoding_dim,
            activation=self.config.activation,
            activity_regularizer=keras.regularizers.l1(self.config.sparsity_regularizer)
        )(encoded)

        # Decoder
        decoded = encoded
        for hidden_dim in reversed(self.config.hidden_layers):
            decoded = layers.Dense(hidden_dim, activation=self.config.activation)(decoded)
        decoded = layers.Dense(input_dim, activation=self.config.output_activation)(decoded)

        # Models
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)

        # Decoder model
        encoded_input = keras.Input(shape=(self.config.encoding_dim,))
        decoder_layer = encoded_input
        for i, hidden_dim in enumerate(reversed(self.config.hidden_layers)):
            decoder_layer = layers.Dense(hidden_dim, activation=self.config.activation)(decoder_layer)
        decoder_output = layers.Dense(input_dim, activation=self.config.output_activation)(decoder_layer)
        decoder = Model(encoded_input, decoder_output)

        return autoencoder, encoder, decoder

    async def _train_model(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, List[float]]:
        """Train the autoencoder model."""
        if not TF_AVAILABLE or self.model is None:
            logger.warning("TensorFlow not available or model not built. Using mock training.")
            return {'loss': [1.0, 0.5, 0.3], 'val_loss': [1.1, 0.6, 0.4]}

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]

        # Train model
        history = self.model.fit(
            X_train, X_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=0
        )

        return history.history

    async def _encode_features(self, X: np.ndarray) -> np.ndarray:
        """Encode features using the encoder."""
        if not TF_AVAILABLE or self.encoder is None:
            # Mock encoding
            return X[:, :self.config.encoding_dim]

        if self.config.autoencoder_type == AutoencoderType.VARIATIONAL:
            # For VAE, use mean of latent distribution
            z_mean, z_log_var, z = self.encoder.predict(X, verbose=0)
            return z_mean
        else:
            return self.encoder.predict(X, verbose=0)

    async def _decode_features(self, encoded: np.ndarray) -> np.ndarray:
        """Decode features using the decoder."""
        if not TF_AVAILABLE or self.decoder is None:
            # Mock decoding - repeat encoded features to match input dimension
            n_features = len(self.feature_names) if self.feature_names else encoded.shape[1] * 2
            decoded = np.tile(encoded, (1, n_features // encoded.shape[1] + 1))
            return decoded[:, :n_features]

        return self.decoder.predict(encoded, verbose=0)

    async def _calculate_reconstruction_error(self,
                                            original: np.ndarray,
                                            reconstructed: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error."""
        return np.mean(np.square(original - reconstructed), axis=1)

    async def _detect_anomalies(self, reconstruction_error: np.ndarray) -> AnomalyDetection:
        """Detect anomalies based on reconstruction error."""
        # Calculate threshold using percentile
        threshold = np.percentile(reconstruction_error, 95.0)

        # Identify anomalies
        anomalies = reconstruction_error > threshold

        return AnomalyDetection(
            anomaly_scores=reconstruction_error,
            threshold=threshold,
            anomalies=anomalies,
            reconstruction_errors=reconstruction_error,
            percentile_threshold=95.0
        )

    def _get_model_summary(self) -> str:
        """Get model summary as string."""
        if not TF_AVAILABLE or self.model is None:
            return "Mock autoencoder model summary"

        import io
        import sys

        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            self.model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout

        return summary

    def get_latent_representation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get latent representation of input data."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before getting latent representation")

        X_processed = asyncio.run(self._prepare_data(X, fit_scaler=False))
        encoded = asyncio.run(self._encode_features(X_processed))

        return pd.DataFrame(
            encoded,
            columns=[f'latent_{i}' for i in range(encoded.shape[1])],
            index=X.index
        )

    def save_model(self, filepath: str):
        """Save the trained model."""
        if not TF_AVAILABLE or self.model is None:
            logger.warning("No model to save")
            return

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot load model.")
            return

        self.model = keras.models.load_model(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
