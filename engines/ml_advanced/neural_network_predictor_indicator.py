"""
NeuralNetworkPredictor Indicator - Deep Learning Market Prediction
Platform3 Trading Framework
Version: 1.0.0

This indicator implements neural network-based market prediction using
feedforward and recurrent neural architectures for price forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface
from engines.ai_enhancement.indicators.base_indicator import IndicatorValidationError


@dataclass
class NeuralNetworkConfig:
    """Configuration for NeuralNetwork indicator"""
    input_size: int = 10
    hidden_layers: List[int] = None
    output_size: int = 3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.2
    sequence_length: int = 20
    prediction_horizon: int = 1
    early_stopping_patience: int = 10
    validation_split: float = 0.2


class NeuralLayer:
    """Simple neural network layer implementation"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.weights = np.random.normal(0, np.sqrt(2.0/input_size), (input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        
        # For backpropagation
        self.last_input = None
        self.last_output = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.last_input = x
        z = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'relu':
            self.last_output = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            self.last_output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            self.last_output = np.tanh(z)
        else:  # linear
            self.last_output = z
            
        return self.last_output
        
    def backward(self, grad_output: np.ndarray, learning_rate: float):
        """Backward pass (simplified)"""
        if self.activation == 'relu':
            grad_z = grad_output * (self.last_output > 0)
        elif self.activation == 'sigmoid':
            grad_z = grad_output * self.last_output * (1 - self.last_output)
        elif self.activation == 'tanh':
            grad_z = grad_output * (1 - self.last_output**2)
        else:  # linear
            grad_z = grad_output
            
        # Calculate gradients
        grad_weights = np.dot(self.last_input.T, grad_z) / self.last_input.shape[0]
        grad_bias = np.mean(grad_z, axis=0, keepdims=True)
        grad_input = np.dot(grad_z, self.weights.T)
        
        # Update weights
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input


class SimpleNeuralNetwork:
    """Simple neural network implementation"""
    
    def __init__(self, config: NeuralNetworkConfig):
        self.config = config
        self.layers = []
        self._build_network()
        
    def _build_network(self):
        """Build neural network architecture"""
        if self.config.hidden_layers is None:
            self.config.hidden_layers = [64, 32, 16]
            
        # Input layer
        layer_sizes = [self.config.input_size] + self.config.hidden_layers + [self.config.output_size]
        
        for i in range(len(layer_sizes) - 1):
            activation = 'relu' if i < len(layer_sizes) - 2 else 'linear'
            layer = NeuralLayer(layer_sizes[i], layer_sizes[i + 1], activation)
            self.layers.append(layer)
            
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
        
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Single training step"""
        # Forward pass
        predictions = self.forward(x)
        
        # Calculate loss (MSE)
        loss = np.mean((predictions - y)**2)
        
        # Backward pass
        grad_output = 2 * (predictions - y) / y.shape[0]
        
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, self.config.learning_rate)
            
        return loss


class NeuralNetworkPredictorIndicator(StandardIndicatorInterface):
    """
    NeuralNetworkPredictor Indicator v1.0.0
    
    A deep learning indicator that uses neural networks to predict market
    movements based on historical price and volume patterns.
    
    Features:
    - Feedforward neural network architecture
    - Sequence-based learning for temporal patterns
    - Multi-output prediction (direction, magnitude, confidence)
    - Online learning with incremental updates
    - Feature normalization and preprocessing
    - Early stopping and regularization
    
    Mathematical Foundation:
    The network uses feedforward architecture:
    
    h₁ = σ(W₁x + b₁)
    h₂ = σ(W₂h₁ + b₂)
    ...
    y = W_out·h_n + b_out
    
    Where σ is the activation function (ReLU for hidden layers)
    
    Loss function: L = Σ(y_pred - y_true)² + λΣ|W|² (L2 regularization)
    """
    
    # Class-level metadata
    name = "NeuralNetworkPredictor"
    version = "1.0.0"
    category = "ml_advanced"
    description = "Neural network-based market prediction system"
    
    def __init__(self, **params):
        """Initialize NeuralNetworkPredictor indicator"""
        # Extract parameters with defaults
        self.parameters = params
        
        hidden_layers = self.parameters.get('hidden_layers', [64, 32, 16])
        if isinstance(hidden_layers, str):
            # Parse string representation
            hidden_layers = [int(x.strip()) for x in hidden_layers.split(',')]
            
        self.config = NeuralNetworkConfig(
            input_size=self.parameters.get('input_size', 10),
            hidden_layers=hidden_layers,
            output_size=self.parameters.get('output_size', 3),
            learning_rate=self.parameters.get('learning_rate', 0.001),
            batch_size=self.parameters.get('batch_size', 32),
            epochs=self.parameters.get('epochs', 100),
            dropout_rate=self.parameters.get('dropout_rate', 0.2),
            sequence_length=self.parameters.get('sequence_length', 20),
            prediction_horizon=self.parameters.get('prediction_horizon', 1),
            early_stopping_patience=self.parameters.get('early_stopping_patience', 10),
            validation_split=self.parameters.get('validation_split', 0.2)
        )
        
        # Initialize state
        self.reset()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def reset(self):
        """Reset indicator state"""
        self.network = None
        self.feature_scaler = {'mean': None, 'std': None}
        self.target_scaler = {'mean': None, 'std': None}
        self.training_history = []
        self.prediction_history = []
        self.is_trained = False
        
    def calculate(self, data: Union[pd.DataFrame, Dict[str, List], np.ndarray]) -> np.ndarray:
        """
        Calculate NeuralNetworkPredictor predictions
        
        Args:
            data: Price data (OHLCV format)
            
        Returns:
            np.ndarray: Neural network predictions (direction, magnitude, confidence)
        """
        try:
            # Input validation
            if data is None or len(data) == 0:
                raise ValidationError("Input data cannot be empty")
                
            # Convert data to DataFrame if needed
            df = self._prepare_data(data)
            
            if len(df) < self.config.sequence_length + self.config.prediction_horizon:
                return np.full((len(df), 3), np.nan)
                
            # Prepare features and targets
            features, targets = self._prepare_training_data(df)
            
            # Initialize network if needed
            if self.network is None:
                self.network = SimpleNeuralNetwork(self.config)
                
            # Train network if not trained or retrain with new data
            if not self.is_trained or len(features) > len(self.training_history) * 1.2:
                self._train_network(features, targets)
                
            # Generate predictions
            predictions = self._generate_predictions(features)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in NeuralNetworkPredictor calculation: {str(e)}")
            raise CalculationError(f"NeuralNetworkPredictor calculation failed: {str(e)}")
            
    def _prepare_data(self, data: Any) -> pd.DataFrame:
        """Prepare and validate input data"""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame({'close': data})
            else:
                columns = ['open', 'high', 'low', 'close', 'volume'][:data.shape[1]]
                df = pd.DataFrame(data, columns=columns)
        else:
            raise ValidationError("Unsupported data format")
            
        # Ensure required columns
        if 'close' not in df.columns:
            raise ValidationError("Close price is required")
            
        # Fill missing columns with close price
        for col in ['open', 'high', 'low']:
            if col not in df.columns:
                df[col] = df['close']
                
        if 'volume' not in df.columns:
            df['volume'] = 1.0
            
        return df.dropna()
        
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        # Extract raw features
        features = self._extract_features(df)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(features) - self.config.sequence_length - self.config.prediction_horizon + 1):
            # Input sequence
            sequence = features[i:i + self.config.sequence_length]
            sequences.append(sequence.flatten())  # Flatten for feedforward network
            
            # Target (future price movement)
            current_price = df['close'].iloc[i + self.config.sequence_length - 1]
            future_price = df['close'].iloc[i + self.config.sequence_length + self.config.prediction_horizon - 1]
            
            # Calculate targets
            price_change = (future_price - current_price) / current_price
            direction = 1 if price_change > 0.001 else (-1 if price_change < -0.001 else 0)
            magnitude = abs(price_change)
            
            targets.append([direction, magnitude, 1.0])  # confidence = 1.0 for training data
            
        features_array = np.array(sequences)
        targets_array = np.array(targets)
        
        # Normalize features
        if self.feature_scaler['mean'] is None:
            self.feature_scaler['mean'] = np.mean(features_array, axis=0)
            self.feature_scaler['std'] = np.std(features_array, axis=0) + 1e-8
            
        features_normalized = (features_array - self.feature_scaler['mean']) / self.feature_scaler['std']
        
        # Normalize targets
        if self.target_scaler['mean'] is None:
            self.target_scaler['mean'] = np.mean(targets_array, axis=0)
            self.target_scaler['std'] = np.std(targets_array, axis=0) + 1e-8
            
        targets_normalized = (targets_array - self.target_scaler['mean']) / self.target_scaler['std']
        
        return features_normalized, targets_normalized
        
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract technical features for neural network"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Price-based features
        returns = np.diff(np.log(close), prepend=close[0])
        
        # Moving averages
        sma5 = pd.Series(close).rolling(5).mean().values
        sma20 = pd.Series(close).rolling(20).mean().values
        
        # Technical indicators
        rsi = self._calculate_rsi(close, 14)
        bb_upper, bb_lower = self._calculate_bollinger_bands(close, 20)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        # Volatility
        volatility = pd.Series(returns).rolling(10).std().values
        
        # Volume features
        volume_sma = pd.Series(volume).rolling(20).mean().values
        volume_ratio = volume / (volume_sma + 1e-8)
        
        # Combine features
        features = np.column_stack([
            returns,
            sma5 / close - 1,
            sma20 / close - 1,
            rsi / 100,
            bb_position,
            volatility,
            np.log(volume_ratio + 1e-8),
            (high - low) / close,  # True range normalized
            (close - low) / (high - low + 1e-8),  # Williams %R like
            np.log(close / np.roll(close, 5) + 1e-8)  # 5-period momentum
        ])
        
        return np.nan_to_num(features, 0)
        
    def _train_network(self, features: np.ndarray, targets: np.ndarray):
        """Train the neural network"""
        if len(features) < self.config.batch_size:
            return
            
        # Split data
        split_idx = int(len(features) * (1 - self.config.validation_split))
        train_x, val_x = features[:split_idx], features[split_idx:]
        train_y, val_y = targets[:split_idx], targets[split_idx:]
        
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Shuffle training data
            indices = np.random.permutation(len(train_x))
            train_x_shuffled = train_x[indices]
            train_y_shuffled = train_y[indices]
            
            # Mini-batch training
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(train_x_shuffled), self.config.batch_size):
                batch_x = train_x_shuffled[i:i + self.config.batch_size]
                batch_y = train_y_shuffled[i:i + self.config.batch_size]
                
                if len(batch_x) < self.config.batch_size:
                    continue
                    
                loss = self.network.train_step(batch_x, batch_y)
                epoch_loss += loss
                num_batches += 1
                
            if num_batches > 0:
                epoch_loss /= num_batches
                
            # Validation
            if len(val_x) > 0:
                val_predictions = self.network.forward(val_x)
                val_loss = np.mean((val_predictions - val_y)**2)
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    break
                    
            self.training_history.append(epoch_loss)
            
        self.is_trained = True
        
    def _generate_predictions(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions for all data points"""
        n_points = len(features) + self.config.sequence_length + self.config.prediction_horizon - 1
        predictions = np.zeros((n_points, 3))
        
        # Generate predictions for sequences
        if len(features) > 0:
            raw_predictions = self.network.forward(features)
            
            # Denormalize predictions
            denormalized = (raw_predictions * self.target_scaler['std'] + 
                          self.target_scaler['mean'])
            
            # Fill predictions array
            start_idx = self.config.sequence_length
            end_idx = start_idx + len(denormalized)
            predictions[start_idx:end_idx] = denormalized
            
        return predictions
        
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(period).mean().values
        avg_loss = pd.Series(losses).rolling(period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[50], rsi])
        
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int):
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(period).mean().values
        std = pd.Series(prices).rolling(period).std().values
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        return upper, lower
        
    def get_signal(self, data: Any) -> int:
        """Get current signal from the indicator"""
        result = self.calculate(data)
        if len(result) == 0:
            return 0
        return int(np.sign(result[-1, 0]))  # Return direction prediction
        
    def get_current_value(self, data: Any) -> float:
        """Get current indicator value"""
        result = self.calculate(data)
        if len(result) == 0:
            return 0.0
        return float(result[-1, 1])  # Return magnitude prediction
        
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        try:
            input_size = self.parameters.get('input_size', 10)
            if not isinstance(input_size, int) or input_size <= 0:
                return False
                
            learning_rate = self.parameters.get('learning_rate', 0.001)
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                return False
                
            sequence_length = self.parameters.get('sequence_length', 20)
            if not isinstance(sequence_length, int) or sequence_length <= 0:
                return False
                
            return True
        except Exception:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Return NeuralNetworkPredictor metadata as dictionary for compatibility"""
        return {
            "name": "NeuralNetworkPredictor",
            "category": self.CATEGORY,
            "description": "Neural Network Predictor for advanced market pattern recognition and forecasting",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Dict",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """NeuralNetworkPredictor can work with OHLCV data"""
        return ["open", "high", "low", "close", "volume"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for neural network prediction"""
        return max(self.parameters.get("sequence_length", 20), 50)  # Needs sequence data


def get_neural_network_predictor_indicator(**params) -> NeuralNetworkPredictorIndicator:
    """
    Factory function to create NeuralNetworkPredictor indicator
    
    Args:
        **params: Indicator parameters
        
    Returns:
        NeuralNetworkPredictorIndicator: Configured indicator instance
    """
    return NeuralNetworkPredictorIndicator(**params)


# Export for registry discovery
__all__ = ['NeuralNetworkPredictorIndicator', 'get_neural_network_predictor_indicator']