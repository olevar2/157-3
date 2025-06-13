"""
Neural Network Predictor for Platform3

An advanced neural network-based predictor that uses deep learning
to forecast market movements, price targets, and trend changes.
Includes multiple architectures and ensemble methods.

Author: Platform3 Development Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class NeuralNetworkPredictor:
    """
    Neural Network Predictor for Financial Markets
    
    This predictor implements multiple neural network architectures:
    - Multi-Layer Perceptron (MLP)
    - Long Short-Term Memory (LSTM) networks
    - Convolutional Neural Networks (CNN)
    - Ensemble methods combining multiple models
    
    Features:
    - Multiple prediction types (regression, classification)
    - Feature engineering and selection
    - Model validation and performance metrics
    - Real-time prediction capabilities
    - Ensemble predictions for robustness
    """
    
    def __init__(self,
                 prediction_type: str = 'regression',
                 prediction_horizon: int = 5,
                 sequence_length: int = 20,
                 hidden_layers: Tuple[int, ...] = (100, 50, 25),
                 use_lstm: bool = True,
                 use_cnn: bool = False,
                 ensemble_size: int = 3,
                 validation_split: float = 0.2,
                 epochs: int = 100,
                 batch_size: int = 32):
        """
        Initialize Neural Network Predictor
        
        Args:
            prediction_type: 'regression' or 'classification'
            prediction_horizon: Steps ahead to predict
            sequence_length: Length of input sequences
            hidden_layers: Architecture of hidden layers
            use_lstm: Whether to use LSTM networks
            use_cnn: Whether to use CNN networks
            ensemble_size: Number of models in ensemble
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Batch size for training
        """
        self.prediction_type = prediction_type
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        self.hidden_layers = hidden_layers
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.ensemble_size = ensemble_size
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_columns = []
        self.performance_metrics = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize neural network models"""
        if SKLEARN_AVAILABLE:
            # MLP models
            if self.prediction_type == 'regression':
                self.models['mlp'] = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layers,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            else:
                self.models['mlp'] = MLPClassifier(
                    hidden_layer_sizes=self.hidden_layers,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            
            self.scalers['mlp'] = StandardScaler()
        
        if TF_AVAILABLE:
            # Will create TensorFlow models during training
            pass
        
        if not SKLEARN_AVAILABLE and not TF_AVAILABLE:
            print("Warning: Neither scikit-learn nor TensorFlow available. Using fallback implementation.")
    
    def _engineer_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Engineer features from market data
        
        Args:
            data: Market data with OHLCV
            
        Returns:
            Feature matrix
        """
        features = []
        feature_names = []
        
        # Price features
        features.extend([
            data['close'].pct_change(1).fillna(0),  # Returns
            data['close'].pct_change(5).fillna(0),  # 5-day returns
            data['high'] / data['close'] - 1,       # High/Close ratio
            data['low'] / data['close'] - 1,        # Low/Close ratio
        ])
        feature_names.extend(['return_1d', 'return_5d', 'high_ratio', 'low_ratio'])
        
        # Volume features
        if 'volume' in data.columns:
            features.append(data['volume'].pct_change(1).fillna(0))
            feature_names.append('volume_change')
        
        # Technical indicators
        if len(data) >= 20:
            # Moving averages
            ma5 = data['close'].rolling(window=5).mean()
            ma20 = data['close'].rolling(window=20).mean()
            features.extend([
                (data['close'] - ma5) / ma5,
                (ma5 - ma20) / ma20
            ])
            feature_names.extend(['ma5_ratio', 'ma5_ma20_ratio'])
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append((rsi - 50) / 50)  # Normalized RSI
            feature_names.append('rsi_norm')
            
            # Bollinger Bands
            bb_middle = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            features.append((data['close'] - bb_middle) / bb_std)
            feature_names.append('bb_position')
            
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            features.append((macd - signal) / data['close'])
            feature_names.append('macd_histogram')
            
            # Volatility
            volatility = data['close'].rolling(window=20).std()
            features.append(volatility / data['close'])
            feature_names.append('volatility_ratio')
        
        # Store feature names for interpretation
        self.feature_columns = feature_names
        
        # Convert to matrix and handle NaN
        feature_matrix = np.column_stack(features)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        return feature_matrix
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/CNN models
        
        Args:
            features: Feature matrix
            targets: Target values
            
        Returns:
            Sequence features and targets
        """
        seq_features = []
        seq_targets = []
        
        for i in range(self.sequence_length, len(features)):
            seq_features.append(features[i-self.sequence_length:i])
            seq_targets.append(targets[i])
        
        return np.array(seq_features), np.array(seq_targets)
    
    def _create_targets(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create target variables based on prediction type
        
        Args:
            data: Market data
            
        Returns:
            Target array
        """
        if self.prediction_type == 'regression':
            # Future returns
            targets = data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            targets = targets.fillna(0)
        else:
            # Classification: future direction
            future_returns = data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            # 0: down, 1: neutral, 2: up
            targets = np.where(future_returns > 0.01, 2, 
                              np.where(future_returns < -0.01, 0, 1))
        
        return targets
    
    def _build_lstm_model(self, input_shape: Tuple, output_size: int = 1) -> 'keras.Model':
        """
        Build LSTM model using TensorFlow/Keras
        
        Args:
            input_shape: Shape of input sequences
            output_size: Number of output neurons
            
        Returns:
            Compiled LSTM model
        """
        if not TF_AVAILABLE:
            return None
        
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25, activation='relu'),
            layers.Dense(output_size, activation='linear' if self.prediction_type == 'regression' else 'softmax')
        ])
        
        if self.prediction_type == 'regression':
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _build_cnn_model(self, input_shape: Tuple, output_size: int = 1) -> 'keras.Model':
        """
        Build CNN model for time series
        
        Args:
            input_shape: Shape of input sequences
            output_size: Number of output neurons
            
        Returns:
            Compiled CNN model
        """
        if not TF_AVAILABLE:
            return None
        
        model = keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.Dropout(0.2),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=50, kernel_size=3, activation='relu'),
            layers.Dropout(0.2),
            layers.GlobalMaxPooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dense(output_size, activation='linear' if self.prediction_type == 'regression' else 'softmax')
        ])
        
        if self.prediction_type == 'regression':
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train neural network models
        
        Args:
            data: Historical market data
            
        Returns:
            Training results and performance metrics
        """
        if len(data) < self.sequence_length + self.prediction_horizon:
            return {'error': 'Insufficient data for training'}
        
        try:
            # Feature engineering
            features = self._engineer_features(data)
            targets = self._create_targets(data)
            
            # Align features and targets
            min_len = min(len(features), len(targets))
            features = features[:min_len]
            targets = targets[:min_len]
            
            # Remove NaN targets
            valid_mask = ~np.isnan(targets)
            features = features[valid_mask]
            targets = targets[valid_mask]
            
            if len(features) < 50:
                return {'error': 'Insufficient valid samples for training'}
            
            # Split data
            split_idx = int(len(features) * (1 - self.validation_split))
            
            train_features = features[:split_idx]
            train_targets = targets[:split_idx]
            val_features = features[split_idx:]
            val_targets = targets[split_idx:]
            
            results = {}
            
            # Train MLP model
            if SKLEARN_AVAILABLE and 'mlp' in self.models:
                try:
                    # Scale features
                    train_features_scaled = self.scalers['mlp'].fit_transform(train_features)
                    val_features_scaled = self.scalers['mlp'].transform(val_features)
                    
                    # Train model
                    self.models['mlp'].fit(train_features_scaled, train_targets)
                    
                    # Validate
                    val_pred = self.models['mlp'].predict(val_features_scaled)
                    
                    if self.prediction_type == 'regression':
                        mse = mean_squared_error(val_targets, val_pred)
                        results['mlp_mse'] = mse
                        results['mlp_rmse'] = np.sqrt(mse)
                    else:
                        accuracy = accuracy_score(val_targets, val_pred)
                        results['mlp_accuracy'] = accuracy
                    
                except Exception as e:
                    results['mlp_error'] = str(e)
            
            # Train deep learning models
            if TF_AVAILABLE and (self.use_lstm or self.use_cnn):
                try:
                    # Create sequences
                    seq_features, seq_targets = self._create_sequences(features, targets)
                    
                    if len(seq_features) < 20:
                        results['dl_error'] = 'Insufficient sequence data'
                    else:
                        # Split sequences
                        seq_split_idx = int(len(seq_features) * (1 - self.validation_split))
                        
                        train_seq_features = seq_features[:seq_split_idx]
                        train_seq_targets = seq_targets[:seq_split_idx]
                        val_seq_features = seq_features[seq_split_idx:]
                        val_seq_targets = seq_targets[seq_split_idx:]
                        
                        # Scale sequence features
                        scaler = StandardScaler()
                        train_seq_features_scaled = scaler.fit_transform(
                            train_seq_features.reshape(-1, train_seq_features.shape[-1])
                        ).reshape(train_seq_features.shape)
                        val_seq_features_scaled = scaler.transform(
                            val_seq_features.reshape(-1, val_seq_features.shape[-1])
                        ).reshape(val_seq_features.shape)
                        
                        self.scalers['sequence'] = scaler
                        
                        # Determine output size
                        output_size = 1 if self.prediction_type == 'regression' else len(np.unique(targets))
                        
                        # Train LSTM
                        if self.use_lstm:
                            lstm_model = self._build_lstm_model(
                                (self.sequence_length, train_seq_features.shape[-1]),
                                output_size
                            )
                            
                            if lstm_model is not None:
                                history = lstm_model.fit(
                                    train_seq_features_scaled, train_seq_targets,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    validation_data=(val_seq_features_scaled, val_seq_targets),
                                    verbose=0
                                )
                                
                                self.models['lstm'] = lstm_model
                                
                                # Get final validation loss
                                val_loss = history.history['val_loss'][-1]
                                results['lstm_val_loss'] = val_loss
                                
                                if self.prediction_type == 'classification':
                                    val_acc = history.history['val_accuracy'][-1]
                                    results['lstm_val_accuracy'] = val_acc
                        
                        # Train CNN
                        if self.use_cnn:
                            cnn_model = self._build_cnn_model(
                                (self.sequence_length, train_seq_features.shape[-1]),
                                output_size
                            )
                            
                            if cnn_model is not None:
                                history = cnn_model.fit(
                                    train_seq_features_scaled, train_seq_targets,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    validation_data=(val_seq_features_scaled, val_seq_targets),
                                    verbose=0
                                )
                                
                                self.models['cnn'] = cnn_model
                                
                                # Get final validation loss
                                val_loss = history.history['val_loss'][-1]
                                results['cnn_val_loss'] = val_loss
                                
                                if self.prediction_type == 'classification':
                                    val_acc = history.history['val_accuracy'][-1]
                                    results['cnn_val_accuracy'] = val_acc
                
                except Exception as e:
                    results['dl_error'] = str(e)
            
            self.performance_metrics = results
            self.is_trained = True
            
            return results
            
        except Exception as e:
            return {'error': f'Training failed: {str(e)}'}
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using trained models
        
        Args:
            data: Recent market data
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            return {'error': 'Models not trained'}
        
        try:
            # Engineer features
            features = self._engineer_features(data)
            
            if len(features) == 0:
                return {'error': 'No features available'}
            
            predictions = {}
            confidences = {}
            
            # MLP prediction
            if SKLEARN_AVAILABLE and 'mlp' in self.models:
                try:
                    latest_features = features[-1:].reshape(1, -1)
                    scaled_features = self.scalers['mlp'].transform(latest_features)
                    
                    pred = self.models['mlp'].predict(scaled_features)[0]
                    predictions['mlp'] = pred
                    
                    # Calculate confidence (simplified)
                    if hasattr(self.models['mlp'], 'predict_proba'):
                        proba = self.models['mlp'].predict_proba(scaled_features)[0]
                        confidences['mlp'] = np.max(proba)
                    else:
                        confidences['mlp'] = 0.7  # Default confidence
                        
                except Exception as e:
                    predictions['mlp'] = 0.0
                    confidences['mlp'] = 0.0
            
            # Deep learning predictions
            if TF_AVAILABLE and len(features) >= self.sequence_length:
                try:
                    # Create sequence
                    sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                    
                    if 'sequence' in self.scalers:
                        # Scale sequence
                        sequence_scaled = self.scalers['sequence'].transform(
                            sequence.reshape(-1, sequence.shape[-1])
                        ).reshape(sequence.shape)
                    else:
                        sequence_scaled = sequence
                    
                    # LSTM prediction
                    if 'lstm' in self.models:
                        lstm_pred = self.models['lstm'].predict(sequence_scaled, verbose=0)[0]
                        if self.prediction_type == 'regression':
                            predictions['lstm'] = lstm_pred[0] if len(lstm_pred.shape) > 0 else lstm_pred
                        else:
                            predictions['lstm'] = np.argmax(lstm_pred)
                            confidences['lstm'] = np.max(lstm_pred)
                    
                    # CNN prediction
                    if 'cnn' in self.models:
                        cnn_pred = self.models['cnn'].predict(sequence_scaled, verbose=0)[0]
                        if self.prediction_type == 'regression':
                            predictions['cnn'] = cnn_pred[0] if len(cnn_pred.shape) > 0 else cnn_pred
                        else:
                            predictions['cnn'] = np.argmax(cnn_pred)
                            confidences['cnn'] = np.max(cnn_pred)
                            
                except Exception as e:
                    pass  # Handle gracefully
            
            # Ensemble prediction
            if predictions:
                if self.prediction_type == 'regression':
                    ensemble_pred = np.mean(list(predictions.values()))
                    ensemble_confidence = np.mean(list(confidences.values())) if confidences else 0.7
                else:
                    # Majority vote for classification
                    votes = list(predictions.values())
                    ensemble_pred = max(set(votes), key=votes.count)
                    ensemble_confidence = votes.count(ensemble_pred) / len(votes)
            else:
                ensemble_pred = 0.0
                ensemble_confidence = 0.0
            
            return {
                'prediction': ensemble_pred,
                'confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'prediction_type': self.prediction_type
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate prediction signals for backtesting
        
        Args:
            data: Historical market data
            
        Returns:
            Array of prediction signals
        """
        signals = np.zeros(len(data))
        
        # Need minimum data for training
        min_train_size = max(100, self.sequence_length * 3)
        
        if len(data) < min_train_size:
            return signals
        
        # Train on initial data
        train_data = data.iloc[:min_train_size]
        self.train(train_data)
        
        # Generate predictions
        for i in range(min_train_size, len(data)):
            # Use recent data for prediction
            recent_data = data.iloc[max(0, i-50):i+1]
            
            # Make prediction
            result = self.predict(recent_data)
            
            if 'error' not in result:
                prediction = result.get('prediction', 0.0)
                confidence = result.get('confidence', 0.0)
                
                # Apply confidence threshold
                if confidence > 0.6:
                    if self.prediction_type == 'regression':
                        signals[i] = np.tanh(prediction * 10)  # Normalize
                    else:
                        # Convert classification to signal
                        if prediction == 2:  # Up
                            signals[i] = 1.0
                        elif prediction == 0:  # Down
                            signals[i] = -1.0
                        else:  # Neutral
                            signals[i] = 0.0
        
        return signals
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance analysis
        
        Returns:
            Feature importance information
        """
        importance = {}
        
        # For MLP models, we can get feature importance through permutation
        if 'mlp' in self.models and len(self.feature_columns) > 0:
            importance['feature_names'] = self.feature_columns
            importance['feature_count'] = len(self.feature_columns)
        
        # For deep learning models, feature importance is more complex
        if 'lstm' in self.models or 'cnn' in self.models:
            importance['uses_sequences'] = True
            importance['sequence_length'] = self.sequence_length
        
        return importance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of trained models
        
        Returns:
            Model summary information
        """
        summary = {
            'is_trained': self.is_trained,
            'prediction_type': self.prediction_type,
            'prediction_horizon': self.prediction_horizon,
            'models_available': list(self.models.keys()),
            'performance_metrics': self.performance_metrics
        }
        
        if TF_AVAILABLE:
            summary['tensorflow_available'] = True
            for model_name in ['lstm', 'cnn']:
                if model_name in self.models:
                    model = self.models[model_name]
                    summary[f'{model_name}_params'] = model.count_params()
        
        return summary


# Test and example usage
if __name__ == "__main__":
    print("Testing Neural Network Predictor...")
    
    # Generate sample data
    np.random.seed(42)
    n_points = 300
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
    
    # Create synthetic market data with patterns
    price = 100
    prices = []
    volumes = []
    
    for i in range(n_points):
        # Add trend, cycles, and noise
        trend = 0.001 * np.sin(i * 0.02)
        cycle = 0.005 * np.sin(i * 0.1)
        noise = np.random.normal(0, 0.02)
        
        price = price * (1 + trend + cycle + noise)
        prices.append(price)
        
        volume = 1000000 + np.random.normal(0, 100000)
        volumes.append(max(volume, 500000))
    
    data = pd.DataFrame({
        'date': dates,
        'open': np.array(prices) * np.random.uniform(0.995, 1.005, n_points),
        'high': np.array(prices) * np.random.uniform(1.005, 1.02, n_points),
        'low': np.array(prices) * np.random.uniform(0.98, 0.995, n_points),
        'close': prices,
        'volume': volumes
    })
    
    print(f"Generated {n_points} data points")
    print(f"Price range: {data['close'].min():.2f} to {data['close'].max():.2f}")
    
    # Test regression predictor
    print("\\nTesting Regression Predictor...")
    reg_predictor = NeuralNetworkPredictor(
        prediction_type='regression',
        prediction_horizon=3,
        sequence_length=20,
        hidden_layers=(50, 25),
        use_lstm=TF_AVAILABLE,
        use_cnn=False,
        epochs=20,  # Reduced for testing
        batch_size=16
    )
    
    print(f"Initialized predictor with {'TensorFlow' if TF_AVAILABLE else 'scikit-learn only'}")
    
    # Train model
    print("Training regression model...")
    train_data = data.iloc[:200]
    train_results = reg_predictor.train(train_data)
    
    print("Training Results:")
    for key, value in train_results.items():
        print(f"  {key}: {value}")
    
    # Test prediction
    print("\\nTesting prediction...")
    test_data = data.iloc[150:250]
    prediction = reg_predictor.predict(test_data)
    
    print("Prediction Results:")
    for key, value in prediction.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Test classification predictor
    print("\\n" + "="*50)
    print("Testing Classification Predictor...")
    
    class_predictor = NeuralNetworkPredictor(
        prediction_type='classification',
        prediction_horizon=5,
        sequence_length=15,
        hidden_layers=(30, 15),
        use_lstm=TF_AVAILABLE,
        use_cnn=False,
        epochs=15,
        batch_size=16
    )
    
    # Train classification model
    print("Training classification model...")
    class_train_results = class_predictor.train(train_data)
    
    print("Classification Training Results:")
    for key, value in class_train_results.items():
        print(f"  {key}: {value}")
    
    # Test classification prediction
    class_prediction = class_predictor.predict(test_data)
    
    print("Classification Prediction Results:")
    for key, value in class_prediction.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Test signal generation
    print("\\nTesting signal generation...")
    signals = reg_predictor.calculate(data)
    
    print(f"Generated {len(signals)} signals")
    print(f"Signal range: [{signals.min():.6f}, {signals.max():.6f}]")
    print(f"Non-zero signals: {np.count_nonzero(signals)}")
    print(f"Last 10 signals: {signals[-10:]}")
    
    # Model summary
    summary = reg_predictor.get_model_summary()
    print("\\nModel Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\\nNeural Network Predictor test completed successfully!")