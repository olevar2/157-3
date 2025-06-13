"""
Advanced ML Engine Indicator for Platform3

A comprehensive machine learning engine that combines multiple algorithms
for market prediction and analysis. This indicator implements ensemble
methods, deep learning models, and advanced feature engineering.

Author: Platform3 Development Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        AdaBoostRegressor, VotingRegressor
    )
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Fallback base class when sklearn not available
if not SKLEARN_AVAILABLE:
    class BaseIndicator:
        def __init__(self, **kwargs):
            pass
        
        def calculate(self, data):
            return np.zeros(len(data))


class AdvancedMLEngine:
    """
    Advanced Machine Learning Engine for Market Analysis
    
    This indicator implements a sophisticated ensemble of machine learning
    algorithms to predict market movements and identify patterns.
    
    Features:
    - Multiple algorithm ensemble (Random Forest, Gradient Boosting, Neural Network)
    - Advanced feature engineering
    - Model validation and performance metrics
    - Real-time prediction capabilities
    - Adaptive learning mechanisms
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 feature_window: int = 10,
                 prediction_horizon: int = 5,
                 ensemble_size: int = 3,
                 neural_hidden_layers: Tuple[int, ...] = (100, 50),
                 validation_split: float = 0.2,
                 retrain_frequency: int = 100):
        """
        Initialize Advanced ML Engine
        
        Args:
            lookback_period: Historical data points for training
            feature_window: Window for feature engineering
            prediction_horizon: Steps ahead to predict
            ensemble_size: Number of models in ensemble
            neural_hidden_layers: Neural network architecture
            validation_split: Fraction for validation
            retrain_frequency: Frequency of model retraining
        """
        self.lookback_period = lookback_period
        self.feature_window = feature_window
        self.prediction_horizon = prediction_horizon
        self.ensemble_size = ensemble_size
        self.neural_hidden_layers = neural_hidden_layers
        self.validation_split = validation_split
        self.retrain_frequency = retrain_frequency
        
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.training_count = 0
        self.performance_metrics = {}
        
        if SKLEARN_AVAILABLE:
            self._initialize_models()
        else:
            print("Warning: scikit-learn not available. Using fallback implementation.")
    
    def _initialize_models(self):
        """Initialize the ensemble of ML models"""
        if not SKLEARN_AVAILABLE:
            return
            
        # Random Forest
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Gradient Boosting
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Neural Network
        self.models['nn'] = MLPRegressor(
            hidden_layer_sizes=self.neural_hidden_layers,
            max_iter=500,
            random_state=42
        )
        
        # AdaBoost
        self.models['ada'] = AdaBoostRegressor(
            n_estimators=50,
            random_state=42
        )
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def _engineer_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Engineer features from market data
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Feature matrix
        """
        features = []
        
        # Basic price features
        features.extend([
            data['close'].pct_change(1).fillna(0),  # Returns
            data['close'].pct_change(5).fillna(0),  # 5-day returns
            data['high'] / data['close'] - 1,       # High/Close ratio
            data['low'] / data['close'] - 1,        # Low/Close ratio
            data['volume'].pct_change(1).fillna(0)  # Volume change
        ])
        
        # Technical indicators as features
        if len(data) >= self.feature_window:
            # Moving averages
            ma_short = data['close'].rolling(window=5).mean()
            ma_long = data['close'].rolling(window=20).mean()
            features.append((data['close'] - ma_short) / ma_short)
            features.append((ma_short - ma_long) / ma_long)
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append((rsi - 50) / 50)  # Normalized RSI
            
            # Bollinger Bands
            bb_middle = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            features.append((data['close'] - bb_middle) / bb_std)
            
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            features.append((macd - signal) / data['close'])
        
        # Convert to matrix and handle NaN values
        feature_matrix = np.column_stack(features)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        return feature_matrix
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with features and targets
        
        Args:
            data: Market data
            
        Returns:
            Features and targets for training
        """
        features = self._engineer_features(data)
        
        # Create targets (future returns)
        targets = data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        targets = targets.fillna(0)
        
        # Remove NaN rows and align data
        valid_indices = ~np.isnan(targets.values)
        features = features[valid_indices]
        targets = targets.values[valid_indices]
        
        return features, targets
    
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the ML ensemble
        
        Args:
            data: Historical market data
            
        Returns:
            Training performance metrics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
            
        if len(data) < self.lookback_period:
            return {'error': 'Insufficient data for training'}
        
        try:
            # Prepare training data
            features, targets = self._prepare_training_data(data)
            
            if len(features) < 10:  # Minimum samples for training
                return {'error': 'Insufficient valid samples'}
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, 
                test_size=self.validation_split,
                random_state=42
            )
            
            # Train each model
            performance = {}
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_val_scaled = self.scalers[model_name].transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Validate
                    y_pred = model.predict(X_val_scaled)
                    predictions[model_name] = y_pred
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    performance[f'{model_name}_mse'] = mse
                    performance[f'{model_name}_r2'] = r2
                    
                except Exception as e:
                    performance[f'{model_name}_error'] = str(e)
            
            # Ensemble prediction
            if predictions:
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
                ensemble_mse = mean_squared_error(y_val, ensemble_pred)
                ensemble_r2 = r2_score(y_val, ensemble_pred)
                
                performance['ensemble_mse'] = ensemble_mse
                performance['ensemble_r2'] = ensemble_r2
            
            self.performance_metrics = performance
            self.is_trained = True
            self.training_count += 1
            
            return performance
            
        except Exception as e:
            return {'error': f'Training failed: {str(e)}'}
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using the trained ensemble
        
        Args:
            data: Recent market data
            
        Returns:
            Prediction results
        """
        if not SKLEARN_AVAILABLE:
            return {'prediction': 0.0, 'confidence': 0.0, 'error': 'scikit-learn not available'}
            
        if not self.is_trained:
            return {'prediction': 0.0, 'confidence': 0.0, 'error': 'Model not trained'}
        
        try:
            # Engineer features for latest data
            features = self._engineer_features(data)
            
            if len(features) == 0:
                return {'prediction': 0.0, 'confidence': 0.0, 'error': 'No features available'}
            
            # Use the last row for prediction
            latest_features = features[-1:].reshape(1, -1)
            
            # Get predictions from each model
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    scaled_features = self.scalers[model_name].transform(latest_features)
                    
                    # Predict
                    pred = model.predict(scaled_features)[0]
                    predictions[model_name] = pred
                    
                    # Calculate confidence based on training performance
                    r2_score = self.performance_metrics.get(f'{model_name}_r2', 0)
                    confidence = max(0, min(1, r2_score))
                    confidences[model_name] = confidence
                    
                except Exception as e:
                    predictions[model_name] = 0.0
                    confidences[model_name] = 0.0
            
            # Ensemble prediction with confidence weighting
            if predictions:
                total_confidence = sum(confidences.values())
                if total_confidence > 0:
                    weighted_pred = sum(
                        pred * confidences[name] 
                        for name, pred in predictions.items()
                    ) / total_confidence
                    avg_confidence = total_confidence / len(predictions)
                else:
                    weighted_pred = np.mean(list(predictions.values()))
                    avg_confidence = 0.5
            else:
                weighted_pred = 0.0
                avg_confidence = 0.0
            
            return {
                'prediction': weighted_pred,
                'confidence': avg_confidence,
                'individual_predictions': predictions,
                'individual_confidences': confidences
            }
            
        except Exception as e:
            return {'prediction': 0.0, 'confidence': 0.0, 'error': f'Prediction failed: {str(e)}'}
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate ML engine signals for backtesting
        
        Args:
            data: Historical market data
            
        Returns:
            Array of prediction signals
        """
        if not SKLEARN_AVAILABLE:
            return np.zeros(len(data))
            
        signals = np.zeros(len(data))
        
        # Train initially if we have enough data
        if len(data) >= self.lookback_period and not self.is_trained:
            train_data = data.iloc[:self.lookback_period]
            self.train(train_data)
        
        # Generate signals for each point
        for i in range(self.lookback_period, len(data)):
            # Retrain periodically
            if i % self.retrain_frequency == 0 and i > self.lookback_period:
                train_data = data.iloc[i-self.lookback_period:i]
                self.train(train_data)
            
            # Make prediction
            recent_data = data.iloc[max(0, i-self.feature_window):i+1]
            prediction_result = self.predict(recent_data)
            
            # Convert prediction to signal
            prediction = prediction_result.get('prediction', 0.0)
            confidence = prediction_result.get('confidence', 0.0)
            
            # Apply confidence threshold
            if confidence > 0.6:  # Only use high-confidence predictions
                signals[i] = np.tanh(prediction * 10)  # Normalize to [-1, 1]
            else:
                signals[i] = 0.0
        
        return signals
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained models
        
        Returns:
            Feature importance scores
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return {}
        
        importance_scores = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_scores[model_name] = model.feature_importances_.tolist()
        
        return importance_scores
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()


# Test and example usage
if __name__ == "__main__":
    print("Testing Advanced ML Engine Indicator...")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Create synthetic market data with trends and patterns
    price = 100
    prices = []
    volumes = []
    
    for i in range(200):
        # Add trend and noise
        trend = 0.001 * np.sin(i * 0.1)
        noise = np.random.normal(0, 0.02)
        price = price * (1 + trend + noise)
        prices.append(price)
        
        # Volume with some correlation to price movements
        volume = 1000000 + np.random.normal(0, 100000)
        volumes.append(max(volume, 500000))
    
    data = pd.DataFrame({
        'date': dates,
        'open': np.array(prices) * np.random.uniform(0.995, 1.005, 200),
        'high': np.array(prices) * np.random.uniform(1.005, 1.02, 200),
        'low': np.array(prices) * np.random.uniform(0.98, 0.995, 200),
        'close': prices,
        'volume': volumes
    })
    
    # Initialize and test the ML engine
    ml_engine = AdvancedMLEngine(
        lookback_period=50,
        feature_window=20,
        prediction_horizon=3
    )
    
    print(f"Initialized ML Engine with {len(ml_engine.models) if SKLEARN_AVAILABLE else 0} models")
    
    # Test training
    print("\nTraining ML models...")
    training_data = data.iloc[:100]
    train_results = ml_engine.train(training_data)
    print("Training Results:")
    for key, value in train_results.items():
        print(f"  {key}: {value}")
    
    # Test prediction
    print("\nTesting prediction...")
    recent_data = data.iloc[90:110]
    prediction = ml_engine.predict(recent_data)
    print("Prediction Results:")
    for key, value in prediction.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Test signal calculation
    print("\nCalculating signals...")
    signals = ml_engine.calculate(data)
    
    print(f"Generated {len(signals)} signals")
    print(f"Signal range: [{signals.min():.6f}, {signals.max():.6f}]")
    print(f"Non-zero signals: {np.count_nonzero(signals)}")
    print(f"Last 10 signals: {signals[-10:]}")
    
    # Performance metrics
    if ml_engine.is_trained:
        print("\nPerformance Metrics:")
        metrics = ml_engine.get_performance_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Feature importance
        importance = ml_engine.get_feature_importance()
        if importance:
            print("\nFeature Importance (first model):")
            first_model = list(importance.keys())[0]
            for i, score in enumerate(importance[first_model][:5]):
                print(f"  Feature {i}: {score:.4f}")
    
    print("\nAdvanced ML Engine test completed successfully!")