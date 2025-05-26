"""
Platform3 Forex Trading Platform
Spread Predictor - ML-Based Bid/Ask Spread Forecasting

This module provides advanced machine learning models for predicting
bid/ask spreads to optimize scalping entry and exit timing.

Author: Platform3 Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import asyncio
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpreadData:
    """Spread data structure"""
    timestamp: datetime
    bid: float
    ask: float
    spread: float
    volume: float
    volatility: float
    session: str
    pair: str

@dataclass
class SpreadPrediction:
    """Spread prediction result"""
    predicted_spread: float
    confidence: float
    prediction_horizon: int  # seconds
    optimal_entry_time: datetime
    expected_spread_range: Tuple[float, float]
    model_used: str
    timestamp: datetime

@dataclass
class MarketMicrostructure:
    """Market microstructure features"""
    order_book_imbalance: float
    tick_direction: int  # -1, 0, 1
    trade_intensity: float
    price_impact: float
    liquidity_score: float

class SpreadPredictor:
    """
    Advanced spread prediction using ensemble ML models
    
    Features:
    - Multi-model ensemble prediction
    - Real-time feature engineering
    - Session-aware spread modeling
    - Volatility-adjusted predictions
    - Optimal entry timing calculation
    - Continuous model retraining
    """
    
    def __init__(self):
        """Initialize the spread predictor"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'linear': LinearRegression()
        }
        
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        self.feature_columns = [
            'volume', 'volatility', 'tick_direction', 'trade_intensity',
            'order_book_imbalance', 'price_impact', 'liquidity_score',
            'hour', 'minute', 'session_encoded', 'spread_ma_5', 'spread_ma_20',
            'volume_ma_5', 'volatility_ma_10', 'spread_std_5'
        ]
        
        self.session_encoding = {
            'Asian': 0,
            'London': 1,
            'NY': 2,
            'Overlap': 3
        }
        
        self.model_weights = {
            'random_forest': 0.4,
            'gradient_boost': 0.4,
            'linear': 0.2
        }
        
        self.is_trained = False
        self.training_data = []
        self.prediction_history = []
        self.performance_metrics = {}
        
    async def predict_spread(
        self,
        current_data: SpreadData,
        microstructure: MarketMicrostructure,
        prediction_horizon: int = 30  # seconds
    ) -> SpreadPrediction:
        """
        Predict future spread with confidence estimation
        
        Args:
            current_data: Current market data
            microstructure: Market microstructure features
            prediction_horizon: Prediction time horizon in seconds
            
        Returns:
            SpreadPrediction with detailed forecast
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained, using fallback prediction")
                return self._fallback_prediction(current_data, prediction_horizon)
            
            # Engineer features
            features = await self._engineer_features(current_data, microstructure)
            
            # Make ensemble prediction
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    scaled_features = self.scalers['standard'].transform([features])
                    
                    # Predict
                    pred = model.predict(scaled_features)[0]
                    predictions[model_name] = max(0.0001, pred)  # Minimum spread
                    
                    # Calculate confidence (simplified)
                    if hasattr(model, 'predict_proba'):
                        confidence = 0.8  # Default for tree-based models
                    else:
                        confidence = 0.6  # Default for linear models
                    
                    confidences[model_name] = confidence
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
                    predictions[model_name] = current_data.spread
                    confidences[model_name] = 0.3
            
            # Ensemble prediction
            weighted_prediction = sum(
                predictions[model] * self.model_weights[model]
                for model in predictions
            )
            
            # Ensemble confidence
            weighted_confidence = sum(
                confidences[model] * self.model_weights[model]
                for model in confidences
            )
            
            # Calculate prediction range
            prediction_std = np.std(list(predictions.values()))
            spread_range = (
                max(0.0001, weighted_prediction - prediction_std),
                weighted_prediction + prediction_std
            )
            
            # Calculate optimal entry time
            optimal_entry = self._calculate_optimal_entry_time(
                current_data, weighted_prediction, prediction_horizon
            )
            
            # Create prediction result
            prediction = SpreadPrediction(
                predicted_spread=weighted_prediction,
                confidence=weighted_confidence,
                prediction_horizon=prediction_horizon,
                optimal_entry_time=optimal_entry,
                expected_spread_range=spread_range,
                model_used="ensemble",
                timestamp=datetime.now()
            )
            
            # Store prediction
            self.prediction_history.append(prediction)
            
            logger.info(f"Spread predicted: {weighted_prediction:.5f} (confidence: {weighted_confidence:.3f})")
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting spread: {e}")
            return self._fallback_prediction(current_data, prediction_horizon)
    
    async def _engineer_features(
        self,
        data: SpreadData,
        microstructure: MarketMicrostructure
    ) -> List[float]:
        """Engineer features for prediction"""
        # Get historical data for moving averages (simplified)
        recent_spreads = [d.spread for d in self.training_data[-20:]] if self.training_data else [data.spread] * 20
        recent_volumes = [d.volume for d in self.training_data[-20:]] if self.training_data else [data.volume] * 20
        recent_volatilities = [d.volatility for d in self.training_data[-20:]] if self.training_data else [data.volatility] * 20
        
        # Calculate moving averages
        spread_ma_5 = np.mean(recent_spreads[-5:])
        spread_ma_20 = np.mean(recent_spreads)
        volume_ma_5 = np.mean(recent_volumes[-5:])
        volatility_ma_10 = np.mean(recent_volatilities[-10:])
        spread_std_5 = np.std(recent_spreads[-5:])
        
        # Time features
        hour = data.timestamp.hour
        minute = data.timestamp.minute
        session_encoded = self.session_encoding.get(data.session, 0)
        
        # Compile features
        features = [
            data.volume,
            data.volatility,
            microstructure.tick_direction,
            microstructure.trade_intensity,
            microstructure.order_book_imbalance,
            microstructure.price_impact,
            microstructure.liquidity_score,
            hour,
            minute,
            session_encoded,
            spread_ma_5,
            spread_ma_20,
            volume_ma_5,
            volatility_ma_10,
            spread_std_5
        ]
        
        return features
    
    def _calculate_optimal_entry_time(
        self,
        current_data: SpreadData,
        predicted_spread: float,
        horizon: int
    ) -> datetime:
        """Calculate optimal entry timing"""
        current_spread = current_data.spread
        
        if predicted_spread < current_spread:
            # Spread expected to tighten - enter soon
            delay_seconds = min(10, horizon // 3)
        else:
            # Spread expected to widen - wait
            delay_seconds = min(horizon - 5, horizon // 2)
        
        return current_data.timestamp + timedelta(seconds=delay_seconds)
    
    def _fallback_prediction(
        self,
        data: SpreadData,
        horizon: int
    ) -> SpreadPrediction:
        """Fallback prediction when models are not available"""
        # Simple heuristic based on session and volatility
        session_multipliers = {
            'Asian': 0.9,
            'London': 1.1,
            'NY': 1.05,
            'Overlap': 1.2
        }
        
        volatility_adjustment = 1.0 + (data.volatility - 0.5) * 0.3
        session_adjustment = session_multipliers.get(data.session, 1.0)
        
        predicted_spread = data.spread * volatility_adjustment * session_adjustment
        
        return SpreadPrediction(
            predicted_spread=predicted_spread,
            confidence=0.4,  # Low confidence for fallback
            prediction_horizon=horizon,
            optimal_entry_time=data.timestamp + timedelta(seconds=horizon//2),
            expected_spread_range=(predicted_spread * 0.8, predicted_spread * 1.2),
            model_used="fallback",
            timestamp=datetime.now()
        )
    
    async def train_models(
        self,
        training_data: List[SpreadData],
        microstructure_data: List[MarketMicrostructure]
    ) -> Dict[str, float]:
        """
        Train all models with historical data
        
        Args:
            training_data: Historical spread data
            microstructure_data: Corresponding microstructure data
            
        Returns:
            Training performance metrics
        """
        try:
            if len(training_data) < 100:
                logger.warning("Insufficient training data")
                return {}
            
            logger.info(f"Training models with {len(training_data)} samples")
            
            # Prepare features and targets
            features = []
            targets = []
            
            for i, (data, micro) in enumerate(zip(training_data[:-1], microstructure_data[:-1])):
                # Use next spread as target
                target_spread = training_data[i + 1].spread
                
                # Engineer features
                feature_vector = await self._engineer_features(data, micro)
                
                features.append(feature_vector)
                targets.append(target_spread)
            
            features = np.array(features)
            targets = np.array(targets)
            
            # Split data
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            # Scale features
            X_train_scaled = self.scalers['standard'].fit_transform(X_train)
            X_test_scaled = self.scalers['standard'].transform(X_test)
            
            # Train models
            performance = {}
            
            for model_name, model in self.models.items():
                try:
                    # Train
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    performance[model_name] = {
                        'mae': mae,
                        'mse': mse,
                        'r2': r2
                    }
                    
                    logger.info(f"Model {model_name} - MAE: {mae:.6f}, R2: {r2:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training model {model_name}: {e}")
                    performance[model_name] = {'error': str(e)}
            
            self.is_trained = True
            self.training_data = training_data
            self.performance_metrics = performance
            
            logger.info("Model training completed successfully")
            return performance
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return {
            'is_trained': self.is_trained,
            'training_samples': len(self.training_data),
            'predictions_made': len(self.prediction_history),
            'model_performance': self.performance_metrics,
            'recent_predictions': len([p for p in self.prediction_history 
                                    if (datetime.now() - p.timestamp).total_seconds() < 3600])
        }
    
    async def update_model_weights(self, performance_feedback: Dict[str, float]):
        """Update ensemble weights based on performance feedback"""
        total_performance = sum(performance_feedback.values())
        
        if total_performance > 0:
            for model_name in self.model_weights:
                if model_name in performance_feedback:
                    # Update weight based on relative performance
                    self.model_weights[model_name] = performance_feedback[model_name] / total_performance
            
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight
            
            logger.info(f"Updated model weights: {self.model_weights}")
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'model_weights': self.model_weights,
                'is_trained': self.is_trained,
                'performance_metrics': self.performance_metrics
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_weights = model_data['model_weights']
            self.is_trained = model_data['is_trained']
            self.performance_metrics = model_data['performance_metrics']
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def test_spread_predictor():
        predictor = SpreadPredictor()
        
        # Create test data
        current_data = SpreadData(
            timestamp=datetime.now(),
            bid=1.25000,
            ask=1.25020,
            spread=0.00020,
            volume=1000,
            volatility=0.5,
            session="London",
            pair="EURUSD"
        )
        
        microstructure = MarketMicrostructure(
            order_book_imbalance=0.1,
            tick_direction=1,
            trade_intensity=0.7,
            price_impact=0.3,
            liquidity_score=0.8
        )
        
        # Make prediction (will use fallback since not trained)
        prediction = await predictor.predict_spread(current_data, microstructure)
        
        print(f"Predicted Spread: {prediction.predicted_spread:.5f}")
        print(f"Confidence: {prediction.confidence:.3f}")
        print(f"Optimal Entry: {prediction.optimal_entry_time}")
        print(f"Expected Range: {prediction.expected_spread_range}")
        print(f"Model Used: {prediction.model_used}")
    
    # Run test
    asyncio.run(test_spread_predictor())
