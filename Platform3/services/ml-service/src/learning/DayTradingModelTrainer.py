"""
Platform3 Forex Trading Platform
Day Trading Model Trainer - Intraday Pattern Learning

This module provides specialized training for day trading models optimized
for M15-H1 timeframes with session-based pattern recognition.

Author: Platform3 Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Trading sessions for day trading"""
    ASIAN = "asian"
    LONDON = "london"
    NY = "ny"
    OVERLAP_LONDON_NY = "overlap_london_ny"

class SignalType(Enum):
    """Day trading signal types"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

class TimeFrame(Enum):
    """Day trading timeframes"""
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"

@dataclass
class DayTradingFeatures:
    """Day trading feature set"""
    timestamp: datetime
    session: TradingSession
    timeframe: TimeFrame
    price_features: Dict[str, float]
    momentum_features: Dict[str, float]
    volatility_features: Dict[str, float]
    volume_features: Dict[str, float]
    session_features: Dict[str, float]
    target: SignalType

@dataclass
class TrainingResult:
    """Training result with performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    feature_importance: Dict[str, float]
    confusion_matrix: np.ndarray
    training_time: float
    cross_val_scores: List[float]

class DayTradingModelTrainer:
    """
    Specialized trainer for day trading models
    
    Features:
    - Session-aware feature engineering
    - Multi-timeframe pattern recognition
    - Momentum and breakout detection
    - Volume confirmation training
    - Real-time model validation
    - Performance optimization for intraday trading
    """
    
    def __init__(self):
        """Initialize the day trading model trainer"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.trained_models = {}
        self.training_history = []
        
        # Session-specific parameters
        self.session_weights = {
            TradingSession.ASIAN: 0.8,      # Lower volatility
            TradingSession.LONDON: 1.2,     # High volatility
            TradingSession.NY: 1.1,         # High volatility
            TradingSession.OVERLAP_LONDON_NY: 1.3  # Highest volatility
        }
        
        # Timeframe importance weights
        self.timeframe_weights = {
            TimeFrame.M15: 0.3,
            TimeFrame.M30: 0.4,
            TimeFrame.H1: 0.3
        }
        
    async def train_models(
        self,
        training_data: List[DayTradingFeatures],
        validation_split: float = 0.2
    ) -> Dict[str, TrainingResult]:
        """
        Train day trading models with comprehensive evaluation
        
        Args:
            training_data: List of training features
            validation_split: Validation data percentage
            
        Returns:
            Dictionary of training results for each model
        """
        try:
            if len(training_data) < 100:
                logger.warning("Insufficient training data for day trading models")
                return {}
            
            logger.info(f"Training day trading models with {len(training_data)} samples")
            
            # Prepare features and targets
            X, y, feature_names = await self._prepare_training_data(training_data)
            self.feature_names = feature_names
            
            # Split data chronologically
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers['main'] = scaler
            
            # Train models
            results = {}
            
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name} model...")
                
                start_time = datetime.now()
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                result = await self._evaluate_model(
                    model, model_name, X_val_scaled, y_val, feature_names
                )
                
                # Calculate training time
                training_time = (datetime.now() - start_time).total_seconds()
                result.training_time = training_time
                
                # Cross-validation
                cv_scores = await self._cross_validate_model(
                    model, X_train_scaled, y_train
                )
                result.cross_val_scores = cv_scores
                
                results[model_name] = result
                self.trained_models[model_name] = model
                
                logger.info(f"{model_name} - Accuracy: {result.accuracy:.4f}, "
                           f"AUC: {result.auc_score:.4f}")
            
            # Store training history
            self.training_history.append({
                'timestamp': datetime.now(),
                'samples': len(training_data),
                'results': results
            })
            
            logger.info("Day trading model training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training day trading models: {e}")
            return {}
    
    async def _prepare_training_data(
        self,
        training_data: List[DayTradingFeatures]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and targets for training"""
        features = []
        targets = []
        feature_names = []
        
        for data in training_data:
            # Extract all features
            feature_vector = []
            
            # Price features
            for name, value in data.price_features.items():
                feature_vector.append(value)
                if not feature_names:
                    feature_names.append(f"price_{name}")
            
            # Momentum features
            for name, value in data.momentum_features.items():
                feature_vector.append(value)
                if len(feature_names) == len(data.price_features):
                    feature_names.append(f"momentum_{name}")
            
            # Volatility features
            for name, value in data.volatility_features.items():
                feature_vector.append(value)
                if len(feature_names) == len(data.price_features) + len(data.momentum_features):
                    feature_names.append(f"volatility_{name}")
            
            # Volume features
            for name, value in data.volume_features.items():
                feature_vector.append(value)
                if len(feature_names) == (len(data.price_features) + 
                                        len(data.momentum_features) + 
                                        len(data.volatility_features)):
                    feature_names.append(f"volume_{name}")
            
            # Session features
            for name, value in data.session_features.items():
                feature_vector.append(value)
                if len(feature_names) == (len(data.price_features) + 
                                        len(data.momentum_features) + 
                                        len(data.volatility_features) +
                                        len(data.volume_features)):
                    feature_names.append(f"session_{name}")
            
            # Add session encoding
            session_encoding = self._encode_session(data.session)
            feature_vector.extend(session_encoding)
            if len(feature_names) == (len(data.price_features) + 
                                    len(data.momentum_features) + 
                                    len(data.volatility_features) +
                                    len(data.volume_features) +
                                    len(data.session_features)):
                feature_names.extend(['session_asian', 'session_london', 'session_ny', 'session_overlap'])
            
            # Add timeframe encoding
            timeframe_encoding = self._encode_timeframe(data.timeframe)
            feature_vector.extend(timeframe_encoding)
            if len(feature_names) == (len(data.price_features) + 
                                    len(data.momentum_features) + 
                                    len(data.volatility_features) +
                                    len(data.volume_features) +
                                    len(data.session_features) + 4):
                feature_names.extend(['timeframe_m15', 'timeframe_m30', 'timeframe_h1'])
            
            features.append(feature_vector)
            targets.append(data.target.value)
        
        return np.array(features), np.array(targets), feature_names
    
    def _encode_session(self, session: TradingSession) -> List[float]:
        """One-hot encode trading session"""
        encoding = [0.0, 0.0, 0.0, 0.0]
        if session == TradingSession.ASIAN:
            encoding[0] = 1.0
        elif session == TradingSession.LONDON:
            encoding[1] = 1.0
        elif session == TradingSession.NY:
            encoding[2] = 1.0
        elif session == TradingSession.OVERLAP_LONDON_NY:
            encoding[3] = 1.0
        return encoding
    
    def _encode_timeframe(self, timeframe: TimeFrame) -> List[float]:
        """One-hot encode timeframe"""
        encoding = [0.0, 0.0, 0.0]
        if timeframe == TimeFrame.M15:
            encoding[0] = 1.0
        elif timeframe == TimeFrame.M30:
            encoding[1] = 1.0
        elif timeframe == TimeFrame.H1:
            encoding[2] = 1.0
        return encoding
    
    async def _evaluate_model(
        self,
        model: Any,
        model_name: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str]
    ) -> TrainingResult:
        """Evaluate trained model performance"""
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # AUC score (for multiclass)
        try:
            if y_pred_proba is not None:
                auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                auc = 0.0
        except:
            auc = 0.0
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = importance
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        return TrainingResult(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            feature_importance=feature_importance,
            confusion_matrix=cm,
            training_time=0.0,  # Will be set by caller
            cross_val_scores=[]  # Will be set by caller
        )
    
    async def _cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> List[float]:
        """Perform time series cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            return cv_scores.tolist()
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return []
    
    async def predict_signal(
        self,
        features: DayTradingFeatures,
        model_name: str = 'random_forest'
    ) -> Tuple[SignalType, float]:
        """
        Predict trading signal for given features
        
        Args:
            features: Day trading features
            model_name: Model to use for prediction
            
        Returns:
            Tuple of (predicted signal, confidence)
        """
        try:
            if model_name not in self.trained_models:
                logger.warning(f"Model {model_name} not trained")
                return SignalType.HOLD, 0.0
            
            # Prepare features
            feature_vector = await self._prepare_single_prediction(features)
            
            # Scale features
            if 'main' in self.scalers:
                feature_vector_scaled = self.scalers['main'].transform([feature_vector])
            else:
                feature_vector_scaled = [feature_vector]
            
            # Make prediction
            model = self.trained_models[model_name]
            prediction = model.predict(feature_vector_scaled)[0]
            
            # Get confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_vector_scaled)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.7  # Default confidence
            
            # Convert to signal type
            signal_type = SignalType(prediction)
            
            return signal_type, confidence
            
        except Exception as e:
            logger.error(f"Error predicting signal: {e}")
            return SignalType.HOLD, 0.0
    
    async def _prepare_single_prediction(self, features: DayTradingFeatures) -> List[float]:
        """Prepare features for single prediction"""
        feature_vector = []
        
        # Add all feature categories
        for value in features.price_features.values():
            feature_vector.append(value)
        
        for value in features.momentum_features.values():
            feature_vector.append(value)
        
        for value in features.volatility_features.values():
            feature_vector.append(value)
        
        for value in features.volume_features.values():
            feature_vector.append(value)
        
        for value in features.session_features.values():
            feature_vector.append(value)
        
        # Add encodings
        feature_vector.extend(self._encode_session(features.session))
        feature_vector.extend(self._encode_timeframe(features.timeframe))
        
        return feature_vector
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance summary"""
        if not self.training_history:
            return {}
        
        latest_training = self.training_history[-1]
        
        return {
            'last_training': latest_training['timestamp'].isoformat(),
            'training_samples': latest_training['samples'],
            'model_performance': {
                name: {
                    'accuracy': result.accuracy,
                    'f1_score': result.f1_score,
                    'auc_score': result.auc_score,
                    'training_time': result.training_time
                }
                for name, result in latest_training['results'].items()
            },
            'best_model': max(
                latest_training['results'].items(),
                key=lambda x: x[1].accuracy
            )[0] if latest_training['results'] else None,
            'feature_importance': {
                name: dict(list(result.feature_importance.items())[:10])  # Top 10 features
                for name, result in latest_training['results'].items()
            }
        }
    
    def save_models(self, filepath: str):
        """Save trained models"""
        try:
            import joblib
            model_data = {
                'models': self.trained_models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'training_history': self.training_history
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Day trading models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        try:
            import joblib
            model_data = joblib.load(filepath)
            self.trained_models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data['training_history']
            logger.info(f"Day trading models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# Example usage and testing
if __name__ == "__main__":
    async def test_day_trading_trainer():
        trainer = DayTradingModelTrainer()
        
        # Create test training data
        training_data = []
        for i in range(500):
            features = DayTradingFeatures(
                timestamp=datetime.now() + timedelta(minutes=i*15),
                session=TradingSession.LONDON,
                timeframe=TimeFrame.M15,
                price_features={
                    'close': 1.2500 + np.random.normal(0, 0.001),
                    'high': 1.2510 + np.random.normal(0, 0.001),
                    'low': 1.2490 + np.random.normal(0, 0.001),
                    'rsi': np.random.uniform(30, 70)
                },
                momentum_features={
                    'macd': np.random.normal(0, 0.0001),
                    'stochastic': np.random.uniform(20, 80),
                    'williams_r': np.random.uniform(-80, -20)
                },
                volatility_features={
                    'atr': np.random.uniform(0.0005, 0.002),
                    'bollinger_width': np.random.uniform(0.001, 0.005)
                },
                volume_features={
                    'volume': np.random.uniform(1000, 5000),
                    'volume_sma': np.random.uniform(1500, 4000)
                },
                session_features={
                    'session_volatility': np.random.uniform(0.3, 0.8),
                    'session_volume': np.random.uniform(0.5, 1.5)
                },
                target=SignalType(np.random.choice([-2, -1, 0, 1, 2]))
            )
            training_data.append(features)
        
        # Train models
        results = await trainer.train_models(training_data)
        
        print("Training Results:")
        for model_name, result in results.items():
            print(f"{model_name}:")
            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  F1 Score: {result.f1_score:.4f}")
            print(f"  AUC Score: {result.auc_score:.4f}")
            print(f"  Training Time: {result.training_time:.2f}s")
        
        # Test prediction
        test_features = training_data[0]
        signal, confidence = await trainer.predict_signal(test_features)
        print(f"\nPrediction: {signal.name} (confidence: {confidence:.3f})")
    
    # Run test
    asyncio.run(test_day_trading_trainer())
