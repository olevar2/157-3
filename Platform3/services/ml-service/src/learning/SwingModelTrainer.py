"""
Platform3 Forex Trading Platform
Swing Model Trainer - Short-Term Swing Learning

This module provides specialized training for swing trading models optimized
for H4 timeframes with 1-5 day maximum holding periods.

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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Trading sessions for swing trading"""
    ASIAN = "asian"
    LONDON = "london"
    NY = "ny"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    WEEKEND = "weekend"

class SwingSignalType(Enum):
    """Swing trading signal types"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

class SwingTimeFrame(Enum):
    """Swing trading timeframes"""
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SwingPattern(Enum):
    """Swing trading patterns"""
    ELLIOTT_WAVE_5 = "elliott_wave_5"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    SUPPORT_RESISTANCE = "support_resistance"
    TREND_CONTINUATION = "trend_continuation"
    REVERSAL_PATTERN = "reversal_pattern"

@dataclass
class SwingTradingFeatures:
    """Swing trading feature set"""
    timestamp: datetime
    session: TradingSession
    timeframe: SwingTimeFrame
    price_features: Dict[str, float]
    pattern_features: Dict[str, float]
    momentum_features: Dict[str, float]
    volatility_features: Dict[str, float]
    volume_features: Dict[str, float]
    fibonacci_features: Dict[str, float]
    elliott_wave_features: Dict[str, float]
    support_resistance_features: Dict[str, float]
    multi_timeframe_features: Dict[str, float]
    target: SwingSignalType
    holding_period_days: int  # Expected holding period (1-5 days)

@dataclass
class SwingTrainingResult:
    """Swing training result with performance metrics"""
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
    pattern_accuracy: Dict[str, float]  # Accuracy per pattern type
    holding_period_performance: Dict[int, float]  # Performance per holding period

class SwingModelTrainer:
    """
    Specialized trainer for swing trading models

    Features:
    - H4 timeframe optimization for 1-5 day trades
    - Elliott wave pattern recognition
    - Fibonacci retracement analysis
    - Multi-timeframe confluence detection
    - Support/resistance level training
    - Session-aware pattern recognition
    - Risk-adjusted position sizing for swing trades
    - Performance optimization for swing trading
    """

    def __init__(self):
        """Initialize the swing trading model trainer"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=120,
                max_depth=10,
                learning_rate=0.08,
                random_state=42,
                subsample=0.8
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',
                solver='liblinear'
            )
        }

        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.trained_models = {}
        self.training_history = []

        # Session-specific parameters for swing trading
        self.session_weights = {
            TradingSession.ASIAN: 0.7,      # Lower volatility, longer patterns
            TradingSession.LONDON: 1.3,     # High volatility, strong patterns
            TradingSession.NY: 1.2,         # High volatility, trend continuation
            TradingSession.OVERLAP_LONDON_NY: 1.4,  # Highest volatility, breakouts
            TradingSession.WEEKEND: 0.5     # Gap analysis
        }

        # Timeframe importance weights for swing trading
        self.timeframe_weights = {
            SwingTimeFrame.H1: 0.2,   # Confirmation timeframe
            SwingTimeFrame.H4: 0.6,   # Primary swing timeframe
            SwingTimeFrame.D1: 0.2    # Trend direction
        }

        # Pattern-specific weights
        self.pattern_weights = {
            SwingPattern.ELLIOTT_WAVE_5: 1.2,
            SwingPattern.FIBONACCI_RETRACEMENT: 1.1,
            SwingPattern.SUPPORT_RESISTANCE: 1.0,
            SwingPattern.TREND_CONTINUATION: 0.9,
            SwingPattern.REVERSAL_PATTERN: 1.3
        }

        # Holding period weights (1-5 days)
        self.holding_period_weights = {
            1: 0.8,  # Short swing
            2: 1.0,  # Optimal swing
            3: 1.2,  # Extended swing
            4: 1.1,  # Long swing
            5: 0.9   # Maximum swing
        }

    async def train_models(
        self,
        training_data: List[SwingTradingFeatures],
        validation_split: float = 0.2
    ) -> Dict[str, SwingTrainingResult]:
        """
        Train swing trading models with comprehensive evaluation

        Args:
            training_data: List of swing trading features
            validation_split: Validation data percentage

        Returns:
            Dictionary of training results for each model
        """
        try:
            if len(training_data) < 200:
                logger.warning("Insufficient training data for swing trading models")
                return {}

            logger.info(f"Training swing trading models with {len(training_data)} samples")

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
                logger.info(f"Training {model_name} swing model...")

                start_time = datetime.now()

                # Train model
                model.fit(X_train_scaled, y_train)

                # Evaluate model
                result = await self._evaluate_model(
                    model, model_name, X_val_scaled, y_val, feature_names, training_data[split_idx:]
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

            logger.info("Swing trading model training completed")
            return results

        except Exception as e:
            logger.error(f"Error training swing trading models: {e}")
            return {}

    async def _prepare_training_data(
        self,
        training_data: List[SwingTradingFeatures]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and targets for swing trading training"""
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

            # Pattern features (swing-specific)
            for name, value in data.pattern_features.items():
                feature_vector.append(value)
                if len(feature_names) == len(data.price_features):
                    feature_names.append(f"pattern_{name}")

            # Momentum features
            for name, value in data.momentum_features.items():
                feature_vector.append(value)
                if len(feature_names) == len(data.price_features) + len(data.pattern_features):
                    feature_names.append(f"momentum_{name}")

            # Volatility features
            for name, value in data.volatility_features.items():
                feature_vector.append(value)
                if len(feature_names) == (len(data.price_features) + len(data.pattern_features) +
                                        len(data.momentum_features)):
                    feature_names.append(f"volatility_{name}")

            # Volume features
            for name, value in data.volume_features.items():
                feature_vector.append(value)
                if len(feature_names) == (len(data.price_features) + len(data.pattern_features) +
                                        len(data.momentum_features) + len(data.volatility_features)):
                    feature_names.append(f"volume_{name}")

            # Fibonacci features (swing-specific)
            for name, value in data.fibonacci_features.items():
                feature_vector.append(value)
                if len(feature_names) == (len(data.price_features) + len(data.pattern_features) +
                                        len(data.momentum_features) + len(data.volatility_features) +
                                        len(data.volume_features)):
                    feature_names.append(f"fibonacci_{name}")

            # Elliott wave features (swing-specific)
            for name, value in data.elliott_wave_features.items():
                feature_vector.append(value)
                if len(feature_names) == (len(data.price_features) + len(data.pattern_features) +
                                        len(data.momentum_features) + len(data.volatility_features) +
                                        len(data.volume_features) + len(data.fibonacci_features)):
                    feature_names.append(f"elliott_{name}")

            # Support/Resistance features (swing-specific)
            for name, value in data.support_resistance_features.items():
                feature_vector.append(value)
                if len(feature_names) == (len(data.price_features) + len(data.pattern_features) +
                                        len(data.momentum_features) + len(data.volatility_features) +
                                        len(data.volume_features) + len(data.fibonacci_features) +
                                        len(data.elliott_wave_features)):
                    feature_names.append(f"sr_{name}")

            # Multi-timeframe features (swing-specific)
            for name, value in data.multi_timeframe_features.items():
                feature_vector.append(value)
                if len(feature_names) == (len(data.price_features) + len(data.pattern_features) +
                                        len(data.momentum_features) + len(data.volatility_features) +
                                        len(data.volume_features) + len(data.fibonacci_features) +
                                        len(data.elliott_wave_features) + len(data.support_resistance_features)):
                    feature_names.append(f"mtf_{name}")

            # Add session encoding
            session_encoding = self._encode_session(data.session)
            feature_vector.extend(session_encoding)
            if len(feature_names) == (len(data.price_features) + len(data.pattern_features) +
                                    len(data.momentum_features) + len(data.volatility_features) +
                                    len(data.volume_features) + len(data.fibonacci_features) +
                                    len(data.elliott_wave_features) + len(data.support_resistance_features) +
                                    len(data.multi_timeframe_features)):
                feature_names.extend(['session_asian', 'session_london', 'session_ny', 'session_overlap', 'session_weekend'])

            # Add timeframe encoding
            timeframe_encoding = self._encode_timeframe(data.timeframe)
            feature_vector.extend(timeframe_encoding)
            if len(feature_names) == (len(data.price_features) + len(data.pattern_features) +
                                    len(data.momentum_features) + len(data.volatility_features) +
                                    len(data.volume_features) + len(data.fibonacci_features) +
                                    len(data.elliott_wave_features) + len(data.support_resistance_features) +
                                    len(data.multi_timeframe_features) + 5):
                feature_names.extend(['timeframe_h1', 'timeframe_h4', 'timeframe_d1'])

            # Add holding period encoding
            holding_period_encoding = self._encode_holding_period(data.holding_period_days)
            feature_vector.extend(holding_period_encoding)
            if len(feature_names) == (len(data.price_features) + len(data.pattern_features) +
                                    len(data.momentum_features) + len(data.volatility_features) +
                                    len(data.volume_features) + len(data.fibonacci_features) +
                                    len(data.elliott_wave_features) + len(data.support_resistance_features) +
                                    len(data.multi_timeframe_features) + 8):
                feature_names.extend(['holding_1d', 'holding_2d', 'holding_3d', 'holding_4d', 'holding_5d'])

            features.append(feature_vector)
            targets.append(data.target.value)

        return np.array(features), np.array(targets), feature_names

    def _encode_session(self, session: TradingSession) -> List[float]:
        """One-hot encode trading session"""
        encoding = [0.0, 0.0, 0.0, 0.0, 0.0]
        if session == TradingSession.ASIAN:
            encoding[0] = 1.0
        elif session == TradingSession.LONDON:
            encoding[1] = 1.0
        elif session == TradingSession.NY:
            encoding[2] = 1.0
        elif session == TradingSession.OVERLAP_LONDON_NY:
            encoding[3] = 1.0
        elif session == TradingSession.WEEKEND:
            encoding[4] = 1.0
        return encoding

    def _encode_timeframe(self, timeframe: SwingTimeFrame) -> List[float]:
        """One-hot encode timeframe"""
        encoding = [0.0, 0.0, 0.0]
        if timeframe == SwingTimeFrame.H1:
            encoding[0] = 1.0
        elif timeframe == SwingTimeFrame.H4:
            encoding[1] = 1.0
        elif timeframe == SwingTimeFrame.D1:
            encoding[2] = 1.0
        return encoding

    def _encode_holding_period(self, holding_period: int) -> List[float]:
        """One-hot encode holding period (1-5 days)"""
        encoding = [0.0, 0.0, 0.0, 0.0, 0.0]
        if 1 <= holding_period <= 5:
            encoding[holding_period - 1] = 1.0
        return encoding

    async def _evaluate_model(
        self,
        model: Any,
        model_name: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        validation_data: List[SwingTradingFeatures]
    ) -> SwingTrainingResult:
        """Evaluate trained swing model performance"""
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

        # Pattern-specific accuracy analysis
        pattern_accuracy = await self._analyze_pattern_accuracy(
            y_val, y_pred, validation_data
        )

        # Holding period performance analysis
        holding_period_performance = await self._analyze_holding_period_performance(
            y_val, y_pred, validation_data
        )

        return SwingTrainingResult(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            feature_importance=feature_importance,
            confusion_matrix=cm,
            training_time=0.0,  # Will be set by caller
            cross_val_scores=[],  # Will be set by caller
            pattern_accuracy=pattern_accuracy,
            holding_period_performance=holding_period_performance
        )

    async def _analyze_pattern_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        validation_data: List[SwingTradingFeatures]
    ) -> Dict[str, float]:
        """Analyze accuracy by pattern type"""
        pattern_accuracy = {}

        # Group predictions by pattern features
        pattern_groups = {}
        for i, data in enumerate(validation_data):
            # Determine dominant pattern based on feature values
            dominant_pattern = self._get_dominant_pattern(data.pattern_features)
            if dominant_pattern not in pattern_groups:
                pattern_groups[dominant_pattern] = []
            pattern_groups[dominant_pattern].append(i)

        # Calculate accuracy for each pattern
        for pattern, indices in pattern_groups.items():
            if len(indices) > 0:
                pattern_y_true = y_true[indices]
                pattern_y_pred = y_pred[indices]
                pattern_accuracy[pattern] = accuracy_score(pattern_y_true, pattern_y_pred)

        return pattern_accuracy

    async def _analyze_holding_period_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        validation_data: List[SwingTradingFeatures]
    ) -> Dict[int, float]:
        """Analyze performance by holding period"""
        holding_period_performance = {}

        # Group predictions by holding period
        period_groups = {}
        for i, data in enumerate(validation_data):
            period = data.holding_period_days
            if period not in period_groups:
                period_groups[period] = []
            period_groups[period].append(i)

        # Calculate accuracy for each holding period
        for period, indices in period_groups.items():
            if len(indices) > 0:
                period_y_true = y_true[indices]
                period_y_pred = y_pred[indices]
                holding_period_performance[period] = accuracy_score(period_y_true, period_y_pred)

        return holding_period_performance

    def _get_dominant_pattern(self, pattern_features: Dict[str, float]) -> str:
        """Determine dominant pattern from features"""
        if not pattern_features:
            return "unknown"

        # Find the pattern with highest confidence
        max_value = max(pattern_features.values())
        for pattern_name, value in pattern_features.items():
            if value == max_value:
                return pattern_name

        return "unknown"

    async def _cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> List[float]:
        """Perform time series cross-validation for swing trading"""
        try:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            return cv_scores.tolist()
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return []

    async def predict_signal(
        self,
        features: SwingTradingFeatures,
        model_name: str = 'random_forest'
    ) -> Tuple[SwingSignalType, float, Dict[str, Any]]:
        """
        Predict swing trading signal for given features

        Args:
            features: Swing trading features
            model_name: Model to use for prediction

        Returns:
            Tuple of (predicted signal, confidence, additional_info)
        """
        try:
            if model_name not in self.trained_models:
                logger.warning(f"Model {model_name} not trained")
                return SwingSignalType.HOLD, 0.0, {}

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

            # Get confidence and probabilities
            additional_info = {}
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_vector_scaled)[0]
                confidence = np.max(probabilities)

                # Add probability distribution
                signal_types = [SwingSignalType.STRONG_SELL, SwingSignalType.SELL,
                              SwingSignalType.HOLD, SwingSignalType.BUY, SwingSignalType.STRONG_BUY]
                additional_info['probabilities'] = {
                    signal.name: prob for signal, prob in zip(signal_types, probabilities)
                }
            else:
                confidence = 0.7  # Default confidence

            # Add pattern analysis
            additional_info['dominant_pattern'] = self._get_dominant_pattern(features.pattern_features)
            additional_info['holding_period'] = features.holding_period_days
            additional_info['session'] = features.session.value
            additional_info['timeframe'] = features.timeframe.value

            # Convert to signal type
            signal_type = SwingSignalType(prediction)

            return signal_type, confidence, additional_info

        except Exception as e:
            logger.error(f"Error predicting swing signal: {e}")
            return SwingSignalType.HOLD, 0.0, {}

    async def _prepare_single_prediction(self, features: SwingTradingFeatures) -> List[float]:
        """Prepare features for single swing prediction"""
        feature_vector = []

        # Add all feature categories in the same order as training
        for value in features.price_features.values():
            feature_vector.append(value)

        for value in features.pattern_features.values():
            feature_vector.append(value)

        for value in features.momentum_features.values():
            feature_vector.append(value)

        for value in features.volatility_features.values():
            feature_vector.append(value)

        for value in features.volume_features.values():
            feature_vector.append(value)

        for value in features.fibonacci_features.values():
            feature_vector.append(value)

        for value in features.elliott_wave_features.values():
            feature_vector.append(value)

        for value in features.support_resistance_features.values():
            feature_vector.append(value)

        for value in features.multi_timeframe_features.values():
            feature_vector.append(value)

        # Add encodings
        feature_vector.extend(self._encode_session(features.session))
        feature_vector.extend(self._encode_timeframe(features.timeframe))
        feature_vector.extend(self._encode_holding_period(features.holding_period_days))

        return feature_vector

    def get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive swing model performance summary"""
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
                    'training_time': result.training_time,
                    'pattern_accuracy': result.pattern_accuracy,
                    'holding_period_performance': result.holding_period_performance
                }
                for name, result in latest_training['results'].items()
            },
            'best_model': max(
                latest_training['results'].items(),
                key=lambda x: x[1].accuracy
            )[0] if latest_training['results'] else None,
            'feature_importance': {
                name: dict(list(result.feature_importance.items())[:15])  # Top 15 features
                for name, result in latest_training['results'].items()
            },
            'pattern_analysis': self._get_pattern_analysis(latest_training['results']),
            'holding_period_analysis': self._get_holding_period_analysis(latest_training['results'])
        }

    def _get_pattern_analysis(self, results: Dict[str, SwingTrainingResult]) -> Dict[str, Any]:
        """Analyze pattern performance across models"""
        pattern_analysis = {}

        for model_name, result in results.items():
            for pattern, accuracy in result.pattern_accuracy.items():
                if pattern not in pattern_analysis:
                    pattern_analysis[pattern] = {}
                pattern_analysis[pattern][model_name] = accuracy

        # Calculate average accuracy per pattern
        for pattern in pattern_analysis:
            accuracies = list(pattern_analysis[pattern].values())
            pattern_analysis[pattern]['average'] = np.mean(accuracies) if accuracies else 0.0

        return pattern_analysis

    def _get_holding_period_analysis(self, results: Dict[str, SwingTrainingResult]) -> Dict[str, Any]:
        """Analyze holding period performance across models"""
        period_analysis = {}

        for model_name, result in results.items():
            for period, performance in result.holding_period_performance.items():
                period_key = f"{period}_days"
                if period_key not in period_analysis:
                    period_analysis[period_key] = {}
                period_analysis[period_key][model_name] = performance

        # Calculate average performance per holding period
        for period_key in period_analysis:
            performances = list(period_analysis[period_key].values())
            period_analysis[period_key]['average'] = np.mean(performances) if performances else 0.0

        return period_analysis

    def save_models(self, filepath: str):
        """Save trained swing models"""
        try:
            import joblib
            model_data = {
                'models': self.trained_models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'session_weights': self.session_weights,
                'timeframe_weights': self.timeframe_weights,
                'pattern_weights': self.pattern_weights,
                'holding_period_weights': self.holding_period_weights
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Swing trading models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving swing models: {e}")

    def load_models(self, filepath: str):
        """Load trained swing models"""
        try:
            import joblib
            model_data = joblib.load(filepath)
            self.trained_models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data['training_history']

            # Load weights if available
            if 'session_weights' in model_data:
                self.session_weights = model_data['session_weights']
            if 'timeframe_weights' in model_data:
                self.timeframe_weights = model_data['timeframe_weights']
            if 'pattern_weights' in model_data:
                self.pattern_weights = model_data['pattern_weights']
            if 'holding_period_weights' in model_data:
                self.holding_period_weights = model_data['holding_period_weights']

            logger.info(f"Swing trading models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading swing models: {e}")

    def get_pattern_recommendations(self, features: SwingTradingFeatures) -> Dict[str, Any]:
        """Get pattern-based trading recommendations"""
        recommendations = {
            'dominant_pattern': self._get_dominant_pattern(features.pattern_features),
            'session_strength': self.session_weights.get(features.session, 1.0),
            'timeframe_importance': self.timeframe_weights.get(features.timeframe, 1.0),
            'holding_period_weight': self.holding_period_weights.get(features.holding_period_days, 1.0),
            'fibonacci_levels': self._analyze_fibonacci_levels(features.fibonacci_features),
            'elliott_wave_position': self._analyze_elliott_wave(features.elliott_wave_features),
            'support_resistance_strength': self._analyze_support_resistance(features.support_resistance_features)
        }

        # Calculate overall recommendation score
        pattern_score = features.pattern_features.get('confidence', 0.5)
        session_score = recommendations['session_strength']
        timeframe_score = recommendations['timeframe_importance']
        holding_score = recommendations['holding_period_weight']

        recommendations['overall_score'] = (pattern_score * session_score * timeframe_score * holding_score)
        recommendations['recommendation'] = self._get_recommendation_level(recommendations['overall_score'])

        return recommendations

    def _analyze_fibonacci_levels(self, fibonacci_features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze Fibonacci retracement levels"""
        if not fibonacci_features:
            return {'status': 'no_data'}

        # Common Fibonacci levels
        fib_levels = ['23.6', '38.2', '50.0', '61.8', '78.6']
        analysis = {}

        for level in fib_levels:
            level_key = f'fib_{level.replace(".", "_")}'
            if level_key in fibonacci_features:
                analysis[f'{level}%'] = {
                    'value': fibonacci_features[level_key],
                    'strength': 'strong' if fibonacci_features[level_key] > 0.7 else
                               'moderate' if fibonacci_features[level_key] > 0.4 else 'weak'
                }

        return analysis

    def _analyze_elliott_wave(self, elliott_features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze Elliott Wave position"""
        if not elliott_features:
            return {'status': 'no_data'}

        wave_analysis = {}
        wave_positions = ['wave_1', 'wave_2', 'wave_3', 'wave_4', 'wave_5']

        for wave in wave_positions:
            if wave in elliott_features:
                wave_analysis[wave] = {
                    'probability': elliott_features[wave],
                    'strength': 'high' if elliott_features[wave] > 0.7 else
                               'medium' if elliott_features[wave] > 0.4 else 'low'
                }

        # Determine most likely wave position
        if wave_analysis:
            most_likely_wave = max(wave_analysis.items(), key=lambda x: x[1]['probability'])
            wave_analysis['most_likely'] = most_likely_wave[0]
            wave_analysis['confidence'] = most_likely_wave[1]['probability']

        return wave_analysis

    def _analyze_support_resistance(self, sr_features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze support and resistance levels"""
        if not sr_features:
            return {'status': 'no_data'}

        sr_analysis = {
            'support_strength': sr_features.get('support_strength', 0.0),
            'resistance_strength': sr_features.get('resistance_strength', 0.0),
            'breakout_probability': sr_features.get('breakout_probability', 0.0),
            'level_type': 'support' if sr_features.get('support_strength', 0) > sr_features.get('resistance_strength', 0) else 'resistance'
        }

        # Determine strength level
        max_strength = max(sr_analysis['support_strength'], sr_analysis['resistance_strength'])
        sr_analysis['overall_strength'] = (
            'very_strong' if max_strength > 0.8 else
            'strong' if max_strength > 0.6 else
            'moderate' if max_strength > 0.4 else
            'weak'
        )

        return sr_analysis

    def _get_recommendation_level(self, score: float) -> str:
        """Get recommendation level based on overall score"""
        if score > 1.5:
            return 'very_strong'
        elif score > 1.2:
            return 'strong'
        elif score > 0.8:
            return 'moderate'
        elif score > 0.5:
            return 'weak'
        else:
            return 'very_weak'

# Example usage and testing
if __name__ == "__main__":
    async def test_swing_trading_trainer():
        trainer = SwingModelTrainer()

        # Create test training data for swing trading
        training_data = []
        sessions = [TradingSession.LONDON, TradingSession.NY, TradingSession.OVERLAP_LONDON_NY]
        timeframes = [SwingTimeFrame.H1, SwingTimeFrame.H4, SwingTimeFrame.D1]

        for i in range(800):  # More data for swing trading
            session = np.random.choice(sessions)
            timeframe = np.random.choice(timeframes)
            holding_period = np.random.randint(1, 6)  # 1-5 days

            features = SwingTradingFeatures(
                timestamp=datetime.now() + timedelta(hours=i*4),
                session=session,
                timeframe=timeframe,
                price_features={
                    'close': 1.2500 + np.random.normal(0, 0.002),
                    'high': 1.2520 + np.random.normal(0, 0.002),
                    'low': 1.2480 + np.random.normal(0, 0.002),
                    'rsi': np.random.uniform(25, 75),
                    'sma_20': 1.2495 + np.random.normal(0, 0.001),
                    'ema_50': 1.2490 + np.random.normal(0, 0.001)
                },
                pattern_features={
                    'elliott_wave_confidence': np.random.uniform(0.3, 0.9),
                    'fibonacci_confluence': np.random.uniform(0.2, 0.8),
                    'support_resistance_strength': np.random.uniform(0.4, 0.9),
                    'trend_continuation': np.random.uniform(0.1, 0.7),
                    'reversal_pattern': np.random.uniform(0.2, 0.8)
                },
                momentum_features={
                    'macd': np.random.normal(0, 0.0002),
                    'macd_signal': np.random.normal(0, 0.0001),
                    'stochastic_k': np.random.uniform(15, 85),
                    'stochastic_d': np.random.uniform(15, 85),
                    'williams_r': np.random.uniform(-85, -15),
                    'momentum': np.random.normal(0, 0.001)
                },
                volatility_features={
                    'atr': np.random.uniform(0.001, 0.004),
                    'bollinger_width': np.random.uniform(0.002, 0.008),
                    'volatility_ratio': np.random.uniform(0.5, 2.0),
                    'price_range': np.random.uniform(0.001, 0.005)
                },
                volume_features={
                    'volume': np.random.uniform(2000, 8000),
                    'volume_sma': np.random.uniform(3000, 7000),
                    'volume_ratio': np.random.uniform(0.7, 1.5),
                    'money_flow': np.random.uniform(-0.3, 0.3)
                },
                fibonacci_features={
                    'fib_23_6': np.random.uniform(0.1, 0.8),
                    'fib_38_2': np.random.uniform(0.2, 0.9),
                    'fib_50_0': np.random.uniform(0.3, 0.8),
                    'fib_61_8': np.random.uniform(0.2, 0.9),
                    'fib_78_6': np.random.uniform(0.1, 0.7)
                },
                elliott_wave_features={
                    'wave_1': np.random.uniform(0.1, 0.6),
                    'wave_2': np.random.uniform(0.1, 0.5),
                    'wave_3': np.random.uniform(0.2, 0.8),
                    'wave_4': np.random.uniform(0.1, 0.5),
                    'wave_5': np.random.uniform(0.1, 0.6)
                },
                support_resistance_features={
                    'support_strength': np.random.uniform(0.3, 0.9),
                    'resistance_strength': np.random.uniform(0.3, 0.9),
                    'breakout_probability': np.random.uniform(0.2, 0.8),
                    'level_distance': np.random.uniform(0.001, 0.005)
                },
                multi_timeframe_features={
                    'h1_trend': np.random.uniform(-1, 1),
                    'h4_trend': np.random.uniform(-1, 1),
                    'd1_trend': np.random.uniform(-1, 1),
                    'confluence_score': np.random.uniform(0.3, 0.9)
                },
                target=SwingSignalType(np.random.choice([-2, -1, 0, 1, 2])),
                holding_period_days=holding_period
            )
            training_data.append(features)

        # Train models
        print("Training swing trading models...")
        results = await trainer.train_models(training_data)

        print("\nSwing Trading Model Training Results:")
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  F1 Score: {result.f1_score:.4f}")
            print(f"  AUC Score: {result.auc_score:.4f}")
            print(f"  Training Time: {result.training_time:.2f}s")

            # Show pattern accuracy
            if result.pattern_accuracy:
                print(f"  Pattern Accuracy:")
                for pattern, acc in result.pattern_accuracy.items():
                    print(f"    {pattern}: {acc:.3f}")

            # Show holding period performance
            if result.holding_period_performance:
                print(f"  Holding Period Performance:")
                for period, perf in result.holding_period_performance.items():
                    print(f"    {period} days: {perf:.3f}")

        # Test prediction
        test_features = training_data[0]
        signal, confidence, additional_info = await trainer.predict_signal(test_features)
        print(f"\nSwing Prediction Test:")
        print(f"Signal: {signal.name} (confidence: {confidence:.3f})")
        print(f"Dominant Pattern: {additional_info.get('dominant_pattern', 'unknown')}")
        print(f"Holding Period: {additional_info.get('holding_period', 'unknown')} days")
        print(f"Session: {additional_info.get('session', 'unknown')}")

        # Test pattern recommendations
        recommendations = trainer.get_pattern_recommendations(test_features)
        print(f"\nPattern Recommendations:")
        print(f"Overall Score: {recommendations['overall_score']:.3f}")
        print(f"Recommendation Level: {recommendations['recommendation']}")
        print(f"Dominant Pattern: {recommendations['dominant_pattern']}")

        # Get performance summary
        performance = trainer.get_model_performance()
        if performance:
            print(f"\nPerformance Summary:")
            print(f"Best Model: {performance.get('best_model', 'unknown')}")
            print(f"Training Samples: {performance.get('training_samples', 0)}")

    # Run test
    print("Starting Swing Trading Model Trainer Test...")
    asyncio.run(test_swing_trading_trainer())