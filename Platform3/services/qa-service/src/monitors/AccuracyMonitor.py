"""
AI Prediction Accuracy Monitoring System
Continuous monitoring and validation of AI model prediction accuracy

Features:
- Real-time prediction accuracy tracking
- Model performance validation (>75% target)
- Prediction confidence analysis
- Model drift detection
- Performance degradation alerts
- Accuracy reporting and analytics
- Model comparison and ranking
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import redis
import json
import statistics
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionType(Enum):
    PRICE_DIRECTION = "price_direction"
    TREND_CONTINUATION = "trend_continuation"
    REVERSAL_SIGNAL = "reversal_signal"
    BREAKOUT_PREDICTION = "breakout_prediction"
    VOLATILITY_FORECAST = "volatility_forecast"
    PATTERN_RECOGNITION = "pattern_recognition"

class AccuracyLevel(Enum):
    EXCELLENT = "excellent"     # >90%
    GOOD = "good"              # 80-90%
    ACCEPTABLE = "acceptable"   # 75-80%
    POOR = "poor"              # 60-75%
    CRITICAL = "critical"       # <60%

@dataclass
class Prediction:
    prediction_id: str
    model_name: str
    prediction_type: PredictionType
    symbol: str
    timeframe: str
    predicted_value: float
    confidence: float
    timestamp: datetime
    features_used: Dict[str, float]
    actual_value: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    is_correct: Optional[bool] = None
    error_magnitude: Optional[float] = None

@dataclass
class ModelPerformance:
    model_name: str
    prediction_type: PredictionType
    total_predictions: int
    correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float
    avg_error_magnitude: float
    accuracy_level: AccuracyLevel
    last_updated: datetime
    performance_trend: str  # 'improving', 'stable', 'declining'

@dataclass
class AccuracyAlert:
    alert_id: str
    model_name: str
    alert_type: str
    severity: str
    message: str
    current_accuracy: float
    target_accuracy: float
    timestamp: datetime
    recommended_action: str

class AccuracyMonitor:
    """
    AI Prediction Accuracy Monitoring System
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, db_config: Optional[Dict] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'trading_platform',
            'user': 'postgres',
            'password': 'password'
        }
        
        # Accuracy targets and thresholds
        self.accuracy_targets = {
            PredictionType.PRICE_DIRECTION: 0.75,      # 75% minimum
            PredictionType.TREND_CONTINUATION: 0.70,   # 70% minimum
            PredictionType.REVERSAL_SIGNAL: 0.65,      # 65% minimum
            PredictionType.BREAKOUT_PREDICTION: 0.70,  # 70% minimum
            PredictionType.VOLATILITY_FORECAST: 0.80,  # 80% minimum
            PredictionType.PATTERN_RECOGNITION: 0.75   # 75% minimum
        }
        
        # Monitoring configuration
        self.monitoring_config = {
            'evaluation_window': timedelta(hours=24),   # 24-hour evaluation window
            'min_predictions_for_eval': 50,            # Minimum predictions for evaluation
            'confidence_threshold': 0.60,              # Minimum confidence for inclusion
            'drift_detection_window': timedelta(days=7), # 7-day drift detection
            'alert_cooldown': timedelta(hours=1),      # 1-hour alert cooldown
            'performance_history_days': 30             # 30 days of performance history
        }
        
        # State tracking
        self.predictions_cache = {}
        self.model_performance = {}
        self.alert_history = []
        self.running = False
        
        # Performance statistics
        self.performance_stats = {
            'total_predictions_monitored': 0,
            'total_evaluations_completed': 0,
            'alerts_generated': 0,
            'models_monitored': 0,
            'average_accuracy_across_models': 0.0,
            'best_performing_model': None,
            'worst_performing_model': None,
            'last_evaluation': None
        }
        
        logger.info("AccuracyMonitor initialized")

    async def start_monitoring(self):
        """Start accuracy monitoring"""
        self.running = True
        logger.info("🚀 Starting AI prediction accuracy monitoring...")
        
        # Initialize database tables
        await self._initialize_database()
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_predictions()),
            asyncio.create_task(self._evaluate_model_performance()),
            asyncio.create_task(self._detect_model_drift()),
            asyncio.create_task(self._generate_performance_reports())
        ]
        
        await asyncio.gather(*tasks)

    async def stop_monitoring(self):
        """Stop accuracy monitoring"""
        self.running = False
        logger.info("⏹️ Stopping accuracy monitoring...")

    async def record_prediction(self, prediction: Prediction) -> bool:
        """Record a new AI prediction for monitoring"""
        try:
            # Store prediction in cache
            self.predictions_cache[prediction.prediction_id] = prediction
            
            # Store in database
            await self._store_prediction_in_db(prediction)
            
            # Cache in Redis for real-time access
            await self._cache_prediction(prediction)
            
            self.performance_stats['total_predictions_monitored'] += 1
            
            logger.debug(f"✅ Recorded prediction {prediction.prediction_id} for model {prediction.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record prediction {prediction.prediction_id}: {e}")
            return False

    async def update_prediction_outcome(self, prediction_id: str, actual_value: float, 
                                      outcome_timestamp: datetime) -> bool:
        """Update prediction with actual outcome for accuracy calculation"""
        try:
            # Get prediction from cache or database
            prediction = await self._get_prediction(prediction_id)
            if not prediction:
                logger.warning(f"Prediction {prediction_id} not found")
                return False
            
            # Update prediction with outcome
            prediction.actual_value = actual_value
            prediction.outcome_timestamp = outcome_timestamp
            
            # Calculate accuracy
            prediction.is_correct = await self._calculate_prediction_accuracy(prediction)
            prediction.error_magnitude = abs(prediction.predicted_value - actual_value)
            
            # Update in cache and database
            self.predictions_cache[prediction_id] = prediction
            await self._update_prediction_in_db(prediction)
            
            # Trigger immediate evaluation if enough predictions
            await self._check_immediate_evaluation(prediction.model_name)
            
            logger.debug(f"✅ Updated prediction outcome for {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update prediction outcome {prediction_id}: {e}")
            return False

    async def get_model_performance(self, model_name: str, 
                                  prediction_type: Optional[PredictionType] = None) -> Optional[ModelPerformance]:
        """Get current performance metrics for a model"""
        try:
            # Calculate performance from recent predictions
            predictions = await self._get_recent_predictions(
                model_name, prediction_type, self.monitoring_config['evaluation_window']
            )
            
            if len(predictions) < self.monitoring_config['min_predictions_for_eval']:
                logger.warning(f"Insufficient predictions for {model_name} evaluation")
                return None
            
            # Filter predictions with outcomes and sufficient confidence
            evaluated_predictions = [
                p for p in predictions 
                if p.actual_value is not None and p.confidence >= self.monitoring_config['confidence_threshold']
            ]
            
            if not evaluated_predictions:
                return None
            
            # Calculate performance metrics
            performance = await self._calculate_model_performance(model_name, evaluated_predictions, prediction_type)
            
            # Store performance
            cache_key = f"{model_name}_{prediction_type.value if prediction_type else 'all'}"
            self.model_performance[cache_key] = performance
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Error getting model performance for {model_name}: {e}")
            return None

    async def _calculate_model_performance(self, model_name: str, predictions: List[Prediction],
                                         prediction_type: Optional[PredictionType]) -> ModelPerformance:
        """Calculate comprehensive model performance metrics"""
        try:
            # Basic metrics
            total_predictions = len(predictions)
            correct_predictions = sum(1 for p in predictions if p.is_correct)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Prepare data for sklearn metrics
            y_true = [1 if p.is_correct else 0 for p in predictions]
            y_pred = [1 if p.confidence > 0.5 else 0 for p in predictions]  # Binary classification
            
            # Calculate advanced metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Additional metrics
            avg_confidence = statistics.mean([p.confidence for p in predictions])
            avg_error_magnitude = statistics.mean([p.error_magnitude for p in predictions if p.error_magnitude is not None])
            
            # Determine accuracy level
            accuracy_level = self._determine_accuracy_level(accuracy)
            
            # Calculate performance trend
            performance_trend = await self._calculate_performance_trend(model_name, prediction_type)
            
            return ModelPerformance(
                model_name=model_name,
                prediction_type=prediction_type or PredictionType.PRICE_DIRECTION,
                total_predictions=total_predictions,
                correct_predictions=correct_predictions,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                avg_confidence=avg_confidence,
                avg_error_magnitude=avg_error_magnitude,
                accuracy_level=accuracy_level,
                last_updated=datetime.now(),
                performance_trend=performance_trend
            )
            
        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            raise

    def _determine_accuracy_level(self, accuracy: float) -> AccuracyLevel:
        """Determine accuracy level based on accuracy score"""
        if accuracy >= 0.90:
            return AccuracyLevel.EXCELLENT
        elif accuracy >= 0.80:
            return AccuracyLevel.GOOD
        elif accuracy >= 0.75:
            return AccuracyLevel.ACCEPTABLE
        elif accuracy >= 0.60:
            return AccuracyLevel.POOR
        else:
            return AccuracyLevel.CRITICAL

    async def _monitor_predictions(self):
        """Continuous prediction monitoring"""
        while self.running:
            try:
                # Check for predictions needing outcome updates
                pending_predictions = await self._get_pending_predictions()
                
                for prediction in pending_predictions:
                    # Check if outcome should be available
                    if self._should_have_outcome(prediction):
                        await self._fetch_actual_outcome(prediction)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in prediction monitoring: {e}")
                await asyncio.sleep(60)

    async def _evaluate_model_performance(self):
        """Periodic model performance evaluation"""
        while self.running:
            try:
                # Get all active models
                active_models = await self._get_active_models()
                
                for model_name in active_models:
                    for prediction_type in PredictionType:
                        performance = await self.get_model_performance(model_name, prediction_type)
                        
                        if performance:
                            await self._check_performance_alerts(performance)
                            await self._store_performance_history(performance)
                
                self.performance_stats['total_evaluations_completed'] += 1
                self.performance_stats['last_evaluation'] = datetime.now()
                
                await asyncio.sleep(3600)  # Evaluate every hour
                
            except Exception as e:
                logger.error(f"Error in performance evaluation: {e}")
                await asyncio.sleep(3600)

    async def _check_performance_alerts(self, performance: ModelPerformance):
        """Check if performance alerts should be generated"""
        try:
            target_accuracy = self.accuracy_targets.get(performance.prediction_type, 0.75)
            
            # Check accuracy threshold
            if performance.accuracy < target_accuracy:
                await self._generate_accuracy_alert(
                    performance,
                    "ACCURACY_BELOW_TARGET",
                    "HIGH",
                    f"Model accuracy {performance.accuracy:.2%} below target {target_accuracy:.2%}",
                    "Review model parameters and retrain if necessary"
                )
            
            # Check for critical accuracy
            if performance.accuracy_level == AccuracyLevel.CRITICAL:
                await self._generate_accuracy_alert(
                    performance,
                    "CRITICAL_ACCURACY",
                    "CRITICAL",
                    f"Model accuracy critically low: {performance.accuracy:.2%}",
                    "Immediate model review and retraining required"
                )
            
            # Check performance trend
            if performance.performance_trend == "declining":
                await self._generate_accuracy_alert(
                    performance,
                    "DECLINING_PERFORMANCE",
                    "MEDIUM",
                    f"Model performance declining: {performance.accuracy:.2%}",
                    "Monitor closely and consider model refresh"
                )
                
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")

    async def _generate_accuracy_alert(self, performance: ModelPerformance, alert_type: str,
                                     severity: str, message: str, recommended_action: str):
        """Generate accuracy alert"""
        try:
            alert = AccuracyAlert(
                alert_id=f"ACC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_name=performance.model_name,
                alert_type=alert_type,
                severity=severity,
                message=message,
                current_accuracy=performance.accuracy,
                target_accuracy=self.accuracy_targets.get(performance.prediction_type, 0.75),
                timestamp=datetime.now(),
                recommended_action=recommended_action
            )
            
            # Check alert cooldown
            if await self._should_send_alert(alert):
                await self._send_alert(alert)
                self.alert_history.append(alert)
                self.performance_stats['alerts_generated'] += 1
                
                logger.warning(f"🚨 Accuracy Alert: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error generating accuracy alert: {e}")

    async def get_accuracy_report(self, timeframe: timedelta = timedelta(days=7)) -> Dict:
        """Generate comprehensive accuracy report"""
        try:
            end_time = datetime.now()
            start_time = end_time - timeframe
            
            # Get all model performances
            model_performances = []
            active_models = await self._get_active_models()
            
            for model_name in active_models:
                for prediction_type in PredictionType:
                    performance = await self.get_model_performance(model_name, prediction_type)
                    if performance:
                        model_performances.append(performance)
            
            # Calculate overall statistics
            overall_accuracy = statistics.mean([p.accuracy for p in model_performances]) if model_performances else 0
            best_model = max(model_performances, key=lambda p: p.accuracy) if model_performances else None
            worst_model = min(model_performances, key=lambda p: p.accuracy) if model_performances else None
            
            # Update performance stats
            self.performance_stats['average_accuracy_across_models'] = overall_accuracy
            self.performance_stats['best_performing_model'] = best_model.model_name if best_model else None
            self.performance_stats['worst_performing_model'] = worst_model.model_name if worst_model else None
            self.performance_stats['models_monitored'] = len(set(p.model_name for p in model_performances))
            
            return {
                'timeframe': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': timeframe.total_seconds() / 3600
                },
                'overall_statistics': {
                    'average_accuracy': overall_accuracy,
                    'models_monitored': len(set(p.model_name for p in model_performances)),
                    'total_predictions': sum(p.total_predictions for p in model_performances),
                    'best_performing_model': best_model.model_name if best_model else None,
                    'worst_performing_model': worst_model.model_name if worst_model else None
                },
                'model_performances': [
                    {
                        'model_name': p.model_name,
                        'prediction_type': p.prediction_type.value,
                        'accuracy': p.accuracy,
                        'accuracy_level': p.accuracy_level.value,
                        'total_predictions': p.total_predictions,
                        'performance_trend': p.performance_trend
                    }
                    for p in model_performances
                ],
                'accuracy_targets': {k.value: v for k, v in self.accuracy_targets.items()},
                'recent_alerts': [
                    {
                        'alert_type': alert.alert_type,
                        'model_name': alert.model_name,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.alert_history[-10:]  # Last 10 alerts
                ],
                'performance_stats': self.performance_stats
            }
            
        except Exception as e:
            logger.error(f"Error generating accuracy report: {e}")
            return {'error': str(e)}

    async def get_performance_stats(self) -> Dict:
        """Get monitoring performance statistics"""
        return {
            **self.performance_stats,
            'monitoring_status': 'running' if self.running else 'stopped',
            'accuracy_targets': {k.value: v for k, v in self.accuracy_targets.items()},
            'monitoring_config': {
                'evaluation_window_hours': self.monitoring_config['evaluation_window'].total_seconds() / 3600,
                'min_predictions_for_eval': self.monitoring_config['min_predictions_for_eval'],
                'confidence_threshold': self.monitoring_config['confidence_threshold']
            }
        }

    # Additional helper methods would be implemented here for:
    # - _initialize_database()
    # - _store_prediction_in_db()
    # - _cache_prediction()
    # - _get_prediction()
    # - _calculate_prediction_accuracy()
    # - _get_recent_predictions()
    # - _get_pending_predictions()
    # - _should_have_outcome()
    # - _fetch_actual_outcome()
    # - _get_active_models()
    # - _calculate_performance_trend()
    # - _should_send_alert()
    # - _send_alert()
    # - _store_performance_history()
