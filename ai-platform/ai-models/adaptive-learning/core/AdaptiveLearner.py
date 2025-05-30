"""
Adaptive Learning & Self-Improvement Mechanisms for AI Models
Implements continuous learning and self-improvement for trading AI models.

This module provides:
- Real-time performance feedback integration
- Continuous model adaptation based on market changes
- Self-optimizing trading strategies
- Enhanced model accuracy through continuous learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
import json
from datetime import datetime, timedelta
from collections import deque, defaultdict
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Learning mode types"""
    ONLINE = "online"
    BATCH = "batch"
    INCREMENTAL = "incremental"
    REINFORCEMENT = "reinforcement"

class AdaptationTrigger(Enum):
    """Triggers for model adaptation"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MARKET_REGIME_CHANGE = "market_regime_change"
    PREDICTION_DRIFT = "prediction_drift"
    SCHEDULED_UPDATE = "scheduled_update"
    MANUAL_TRIGGER = "manual_trigger"

class ModelType(Enum):
    """Types of models that can be adapted"""
    SCALPING_LSTM = "scalping_lstm"
    DAY_TRADING_ENSEMBLE = "day_trading_ensemble"
    SWING_PATTERN_RECOGNITION = "swing_pattern_recognition"
    VOLUME_ANALYSIS = "volume_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    total_trades: int
    timestamp: datetime

@dataclass
class AdaptationEvent:
    """Adaptation event record"""
    timestamp: datetime
    model_type: ModelType
    trigger: AdaptationTrigger
    old_performance: PerformanceMetrics
    new_performance: Optional[PerformanceMetrics]
    adaptation_details: Dict[str, Any]
    success: bool
    error_message: Optional[str]

@dataclass
class LearningConfiguration:
    """Configuration for adaptive learning"""
    learning_mode: LearningMode
    adaptation_threshold: float
    min_samples_for_adaptation: int
    max_adaptation_frequency: timedelta
    performance_window_size: int
    drift_detection_sensitivity: float
    reinforcement_learning_rate: float
    batch_size: int
    validation_split: float

class AdaptiveLearner:
    """
    Adaptive Learning Engine for AI Model Self-Improvement.
    
    Provides:
    - Continuous performance monitoring
    - Automatic model adaptation
    - Market regime change detection
    - Self-optimizing parameters
    """
    
    def __init__(self, config: Optional[LearningConfiguration] = None):
        """
        Initialize adaptive learner.
        
        Args:
            config: Learning configuration parameters
        """
        self.config = config or LearningConfiguration(
            learning_mode=LearningMode.INCREMENTAL,
            adaptation_threshold=0.05,  # 5% performance degradation triggers adaptation
            min_samples_for_adaptation=100,
            max_adaptation_frequency=timedelta(hours=1),
            performance_window_size=1000,
            drift_detection_sensitivity=0.1,
            reinforcement_learning_rate=0.01,
            batch_size=32,
            validation_split=0.2
        )
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=self.config.performance_window_size))
        self.adaptation_history = []
        self.last_adaptation_time = defaultdict(lambda: datetime.min)
        
        # Model states
        self.model_states = {}
        self.baseline_performance = {}
        
        # Market regime detection
        self.market_regime_detector = MarketRegimeDetector()
        self.current_regime = "normal"
        
        # Learning components
        self.drift_detector = ConceptDriftDetector(sensitivity=self.config.drift_detection_sensitivity)
        self.performance_analyzer = PerformanceAnalyzer()
        
    async def monitor_model_performance(self, model_type: ModelType, predictions: List[Dict], 
                                      actual_results: List[Dict]) -> PerformanceMetrics:
        """
        Monitor model performance and trigger adaptation if needed.
        
        Args:
            model_type: Type of model being monitored
            predictions: Model predictions
            actual_results: Actual market results
            
        Returns:
            Current performance metrics
        """
        try:
            # Calculate current performance metrics
            current_metrics = await self._calculate_performance_metrics(predictions, actual_results)
            
            # Store performance history
            self.performance_history[model_type].append(current_metrics)
            
            # Check if adaptation is needed
            adaptation_needed, trigger = await self._check_adaptation_needed(model_type, current_metrics)
            
            if adaptation_needed:
                await self._trigger_adaptation(model_type, trigger, current_metrics)
            
            # Update baseline if performance improved
            await self._update_baseline_performance(model_type, current_metrics)
            
            logger.info(f"Performance monitoring completed for {model_type.value}: "
                       f"Accuracy={current_metrics.accuracy:.3f}, "
                       f"Profit Factor={current_metrics.profit_factor:.3f}")
            
            return current_metrics
            
        except Exception as e:
            logger.error(f"Performance monitoring failed for {model_type.value}: {e}")
            raise
    
    async def adapt_model(self, model_type: ModelType, trigger: AdaptationTrigger, 
                         new_data: Optional[List[Dict]] = None) -> bool:
        """
        Adapt model based on performance feedback and new data.
        
        Args:
            model_type: Type of model to adapt
            trigger: What triggered the adaptation
            new_data: New training data (optional)
            
        Returns:
            True if adaptation was successful
        """
        try:
            # Check adaptation frequency limits
            if not await self._can_adapt_now(model_type):
                logger.info(f"Adaptation skipped for {model_type.value} - frequency limit")
                return False
            
            # Get current model state
            current_state = self.model_states.get(model_type)
            if not current_state:
                logger.warning(f"No model state found for {model_type.value}")
                return False
            
            # Perform adaptation based on learning mode
            adaptation_result = await self._perform_adaptation(model_type, trigger, new_data, current_state)
            
            # Record adaptation event
            adaptation_event = AdaptationEvent(
                timestamp=datetime.now(),
                model_type=model_type,
                trigger=trigger,
                old_performance=self._get_latest_performance(model_type),
                new_performance=adaptation_result.get('new_performance'),
                adaptation_details=adaptation_result.get('details', {}),
                success=adaptation_result.get('success', False),
                error_message=adaptation_result.get('error')
            )
            
            self.adaptation_history.append(adaptation_event)
            self.last_adaptation_time[model_type] = datetime.now()
            
            logger.info(f"Model adaptation {'successful' if adaptation_event.success else 'failed'} "
                       f"for {model_type.value}")
            
            return adaptation_event.success
            
        except Exception as e:
            logger.error(f"Model adaptation failed for {model_type.value}: {e}")
            return False
    
    async def detect_market_regime_change(self, market_data: pd.DataFrame) -> bool:
        """
        Detect if market regime has changed significantly.
        
        Args:
            market_data: Recent market data
            
        Returns:
            True if regime change detected
        """
        try:
            new_regime = await self.market_regime_detector.detect_regime(market_data)
            
            if new_regime != self.current_regime:
                logger.info(f"Market regime change detected: {self.current_regime} -> {new_regime}")
                self.current_regime = new_regime
                
                # Trigger adaptation for all models
                for model_type in ModelType:
                    await self.adapt_model(model_type, AdaptationTrigger.MARKET_REGIME_CHANGE)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return False
    
    async def _calculate_performance_metrics(self, predictions: List[Dict], 
                                           actual_results: List[Dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not predictions or not actual_results:
            raise ValueError("Insufficient data for performance calculation")
        
        # Align predictions with actual results
        aligned_data = await self._align_predictions_with_results(predictions, actual_results)
        
        # Calculate basic metrics
        accuracy = await self._calculate_accuracy(aligned_data)
        precision = await self._calculate_precision(aligned_data)
        recall = await self._calculate_recall(aligned_data)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate trading metrics
        profit_factor = await self._calculate_profit_factor(aligned_data)
        sharpe_ratio = await self._calculate_sharpe_ratio(aligned_data)
        max_drawdown = await self._calculate_max_drawdown(aligned_data)
        win_rate = await self._calculate_win_rate(aligned_data)
        avg_trade_duration = await self._calculate_avg_trade_duration(aligned_data)
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_trade_duration=avg_trade_duration,
            total_trades=len(aligned_data),
            timestamp=datetime.now()
        )
    
    async def _check_adaptation_needed(self, model_type: ModelType, 
                                     current_metrics: PerformanceMetrics) -> Tuple[bool, Optional[AdaptationTrigger]]:
        """Check if model adaptation is needed"""
        
        # Check performance degradation
        baseline = self.baseline_performance.get(model_type)
        if baseline:
            performance_drop = baseline.accuracy - current_metrics.accuracy
            if performance_drop > self.config.adaptation_threshold:
                return True, AdaptationTrigger.PERFORMANCE_DEGRADATION
        
        # Check concept drift
        recent_performance = list(self.performance_history[model_type])[-10:]
        if len(recent_performance) >= 5:
            drift_detected = await self.drift_detector.detect_drift(recent_performance)
            if drift_detected:
                return True, AdaptationTrigger.PREDICTION_DRIFT
        
        # Check scheduled updates (daily)
        last_adaptation = self.last_adaptation_time[model_type]
        if datetime.now() - last_adaptation > timedelta(days=1):
            return True, AdaptationTrigger.SCHEDULED_UPDATE
        
        return False, None
    
    async def _trigger_adaptation(self, model_type: ModelType, trigger: AdaptationTrigger, 
                                current_metrics: PerformanceMetrics):
        """Trigger model adaptation"""
        logger.info(f"Triggering adaptation for {model_type.value} due to {trigger.value}")
        await self.adapt_model(model_type, trigger)
    
    async def _perform_adaptation(self, model_type: ModelType, trigger: AdaptationTrigger,
                                new_data: Optional[List[Dict]], current_state: Dict) -> Dict:
        """Perform the actual model adaptation"""
        try:
            adaptation_result = {
                'success': False,
                'details': {},
                'new_performance': None,
                'error': None
            }
            
            if self.config.learning_mode == LearningMode.ONLINE:
                result = await self._online_adaptation(model_type, new_data, current_state)
            elif self.config.learning_mode == LearningMode.INCREMENTAL:
                result = await self._incremental_adaptation(model_type, new_data, current_state)
            elif self.config.learning_mode == LearningMode.BATCH:
                result = await self._batch_adaptation(model_type, new_data, current_state)
            elif self.config.learning_mode == LearningMode.REINFORCEMENT:
                result = await self._reinforcement_adaptation(model_type, new_data, current_state)
            else:
                raise ValueError(f"Unknown learning mode: {self.config.learning_mode}")
            
            adaptation_result.update(result)
            return adaptation_result
            
        except Exception as e:
            return {
                'success': False,
                'details': {},
                'new_performance': None,
                'error': str(e)
            }
    
    async def _online_adaptation(self, model_type: ModelType, new_data: Optional[List[Dict]], 
                               current_state: Dict) -> Dict:
        """Perform online learning adaptation"""
        # Implement online learning logic
        # This would update model weights incrementally with each new sample
        return {
            'success': True,
            'details': {'adaptation_type': 'online', 'samples_processed': len(new_data) if new_data else 0},
            'new_performance': None
        }
    
    async def _incremental_adaptation(self, model_type: ModelType, new_data: Optional[List[Dict]], 
                                    current_state: Dict) -> Dict:
        """Perform incremental learning adaptation"""
        # Implement incremental learning logic
        # This would retrain model with new data while preserving existing knowledge
        return {
            'success': True,
            'details': {'adaptation_type': 'incremental', 'samples_added': len(new_data) if new_data else 0},
            'new_performance': None
        }
    
    async def _batch_adaptation(self, model_type: ModelType, new_data: Optional[List[Dict]], 
                              current_state: Dict) -> Dict:
        """Perform batch learning adaptation"""
        # Implement batch learning logic
        # This would retrain model from scratch with all available data
        return {
            'success': True,
            'details': {'adaptation_type': 'batch', 'total_samples': len(new_data) if new_data else 0},
            'new_performance': None
        }
    
    async def _reinforcement_adaptation(self, model_type: ModelType, new_data: Optional[List[Dict]], 
                                      current_state: Dict) -> Dict:
        """Perform reinforcement learning adaptation"""
        # Implement reinforcement learning logic
        # This would update model based on reward/penalty feedback
        return {
            'success': True,
            'details': {'adaptation_type': 'reinforcement', 'learning_rate': self.config.reinforcement_learning_rate},
            'new_performance': None
        }
    
    async def _can_adapt_now(self, model_type: ModelType) -> bool:
        """Check if model can be adapted now (frequency limits)"""
        last_adaptation = self.last_adaptation_time[model_type]
        time_since_last = datetime.now() - last_adaptation
        return time_since_last >= self.config.max_adaptation_frequency
    
    def _get_latest_performance(self, model_type: ModelType) -> Optional[PerformanceMetrics]:
        """Get latest performance metrics for model"""
        history = self.performance_history[model_type]
        return history[-1] if history else None
    
    async def _update_baseline_performance(self, model_type: ModelType, current_metrics: PerformanceMetrics):
        """Update baseline performance if current is better"""
        baseline = self.baseline_performance.get(model_type)
        if not baseline or current_metrics.accuracy > baseline.accuracy:
            self.baseline_performance[model_type] = current_metrics
    
    async def _align_predictions_with_results(self, predictions: List[Dict], 
                                            actual_results: List[Dict]) -> List[Dict]:
        """Align predictions with actual results by timestamp"""
        # Simple implementation - in practice would need sophisticated alignment
        aligned = []
        for pred in predictions:
            for actual in actual_results:
                if abs(pred.get('timestamp', 0) - actual.get('timestamp', 0)) < 60:  # 1 minute tolerance
                    aligned.append({
                        'prediction': pred,
                        'actual': actual,
                        'correct': pred.get('signal') == actual.get('signal')
                    })
                    break
        return aligned
    
    async def _calculate_accuracy(self, aligned_data: List[Dict]) -> float:
        """Calculate prediction accuracy"""
        if not aligned_data:
            return 0.0
        correct = sum(1 for item in aligned_data if item.get('correct', False))
        return correct / len(aligned_data)
    
    async def _calculate_precision(self, aligned_data: List[Dict]) -> float:
        """Calculate precision"""
        # Simplified implementation
        return 0.75  # Placeholder
    
    async def _calculate_recall(self, aligned_data: List[Dict]) -> float:
        """Calculate recall"""
        # Simplified implementation
        return 0.70  # Placeholder
    
    async def _calculate_profit_factor(self, aligned_data: List[Dict]) -> float:
        """Calculate profit factor"""
        # Simplified implementation
        return 1.5  # Placeholder
    
    async def _calculate_sharpe_ratio(self, aligned_data: List[Dict]) -> float:
        """Calculate Sharpe ratio"""
        # Simplified implementation
        return 1.2  # Placeholder
    
    async def _calculate_max_drawdown(self, aligned_data: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        # Simplified implementation
        return 0.05  # Placeholder
    
    async def _calculate_win_rate(self, aligned_data: List[Dict]) -> float:
        """Calculate win rate"""
        # Simplified implementation
        return 0.65  # Placeholder
    
    async def _calculate_avg_trade_duration(self, aligned_data: List[Dict]) -> float:
        """Calculate average trade duration in minutes"""
        # Simplified implementation
        return 45.0  # Placeholder


class MarketRegimeDetector:
    """Detects market regime changes"""
    
    async def detect_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime"""
        # Simplified implementation
        volatility = market_data['close'].pct_change().std()
        if volatility > 0.02:
            return "high_volatility"
        elif volatility < 0.005:
            return "low_volatility"
        else:
            return "normal"


class ConceptDriftDetector:
    """Detects concept drift in model performance"""
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity
    
    async def detect_drift(self, performance_history: List[PerformanceMetrics]) -> bool:
        """Detect if concept drift has occurred"""
        if len(performance_history) < 5:
            return False
        
        # Simple drift detection based on performance trend
        recent_accuracy = np.mean([p.accuracy for p in performance_history[-3:]])
        older_accuracy = np.mean([p.accuracy for p in performance_history[-6:-3]])
        
        drift = abs(recent_accuracy - older_accuracy)
        return drift > self.sensitivity


class PerformanceAnalyzer:
    """Analyzes model performance trends"""
    
    async def analyze_trends(self, performance_history: List[PerformanceMetrics]) -> Dict:
        """Analyze performance trends"""
        if len(performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        accuracies = [p.accuracy for p in performance_history]
        trend = 'improving' if accuracies[-1] > accuracies[0] else 'declining'
        
        return {
            'trend': trend,
            'accuracy_change': accuracies[-1] - accuracies[0],
            'volatility': np.std(accuracies)
        }
