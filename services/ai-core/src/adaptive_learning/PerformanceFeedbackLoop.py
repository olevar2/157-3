"""
Performance Feedback Loop for AI Model Improvement
Implements real-time performance feedback integration for continuous model enhancement.

This module provides:
- Real-time performance tracking and feedback
- Automated model performance evaluation
- Feedback-driven model optimization
- Performance-based model selection and weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
import json
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of performance feedback"""
    PREDICTION_ACCURACY = "prediction_accuracy"
    TRADING_PROFIT = "trading_profit"
    RISK_METRICS = "risk_metrics"
    EXECUTION_QUALITY = "execution_quality"
    USER_FEEDBACK = "user_feedback"
    MARKET_IMPACT = "market_impact"

class FeedbackPriority(Enum):
    """Priority levels for feedback processing"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ModelComponent(Enum):
    """Model components that can receive feedback"""
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_WEIGHTS = "model_weights"
    HYPERPARAMETERS = "hyperparameters"
    ENSEMBLE_WEIGHTS = "ensemble_weights"
    DECISION_THRESHOLDS = "decision_thresholds"
    RISK_PARAMETERS = "risk_parameters"

@dataclass
class FeedbackEvent:
    """Individual feedback event"""
    timestamp: datetime
    model_id: str
    feedback_type: FeedbackType
    priority: FeedbackPriority
    actual_value: float
    predicted_value: float
    error: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_loss: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    avg_trade_duration: float
    confidence_score: float

@dataclass
class FeedbackSummary:
    """Summary of feedback over a time period"""
    start_time: datetime
    end_time: datetime
    model_id: str
    total_feedback_events: int
    avg_error: float
    error_trend: str
    performance_change: float
    improvement_suggestions: List[str]
    critical_issues: List[str]

class PerformanceFeedbackLoop:
    """
    Performance Feedback Loop for Real-time Model Improvement.
    
    Provides:
    - Real-time feedback collection and processing
    - Performance trend analysis
    - Automated model adjustment recommendations
    - Continuous performance monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize performance feedback loop.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Feedback storage
        self.feedback_buffer = defaultdict(lambda: deque(maxlen=10000))
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Processing configuration
        self.feedback_batch_size = self.config.get('feedback_batch_size', 100)
        self.processing_interval = self.config.get('processing_interval', 60)  # seconds
        self.min_feedback_for_adjustment = self.config.get('min_feedback_for_adjustment', 50)
        
        # Performance thresholds
        self.accuracy_threshold = self.config.get('accuracy_threshold', 0.7)
        self.profit_threshold = self.config.get('profit_threshold', 0.0)
        self.max_drawdown_threshold = self.config.get('max_drawdown_threshold', 0.1)
        
        # Feedback processors
        self.feedback_processors = {
            FeedbackType.PREDICTION_ACCURACY: self._process_accuracy_feedback,
            FeedbackType.TRADING_PROFIT: self._process_profit_feedback,
            FeedbackType.RISK_METRICS: self._process_risk_feedback,
            FeedbackType.EXECUTION_QUALITY: self._process_execution_feedback,
            FeedbackType.USER_FEEDBACK: self._process_user_feedback,
            FeedbackType.MARKET_IMPACT: self._process_market_impact_feedback
        }
        
        # Model adjustment callbacks
        self.adjustment_callbacks = {}
        
        # Processing thread
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
    async def start_feedback_loop(self):
        """Start the feedback processing loop"""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Feedback loop already running")
            return
        
        self.stop_processing.clear()
        self.processing_thread = threading.Thread(target=self._feedback_processing_loop)
        self.processing_thread.start()
        logger.info("Performance feedback loop started")
    
    async def stop_feedback_loop(self):
        """Stop the feedback processing loop"""
        if self.processing_thread:
            self.stop_processing.set()
            self.processing_thread.join(timeout=10)
            logger.info("Performance feedback loop stopped")
    
    async def add_feedback(self, feedback: FeedbackEvent):
        """
        Add feedback event to the processing queue.
        
        Args:
            feedback: Feedback event to process
        """
        try:
            # Add to buffer
            self.feedback_buffer[feedback.model_id].append(feedback)
            
            # Process critical feedback immediately
            if feedback.priority == FeedbackPriority.CRITICAL:
                await self._process_immediate_feedback(feedback)
            
            logger.debug(f"Added {feedback.feedback_type.value} feedback for model {feedback.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            raise
    
    async def get_performance_summary(self, model_id: str, 
                                    time_window: timedelta = timedelta(hours=24)) -> FeedbackSummary:
        """
        Get performance summary for a model over a time window.
        
        Args:
            model_id: Model identifier
            time_window: Time window for summary
            
        Returns:
            Performance feedback summary
        """
        try:
            end_time = datetime.now()
            start_time = end_time - time_window
            
            # Get feedback events in time window
            feedback_events = [
                event for event in self.feedback_buffer[model_id]
                if start_time <= event.timestamp <= end_time
            ]
            
            if not feedback_events:
                return FeedbackSummary(
                    start_time=start_time,
                    end_time=end_time,
                    model_id=model_id,
                    total_feedback_events=0,
                    avg_error=0.0,
                    error_trend="no_data",
                    performance_change=0.0,
                    improvement_suggestions=[],
                    critical_issues=[]
                )
            
            # Calculate summary metrics
            total_events = len(feedback_events)
            avg_error = np.mean([event.error for event in feedback_events])
            
            # Analyze error trend
            error_trend = await self._analyze_error_trend(feedback_events)
            
            # Calculate performance change
            performance_change = await self._calculate_performance_change(model_id, time_window)
            
            # Generate improvement suggestions
            improvement_suggestions = await self._generate_improvement_suggestions(feedback_events)
            
            # Identify critical issues
            critical_issues = await self._identify_critical_issues(feedback_events)
            
            return FeedbackSummary(
                start_time=start_time,
                end_time=end_time,
                model_id=model_id,
                total_feedback_events=total_events,
                avg_error=avg_error,
                error_trend=error_trend,
                performance_change=performance_change,
                improvement_suggestions=improvement_suggestions,
                critical_issues=critical_issues
            )
            
        except Exception as e:
            logger.error(f"Failed to get performance summary for {model_id}: {e}")
            raise
    
    async def register_adjustment_callback(self, model_id: str, component: ModelComponent, 
                                         callback: Callable):
        """
        Register callback for model adjustments.
        
        Args:
            model_id: Model identifier
            component: Model component to adjust
            callback: Callback function for adjustments
        """
        if model_id not in self.adjustment_callbacks:
            self.adjustment_callbacks[model_id] = {}
        
        self.adjustment_callbacks[model_id][component] = callback
        logger.info(f"Registered adjustment callback for {model_id} {component.value}")
    
    def _feedback_processing_loop(self):
        """Main feedback processing loop (runs in separate thread)"""
        while not self.stop_processing.is_set():
            try:
                # Process feedback for all models
                for model_id in list(self.feedback_buffer.keys()):
                    asyncio.run(self._process_model_feedback(model_id))
                
                # Wait for next processing cycle
                self.stop_processing.wait(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in feedback processing loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    async def _process_model_feedback(self, model_id: str):
        """Process feedback for a specific model"""
        feedback_buffer = self.feedback_buffer[model_id]
        
        if len(feedback_buffer) < self.min_feedback_for_adjustment:
            return
        
        # Get recent feedback events
        recent_feedback = list(feedback_buffer)[-self.feedback_batch_size:]
        
        # Process by feedback type
        for feedback_type, processor in self.feedback_processors.items():
            type_feedback = [f for f in recent_feedback if f.feedback_type == feedback_type]
            if type_feedback:
                await processor(model_id, type_feedback)
        
        # Update performance snapshot
        await self._update_performance_snapshot(model_id, recent_feedback)
        
        # Check for adjustment triggers
        await self._check_adjustment_triggers(model_id, recent_feedback)
    
    async def _process_immediate_feedback(self, feedback: FeedbackEvent):
        """Process critical feedback immediately"""
        logger.warning(f"Processing critical feedback for {feedback.model_id}: {feedback.feedback_type.value}")
        
        # Immediate processing based on feedback type
        if feedback.feedback_type == FeedbackType.PREDICTION_ACCURACY and feedback.error > 0.5:
            await self._trigger_emergency_adjustment(feedback.model_id, "high_prediction_error")
        elif feedback.feedback_type == FeedbackType.TRADING_PROFIT and feedback.actual_value < -0.1:
            await self._trigger_emergency_adjustment(feedback.model_id, "significant_loss")
        elif feedback.feedback_type == FeedbackType.RISK_METRICS and feedback.actual_value > 0.2:
            await self._trigger_emergency_adjustment(feedback.model_id, "risk_threshold_exceeded")
    
    async def _process_accuracy_feedback(self, model_id: str, feedback_events: List[FeedbackEvent]):
        """Process prediction accuracy feedback"""
        accuracies = [1.0 - event.error for event in feedback_events]
        avg_accuracy = np.mean(accuracies)
        
        if avg_accuracy < self.accuracy_threshold:
            await self._suggest_model_adjustment(model_id, ModelComponent.MODEL_WEIGHTS, 
                                               f"Low accuracy: {avg_accuracy:.3f}")
    
    async def _process_profit_feedback(self, model_id: str, feedback_events: List[FeedbackEvent]):
        """Process trading profit feedback"""
        profits = [event.actual_value for event in feedback_events]
        total_profit = sum(profits)
        
        if total_profit < self.profit_threshold:
            await self._suggest_model_adjustment(model_id, ModelComponent.DECISION_THRESHOLDS,
                                               f"Low profitability: {total_profit:.3f}")
    
    async def _process_risk_feedback(self, model_id: str, feedback_events: List[FeedbackEvent]):
        """Process risk metrics feedback"""
        risk_values = [event.actual_value for event in feedback_events]
        max_risk = max(risk_values)
        
        if max_risk > self.max_drawdown_threshold:
            await self._suggest_model_adjustment(model_id, ModelComponent.RISK_PARAMETERS,
                                               f"High risk exposure: {max_risk:.3f}")
    
    async def _process_execution_feedback(self, model_id: str, feedback_events: List[FeedbackEvent]):
        """Process execution quality feedback"""
        execution_scores = [1.0 - event.error for event in feedback_events]
        avg_execution = np.mean(execution_scores)
        
        if avg_execution < 0.8:
            await self._suggest_model_adjustment(model_id, ModelComponent.DECISION_THRESHOLDS,
                                               f"Poor execution quality: {avg_execution:.3f}")
    
    async def _process_user_feedback(self, model_id: str, feedback_events: List[FeedbackEvent]):
        """Process user feedback"""
        user_scores = [event.actual_value for event in feedback_events]
        avg_user_score = np.mean(user_scores)
        
        if avg_user_score < 3.0:  # Assuming 1-5 scale
            await self._suggest_model_adjustment(model_id, ModelComponent.HYPERPARAMETERS,
                                               f"Low user satisfaction: {avg_user_score:.1f}")
    
    async def _process_market_impact_feedback(self, model_id: str, feedback_events: List[FeedbackEvent]):
        """Process market impact feedback"""
        impact_values = [event.actual_value for event in feedback_events]
        avg_impact = np.mean(impact_values)
        
        if avg_impact > 0.01:  # 1% market impact threshold
            await self._suggest_model_adjustment(model_id, ModelComponent.ENSEMBLE_WEIGHTS,
                                               f"High market impact: {avg_impact:.4f}")
    
    async def _update_performance_snapshot(self, model_id: str, feedback_events: List[FeedbackEvent]):
        """Update performance snapshot for model"""
        # Calculate current performance metrics
        accuracy_events = [e for e in feedback_events if e.feedback_type == FeedbackType.PREDICTION_ACCURACY]
        profit_events = [e for e in feedback_events if e.feedback_type == FeedbackType.TRADING_PROFIT]
        
        accuracy = np.mean([1.0 - e.error for e in accuracy_events]) if accuracy_events else 0.0
        total_profit = sum([e.actual_value for e in profit_events]) if profit_events else 0.0
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            model_id=model_id,
            accuracy=accuracy,
            precision=0.75,  # Placeholder
            recall=0.70,     # Placeholder
            f1_score=0.72,   # Placeholder
            profit_loss=total_profit,
            sharpe_ratio=1.2,  # Placeholder
            max_drawdown=0.05, # Placeholder
            trade_count=len(profit_events),
            avg_trade_duration=45.0,  # Placeholder
            confidence_score=accuracy
        )
        
        self.performance_history[model_id].append(snapshot)
    
    async def _check_adjustment_triggers(self, model_id: str, feedback_events: List[FeedbackEvent]):
        """Check if model adjustments should be triggered"""
        # Check for consistent poor performance
        recent_performance = list(self.performance_history[model_id])[-5:]
        if len(recent_performance) >= 3:
            avg_accuracy = np.mean([p.accuracy for p in recent_performance])
            if avg_accuracy < self.accuracy_threshold:
                await self._trigger_model_adjustment(model_id, ModelComponent.MODEL_WEIGHTS)
    
    async def _suggest_model_adjustment(self, model_id: str, component: ModelComponent, reason: str):
        """Suggest model adjustment"""
        logger.info(f"Suggesting {component.value} adjustment for {model_id}: {reason}")
        
        # Call registered callback if available
        if model_id in self.adjustment_callbacks and component in self.adjustment_callbacks[model_id]:
            callback = self.adjustment_callbacks[model_id][component]
            try:
                await callback(reason)
            except Exception as e:
                logger.error(f"Adjustment callback failed: {e}")
    
    async def _trigger_model_adjustment(self, model_id: str, component: ModelComponent):
        """Trigger immediate model adjustment"""
        logger.warning(f"Triggering immediate {component.value} adjustment for {model_id}")
        await self._suggest_model_adjustment(model_id, component, "Performance threshold exceeded")
    
    async def _trigger_emergency_adjustment(self, model_id: str, reason: str):
        """Trigger emergency model adjustment"""
        logger.critical(f"Emergency adjustment triggered for {model_id}: {reason}")
        
        # Trigger multiple adjustment types for emergency
        for component in [ModelComponent.DECISION_THRESHOLDS, ModelComponent.RISK_PARAMETERS]:
            await self._suggest_model_adjustment(model_id, component, f"Emergency: {reason}")
    
    async def _analyze_error_trend(self, feedback_events: List[FeedbackEvent]) -> str:
        """Analyze error trend over time"""
        if len(feedback_events) < 3:
            return "insufficient_data"
        
        errors = [event.error for event in sorted(feedback_events, key=lambda x: x.timestamp)]
        
        # Simple trend analysis
        recent_errors = errors[-len(errors)//3:]
        older_errors = errors[:len(errors)//3]
        
        recent_avg = np.mean(recent_errors)
        older_avg = np.mean(older_errors)
        
        if recent_avg < older_avg * 0.9:
            return "improving"
        elif recent_avg > older_avg * 1.1:
            return "deteriorating"
        else:
            return "stable"
    
    async def _calculate_performance_change(self, model_id: str, time_window: timedelta) -> float:
        """Calculate performance change over time window"""
        performance_history = list(self.performance_history[model_id])
        
        if len(performance_history) < 2:
            return 0.0
        
        current_performance = performance_history[-1].accuracy
        past_performance = performance_history[0].accuracy
        
        return current_performance - past_performance
    
    async def _generate_improvement_suggestions(self, feedback_events: List[FeedbackEvent]) -> List[str]:
        """Generate improvement suggestions based on feedback"""
        suggestions = []
        
        # Analyze feedback patterns
        accuracy_events = [e for e in feedback_events if e.feedback_type == FeedbackType.PREDICTION_ACCURACY]
        if accuracy_events:
            avg_error = np.mean([e.error for e in accuracy_events])
            if avg_error > 0.3:
                suggestions.append("Consider retraining model with recent data")
                suggestions.append("Review feature engineering pipeline")
        
        profit_events = [e for e in feedback_events if e.feedback_type == FeedbackType.TRADING_PROFIT]
        if profit_events:
            total_profit = sum([e.actual_value for e in profit_events])
            if total_profit < 0:
                suggestions.append("Adjust risk management parameters")
                suggestions.append("Review position sizing strategy")
        
        return suggestions
    
    async def _identify_critical_issues(self, feedback_events: List[FeedbackEvent]) -> List[str]:
        """Identify critical issues from feedback"""
        issues = []
        
        # Check for critical feedback events
        critical_events = [e for e in feedback_events if e.priority == FeedbackPriority.CRITICAL]
        if critical_events:
            issues.append(f"Found {len(critical_events)} critical performance issues")
        
        # Check for high error rates
        accuracy_events = [e for e in feedback_events if e.feedback_type == FeedbackType.PREDICTION_ACCURACY]
        if accuracy_events:
            high_error_count = sum(1 for e in accuracy_events if e.error > 0.5)
            if high_error_count > len(accuracy_events) * 0.3:
                issues.append("High prediction error rate detected")
        
        return issues
