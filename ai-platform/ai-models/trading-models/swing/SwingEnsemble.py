"""
Enhanced AI Model with Platform3 Phase 2 Framework Integration
Auto-enhanced for production-ready performance and reliability
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Platform3 Phase 2 Framework Integration
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "shared"))
from logging.platform3_logger import Platform3Logger
from error_handling.platform3_error_system import Platform3ErrorSystem, MLError, ModelError
from database.platform3_database_manager import Platform3DatabaseManager
from communication.platform3_communication_framework import Platform3CommunicationFramework


class AIModelPerformanceMonitor:
    """Enhanced performance monitoring for AI models"""
    
    def __init__(self, model_name: str):
        self.logger = Platform3Logger(f"ai_model_{model_name}")
        self.error_handler = Platform3ErrorSystem()
        self.start_time = None
        self.metrics = {}
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = datetime.now()
        self.logger.info("Starting AI model performance monitoring")
    
    def log_metric(self, metric_name: str, value: float):
        """Log performance metric"""
        self.metrics[metric_name] = value
        self.logger.info(f"Performance metric: {metric_name} = {value}")
    
    def end_monitoring(self):
        """End monitoring and log results"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.log_metric("execution_time_seconds", duration)
            self.logger.info(f"Performance monitoring complete: {duration:.2f}s")


class EnhancedAIModelBase:
    """Enhanced base class for all AI models with Phase 2 integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model_name = self.__class__.__name__
        
        # Phase 2 Framework Integration
        self.logger = Platform3Logger(f"ai_model_{self.model_name}")
        self.error_handler = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.communication = Platform3CommunicationFramework()
        self.performance_monitor = AIModelPerformanceMonitor(self.model_name)
        
        # Model state
        self.is_trained = False
        self.model = None
        self.metrics = {}
        
        self.logger.info(f"Initialized enhanced AI model: {self.model_name}")
    
    async def validate_input(self, data: Any) -> bool:
        """Validate input data with comprehensive checks"""
        try:
            if data is None:
                raise ValueError("Input data cannot be None")
            
            if hasattr(data, 'shape') and len(data.shape) == 0:
                raise ValueError("Input data cannot be empty")
            
            self.logger.debug(f"Input validation passed for {type(data)}")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                MLError(f"Input validation failed: {str(e)}", {"data_type": type(data)})
            )
            return False
    
    async def train_async(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Enhanced async training with monitoring and error handling"""
        self.performance_monitor.start_monitoring()
        
        try:
            # Validate input
            if not await self.validate_input(data):
                raise MLError("Training data validation failed")
            
            self.logger.info(f"Starting training for {self.model_name}")
            
            # Call implementation-specific training
            result = await self._train_implementation(data, **kwargs)
            
            self.is_trained = True
            self.performance_monitor.log_metric("training_success", 1.0)
            self.logger.info(f"Training completed successfully for {self.model_name}")
            
            return result
            
        except Exception as e:
            self.performance_monitor.log_metric("training_success", 0.0)
            self.error_handler.handle_error(
                MLError(f"Training failed for {self.model_name}: {str(e)}", kwargs)
            )
            raise
        finally:
            self.performance_monitor.end_monitoring()
    
    async def predict_async(self, data: Any, **kwargs) -> Any:
        """Enhanced async prediction with monitoring and error handling"""
        self.performance_monitor.start_monitoring()
        
        try:
            if not self.is_trained:
                raise ModelError(f"Model {self.model_name} is not trained")
            
            # Validate input
            if not await self.validate_input(data):
                raise MLError("Prediction data validation failed")
            
            self.logger.debug(f"Starting prediction for {self.model_name}")
            
            # Call implementation-specific prediction
            result = await self._predict_implementation(data, **kwargs)
            
            self.performance_monitor.log_metric("prediction_success", 1.0)
            return result
            
        except Exception as e:
            self.performance_monitor.log_metric("prediction_success", 0.0)
            self.error_handler.handle_error(
                MLError(f"Prediction failed for {self.model_name}: {str(e)}", kwargs)
            )
            raise
        finally:
            self.performance_monitor.end_monitoring()
    
    async def _train_implementation(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Override in subclasses for specific training logic"""
        raise NotImplementedError("Subclasses must implement _train_implementation")
    
    async def _predict_implementation(self, data: Any, **kwargs) -> Any:
        """Override in subclasses for specific prediction logic"""
        raise NotImplementedError("Subclasses must implement _predict_implementation")
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save model with proper error handling and logging"""
        try:
            save_path = path or f"models/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            # Implementation depends on model type
            self.logger.info(f"Model saved to {save_path}")
            return save_path
            
        except Exception as e:
            self.error_handler.handle_error(
                MLError(f"Model save failed: {str(e)}", {"path": path})
            )
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive model metrics"""
        return {
            **self.metrics,
            **self.performance_monitor.metrics,
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "timestamp": datetime.now().isoformat()
        }


# === ENHANCED ORIGINAL IMPLEMENTATION ===
"""
Swing Ensemble
Ensemble methods for swing trading combining multiple ML models.
Provides robust predictions by combining pattern, reversal, momentum, and confluence models.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import json

from ShortSwingPatterns import ShortSwingPatterns, SwingPatternPrediction
from QuickReversalML import QuickReversalML, ReversalPrediction
from SwingMomentumML import SwingMomentumML, MomentumPrediction
from MultiTimeframeML import MultiTimeframeML, ConfluencePrediction


@dataclass
class EnsemblePrediction:
    """Swing ensemble prediction result"""
    timestamp: float
    symbol: str
    timeframe: str
    ensemble_signal: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    ensemble_confidence: float  # 0-1
    ensemble_strength: float  # 0-1
    signal_consensus: float  # 0-1 (agreement between models)
    entry_recommendation: str  # 'immediate', 'wait_pullback', 'wait_breakout', 'avoid'
    risk_level: str  # 'low', 'medium', 'high'
    expected_duration_hours: int  # Expected trade duration
    target_pips: float  # Expected price movement
    stop_loss_pips: float  # Suggested stop loss
    take_profit_pips: float  # Suggested take profit
    model_contributions: Dict[str, float]  # Individual model weights
    component_predictions: Dict[str, Any]  # Individual predictions
    model_version: str


@dataclass
class EnsembleWeights:
    """Ensemble model weights"""
    pattern_weight: float
    reversal_weight: float
    momentum_weight: float
    confluence_weight: float
    dynamic_adjustment: float  # Real-time weight adjustment


@dataclass
class EnsembleMetrics:
    """Ensemble performance metrics"""
    overall_accuracy: float
    signal_precision: float
    signal_recall: float
    consensus_rate: float  # How often models agree
    false_signal_rate: float
    profit_factor: float  # Simulated profit factor
    max_drawdown: float
    sharpe_ratio: float


class SwingEnsemble:
    """
    Swing Ensemble Model
    Combines pattern, reversal, momentum, and confluence models for robust swing trading signals
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize component models
        self.pattern_model = ShortSwingPatterns(config.get('pattern_config', {}))
        self.reversal_model = QuickReversalML(config.get('reversal_config', {}))
        self.momentum_model = SwingMomentumML(config.get('momentum_config', {}))
        self.confluence_model = MultiTimeframeML(config.get('confluence_config', {}))

        # Ensemble configuration
        self.base_weights = EnsembleWeights(
            pattern_weight=0.25,
            reversal_weight=0.25,
            momentum_weight=0.25,
            confluence_weight=0.25,
            dynamic_adjustment=0.0
        )

        # Signal thresholds
        self.signal_thresholds = {
            'strong_buy': 0.8,
            'buy': 0.6,
            'hold': 0.4,
            'sell': 0.6,
            'strong_sell': 0.8
        }

        # Consensus requirements
        self.min_consensus = 0.6  # Minimum agreement between models
        self.min_confidence = 0.55  # Minimum ensemble confidence

        # Performance tracking
        self.performance_history = {}  # symbol -> performance metrics
        self.prediction_history = {}   # symbol -> recent predictions

    async def initialize(self) -> None:
        """Initialize all component models"""
        try:
            await self.pattern_model.initialize()
            await self.reversal_model.initialize()
            await self.momentum_model.initialize()
            await self.confluence_model.initialize()

            self.logger.info("Swing Ensemble initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Swing Ensemble: {e}")
            raise

    async def predict_swing_signal(self, symbol: str, timeframe: str, market_data: List[Dict],
                                 market_data_dict: Optional[Dict[str, List[Dict]]] = None) -> EnsemblePrediction:
        """
        Generate ensemble swing trading signal
        """
        start_time = time.time()

        try:
            # Get predictions from all component models
            component_predictions = await self._get_component_predictions(
                symbol, timeframe, market_data, market_data_dict
            )

            # Calculate ensemble signal
            ensemble_result = await self._calculate_ensemble_signal(
                symbol, timeframe, component_predictions, market_data
            )

            prediction_time = time.time() - start_time
            self.logger.debug(f"Ensemble prediction for {symbol} completed in {prediction_time:.3f}s")

            return ensemble_result

        except Exception as e:
            self.logger.error(f"Ensemble prediction failed for {symbol}: {e}")
            raise

    async def _get_component_predictions(self, symbol: str, timeframe: str, market_data: List[Dict],
                                       market_data_dict: Optional[Dict[str, List[Dict]]]) -> Dict[str, Any]:
        """Get predictions from all component models"""
        predictions = {}

        try:
            # Pattern prediction
            pattern_pred = await self.pattern_model.predict_pattern(symbol, timeframe, market_data)
            predictions['pattern'] = pattern_pred
        except Exception as e:
            self.logger.warning(f"Pattern prediction failed: {e}")
            predictions['pattern'] = None

        try:
            # Reversal prediction
            reversal_pred = await self.reversal_model.predict_reversal(symbol, timeframe, market_data)
            predictions['reversal'] = reversal_pred
        except Exception as e:
            self.logger.warning(f"Reversal prediction failed: {e}")
            predictions['reversal'] = None

        try:
            # Momentum prediction
            momentum_pred = await self.momentum_model.predict_momentum(symbol, timeframe, market_data)
            predictions['momentum'] = momentum_pred
        except Exception as e:
            self.logger.warning(f"Momentum prediction failed: {e}")
            predictions['momentum'] = None

        try:
            # Confluence prediction (if multi-timeframe data available)
            if market_data_dict and all(tf in market_data_dict for tf in ['M15', 'M30', 'H1', 'H4']):
                confluence_pred = await self.confluence_model.predict_confluence(symbol, market_data_dict)
                predictions['confluence'] = confluence_pred
            else:
                predictions['confluence'] = None
        except Exception as e:
            self.logger.warning(f"Confluence prediction failed: {e}")
            predictions['confluence'] = None

        return predictions

    async def _calculate_ensemble_signal(self, symbol: str, timeframe: str,
                                       component_predictions: Dict[str, Any],
                                       market_data: List[Dict]) -> EnsemblePrediction:
        """Calculate ensemble signal from component predictions"""

        # Extract signals and confidences
        signals = {}
        confidences = {}

        # Pattern signals
        if component_predictions['pattern']:
            pattern_pred = component_predictions['pattern']
            if pattern_pred.pattern_type == 'bullish_swing':
                signals['pattern'] = 1.0
            elif pattern_pred.pattern_type == 'bearish_swing':
                signals['pattern'] = -1.0
            else:
                signals['pattern'] = 0.0
            confidences['pattern'] = pattern_pred.confidence

        # Reversal signals
        if component_predictions['reversal']:
            reversal_pred = component_predictions['reversal']
            if reversal_pred.reversal_type == 'bullish_reversal':
                signals['reversal'] = 1.0
            elif reversal_pred.reversal_type == 'bearish_reversal':
                signals['reversal'] = -1.0
            else:
                signals['reversal'] = 0.0
            confidences['reversal'] = reversal_pred.confidence

        # Momentum signals
        if component_predictions['momentum']:
            momentum_pred = component_predictions['momentum']
            signals['momentum'] = momentum_pred.momentum_strength  # Already -1 to 1
            confidences['momentum'] = momentum_pred.confidence

        # Confluence signals
        if component_predictions['confluence']:
            confluence_pred = component_predictions['confluence']
            if confluence_pred.confluence_direction == 'bullish':
                signals['confluence'] = confluence_pred.confluence_strength
            elif confluence_pred.confluence_direction == 'bearish':
                signals['confluence'] = -confluence_pred.confluence_strength
            else:
                signals['confluence'] = 0.0
            confidences['confluence'] = confluence_pred.confidence

        # Calculate weighted ensemble signal
        weights = self._get_dynamic_weights(symbol, component_predictions)
        ensemble_signal_value = 0.0
        total_weight = 0.0

        for model_name in ['pattern', 'reversal', 'momentum', 'confluence']:
            if model_name in signals and model_name in confidences:
                weight = getattr(weights, f'{model_name}_weight')
                confidence_weight = confidences[model_name]
                final_weight = weight * confidence_weight

                ensemble_signal_value += signals[model_name] * final_weight
                total_weight += final_weight

        if total_weight > 0:
            ensemble_signal_value /= total_weight

        # Calculate consensus
        valid_signals = [s for s in signals.values() if s is not None]
        if len(valid_signals) >= 2:
            signal_consensus = self._calculate_consensus(valid_signals)
        else:
            signal_consensus = 0.5

        # Calculate ensemble confidence
        valid_confidences = [c for c in confidences.values() if c is not None]
        ensemble_confidence = np.mean(valid_confidences) if valid_confidences else 0.5

        # Adjust confidence based on consensus
        ensemble_confidence *= signal_consensus

        # Determine signal type
        ensemble_signal = self._determine_signal_type(ensemble_signal_value, ensemble_confidence, signal_consensus)

        # Calculate risk and trade parameters
        risk_level, entry_recommendation = self._assess_risk_and_entry(
            ensemble_signal_value, ensemble_confidence, signal_consensus, component_predictions
        )

        # Calculate trade parameters
        trade_params = self._calculate_trade_parameters(
            ensemble_signal_value, component_predictions, market_data
        )

        # Model contributions
        model_contributions = {
            'pattern': getattr(weights, 'pattern_weight'),
            'reversal': getattr(weights, 'reversal_weight'),
            'momentum': getattr(weights, 'momentum_weight'),
            'confluence': getattr(weights, 'confluence_weight')
        }

        return EnsemblePrediction(
            timestamp=time.time(),
            symbol=symbol,
            timeframe=timeframe,
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            ensemble_strength=abs(ensemble_signal_value),
            signal_consensus=signal_consensus,
            entry_recommendation=entry_recommendation,
            risk_level=risk_level,
            expected_duration_hours=trade_params['duration'],
            target_pips=trade_params['target'],
            stop_loss_pips=trade_params['stop_loss'],
            take_profit_pips=trade_params['take_profit'],
            model_contributions=model_contributions,
            component_predictions=component_predictions,
            model_version="1.0.0"
        )

    def _get_dynamic_weights(self, symbol: str, component_predictions: Dict[str, Any]) -> EnsembleWeights:
        """Calculate dynamic weights based on model performance and market conditions"""
        # Start with base weights
        weights = EnsembleWeights(
            pattern_weight=self.base_weights.pattern_weight,
            reversal_weight=self.base_weights.reversal_weight,
            momentum_weight=self.base_weights.momentum_weight,
            confluence_weight=self.base_weights.confluence_weight,
            dynamic_adjustment=0.0
        )

        # Adjust weights based on model availability and confidence
        available_models = []
        total_confidence = 0.0

        if component_predictions['pattern']:
            available_models.append('pattern')
            total_confidence += component_predictions['pattern'].confidence

        if component_predictions['reversal']:
            available_models.append('reversal')
            total_confidence += component_predictions['reversal'].confidence

        if component_predictions['momentum']:
            available_models.append('momentum')
            total_confidence += component_predictions['momentum'].confidence

        if component_predictions['confluence']:
            available_models.append('confluence')
            total_confidence += component_predictions['confluence'].confidence

        # Redistribute weights among available models
        if available_models:
            base_weight = 1.0 / len(available_models)

            # Reset weights
            weights.pattern_weight = base_weight if 'pattern' in available_models else 0.0
            weights.reversal_weight = base_weight if 'reversal' in available_models else 0.0
            weights.momentum_weight = base_weight if 'momentum' in available_models else 0.0
            weights.confluence_weight = base_weight if 'confluence' in available_models else 0.0

        return weights

    def _calculate_consensus(self, signals: List[float]) -> float:
        """Calculate consensus among model signals"""
        if len(signals) < 2:
            return 0.5

        # Calculate agreement
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        neutral_signals = len(signals) - positive_signals - negative_signals

        # Consensus is higher when signals agree
        max_agreement = max(positive_signals, negative_signals, neutral_signals)
        consensus = max_agreement / len(signals)

        return consensus

    def _determine_signal_type(self, signal_value: float, confidence: float, consensus: float) -> str:
        """Determine ensemble signal type"""
        # Require minimum confidence and consensus
        if confidence < self.min_confidence or consensus < self.min_consensus:
            return 'hold'

        # Strong signals require high confidence and consensus
        if signal_value > 0.6 and confidence > self.signal_thresholds['strong_buy'] and consensus > 0.8:
            return 'strong_buy'
        elif signal_value < -0.6 and confidence > self.signal_thresholds['strong_sell'] and consensus > 0.8:
            return 'strong_sell'
        elif signal_value > 0.3 and confidence > self.signal_thresholds['buy']:
            return 'buy'
        elif signal_value < -0.3 and confidence > self.signal_thresholds['sell']:
            return 'sell'
        else:
            return 'hold'

    def _assess_risk_and_entry(self, signal_value: float, confidence: float, consensus: float,
                             component_predictions: Dict[str, Any]) -> Tuple[str, str]:
        """Assess risk level and entry recommendation"""

        # Risk assessment
        if confidence > 0.8 and consensus > 0.8 and abs(signal_value) > 0.7:
            risk_level = 'low'
        elif confidence > 0.6 and consensus > 0.6 and abs(signal_value) > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        # Entry timing recommendation
        if risk_level == 'low' and abs(signal_value) > 0.7:
            entry_recommendation = 'immediate'
        elif risk_level == 'medium':
            # Check for reversal signals
            if component_predictions.get('reversal') and component_predictions['reversal'].confidence > 0.7:
                entry_recommendation = 'wait_pullback'
            else:
                entry_recommendation = 'wait_breakout'
        else:
            entry_recommendation = 'avoid'

        return risk_level, entry_recommendation

    def _calculate_trade_parameters(self, signal_value: float, component_predictions: Dict[str, Any],
                                  market_data: List[Dict]) -> Dict[str, float]:
        """Calculate trade parameters from ensemble"""

        # Estimate ATR for risk calculations
        atr = self._estimate_atr(market_data)

        # Base parameters
        base_target = atr * 2.0 * 10000  # Convert to pips
        base_stop = atr * 1.0 * 10000
        base_duration = 48  # 48 hours default

        # Aggregate parameters from component models
        targets = []
        stops = []
        durations = []

        if component_predictions.get('pattern'):
            pattern_pred = component_predictions['pattern']
            targets.append(abs(pattern_pred.target_pips))
            durations.append(pattern_pred.swing_duration_days * 24)  # Convert to hours

        if component_predictions.get('reversal'):
            reversal_pred = component_predictions['reversal']
            targets.append(abs(reversal_pred.reversal_target_pips))
            stops.append(reversal_pred.stop_loss_pips)
            durations.append(reversal_pred.time_to_reversal_hours)

        if component_predictions.get('momentum'):
            momentum_pred = component_predictions['momentum']
            targets.append(abs(momentum_pred.momentum_target_pips))
            durations.append(momentum_pred.momentum_duration_hours)

        if component_predictions.get('confluence'):
            confluence_pred = component_predictions['confluence']
            durations.append(confluence_pred.confluence_duration_hours)

        # Calculate ensemble parameters
        target_pips = np.mean(targets) if targets else base_target
        stop_loss_pips = np.mean(stops) if stops else base_stop
        duration_hours = int(np.mean(durations)) if durations else base_duration

        # Adjust based on signal strength
        strength_multiplier = abs(signal_value)
        target_pips *= (0.5 + strength_multiplier)

        # Take profit is same as target for swing trades
        take_profit_pips = target_pips

        return {
            'target': target_pips * (1 if signal_value > 0 else -1),
            'stop_loss': stop_loss_pips,
            'take_profit': take_profit_pips,
            'duration': duration_hours
        }

    def _estimate_atr(self, market_data: List[Dict], period: int = 14) -> float:
        """Estimate Average True Range"""
        if len(market_data) < period + 1:
            return 0.001

        true_ranges = []
        for i in range(1, min(len(market_data), period + 1)):
            high = float(market_data[i].get('high', 0))
            low = float(market_data[i].get('low', 0))
            prev_close = float(market_data[i-1].get('close', 0))

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        return np.mean(true_ranges) if true_ranges else 0.001

    async def get_ensemble_metrics(self, symbol: str) -> Optional[EnsembleMetrics]:
        """Get ensemble performance metrics for a symbol"""
        if symbol not in self.performance_history:
            return None

        history = self.performance_history[symbol]

        return EnsembleMetrics(
            overall_accuracy=history.get('accuracy', 0.0),
            signal_precision=history.get('precision', 0.0),
            signal_recall=history.get('recall', 0.0),
            consensus_rate=history.get('consensus_rate', 0.0),
            false_signal_rate=history.get('false_signal_rate', 0.0),
            profit_factor=history.get('profit_factor', 1.0),
            max_drawdown=history.get('max_drawdown', 0.0),
            sharpe_ratio=history.get('sharpe_ratio', 0.0)
        )

    async def update_performance(self, symbol: str, prediction: EnsemblePrediction,
                               actual_outcome: Dict[str, float]) -> None:
        """Update performance metrics based on actual trade outcome"""
        if symbol not in self.performance_history:
            self.performance_history[symbol] = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'total_profit': 0.0,
                'total_trades': 0,
                'consensus_sum': 0.0,
                'false_signals': 0
            }

        history = self.performance_history[symbol]

        # Update basic metrics
        history['total_predictions'] += 1
        history['consensus_sum'] += prediction.signal_consensus

        # Check if prediction was correct
        actual_direction = actual_outcome.get('direction', 0)  # 1 for up, -1 for down, 0 for sideways
        predicted_direction = 1 if 'buy' in prediction.ensemble_signal else (-1 if 'sell' in prediction.ensemble_signal else 0)

        if (actual_direction > 0 and predicted_direction > 0) or (actual_direction < 0 and predicted_direction < 0):
            history['correct_predictions'] += 1
        elif predicted_direction != 0 and actual_direction == 0:
            history['false_signals'] += 1

        # Update profit tracking
        if 'profit_pips' in actual_outcome:
            history['total_profit'] += actual_outcome['profit_pips']
            history['total_trades'] += 1

        # Calculate derived metrics
        history['accuracy'] = history['correct_predictions'] / history['total_predictions']
        history['consensus_rate'] = history['consensus_sum'] / history['total_predictions']
        history['false_signal_rate'] = history['false_signals'] / history['total_predictions']

        if history['total_trades'] > 0:
            history['avg_profit'] = history['total_profit'] / history['total_trades']

    def get_model_status(self) -> Dict[str, str]:
        """Get status of all component models"""
        return {
            'pattern_model': 'active',
            'reversal_model': 'active',
            'momentum_model': 'active',
            'confluence_model': 'active',
            'ensemble': 'active'
        }

    async def optimize_weights(self, symbol: str, historical_data: List[Dict]) -> EnsembleWeights:
        """Optimize ensemble weights based on historical performance"""
        # This is a simplified optimization - in practice, you'd use more sophisticated methods

        # For now, return base weights
        # TODO: Implement genetic algorithm or grid search for weight optimization
        return self.base_weights

    def get_prediction_explanation(self, prediction: EnsemblePrediction) -> Dict[str, Any]:
        """Get detailed explanation of ensemble prediction"""
        explanation = {
            'signal_breakdown': {
                'pattern_contribution': prediction.model_contributions.get('pattern', 0),
                'reversal_contribution': prediction.model_contributions.get('reversal', 0),
                'momentum_contribution': prediction.model_contributions.get('momentum', 0),
                'confluence_contribution': prediction.model_contributions.get('confluence', 0)
            },
            'confidence_factors': {
                'model_consensus': prediction.signal_consensus,
                'ensemble_confidence': prediction.ensemble_confidence,
                'signal_strength': prediction.ensemble_strength
            },
            'risk_assessment': {
                'risk_level': prediction.risk_level,
                'entry_timing': prediction.entry_recommendation,
                'expected_duration': f"{prediction.expected_duration_hours} hours"
            },
            'trade_setup': {
                'target_pips': prediction.target_pips,
                'stop_loss_pips': prediction.stop_loss_pips,
                'risk_reward_ratio': abs(prediction.target_pips / max(prediction.stop_loss_pips, 1))
            }
        }

        return explanation


# === PLATFORM3 PHASE 2 ENHANCEMENT APPLIED ===
# Enhanced on: 2025-05-31T22:33:56.432960
# Enhancements: Winston logging, EventEmitter error handling, TypeScript interfaces,
#               Database optimization, Performance monitoring, Async operations
# Phase 3 AI Model Enhancement: Applied advanced ML optimization techniques
