"""
Scalping Ensemble
Ensemble methods for M1-M5 scalping combining multiple ML models.
Provides robust predictions by combining LSTM, classifiers, and filters.
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

from .ScalpingLSTM import ScalpingLSTM, LSTMPrediction
from .TickClassifier import TickClassifier, TickPrediction
from .SpreadPredictor import SpreadPredictor, SpreadPrediction
from .NoiseFilter import NoiseFilter, FilteredSignal


@dataclass
class EnsemblePrediction:
    """Combined ensemble prediction result"""
    timestamp: float
    symbol: str
    timeframe: str
    
    # Combined predictions
    final_direction: str  # 'buy', 'sell', 'hold'
    final_confidence: float  # 0-1
    final_strength: float  # 0-100
    
    # Price predictions
    predicted_price: float
    price_change_pips: float
    price_confidence: float
    
    # Entry optimization
    optimal_entry_price: float
    optimal_spread: float
    entry_timing: str  # 'immediate', 'wait_short', 'wait_long'
    
    # Risk metrics
    risk_score: float  # 0-1, higher is riskier
    signal_quality: float  # 0-1, higher is better
    noise_level: float  # 0-1, lower is better
    
    # Component predictions
    lstm_prediction: Optional[LSTMPrediction]
    tick_prediction: Optional[TickPrediction]
    spread_prediction: Optional[SpreadPrediction]
    filtered_signals: List[FilteredSignal]
    
    # Ensemble metadata
    model_agreement: float  # 0-1, how well models agree
    prediction_latency_ms: float
    ensemble_version: str


@dataclass
class EnsembleWeights:
    """Weights for ensemble components"""
    lstm_weight: float = 0.4
    tick_classifier_weight: float = 0.3
    spread_predictor_weight: float = 0.2
    noise_filter_weight: float = 0.1


@dataclass
class EnsembleMetrics:
    """Ensemble performance metrics"""
    total_predictions: int
    average_latency_ms: float
    model_agreement_avg: float
    accuracy_by_timeframe: Dict[str, float]
    component_performance: Dict[str, Dict[str, float]]


class ScalpingEnsemble:
    """
    Scalping Ensemble Model
    Combines multiple ML models for robust M1-M5 scalping predictions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Ensemble configuration
        self.ensemble_weights = EnsembleWeights(
            lstm_weight=self.config.get('lstm_weight', 0.4),
            tick_classifier_weight=self.config.get('tick_classifier_weight', 0.3),
            spread_predictor_weight=self.config.get('spread_predictor_weight', 0.2),
            noise_filter_weight=self.config.get('noise_filter_weight', 0.1)
        )
        
        # Component models
        self.lstm_model = ScalpingLSTM(self.config.get('lstm_config', {}))
        self.tick_classifier = TickClassifier(self.config.get('tick_config', {}))
        self.spread_predictor = SpreadPredictor(self.config.get('spread_config', {}))
        self.noise_filter = NoiseFilter(self.config.get('noise_config', {}))
        
        # Ensemble state
        self.prediction_history = {}  # symbol -> deque of predictions
        self.model_performance = {}  # symbol -> performance metrics
        self.adaptive_weights = {}  # symbol -> adaptive weights
        self.max_history_size = 1000
        
        # Agreement thresholds
        self.agreement_thresholds = {
            'high': 0.8,    # High agreement between models
            'medium': 0.6,  # Medium agreement
            'low': 0.4      # Low agreement
        }
        
        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.ensemble_accuracy = {}  # symbol -> accuracy history
        
        # Version
        self.ensemble_version = "1.0.0"
        
    async def initialize(self) -> None:
        """Initialize all ensemble components"""
        try:
            # Initialize component models
            await self.lstm_model.initialize()
            await self.tick_classifier.initialize()
            await self.spread_predictor.initialize()
            await self.noise_filter.initialize()
            
            self.logger.info("Scalping Ensemble initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Scalping Ensemble: {e}")
            raise
    
    async def predict(
        self, 
        symbol: str, 
        timeframe: str, 
        market_data: List[Dict], 
        tick_data: Optional[List[Dict]] = None
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction combining all models
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not market_data:
                raise ValueError("No market data provided")
            
            if timeframe not in ['M1', 'M5']:
                raise ValueError(f"Unsupported timeframe: {timeframe}. Use M1 or M5.")
            
            # Get component predictions
            component_predictions = await self._get_component_predictions(
                symbol, timeframe, market_data, tick_data
            )
            
            # Apply noise filtering to key signals
            filtered_signals = await self._apply_noise_filtering(symbol, market_data)
            
            # Combine predictions using ensemble logic
            ensemble_result = await self._combine_predictions(
                symbol, timeframe, component_predictions, filtered_signals
            )
            
            # Calculate ensemble metrics
            ensemble_metrics = await self._calculate_ensemble_metrics(
                component_predictions, ensemble_result
            )
            
            # Update prediction history
            await self._update_prediction_history(symbol, ensemble_result)
            
            # Update adaptive weights based on performance
            await self._update_adaptive_weights(symbol, component_predictions)
            
            # Update performance tracking
            prediction_time = time.time() - start_time
            self.prediction_count += 1
            self.total_prediction_time += prediction_time
            
            ensemble_result.prediction_latency_ms = prediction_time * 1000
            ensemble_result.ensemble_version = self.ensemble_version
            
            self.logger.debug(f"Ensemble prediction for {symbol} {timeframe} completed in {prediction_time:.3f}s")
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed for {symbol}: {e}")
            raise
    
    async def _get_component_predictions(
        self, 
        symbol: str, 
        timeframe: str, 
        market_data: List[Dict], 
        tick_data: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """Get predictions from all component models"""
        
        predictions = {}
        
        try:
            # LSTM prediction
            lstm_pred = await self.lstm_model.predict_price(symbol, timeframe, market_data)
            predictions['lstm'] = lstm_pred
        except Exception as e:
            self.logger.warning(f"LSTM prediction failed: {e}")
            predictions['lstm'] = None
        
        try:
            # Tick classifier prediction
            if tick_data and len(tick_data) > 0:
                tick_pred = await self.tick_classifier.predict_next_tick(symbol, tick_data)
                predictions['tick'] = tick_pred
            else:
                # Use market data as tick data fallback
                tick_data_fallback = [
                    {
                        'price': data.get('close', 0),
                        'volume': data.get('volume', 0),
                        'spread': data.get('spread', 0.0001),
                        'timestamp': data.get('timestamp', time.time())
                    }
                    for data in market_data[-50:]  # Last 50 bars as ticks
                ]
                tick_pred = await self.tick_classifier.predict_next_tick(symbol, tick_data_fallback)
                predictions['tick'] = tick_pred
        except Exception as e:
            self.logger.warning(f"Tick classifier prediction failed: {e}")
            predictions['tick'] = None
        
        try:
            # Spread prediction
            spread_pred = await self.spread_predictor.predict_spread(symbol, market_data)
            predictions['spread'] = spread_pred
        except Exception as e:
            self.logger.warning(f"Spread prediction failed: {e}")
            predictions['spread'] = None
        
        return predictions
    
    async def _apply_noise_filtering(self, symbol: str, market_data: List[Dict]) -> List[FilteredSignal]:
        """Apply noise filtering to key market signals"""
        
        filtered_signals = []
        
        try:
            # Filter price signal
            prices = [float(data.get('close', 0)) for data in market_data[-100:]]
            if len(prices) >= 10:
                price_filtered = await self.noise_filter.filter_signal(symbol, prices, 'adaptive')
                filtered_signals.append(price_filtered)
            
            # Filter volume signal
            volumes = [float(data.get('volume', 0)) for data in market_data[-100:]]
            if len(volumes) >= 10:
                volume_filtered = await self.noise_filter.filter_signal(symbol, volumes, 'savgol')
                filtered_signals.append(volume_filtered)
            
            # Filter spread signal
            spreads = [float(data.get('spread', 0.0001)) for data in market_data[-100:]]
            if len(spreads) >= 10:
                spread_filtered = await self.noise_filter.filter_signal(symbol, spreads, 'butterworth')
                filtered_signals.append(spread_filtered)
                
        except Exception as e:
            self.logger.warning(f"Noise filtering failed for {symbol}: {e}")
        
        return filtered_signals
    
    async def _combine_predictions(
        self, 
        symbol: str, 
        timeframe: str, 
        component_predictions: Dict[str, Any], 
        filtered_signals: List[FilteredSignal]
    ) -> EnsemblePrediction:
        """Combine component predictions into ensemble result"""
        
        # Get adaptive weights or use defaults
        weights = self.adaptive_weights.get(symbol, self.ensemble_weights)
        
        # Extract component predictions
        lstm_pred = component_predictions.get('lstm')
        tick_pred = component_predictions.get('tick')
        spread_pred = component_predictions.get('spread')
        
        # Initialize ensemble values
        direction_votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        confidence_sum = 0.0
        total_weight = 0.0
        
        # LSTM contribution
        if lstm_pred:
            if lstm_pred.predicted_direction == 'up':
                direction_votes['buy'] += weights.lstm_weight * lstm_pred.confidence
            elif lstm_pred.predicted_direction == 'down':
                direction_votes['sell'] += weights.lstm_weight * lstm_pred.confidence
            else:
                direction_votes['hold'] += weights.lstm_weight * lstm_pred.confidence
            
            confidence_sum += lstm_pred.confidence * weights.lstm_weight
            total_weight += weights.lstm_weight
        
        # Tick classifier contribution
        if tick_pred:
            if tick_pred.predicted_direction == 'up':
                direction_votes['buy'] += weights.tick_classifier_weight * tick_pred.confidence
            elif tick_pred.predicted_direction == 'down':
                direction_votes['sell'] += weights.tick_classifier_weight * tick_pred.confidence
            else:
                direction_votes['hold'] += weights.tick_classifier_weight * tick_pred.confidence
            
            confidence_sum += tick_pred.confidence * weights.tick_classifier_weight
            total_weight += weights.tick_classifier_weight
        
        # Spread predictor contribution (affects timing more than direction)
        if spread_pred:
            # Spread prediction influences confidence and timing
            spread_confidence_factor = 1.0
            if spread_pred.spread_direction == 'tightening':
                spread_confidence_factor = 1.1  # Boost confidence for tighter spreads
            elif spread_pred.spread_direction == 'widening':
                spread_confidence_factor = 0.9  # Reduce confidence for wider spreads
            
            confidence_sum *= spread_confidence_factor
        
        # Determine final direction
        final_direction = max(direction_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate final confidence
        final_confidence = confidence_sum / max(total_weight, 0.1)
        
        # Apply noise filter adjustment
        if filtered_signals:
            avg_signal_quality = np.mean([fs.signal_quality for fs in filtered_signals])
            avg_noise_level = np.mean([fs.noise_level for fs in filtered_signals])
            
            # Adjust confidence based on signal quality
            final_confidence *= avg_signal_quality
            final_confidence *= (1.0 - avg_noise_level * 0.5)  # Reduce confidence for noisy signals
        else:
            avg_signal_quality = 0.7
            avg_noise_level = 0.3
        
        # Calculate final strength (0-100)
        final_strength = final_confidence * 100
        
        # Determine price prediction (weighted average)
        predicted_price = 0.0
        price_confidence = 0.0
        price_change_pips = 0.0
        
        if lstm_pred:
            predicted_price = lstm_pred.predicted_price
            price_confidence = lstm_pred.confidence
            price_change_pips = lstm_pred.price_change_pips
        elif tick_pred:
            # Estimate price from tick prediction
            current_price = float(component_predictions.get('current_price', 1.0))
            if tick_pred.predicted_direction == 'up':
                predicted_price = current_price * 1.0001  # Small upward movement
                price_change_pips = 1.0
            elif tick_pred.predicted_direction == 'down':
                predicted_price = current_price * 0.9999  # Small downward movement
                price_change_pips = -1.0
            else:
                predicted_price = current_price
                price_change_pips = 0.0
            price_confidence = tick_pred.confidence
        
        # Determine optimal entry
        optimal_entry_price = predicted_price
        optimal_spread = 0.0001  # Default spread
        entry_timing = 'immediate'
        
        if spread_pred:
            optimal_spread = spread_pred.predicted_spread
            entry_timing = spread_pred.optimal_entry_timing
            
            # Adjust entry price based on spread prediction
            if spread_pred.spread_direction == 'tightening':
                # Wait for better spread
                entry_timing = 'wait_short'
            elif spread_pred.spread_direction == 'widening':
                # Enter immediately before spread widens
                entry_timing = 'immediate'
        
        # Calculate risk metrics
        risk_score = await self._calculate_risk_score(
            component_predictions, filtered_signals, final_confidence
        )
        
        # Calculate model agreement
        model_agreement = await self._calculate_model_agreement(component_predictions)
        
        return EnsemblePrediction(
            timestamp=time.time(),
            symbol=symbol,
            timeframe=timeframe,
            final_direction=final_direction,
            final_confidence=min(max(final_confidence, 0.0), 1.0),
            final_strength=min(max(final_strength, 0.0), 100.0),
            predicted_price=predicted_price,
            price_change_pips=price_change_pips,
            price_confidence=price_confidence,
            optimal_entry_price=optimal_entry_price,
            optimal_spread=optimal_spread,
            entry_timing=entry_timing,
            risk_score=min(max(risk_score, 0.0), 1.0),
            signal_quality=avg_signal_quality,
            noise_level=avg_noise_level,
            lstm_prediction=lstm_pred,
            tick_prediction=tick_pred,
            spread_prediction=spread_pred,
            filtered_signals=filtered_signals,
            model_agreement=model_agreement,
            prediction_latency_ms=0.0,  # Will be set by caller
            ensemble_version=self.ensemble_version
        )
    
    async def _calculate_risk_score(
        self, 
        component_predictions: Dict[str, Any], 
        filtered_signals: List[FilteredSignal], 
        final_confidence: float
    ) -> float:
        """Calculate overall risk score for the prediction"""
        
        risk_factors = []
        
        # Low confidence increases risk
        confidence_risk = 1.0 - final_confidence
        risk_factors.append(confidence_risk * 0.3)
        
        # Model disagreement increases risk
        model_agreement = await self._calculate_model_agreement(component_predictions)
        disagreement_risk = 1.0 - model_agreement
        risk_factors.append(disagreement_risk * 0.3)
        
        # High noise level increases risk
        if filtered_signals:
            avg_noise = np.mean([fs.noise_level for fs in filtered_signals])
            risk_factors.append(avg_noise * 0.2)
        else:
            risk_factors.append(0.3 * 0.2)  # Default noise risk
        
        # Spread risk
        spread_pred = component_predictions.get('spread')
        if spread_pred and spread_pred.spread_direction == 'widening':
            risk_factors.append(0.5 * 0.2)  # Higher risk for widening spreads
        else:
            risk_factors.append(0.2 * 0.2)  # Lower spread risk
        
        return sum(risk_factors)
    
    async def _calculate_model_agreement(self, component_predictions: Dict[str, Any]) -> float:
        """Calculate agreement between component models"""
        
        predictions = []
        
        # Collect direction predictions
        lstm_pred = component_predictions.get('lstm')
        if lstm_pred:
            if lstm_pred.predicted_direction == 'up':
                predictions.append(1)
            elif lstm_pred.predicted_direction == 'down':
                predictions.append(-1)
            else:
                predictions.append(0)
        
        tick_pred = component_predictions.get('tick')
        if tick_pred:
            if tick_pred.predicted_direction == 'up':
                predictions.append(1)
            elif tick_pred.predicted_direction == 'down':
                predictions.append(-1)
            else:
                predictions.append(0)
        
        # Calculate agreement as inverse of standard deviation
        if len(predictions) < 2:
            return 0.5  # Neutral agreement for single model
        
        agreement = 1.0 - (np.std(predictions) / 1.0)  # Normalize by max possible std
        return max(0.0, min(agreement, 1.0))
    
    async def _update_prediction_history(self, symbol: str, prediction: EnsemblePrediction) -> None:
        """Update prediction history for performance tracking"""
        
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = deque(maxlen=self.max_history_size)
        
        self.prediction_history[symbol].append({
            'timestamp': prediction.timestamp,
            'direction': prediction.final_direction,
            'confidence': prediction.final_confidence,
            'strength': prediction.final_strength,
            'model_agreement': prediction.model_agreement,
            'risk_score': prediction.risk_score
        })
    
    async def _update_adaptive_weights(self, symbol: str, component_predictions: Dict[str, Any]) -> None:
        """Update adaptive weights based on component performance"""
        
        # This is a simplified adaptive weighting scheme
        # In a real implementation, this would track actual performance over time
        
        if symbol not in self.adaptive_weights:
            self.adaptive_weights[symbol] = self.ensemble_weights
        
        # For now, keep weights static
        # Future enhancement: implement performance-based weight adjustment
        pass
    
    async def get_ensemble_metrics(self) -> EnsembleMetrics:
        """Get comprehensive ensemble performance metrics"""
        
        # Component performance
        component_performance = {}
        
        try:
            lstm_metrics = await self.lstm_model.get_performance_metrics()
            component_performance['lstm'] = lstm_metrics
        except Exception:
            component_performance['lstm'] = {}
        
        try:
            tick_metrics = await self.tick_classifier.get_performance_metrics()
            component_performance['tick_classifier'] = tick_metrics
        except Exception:
            component_performance['tick_classifier'] = {}
        
        try:
            spread_metrics = await self.spread_predictor.get_performance_metrics()
            component_performance['spread_predictor'] = spread_metrics
        except Exception:
            component_performance['spread_predictor'] = {}
        
        try:
            noise_metrics = await self.noise_filter.get_performance_metrics()
            component_performance['noise_filter'] = noise_metrics
        except Exception:
            component_performance['noise_filter'] = {}
        
        # Calculate average model agreement
        all_agreements = []
        for symbol_history in self.prediction_history.values():
            agreements = [pred['model_agreement'] for pred in symbol_history]
            all_agreements.extend(agreements)
        
        avg_agreement = np.mean(all_agreements) if all_agreements else 0.5
        
        # Accuracy by timeframe (placeholder)
        accuracy_by_timeframe = {
            'M1': 0.65,  # Placeholder values
            'M5': 0.70
        }
        
        return EnsembleMetrics(
            total_predictions=self.prediction_count,
            average_latency_ms=(self.total_prediction_time / self.prediction_count * 1000) 
                             if self.prediction_count > 0 else 0,
            model_agreement_avg=avg_agreement,
            accuracy_by_timeframe=accuracy_by_timeframe,
            component_performance=component_performance
        )
    
    async def get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance metrics for a specific symbol"""
        
        if symbol not in self.prediction_history:
            return {'status': 'no_data'}
        
        history = list(self.prediction_history[symbol])
        
        if not history:
            return {'status': 'no_data'}
        
        # Calculate statistics
        confidences = [pred['confidence'] for pred in history]
        agreements = [pred['model_agreement'] for pred in history]
        risk_scores = [pred['risk_score'] for pred in history]
        
        return {
            'symbol': symbol,
            'total_predictions': len(history),
            'avg_confidence': np.mean(confidences),
            'avg_model_agreement': np.mean(agreements),
            'avg_risk_score': np.mean(risk_scores),
            'direction_distribution': {
                'buy': sum(1 for pred in history if pred['direction'] == 'buy'),
                'sell': sum(1 for pred in history if pred['direction'] == 'sell'),
                'hold': sum(1 for pred in history if pred['direction'] == 'hold')
            },
            'last_prediction': history[-1]['timestamp']
        }
