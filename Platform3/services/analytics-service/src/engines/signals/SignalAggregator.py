"""
Multi-Timeframe Signal Aggregation Engine
Combines signals from multiple timeframes for enhanced trading decisions

Features:
- Multi-timeframe signal combination (M1-H4)
- Weighted signal aggregation based on timeframe importance
- Signal conflict resolution with priority rules
- Confidence scoring and signal strength assessment
- Real-time signal synchronization
- Adaptive weighting based on market conditions
- Performance tracking and optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import redis
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timeframe(Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class SignalSource(Enum):
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    PATTERN = "pattern"
    VOLUME = "volume"
    ML_MODEL = "ml_model"
    SENTIMENT = "sentiment"

@dataclass
class TradingSignal:
    symbol: str
    timeframe: Timeframe
    signal_type: SignalType
    source: SignalSource
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiry: Optional[datetime] = None

@dataclass
class AggregatedSignal:
    symbol: str
    final_signal: SignalType
    overall_strength: float
    overall_confidence: float
    contributing_signals: List[TradingSignal]
    timeframe_consensus: Dict[Timeframe, SignalType]
    source_consensus: Dict[SignalSource, SignalType]
    risk_score: float
    timestamp: datetime

@dataclass
class SignalWeights:
    timeframe_weights: Dict[Timeframe, float] = field(default_factory=lambda: {
        Timeframe.M1: 0.1,
        Timeframe.M5: 0.15,
        Timeframe.M15: 0.2,
        Timeframe.M30: 0.25,
        Timeframe.H1: 0.2,
        Timeframe.H4: 0.1
    })
    source_weights: Dict[SignalSource, float] = field(default_factory=lambda: {
        SignalSource.TECHNICAL: 0.25,
        SignalSource.MOMENTUM: 0.2,
        SignalSource.PATTERN: 0.15,
        SignalSource.VOLUME: 0.15,
        SignalSource.ML_MODEL: 0.2,
        SignalSource.SENTIMENT: 0.05
    })

class SignalAggregator:
    """
    Multi-timeframe signal aggregation engine for enhanced trading decisions
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Configuration
        self.weights = SignalWeights()
        self.signal_buffer = defaultdict(list)  # symbol -> list of signals
        self.aggregation_config = {
            'max_signal_age': timedelta(minutes=5),  # Maximum age for signals
            'min_signals_required': 3,  # Minimum signals for aggregation
            'consensus_threshold': 0.6,  # Threshold for signal consensus
            'conflict_resolution_method': 'weighted_average',
            'adaptive_weighting': True,
            'real_time_updates': True
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_aggregations': 0,
            'successful_signals': 0,
            'signal_accuracy': 0.0,
            'average_confidence': 0.0,
            'timeframe_performance': {},
            'source_performance': {},
            'last_aggregation': None
        }
        
        # Signal history for performance analysis
        self.signal_history = []
        self.max_history_size = 1000
        
        logger.info("SignalAggregator initialized")

    async def add_signal(self, signal: TradingSignal) -> bool:
        """Add a new trading signal to the aggregation buffer"""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                logger.warning(f"Invalid signal rejected: {signal}")
                return False
            
            # Add to buffer
            self.signal_buffer[signal.symbol].append(signal)
            
            # Clean old signals
            await self._clean_old_signals(signal.symbol)
            
            # Cache signal in Redis for real-time access
            await self._cache_signal(signal)
            
            # Trigger real-time aggregation if enabled
            if self.aggregation_config['real_time_updates']:
                await self._trigger_real_time_aggregation(signal.symbol)
            
            logger.debug(f"✅ Signal added: {signal.symbol} {signal.timeframe.value} {signal.signal_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding signal: {e}")
            return False

    async def aggregate_signals(self, symbol: str, 
                              target_timeframes: Optional[List[Timeframe]] = None) -> Optional[AggregatedSignal]:
        """
        Aggregate signals for a symbol across multiple timeframes
        
        Args:
            symbol: Trading symbol to aggregate signals for
            target_timeframes: Specific timeframes to include (None for all)
            
        Returns:
            AggregatedSignal with combined analysis
        """
        try:
            start_time = datetime.now()
            
            # Get relevant signals
            signals = await self._get_relevant_signals(symbol, target_timeframes)
            
            if len(signals) < self.aggregation_config['min_signals_required']:
                logger.debug(f"Insufficient signals for {symbol}: {len(signals)}")
                return None
            
            # Calculate timeframe consensus
            timeframe_consensus = self._calculate_timeframe_consensus(signals)
            
            # Calculate source consensus
            source_consensus = self._calculate_source_consensus(signals)
            
            # Resolve conflicts and determine final signal
            final_signal = await self._resolve_signal_conflicts(signals, timeframe_consensus, source_consensus)
            
            # Calculate overall strength and confidence
            overall_strength = self._calculate_overall_strength(signals, final_signal)
            overall_confidence = self._calculate_overall_confidence(signals, timeframe_consensus, source_consensus)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(signals, final_signal)
            
            # Create aggregated signal
            aggregated = AggregatedSignal(
                symbol=symbol,
                final_signal=final_signal,
                overall_strength=overall_strength,
                overall_confidence=overall_confidence,
                contributing_signals=signals,
                timeframe_consensus=timeframe_consensus,
                source_consensus=source_consensus,
                risk_score=risk_score,
                timestamp=datetime.now()
            )
            
            # Update performance tracking
            self.performance_stats['total_aggregations'] += 1
            self.performance_stats['last_aggregation'] = datetime.now()
            
            # Store in history
            self._add_to_history(aggregated)
            
            # Cache result
            await self._cache_aggregated_signal(aggregated)
            
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.debug(f"✅ Signal aggregated for {symbol}: {final_signal.value} "
                        f"(confidence: {overall_confidence:.2f}, time: {calculation_time:.1f}ms)")
            
            return aggregated
            
        except Exception as e:
            logger.error(f"❌ Error aggregating signals for {symbol}: {e}")
            return None

    async def get_market_consensus(self, symbols: List[str]) -> Dict[str, AggregatedSignal]:
        """Get aggregated signals for multiple symbols"""
        results = {}
        
        tasks = [self.aggregate_signals(symbol) for symbol in symbols]
        aggregated_signals = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, aggregated_signals):
            if isinstance(result, AggregatedSignal):
                results[symbol] = result
            elif isinstance(result, Exception):
                logger.error(f"Error aggregating {symbol}: {result}")
        
        return results

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate signal data"""
        if not signal.symbol or not signal.timeframe or not signal.signal_type:
            return False
        
        if not (0.0 <= signal.strength <= 1.0) or not (0.0 <= signal.confidence <= 1.0):
            return False
        
        if signal.expiry and signal.expiry < datetime.now():
            return False
        
        return True

    async def _get_relevant_signals(self, symbol: str, 
                                  target_timeframes: Optional[List[Timeframe]] = None) -> List[TradingSignal]:
        """Get relevant signals for aggregation"""
        if symbol not in self.signal_buffer:
            return []
        
        signals = self.signal_buffer[symbol]
        
        # Filter by timeframes if specified
        if target_timeframes:
            signals = [s for s in signals if s.timeframe in target_timeframes]
        
        # Filter by age
        max_age = self.aggregation_config['max_signal_age']
        cutoff_time = datetime.now() - max_age
        signals = [s for s in signals if s.timestamp >= cutoff_time]
        
        # Filter by expiry
        signals = [s for s in signals if not s.expiry or s.expiry >= datetime.now()]
        
        return signals

    def _calculate_timeframe_consensus(self, signals: List[TradingSignal]) -> Dict[Timeframe, SignalType]:
        """Calculate consensus signal for each timeframe"""
        timeframe_signals = defaultdict(list)
        
        # Group signals by timeframe
        for signal in signals:
            timeframe_signals[signal.timeframe].append(signal)
        
        consensus = {}
        for timeframe, tf_signals in timeframe_signals.items():
            # Weight signals by strength and confidence
            signal_scores = defaultdict(float)
            
            for signal in tf_signals:
                weight = signal.strength * signal.confidence
                signal_scores[signal.signal_type] += weight
            
            # Find consensus signal
            if signal_scores:
                consensus_signal = max(signal_scores.items(), key=lambda x: x[1])[0]
                consensus[timeframe] = consensus_signal
        
        return consensus

    def _calculate_source_consensus(self, signals: List[TradingSignal]) -> Dict[SignalSource, SignalType]:
        """Calculate consensus signal for each source"""
        source_signals = defaultdict(list)
        
        # Group signals by source
        for signal in signals:
            source_signals[signal.source].append(signal)
        
        consensus = {}
        for source, src_signals in source_signals.items():
            # Weight signals by strength and confidence
            signal_scores = defaultdict(float)
            
            for signal in src_signals:
                weight = signal.strength * signal.confidence
                signal_scores[signal.signal_type] += weight
            
            # Find consensus signal
            if signal_scores:
                consensus_signal = max(signal_scores.items(), key=lambda x: x[1])[0]
                consensus[source] = consensus_signal
        
        return consensus

    async def _resolve_signal_conflicts(self, signals: List[TradingSignal], 
                                      timeframe_consensus: Dict[Timeframe, SignalType],
                                      source_consensus: Dict[SignalSource, SignalType]) -> SignalType:
        """Resolve conflicts between signals using weighted voting"""
        signal_scores = defaultdict(float)
        
        # Weight by timeframe consensus
        for timeframe, signal_type in timeframe_consensus.items():
            weight = self.weights.timeframe_weights.get(timeframe, 0.1)
            signal_scores[signal_type] += weight
        
        # Weight by source consensus
        for source, signal_type in source_consensus.items():
            weight = self.weights.source_weights.get(source, 0.1)
            signal_scores[signal_type] += weight
        
        # Weight individual signals
        for signal in signals:
            tf_weight = self.weights.timeframe_weights.get(signal.timeframe, 0.1)
            src_weight = self.weights.source_weights.get(signal.source, 0.1)
            combined_weight = tf_weight * src_weight * signal.strength * signal.confidence
            signal_scores[signal.signal_type] += combined_weight
        
        # Apply adaptive weighting if enabled
        if self.aggregation_config['adaptive_weighting']:
            signal_scores = await self._apply_adaptive_weighting(signal_scores, signals)
        
        # Return signal with highest score
        if signal_scores:
            return max(signal_scores.items(), key=lambda x: x[1])[0]
        else:
            return SignalType.HOLD

    async def _apply_adaptive_weighting(self, signal_scores: Dict[SignalType, float], 
                                      signals: List[TradingSignal]) -> Dict[SignalType, float]:
        """Apply adaptive weighting based on recent performance"""
        # Get recent performance data
        recent_performance = await self._get_recent_performance()
        
        # Adjust scores based on performance
        for signal_type, score in signal_scores.items():
            performance_multiplier = recent_performance.get(signal_type, 1.0)
            signal_scores[signal_type] = score * performance_multiplier
        
        return signal_scores

    def _calculate_overall_strength(self, signals: List[TradingSignal], final_signal: SignalType) -> float:
        """Calculate overall signal strength"""
        relevant_signals = [s for s in signals if s.signal_type == final_signal]
        
        if not relevant_signals:
            return 0.0
        
        # Weight by timeframe and source importance
        total_weight = 0.0
        weighted_strength = 0.0
        
        for signal in relevant_signals:
            tf_weight = self.weights.timeframe_weights.get(signal.timeframe, 0.1)
            src_weight = self.weights.source_weights.get(signal.source, 0.1)
            combined_weight = tf_weight * src_weight
            
            weighted_strength += signal.strength * combined_weight
            total_weight += combined_weight
        
        return weighted_strength / total_weight if total_weight > 0 else 0.0

    def _calculate_overall_confidence(self, signals: List[TradingSignal],
                                    timeframe_consensus: Dict[Timeframe, SignalType],
                                    source_consensus: Dict[SignalSource, SignalType]) -> float:
        """Calculate overall confidence score"""
        # Base confidence on signal consensus
        total_signals = len(signals)
        if total_signals == 0:
            return 0.0
        
        # Count signals supporting the majority
        signal_counts = defaultdict(int)
        for signal in signals:
            signal_counts[signal.signal_type] += 1
        
        max_count = max(signal_counts.values()) if signal_counts else 0
        consensus_ratio = max_count / total_signals
        
        # Factor in timeframe and source diversity
        timeframe_diversity = len(timeframe_consensus) / len(Timeframe)
        source_diversity = len(source_consensus) / len(SignalSource)
        
        # Calculate weighted confidence
        base_confidence = consensus_ratio * 0.6
        diversity_bonus = (timeframe_diversity + source_diversity) * 0.2
        
        # Average individual signal confidence
        avg_signal_confidence = np.mean([s.confidence for s in signals])
        
        overall_confidence = base_confidence + diversity_bonus + avg_signal_confidence * 0.2
        
        return min(overall_confidence, 1.0)

    def _calculate_risk_score(self, signals: List[TradingSignal], final_signal: SignalType) -> float:
        """Calculate risk score for the aggregated signal"""
        # Count conflicting signals
        conflicting_signals = [s for s in signals if s.signal_type != final_signal and s.signal_type != SignalType.HOLD]
        conflict_ratio = len(conflicting_signals) / len(signals) if signals else 0
        
        # Factor in signal strength variance
        strengths = [s.strength for s in signals]
        strength_variance = np.var(strengths) if len(strengths) > 1 else 0
        
        # Factor in confidence variance
        confidences = [s.confidence for s in signals]
        confidence_variance = np.var(confidences) if len(confidences) > 1 else 0
        
        # Calculate composite risk score
        risk_score = (conflict_ratio * 0.5 + strength_variance * 0.3 + confidence_variance * 0.2)
        
        return min(risk_score, 1.0)

    async def _clean_old_signals(self, symbol: str):
        """Remove old signals from buffer"""
        if symbol not in self.signal_buffer:
            return
        
        max_age = self.aggregation_config['max_signal_age']
        cutoff_time = datetime.now() - max_age
        
        self.signal_buffer[symbol] = [
            s for s in self.signal_buffer[symbol] 
            if s.timestamp >= cutoff_time and (not s.expiry or s.expiry >= datetime.now())
        ]

    async def _trigger_real_time_aggregation(self, symbol: str):
        """Trigger real-time signal aggregation"""
        try:
            aggregated = await self.aggregate_signals(symbol)
            if aggregated:
                # Publish real-time update
                await self._publish_real_time_update(aggregated)
        except Exception as e:
            logger.error(f"Error in real-time aggregation for {symbol}: {e}")

    async def _cache_signal(self, signal: TradingSignal):
        """Cache signal in Redis"""
        try:
            signal_data = {
                'symbol': signal.symbol,
                'timeframe': signal.timeframe.value,
                'signal_type': signal.signal_type.value,
                'source': signal.source.value,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'timestamp': signal.timestamp.isoformat(),
                'metadata': signal.metadata
            }
            
            key = f"signal:{signal.symbol}:{signal.timeframe.value}:{signal.source.value}"
            await self.redis_client.setex(key, 300, json.dumps(signal_data))  # 5 min TTL
            
        except Exception as e:
            logger.error(f"Error caching signal: {e}")

    async def _cache_aggregated_signal(self, aggregated: AggregatedSignal):
        """Cache aggregated signal in Redis"""
        try:
            aggregated_data = {
                'symbol': aggregated.symbol,
                'final_signal': aggregated.final_signal.value,
                'overall_strength': aggregated.overall_strength,
                'overall_confidence': aggregated.overall_confidence,
                'risk_score': aggregated.risk_score,
                'timestamp': aggregated.timestamp.isoformat(),
                'contributing_signals_count': len(aggregated.contributing_signals)
            }
            
            key = f"aggregated_signal:{aggregated.symbol}"
            await self.redis_client.setex(key, 300, json.dumps(aggregated_data))  # 5 min TTL
            
        except Exception as e:
            logger.error(f"Error caching aggregated signal: {e}")

    async def _publish_real_time_update(self, aggregated: AggregatedSignal):
        """Publish real-time signal update"""
        try:
            update_data = {
                'type': 'signal_update',
                'symbol': aggregated.symbol,
                'signal': aggregated.final_signal.value,
                'strength': aggregated.overall_strength,
                'confidence': aggregated.overall_confidence,
                'timestamp': aggregated.timestamp.isoformat()
            }
            
            await self.redis_client.publish('signal_updates', json.dumps(update_data))
            
        except Exception as e:
            logger.error(f"Error publishing real-time update: {e}")

    def _add_to_history(self, aggregated: AggregatedSignal):
        """Add aggregated signal to history"""
        self.signal_history.append(aggregated)
        
        # Maintain history size
        if len(self.signal_history) > self.max_history_size:
            self.signal_history = self.signal_history[-self.max_history_size:]

    async def _get_recent_performance(self) -> Dict[SignalType, float]:
        """Get recent performance data for adaptive weighting"""
        # Placeholder for performance tracking
        # In a real implementation, this would analyze recent signal accuracy
        return {
            SignalType.BUY: 1.0,
            SignalType.SELL: 1.0,
            SignalType.HOLD: 1.0,
            SignalType.STRONG_BUY: 1.1,
            SignalType.STRONG_SELL: 1.1
        }

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get aggregation performance statistics"""
        return {
            'performance_stats': self.performance_stats,
            'buffer_sizes': {symbol: len(signals) for symbol, signals in self.signal_buffer.items()},
            'configuration': self.aggregation_config,
            'weights': {
                'timeframe_weights': {tf.value: weight for tf, weight in self.weights.timeframe_weights.items()},
                'source_weights': {src.value: weight for src, weight in self.weights.source_weights.items()}
            }
        }

    def update_weights(self, new_weights: SignalWeights):
        """Update signal aggregation weights"""
        self.weights = new_weights
        logger.info("Signal aggregation weights updated")

    def clear_buffer(self, symbol: Optional[str] = None):
        """Clear signal buffer"""
        if symbol:
            if symbol in self.signal_buffer:
                del self.signal_buffer[symbol]
        else:
            self.signal_buffer.clear()
        
        logger.info(f"Signal buffer cleared for {symbol if symbol else 'all symbols'}")
