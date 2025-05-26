"""
Signal Conflict Resolution Engine
Advanced conflict resolution for contradictory trading signals

Features:
- Multi-dimensional conflict detection
- Priority-based signal resolution
- Confidence-weighted decision making
- Temporal signal analysis
- Source reliability tracking
- Adaptive resolution strategies
- Real-time conflict monitoring
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
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConflictType(Enum):
    DIRECTIONAL = "directional"  # Buy vs Sell signals
    STRENGTH = "strength"        # Strong vs Weak signals
    TEMPORAL = "temporal"        # Recent vs Old signals
    SOURCE = "source"           # Different sources disagreeing
    TIMEFRAME = "timeframe"     # Different timeframes disagreeing

class ResolutionStrategy(Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    HIGHEST_CONFIDENCE = "highest_confidence"
    MAJORITY_VOTE = "majority_vote"
    SOURCE_PRIORITY = "source_priority"
    TIMEFRAME_PRIORITY = "timeframe_priority"
    ADAPTIVE = "adaptive"

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

class Timeframe(Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    source: SignalSource
    timeframe: Timeframe
    strength: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SignalConflict:
    conflict_id: str
    conflict_type: ConflictType
    conflicting_signals: List[TradingSignal]
    severity: float  # 0.0 to 1.0
    detected_at: datetime
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolved_signal: Optional[TradingSignal] = None
    resolution_confidence: float = 0.0

@dataclass
class ResolutionWeights:
    source_weights: Dict[SignalSource, float] = field(default_factory=lambda: {
        SignalSource.ML_MODEL: 0.3,
        SignalSource.TECHNICAL: 0.25,
        SignalSource.MOMENTUM: 0.2,
        SignalSource.PATTERN: 0.15,
        SignalSource.VOLUME: 0.1,
        SignalSource.SENTIMENT: 0.05
    })
    timeframe_weights: Dict[Timeframe, float] = field(default_factory=lambda: {
        Timeframe.M1: 0.1,
        Timeframe.M5: 0.15,
        Timeframe.M15: 0.2,
        Timeframe.M30: 0.25,
        Timeframe.H1: 0.2,
        Timeframe.H4: 0.1
    })
    recency_decay: float = 0.95  # Decay factor for older signals

class ConflictResolver:
    """
    Advanced signal conflict resolution engine
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Configuration
        self.weights = ResolutionWeights()
        self.default_strategy = ResolutionStrategy.ADAPTIVE
        self.conflict_threshold = 0.3  # Minimum difference to consider conflict
        
        # Conflict tracking
        self.active_conflicts = {}
        self.conflict_history = deque(maxlen=1000)
        self.source_reliability = defaultdict(lambda: 0.8)  # Default reliability
        
        # Performance tracking
        self.resolution_stats = {
            'total_conflicts': 0,
            'resolved_conflicts': 0,
            'resolution_accuracy': 0.0,
            'average_resolution_time': 0.0,
            'strategy_performance': defaultdict(lambda: {'count': 0, 'accuracy': 0.0})
        }
        
        # Adaptive learning
        self.strategy_success_rates = defaultdict(lambda: 0.5)
        self.learning_rate = 0.1
        
        logger.info("ConflictResolver initialized")

    async def detect_conflicts(self, signals: List[TradingSignal]) -> List[SignalConflict]:
        """Detect conflicts among trading signals"""
        try:
            conflicts = []
            
            if len(signals) < 2:
                return conflicts
            
            # Group signals by symbol
            symbol_signals = defaultdict(list)
            for signal in signals:
                symbol_signals[signal.symbol].append(signal)
            
            # Detect conflicts for each symbol
            for symbol, symbol_signal_list in symbol_signals.items():
                symbol_conflicts = await self._detect_symbol_conflicts(symbol, symbol_signal_list)
                conflicts.extend(symbol_conflicts)
            
            # Store conflicts
            for conflict in conflicts:
                self.active_conflicts[conflict.conflict_id] = conflict
                self.conflict_history.append(conflict)
            
            self.resolution_stats['total_conflicts'] += len(conflicts)
            
            logger.debug(f"Detected {len(conflicts)} signal conflicts")
            
            return conflicts
            
        except Exception as e:
            logger.error(f"❌ Error detecting conflicts: {e}")
            return []

    async def resolve_conflict(self, conflict: SignalConflict, 
                             strategy: Optional[ResolutionStrategy] = None) -> Optional[TradingSignal]:
        """Resolve a signal conflict using specified or adaptive strategy"""
        try:
            start_time = datetime.now()
            
            # Choose resolution strategy
            if strategy is None:
                strategy = await self._choose_optimal_strategy(conflict)
            
            # Apply resolution strategy
            resolved_signal = await self._apply_resolution_strategy(conflict, strategy)
            
            if resolved_signal:
                # Update conflict with resolution
                conflict.resolution_strategy = strategy
                conflict.resolved_signal = resolved_signal
                conflict.resolution_confidence = await self._calculate_resolution_confidence(
                    conflict, resolved_signal
                )
                
                # Update performance tracking
                resolution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.resolution_stats['resolved_conflicts'] += 1
                self.resolution_stats['average_resolution_time'] = (
                    (self.resolution_stats['average_resolution_time'] * 
                     (self.resolution_stats['resolved_conflicts'] - 1) + resolution_time) /
                    self.resolution_stats['resolved_conflicts']
                )
                
                # Cache resolution
                await self._cache_resolution(conflict)
                
                logger.debug(f"Conflict resolved using {strategy.value}: "
                           f"{resolved_signal.signal_type.value} "
                           f"(confidence: {conflict.resolution_confidence:.3f})")
                
                return resolved_signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error resolving conflict: {e}")
            return None

    async def resolve_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Detect and resolve all conflicts in a list of signals"""
        try:
            # Detect conflicts
            conflicts = await self.detect_conflicts(signals)
            
            if not conflicts:
                return signals  # No conflicts to resolve
            
            # Resolve each conflict
            resolved_signals = []
            unresolved_signals = signals.copy()
            
            for conflict in conflicts:
                resolved_signal = await self.resolve_conflict(conflict)
                
                if resolved_signal:
                    resolved_signals.append(resolved_signal)
                    
                    # Remove conflicting signals from unresolved list
                    for conflicting_signal in conflict.conflicting_signals:
                        if conflicting_signal in unresolved_signals:
                            unresolved_signals.remove(conflicting_signal)
            
            # Combine resolved and non-conflicting signals
            final_signals = resolved_signals + unresolved_signals
            
            logger.debug(f"Resolved {len(conflicts)} conflicts, "
                        f"returning {len(final_signals)} signals")
            
            return final_signals
            
        except Exception as e:
            logger.error(f"❌ Error resolving signals: {e}")
            return signals

    async def _detect_symbol_conflicts(self, symbol: str, 
                                     signals: List[TradingSignal]) -> List[SignalConflict]:
        """Detect conflicts for signals of a specific symbol"""
        conflicts = []
        
        try:
            # Directional conflicts (Buy vs Sell)
            directional_conflicts = await self._detect_directional_conflicts(symbol, signals)
            conflicts.extend(directional_conflicts)
            
            # Strength conflicts (Strong vs Weak)
            strength_conflicts = await self._detect_strength_conflicts(symbol, signals)
            conflicts.extend(strength_conflicts)
            
            # Temporal conflicts (Recent vs Old)
            temporal_conflicts = await self._detect_temporal_conflicts(symbol, signals)
            conflicts.extend(temporal_conflicts)
            
            # Source conflicts (Different sources disagreeing)
            source_conflicts = await self._detect_source_conflicts(symbol, signals)
            conflicts.extend(source_conflicts)
            
            # Timeframe conflicts (Different timeframes disagreeing)
            timeframe_conflicts = await self._detect_timeframe_conflicts(symbol, signals)
            conflicts.extend(timeframe_conflicts)
            
        except Exception as e:
            logger.error(f"Error detecting conflicts for {symbol}: {e}")
        
        return conflicts

    async def _detect_directional_conflicts(self, symbol: str, 
                                          signals: List[TradingSignal]) -> List[SignalConflict]:
        """Detect directional conflicts (Buy vs Sell)"""
        conflicts = []
        
        # Group by direction
        buy_signals = [s for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]]
        
        # Check for conflicts
        if buy_signals and sell_signals:
            # Calculate conflict severity
            buy_strength = sum(s.strength * s.confidence for s in buy_signals) / len(buy_signals)
            sell_strength = sum(s.strength * s.confidence for s in sell_signals) / len(sell_signals)
            
            severity = min(buy_strength, sell_strength) / max(buy_strength, sell_strength)
            
            if severity > self.conflict_threshold:
                conflict = SignalConflict(
                    conflict_id=f"directional_{symbol}_{datetime.now().timestamp()}",
                    conflict_type=ConflictType.DIRECTIONAL,
                    conflicting_signals=buy_signals + sell_signals,
                    severity=severity,
                    detected_at=datetime.now()
                )
                conflicts.append(conflict)
        
        return conflicts

    async def _detect_strength_conflicts(self, symbol: str, 
                                       signals: List[TradingSignal]) -> List[SignalConflict]:
        """Detect strength conflicts (Strong vs Weak signals of same direction)"""
        conflicts = []
        
        # Group by direction and check for strength conflicts
        for direction in [SignalType.BUY, SignalType.SELL]:
            direction_signals = [s for s in signals if s.signal_type.value.endswith(direction.value)]
            
            if len(direction_signals) > 1:
                strengths = [s.strength for s in direction_signals]
                strength_variance = np.var(strengths)
                
                if strength_variance > 0.2:  # Significant variance in strength
                    conflict = SignalConflict(
                        conflict_id=f"strength_{symbol}_{direction.value}_{datetime.now().timestamp()}",
                        conflict_type=ConflictType.STRENGTH,
                        conflicting_signals=direction_signals,
                        severity=min(strength_variance, 1.0),
                        detected_at=datetime.now()
                    )
                    conflicts.append(conflict)
        
        return conflicts

    async def _detect_temporal_conflicts(self, symbol: str, 
                                       signals: List[TradingSignal]) -> List[SignalConflict]:
        """Detect temporal conflicts (Recent vs Old signals)"""
        conflicts = []
        
        if len(signals) < 2:
            return conflicts
        
        # Sort by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)
        
        # Check for significant time gaps with different signals
        for i in range(len(sorted_signals) - 1):
            current_signal = sorted_signals[i]
            next_signal = sorted_signals[i + 1]
            
            time_gap = (next_signal.timestamp - current_signal.timestamp).total_seconds()
            
            # If signals are far apart in time but conflicting
            if (time_gap > 300 and  # 5 minutes apart
                current_signal.signal_type != next_signal.signal_type and
                current_signal.signal_type != SignalType.HOLD and
                next_signal.signal_type != SignalType.HOLD):
                
                severity = min(time_gap / 3600, 1.0)  # Normalize to hours
                
                conflict = SignalConflict(
                    conflict_id=f"temporal_{symbol}_{datetime.now().timestamp()}",
                    conflict_type=ConflictType.TEMPORAL,
                    conflicting_signals=[current_signal, next_signal],
                    severity=severity,
                    detected_at=datetime.now()
                )
                conflicts.append(conflict)
        
        return conflicts

    async def _detect_source_conflicts(self, symbol: str, 
                                     signals: List[TradingSignal]) -> List[SignalConflict]:
        """Detect source conflicts (Different sources disagreeing)"""
        conflicts = []
        
        # Group by source
        source_signals = defaultdict(list)
        for signal in signals:
            source_signals[signal.source].append(signal)
        
        # Check for conflicts between sources
        sources = list(source_signals.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1_signals = source_signals[sources[i]]
                source2_signals = source_signals[sources[j]]
                
                # Check if sources have conflicting signals
                source1_direction = self._get_dominant_direction(source1_signals)
                source2_direction = self._get_dominant_direction(source2_signals)
                
                if (source1_direction != source2_direction and
                    source1_direction != SignalType.HOLD and
                    source2_direction != SignalType.HOLD):
                    
                    severity = self._calculate_source_conflict_severity(
                        source1_signals, source2_signals
                    )
                    
                    if severity > self.conflict_threshold:
                        conflict = SignalConflict(
                            conflict_id=f"source_{symbol}_{sources[i].value}_{sources[j].value}_{datetime.now().timestamp()}",
                            conflict_type=ConflictType.SOURCE,
                            conflicting_signals=source1_signals + source2_signals,
                            severity=severity,
                            detected_at=datetime.now()
                        )
                        conflicts.append(conflict)
        
        return conflicts

    async def _detect_timeframe_conflicts(self, symbol: str, 
                                        signals: List[TradingSignal]) -> List[SignalConflict]:
        """Detect timeframe conflicts (Different timeframes disagreeing)"""
        conflicts = []
        
        # Group by timeframe
        timeframe_signals = defaultdict(list)
        for signal in signals:
            timeframe_signals[signal.timeframe].append(signal)
        
        # Check for conflicts between timeframes
        timeframes = list(timeframe_signals.keys())
        for i in range(len(timeframes)):
            for j in range(i + 1, len(timeframes)):
                tf1_signals = timeframe_signals[timeframes[i]]
                tf2_signals = timeframe_signals[timeframes[j]]
                
                # Check if timeframes have conflicting signals
                tf1_direction = self._get_dominant_direction(tf1_signals)
                tf2_direction = self._get_dominant_direction(tf2_signals)
                
                if (tf1_direction != tf2_direction and
                    tf1_direction != SignalType.HOLD and
                    tf2_direction != SignalType.HOLD):
                    
                    severity = self._calculate_timeframe_conflict_severity(
                        tf1_signals, tf2_signals, timeframes[i], timeframes[j]
                    )
                    
                    if severity > self.conflict_threshold:
                        conflict = SignalConflict(
                            conflict_id=f"timeframe_{symbol}_{timeframes[i].value}_{timeframes[j].value}_{datetime.now().timestamp()}",
                            conflict_type=ConflictType.TIMEFRAME,
                            conflicting_signals=tf1_signals + tf2_signals,
                            severity=severity,
                            detected_at=datetime.now()
                        )
                        conflicts.append(conflict)
        
        return conflicts

    def _get_dominant_direction(self, signals: List[TradingSignal]) -> SignalType:
        """Get dominant signal direction from a list of signals"""
        if not signals:
            return SignalType.HOLD
        
        # Weight signals by strength and confidence
        signal_scores = defaultdict(float)
        for signal in signals:
            weight = signal.strength * signal.confidence
            signal_scores[signal.signal_type] += weight
        
        if signal_scores:
            return max(signal_scores.items(), key=lambda x: x[1])[0]
        else:
            return SignalType.HOLD

    def _calculate_source_conflict_severity(self, signals1: List[TradingSignal], 
                                          signals2: List[TradingSignal]) -> float:
        """Calculate severity of conflict between two sources"""
        # Calculate average strength and confidence for each source
        avg_strength1 = np.mean([s.strength for s in signals1])
        avg_confidence1 = np.mean([s.confidence for s in signals1])
        
        avg_strength2 = np.mean([s.strength for s in signals2])
        avg_confidence2 = np.mean([s.confidence for s in signals2])
        
        # Severity based on how strong both conflicting signals are
        severity = min(avg_strength1 * avg_confidence1, avg_strength2 * avg_confidence2)
        
        return severity

    def _calculate_timeframe_conflict_severity(self, signals1: List[TradingSignal], 
                                             signals2: List[TradingSignal],
                                             tf1: Timeframe, tf2: Timeframe) -> float:
        """Calculate severity of conflict between two timeframes"""
        # Base severity on signal strength
        base_severity = self._calculate_source_conflict_severity(signals1, signals2)
        
        # Adjust based on timeframe importance difference
        tf1_weight = self.weights.timeframe_weights.get(tf1, 0.1)
        tf2_weight = self.weights.timeframe_weights.get(tf2, 0.1)
        
        weight_factor = min(tf1_weight, tf2_weight) / max(tf1_weight, tf2_weight)
        
        return base_severity * weight_factor

    async def _choose_optimal_strategy(self, conflict: SignalConflict) -> ResolutionStrategy:
        """Choose optimal resolution strategy based on conflict type and history"""
        if self.default_strategy == ResolutionStrategy.ADAPTIVE:
            # Choose strategy based on conflict type and success rates
            strategies_by_type = {
                ConflictType.DIRECTIONAL: [ResolutionStrategy.WEIGHTED_AVERAGE, ResolutionStrategy.HIGHEST_CONFIDENCE],
                ConflictType.STRENGTH: [ResolutionStrategy.HIGHEST_CONFIDENCE, ResolutionStrategy.WEIGHTED_AVERAGE],
                ConflictType.TEMPORAL: [ResolutionStrategy.WEIGHTED_AVERAGE, ResolutionStrategy.SOURCE_PRIORITY],
                ConflictType.SOURCE: [ResolutionStrategy.SOURCE_PRIORITY, ResolutionStrategy.WEIGHTED_AVERAGE],
                ConflictType.TIMEFRAME: [ResolutionStrategy.TIMEFRAME_PRIORITY, ResolutionStrategy.WEIGHTED_AVERAGE]
            }
            
            candidate_strategies = strategies_by_type.get(conflict.conflict_type, 
                                                        [ResolutionStrategy.WEIGHTED_AVERAGE])
            
            # Choose strategy with highest success rate
            best_strategy = max(candidate_strategies, 
                              key=lambda s: self.strategy_success_rates[s])
            
            return best_strategy
        else:
            return self.default_strategy

    async def _apply_resolution_strategy(self, conflict: SignalConflict, 
                                       strategy: ResolutionStrategy) -> Optional[TradingSignal]:
        """Apply specific resolution strategy to resolve conflict"""
        try:
            signals = conflict.conflicting_signals
            
            if strategy == ResolutionStrategy.WEIGHTED_AVERAGE:
                return await self._weighted_average_resolution(signals)
            
            elif strategy == ResolutionStrategy.HIGHEST_CONFIDENCE:
                return await self._highest_confidence_resolution(signals)
            
            elif strategy == ResolutionStrategy.MAJORITY_VOTE:
                return await self._majority_vote_resolution(signals)
            
            elif strategy == ResolutionStrategy.SOURCE_PRIORITY:
                return await self._source_priority_resolution(signals)
            
            elif strategy == ResolutionStrategy.TIMEFRAME_PRIORITY:
                return await self._timeframe_priority_resolution(signals)
            
            else:
                # Default to weighted average
                return await self._weighted_average_resolution(signals)
                
        except Exception as e:
            logger.error(f"Error applying resolution strategy {strategy.value}: {e}")
            return None

    async def _weighted_average_resolution(self, signals: List[TradingSignal]) -> TradingSignal:
        """Resolve conflict using weighted average of signals"""
        # Calculate weighted scores for each signal type
        signal_scores = defaultdict(float)
        total_weight = 0.0
        
        for signal in signals:
            # Calculate weight based on multiple factors
            source_weight = self.weights.source_weights.get(signal.source, 0.1)
            timeframe_weight = self.weights.timeframe_weights.get(signal.timeframe, 0.1)
            recency_weight = self._calculate_recency_weight(signal.timestamp)
            reliability_weight = self.source_reliability[signal.source]
            
            combined_weight = (source_weight * timeframe_weight * recency_weight * 
                             reliability_weight * signal.strength * signal.confidence)
            
            signal_scores[signal.signal_type] += combined_weight
            total_weight += combined_weight
        
        # Find signal type with highest weighted score
        if signal_scores:
            resolved_signal_type = max(signal_scores.items(), key=lambda x: x[1])[0]
            
            # Calculate average properties for resolved signal
            relevant_signals = [s for s in signals if s.signal_type == resolved_signal_type]
            
            avg_strength = np.mean([s.strength for s in relevant_signals])
            avg_confidence = np.mean([s.confidence for s in relevant_signals])
            
            # Create resolved signal
            resolved_signal = TradingSignal(
                symbol=signals[0].symbol,
                signal_type=resolved_signal_type,
                source=SignalSource.TECHNICAL,  # Composite source
                timeframe=Timeframe.M15,  # Default timeframe
                strength=avg_strength,
                confidence=avg_confidence,
                timestamp=datetime.now(),
                metadata={'resolution_method': 'weighted_average', 'source_signals': len(signals)}
            )
            
            return resolved_signal
        
        return signals[0]  # Fallback

    async def _highest_confidence_resolution(self, signals: List[TradingSignal]) -> TradingSignal:
        """Resolve conflict by choosing signal with highest confidence"""
        return max(signals, key=lambda s: s.confidence)

    async def _majority_vote_resolution(self, signals: List[TradingSignal]) -> TradingSignal:
        """Resolve conflict using majority vote"""
        signal_counts = defaultdict(int)
        for signal in signals:
            signal_counts[signal.signal_type] += 1
        
        majority_signal_type = max(signal_counts.items(), key=lambda x: x[1])[0]
        
        # Return first signal of majority type
        for signal in signals:
            if signal.signal_type == majority_signal_type:
                return signal
        
        return signals[0]  # Fallback

    async def _source_priority_resolution(self, signals: List[TradingSignal]) -> TradingSignal:
        """Resolve conflict based on source priority"""
        # Sort by source weight (highest first)
        sorted_signals = sorted(signals, 
                              key=lambda s: self.weights.source_weights.get(s.source, 0.0),
                              reverse=True)
        
        return sorted_signals[0]

    async def _timeframe_priority_resolution(self, signals: List[TradingSignal]) -> TradingSignal:
        """Resolve conflict based on timeframe priority"""
        # Sort by timeframe weight (highest first)
        sorted_signals = sorted(signals,
                              key=lambda s: self.weights.timeframe_weights.get(s.timeframe, 0.0),
                              reverse=True)
        
        return sorted_signals[0]

    def _calculate_recency_weight(self, timestamp: datetime) -> float:
        """Calculate weight based on signal recency"""
        age_seconds = (datetime.now() - timestamp).total_seconds()
        age_minutes = age_seconds / 60.0
        
        # Exponential decay
        return self.weights.recency_decay ** age_minutes

    async def _calculate_resolution_confidence(self, conflict: SignalConflict, 
                                             resolved_signal: TradingSignal) -> float:
        """Calculate confidence in the resolution"""
        # Base confidence on resolved signal's confidence
        base_confidence = resolved_signal.confidence
        
        # Adjust based on conflict severity (lower severity = higher confidence)
        severity_factor = 1.0 - conflict.severity
        
        # Adjust based on number of supporting signals
        supporting_signals = [s for s in conflict.conflicting_signals 
                            if s.signal_type == resolved_signal.signal_type]
        support_factor = len(supporting_signals) / len(conflict.conflicting_signals)
        
        # Combine factors
        resolution_confidence = base_confidence * severity_factor * support_factor
        
        return min(max(resolution_confidence, 0.0), 1.0)

    async def _cache_resolution(self, conflict: SignalConflict):
        """Cache conflict resolution in Redis"""
        try:
            resolution_data = {
                'conflict_id': conflict.conflict_id,
                'conflict_type': conflict.conflict_type.value,
                'severity': conflict.severity,
                'resolution_strategy': conflict.resolution_strategy.value if conflict.resolution_strategy else None,
                'resolved_signal': {
                    'signal_type': conflict.resolved_signal.signal_type.value,
                    'strength': conflict.resolved_signal.strength,
                    'confidence': conflict.resolved_signal.confidence
                } if conflict.resolved_signal else None,
                'resolution_confidence': conflict.resolution_confidence,
                'timestamp': conflict.detected_at.isoformat()
            }
            
            await self.redis_client.setex(
                f"conflict_resolution:{conflict.conflict_id}",
                3600,  # 1 hour TTL
                json.dumps(resolution_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching resolution: {e}")

    async def update_strategy_performance(self, strategy: ResolutionStrategy, 
                                        success: bool):
        """Update strategy performance based on feedback"""
        current_rate = self.strategy_success_rates[strategy]
        
        if success:
            new_rate = current_rate + self.learning_rate * (1.0 - current_rate)
        else:
            new_rate = current_rate - self.learning_rate * current_rate
        
        self.strategy_success_rates[strategy] = max(min(new_rate, 1.0), 0.0)
        
        # Update strategy performance stats
        self.resolution_stats['strategy_performance'][strategy]['count'] += 1
        if success:
            old_accuracy = self.resolution_stats['strategy_performance'][strategy]['accuracy']
            count = self.resolution_stats['strategy_performance'][strategy]['count']
            new_accuracy = (old_accuracy * (count - 1) + 1.0) / count
            self.resolution_stats['strategy_performance'][strategy]['accuracy'] = new_accuracy

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get conflict resolution performance statistics"""
        return {
            'resolution_stats': dict(self.resolution_stats),
            'strategy_success_rates': dict(self.strategy_success_rates),
            'source_reliability': dict(self.source_reliability),
            'active_conflicts': len(self.active_conflicts),
            'conflict_history_size': len(self.conflict_history),
            'weights': {
                'source_weights': {src.value: weight for src, weight in self.weights.source_weights.items()},
                'timeframe_weights': {tf.value: weight for tf, weight in self.weights.timeframe_weights.items()},
                'recency_decay': self.weights.recency_decay
            },
            'configuration': {
                'default_strategy': self.default_strategy.value,
                'conflict_threshold': self.conflict_threshold,
                'learning_rate': self.learning_rate
            }
        }

    def update_weights(self, new_weights: ResolutionWeights):
        """Update resolution weights"""
        self.weights = new_weights
        logger.info("Conflict resolution weights updated")

    def clear_conflicts(self):
        """Clear all active conflicts"""
        self.active_conflicts.clear()
        logger.info("Active conflicts cleared")
