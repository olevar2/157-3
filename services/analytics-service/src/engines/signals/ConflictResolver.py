"""
Conflict Resolution Engine
Resolves conflicting signals from different timeframes and sources.
Implements sophisticated logic to handle signal disagreements.
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum

from .SignalAggregator import SignalInput, Timeframe


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving signal conflicts"""
    TIMEFRAME_HIERARCHY = "timeframe_hierarchy"  # Higher timeframes win
    STRENGTH_WEIGHTED = "strength_weighted"      # Stronger signals win
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # More confident signals win
    MAJORITY_VOTE = "majority_vote"             # Most common signal wins
    HYBRID = "hybrid"                           # Combination of strategies


@dataclass
class ConflictAnalysis:
    """Analysis of signal conflicts"""
    total_signals: int
    conflicting_signals: int
    agreement_percentage: float
    dominant_signal: str
    conflict_severity: str  # 'low', 'medium', 'high'
    timeframe_conflicts: Dict[str, List[str]]
    source_conflicts: Dict[str, List[str]]


@dataclass
class ResolvedSignal:
    """Result of conflict resolution"""
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-100
    confidence: float  # 0-1
    resolution_method: str
    conflict_analysis: ConflictAnalysis
    contributing_signals: List[SignalInput]
    discarded_signals: List[SignalInput]
    resolution_confidence: float  # How confident we are in the resolution


class ConflictResolver:
    """
    Conflict Resolution Engine
    Resolves disagreements between signals from different timeframes and sources
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default resolution strategy
        self.default_strategy = ConflictResolutionStrategy(
            self.config.get('default_strategy', 'hybrid')
        )
        
        # Timeframe hierarchy weights (higher = more important)
        self.timeframe_hierarchy = {
            Timeframe.H4: 6,
            Timeframe.H1: 5,
            Timeframe.M30: 4,
            Timeframe.M15: 3,
            Timeframe.M5: 2,
            Timeframe.M1: 1
        }
        
        # Source reliability weights
        self.source_weights = {
            'trend': 0.3,      # Trend signals are most reliable
            'momentum': 0.25,  # Momentum is important for timing
            'volume': 0.2,     # Volume confirms moves
            'pattern': 0.15,   # Patterns provide context
            'volatility': 0.1  # Volatility is supplementary
        }
        
        # Conflict thresholds
        self.conflict_thresholds = {
            'low': 0.2,     # < 20% conflicting signals
            'medium': 0.5,  # 20-50% conflicting signals
            'high': 0.8     # > 50% conflicting signals
        }
        
        # Performance tracking
        self.resolution_count = 0
        self.total_resolution_time = 0.0
        
    async def resolve_conflicts(self, signals: List[SignalInput]) -> ResolvedSignal:
        """
        Main conflict resolution method
        Analyzes conflicts and returns a resolved signal
        """
        start_time = time.time()
        
        try:
            if not signals:
                raise ValueError("No signals provided for conflict resolution")
            
            # Analyze conflicts
            conflict_analysis = await self._analyze_conflicts(signals)
            
            # Choose resolution strategy based on conflict severity
            strategy = await self._choose_resolution_strategy(conflict_analysis)
            
            # Resolve conflicts using chosen strategy
            resolved_signal = await self._resolve_using_strategy(signals, strategy, conflict_analysis)
            
            # Update performance tracking
            resolution_time = time.time() - start_time
            self.resolution_count += 1
            self.total_resolution_time += resolution_time
            
            self.logger.debug(f"Resolved conflicts for {len(signals)} signals using {strategy.value}")
            
            return resolved_signal
            
        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            raise
    
    async def _analyze_conflicts(self, signals: List[SignalInput]) -> ConflictAnalysis:
        """Analyze the nature and severity of signal conflicts"""
        
        # Count signal types
        signal_counts = Counter(signal.signal_type for signal in signals)
        total_signals = len(signals)
        
        # Find dominant signal
        dominant_signal = signal_counts.most_common(1)[0][0]
        dominant_count = signal_counts[dominant_signal]
        
        # Calculate agreement percentage
        agreement_percentage = dominant_count / total_signals
        
        # Count conflicting signals
        conflicting_signals = total_signals - dominant_count
        
        # Determine conflict severity
        conflict_ratio = conflicting_signals / total_signals
        if conflict_ratio <= self.conflict_thresholds['low']:
            conflict_severity = 'low'
        elif conflict_ratio <= self.conflict_thresholds['medium']:
            conflict_severity = 'medium'
        else:
            conflict_severity = 'high'
        
        # Analyze timeframe conflicts
        timeframe_conflicts = defaultdict(list)
        timeframe_signals = defaultdict(list)
        
        for signal in signals:
            timeframe_signals[signal.timeframe.value].append(signal.signal_type)
        
        for timeframe, tf_signals in timeframe_signals.items():
            unique_signals = set(tf_signals)
            if len(unique_signals) > 1:
                timeframe_conflicts[timeframe] = list(unique_signals)
        
        # Analyze source conflicts
        source_conflicts = defaultdict(list)
        source_signals = defaultdict(list)
        
        for signal in signals:
            source_signals[signal.source].append(signal.signal_type)
        
        for source, src_signals in source_signals.items():
            unique_signals = set(src_signals)
            if len(unique_signals) > 1:
                source_conflicts[source] = list(unique_signals)
        
        return ConflictAnalysis(
            total_signals=total_signals,
            conflicting_signals=conflicting_signals,
            agreement_percentage=agreement_percentage,
            dominant_signal=dominant_signal,
            conflict_severity=conflict_severity,
            timeframe_conflicts=dict(timeframe_conflicts),
            source_conflicts=dict(source_conflicts)
        )
    
    async def _choose_resolution_strategy(self, conflict_analysis: ConflictAnalysis) -> ConflictResolutionStrategy:
        """Choose the best resolution strategy based on conflict analysis"""
        
        # For low conflicts, use majority vote
        if conflict_analysis.conflict_severity == 'low':
            return ConflictResolutionStrategy.MAJORITY_VOTE
        
        # For medium conflicts, use timeframe hierarchy
        elif conflict_analysis.conflict_severity == 'medium':
            return ConflictResolutionStrategy.TIMEFRAME_HIERARCHY
        
        # For high conflicts, use hybrid approach
        else:
            return ConflictResolutionStrategy.HYBRID
    
    async def _resolve_using_strategy(
        self, 
        signals: List[SignalInput], 
        strategy: ConflictResolutionStrategy,
        conflict_analysis: ConflictAnalysis
    ) -> ResolvedSignal:
        """Resolve conflicts using the specified strategy"""
        
        if strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            return await self._resolve_majority_vote(signals, conflict_analysis)
        
        elif strategy == ConflictResolutionStrategy.TIMEFRAME_HIERARCHY:
            return await self._resolve_timeframe_hierarchy(signals, conflict_analysis)
        
        elif strategy == ConflictResolutionStrategy.STRENGTH_WEIGHTED:
            return await self._resolve_strength_weighted(signals, conflict_analysis)
        
        elif strategy == ConflictResolutionStrategy.CONFIDENCE_WEIGHTED:
            return await self._resolve_confidence_weighted(signals, conflict_analysis)
        
        elif strategy == ConflictResolutionStrategy.HYBRID:
            return await self._resolve_hybrid(signals, conflict_analysis)
        
        else:
            # Fallback to majority vote
            return await self._resolve_majority_vote(signals, conflict_analysis)
    
    async def _resolve_majority_vote(self, signals: List[SignalInput], conflict_analysis: ConflictAnalysis) -> ResolvedSignal:
        """Resolve using simple majority vote"""
        
        # Use the dominant signal from analysis
        winning_signal_type = conflict_analysis.dominant_signal
        
        # Find all signals that match the winning type
        contributing_signals = [s for s in signals if s.signal_type == winning_signal_type]
        discarded_signals = [s for s in signals if s.signal_type != winning_signal_type]
        
        # Calculate average strength and confidence
        avg_strength = np.mean([s.strength for s in contributing_signals])
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        
        # Resolution confidence based on agreement percentage
        resolution_confidence = conflict_analysis.agreement_percentage
        
        return ResolvedSignal(
            signal_type=winning_signal_type,
            strength=avg_strength,
            confidence=avg_confidence,
            resolution_method='majority_vote',
            conflict_analysis=conflict_analysis,
            contributing_signals=contributing_signals,
            discarded_signals=discarded_signals,
            resolution_confidence=resolution_confidence
        )
    
    async def _resolve_timeframe_hierarchy(self, signals: List[SignalInput], conflict_analysis: ConflictAnalysis) -> ResolvedSignal:
        """Resolve using timeframe hierarchy (higher timeframes win)"""
        
        # Group signals by timeframe
        timeframe_groups = defaultdict(list)
        for signal in signals:
            timeframe_groups[signal.timeframe].append(signal)
        
        # Find the highest timeframe with signals
        highest_timeframe = max(timeframe_groups.keys(), key=lambda tf: self.timeframe_hierarchy[tf])
        highest_tf_signals = timeframe_groups[highest_timeframe]
        
        # If multiple signals in highest timeframe, use majority vote within that timeframe
        if len(highest_tf_signals) == 1:
            winning_signal = highest_tf_signals[0]
            contributing_signals = [winning_signal]
        else:
            signal_counts = Counter(s.signal_type for s in highest_tf_signals)
            winning_signal_type = signal_counts.most_common(1)[0][0]
            contributing_signals = [s for s in highest_tf_signals if s.signal_type == winning_signal_type]
            winning_signal = contributing_signals[0]  # Use first as representative
        
        discarded_signals = [s for s in signals if s not in contributing_signals]
        
        # Calculate weighted strength and confidence
        avg_strength = np.mean([s.strength for s in contributing_signals])
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        
        # Higher resolution confidence for timeframe hierarchy
        resolution_confidence = 0.8 + (self.timeframe_hierarchy[highest_timeframe] / 10)
        resolution_confidence = min(resolution_confidence, 1.0)
        
        return ResolvedSignal(
            signal_type=winning_signal.signal_type,
            strength=avg_strength,
            confidence=avg_confidence,
            resolution_method='timeframe_hierarchy',
            conflict_analysis=conflict_analysis,
            contributing_signals=contributing_signals,
            discarded_signals=discarded_signals,
            resolution_confidence=resolution_confidence
        )
    
    async def _resolve_strength_weighted(self, signals: List[SignalInput], conflict_analysis: ConflictAnalysis) -> ResolvedSignal:
        """Resolve using strength-weighted voting"""
        
        # Calculate weighted votes for each signal type
        signal_weights = defaultdict(float)
        signal_groups = defaultdict(list)
        
        for signal in signals:
            weight = signal.strength / 100.0  # Normalize to 0-1
            signal_weights[signal.signal_type] += weight
            signal_groups[signal.signal_type].append(signal)
        
        # Find winning signal type
        winning_signal_type = max(signal_weights.items(), key=lambda x: x[1])[0]
        
        contributing_signals = signal_groups[winning_signal_type]
        discarded_signals = [s for s in signals if s.signal_type != winning_signal_type]
        
        # Calculate weighted averages
        total_weight = sum(s.strength for s in contributing_signals)
        avg_strength = np.mean([s.strength for s in contributing_signals])
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        
        # Resolution confidence based on weight dominance
        total_possible_weight = sum(s.strength for s in signals)
        resolution_confidence = signal_weights[winning_signal_type] / (total_possible_weight / 100.0)
        
        return ResolvedSignal(
            signal_type=winning_signal_type,
            strength=avg_strength,
            confidence=avg_confidence,
            resolution_method='strength_weighted',
            conflict_analysis=conflict_analysis,
            contributing_signals=contributing_signals,
            discarded_signals=discarded_signals,
            resolution_confidence=min(resolution_confidence, 1.0)
        )
    
    async def _resolve_confidence_weighted(self, signals: List[SignalInput], conflict_analysis: ConflictAnalysis) -> ResolvedSignal:
        """Resolve using confidence-weighted voting"""
        
        # Calculate weighted votes for each signal type
        signal_weights = defaultdict(float)
        signal_groups = defaultdict(list)
        
        for signal in signals:
            weight = signal.confidence
            signal_weights[signal.signal_type] += weight
            signal_groups[signal.signal_type].append(signal)
        
        # Find winning signal type
        winning_signal_type = max(signal_weights.items(), key=lambda x: x[1])[0]
        
        contributing_signals = signal_groups[winning_signal_type]
        discarded_signals = [s for s in signals if s.signal_type != winning_signal_type]
        
        # Calculate weighted averages
        avg_strength = np.mean([s.strength for s in contributing_signals])
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        
        # Resolution confidence based on confidence dominance
        total_confidence = sum(s.confidence for s in signals)
        resolution_confidence = signal_weights[winning_signal_type] / total_confidence
        
        return ResolvedSignal(
            signal_type=winning_signal_type,
            strength=avg_strength,
            confidence=avg_confidence,
            resolution_method='confidence_weighted',
            conflict_analysis=conflict_analysis,
            contributing_signals=contributing_signals,
            discarded_signals=discarded_signals,
            resolution_confidence=min(resolution_confidence, 1.0)
        )
    
    async def _resolve_hybrid(self, signals: List[SignalInput], conflict_analysis: ConflictAnalysis) -> ResolvedSignal:
        """Resolve using hybrid approach combining multiple strategies"""
        
        # Calculate composite scores for each signal
        signal_scores = defaultdict(float)
        signal_groups = defaultdict(list)
        
        for signal in signals:
            # Timeframe weight
            tf_weight = self.timeframe_hierarchy[signal.timeframe] / 6.0  # Normalize to 0-1
            
            # Source weight
            src_weight = self.source_weights.get(signal.source, 0.1)
            
            # Strength weight
            strength_weight = signal.strength / 100.0
            
            # Confidence weight
            confidence_weight = signal.confidence
            
            # Composite score
            composite_score = (tf_weight * 0.3 + src_weight * 0.2 + 
                             strength_weight * 0.25 + confidence_weight * 0.25)
            
            signal_scores[signal.signal_type] += composite_score
            signal_groups[signal.signal_type].append(signal)
        
        # Find winning signal type
        winning_signal_type = max(signal_scores.items(), key=lambda x: x[1])[0]
        
        contributing_signals = signal_groups[winning_signal_type]
        discarded_signals = [s for s in signals if s.signal_type != winning_signal_type]
        
        # Calculate weighted averages
        avg_strength = np.mean([s.strength for s in contributing_signals])
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        
        # Resolution confidence based on score dominance
        total_score = sum(signal_scores.values())
        resolution_confidence = signal_scores[winning_signal_type] / total_score
        
        return ResolvedSignal(
            signal_type=winning_signal_type,
            strength=avg_strength,
            confidence=avg_confidence,
            resolution_method='hybrid',
            conflict_analysis=conflict_analysis,
            contributing_signals=contributing_signals,
            discarded_signals=discarded_signals,
            resolution_confidence=min(resolution_confidence, 1.0)
        )
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get conflict resolution performance metrics"""
        return {
            'total_resolutions': self.resolution_count,
            'average_resolution_time_ms': (self.total_resolution_time / self.resolution_count * 1000) 
                                        if self.resolution_count > 0 else 0
        }
