"""
Fast Signal Aggregation Engine
Multi-timeframe signal combination for short-term trading (M1-H4).
Aggregates signals from different engines and timeframes for enhanced accuracy.
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

from .TimeframeSynchronizer import TimeframeSynchronizer, SynchronizedSignals
from .ConflictResolver import ConflictResolver, ResolvedSignal
from .ConfidenceCalculator import ConfidenceCalculator, ConfidenceMetrics
from .QuickDecisionMatrix import QuickDecisionMatrix, TradingDecision


class Timeframe(Enum):
    """Supported timeframes for signal aggregation"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"


@dataclass
class SignalInput:
    """Input signal from any engine"""
    source: str  # Engine name (momentum, volume, pattern, etc.)
    timeframe: Timeframe
    symbol: str
    timestamp: float
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-100
    confidence: float  # 0-1
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AggregatedSignal:
    """Final aggregated signal result"""
    symbol: str
    timestamp: float
    signal_type: str  # 'buy', 'sell', 'hold'
    overall_strength: float  # 0-100
    overall_confidence: float  # 0-1
    timeframe_consensus: Dict[str, str]  # Timeframe -> signal_type
    source_consensus: Dict[str, str]  # Source -> signal_type
    confluence_score: float  # 0-1 (higher = more timeframes agree)
    risk_reward_ratio: float
    entry_price: float
    stop_loss: float
    take_profit: float
    supporting_signals: List[SignalInput]
    conflicting_signals: List[SignalInput]
    confidence_metrics: ConfidenceMetrics
    execution_priority: str  # 'high', 'medium', 'low'


class SignalAggregator:
    """
    Fast Signal Aggregation Engine
    Combines signals from multiple timeframes and sources for enhanced trading decisions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-components
        self.timeframe_synchronizer = TimeframeSynchronizer()
        self.conflict_resolver = ConflictResolver()
        self.confidence_calculator = ConfidenceCalculator()
        self.decision_matrix = QuickDecisionMatrix()
        
        # Signal storage
        self.signal_buffer = defaultdict(lambda: defaultdict(deque))  # symbol -> timeframe -> signals
        self.max_buffer_size = self.config.get('max_buffer_size', 100)
        
        # Timeframe weights for aggregation
        self.timeframe_weights = {
            Timeframe.M1: 0.1,   # Lower weight for noise
            Timeframe.M5: 0.15,
            Timeframe.M15: 0.2,
            Timeframe.M30: 0.25,
            Timeframe.H1: 0.3,   # Higher weight for trend
            Timeframe.H4: 0.35   # Highest weight for major trend
        }
        
        # Source weights
        self.source_weights = {
            'momentum': 0.25,
            'volume': 0.2,
            'pattern': 0.2,
            'trend': 0.25,
            'volatility': 0.1
        }
        
        # Performance tracking
        self.aggregation_count = 0
        self.total_aggregation_time = 0.0
        
    async def add_signal(self, signal: SignalInput) -> None:
        """Add a new signal to the aggregation buffer"""
        try:
            # Add to buffer
            buffer = self.signal_buffer[signal.symbol][signal.timeframe]
            buffer.append(signal)
            
            # Maintain buffer size
            if len(buffer) > self.max_buffer_size:
                buffer.popleft()
                
            self.logger.debug(f"Added {signal.source} signal for {signal.symbol} {signal.timeframe.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to add signal: {e}")
            raise
    
    async def aggregate_signals(self, symbol: str, max_age_minutes: int = 5) -> Optional[AggregatedSignal]:
        """
        Aggregate all recent signals for a symbol into a single trading decision
        """
        start_time = time.time()
        
        try:
            # Get recent signals
            recent_signals = await self._get_recent_signals(symbol, max_age_minutes)
            
            if not recent_signals:
                self.logger.debug(f"No recent signals found for {symbol}")
                return None
            
            # Synchronize signals across timeframes
            synchronized_signals = await self.timeframe_synchronizer.synchronize_signals(recent_signals)
            
            # Resolve conflicts between signals
            resolved_signal = await self.conflict_resolver.resolve_conflicts(synchronized_signals)
            
            # Calculate confidence metrics
            confidence_metrics = await self.confidence_calculator.calculate_confidence(
                recent_signals, resolved_signal
            )
            
            # Generate final trading decision
            trading_decision = await self.decision_matrix.make_decision(
                resolved_signal, confidence_metrics
            )
            
            # Build aggregated signal
            aggregated_signal = await self._build_aggregated_signal(
                symbol, recent_signals, resolved_signal, confidence_metrics, trading_decision
            )
            
            # Update performance tracking
            calculation_time = time.time() - start_time
            self.aggregation_count += 1
            self.total_aggregation_time += calculation_time
            
            self.logger.info(f"Aggregated {len(recent_signals)} signals for {symbol} in {calculation_time:.3f}s")
            
            return aggregated_signal
            
        except Exception as e:
            self.logger.error(f"Signal aggregation failed for {symbol}: {e}")
            raise
    
    async def _get_recent_signals(self, symbol: str, max_age_minutes: int) -> List[SignalInput]:
        """Get all recent signals for a symbol within the specified time window"""
        recent_signals = []
        cutoff_time = time.time() - (max_age_minutes * 60)
        
        if symbol not in self.signal_buffer:
            return recent_signals
        
        for timeframe, signal_buffer in self.signal_buffer[symbol].items():
            for signal in signal_buffer:
                if signal.timestamp >= cutoff_time:
                    recent_signals.append(signal)
        
        # Sort by timestamp (newest first)
        recent_signals.sort(key=lambda x: x.timestamp, reverse=True)
        
        return recent_signals
    
    async def _build_aggregated_signal(
        self, 
        symbol: str,
        recent_signals: List[SignalInput],
        resolved_signal: ResolvedSignal,
        confidence_metrics: ConfidenceMetrics,
        trading_decision: TradingDecision
    ) -> AggregatedSignal:
        """Build the final aggregated signal result"""
        
        # Calculate timeframe consensus
        timeframe_consensus = {}
        for signal in recent_signals:
            tf = signal.timeframe.value
            if tf not in timeframe_consensus:
                timeframe_consensus[tf] = signal.signal_type
        
        # Calculate source consensus
        source_consensus = {}
        for signal in recent_signals:
            source = signal.source
            if source not in source_consensus:
                source_consensus[source] = signal.signal_type
        
        # Calculate confluence score
        confluence_score = await self._calculate_confluence_score(recent_signals)
        
        # Separate supporting and conflicting signals
        supporting_signals = []
        conflicting_signals = []
        
        for signal in recent_signals:
            if signal.signal_type == resolved_signal.signal_type:
                supporting_signals.append(signal)
            else:
                conflicting_signals.append(signal)
        
        # Determine execution priority
        execution_priority = self._determine_execution_priority(
            confluence_score, confidence_metrics.overall_confidence
        )
        
        return AggregatedSignal(
            symbol=symbol,
            timestamp=time.time(),
            signal_type=resolved_signal.signal_type,
            overall_strength=resolved_signal.strength,
            overall_confidence=confidence_metrics.overall_confidence,
            timeframe_consensus=timeframe_consensus,
            source_consensus=source_consensus,
            confluence_score=confluence_score,
            risk_reward_ratio=trading_decision.risk_reward_ratio,
            entry_price=trading_decision.entry_price,
            stop_loss=trading_decision.stop_loss,
            take_profit=trading_decision.take_profit,
            supporting_signals=supporting_signals,
            conflicting_signals=conflicting_signals,
            confidence_metrics=confidence_metrics,
            execution_priority=execution_priority
        )
    
    async def _calculate_confluence_score(self, signals: List[SignalInput]) -> float:
        """Calculate how well signals agree across timeframes"""
        if not signals:
            return 0.0
        
        # Group signals by timeframe
        timeframe_signals = defaultdict(list)
        for signal in signals:
            timeframe_signals[signal.timeframe].append(signal)
        
        # Calculate agreement within each timeframe
        timeframe_agreements = {}
        for timeframe, tf_signals in timeframe_signals.items():
            if len(tf_signals) == 1:
                timeframe_agreements[timeframe] = tf_signals[0].signal_type
            else:
                # Find most common signal type
                signal_counts = defaultdict(int)
                for signal in tf_signals:
                    signal_counts[signal.signal_type] += 1
                
                most_common = max(signal_counts.items(), key=lambda x: x[1])
                timeframe_agreements[timeframe] = most_common[0]
        
        # Calculate overall confluence
        if len(timeframe_agreements) <= 1:
            return 1.0
        
        # Find most common signal across timeframes
        signal_counts = defaultdict(int)
        for signal_type in timeframe_agreements.values():
            signal_counts[signal_type] += 1
        
        most_common_count = max(signal_counts.values())
        total_timeframes = len(timeframe_agreements)
        
        return most_common_count / total_timeframes
    
    def _determine_execution_priority(self, confluence_score: float, confidence: float) -> str:
        """Determine execution priority based on confluence and confidence"""
        combined_score = (confluence_score + confidence) / 2
        
        if combined_score >= 0.8:
            return 'high'
        elif combined_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get aggregation performance metrics"""
        return {
            'total_aggregations': self.aggregation_count,
            'average_aggregation_time_ms': (self.total_aggregation_time / self.aggregation_count * 1000) 
                                         if self.aggregation_count > 0 else 0,
            'signals_in_buffer': sum(
                sum(len(tf_buffer) for tf_buffer in symbol_buffers.values())
                for symbol_buffers in self.signal_buffer.values()
            )
        }
    
    async def clear_old_signals(self, max_age_hours: int = 24) -> int:
        """Clear signals older than specified hours"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleared_count = 0
        
        for symbol_buffers in self.signal_buffer.values():
            for timeframe_buffer in symbol_buffers.values():
                original_length = len(timeframe_buffer)
                
                # Keep only recent signals
                while timeframe_buffer and timeframe_buffer[0].timestamp < cutoff_time:
                    timeframe_buffer.popleft()
                    cleared_count += 1
        
        self.logger.info(f"Cleared {cleared_count} old signals")
        return cleared_count
