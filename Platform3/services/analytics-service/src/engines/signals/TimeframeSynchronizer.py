"""
Platform3 Forex Trading Platform
Timeframe Synchronizer - Multi-Timeframe Signal Alignment

This module synchronizes signals across multiple timeframes to ensure
coherent trading decisions and optimal signal timing.

Author: Platform3 Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFrame(Enum):
    """Trading timeframes with minute values"""
    M1 = 1
    M5 = 5
    M15 = 15
    M30 = 30
    H1 = 60
    H4 = 240
    D1 = 1440

class SyncStatus(Enum):
    """Synchronization status"""
    SYNCHRONIZED = "SYNCHRONIZED"
    PARTIAL_SYNC = "PARTIAL_SYNC"
    DESYNCHRONIZED = "DESYNCHRONIZED"
    CONFLICTING = "CONFLICTING"

@dataclass
class TimeframedSignal:
    """Signal with timeframe information"""
    timeframe: TimeFrame
    signal_type: str
    strength: float
    timestamp: datetime
    price: float
    indicator: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SynchronizationResult:
    """Result of timeframe synchronization"""
    sync_status: SyncStatus
    synchronized_signals: List[TimeframedSignal]
    alignment_score: float  # 0.0 to 1.0
    dominant_timeframe: TimeFrame
    conflicting_signals: List[TimeframedSignal]
    sync_timestamp: datetime
    execution_window: Tuple[datetime, datetime]

class TimeframeSynchronizer:
    """
    Advanced timeframe synchronizer for multi-timeframe analysis
    
    Features:
    - Multi-timeframe signal alignment
    - Temporal synchronization across different timeframes
    - Conflict detection and resolution
    - Optimal execution timing calculation
    - Real-time synchronization monitoring
    """
    
    def __init__(self):
        """Initialize the timeframe synchronizer"""
        self.timeframe_hierarchy = {
            TimeFrame.D1: 6,
            TimeFrame.H4: 5,
            TimeFrame.H1: 4,
            TimeFrame.M30: 3,
            TimeFrame.M15: 2,
            TimeFrame.M5: 1,
            TimeFrame.M1: 0
        }
        
        self.sync_tolerances = {
            TimeFrame.M1: timedelta(seconds=30),
            TimeFrame.M5: timedelta(minutes=2),
            TimeFrame.M15: timedelta(minutes=5),
            TimeFrame.M30: timedelta(minutes=10),
            TimeFrame.H1: timedelta(minutes=20),
            TimeFrame.H4: timedelta(hours=1),
            TimeFrame.D1: timedelta(hours=4)
        }
        
        self.signal_buffer = defaultdict(list)
        self.sync_history = []
        
    async def synchronize_signals(
        self,
        signals: List[TimeframedSignal],
        reference_time: Optional[datetime] = None
    ) -> SynchronizationResult:
        """
        Synchronize signals across multiple timeframes
        
        Args:
            signals: List of signals from different timeframes
            reference_time: Reference time for synchronization
            
        Returns:
            SynchronizationResult with alignment analysis
        """
        try:
            if not signals:
                return self._create_empty_sync_result()
            
            reference_time = reference_time or datetime.now()
            
            # Group signals by timeframe
            timeframe_groups = self._group_by_timeframe(signals)
            
            # Align signals to common time grid
            aligned_signals = await self._align_to_time_grid(
                timeframe_groups, reference_time
            )
            
            # Calculate synchronization score
            alignment_score = self._calculate_alignment_score(aligned_signals)
            
            # Detect conflicts
            conflicting_signals = self._detect_conflicts(aligned_signals)
            
            # Determine synchronization status
            sync_status = self._determine_sync_status(
                aligned_signals, conflicting_signals, alignment_score
            )
            
            # Find dominant timeframe
            dominant_timeframe = self._find_dominant_timeframe(aligned_signals)
            
            # Calculate execution window
            execution_window = self._calculate_execution_window(
                aligned_signals, dominant_timeframe
            )
            
            # Create result
            result = SynchronizationResult(
                sync_status=sync_status,
                synchronized_signals=aligned_signals,
                alignment_score=alignment_score,
                dominant_timeframe=dominant_timeframe,
                conflicting_signals=conflicting_signals,
                sync_timestamp=reference_time,
                execution_window=execution_window
            )
            
            # Store for analysis
            self.sync_history.append(result)
            
            logger.info(f"Synchronization complete: {sync_status.value} - Score: {alignment_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error synchronizing signals: {e}")
            return self._create_empty_sync_result()
    
    def _group_by_timeframe(self, signals: List[TimeframedSignal]) -> Dict[TimeFrame, List[TimeframedSignal]]:
        """Group signals by their timeframe"""
        groups = defaultdict(list)
        for signal in signals:
            groups[signal.timeframe].append(signal)
        return dict(groups)
    
    async def _align_to_time_grid(
        self,
        timeframe_groups: Dict[TimeFrame, List[TimeframedSignal]],
        reference_time: datetime
    ) -> List[TimeframedSignal]:
        """Align signals to a common time grid"""
        aligned_signals = []
        
        for timeframe, signals in timeframe_groups.items():
            # Calculate timeframe boundaries
            timeframe_minutes = timeframe.value
            
            for signal in signals:
                # Align timestamp to timeframe boundary
                aligned_timestamp = self._align_timestamp_to_timeframe(
                    signal.timestamp, timeframe_minutes
                )
                
                # Check if signal is within sync tolerance
                time_diff = abs((aligned_timestamp - reference_time).total_seconds())
                tolerance = self.sync_tolerances[timeframe].total_seconds()
                
                if time_diff <= tolerance:
                    # Create aligned signal
                    aligned_signal = TimeframedSignal(
                        timeframe=signal.timeframe,
                        signal_type=signal.signal_type,
                        strength=signal.strength,
                        timestamp=aligned_timestamp,
                        price=signal.price,
                        indicator=signal.indicator,
                        metadata=signal.metadata
                    )
                    aligned_signals.append(aligned_signal)
        
        return aligned_signals
    
    def _align_timestamp_to_timeframe(self, timestamp: datetime, timeframe_minutes: int) -> datetime:
        """Align timestamp to timeframe boundary"""
        # Round down to nearest timeframe boundary
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
        aligned_minutes = (minutes_since_midnight // timeframe_minutes) * timeframe_minutes
        
        aligned_time = timestamp.replace(
            hour=aligned_minutes // 60,
            minute=aligned_minutes % 60,
            second=0,
            microsecond=0
        )
        
        return aligned_time
    
    def _calculate_alignment_score(self, signals: List[TimeframedSignal]) -> float:
        """Calculate how well signals are aligned"""
        if len(signals) < 2:
            return 0.0
        
        # Group by signal type
        signal_types = defaultdict(list)
        for signal in signals:
            signal_types[signal.signal_type].append(signal)
        
        # Calculate alignment for each signal type
        alignment_scores = []
        
        for signal_type, type_signals in signal_types.items():
            if len(type_signals) < 2:
                continue
            
            # Calculate time variance
            timestamps = [s.timestamp for s in type_signals]
            time_variance = np.var([(t - timestamps[0]).total_seconds() for t in timestamps])
            
            # Calculate strength consistency
            strengths = [s.strength for s in type_signals]
            strength_variance = np.var(strengths)
            
            # Combined alignment score
            time_score = max(0.0, 1.0 - (time_variance / 3600))  # 1-hour normalization
            strength_score = max(0.0, 1.0 - strength_variance)
            
            alignment_scores.append((time_score + strength_score) / 2)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
    
    def _detect_conflicts(self, signals: List[TimeframedSignal]) -> List[TimeframedSignal]:
        """Detect conflicting signals"""
        conflicting = []
        
        # Group by timestamp windows
        time_windows = defaultdict(list)
        
        for signal in signals:
            # Create 5-minute time windows
            window_key = signal.timestamp.replace(minute=(signal.timestamp.minute // 5) * 5, second=0, microsecond=0)
            time_windows[window_key].append(signal)
        
        # Check for conflicts within each window
        for window_signals in time_windows.values():
            if len(window_signals) < 2:
                continue
            
            signal_types = set(s.signal_type for s in window_signals)
            
            # Check for opposing signals
            if 'BUY' in signal_types and 'SELL' in signal_types:
                conflicting.extend(window_signals)
            elif 'STRONG_BUY' in signal_types and ('SELL' in signal_types or 'STRONG_SELL' in signal_types):
                conflicting.extend(window_signals)
            elif 'STRONG_SELL' in signal_types and ('BUY' in signal_types or 'STRONG_BUY' in signal_types):
                conflicting.extend(window_signals)
        
        return conflicting
    
    def _determine_sync_status(
        self,
        signals: List[TimeframedSignal],
        conflicts: List[TimeframedSignal],
        alignment_score: float
    ) -> SyncStatus:
        """Determine overall synchronization status"""
        if not signals:
            return SyncStatus.DESYNCHRONIZED
        
        if conflicts:
            return SyncStatus.CONFLICTING
        
        if alignment_score >= 0.8:
            return SyncStatus.SYNCHRONIZED
        elif alignment_score >= 0.5:
            return SyncStatus.PARTIAL_SYNC
        else:
            return SyncStatus.DESYNCHRONIZED
    
    def _find_dominant_timeframe(self, signals: List[TimeframedSignal]) -> TimeFrame:
        """Find the dominant timeframe based on signal strength and hierarchy"""
        if not signals:
            return TimeFrame.M15  # Default
        
        # Calculate weighted scores for each timeframe
        timeframe_scores = defaultdict(float)
        
        for signal in signals:
            hierarchy_weight = self.timeframe_hierarchy[signal.timeframe]
            timeframe_scores[signal.timeframe] += signal.strength * hierarchy_weight
        
        # Return timeframe with highest score
        return max(timeframe_scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_execution_window(
        self,
        signals: List[TimeframedSignal],
        dominant_timeframe: TimeFrame
    ) -> Tuple[datetime, datetime]:
        """Calculate optimal execution time window"""
        if not signals:
            now = datetime.now()
            return (now, now + timedelta(minutes=5))
        
        # Find latest signal timestamp
        latest_timestamp = max(s.timestamp for s in signals)
        
        # Calculate window based on dominant timeframe
        window_duration = timedelta(minutes=dominant_timeframe.value // 2)
        
        start_time = latest_timestamp
        end_time = latest_timestamp + window_duration
        
        return (start_time, end_time)
    
    def _create_empty_sync_result(self) -> SynchronizationResult:
        """Create empty synchronization result for error cases"""
        now = datetime.now()
        return SynchronizationResult(
            sync_status=SyncStatus.DESYNCHRONIZED,
            synchronized_signals=[],
            alignment_score=0.0,
            dominant_timeframe=TimeFrame.M15,
            conflicting_signals=[],
            sync_timestamp=now,
            execution_window=(now, now + timedelta(minutes=5))
        )
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        if not self.sync_history:
            return {}
        
        recent_syncs = self.sync_history[-50:]  # Last 50 synchronizations
        
        return {
            'average_alignment_score': np.mean([s.alignment_score for s in recent_syncs]),
            'sync_status_distribution': {
                status.value: len([s for s in recent_syncs if s.sync_status == status])
                for status in SyncStatus
            },
            'dominant_timeframes': {
                tf.name: len([s for s in recent_syncs if s.dominant_timeframe == tf])
                for tf in TimeFrame
            },
            'conflict_rate': len([s for s in recent_syncs if s.conflicting_signals]) / len(recent_syncs)
        }
    
    async def real_time_sync_monitor(self, signal_stream):
        """Real-time synchronization monitoring"""
        while True:
            try:
                # Collect signals from stream
                signals = []
                async for signal in signal_stream:
                    signals.append(signal)
                    
                    # Synchronize when we have enough signals
                    if len(signals) >= 3:
                        result = await self.synchronize_signals(signals)
                        
                        if result.sync_status == SyncStatus.SYNCHRONIZED:
                            logger.info(f"Synchronized signals detected: {result.alignment_score:.3f}")
                        
                        signals = []  # Reset buffer
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in real-time sync monitor: {e}")
                await asyncio.sleep(5)

# Example usage and testing
if __name__ == "__main__":
    async def test_timeframe_synchronizer():
        synchronizer = TimeframeSynchronizer()
        
        # Create test signals
        now = datetime.now()
        test_signals = [
            TimeframedSignal(
                timeframe=TimeFrame.H1,
                signal_type="BUY",
                strength=0.8,
                timestamp=now,
                price=1.2500,
                indicator="RSI"
            ),
            TimeframedSignal(
                timeframe=TimeFrame.M15,
                signal_type="BUY",
                strength=0.6,
                timestamp=now + timedelta(minutes=2),
                price=1.2501,
                indicator="MACD"
            ),
            TimeframedSignal(
                timeframe=TimeFrame.M5,
                signal_type="BUY",
                strength=0.7,
                timestamp=now + timedelta(minutes=1),
                price=1.2499,
                indicator="Stochastic"
            )
        ]
        
        # Synchronize signals
        result = await synchronizer.synchronize_signals(test_signals)
        
        print(f"Sync Status: {result.sync_status.value}")
        print(f"Alignment Score: {result.alignment_score:.3f}")
        print(f"Dominant Timeframe: {result.dominant_timeframe.name}")
        print(f"Synchronized Signals: {len(result.synchronized_signals)}")
        print(f"Execution Window: {result.execution_window[0]} - {result.execution_window[1]}")
    
    # Run test
    asyncio.run(test_timeframe_synchronizer())
