"""
Timeframe Synchronizer Engine
Multi-timeframe alignment for signal aggregation.
Synchronizes signals across different timeframes for coherent analysis.
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

from .SignalAggregator import SignalInput, Timeframe


@dataclass
class TimeframeWindow:
    """Time window for synchronizing signals"""
    start_time: float
    end_time: float
    timeframe: Timeframe
    duration_seconds: int


@dataclass
class SynchronizedSignals:
    """Synchronized signals across timeframes"""
    primary_timeframe: Timeframe
    sync_window: TimeframeWindow
    aligned_signals: Dict[Timeframe, List[SignalInput]]
    temporal_weights: Dict[Timeframe, float]
    synchronization_quality: float  # 0-1 quality of synchronization
    missing_timeframes: List[Timeframe]
    sync_metadata: Dict[str, Any]


class TimeframeSynchronizer:
    """
    Timeframe Synchronizer Engine
    Aligns signals across multiple timeframes for coherent multi-TF analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Timeframe durations in seconds
        self.timeframe_durations = {
            Timeframe.M1: 60,
            Timeframe.M5: 300,
            Timeframe.M15: 900,
            Timeframe.M30: 1800,
            Timeframe.H1: 3600,
            Timeframe.H4: 14400
        }
        
        # Synchronization tolerances (how far signals can be from sync point)
        self.sync_tolerances = {
            Timeframe.M1: 30,    # 30 seconds
            Timeframe.M5: 150,   # 2.5 minutes
            Timeframe.M15: 450,  # 7.5 minutes
            Timeframe.M30: 900,  # 15 minutes
            Timeframe.H1: 1800,  # 30 minutes
            Timeframe.H4: 7200   # 2 hours
        }
        
        # Temporal decay weights (how signal relevance decreases over time)
        self.temporal_decay_rates = {
            Timeframe.M1: 0.1,   # Fast decay for M1
            Timeframe.M5: 0.05,
            Timeframe.M15: 0.03,
            Timeframe.M30: 0.02,
            Timeframe.H1: 0.01,
            Timeframe.H4: 0.005  # Slow decay for H4
        }
        
        # Performance tracking
        self.sync_count = 0
        self.total_sync_time = 0.0
        
    async def synchronize_signals(self, signals: List[SignalInput]) -> SynchronizedSignals:
        """
        Synchronize signals across multiple timeframes
        """
        start_time = time.time()
        
        try:
            if not signals:
                raise ValueError("No signals provided for synchronization")
            
            # Group signals by timeframe
            timeframe_groups = await self._group_signals_by_timeframe(signals)
            
            # Determine primary timeframe (highest timeframe with signals)
            primary_timeframe = await self._determine_primary_timeframe(timeframe_groups)
            
            # Create synchronization window
            sync_window = await self._create_sync_window(signals, primary_timeframe)
            
            # Align signals to synchronization window
            aligned_signals = await self._align_signals_to_window(timeframe_groups, sync_window)
            
            # Calculate temporal weights
            temporal_weights = await self._calculate_temporal_weights(aligned_signals, sync_window)
            
            # Assess synchronization quality
            sync_quality = await self._assess_synchronization_quality(aligned_signals, timeframe_groups)
            
            # Identify missing timeframes
            missing_timeframes = await self._identify_missing_timeframes(aligned_signals)
            
            # Create sync metadata
            sync_metadata = await self._create_sync_metadata(signals, aligned_signals, sync_window)
            
            # Update performance tracking
            sync_time = time.time() - start_time
            self.sync_count += 1
            self.total_sync_time += sync_time
            
            self.logger.debug(f"Synchronized {len(signals)} signals across {len(aligned_signals)} timeframes")
            
            return SynchronizedSignals(
                primary_timeframe=primary_timeframe,
                sync_window=sync_window,
                aligned_signals=aligned_signals,
                temporal_weights=temporal_weights,
                synchronization_quality=sync_quality,
                missing_timeframes=missing_timeframes,
                sync_metadata=sync_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Signal synchronization failed: {e}")
            raise
    
    async def _group_signals_by_timeframe(self, signals: List[SignalInput]) -> Dict[Timeframe, List[SignalInput]]:
        """Group signals by their timeframes"""
        timeframe_groups = defaultdict(list)
        
        for signal in signals:
            timeframe_groups[signal.timeframe].append(signal)
        
        # Sort signals within each timeframe by timestamp
        for timeframe in timeframe_groups:
            timeframe_groups[timeframe].sort(key=lambda x: x.timestamp, reverse=True)
        
        return dict(timeframe_groups)
    
    async def _determine_primary_timeframe(self, timeframe_groups: Dict[Timeframe, List[SignalInput]]) -> Timeframe:
        """Determine the primary timeframe for synchronization"""
        
        # Priority order: H4 > H1 > M30 > M15 > M5 > M1
        priority_order = [Timeframe.H4, Timeframe.H1, Timeframe.M30, Timeframe.M15, Timeframe.M5, Timeframe.M1]
        
        for timeframe in priority_order:
            if timeframe in timeframe_groups and timeframe_groups[timeframe]:
                return timeframe
        
        # Fallback to any available timeframe
        if timeframe_groups:
            return list(timeframe_groups.keys())[0]
        
        raise ValueError("No timeframes available for synchronization")
    
    async def _create_sync_window(self, signals: List[SignalInput], primary_timeframe: Timeframe) -> TimeframeWindow:
        """Create synchronization window based on primary timeframe"""
        
        # Get the most recent signal timestamp
        latest_timestamp = max(signal.timestamp for signal in signals)
        
        # Calculate window duration based on primary timeframe
        window_duration = self.timeframe_durations[primary_timeframe]
        
        # Create window ending at latest timestamp
        end_time = latest_timestamp
        start_time = end_time - window_duration
        
        return TimeframeWindow(
            start_time=start_time,
            end_time=end_time,
            timeframe=primary_timeframe,
            duration_seconds=window_duration
        )
    
    async def _align_signals_to_window(
        self, 
        timeframe_groups: Dict[Timeframe, List[SignalInput]], 
        sync_window: TimeframeWindow
    ) -> Dict[Timeframe, List[SignalInput]]:
        """Align signals to the synchronization window"""
        
        aligned_signals = {}
        
        for timeframe, signals in timeframe_groups.items():
            # Get tolerance for this timeframe
            tolerance = self.sync_tolerances[timeframe]
            
            # Find signals within the sync window (with tolerance)
            window_start = sync_window.start_time - tolerance
            window_end = sync_window.end_time + tolerance
            
            aligned_timeframe_signals = []
            for signal in signals:
                if window_start <= signal.timestamp <= window_end:
                    aligned_timeframe_signals.append(signal)
            
            if aligned_timeframe_signals:
                aligned_signals[timeframe] = aligned_timeframe_signals
        
        return aligned_signals
    
    async def _calculate_temporal_weights(
        self, 
        aligned_signals: Dict[Timeframe, List[SignalInput]], 
        sync_window: TimeframeWindow
    ) -> Dict[Timeframe, float]:
        """Calculate temporal weights for each timeframe based on signal recency"""
        
        temporal_weights = {}
        reference_time = sync_window.end_time
        
        for timeframe, signals in aligned_signals.items():
            if not signals:
                temporal_weights[timeframe] = 0.0
                continue
            
            # Calculate weight based on most recent signal in timeframe
            most_recent_signal = max(signals, key=lambda x: x.timestamp)
            time_diff = reference_time - most_recent_signal.timestamp
            
            # Apply temporal decay
            decay_rate = self.temporal_decay_rates[timeframe]
            weight = np.exp(-decay_rate * time_diff / 60)  # Decay per minute
            
            temporal_weights[timeframe] = min(weight, 1.0)
        
        return temporal_weights
    
    async def _assess_synchronization_quality(
        self, 
        aligned_signals: Dict[Timeframe, List[SignalInput]], 
        original_groups: Dict[Timeframe, List[SignalInput]]
    ) -> float:
        """Assess the quality of synchronization"""
        
        if not original_groups:
            return 0.0
        
        # Calculate coverage ratio (how many timeframes have aligned signals)
        total_timeframes = len(original_groups)
        aligned_timeframes = len(aligned_signals)
        coverage_ratio = aligned_timeframes / total_timeframes
        
        # Calculate signal retention ratio (how many signals were kept)
        total_signals = sum(len(signals) for signals in original_groups.values())
        aligned_signal_count = sum(len(signals) for signals in aligned_signals.values())
        retention_ratio = aligned_signal_count / total_signals if total_signals > 0 else 0
        
        # Calculate temporal coherence (how close signals are to sync window)
        temporal_coherence = await self._calculate_temporal_coherence(aligned_signals)
        
        # Combine metrics
        quality_score = (coverage_ratio * 0.4 + retention_ratio * 0.3 + temporal_coherence * 0.3)
        
        return min(quality_score, 1.0)
    
    async def _calculate_temporal_coherence(self, aligned_signals: Dict[Timeframe, List[SignalInput]]) -> float:
        """Calculate how temporally coherent the aligned signals are"""
        
        if not aligned_signals:
            return 0.0
        
        # Get all signal timestamps
        all_timestamps = []
        for signals in aligned_signals.values():
            all_timestamps.extend([signal.timestamp for signal in signals])
        
        if len(all_timestamps) < 2:
            return 1.0
        
        # Calculate timestamp variance (lower variance = better coherence)
        timestamp_variance = np.var(all_timestamps)
        
        # Normalize variance to coherence score (0-1)
        # Lower variance = higher coherence
        max_expected_variance = 3600  # 1 hour variance
        coherence = max(0, 1 - (timestamp_variance / max_expected_variance))
        
        return coherence
    
    async def _identify_missing_timeframes(self, aligned_signals: Dict[Timeframe, List[SignalInput]]) -> List[Timeframe]:
        """Identify timeframes that are missing from aligned signals"""
        
        all_timeframes = set(Timeframe)
        present_timeframes = set(aligned_signals.keys())
        missing_timeframes = list(all_timeframes - present_timeframes)
        
        return missing_timeframes
    
    async def _create_sync_metadata(
        self, 
        original_signals: List[SignalInput], 
        aligned_signals: Dict[Timeframe, List[SignalInput]], 
        sync_window: TimeframeWindow
    ) -> Dict[str, Any]:
        """Create metadata about the synchronization process"""
        
        return {
            'original_signal_count': len(original_signals),
            'aligned_signal_count': sum(len(signals) for signals in aligned_signals.values()),
            'timeframes_present': len(aligned_signals),
            'sync_window_duration': sync_window.duration_seconds,
            'sync_window_start': sync_window.start_time,
            'sync_window_end': sync_window.end_time,
            'primary_timeframe': sync_window.timeframe.value,
            'signal_sources': list(set(signal.source for signal in original_signals)),
            'signal_types': list(set(signal.signal_type for signal in original_signals))
        }
    
    async def get_timeframe_hierarchy(self) -> List[Timeframe]:
        """Get timeframe hierarchy from highest to lowest"""
        return [Timeframe.H4, Timeframe.H1, Timeframe.M30, Timeframe.M15, Timeframe.M5, Timeframe.M1]
    
    async def get_sync_statistics(self, synchronized_signals: SynchronizedSignals) -> Dict[str, Any]:
        """Get detailed synchronization statistics"""
        
        stats = {
            'synchronization_quality': synchronized_signals.synchronization_quality,
            'primary_timeframe': synchronized_signals.primary_timeframe.value,
            'timeframes_synchronized': len(synchronized_signals.aligned_signals),
            'missing_timeframes': [tf.value for tf in synchronized_signals.missing_timeframes],
            'temporal_weights': {tf.value: weight for tf, weight in synchronized_signals.temporal_weights.items()},
            'sync_window_duration': synchronized_signals.sync_window.duration_seconds,
            'total_aligned_signals': sum(len(signals) for signals in synchronized_signals.aligned_signals.values())
        }
        
        # Add per-timeframe statistics
        timeframe_stats = {}
        for timeframe, signals in synchronized_signals.aligned_signals.items():
            timeframe_stats[timeframe.value] = {
                'signal_count': len(signals),
                'signal_types': list(set(signal.signal_type for signal in signals)),
                'sources': list(set(signal.source for signal in signals)),
                'avg_strength': np.mean([signal.strength for signal in signals]) if signals else 0,
                'avg_confidence': np.mean([signal.confidence for signal in signals]) if signals else 0
            }
        
        stats['timeframe_details'] = timeframe_stats
        
        return stats
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get synchronization performance metrics"""
        return {
            'total_synchronizations': self.sync_count,
            'average_sync_time_ms': (self.total_sync_time / self.sync_count * 1000) 
                                  if self.sync_count > 0 else 0
        }
