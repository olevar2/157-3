"""
Fast Signal Aggregation Engine Package
Multi-timeframe signal combination for short-term trading.

This package provides comprehensive signal aggregation capabilities including:
- Multi-timeframe signal synchronization
- Conflict resolution between competing signals
- Confidence calculation and scoring
- Fast decision matrix for trading execution

Components:
- SignalAggregator: Main aggregation engine for M1-H4 signal combination
- ConflictResolver: Handles conflicting signals from different timeframes
- ConfidenceCalculator: Calculates signal strength and confidence scores
- TimeframeSynchronizer: Aligns signals across multiple timeframes
- QuickDecisionMatrix: Fast buy/sell/hold decision making
"""

from .SignalAggregator import (
    SignalAggregator,
    SignalInput,
    AggregatedSignal,
    Timeframe
)

from .ConflictResolver import (
    ConflictResolver,
    ResolvedSignal,
    ConflictAnalysis,
    ConflictResolutionStrategy
)

from .ConfidenceCalculator import (
    ConfidenceCalculator,
    ConfidenceMetrics,
    ConfidenceFactors
)

from .TimeframeSynchronizer import (
    TimeframeSynchronizer,
    SynchronizedSignals,
    TimeframeWindow
)

from .QuickDecisionMatrix import (
    QuickDecisionMatrix,
    TradingDecision,
    DecisionType,
    RiskLevel
)

__all__ = [
    # Main classes
    'SignalAggregator',
    'ConflictResolver', 
    'ConfidenceCalculator',
    'TimeframeSynchronizer',
    'QuickDecisionMatrix',
    
    # Data classes
    'SignalInput',
    'AggregatedSignal',
    'ResolvedSignal',
    'ConflictAnalysis',
    'ConfidenceMetrics',
    'SynchronizedSignals',
    'TimeframeWindow',
    'TradingDecision',
    
    # Enums
    'Timeframe',
    'ConflictResolutionStrategy',
    'ConfidenceFactors',
    'DecisionType',
    'RiskLevel'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Fast Signal Aggregation Engine for Multi-Timeframe Trading"
