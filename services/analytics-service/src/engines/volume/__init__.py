"""
High-Frequency Volume Analysis Engine for Scalping and Day Trading

This package provides comprehensive volume analysis tools optimized for short-term trading:
- Tick volume analysis for M1-M5 scalping strategies
- Volume Spread Analysis (VSA) for day trading
- Order flow imbalance detection for institutional activity
- Session-based volume profiles for key level identification
- Smart money indicators for institutional flow detection

Expected Benefits:
- Real-time volume confirmation for scalping entries
- Smart money flow detection for day trading
- Volume-based breakout validation
- Order flow imbalance alerts for quick profits
- Session-based volume analysis for optimal timing
"""

from .TickVolumeIndicators import (
    TickVolumeIndicators,
    TickVolumeSignal,
    VolumeAnalysisResult,
    VolumeTrend,
    VolumeStrength
)

from .VolumeSpreadAnalysis import (
    VolumeSpreadAnalysis,
    VSAAnalysisResult,
    VSABar,
    VSASignalType,
    VolumeStrength as VSAVolumeStrength,
    SpreadSize
)

from .OrderFlowImbalance import (
    OrderFlowImbalance,
    OrderFlowAnalysisResult,
    OrderFlowBar,
    ImbalanceType,
    ImbalanceStrength
)

from .VolumeProfiles import (
    VolumeProfiles,
    VolumeProfileAnalysisResult,
    VolumeProfile,
    VolumeNode,
    ValueArea,
    TradingSession,
    VolumeNodeType
)

from .SmartMoneyIndicators import (
    SmartMoneyIndicators,
    SmartMoneyAnalysisResult,
    SmartMoneySignal,
    InstitutionalFootprint,
    SmartMoneyActivity,
    InstitutionalBehavior,
    FlowStrength
)

__all__ = [
    # Main analyzer classes
    'TickVolumeIndicators',
    'VolumeSpreadAnalysis', 
    'OrderFlowImbalance',
    'VolumeProfiles',
    'SmartMoneyIndicators',
    
    # Tick volume components
    'TickVolumeSignal',
    'VolumeAnalysisResult',
    'VolumeTrend',
    'VolumeStrength',
    
    # VSA components
    'VSAAnalysisResult',
    'VSABar',
    'VSASignalType',
    'VSAVolumeStrength',
    'SpreadSize',
    
    # Order flow components
    'OrderFlowAnalysisResult',
    'OrderFlowBar',
    'ImbalanceType',
    'ImbalanceStrength',
    
    # Volume profile components
    'VolumeProfileAnalysisResult',
    'VolumeProfile',
    'VolumeNode',
    'ValueArea',
    'TradingSession',
    'VolumeNodeType',
    
    # Smart money components
    'SmartMoneyAnalysisResult',
    'SmartMoneySignal',
    'InstitutionalFootprint',
    'SmartMoneyActivity',
    'InstitutionalBehavior',
    'FlowStrength'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "High-frequency volume analysis for scalping and day trading"
