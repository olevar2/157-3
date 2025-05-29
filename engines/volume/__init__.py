"""
<<<<<<< HEAD
High-Frequency Volume Analysis Package
Volume-based analysis for short-term trading validation (SCALPING/DAY TRADING FOCUS)

This package provides comprehensive volume analysis including:
- Tick volume indicators for M1-M5 analysis
- Volume Spread Analysis (VSA) for day trading
- Order flow imbalance detection for scalping
- Session-based volume profiles for breakout validation
- Smart money indicators for institutional flow detection

All modules are optimized for scalping and day trading with real-time volume confirmation.
"""

from .TickVolumeIndicators import TickVolumeIndicators, TickVolumeResult
from .VolumeSpreadAnalysis import VolumeSpreadAnalysis, VolumeSpreadResult
from .OrderFlowImbalance import OrderFlowImbalance, OrderFlowResult
from .VolumeProfiles import VolumeProfiles, VolumeProfileResult
from .SmartMoneyIndicators import SmartMoneyIndicators, SmartMoneyResult

__all__ = [
    'TickVolumeIndicators',
    'TickVolumeResult',
    'VolumeSpreadAnalysis',
    'VolumeSpreadResult',
    'OrderFlowImbalance',
    'OrderFlowResult',
    'VolumeProfiles',
    'VolumeProfileResult',
    'SmartMoneyIndicators',
    'SmartMoneyResult'
]

__version__ = '1.0.0'
__author__ = 'Platform3 Analytics Team'
__description__ = 'High-Frequency Volume Analysis for Scalping/Day Trading Focus'
=======
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
>>>>>>> 5e659b3064c215382ffc9ef1f13510cbfdd547a7
