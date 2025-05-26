"""
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
