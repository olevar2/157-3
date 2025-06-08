# -*- coding: utf-8 -*-
"""
Platform3 Volume Analysis Module
===============================

High-performance volume analysis indicators for humanitarian forex trading.
"""

# Import only the classes that actually exist
try:
    from .TickVolumeIndicators import Tickvolumeindicators, VolumeData
except ImportError as e:
    print(f"Warning: TickVolumeIndicators import failed: {e}")
    Tickvolumeindicators = None

try:
    from .VolumeSpreadAnalysis import Volumespreadanalysis
except ImportError as e:
    print(f"Warning: VolumeSpreadAnalysis import failed: {e}")
    Volumespreadanalysis = None

try:
    from .OrderFlowImbalance import Orderflowimbalance
except ImportError as e:
    print(f"Warning: OrderFlowImbalance import failed: {e}")
    Orderflowimbalance = None

try:
    from .VolumeProfiles import Volumeprofiles
except ImportError as e:
    print(f"Warning: VolumeProfiles import failed: {e}")
    Volumeprofiles = None

try:
    from .SmartMoneyIndicators import Smartmoneyindicators
except ImportError as e:
    print(f"Warning: SmartMoneyIndicators import failed: {e}")
    Smartmoneyindicators = None

# Import new volume indicators
try:
    from .institutional_flow_detector import InstitutionalFlowDetector
except ImportError as e:
    print(f"Warning: InstitutionalFlowDetector import failed: {e}")
    InstitutionalFlowDetector = None

try:
    from .tick_volume_analyzer import TickVolumeAnalyzer
except ImportError as e:
    print(f"Warning: TickVolumeAnalyzer import failed: {e}")
    TickVolumeAnalyzer = None

try:
    from .market_microstructure_indicator import MarketMicrostructureIndicator
except ImportError as e:
    print(f"Warning: MarketMicrostructureIndicator import failed: {e}")
    MarketMicrostructureIndicator = None

try:
    from .volume_delta_indicator import VolumeDeltaIndicator
except ImportError as e:
    print(f"Warning: VolumeDeltaIndicator import failed: {e}")
    VolumeDeltaIndicator = None

try:
    from .liquidity_flow_indicator import LiquidityFlowIndicator
except ImportError as e:
    print(f"Warning: LiquidityFlowIndicator import failed: {e}")
    LiquidityFlowIndicator = None

# Add more volume indicators for Phase 3B completion
try:
    from .accumulation_distribution import AccumulationDistributionLine
except ImportError as e:
    print(f"Warning: AccumulationDistributionLine import failed: {e}")
    AccumulationDistributionLine = None

try:
    from .chaikin_money_flow import ChaikinMoneyFlow
except ImportError as e:
    print(f"Warning: ChaikinMoneyFlow import failed: {e}")
    ChaikinMoneyFlow = None

try:
    from .volume_price_trend import VolumePriceTrend
except ImportError as e:
    print(f"Warning: VolumePriceTrend import failed: {e}")
    VolumePriceTrend = None

try:
    from .order_flow_block_trade_detector import OrderFlowBlockTradeDetector
except ImportError as e:
    print(f"Warning: OrderFlowBlockTradeDetector import failed: {e}")
    OrderFlowBlockTradeDetector = None

try:
    from .order_flow_sequence_analyzer import OrderFlowSequenceAnalyzer
except ImportError as e:
    print(f"Warning: OrderFlowSequenceAnalyzer import failed: {e}")
    OrderFlowSequenceAnalyzer = None

try:
    from .volume_breakout_detector import VolumeBreakoutDetector
except ImportError as e:
    print(f"Warning: VolumeBreakoutDetector import failed: {e}")
    VolumeBreakoutDetector = None

try:
    from .volume_weighted_market_depth import VolumeWeightedMarketDepthIndicator
except ImportError as e:
    print(f"Warning: VolumeWeightedMarketDepthIndicator import failed: {e}")
    VolumeWeightedMarketDepthIndicator = None

try:
    from .ease_of_movement import EaseOfMovement
except ImportError as e:
    print(f"Warning: EaseOfMovement import failed: {e}")
    EaseOfMovement = None

try:
    from .force_index import ForceIndex
except ImportError as e:
    print(f"Warning: ForceIndex import failed: {e}")
    ForceIndex = None

try:
    from .volume_oscillator import VolumeOscillator
except ImportError as e:
    print(f"Warning: VolumeOscillator import failed: {e}")
    VolumeOscillator = None

try:
    from .klinger_oscillator import KlingerOscillator
except ImportError as e:
    print(f"Warning: KlingerOscillator import failed: {e}")
    KlingerOscillator = None

try:
    from .volume_rate_of_change import VolumeRateOfChange
except ImportError as e:
    print(f"Warning: VolumeRateOfChange import failed: {e}")
    VolumeRateOfChange = None

try:
    from .negative_volume_index import NegativeVolumeIndex
except ImportError as e:
    print(f"Warning: NegativeVolumeIndex import failed: {e}")
    NegativeVolumeIndex = None

try:
    from .positive_volume_index import PositiveVolumeIndex
except ImportError as e:
    print(f"Warning: PositiveVolumeIndex import failed: {e}")
    PositiveVolumeIndex = None

try:
    from .price_volume_rank import PriceVolumeRank
except ImportError as e:
    print(f"Warning: PriceVolumeRank import failed: {e}")
    PriceVolumeRank = None

try:
    from .obv import OnBalanceVolume
except ImportError as e:
    print(f"Warning: OnBalanceVolume import failed: {e}")
    OnBalanceVolume = None

try:
    from .vwap import VolumeWeightedAveragePrice
except ImportError as e:
    print(f"Warning: VolumeWeightedAveragePrice import failed: {e}")
    VolumeWeightedAveragePrice = None

# Export only successful imports
__all__ = []
if Tickvolumeindicators:
    __all__.extend(['Tickvolumeindicators', 'VolumeData'])
if Volumespreadanalysis:
    __all__.append('Volumespreadanalysis')
if Orderflowimbalance:
    __all__.append('Orderflowimbalance')
if Volumeprofiles:
    __all__.append('Volumeprofiles')
if Smartmoneyindicators:
    __all__.append('Smartmoneyindicators')
    
# Add new volume indicators to exports
if InstitutionalFlowDetector:
    __all__.append('InstitutionalFlowDetector')
if TickVolumeAnalyzer:
    __all__.append('TickVolumeAnalyzer')
if MarketMicrostructureIndicator:
    __all__.append('MarketMicrostructureIndicator')
if VolumeDeltaIndicator:
    __all__.append('VolumeDeltaIndicator')
if LiquidityFlowIndicator:
    __all__.append('LiquidityFlowIndicator')

# Add Phase 3B volume indicators to exports
if AccumulationDistributionLine:
    __all__.append('AccumulationDistributionLine')
if ChaikinMoneyFlow:
    __all__.append('ChaikinMoneyFlow')
if VolumePriceTrend:
    __all__.append('VolumePriceTrend')
if OrderFlowBlockTradeDetector:
    __all__.append('OrderFlowBlockTradeDetector')
if OrderFlowSequenceAnalyzer:
    __all__.append('OrderFlowSequenceAnalyzer')
if VolumeBreakoutDetector:
    __all__.append('VolumeBreakoutDetector')
if VolumeWeightedMarketDepthIndicator:
    __all__.append('VolumeWeightedMarketDepthIndicator')
if EaseOfMovement:
    __all__.append('EaseOfMovement')
if ForceIndex:
    __all__.append('ForceIndex')
if VolumeOscillator:
    __all__.append('VolumeOscillator')
if KlingerOscillator:
    __all__.append('KlingerOscillator')
if VolumeRateOfChange:
    __all__.append('VolumeRateOfChange')
if NegativeVolumeIndex:
    __all__.append('NegativeVolumeIndex')
if PositiveVolumeIndex:
    __all__.append('PositiveVolumeIndex')
if PriceVolumeRank:
    __all__.append('PriceVolumeRank')
if OnBalanceVolume:
    __all__.append('OnBalanceVolume')
if VolumeWeightedAveragePrice:
    __all__.append('VolumeWeightedAveragePrice')

__version__ = '1.0.0'
__author__ = 'Platform3 Analytics Team'