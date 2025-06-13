"""
Momentum Indicators Package

This package contains individual momentum indicator implementations
that inherit from StandardIndicatorInterface for trading-grade accuracy.
"""

from .awesome_oscillator import AwesomeOscillatorIndicator
from .chande_momentum_oscillator import ChandeMomentumOscillatorIndicator
from .commodity_channel_index import CommodityChannelIndex
from .correlation_matrix import CorrelationMatrixIndicator
from .detrended_price_oscillator import DetrendedPriceOscillatorIndicator
from .know_sure_thing import KnowSureThingIndicator
from .macd import MovingAverageConvergenceDivergenceIndicator
from .momentum import MomentumIndicator
from .money_flow_index import MoneyFlowIndexIndicator
from .percentage_price_oscillator import PercentagePriceOscillatorIndicator
from .rate_of_change import RateOfChangeIndicator
from .rsi import RelativeStrengthIndexIndicator
from .stochastic_oscillator import StochasticOscillator
from .trix import TRIXIndicator
from .true_strength_index import TrueStrengthIndexIndicator
from .ultimate_oscillator import UltimateOscillatorIndicator
from .williams_r import WilliamsRIndicator

__all__ = [
    "AwesomeOscillatorIndicator",
    "ChandeMomentumOscillatorIndicator",
    "CommodityChannelIndex",
    "CorrelationMatrixIndicator",
    "DetrendedPriceOscillatorIndicator",
    "KnowSureThingIndicator",
    "MomentumIndicator",
    "MoneyFlowIndexIndicator",
    "MovingAverageConvergenceDivergenceIndicator",
    "PercentagePriceOscillatorIndicator",
    "RateOfChangeIndicator",
    "RelativeStrengthIndexIndicator",
    "StochasticOscillator",
    "TRIXIndicator",
    "TrueStrengthIndexIndicator",
    "UltimateOscillatorIndicator",
    "WilliamsRIndicator",
]
