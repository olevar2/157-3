#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Momentum Indicators Module
==========================

Complete collection of momentum-based technical indicators for Platform3.

Author: Platform3 AI System
Created: June 6, 2025
"""

# Core momentum indicators
from .rsi import RelativeStrengthIndex
from .macd import MovingAverageConvergenceDivergence as MACD
from .stochastic import StochasticOscillator
from .williams_r import WilliamsR
from .roc import RateOfChange
from .momentum import MomentumIndicator
from .cci import CommodityChannelIndex
from .awesome_oscillator import AwesomeOscillator
from .mfi import MoneyFlowIndex
from .trix import TRIX
from .ultimate_oscillator import UltimateOscillator
from .true_strength_index import TrueStrengthIndex
from .percentage_price_oscillator import PercentagePriceOscillator
from .detrended_price_oscillator import DetrendedPriceOscillator
from .chande_momentum_oscillator import ChandeMomentumOscillator
from .know_sure_thing import KnowSureThing

# Advanced correlation and momentum indicators
from .correlation_momentum import (
    DynamicCorrelationIndicator,
    RelativeMomentumIndicator,
    CorrelationMatrix,
    MomentumMetrics,
    create_dynamic_correlation_indicator,
    create_relative_momentum_indicator
)

# Export all momentum indicators
__all__ = [
    # Core momentum indicators
    'RelativeStrengthIndex',
    'MACD',
    'StochasticOscillator', 
    'WilliamsR',
    'RateOfChange',
    'MomentumIndicator',
    'CommodityChannelIndex',
    'AwesomeOscillator',
    'MoneyFlowIndex',
    'TRIX',
    'UltimateOscillator',
    'TrueStrengthIndex',
    'PercentagePriceOscillator',
    'DetrendedPriceOscillator',
    'ChandeMomentumOscillator',
    'KnowSureThing',
    
    # Advanced correlation and momentum indicators
    'DynamicCorrelationIndicator',
    'RelativeMomentumIndicator',
    'CorrelationMatrix',
    'MomentumMetrics',
    'create_dynamic_correlation_indicator',
    'create_relative_momentum_indicator'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Platform3 AI System"
__description__ = "Complete momentum indicators collection for Platform3 trading engine"
