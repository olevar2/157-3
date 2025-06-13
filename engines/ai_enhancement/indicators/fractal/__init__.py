"""
Fractal Indicators Module

This module contains all fractal-based technical indicators for the Platform3 system.
Fractal indicators analyze market patterns using fractal geometry and chaos theory.
"""

from .fractal_adaptive_moving_average import FractalAdaptiveMovingAverage
from .fractal_breakout_indicator import FractalBreakoutIndicator
from .fractal_channel_indicator import FractalChannelIndicator
from .fractal_chaos_oscillator import FractalChaosOscillator
from .fractal_dimension_indicator import FractalDimensionIndicator
from .fractal_energy_indicator import FractalEnergyIndicator
from .fractal_volume_indicator import FractalVolumeIndicator
from .mandelbrot_fractal_indicator import MandelbrotFractalIndicator
from .chaos_fractal_dimension import ChaosFractalDimension
from .fractal_efficiency_ratio import FractalEfficiencyRatio
from .fractal_market_hypothesis import FractalMarketHypothesis
from .multifractal_dfa import MultifractalDFA, MultiFractalDFA  # Import both names
from .fractal_market_profile import FractalMarketProfile

# Import all data classes
from .fractal_data_classes import (
    FractalPoint,
    FractalChannelResult,
    FractalChaosResult,
    FractalEnergyResult,
    FractalAdaptiveResult,
    MandelbrotResult,
    FractalDimensionResult,
    FractalBreakoutResult,
    FractalVolumeResult,
    MultifractalDFAResult,
    ChaosFractalDimensionResult,
    FractalEfficiencyResult,
    FractalMarketHypothesisResult,
    FractalMarketProfileResult
)

__all__ = [
    # Indicator classes
    'FractalAdaptiveMovingAverage',
    'FractalBreakoutIndicator',
    'FractalChannelIndicator',
    'FractalChaosOscillator',
    'FractalDimensionIndicator',
    'FractalEnergyIndicator',
    'FractalVolumeIndicator',
    'MandelbrotFractalIndicator',
    'ChaosFractalDimension',
    'FractalEfficiencyRatio',
    'FractalMarketHypothesis',
    'MultifractalDFA',
    'MultiFractalDFA',
    'FractalMarketProfile',
    
    # Data classes
    'FractalPoint',
    'FractalChannelResult',
    'FractalChaosResult',
    'FractalEnergyResult',
    'FractalAdaptiveResult',
    'MandelbrotResult',
    'FractalDimensionResult',
    'FractalBreakoutResult',
    'FractalVolumeResult',
    'MultifractalDFAResult',
    'ChaosFractalDimensionResult',
    'FractalEfficiencyResult',
    'FractalMarketHypothesisResult',
    'FractalMarketProfileResult'
]