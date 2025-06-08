# -*- coding: utf-8 -*-
"""
Fractal Geometry Analysis Engine
Advanced chaos theory and fractal analysis for market complexity measurement.

This engine contains indicators for fractal analysis and chaos theory applications
to financial markets. The indicators detect self-similarity, fractal patterns,
multi-fractal properties, and chaotic behaviors in price movement.
"""

from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time

# Base indicators
from .fractal_dimension_calculator import FractalDimensionCalculator
from .chaos_theory_indicators import ChaosTheoryIndicators
from .self_similarity_detector import SelfSimilarityDetector

# Core fractal indicators
from .fractal_breakout import FractalBreakoutIndicator
from .fractal_channel import FractalChannelIndicator
from .fractal_chaos_oscillator import FractalChaosOscillator
from .fractal_correlation_dimension import FractalCorrelationDimension
from .fractal_efficiency_ratio import FractalEfficiencyRatio
from .fractal_energy_indicator import FractalEnergyIndicator

# Market structure indicators
from .fractal_market_hypothesis import FractalMarketHypothesis
from .fractal_market_profile import FractalMarketProfile
from .fractal_momentum_oscillator import FractalMomentumOscillator
from .fractal_volume_analysis import FractalVolumeAnalysis
from .fractal_wave_counter import FractalWaveCounter

# Advanced indicators
from .frama import FractalAdaptiveMovingAverage
from .mfdfa import MultiFractalDFA
from .mandelbrot_fractal import MandelbrotFractalIndicator
from .hurst_exponent import HurstExponentCalculator

# Implementation template
from .implementation_template import FractalIndicatorTemplate

__all__ = [
    # Base indicators
    'FractalDimensionCalculator',
    'ChaosTheoryIndicators', 
    'SelfSimilarityDetector',
    
    # Core fractal indicators
    'FractalBreakoutIndicator',
    'FractalChannelIndicator',
    'FractalChaosOscillator',
    'FractalCorrelationDimension',
    'FractalEfficiencyRatio',
    'FractalEnergyIndicator',
    
    # Market structure indicators
    'FractalMarketHypothesis',
    'FractalMarketProfile',
    'FractalMomentumOscillator',
    'FractalVolumeAnalysis',
    'FractalWaveCounter',
    
    # Advanced indicators
    'FractalAdaptiveMovingAverage',
    'MultiFractalDFA',
    'MandelbrotFractalIndicator',
    'HurstExponentCalculator',
    
    # Implementation template
    'FractalIndicatorTemplate'
]
