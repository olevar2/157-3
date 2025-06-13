"""
Fractal Data Classes

This module contains all data class definitions used by fractal indicators
in the Platform3 system. These classes provide structured output formats
for fractal analysis results.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np


@dataclass
class FractalPoint:
    """Represents a fractal point in price data"""
    index: int
    price: float
    fractal_type: str  # 'high' or 'low'
    strength: float = 1.0


@dataclass
class FractalChannelResult:
    """Result structure for Fractal Channel analysis"""
    upper_channel: float
    lower_channel: float
    middle_channel: float
    channel_width: float
    fractal_high: Optional[float] = None
    fractal_low: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    breakout_probability: float = 0.0


@dataclass
class FractalChaosResult:
    """Result structure for Fractal Chaos analysis"""
    chaos_value: float
    fractal_dimension: float
    market_regime: str  # 'ranging', 'trending', 'chaotic'
    complexity_score: float
    predictability_index: float
    regime_strength: float


@dataclass
class FractalEnergyResult:
    """Result structure for Fractal Energy analysis"""
    energy_level: float
    momentum_strength: float
    energy_direction: str  # 'bullish', 'bearish', 'neutral'
    energy_sustainability: float
    power_ratio: float
    kinetic_energy: float
    potential_energy: float


@dataclass
class FractalAdaptiveResult:
    """Result structure for Fractal Adaptive Moving Average"""
    frama_value: float
    fractal_dimension: float
    smoothing_factor: float
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    adaptation_speed: float
    signal_strength: float


@dataclass
class MandelbrotResult:
    """Result structure for Mandelbrot Fractal analysis"""
    mandelbrot_value: float
    complexity_index: float
    fractal_stability: float
    convergence_iterations: int
    pattern_strength: float
    fractal_type: str  # 'stable', 'chaotic', 'transitional'


@dataclass
class FractalDimensionResult:
    """Result structure for Fractal Dimension analysis"""
    fractal_dimension: float
    box_counting_dimension: float
    correlation_dimension: float
    capacity_dimension: float
    scaling_exponent: float
    dimension_confidence: float


@dataclass
class FractalBreakoutResult:
    """Result structure for Fractal Breakout analysis"""
    breakout_signal: bool
    breakout_direction: str  # 'bullish', 'bearish', 'none'
    breakout_strength: float
    fractal_level: float
    confirmation_period: int
    reliability_score: float


@dataclass
class FractalVolumeResult:
    """Result structure for Fractal Volume analysis"""
    fractal_volume: float
    volume_dimension: float
    volume_pattern: str  # 'accumulation', 'distribution', 'neutral'
    volume_strength: float
    fractal_volume_ratio: float
    anomaly_score: float


@dataclass
class MultifractalDFAResult:
    """Result structure for Multi-fractal Detrended Fluctuation Analysis"""
    hurst_exponents: np.ndarray
    multifractal_spectrum: Dict
    singularity_spectrum: Dict
    is_multifractal: bool
    multifractal_strength: float
    market_efficiency: float
    box_sizes: np.ndarray
    q_values: np.ndarray
    fluctuation_functions: np.ndarray


@dataclass 
class ChaosFractalDimensionResult:
    """Result structure for Chaos Fractal Dimension analysis"""
    fractal_dimension: float
    lyapunov_exponent: float
    entropy: float
    predictability: float
    chaos_level: str


@dataclass
class FractalEfficiencyResult:
    """Result structure for Fractal Efficiency Ratio analysis"""
    efficiency_ratio: float
    market_efficiency: str
    trend_strength: float
    noise_level: float


@dataclass
class FractalMarketHypothesisResult:
    """Result structure for Fractal Market Hypothesis analysis"""
    hurst_exponent: float
    market_regime: str
    persistence: float
    mean_reversion_tendency: float


@dataclass
class FractalMarketProfileResult:
    """Result structure for Fractal Market Profile analysis"""
    profile_levels: List[float]
    volume_at_price: Dict[float, float]
    poc_level: float  # Point of Control
    value_area_high: float
    value_area_low: float
    market_balance: str


# Export all result classes
__all__ = [
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