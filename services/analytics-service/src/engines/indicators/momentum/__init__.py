"""
Momentum Indicators Module

This module provides comprehensive momentum-based technical indicators for forex trading.
Momentum indicators measure the rate of change in price movements and help identify
the strength and direction of trends, as well as potential reversal points.

Key Indicators:
- RSI (Relative Strength Index): Momentum oscillator for overbought/oversold conditions
- MACD (Moving Average Convergence Divergence): Trend-following momentum indicator
- Stochastic: Momentum oscillator comparing closing price to price range

Expected Benefits:
- Comprehensive momentum analysis for trading decisions
- Overbought/oversold condition identification
- Trend strength and direction assessment
- Divergence detection for reversal signals
- Multi-timeframe momentum confirmation
"""

from .RSI import (
    RSI,
    RSISignal,
    SmoothingMethod,
    RSIResult
)

from .MACD import (
    MACD,
    MACDSignal,
    MACDResult,
    MACDData
)

from .Stochastic import (
    Stochastic,
    StochasticSignal,
    StochasticType,
    StochasticResult,
    StochasticData
)

__all__ = [
    # RSI components
    'RSI',
    'RSISignal',
    'SmoothingMethod',
    'RSIResult',
    
    # MACD components
    'MACD',
    'MACDSignal',
    'MACDResult',
    'MACDData',
    
    # Stochastic components
    'Stochastic',
    'StochasticSignal',
    'StochasticType',
    'StochasticResult',
    'StochasticData'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Comprehensive momentum indicators for technical analysis and trading signals"
