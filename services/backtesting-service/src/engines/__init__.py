"""
High-Frequency Backtesting Engine Package
Ultra-fast backtesting for scalping, day trading, and swing strategies.

This package provides comprehensive backtesting capabilities for short-term
trading strategies with tick-accurate simulation and realistic execution modeling.

Components:
- ScalpingBacktester: M1-M5 tick-accurate backtesting
- DayTradingBacktester: M15-H1 session-based backtesting  
- SwingBacktester: H4 short-term swing backtesting
- MultiTimeframeBacktester: M1-H4 strategy combinations
- SpeedOptimizedEngine: Vectorized backtesting for speed
"""

from .ScalpingBacktester import ScalpingBacktester, TickData, BacktestResult
from .DayTradingBacktester import (
    DayTradingBacktester, 
    TradingSession, 
    SessionData, 
    DayTradingResult
)
from .SwingBacktester import (
    SwingBacktester,
    SwingPattern,
    SwingSetup,
    SwingBacktestResult
)

__all__ = [
    # Main backtesting engines
    'ScalpingBacktester',
    'DayTradingBacktester', 
    'SwingBacktester',
    
    # Scalping components
    'TickData',
    'BacktestResult',
    
    # Day trading components
    'TradingSession',
    'SessionData',
    'DayTradingResult',
    
    # Swing trading components
    'SwingPattern',
    'SwingSetup',
    'SwingBacktestResult'
]

__version__ = "1.0.0"
__author__ = "Platform3 Backtesting Team"
__description__ = "High-Frequency Backtesting Engine for Short-Term Trading Strategies"
