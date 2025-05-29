"""
Platform3 Advanced Trading Engines
===================================
Sophisticated analysis engines for maximum accuracy trading platform.

For Humanitarian Profit Generation - Using Mathematical Precision
"""

import sys
import os

# Add the actual analytics service path
current_dir = os.path.dirname(os.path.abspath(__file__))
analytics_path = os.path.join(current_dir, '..', 'services', 'analytics-service', 'src')
if analytics_path not in sys.path:
    sys.path.insert(0, analytics_path)

__version__ = "3.0.0"
__author__ = "Platform3 AI Team"
__purpose__ = "Humanitarian Profit Generation Through Mathematical Precision"

# Import from actual implementations
try:
    from engines.gann.GannAnglesCalculator import GannAnglesCalculator
    from engines.gann.GannSquareOfNine import GannSquareOfNine
    from engines.gann.GannFanAnalysis import GannFanAnalysis
    from engines.gann.GannTimePrice import GannTimePrice
    from engines.gann.GannPatternDetector import GannPatternDetector
except ImportError as e:
    print(f"Warning: Gann indicators not available: {e}")

try:
    from engines.fibonacci.FibonacciRetracement import FibonacciRetracement
    from engines.fibonacci.FibonacciExtension import FibonacciExtension
    from engines.fibonacci.ConfluenceDetector import ConfluenceDetector
    from engines.fibonacci.TimeZoneAnalysis import TimeZoneAnalysis
    from engines.fibonacci.ProjectionArcCalculator import ProjectionArcCalculator
except ImportError as e:
    print(f"Warning: Fibonacci indicators not available: {e}")

try:
    from engines.indicators.momentum.RSI import RSI
    from engines.indicators.momentum.MACD import MACD
    from engines.indicators.momentum.Stochastic import Stochastic
except ImportError as e:
    print(f"Warning: Technical indicators not available: {e}")

__all__ = [
    # Gann Analysis
    'GannAnglesCalculator',
    'GannSquareOfNine', 
    'GannFanAnalysis',
    'GannTimePrice',
    'GannPatternDetector',
    
    # Fibonacci Analysis
    'FibonacciRetracement',
    'FibonacciExtension',
    'ConfluenceDetector',
    'TimeZoneAnalysis',
    'ProjectionArcCalculator',
    
    # Technical Indicators
    'RSI',
    'MACD',
    'Stochastic',
]
