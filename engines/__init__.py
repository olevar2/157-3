from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import (
    Platform3ErrorSystem,
    ServiceError,
)
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import (
    Platform3CommunicationFramework,
)
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time

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
analytics_path = os.path.join(current_dir, "..", "services", "analytics-service", "src")
if analytics_path not in sys.path:
    sys.path.insert(0, analytics_path)

__version__ = "3.0.0"
__author__ = "Platform3 AI Team"
__purpose__ = "Humanitarian Profit Generation Through Mathematical Precision"

# Import from actual implementations
try:
    from engines.gann.GannAnglesCalculator import GannAnglesCalculator
    from engines.gann.gann_square_of_nine import GannSquareOfNine

    # Commented out missing module: from engines.gann.GannFanAnalysis import GannFanAnalysis
    from engines.gann.gann_fan_lines import (
        GannFanAnalysis,
    )  # Use the class from gann_fan_lines.py
    from engines.gann.GannTimePrice import GannTimePrice
    from engines.gann.GannPatternDetector import GannPatternDetector
except ImportError as e:
    print(f"Warning: Gann indicators not available: {e}")

try:
    from engines.fibonacci.FibonacciRetracement import FibonacciRetracement
    from engines.fibonacci.FibonacciExtension import FibonacciExtension
    from engines.fibonacci.FibonacciFan import FibonacciFanIndicator
    from engines.fibonacci.ConfluenceDetector import ConfluenceDetector
    from engines.fibonacci.TimeZoneAnalysis import TimeZoneAnalysis
    from engines.fibonacci.ProjectionArcCalculator import ProjectionArcCalculator
except ImportError as e:
    print(f"Warning: Fibonacci indicators not available: {e}")

try:
    from engines.ml_advanced.neural_network_predictor import NeuralNetworkPredictor
    from engines.ml_advanced.genetic_algorithm_optimizer import (
        GeneticAlgorithmOptimizer,
    )
except ImportError as e:
    print(f"Warning: ML Advanced indicators not available: {e}")

try:
    from engines.momentum.rsi import RelativeStrengthIndex as RSI
    from engines.momentum.macd import MovingAverageConvergenceDivergence as MACD
    from engines.momentum.stochastic import StochasticOscillator as Stochastic
    from engines.momentum.correlation_momentum import (
        DynamicCorrelationIndicator,
        RelativeMomentumIndicator,
    )
except ImportError as e:
    print(f"Warning: Technical indicators not available: {e}")

__all__ = [
    # Gann Analysis
    "GannAnglesCalculator",
    "GannSquareOfNine",
    "GannFanAnalysis",
    "GannTimePrice",
    "GannPatternDetector",
    # Fibonacci Analysis
    "FibonacciRetracement",
    "FibonacciExtension",
    "FibonacciFanIndicator",
    "ConfluenceDetector",
    "TimeZoneAnalysis",
    "ProjectionArcCalculator",
    # ML Advanced Analysis
    "NeuralNetworkPredictor",
    "GeneticAlgorithmOptimizer",
    # Technical Indicators
    "RSI",
    "MACD",
    "Stochastic",
    "DynamicCorrelationIndicator",
    "RelativeMomentumIndicator",
]
