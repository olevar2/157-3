from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time
"""
Walk-Forward Optimization Module
Advanced optimization techniques for trading strategy validation and robustness testing.

This module provides comprehensive walk-forward optimization capabilities to prevent
overfitting and ensure strategy robustness across different market conditions.

Key Features:
- Walk-forward optimization with rolling windows
- Overfitting detection and prevention
- Parameter optimization across time periods
- Out-of-sample validation
- Robust performance metrics
- Statistical significance testing

Components:
- WalkForwardOptimizer: Main optimization engine
- OverfitDetector: Overfitting detection and prevention
- OptimizationResult: Results container
- ParameterSpace: Parameter space definition
- ValidationMetrics: Performance validation metrics

Expected Benefits:
- Prevention of strategy overfitting through walk-forward analysis
- Robust parameter optimization across different market periods
- Enhanced strategy validation and reliability
- Automated overfitting detection and prevention

Author: Platform3 Analytics Team
Version: 1.0.0
"""

from .WalkForwardOptimizer import (
    WalkForwardOptimizer,
    OptimizationResult,
    ParameterSpace,
    OptimizationConfig,
    WalkForwardWindow,
    OptimizationMetrics
)

from .OverfitDetector import (
    OverfitDetector,
    OverfitResult,
    OverfitMetrics,
    OverfitStatus,
    StatisticalTest,
    RobustnessScore
)

__all__ = [
    # Main optimization classes
    'WalkForwardOptimizer',
    'OverfitDetector',
    
    # Optimization components
    'OptimizationResult',
    'ParameterSpace',
    'OptimizationConfig',
    'WalkForwardWindow',
    'OptimizationMetrics',
    
    # Overfitting components
    'OverfitResult',
    'OverfitMetrics',
    'OverfitStatus',
    'StatisticalTest',
    'RobustnessScore'
]

__version__ = "1.0.0"
__author__ = "Platform3 Analytics Team"
__description__ = "Walk-forward optimization and overfitting prevention for trading strategies"
