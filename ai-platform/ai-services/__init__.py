# -*- coding: utf-8 -*-
"""
Platform3 AI Services Module
Humanitarian Forex Trading Platform - AI Services Integration
Bridges Python indicators with AI agents for autonomous trading decisions
"""

from .ai_coordinator import AICoordinator
from .decision_engine import DecisionEngine
from .strategy_generator import StrategyGenerator
from .risk_manager import RiskManager
from .performance_optimizer import PerformanceOptimizer

__all__ = [
    'AICoordinator',
    'DecisionEngine', 
    'StrategyGenerator',
    'RiskManager',
    'PerformanceOptimizer'
]

__version__ = "1.0.0"
__description__ = "AI Services for Platform3 Humanitarian Trading Platform"