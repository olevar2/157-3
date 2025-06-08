# -*- coding: utf-8 -*-
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
AI Enhancement Module Initialization

This module provides AI-powered enhancements to traditional technical analysis,
including adaptive indicators, machine learning signal generation, market
microstructure analysis, and intelligent pattern recognition.

Key Components:
- AdaptiveIndicators: Self-adjusting parameter indicators
- MLSignalGenerator: Machine learning signal classification
- MarketMicrostructureAnalysis: Order flow and depth analysis
- SentimentIntegration: News/social sentiment incorporation
- MultiAssetCorrelation: Cross-market relationship analysis
- RegimeDetectionAI: AI-powered market regime identification
- PatternRecognitionAI: Deep learning pattern detection
- RiskAssessmentAI: AI-driven risk measurement
- SignalConfidenceAI: AI confidence scoring for signals

Author: Platform3 Trading System
Version: 1.0.0 - AI Enhancement Suite
"""

from .adaptive_indicators import AdaptiveIndicators
from .ml_signal_generator import MLSignalGenerator
from .market_microstructure_analysis import MarketMicrostructureAnalysis
from .sentiment_integration import SentimentIntegration
from .multi_asset_correlation import MultiAssetCorrelation
from .regime_detection_ai import RegimeDetectionAI
from .pattern_recognition_ai import PatternRecognitionAI
from .risk_assessment_ai import RiskAssessmentAI
from .signal_confidence_ai import SignalConfidenceAI

# NEW: Enhanced Integration Layer for Genius Agents
from .adaptive_indicator_bridge import (
    AdaptiveIndicatorBridge, 
    adaptive_indicator_bridge,
    GeniusAgentType,
    IndicatorPackage
)
from .adaptive_indicator_coordinator import AdaptiveIndicatorCoordinator

class IndicatorGeniusBridge:
    """Bridge between indicators and genius agents for seamless integration"""
    
    def __init__(self):
        self.adaptive_coordinator = AdaptiveIndicatorCoordinator()
        self.genius_indicator_cache = {}
        
    async def prepare_indicators_for_genius(self, market_data, genius_type):
        """Prepare optimized indicator data for specific genius agents"""
        if genius_type == 'risk_genius':
            return await self._prepare_risk_indicators(market_data)
        elif genius_type == 'session_expert':
            return await self._prepare_session_indicators(market_data)
        elif genius_type == 'pattern_master':
            return await self._prepare_pattern_indicators(market_data)
        # ...other genius types...
        
    async def _prepare_risk_indicators(self, market_data):
        """Prepare risk-focused indicators for Risk Genius"""
        return {
            'volatility_indicators': self._get_volatility_suite(market_data),
            'momentum_indicators': self._get_momentum_suite(market_data),
            'trend_strength': self._get_trend_strength_suite(market_data),
            'risk_metrics': self._calculate_risk_metrics(market_data)
        }

__all__ = [
    'AdaptiveIndicators',
    'MLSignalGenerator',
    'MarketMicrostructureAnalysis',
    'SentimentIntegration',
    'MultiAssetCorrelation',
    'RegimeDetectionAI',
    'PatternRecognitionAI',
    'RiskAssessmentAI',
    'SignalConfidenceAI',
    'AdaptiveIndicatorBridge',
    'adaptive_indicator_bridge',
    'GeniusAgentType',
    'IndicatorPackage',
    'AdaptiveIndicatorCoordinator'
]
