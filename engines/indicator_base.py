#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Indicator Base System with Signal Support
Platform3 Trading Engine - Complete indicator infrastructure
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Generic, TypeVar
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Platform3 imports with fallback handling
try:
    from shared.logging.platform3_logger import Platform3Logger
    from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
    from shared.database.platform3_database_manager import Platform3DatabaseManager
    from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
except ImportError:
    # Fallback implementations for missing Platform3 components
    import logging
    
    class Platform3Logger:
        def __init__(self, name):
            self.logger = logging.getLogger(name)
            logging.basicConfig(level=logging.INFO)
        def info(self, msg): print(f"[INFO] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
    
    class ServiceError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.message = message
        def to_dict(self):
            return {"error": self.message}
    
    class Platform3ErrorSystem:
        def handle_error(self, error):
            print(f"[ERROR SYSTEM] {error}")    
    class Platform3DatabaseManager:
        def __init__(self):
            pass
    
    class Platform3CommunicationFramework:
        def __init__(self):
            pass

# Generic type for indicator configuration
ConfigType = TypeVar('ConfigType')

class IndicatorType(Enum):
    """Classification of indicator types for organization and analysis."""
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    CYCLE = "cycle"
    PATTERN = "pattern"
    FIBONACCI = "fibonacci"
    GANN = "gann"
    ELLIOTT_WAVE = "elliott_wave"
    FRACTAL = "fractal"
    DIVERGENCE = "divergence"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "ml"
    COMPOSITE = "composite"

class SignalType(Enum):
    """Standard signal types for all indicators."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    WARNING = "warning"

class TimeFrame(Enum):
    """Supported timeframes for multi-timeframe analysis."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

class IndicatorStatus(Enum):
    """Status tracking for indicator health monitoring."""
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"
    CALCULATING = "calculating"
    NO_DATA = "no_data"

class SignalStrength(Enum):
    """Signal strength classification for trading decisions."""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    NEUTRAL = "neutral"

class MarketCondition(Enum):
    """Market condition classification for context-aware trading."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    REVERSAL = "reversal"

@dataclass
class MarketData:
    """Standardized market data structure for all indicators."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: TimeFrame

@dataclass
class IndicatorSignal:
    """Standardized signal output for all indicators."""
    timestamp: datetime
    indicator_name: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class IndicatorResult:
    """Standardized result structure for all indicators."""
    timestamp: datetime
    indicator_name: str
    indicator_type: IndicatorType
    timeframe: TimeFrame
    value: Union[float, Dict[str, float], List[float]]
    signal: Optional[IndicatorSignal] = None
    raw_data: Optional[Dict[str, Any]] = None
    calculation_time_ms: Optional[float] = None

@dataclass
class IndicatorConfig:
    """Configuration structure for all indicators."""
    name: str
    indicator_type: IndicatorType
    timeframe: TimeFrame
    lookback_periods: int = 20
    parameters: Optional[Dict[str, Any]] = None
    enabled: bool = True

class TechnicalIndicator(ABC):
    """Abstract base class for all technical indicators."""
    
    def __init__(self, config: IndicatorConfig):
        self.config = config
        self.status = IndicatorStatus.ACTIVE
        self.logger = Platform3Logger(f'TechnicalIndicator.{config.name}')
        
    @abstractmethod
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """Calculate indicator values from market data."""
        pass
        
    @abstractmethod
    def generate_signal(self, data: List[MarketData]) -> Optional[IndicatorSignal]:
        """Generate trading signal from indicator values."""
        pass

class IndicatorBase(Generic[ConfigType]):
    """
    Base class for all Platform3 indicators providing:
    - Standardized interface
    - Performance monitoring 
    - Signal generation support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize IndicatorBase with Platform3 framework components"""
        self.config = config or {}
        
        # Initialize Platform3 framework components
        self.logger = Platform3Logger('IndicatorBase')
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
          # Performance monitoring
        self.calculation_times = []
        self.last_calculation = None
        # Signal tracking
        self.last_signal = None
        self.signal_history = []
        
        self.logger.info("IndicatorBase initialized successfully")
    
    def _validate_data(self, data: Union[List[Dict[str, Any]], pd.DataFrame], *args) -> bool:
        """
        Internal method to validate data before calculation.
        Subclass indicators use this method to ensure data is properly formatted.
        
        Args:
            data: List of market data dictionaries or pandas DataFrame
            *args: Additional arguments for backward compatibility
            
        Returns:
            bool: True if data is valid, False otherwise
        
        Raises:
            ServiceError: If data has critical formatting issues
        """
        import pandas as pd
        
        # Handle pandas DataFrame input
        if isinstance(data, pd.DataFrame):
            # Check for empty DataFrame
            if data.empty:
                self.logger.warning("Input DataFrame is empty.")
                return False
                
            # Check required columns
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_fields = [field for field in required_fields if field not in data.columns]
            
            if missing_fields:
                self.logger.warning(f"DataFrame missing required columns: {missing_fields}")
                return False
                
            return True
            
        # Handle list input
        if isinstance(data, list):
            # Check for empty list
            if not data: # More Pythonic way to check for an empty list
                self.logger.warning("Input data list is empty.")
                return False
        else: # Not a list or DataFrame
            self.logger.warning(f"Input data must be a list or pandas DataFrame, got {type(data)}.")
            return False
            
        # Verify each data point in the list has required fields
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Allow validation to skip some items if there are many
        validation_limit = min(len(data), 100)  # Validate at most 100 items for performance
        
        for i, item in enumerate(data):
            if i >= validation_limit:
                break
                
            if not isinstance(item, dict):
                self.logger.warning(f"Item at index {i} is not a dictionary: {item}")
                return False
                
            # Check required fields exist
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                self.logger.warning(f"Data item at index {i} missing required fields: {missing_fields}. Item: {item}")
                return False
                
            # Verify numeric types where appropriate
            numeric_fields = ['open', 'high', 'low', 'close', 'volume']
            for field in numeric_fields:
                if field in item and not isinstance(item[field], (int, float, np.number)): # Added np.number for numpy numeric types
                    try:
                        # Attempt to convert to float if not already numeric, to handle strings that are valid numbers
                        item[field] = float(item[field])
                    except (ValueError, TypeError):
                        self.logger.warning(f"Field '{field}' in item at index {i} must be numeric or convertible to numeric, got {type(item[field])} with value {item[field]}. Item: {item}")
                        return False
        
        return True
    
    def calculate(self, data: Union[List[Dict[str, Any]], pd.DataFrame]) -> Dict[str, Any]:
        """Calculate trading engine values with enhanced accuracy"""
        start_time = datetime.now()
        
        try:
            # Validate input data
            if not data:
                raise ServiceError("Input data cannot be empty")
                
            if not isinstance(data, list):
                raise ServiceError("Input data must be a list")
            
            # Call internal validate method that subclasses inherit
            if not self._validate_data(data):
                raise ServiceError("Input data validation failed")
                  # Perform calculation (to be implemented by subclasses)
            result = self._perform_calculation(data)
            
            # Track performance
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.calculation_times.append(calculation_time)
            self.last_calculation = datetime.now()
            
            self.logger.info(f"Calculation completed in {calculation_time:.4f}s")
            
            return {
                'success': True,
                'data': result,
                'timestamp': self.last_calculation.isoformat(),
                'calculation_time': calculation_time
            }
            
        except ServiceError as e:
            self.logger.error(f"Service error: {e}")
            self.error_system.handle_error(e)
            raise
        except ValueError as e:
            # Centralized handling for insufficient data errors
            if "Insufficient data" in str(e) or "need at least" in str(e):
                self.logger.warning(f"{self.__class__.__name__}: {e}")
                return {
                    'success': False,
                    'data': None,
                    'timestamp': datetime.now().isoformat(),
                    'error': 'insufficient_data',
                    'message': str(e)
                }
            # Re-raise other ValueError exceptions
            error = ServiceError(f"Calculation failed: {str(e)}")
            self.logger.error(f"Value error: {e}")
            self.error_system.handle_error(error)
            raise error
        except Exception as e:
            error = ServiceError(f"Calculation failed: {str(e)}")
            self.logger.error(f"Unexpected error: {e}")
            self.error_system.handle_error(error)
            raise error
    
    def _perform_calculation(self, data: List[Dict[str, Any]]) -> Any:
        """Perform the actual indicator calculation - override in subclasses"""
        raise NotImplementedError("Subclasses must implement _perform_calculation")
    
    def generate_signal(self, data: List[Dict[str, Any]]) -> Optional[IndicatorSignal]:
        """Generate trading signal based on indicator values"""
        # Default implementation - to be overridden by subclasses
        return None
    
    async def calculate_async(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Asynchronous version of calculate for high-frequency trading"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.calculate, data
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the indicator"""
        if not self.calculation_times:
            return {"status": "no_calculations"}
            
        return {
            "total_calculations": len(self.calculation_times),
            "average_time": np.mean(self.calculation_times),
            "min_time": np.min(self.calculation_times),
            "max_time": np.max(self.calculation_times),
            "last_calculation": self.last_calculation.isoformat() if self.last_calculation else None,
            "signal_count": len(self.signal_history)
        }
    
    def reset_performance_metrics(self):
        """Reset performance tracking metrics"""
        self.calculation_times = []
        self.last_calculation = None
        self.signal_history = []
        self.last_signal = None
        self.logger.info("Performance metrics reset")
    
    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """Validate input data format and content"""
        if not data or not isinstance(data, list):
            return False
            
        # Check for required fields in each data point
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        for item in data:
            if not isinstance(item, dict):
                return False
                
            for field in required_fields:
                if field not in item:
                    return False
                    
        return True
    
    def get_signal_history(self, limit: Optional[int] = None) -> List[IndicatorSignal]:
        """Get signal history"""
        if limit:
            return self.signal_history[-limit:]
        return self.signal_history.copy()
    
    def reset(self):
        """Reset indicator internal state"""
        self.calculation_times = []
        self.last_calculation = None
        self.signal_history = []
        self.last_signal = None
        self.logger.info(f"{self.__class__.__name__} reset successfully")
    
    def __str__(self) -> str:
        """String representation of the indicator"""
        return f"IndicatorBase(calculations={len(self.calculation_times)}, signals={len(self.signal_history)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the indicator"""
        return f"IndicatorBase(config={self.config}, calculations={len(self.calculation_times)}, signals={len(self.signal_history)})"

# Export for use by other modules
__all__ = [
    'IndicatorBase', 'TechnicalIndicator', 'IndicatorConfig', 'IndicatorResult',
    'IndicatorSignal', 'MarketData', 'SignalType', 'IndicatorType', 
    'TimeFrame', 'IndicatorStatus'
]
