#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Pattern Engine for Japanese Candlestick and Chart Pattern Recognition
Platform3 Trading Engine - Pattern Detection Infrastructure
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import the base indicator
from .indicator_base import IndicatorBase, IndicatorSignal, SignalType, IndicatorType


class PatternType(Enum):
    """Classification of pattern types"""
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    INDECISION = "indecision"
    BREAKOUT = "breakout"


class PatternStrength(Enum):
    """Pattern strength classification"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class PatternSignal(IndicatorSignal):
    """Extended signal structure for pattern-specific data"""
    pattern_type: str = "unknown"
    pattern_strength: PatternStrength = PatternStrength.MODERATE
    candles_involved: List[Dict[str, Any]] = field(default_factory=list)
    confirmation_level: float = 0.5  # 0.0 to 1.0
    reliability: float = 0.5  # 0.0 to 1.0


class BasePatternEngine(IndicatorBase):
    """
    Base class for all pattern recognition engines providing:
    - Standardized pattern detection interface
    - Signal generation for pattern-based trading
    - Pattern validation and confirmation logic
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize BasePatternEngine with pattern-specific configuration"""
        super().__init__(config)
        
        # Pattern-specific configuration
        # Ensure config is not None before accessing get method
        cfg = config if config is not None else {}
        self.min_confirmation_level = cfg.get('min_confirmation_level', 0.7)
        self.require_volume_confirmation = cfg.get('require_volume_confirmation', False)
        self.lookback_periods = cfg.get('lookback_periods', 20)
        
        # Pattern tracking
        self.detected_patterns = []
        self.pattern_history = []
        
        self.logger.info("BasePatternEngine initialized successfully")

    def reset(self):
        """Reset internal state of the pattern engine."""
        super().reset() # Call parent's reset if it exists and is needed
        self.detected_patterns = []
        self.pattern_history = []
        # Reset any other pattern-specific state
        # self.logger.debug(f"{self.name} reset.")

    def _perform_calculation(self, data: Union[List[Dict[str, Any]], Any]) -> Any:
        """Perform pattern detection calculation"""
        # Convert DataFrame to list of dictionaries if needed
        if hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')) and hasattr(data, 'empty'): # Check if it's DataFrame-like
            if data.empty:
                self.logger.warning(f"{self.config.get('name', 'UnnamedPatternEngine')}: Input DataFrame is empty.")
                return {
                    'patterns': [],
                    'signals': [],
                    'pattern_count': 0,
                    'error': 'empty_input_dataframe'
                }
            data_dict = data.to_dict('records')
        elif isinstance(data, list):
            if not data:
                self.logger.warning(f"{self.config.get('name', 'UnnamedPatternEngine')}: Input data list is empty.")
                return {
                    'patterns': [],
                    'signals': [],
                    'pattern_count': 0,
                    'error': 'empty_input_list'
                }
            data_dict = data
        else:
            self.logger.error(f"Invalid input data format for pattern calculation: {type(data)}")
            raise ValueError("Invalid input data format for pattern calculation")
        
        if not self.validate_data(data_dict): # validate_data should handle empty list if that's invalid
            self.logger.error("Invalid input data for pattern calculation after initial conversion.")
            raise ValueError("Invalid input data for pattern calculation")
        
        # Detect patterns in the data
        patterns = self.detect_patterns(data_dict)
        
        # Generate signals from detected patterns
        signals = []
        for pattern in patterns:
            signal = self.generate_pattern_signal(pattern, data_dict)
            if signal:
                signals.append(signal)
        
        return {
            'patterns': patterns,
            'signals': signals,
            'pattern_count': len(patterns)
        }
    
    @abstractmethod
    def detect_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in the provided data - to be implemented by subclasses"""
        pass
    
    def generate_pattern_signal(self, pattern: Dict[str, Any], data: List[Dict[str, Any]]) -> Optional[PatternSignal]:
        """Generate trading signal from detected pattern"""
        # Default implementation - can be overridden by subclasses
        if not pattern or pattern.get('confidence', 0) < self.min_confirmation_level:
            return None
        
        # Determine signal type based on pattern
        signal_type = self._determine_signal_type(pattern)
          # Calculate entry price and risk management levels
        entry_price = pattern.get('entry_price', data[-1]['close'])
        stop_loss = pattern.get('stop_loss')
        take_profit = pattern.get('take_profit')
        
        return PatternSignal(
            timestamp=datetime.now(),
            indicator_name=self.__class__.__name__,
            signal_type=signal_type,
            strength=pattern.get('strength', 0.5),
            confidence=pattern.get('confidence', 0.5),
            price_target=take_profit,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_type=pattern.get('pattern_type', 'unknown'),
            pattern_strength=PatternStrength(pattern.get('pattern_strength', 'moderate')),
            candles_involved=pattern.get('candles_involved', []),
            confirmation_level=pattern.get('confidence', 0.5),
            reliability=pattern.get('reliability', 0.5)
        )
    
    def _determine_signal_type(self, pattern: Dict[str, Any]) -> SignalType:
        """Determine signal type based on pattern characteristics"""
        pattern_type = pattern.get('pattern_type', '').lower()
        
        if 'bullish' in pattern_type or 'buy' in pattern_type:
            return SignalType.BUY
        elif 'bearish' in pattern_type or 'sell' in pattern_type:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL
    
    def validate_pattern(self, pattern: Dict[str, Any], data: List[Dict[str, Any]]) -> bool:
        """Validate detected pattern against market context"""
        # Basic validation - can be extended by subclasses
        if not pattern:
            return False
        
        # Check confidence level
        confidence = pattern.get('confidence', 0)
        if confidence < self.min_confirmation_level:
            return False
        
        # Volume confirmation if required
        if self.require_volume_confirmation:
            volume_confirmed = pattern.get('volume_confirmed', False)
            if not volume_confirmed:
                return False
        
        return True
    
    def calculate_pattern_reliability(self, pattern: Dict[str, Any], data: List[Dict[str, Any]]) -> float:
        """Calculate pattern reliability based on market context"""
        # Basic reliability calculation - can be enhanced by subclasses
        base_reliability = pattern.get('confidence', 0.5)
        
        # Adjust based on market volatility
        if len(data) >= 5:
            recent_volatility = self._calculate_volatility(data[-5:])
            if recent_volatility > 0.03:  # High volatility
                base_reliability *= 0.9
        
        # Adjust based on volume
        if pattern.get('volume_confirmed', False):
            base_reliability *= 1.1
        
        return min(base_reliability, 1.0)
    
    def _calculate_volatility(self, data: List[Dict[str, Any]]) -> float:
        """Calculate simple volatility measure"""
        if len(data) < 2:
            return 0
        
        prices = [candle['close'] for candle in data]
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        return np.std(returns) if returns else 0
    
    def get_pattern_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get pattern detection history"""
        if limit:
            return self.pattern_history[-limit:]
        return self.pattern_history.copy()
    
    def is_doji(self, candle: Dict[str, Any], max_body_ratio: float = 0.1) -> bool:
        """Helper method to identify doji candles"""
        body = abs(candle['close'] - candle['open'])
        range_size = candle['high'] - candle['low']
        
        if range_size == 0:
            return True
        
        body_ratio = body / range_size
        return body_ratio <= max_body_ratio
    
    def calculate_body_size(self, candle: Dict[str, Any]) -> float:
        """Calculate candle body size"""
        return abs(candle['close'] - candle['open'])
    
    def calculate_upper_shadow(self, candle: Dict[str, Any]) -> float:
        """Calculate upper shadow size"""
        return candle['high'] - max(candle['open'], candle['close'])
    
    def calculate_lower_shadow(self, candle: Dict[str, Any]) -> float:
        """Calculate lower shadow size"""
        return min(candle['open'], candle['close']) - candle['low']
    
    def is_bullish_candle(self, candle: Dict[str, Any]) -> bool:
        """Check if candle is bullish"""
        return candle['close'] > candle['open']
    
    def is_bearish_candle(self, candle: Dict[str, Any]) -> bool:
        """Check if candle is bearish"""
        return candle['close'] < candle['open']


# Export for use by other modules
__all__ = [
    'BasePatternEngine', 'PatternSignal', 'PatternType', 'PatternStrength'
]
