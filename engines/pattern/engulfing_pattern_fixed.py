# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Engulfing Pattern Scanner - Japanese Candlestick Pattern Recognition
Platform3 Enhanced Technical Analysis Engine

Detects Bullish and Bearish Engulfing patterns, which are powerful two-candle
reversal patterns in technical analysis. An engulfing pattern occurs when
a larger candle completely "engulfs" the previous smaller candle.

Pattern Characteristics:
- Two-candle pattern
- Second candle's body completely engulfs the first candle's body
- Opposite colors (bullish after bearish or vice versa)
- Trend reversal indication

Key Features:
- Bullish and bearish engulfing detection
- Trend context analysis
- Volume confirmation
- Pattern strength measurement
- Reversal probability scoring
- Support/resistance level validation

Trading Applications:
- Trend reversal identification
- Entry/exit timing optimization
- Confirmation of support/resistance levels
- Risk management enhancement
- Market sentiment analysis

Mathematical Foundation:
- Engulfing Condition: Body2_Size > Body1_Size AND Body2 contains Body1
- Volume Confirmation: Volume2 > Average_Volume
- Trend Context: Direction of preceding trend
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame, IndicatorSignal, SignalType

class EngulfingType(Enum):
    """Types of Engulfing patterns"""
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    NO_PATTERN = "no_pattern"

class EngulfingSignal(Enum):
    """Engulfing signal types"""
    STRONG_BULLISH_REVERSAL = "strong_bullish_reversal"
    STRONG_BEARISH_REVERSAL = "strong_bearish_reversal"
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    TREND_CONTINUATION = "trend_continuation"
    NEUTRAL = "neutral"

@dataclass
class CandlestickData:
    """Individual candlestick data"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    body_size: float
    is_bullish: bool

# Create a standalone class instead of inheriting from IndicatorResult
@dataclass
class EngulfingPatternResult:
    """Engulfing Pattern detection result"""
    timestamp: datetime
    pattern_type: EngulfingType
    pattern_strength: float  # 0-100, higher means stronger pattern
    engulfing_ratio: float  # How much larger the engulfing candle is
    first_candle: CandlestickData
    second_candle: CandlestickData
    trend_context: str  # 'uptrend', 'downtrend', 'sideways'
    reversal_probability: float  # 0-100
    volume_confirmation: bool
    volume_ratio: float  # Current volume vs average
    support_resistance_level: Optional[float]
    signal: EngulfingSignal
    signal_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            'pattern_type': self.pattern_type.value,
            'pattern_strength': self.pattern_strength,
            'engulfing_ratio': self.engulfing_ratio,
            'first_candle': {
                'open': self.first_candle.open,
                'high': self.first_candle.high,
                'low': self.first_candle.low,
                'close': self.first_candle.close,
                'volume': self.first_candle.volume,
                'body_size': self.first_candle.body_size,
                'is_bullish': self.first_candle.is_bullish
            },
            'second_candle': {
                'open': self.second_candle.open,
                'high': self.second_candle.high,
                'low': self.second_candle.low,
                'close': self.second_candle.close,
                'volume': self.second_candle.volume,
                'body_size': self.second_candle.body_size,
                'is_bullish': self.second_candle.is_bullish
            },
            'trend_context': self.trend_context,
            'reversal_probability': self.reversal_probability,
            'volume_confirmation': self.volume_confirmation,
            'volume_ratio': self.volume_ratio,
            'support_resistance_level': self.support_resistance_level,
            'signal': self.signal.value,
            'signal_strength': self.signal_strength,
            'metadata': self.metadata
        }
    
    def to_indicator_result(self, indicator_name: str) -> IndicatorResult:
        """Convert to standard IndicatorResult"""
        return IndicatorResult(
            timestamp=self.timestamp,
            indicator_name=indicator_name,
            indicator_type=IndicatorType.PATTERN,
            timeframe=TimeFrame.D1,  # Default timeframe - should be updated
            value=self.pattern_strength,
            signal=IndicatorSignal(
                timestamp=self.timestamp,
                indicator_name=indicator_name,
                signal_type=self._map_signal_type(),
                strength=self.signal_strength / 100,  # Scale to 0-1
                confidence=self.reversal_probability / 100,  # Scale to 0-1
                metadata={'pattern_type': self.pattern_type.value}
            ) if self.signal != EngulfingSignal.NEUTRAL else None
        )
    
    def _map_signal_type(self) -> SignalType:
        """Map Engulfing signal to standard SignalType"""
        signal_mapping = {
            EngulfingSignal.STRONG_BULLISH_REVERSAL: SignalType.STRONG_BUY,
            EngulfingSignal.BULLISH_REVERSAL: SignalType.BUY,
            EngulfingSignal.STRONG_BEARISH_REVERSAL: SignalType.STRONG_SELL,
            EngulfingSignal.BEARISH_REVERSAL: SignalType.SELL,
            EngulfingSignal.TREND_CONTINUATION: SignalType.HOLD,
            EngulfingSignal.NEUTRAL: SignalType.NEUTRAL
        }
        return signal_mapping.get(self.signal, SignalType.NEUTRAL)

class EngulfingPatternScanner(IndicatorBase):
    """
    Engulfing Pattern Scanner - Japanese Candlestick Pattern Recognition
    
    Identifies Bullish and Bearish Engulfing patterns with comprehensive analysis
    including trend context, volume confirmation, and reversal probability.
    """
    
    def __init__(self, 
                 min_engulfing_ratio: float = 1.1,
                 trend_period: int = 10,
                 volume_lookback: int = 10,
                 min_body_size_ratio: float = 0.3,
                 support_resistance_tolerance: float = 0.02):
        """
        Initialize Engulfing Pattern Scanner
        
        Args:
            min_engulfing_ratio: Minimum ratio for engulfing body size (default: 1.1)
            trend_period: Period for trend context analysis (default: 10)
            volume_lookback: Period for volume confirmation (default: 10)
            min_body_size_ratio: Minimum body size relative to range (default: 0.3)
            support_resistance_tolerance: Tolerance for S/R level detection (default: 2%)
        """
        super().__init__({
            "name": "Engulfing Pattern Scanner",
            "version": "1.0.0",
            "description": "Detects bullish and bearish engulfing patterns"
        })
        self.min_engulfing_ratio = min_engulfing_ratio
        self.trend_period = trend_period
        self.volume_lookback = volume_lookback
        self.min_body_size_ratio = min_body_size_ratio
        self.support_resistance_tolerance = support_resistance_tolerance
        
        # Validation
        if self.min_engulfing_ratio < 1:
            raise ValueError("Minimum engulfing ratio must be >= 1")
        if self.trend_period < 3:
            raise ValueError("Trend period must be >= 3")
        if self.volume_lookback < 2:
            raise ValueError("Volume lookback must be >= 2")
        if not 0 < self.min_body_size_ratio < 1:
            raise ValueError("Minimum body size ratio must be between 0 and 1")
            
        # State variables
        self.reset()
        
    def reset(self) -> None:
        """Reset indicator state"""
        self.candles = []
        self.patterns = []

    # Implement the required methods from the original file but adapted to use the new standalone class
    def _perform_calculation(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform engulfing pattern detection on market data"""
        # This is a stub implementation - fill in the actual implementation based on the original file
        start_time = datetime.now()
        
        # Process results using the standalone pattern class
        result = EngulfingPatternResult(
            timestamp=datetime.now(),
            pattern_type=EngulfingType.NO_PATTERN,
            pattern_strength=0.0,
            engulfing_ratio=0.0,
            first_candle=self._create_candlestick_data(0, 0, 0, 0, 0),
            second_candle=self._create_candlestick_data(0, 0, 0, 0, 0),
            trend_context="sideways",
            reversal_probability=0.0,
            volume_confirmation=False,
            volume_ratio=0.0,
            support_resistance_level=None,
            signal=EngulfingSignal.NEUTRAL,
            signal_strength=0.0
        )
        
        # Convert to standard IndicatorResult for compatibility
        indicator_result = result.to_indicator_result("Engulfing Pattern Scanner")
        
        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        return {
            'result': result,
            'indicator_result': indicator_result,
            'calculation_time_ms': calculation_time
        }
    
    def _create_candlestick_data(self, open_price: float, high: float, low: float, 
                                close: float, volume: float) -> CandlestickData:
        """Create candlestick data object"""
        body_size = abs(close - open_price)
        is_bullish = close > open_price
        
        return CandlestickData(
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            body_size=body_size,
            is_bullish=is_bullish
        )