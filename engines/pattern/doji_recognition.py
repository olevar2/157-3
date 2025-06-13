# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Doji Recognition Engine - Advanced Japanese Candlestick Pattern Detection
Platform3 Enhanced Technical Analysis Engine

The Doji Recognition Engine identifies all variations of Doji candlestick patterns,
which indicate market indecision and potential trend reversals.

Doji Types Detected:
- Standard Doji - Open and close are virtually equal
- Long-Legged Doji - Long shadows, small body
- Dragonfly Doji - Long lower shadow, no upper shadow
- Gravestone Doji - Long upper shadow, no lower shadow
- Four Price Doji - Open, high, low, close all equal (rare)

Key Features:
- Multi-type doji pattern recognition
- Trend context analysis
- Volume confirmation
- Reversal probability scoring
- Market indecision measurement
- Support/resistance level integration

Trading Applications:
- Trend reversal identification
- Support/resistance confirmation
- Entry/exit timing optimization
- Risk management enhancement
- Market sentiment analysis

Mathematical Foundation:
- Body Size = |Close - Open|
- Shadow Ratios = Upper Shadow / Lower Shadow
- Doji Threshold = Body Size / Total Range < 0.1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame, IndicatorSignal, SignalType

class DojiType(Enum):
    """Types of Doji patterns"""
    STANDARD_DOJI = "standard_doji"
    LONG_LEGGED_DOJI = "long_legged_doji"
    DRAGONFLY_DOJI = "dragonfly_doji"
    GRAVESTONE_DOJI = "gravestone_doji"
    FOUR_PRICE_DOJI = "four_price_doji"
    NO_DOJI = "no_doji"

class DojiSignal(Enum):
    """Doji signal types"""
    REVERSAL_BULLISH = "reversal_bullish"
    REVERSAL_BEARISH = "reversal_bearish"
    INDECISION = "indecision"
    CONTINUATION = "continuation"
    NEUTRAL = "neutral"

# Create a standalone dataclass instead of inheriting from IndicatorResult
@dataclass
class DojiRecognitionResult:
    """Doji Recognition calculation result"""
    timestamp: datetime
    doji_type: DojiType
    doji_strength: float  # 0-100, higher means stronger pattern
    body_size: float
    upper_shadow: float
    lower_shadow: float
    total_range: float
    body_ratio: float  # Body size / Total range
    shadow_ratio: float  # Upper shadow / Lower shadow
    trend_context: str  # 'uptrend', 'downtrend', 'sideways'
    reversal_probability: float  # 0-100
    volume_confirmation: bool
    signal: DojiSignal
    signal_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            'doji_type': self.doji_type.value,
            'doji_strength': self.doji_strength,
            'body_size': self.body_size,
            'upper_shadow': self.upper_shadow,
            'lower_shadow': self.lower_shadow,
            'total_range': self.total_range,
            'body_ratio': self.body_ratio,
            'shadow_ratio': self.shadow_ratio,
            'trend_context': self.trend_context,
            'reversal_probability': self.reversal_probability,
            'volume_confirmation': self.volume_confirmation,
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
            value=self.doji_strength,
            signal=IndicatorSignal(
                timestamp=self.timestamp,
                indicator_name=indicator_name,
                signal_type=self._map_signal_type(),
                strength=self.signal_strength / 100,  # Scale to 0-1
                confidence=self.reversal_probability / 100,  # Scale to 0-1
                metadata={'doji_type': self.doji_type.value}
            ) if self.signal != DojiSignal.NEUTRAL else None
        )
    
    def _map_signal_type(self) -> SignalType:
        """Map Doji signal to standard SignalType"""
        if self.signal == DojiSignal.REVERSAL_BULLISH:
            return SignalType.BUY
        elif self.signal == DojiSignal.REVERSAL_BEARISH:
            return SignalType.SELL
        elif self.signal == DojiSignal.CONTINUATION:
            return SignalType.HOLD
        else:
            return SignalType.NEUTRAL

class DojiRecognitionEngine(IndicatorBase):
    """
    Doji Recognition Engine - Japanese Candlestick Pattern Detection
    
    Identifies all variations of Doji patterns with trend context analysis
    and reversal probability scoring.
    """
    
    def __init__(self, 
                 doji_threshold: float = 0.1,
                 long_shadow_ratio: float = 2.0,
                 trend_period: int = 10,
                 volume_lookback: int = 5):
        """
        Initialize Doji Recognition Engine
        
        Args:
            doji_threshold: Maximum body ratio for doji identification (default: 0.1)
            long_shadow_ratio: Minimum shadow ratio for long-legged classification (default: 2.0)
            trend_period: Period for trend context analysis (default: 10)
            volume_lookback: Period for volume confirmation (default: 5)
        """
        super().__init__("Doji Recognition Engine")
        self.doji_threshold = doji_threshold
        self.long_shadow_ratio = long_shadow_ratio
        self.trend_period = trend_period
        self.volume_lookback = volume_lookback
        
        # Validation
        if self.doji_threshold <= 0 or self.doji_threshold > 0.5:
            raise ValueError("Doji threshold must be between 0 and 0.5")
        if self.long_shadow_ratio < 1:
            raise ValueError("Long shadow ratio must be >= 1")
        if self.trend_period < 3:
            raise ValueError("Trend period must be >= 3")
        if self.volume_lookback < 1:
            raise ValueError("Volume lookback must be >= 1")
            
        # State variables
        self.reset()
        
    def reset(self) -> None:
        """Reset indicator state"""
        super().reset()
        self.highs = []
        self.lows = []
        self.opens = []
        self.closes = []
        self.volumes = []
        self.doji_patterns = []
        
    def _calculate_candle_components(self, open_price: float, high: float, 
                                   low: float, close: float) -> Tuple[float, float, float, float]:
        """Calculate candlestick components"""
        body_size = abs(close - open_price)
        total_range = high - low
        
        # Calculate shadows
        if close >= open_price:  # Bullish candle
            upper_shadow = high - close
            lower_shadow = open_price - low
        else:  # Bearish candle
            upper_shadow = high - open_price
            lower_shadow = close - low
            
        return body_size, upper_shadow, lower_shadow, total_range
    
    def _identify_doji_type(self, body_size: float, upper_shadow: float, 
                           lower_shadow: float, total_range: float) -> DojiType:
        """Identify the specific type of Doji pattern"""
        if total_range == 0:
            return DojiType.FOUR_PRICE_DOJI
            
        body_ratio = body_size / total_range if total_range > 0 else 0
        
        # Check if it's a Doji based on body ratio
        if body_ratio > self.doji_threshold:
            return DojiType.NO_DOJI
            
        # Classify Doji type based on shadows
        shadow_threshold = total_range * 0.1  # 10% of total range
        
        # Four Price Doji (very rare)
        if body_size == 0 and upper_shadow == 0 and lower_shadow == 0:
            return DojiType.FOUR_PRICE_DOJI
            
        # Gravestone Doji - long upper shadow, minimal lower shadow
        if upper_shadow > total_range * 0.6 and lower_shadow < shadow_threshold:
            return DojiType.GRAVESTONE_DOJI
            
        # Dragonfly Doji - long lower shadow, minimal upper shadow  
        if lower_shadow > total_range * 0.6 and upper_shadow < shadow_threshold:
            return DojiType.DRAGONFLY_DOJI
            
        # Long-Legged Doji - both shadows are significant
        min_shadow = min(upper_shadow, lower_shadow)
        max_shadow = max(upper_shadow, lower_shadow)
        if min_shadow > total_range * 0.3 and max_shadow > total_range * 0.4:
            return DojiType.LONG_LEGGED_DOJI
            
        # Standard Doji
        return DojiType.STANDARD_DOJI
    
    def _calculate_doji_strength(self, doji_type: DojiType, body_ratio: float,
                                upper_shadow: float, lower_shadow: float, 
                                total_range: float) -> float:
        """Calculate the strength of the Doji pattern (0-100)"""
        if doji_type == DojiType.NO_DOJI:
            return 0.0
            
        # Base strength from small body
        body_strength = max(0, (self.doji_threshold - body_ratio) / self.doji_threshold * 100)
        
        # Adjust based on Doji type
        type_multiplier = {
            DojiType.FOUR_PRICE_DOJI: 1.0,
            DojiType.GRAVESTONE_DOJI: 0.9,
            DojiType.DRAGONFLY_DOJI: 0.9,
            DojiType.LONG_LEGGED_DOJI: 0.8,
            DojiType.STANDARD_DOJI: 0.7
        }.get(doji_type, 0.5)
        
        # Shadow quality adjustment
        if total_range > 0:
            shadow_balance = 1 - abs(upper_shadow - lower_shadow) / total_range
            shadow_quality = shadow_balance * 0.2  # 20% weight for shadow balance
        else:
            shadow_quality = 0
            
        final_strength = (body_strength * type_multiplier) + (shadow_quality * 100)
        return min(100, max(0, final_strength))
    
    def _determine_trend_context(self) -> str:
        """Determine current trend context"""
        if len(self.closes) < self.trend_period:
            return 'sideways'
            
        recent_closes = self.closes[-self.trend_period:]
        
        # Calculate trend slope
        x = np.arange(len(recent_closes))
        slope = np.polyfit(x, recent_closes, 1)[0]
        
        # Classify trend based on slope magnitude
        price_range = max(recent_closes) - min(recent_closes)
        relative_slope = slope / price_range if price_range > 0 else 0
        
        if relative_slope > 0.05:
            return 'uptrend'
        elif relative_slope < -0.05:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_reversal_probability(self, doji_type: DojiType, doji_strength: float,
                                       trend_context: str, volume_confirmation: bool) -> float:
        """Calculate probability of trend reversal (0-100)"""
        base_probability = {
            DojiType.GRAVESTONE_DOJI: 75,
            DojiType.DRAGONFLY_DOJI: 75,
            DojiType.LONG_LEGGED_DOJI: 65,
            DojiType.FOUR_PRICE_DOJI: 70,
            DojiType.STANDARD_DOJI: 50,
            DojiType.NO_DOJI: 0
        }.get(doji_type, 0)
        
        # Adjust for trend context
        if trend_context == 'uptrend' and doji_type == DojiType.GRAVESTONE_DOJI:
            trend_adjustment = 1.2
        elif trend_context == 'downtrend' and doji_type == DojiType.DRAGONFLY_DOJI:
            trend_adjustment = 1.2
        elif trend_context == 'sideways':
            trend_adjustment = 0.8
        else:
            trend_adjustment = 1.0
            
        # Adjust for pattern strength
        strength_adjustment = doji_strength / 100
        
        # Adjust for volume confirmation
        volume_adjustment = 1.1 if volume_confirmation else 0.9
        
        probability = base_probability * trend_adjustment * strength_adjustment * volume_adjustment
        return min(100, max(0, probability))
    
    def _check_volume_confirmation(self) -> bool:
        """Check if current volume confirms the pattern"""
        if len(self.volumes) < self.volume_lookback + 1:
            return False
            
        current_volume = self.volumes[-1]
        avg_volume = sum(self.volumes[-self.volume_lookback-1:-1]) / self.volume_lookback
        
        # Volume confirmation if current volume is above average
        return current_volume > avg_volume * 1.1
    
    def _generate_signal(self, doji_type: DojiType, reversal_probability: float,
                        trend_context: str) -> Tuple[DojiSignal, float]:
        """Generate trading signal based on Doji analysis"""
        if doji_type == DojiType.NO_DOJI:
            return DojiSignal.NEUTRAL, 25
            
        # High reversal probability signals
        if reversal_probability > 70:
            if trend_context == 'uptrend':
                return DojiSignal.REVERSAL_BEARISH, reversal_probability
            elif trend_context == 'downtrend':
                return DojiSignal.REVERSAL_BULLISH, reversal_probability
            else:
                return DojiSignal.INDECISION, reversal_probability * 0.8
                
        # Medium reversal probability
        elif reversal_probability > 50:
            return DojiSignal.INDECISION, reversal_probability
            
        # Low reversal probability
        elif reversal_probability > 30:
            return DojiSignal.CONTINUATION, 100 - reversal_probability
            
        return DojiSignal.NEUTRAL, 25
    
    def update(self, open_price: float, high: float, low: float, close: float, 
               volume: float = 0, timestamp: Optional[pd.Timestamp] = None) -> DojiRecognitionResult:
        """
        Update Doji Recognition with new candlestick data
        
        Args:
            open_price: Opening price
            high: High price
            low: Low price
            close: Close price
            volume: Volume (optional)
            timestamp: Optional timestamp
            
        Returns:
            DojiRecognitionResult: Pattern recognition result
        """
        try:
            # Store data
            self.opens.append(open_price)
            self.highs.append(high)
            self.lows.append(low)
            self.closes.append(close)
            self.volumes.append(volume)
            
            # Calculate candlestick components
            body_size, upper_shadow, lower_shadow, total_range = self._calculate_candle_components(
                open_price, high, low, close
            )
            
            # Calculate ratios
            body_ratio = body_size / total_range if total_range > 0 else 0
            shadow_ratio = upper_shadow / lower_shadow if lower_shadow > 0 else float('inf')
            
            # Identify Doji type
            doji_type = self._identify_doji_type(body_size, upper_shadow, lower_shadow, total_range)
            
            # Calculate pattern strength
            doji_strength = self._calculate_doji_strength(
                doji_type, body_ratio, upper_shadow, lower_shadow, total_range
            )
            
            # Determine trend context
            trend_context = self._determine_trend_context()
            
            # Check volume confirmation
            volume_confirmation = self._check_volume_confirmation()
            
            # Calculate reversal probability
            reversal_probability = self._calculate_reversal_probability(
                doji_type, doji_strength, trend_context, volume_confirmation
            )
            
            # Generate signal
            signal, signal_strength = self._generate_signal(
                doji_type, reversal_probability, trend_context
            )
            
            # Create result
            result = DojiRecognitionResult(
                timestamp=timestamp or pd.Timestamp.now(),
                value=doji_strength,
                doji_type=doji_type,
                doji_strength=doji_strength,
                body_size=body_size,
                upper_shadow=upper_shadow,
                lower_shadow=lower_shadow,
                total_range=total_range,
                body_ratio=body_ratio,
                shadow_ratio=shadow_ratio if shadow_ratio != float('inf') else 999,
                trend_context=trend_context,
                reversal_probability=reversal_probability,
                volume_confirmation=volume_confirmation,
                signal=signal,
                signal_strength=signal_strength,
                metadata={
                    'doji_threshold': self.doji_threshold,
                    'long_shadow_ratio': self.long_shadow_ratio,
                    'trend_period': self.trend_period,
                    'volume_lookback': self.volume_lookback
                }
            )
            
            self.doji_patterns.append(result)
            self.last_result = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error updating Doji Recognition: {str(e)}")
            raise
    
    def get_recent_patterns(self, lookback: int = 10) -> List[DojiRecognitionResult]:
        """Get recent Doji patterns"""
        return self.doji_patterns[-lookback:] if len(self.doji_patterns) >= lookback else self.doji_patterns
    
    def __str__(self) -> str:
        return f"DojiRecognitionEngine(threshold={self.doji_threshold}, trend_period={self.trend_period})"
