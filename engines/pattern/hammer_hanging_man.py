# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Hammer/Hanging Man Detector - Japanese Candlestick Pattern Recognition
Platform3 Enhanced Technical Analysis Engine

Detects Hammer and Hanging Man candlestick patterns, which are critical reversal
indicators in technical analysis. These patterns have identical appearance but
different meanings based on trend context.

Pattern Characteristics:
- Small real body at upper end of trading range
- Long lower shadow (at least twice the body size)
- Little to no upper shadow
- Can be bullish (Hammer) or bearish (Hanging Man) depending on context

Key Features:
- Trend context analysis
- Pattern strength measurement
- Volume confirmation
- Reversal probability scoring
- Support/resistance level validation
- Multiple timeframe coordination

Trading Applications:
- Trend reversal identification
- Support level confirmation
- Entry/exit timing optimization
- Risk management enhancement
- Market sentiment analysis

Mathematical Foundation:
- Body Size = |Close - Open|
- Lower Shadow = Min(Open, Close) - Low
- Upper Shadow = High - Max(Open, Close)
- Body Position = (Max(Open, Close) - Low) / (High - Low)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame, IndicatorSignal, SignalType

class HammerType(Enum):
    """Types of Hammer/Hanging Man patterns"""
    HAMMER = "hammer"  # Bullish reversal in downtrend
    HANGING_MAN = "hanging_man"  # Bearish reversal in uptrend
    INVERTED_HAMMER = "inverted_hammer"  # Bullish reversal with long upper shadow
    SHOOTING_STAR = "shooting_star"  # Bearish reversal with long upper shadow
    NO_PATTERN = "no_pattern"

class HammerSignal(Enum):
    """Hammer signal types"""
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    SUPPORT_CONFIRMATION = "support_confirmation"
    RESISTANCE_CONFIRMATION = "resistance_confirmation"
    TREND_CONTINUATION = "trend_continuation"
    NEUTRAL = "neutral"

# Create a standalone class instead of inheriting from IndicatorResult
@dataclass
class HammerDetectorResult:
    """Hammer/Hanging Man detection result"""
    timestamp: datetime
    pattern_type: HammerType
    pattern_strength: float  # 0-100, higher means stronger pattern
    body_size: float
    lower_shadow: float
    upper_shadow: float
    total_range: float
    body_position: float  # Position of body in range (0-1)
    shadow_ratio: float  # Lower shadow / Body size
    trend_context: str  # 'uptrend', 'downtrend', 'sideways'
    reversal_probability: float  # 0-100
    volume_confirmation: bool
    support_resistance_level: Optional[float]
    signal: HammerSignal
    signal_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            'pattern_type': self.pattern_type.value,
            'pattern_strength': self.pattern_strength,
            'body_size': self.body_size,
            'lower_shadow': self.lower_shadow,
            'upper_shadow': self.upper_shadow,
            'total_range': self.total_range,
            'body_position': self.body_position,
            'shadow_ratio': self.shadow_ratio,
            'trend_context': self.trend_context,
            'reversal_probability': self.reversal_probability,
            'volume_confirmation': self.volume_confirmation,
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
                metadata={
                    'pattern_type': self.pattern_type.value,
                    'body_position': self.body_position,
                    'shadow_ratio': self.shadow_ratio
                }
            ) if self.signal != HammerSignal.NEUTRAL else None
        )
    
    def _map_signal_type(self) -> SignalType:
        """Map Hammer signal to standard SignalType"""
        if self.signal == HammerSignal.BULLISH_REVERSAL:
            return SignalType.BUY
        elif self.signal == HammerSignal.BEARISH_REVERSAL:
            return SignalType.SELL
        elif self.signal == HammerSignal.TREND_CONTINUATION:
            return SignalType.HOLD
        else:
            return SignalType.NEUTRAL

class HammerHangingManDetector(IndicatorBase):
    """
    Hammer/Hanging Man Detector - Japanese Candlestick Pattern Recognition
    
    Identifies Hammer and Hanging Man patterns with comprehensive analysis
    including trend context, volume confirmation, and reversal probability.
    """
    
    def __init__(self, 
                 min_shadow_ratio: float = 2.0,
                 max_upper_shadow_ratio: float = 0.1,
                 min_body_position: float = 0.7,
                 trend_period: int = 10,
                 volume_lookback: int = 5,
                 support_resistance_tolerance: float = 0.02):
        """
        Initialize Hammer/Hanging Man Detector
        
        Args:
            min_shadow_ratio: Minimum ratio of lower shadow to body (default: 2.0)
            max_upper_shadow_ratio: Maximum ratio of upper shadow to body (default: 0.1)
            min_body_position: Minimum position of body in range for pattern (default: 0.7)
            trend_period: Period for trend context analysis (default: 10)
            volume_lookback: Period for volume confirmation (default: 5)
            support_resistance_tolerance: Tolerance for S/R level detection (default: 2%)
        """
        super().__init__("Hammer/Hanging Man Detector")
        self.min_shadow_ratio = min_shadow_ratio
        self.max_upper_shadow_ratio = max_upper_shadow_ratio
        self.min_body_position = min_body_position
        self.trend_period = trend_period
        self.volume_lookback = volume_lookback
        self.support_resistance_tolerance = support_resistance_tolerance
        
        # Validation
        if self.min_shadow_ratio < 1:
            raise ValueError("Minimum shadow ratio must be >= 1")
        if self.max_upper_shadow_ratio < 0:
            raise ValueError("Maximum upper shadow ratio must be >= 0")
        if not 0 <= self.min_body_position <= 1:
            raise ValueError("Minimum body position must be between 0 and 1")
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
        self.patterns = []
        self.support_resistance_levels = []
        
    def _calculate_candle_components(self, open_price: float, high: float, 
                                   low: float, close: float) -> Tuple[float, float, float, float, float]:
        """Calculate candlestick components"""
        body_size = abs(close - open_price)
        total_range = high - low
        
        # Calculate shadows and body position
        body_top = max(open_price, close)
        body_bottom = min(open_price, close)
        
        upper_shadow = high - body_top
        lower_shadow = body_bottom - low
        
        # Body position in the total range (0 = bottom, 1 = top)
        body_position = (body_top - low) / total_range if total_range > 0 else 0.5
        
        return body_size, upper_shadow, lower_shadow, total_range, body_position
    
    def _identify_pattern_type(self, body_size: float, upper_shadow: float, 
                              lower_shadow: float, body_position: float,
                              trend_context: str) -> HammerType:
        """Identify the specific type of Hammer/Hanging Man pattern"""
        if body_size == 0:  # Doji-like patterns
            return HammerType.NO_PATTERN
            
        # Calculate ratios
        lower_shadow_ratio = lower_shadow / body_size if body_size > 0 else 0
        upper_shadow_ratio = upper_shadow / body_size if body_size > 0 else 0
        
        # Check for Hammer/Hanging Man (long lower shadow, small upper shadow)
        is_hammer_hanging = (
            lower_shadow_ratio >= self.min_shadow_ratio and
            upper_shadow_ratio <= self.max_upper_shadow_ratio and
            body_position >= self.min_body_position
        )
        
        # Check for Inverted Hammer/Shooting Star (long upper shadow, small lower shadow)
        is_inverted_hammer_shooting = (
            upper_shadow_ratio >= self.min_shadow_ratio and
            lower_shadow_ratio <= self.max_upper_shadow_ratio and
            body_position <= (1 - self.min_body_position)
        )
        
        if is_hammer_hanging:
            if trend_context == 'downtrend':
                return HammerType.HAMMER
            elif trend_context == 'uptrend':
                return HammerType.HANGING_MAN
            else:
                return HammerType.HAMMER  # Default to hammer in sideways
                
        elif is_inverted_hammer_shooting:
            if trend_context == 'downtrend':
                return HammerType.INVERTED_HAMMER
            elif trend_context == 'uptrend':
                return HammerType.SHOOTING_STAR
            else:
                return HammerType.INVERTED_HAMMER  # Default to inverted hammer
                
        return HammerType.NO_PATTERN
    
    def _calculate_pattern_strength(self, pattern_type: HammerType, body_size: float,
                                   upper_shadow: float, lower_shadow: float,
                                   total_range: float, body_position: float) -> float:
        """Calculate the strength of the pattern (0-100)"""
        if pattern_type == HammerType.NO_PATTERN:
            return 0.0
            
        if body_size == 0 or total_range == 0:
            return 0.0
            
        # Base strength from shadow ratios
        if pattern_type in [HammerType.HAMMER, HammerType.HANGING_MAN]:
            primary_shadow = lower_shadow
            secondary_shadow = upper_shadow
        else:  # Inverted patterns
            primary_shadow = upper_shadow
            secondary_shadow = lower_shadow
            
        shadow_ratio = primary_shadow / body_size
        
        # Strength increases with longer primary shadow
        shadow_strength = min(100, (shadow_ratio / self.min_shadow_ratio) * 50)
        
        # Penalty for secondary shadow being too large
        secondary_ratio = secondary_shadow / body_size
        secondary_penalty = min(30, secondary_ratio * 100)
        
        # Body position strength
        if pattern_type in [HammerType.HAMMER, HammerType.HANGING_MAN]:
            position_strength = (body_position - self.min_body_position) / (1 - self.min_body_position) * 30
        else:  # Inverted patterns
            position_strength = (1 - self.min_body_position - body_position) / (1 - self.min_body_position) * 30
            
        # Total range consideration (bigger candles are more significant)
        range_bonus = min(20, (total_range / np.mean(self.highs[-20:]) if len(self.highs) >= 20 else total_range) * 100)
        
        final_strength = shadow_strength + position_strength + range_bonus - secondary_penalty
        return min(100, max(0, final_strength))
    
    def _determine_trend_context(self) -> str:
        """Determine current trend context"""
        if len(self.closes) < self.trend_period:
            return 'sideways'
            
        recent_closes = self.closes[-self.trend_period:]
        
        # Calculate trend slope using linear regression
        x = np.arange(len(recent_closes))
        slope = np.polyfit(x, recent_closes, 1)[0]
        
        # Normalize slope by price range
        price_range = max(recent_closes) - min(recent_closes)
        relative_slope = slope / price_range if price_range > 0 else 0
        
        # Classify trend
        if relative_slope > 0.1:
            return 'uptrend'
        elif relative_slope < -0.1:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _find_support_resistance_level(self, low: float, high: float) -> Optional[float]:
        """Find nearby support/resistance levels"""
        if len(self.lows) < 5:
            return None
            
        # Look for recent swing highs and lows
        recent_lows = self.lows[-20:] if len(self.lows) >= 20 else self.lows
        recent_highs = self.highs[-20:] if len(self.highs) >= 20 else self.highs
        
        all_levels = recent_lows + recent_highs
        
        # Find levels near current candle range
        tolerance = (high - low) * self.support_resistance_tolerance
        
        for level in all_levels:
            if abs(level - low) <= tolerance or abs(level - high) <= tolerance:
                return level
                
        return None
    
    def _calculate_reversal_probability(self, pattern_type: HammerType, pattern_strength: float,
                                       trend_context: str, volume_confirmation: bool,
                                       support_resistance_level: Optional[float]) -> float:
        """Calculate probability of trend reversal (0-100)"""
        base_probability = {
            HammerType.HAMMER: 70,
            HammerType.HANGING_MAN: 65,
            HammerType.INVERTED_HAMMER: 60,
            HammerType.SHOOTING_STAR: 75,
            HammerType.NO_PATTERN: 0
        }.get(pattern_type, 0)
        
        # Adjust for trend context appropriateness
        if pattern_type == HammerType.HAMMER and trend_context == 'downtrend':
            trend_adjustment = 1.2
        elif pattern_type == HammerType.HANGING_MAN and trend_context == 'uptrend':
            trend_adjustment = 1.2
        elif pattern_type == HammerType.INVERTED_HAMMER and trend_context == 'downtrend':
            trend_adjustment = 1.1
        elif pattern_type == HammerType.SHOOTING_STAR and trend_context == 'uptrend':
            trend_adjustment = 1.3
        else:
            trend_adjustment = 0.8  # Pattern not in ideal trend context
            
        # Adjust for pattern strength
        strength_adjustment = pattern_strength / 100
        
        # Adjust for volume confirmation
        volume_adjustment = 1.15 if volume_confirmation else 0.9
        
        # Adjust for support/resistance level
        sr_adjustment = 1.1 if support_resistance_level is not None else 1.0
        
        probability = base_probability * trend_adjustment * strength_adjustment * volume_adjustment * sr_adjustment
        return min(100, max(0, probability))
    
    def _check_volume_confirmation(self) -> bool:
        """Check if current volume confirms the pattern"""
        if len(self.volumes) < self.volume_lookback + 1:
            return False
            
        current_volume = self.volumes[-1]
        avg_volume = sum(self.volumes[-self.volume_lookback-1:-1]) / self.volume_lookback
        
        # Volume confirmation if current volume is significantly above average
        return current_volume > avg_volume * 1.2
    
    def _generate_signal(self, pattern_type: HammerType, reversal_probability: float,
                        trend_context: str, support_resistance_level: Optional[float]) -> Tuple[HammerSignal, float]:
        """Generate trading signal based on pattern analysis"""
        if pattern_type == HammerType.NO_PATTERN:
            return HammerSignal.NEUTRAL, 25
            
        # High probability reversal signals
        if reversal_probability > 75:
            if pattern_type in [HammerType.HAMMER, HammerType.INVERTED_HAMMER]:
                if support_resistance_level is not None:
                    return HammerSignal.SUPPORT_CONFIRMATION, reversal_probability
                return HammerSignal.BULLISH_REVERSAL, reversal_probability
            else:  # Hanging Man or Shooting Star
                if support_resistance_level is not None:
                    return HammerSignal.RESISTANCE_CONFIRMATION, reversal_probability
                return HammerSignal.BEARISH_REVERSAL, reversal_probability
                
        # Medium probability signals
        elif reversal_probability > 50:
            if pattern_type in [HammerType.HAMMER, HammerType.INVERTED_HAMMER]:
                return HammerSignal.BULLISH_REVERSAL, reversal_probability
            else:
                return HammerSignal.BEARISH_REVERSAL, reversal_probability
                
        # Low probability - trend continuation more likely
        elif reversal_probability > 30:
            return HammerSignal.TREND_CONTINUATION, 100 - reversal_probability
            
        return HammerSignal.NEUTRAL, 25
    
    def update(self, open_price: float, high: float, low: float, close: float, 
               volume: float = 0, timestamp: Optional[pd.Timestamp] = None) -> HammerDetectorResult:
        """
        Update Hammer/Hanging Man Detector with new candlestick data
        
        Args:
            open_price: Opening price
            high: High price
            low: Low price
            close: Close price
            volume: Volume (optional)
            timestamp: Optional timestamp
            
        Returns:
            HammerDetectorResult: Pattern detection result
        """
        try:
            # Store data
            self.opens.append(open_price)
            self.highs.append(high)
            self.lows.append(low)
            self.closes.append(close)
            self.volumes.append(volume)
            
            # Calculate candlestick components
            body_size, upper_shadow, lower_shadow, total_range, body_position = self._calculate_candle_components(
                open_price, high, low, close
            )
            
            # Determine trend context
            trend_context = self._determine_trend_context()
            
            # Identify pattern type
            pattern_type = self._identify_pattern_type(
                body_size, upper_shadow, lower_shadow, body_position, trend_context
            )
            
            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength(
                pattern_type, body_size, upper_shadow, lower_shadow, total_range, body_position
            )
            
            # Calculate shadow ratio for primary shadow
            if pattern_type in [HammerType.HAMMER, HammerType.HANGING_MAN]:
                shadow_ratio = lower_shadow / body_size if body_size > 0 else 0
            elif pattern_type in [HammerType.INVERTED_HAMMER, HammerType.SHOOTING_STAR]:
                shadow_ratio = upper_shadow / body_size if body_size > 0 else 0
            else:
                shadow_ratio = 0
                
            # Find support/resistance level
            support_resistance_level = self._find_support_resistance_level(low, high)
            
            # Check volume confirmation
            volume_confirmation = self._check_volume_confirmation()
            
            # Calculate reversal probability
            reversal_probability = self._calculate_reversal_probability(
                pattern_type, pattern_strength, trend_context, volume_confirmation, support_resistance_level
            )
            
            # Generate signal
            signal, signal_strength = self._generate_signal(
                pattern_type, reversal_probability, trend_context, support_resistance_level
            )
            
            # Create result
            result = HammerDetectorResult(
                timestamp=timestamp or pd.Timestamp.now(),
                value=pattern_strength,
                pattern_type=pattern_type,
                pattern_strength=pattern_strength,
                body_size=body_size,
                lower_shadow=lower_shadow,
                upper_shadow=upper_shadow,
                total_range=total_range,
                body_position=body_position,
                shadow_ratio=shadow_ratio,
                trend_context=trend_context,
                reversal_probability=reversal_probability,
                volume_confirmation=volume_confirmation,
                support_resistance_level=support_resistance_level,
                signal=signal,
                signal_strength=signal_strength,
                metadata={
                    'min_shadow_ratio': self.min_shadow_ratio,
                    'max_upper_shadow_ratio': self.max_upper_shadow_ratio,
                    'min_body_position': self.min_body_position,
                    'trend_period': self.trend_period,
                    'volume_lookback': self.volume_lookback
                }
            )
            
            self.patterns.append(result)
            self.last_result = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error updating Hammer/Hanging Man Detector: {str(e)}")
            raise
    
    def get_recent_patterns(self, lookback: int = 10) -> List[HammerDetectorResult]:
        """Get recent patterns"""
        return self.patterns[-lookback:] if len(self.patterns) >= lookback else self.patterns
    
    def __str__(self) -> str:
        return f"HammerHangingManDetector(shadow_ratio={self.min_shadow_ratio}, trend_period={self.trend_period})"
