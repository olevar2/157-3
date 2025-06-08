# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Piercing Line Pattern Scanner - Japanese Candlestick Pattern Recognition
Platform3 Enhanced Technical Analysis Engine

The Piercing Line is a two-candle bullish reversal pattern that appears at the 
bottom of a downtrend. It consists of a bearish candle followed by a bullish 
candle that opens below the previous candle's low but closes above its midpoint.

Pattern Characteristics:
- Two-candle pattern
- First candle: Bearish (black/red)
- Second candle: Bullish (white/green)
- Second candle opens below first candle's low
- Second candle closes above midpoint of first candle
- Indicates potential bullish reversal

Key Features:
- Downtrend reversal signal
- Volume confirmation analysis
- Pattern strength measurement
- Support level validation
- Entry/exit timing optimization

Trading Applications:
- Reversal identification in downtrends
- Long position entry signals
- Stop-loss placement guidance
- Trend change confirmation
- Risk management enhancement

Mathematical Foundation:
- Midpoint = (Open1 + Close1) / 2
- Condition: Open2 < Low1 AND Close2 > Midpoint AND Close2 < Open1
- Volume2 > Average_Volume (preferred)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame

class PiercingLineType(Enum):
    """Types of Piercing Line patterns"""
    STANDARD = "standard"           # Classic piercing line
    DEEP_PIERCE = "deep_pierce"     # Pierces >75% of previous candle
    SHALLOW_PIERCE = "shallow_pierce" # Pierces 50-65% of previous candle

@dataclass
class CandleData:
    """Represents a single candlestick"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: any
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def midpoint(self) -> float:
        return (self.open + self.close) / 2
    
    @property
    def body_top(self) -> float:
        return max(self.open, self.close)
    
    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)

@dataclass
class PiercingLineResult:
    """Result of Piercing Line pattern detection"""
    pattern_type: PiercingLineType
    strength: float              # 0-100
    confidence: float            # 0-1
    position: int               # Index where pattern was found
    candle1: CandleData         # Bearish candle
    candle2: CandleData         # Bullish candle (piercing)
    pierce_percentage: float    # How much of candle1 body is pierced
    volume_confirmation: bool   # Volume support
    support_level: float        # Key support level
    target_levels: List[float]  # Potential price targets
    stop_loss: float           # Suggested stop loss
    metadata: Dict[str, Any] = field(default_factory=dict)

class PiercingLinePattern(IndicatorBase):
    """
    Piercing Line Pattern Recognition and Analysis Engine
    
    Specialized scanner for detecting and analyzing Piercing Line patterns
    with comprehensive strength assessment and trading signal generation.
    """
    
    def __init__(self, 
                 min_pierce_percentage: float = 0.5,
                 volume_threshold: float = 1.2,
                 trend_lookback: int = 10,
                 min_body_size: float = 0.3):
        """
        Initialize Piercing Line pattern scanner
        
        Args:
            min_pierce_percentage: Minimum percentage of first candle body to pierce (0.5 = 50%)
            volume_threshold: Volume multiplier vs average (1.2 = 20% above average)
            trend_lookback: Number of periods to analyze for trend context
            min_body_size: Minimum body size as percentage of total range
        """
        super().__init__()
        self.min_pierce_percentage = min_pierce_percentage
        self.volume_threshold = volume_threshold
        self.trend_lookback = trend_lookback
        self.min_body_size = min_body_size
        
        self.detected_patterns: List[PiercingLineResult] = []
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate Piercing Line patterns in the given data
        
        Args:
            data: DataFrame with OHLCV data            
        Returns:
            IndicatorResult containing detected patterns
        """
        try:
            if len(data) < 2:
                return IndicatorResult(
                    timestamp=datetime.now(),
                    indicator_name="Piercing Line Pattern",
                    indicator_type=IndicatorType.PATTERN,
                    timeframe=TimeFrame.DAILY,
                    value=[],
                    signal=[],
                    raw_data={"error": "Insufficient data for pattern detection"}
                )
            
            # Reset previous results
            self.detected_patterns = []
            # Convert to candle objects
            candles = self._create_candle_objects(data)
            
            # Detect patterns
            self._detect_piercing_line_patterns(candles)
            
            # Create signals
            signals = self._create_trading_signals()
            
            return IndicatorResult(
                timestamp=datetime.now(),
                indicator_name="Piercing Line Pattern",
                indicator_type=IndicatorType.PATTERN,
                timeframe=TimeFrame.DAILY,
                value=self.detected_patterns,
                signal=signals,
                raw_data={
                    "total_patterns": len(self.detected_patterns),
                    "pattern_types": self._get_pattern_type_counts(),
                    "strongest_pattern": self._get_strongest_pattern()
                }
            )
        except Exception as e:
            self.logger.error(f"Error in Piercing Line calculation: {e}")
            return IndicatorResult(
                timestamp=datetime.now(),
                indicator_name="Piercing Line Pattern",
                indicator_type=IndicatorType.PATTERN,
                timeframe=TimeFrame.DAILY,
                value=[],
                signal=[],
                raw_data={"error": str(e)}
            )
    
    def _create_candle_objects(self, data: pd.DataFrame) -> List[CandleData]:
        """Convert DataFrame to CandleData objects"""
        candles = []
        
        for idx, row in data.iterrows():
            candle = CandleData(
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timestamp=idx
            )
            candles.append(candle)
        
        return candles
    
    def _detect_piercing_line_patterns(self, candles: List[CandleData]):
        """Detect Piercing Line patterns in the candle data"""
        for i in range(1, len(candles)):
            candle1 = candles[i-1]  # First candle (bearish)
            candle2 = candles[i]    # Second candle (bullish)
            
            # Check basic pattern requirements
            if self._is_valid_piercing_line(candle1, candle2):
                # Calculate pattern details
                pierce_percentage = self._calculate_pierce_percentage(candle1, candle2)
                strength = self._calculate_pattern_strength(candle1, candle2, candles, i)
                confidence = self._calculate_confidence(candle1, candle2, pierce_percentage)
                
                # Determine pattern type
                pattern_type = self._classify_pattern_type(pierce_percentage)
                
                # Check volume confirmation
                volume_confirmation = self._check_volume_confirmation(candle2, candles, i)
                
                # Calculate trading levels
                support_level = min(candle1.low, candle2.low)
                targets = self._calculate_target_levels(candle1, candle2)
                stop_loss = self._calculate_stop_loss(candle1, candle2)
                
                # Create pattern result
                pattern = PiercingLineResult(
                    pattern_type=pattern_type,
                    strength=strength,
                    confidence=confidence,
                    position=i,
                    candle1=candle1,
                    candle2=candle2,
                    pierce_percentage=pierce_percentage,
                    volume_confirmation=volume_confirmation,
                    support_level=support_level,
                    target_levels=targets,
                    stop_loss=stop_loss,
                    metadata={
                        "trend_context": self._get_trend_context(candles, i),
                        "atr": self._calculate_atr(candles, i),
                        "pattern_location": "downtrend_reversal"
                    }
                )
                
                self.detected_patterns.append(pattern)
    
    def _is_valid_piercing_line(self, candle1: CandleData, candle2: CandleData) -> bool:
        """Check if two candles form a valid Piercing Line pattern"""
        # First candle must be bearish
        if candle1.is_bullish:
            return False
        
        # Second candle must be bullish
        if not candle2.is_bullish:
            return False
        
        # Second candle opens below first candle's low
        if candle2.open >= candle1.low:
            return False
        
        # Second candle closes above first candle's midpoint
        if candle2.close <= candle1.midpoint:
            return False
        
        # Second candle closes below first candle's open (doesn't engulf)
        if candle2.close >= candle1.open:
            return False
        
        # Check minimum body sizes
        candle1_range = candle1.high - candle1.low
        candle2_range = candle2.high - candle2.low
        
        if candle1_range > 0 and candle1.body_size / candle1_range < self.min_body_size:
            return False
        
        if candle2_range > 0 and candle2.body_size / candle2_range < self.min_body_size:
            return False
        
        return True
    
    def _calculate_pierce_percentage(self, candle1: CandleData, candle2: CandleData) -> float:
        """Calculate what percentage of candle1's body is pierced by candle2"""
        if candle1.body_size == 0:
            return 0.0
        
        pierced_amount = candle2.close - candle1.close
        return (pierced_amount / candle1.body_size) * 100
    
    def _classify_pattern_type(self, pierce_percentage: float) -> PiercingLineType:
        """Classify the type of Piercing Line based on pierce percentage"""
        if pierce_percentage >= 75:
            return PiercingLineType.DEEP_PIERCE
        elif pierce_percentage >= 50:
            return PiercingLineType.STANDARD
        else:
            return PiercingLineType.SHALLOW_PIERCE
    
    def _calculate_pattern_strength(self, candle1: CandleData, candle2: CandleData, 
                                  candles: List[CandleData], position: int) -> float:
        """Calculate the strength of the Piercing Line pattern (0-100)"""
        strength = 50.0  # Base strength
        
        # Pierce percentage bonus
        pierce_pct = self._calculate_pierce_percentage(candle1, candle2)
        if pierce_pct >= 75:
            strength += 25
        elif pierce_pct >= 65:
            strength += 15
        elif pierce_pct >= 55:
            strength += 10
        
        # Body size bonus
        avg_range = np.mean([c.high - c.low for c in candles[max(0, position-10):position]])
        if avg_range > 0:
            if candle1.body_size / avg_range > 1.5:
                strength += 10
            if candle2.body_size / avg_range > 1.5:
                strength += 10
        
        # Gap bonus (larger gap = stronger)
        gap_size = candle1.low - candle2.open
        if gap_size > 0 and avg_range > 0:
            gap_ratio = gap_size / avg_range
            if gap_ratio > 0.5:
                strength += 15
            elif gap_ratio > 0.3:
                strength += 10
            elif gap_ratio > 0.1:
                strength += 5
        
        # Trend context bonus
        trend = self._get_trend_context(candles, position)
        if trend == "strong_downtrend":
            strength += 15
        elif trend == "downtrend":
            strength += 10
        
        # Volume confirmation bonus
        if self._check_volume_confirmation(candle2, candles, position):
            strength += 10
        
        return min(100, max(0, strength))
    
    def _calculate_confidence(self, candle1: CandleData, candle2: CandleData, 
                            pierce_percentage: float) -> float:
        """Calculate confidence level (0-1)"""
        base_confidence = 0.6
        
        # Pierce percentage influence
        if pierce_percentage >= 75:
            base_confidence += 0.2
        elif pierce_percentage >= 65:
            base_confidence += 0.15
        elif pierce_percentage >= 55:
            base_confidence += 0.1
        
        # Body ratio influence
        if candle1.body_size > 0 and candle2.body_size > 0:
            body_ratio = candle2.body_size / candle1.body_size
            if body_ratio > 0.8:  # Similar or larger body
                base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _get_trend_context(self, candles: List[CandleData], position: int) -> str:
        """Determine trend context at the pattern position"""
        if position < self.trend_lookback:
            return "insufficient_data"
        
        lookback_candles = candles[position - self.trend_lookback:position]
        closes = [c.close for c in lookback_candles]
        
        # Simple trend analysis
        start_price = closes[0]
        end_price = closes[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        
        if change_pct <= -10:
            return "strong_downtrend"
        elif change_pct <= -3:
            return "downtrend"
        elif change_pct >= 3:
            return "uptrend"
        else:
            return "sideways"
    
    def _check_volume_confirmation(self, candle2: CandleData, 
                                 candles: List[CandleData], position: int) -> bool:
        """Check if volume confirms the pattern"""
        if position < 10:
            return False
        
        recent_volumes = [c.volume for c in candles[position-10:position]]
        avg_volume = np.mean(recent_volumes)
        
        return candle2.volume >= (avg_volume * self.volume_threshold)
    
    def _calculate_target_levels(self, candle1: CandleData, candle2: CandleData) -> List[float]:
        """Calculate potential price targets"""
        pattern_height = candle1.open - candle1.close
        
        target1 = candle2.close + (pattern_height * 0.5)  # 50% projection
        target2 = candle2.close + pattern_height           # 100% projection
        target3 = candle2.close + (pattern_height * 1.618) # Fibonacci extension
        
        return [target1, target2, target3]
    
    def _calculate_stop_loss(self, candle1: CandleData, candle2: CandleData) -> float:
        """Calculate suggested stop loss level"""
        return min(candle1.low, candle2.low) - (candle2.body_size * 0.1)
    
    def _calculate_atr(self, candles: List[CandleData], position: int, period: int = 14) -> float:
        """Calculate Average True Range"""
        if position < period:
            return 0.0
        
        tr_values = []
        for i in range(position - period, position):
            if i > 0:
                high_low = candles[i].high - candles[i].low
                high_close = abs(candles[i].high - candles[i-1].close)
                low_close = abs(candles[i].low - candles[i-1].close)
                tr = max(high_low, high_close, low_close)
                tr_values.append(tr)
        
        return np.mean(tr_values) if tr_values else 0.0
    
    def _create_trading_signals(self) -> List[Dict[str, Any]]:
        """Create trading signals from detected patterns"""
        signals = []
        
        for pattern in self.detected_patterns:
            if pattern.strength >= 70:  # High-quality patterns only
                signal = {
                    "type": "BUY",
                    "pattern": "Piercing Line",
                    "strength": pattern.strength,
                    "confidence": pattern.confidence,
                    "entry_price": pattern.candle2.close,
                    "stop_loss": pattern.stop_loss,
                    "targets": pattern.target_levels,
                    "position": pattern.position,
                    "volume_confirmed": pattern.volume_confirmation,
                    "pattern_type": pattern.pattern_type.value,
                    "pierce_percentage": pattern.pierce_percentage
                }
                signals.append(signal)
        
        return signals
    
    def _get_pattern_type_counts(self) -> Dict[str, int]:
        """Get count of each pattern type"""
        counts = {ptype.value: 0 for ptype in PiercingLineType}
        for pattern in self.detected_patterns:
            counts[pattern.pattern_type.value] += 1
        return counts
    
    def _get_strongest_pattern(self) -> Optional[Dict[str, Any]]:
        """Get the strongest detected pattern"""
        if not self.detected_patterns:
            return None
        
        strongest = max(self.detected_patterns, key=lambda p: p.strength)
        return {
            "strength": strongest.strength,
            "confidence": strongest.confidence,
            "position": strongest.position,
            "pattern_type": strongest.pattern_type.value,
            "pierce_percentage": strongest.pierce_percentage
        }
    
    def get_pattern_description(self) -> str:
        """Get description of the pattern"""
        return (
            "Piercing Line: A two-candle bullish reversal pattern appearing in downtrends. "
            "Consists of a bearish candle followed by a bullish candle that opens below "
            "the previous low but closes above its midpoint, signaling potential upward reversal."
        )
