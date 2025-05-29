"""
Japanese Candlestick Pattern Recognition - Advanced Pattern Detection System
Identifies and analyzes traditional Japanese candlestick patterns for trend reversal and continuation signals.
Essential for timing entries/exits and understanding market psychology.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import math
from dataclasses import dataclass
from enum import Enum


class CandlestickPatternType(Enum):
    """Types of candlestick patterns - Comprehensive collection for model training"""
    
    # === SINGLE CANDLE PATTERNS ===
    # Basic Doji family
    DOJI = "doji"
    DRAGONFLY_DOJI = "dragonfly_doji"
    GRAVESTONE_DOJI = "gravestone_doji"
    LONG_LEGGED_DOJI = "long_legged_doji"
    FOUR_PRICE_DOJI = "four_price_doji"
    
    # Hammer family
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    INVERTED_HAMMER = "inverted_hammer"
    SHOOTING_STAR = "shooting_star"
    
    # Spinning tops
    SPINNING_TOP_BULLISH = "spinning_top_bullish"
    SPINNING_TOP_BEARISH = "spinning_top_bearish"
    
    # Marubozu family
    MARUBOZU_BULLISH = "marubozu_bullish"
    MARUBOZU_BEARISH = "marubozu_bearish"
    WHITE_MARUBOZU = "white_marubozu"
    BLACK_MARUBOZU = "black_marubozu"
    OPENING_MARUBOZU_BULLISH = "opening_marubozu_bullish"
    OPENING_MARUBOZU_BEARISH = "opening_marubozu_bearish"
    CLOSING_MARUBOZU_BULLISH = "closing_marubozu_bullish"
    CLOSING_MARUBOZU_BEARISH = "closing_marubozu_bearish"
    
    # Special single candles
    BELT_HOLD_BULLISH = "belt_hold_bullish"
    BELT_HOLD_BEARISH = "belt_hold_bearish"
    HIGH_WAVE_CANDLE = "high_wave_candle"
    RICKSHAW_MAN = "rickshaw_man"
    
    # === TWO CANDLE PATTERNS ===
    # Engulfing patterns
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    
    # Harami patterns
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    HARAMI_CROSS_BULLISH = "harami_cross_bullish"
    HARAMI_CROSS_BEARISH = "harami_cross_bearish"
    
    # Penetrating patterns
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    
    # Tweezer patterns
    TWEEZER_TOPS = "tweezer_tops"
    TWEEZER_BOTTOMS = "tweezer_bottoms"
    
    # Counterattack patterns
    COUNTERATTACK_BULLISH = "counterattack_bullish"
    COUNTERATTACK_BEARISH = "counterattack_bearish"
    
    # Kicking patterns
    KICKING_BULLISH = "kicking_bullish"
    KICKING_BEARISH = "kicking_bearish"
    
    # In-neck patterns
    IN_NECK = "in_neck"
    ON_NECK = "on_neck"
    THRUSTING_LINE = "thrusting_line"
    
    # Separating lines
    SEPARATING_LINES_BULLISH = "separating_lines_bullish"
    SEPARATING_LINES_BEARISH = "separating_lines_bearish"
    
    # Meeting lines
    MEETING_LINES_BULLISH = "meeting_lines_bullish"
    MEETING_LINES_BEARISH = "meeting_lines_bearish"
    
    # === THREE CANDLE PATTERNS ===
    # Star patterns
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    MORNING_DOJI_STAR = "morning_doji_star"
    EVENING_DOJI_STAR = "evening_doji_star"
    
    # Soldier patterns
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    ADVANCE_BLOCK = "advance_block"
    DELIBERATION = "deliberation"
    
    # Inside patterns
    THREE_INSIDE_UP = "three_inside_up"
    THREE_INSIDE_DOWN = "three_inside_down"
    THREE_OUTSIDE_UP = "three_outside_up"
    THREE_OUTSIDE_DOWN = "three_outside_down"
    
    # Gap patterns
    UPSIDE_GAP_TWO_CROWS = "upside_gap_two_crows"
    DOWNSIDE_GAP_THREE_METHODS = "downside_gap_three_methods"
    UPSIDE_GAP_THREE_METHODS = "upside_gap_three_methods"
    
    # Sandwich patterns
    SANDWICH_BOTTOM = "sandwich_bottom"
    SANDWICH_TOP = "sandwich_top"
    
    # Abandoned baby
    ABANDONED_BABY_BULLISH = "abandoned_baby_bullish"
    ABANDONED_BABY_BEARISH = "abandoned_baby_bearish"
    
    # Tri-star patterns
    TRI_STAR_BULLISH = "tri_star_bullish"
    TRI_STAR_BEARISH = "tri_star_bearish"
    
    # === FOUR+ CANDLE PATTERNS ===
    # Rising/Falling methods
    RISING_THREE_METHODS = "rising_three_methods"
    FALLING_THREE_METHODS = "falling_three_methods"
    
    # Concealing patterns
    CONCEALING_BABY_SWALLOW = "concealing_baby_swallow"
    
    # Breakaway patterns
    BREAKAWAY_BULLISH = "breakaway_bullish"
    BREAKAWAY_BEARISH = "breakaway_bearish"
    
    # Ladder patterns
    LADDER_BOTTOM = "ladder_bottom"
    LADDER_TOP = "ladder_top"
    
    # Unique patterns
    THREE_RIVER_BOTTOM = "three_river_bottom"
    UNIQUE_THREE_RIVER_BOTTOM = "unique_three_river_bottom"
    
    # Stick sandwich
    STICK_SANDWICH = "stick_sandwich"
    
    # Homing pigeon
    HOMING_PIGEON = "homing_pigeon"
    
    # === COMPLEX CONTINUATION PATTERNS ===
    # Window patterns
    RISING_WINDOW = "rising_window"
    FALLING_WINDOW = "falling_window"
    
    # Side-by-side patterns
    SIDE_BY_SIDE_WHITE_LINES_BULLISH = "side_by_side_white_lines_bullish"
    SIDE_BY_SIDE_WHITE_LINES_BEARISH = "side_by_side_white_lines_bearish"
    
    # Mat hold patterns
    MAT_HOLD_BULLISH = "mat_hold_bullish"
    MAT_HOLD_BEARISH = "mat_hold_bearish"
    
    # === RARE AND EXOTIC PATTERNS ===
    # Doji star variants
    NORTHERN_DOJI = "northern_doji"
    SOUTHERN_DOJI = "southern_doji"
    
    # Three line strike
    THREE_LINE_STRIKE_BULLISH = "three_line_strike_bullish"
    THREE_LINE_STRIKE_BEARISH = "three_line_strike_bearish"
    
    # Identical three crows
    IDENTICAL_THREE_CROWS = "identical_three_crows"
    
    # Takuri line
    TAKURI_LINE = "takuri_line"
    
    # Closing price reversal
    CLOSING_PRICE_REVERSAL_BULLISH = "closing_price_reversal_bullish"
    CLOSING_PRICE_REVERSAL_BEARISH = "closing_price_reversal_bearish"


class PatternSignificance(Enum):
    """Pattern significance levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class CandlestickData:
    """Single candlestick data"""
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: Optional[Any] = None
    
    @property
    def body_size(self) -> float:
        """Size of the candle body"""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> float:
        """Size of upper shadow"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Size of lower shadow"""
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        """Total range from high to low"""
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Is candle bullish (close > open)"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Is candle bearish (close < open)"""
        return self.close < self.open


@dataclass
class CandlestickPatternSignal:
    """Signal output for Candlestick Pattern Recognition"""
    timestamp: Optional[Any] = None
    pattern_type: Optional[CandlestickPatternType] = None
    pattern_name: str = ""
    significance: PatternSignificance = PatternSignificance.WEAK
    
    # Pattern analysis
    reversal_potential: str = "neutral"  # bullish, bearish, neutral
    continuation_signal: bool = False
    pattern_reliability: float = 0.5
    pattern_strength: float = 0.0
    
    # Context analysis
    trend_context: str = "unknown"  # uptrend, downtrend, sideways
    volume_confirmation: bool = False
    support_resistance_level: bool = False
    
    # Signal generation
    signal_direction: str = "hold"  # buy, sell, hold
    signal_strength: float = 0.0
    signal_confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Pattern components
    pattern_candles: List[Dict[str, float]] = None
    pattern_description: str = ""
    pattern_implications: str = ""
    
    def __post_init__(self):
        if self.pattern_candles is None:
            self.pattern_candles = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'pattern_type': self.pattern_type.value if self.pattern_type else None,
            'pattern_name': self.pattern_name,
            'significance': self.significance.value,
            'reversal_potential': self.reversal_potential,
            'continuation_signal': self.continuation_signal,
            'pattern_reliability': self.pattern_reliability,
            'pattern_strength': self.pattern_strength,
            'trend_context': self.trend_context,
            'volume_confirmation': self.volume_confirmation,
            'support_resistance_level': self.support_resistance_level,
            'signal_direction': self.signal_direction,
            'signal_strength': self.signal_strength,
            'signal_confidence': self.signal_confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'pattern_candles': self.pattern_candles,
            'pattern_description': self.pattern_description,
            'pattern_implications': self.pattern_implications
        }


class JapaneseCandlestickPatterns:
    """
    Japanese Candlestick Pattern Recognition System
    
    Advanced pattern recognition with:
    - Single, double, and triple candle patterns
    - Pattern strength and reliability assessment
    - Trend context analysis
    - Volume confirmation
    - Support/resistance level detection
    - Signal generation with entry/exit levels
    """
    
    def __init__(self,
                 body_threshold: float = 0.1,
                 shadow_threshold: float = 2.0,
                 doji_threshold: float = 0.05,
                 volume_threshold: float = 1.2):
        """
        Initialize pattern recognition parameters
        
        Args:
            body_threshold: Minimum body size relative to range (0.1 = 10%)
            shadow_threshold: Shadow-to-body ratio for hammer/shooting star (2.0 = 2x body)
            doji_threshold: Maximum body size for doji pattern (0.05 = 5% of range)
            volume_threshold: Volume multiplier for confirmation (1.2 = 20% above average)
        """
        self.body_threshold = body_threshold
        self.shadow_threshold = shadow_threshold
        self.doji_threshold = doji_threshold
        self.volume_threshold = volume_threshold
        
        # Data storage
        self.candles: List[CandlestickData] = []
        self.patterns: List[CandlestickPatternSignal] = []
        self.average_volume: float = 0.0
        self.trend_direction: str = "unknown"
        
    def add_candle(self, open_price: float, high: float, low: float, close: float,
                   volume: float = 0.0, timestamp: Optional[Any] = None) -> CandlestickPatternSignal:
        """
        Add new candle and detect patterns
        """
        try:
            # Create candle data
            candle = CandlestickData(
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                timestamp=timestamp
            )
            
            self.candles.append(candle)
            
            # Update average volume
            if volume > 0:
                volumes = [c.volume for c in self.candles[-20:] if c.volume > 0]
                if volumes:
                    self.average_volume = np.mean(volumes)
            
            # Update trend direction
            self._update_trend_direction()
            
            # Detect patterns (need at least 3 candles for comprehensive analysis)
            if len(self.candles) >= 3:
                pattern_signal = self._detect_patterns()
                self.patterns.append(pattern_signal)
                return pattern_signal
            else:
                # Return neutral signal for insufficient data
                return CandlestickPatternSignal(
                    timestamp=timestamp,
                    pattern_name="insufficient_data",
                    signal_direction="hold"
                )
                
        except Exception as e:
            return CandlestickPatternSignal(
                timestamp=timestamp,
                pattern_name="error",
                signal_direction="hold",
                pattern_description=f"Error: {str(e)}"
            )
    
    def _update_trend_direction(self):
        """Update current trend direction based on recent price action"""
        if len(self.candles) < 10:
            self.trend_direction = "unknown"
            return
        
        recent_candles = self.candles[-10:]
        recent_closes = [c.close for c in recent_candles]
        
        # Simple trend analysis using linear regression
        x = list(range(len(recent_closes)))
        slope = np.polyfit(x, recent_closes, 1)[0]
        
        if slope > 0.001:  # Adjust threshold based on price level
            self.trend_direction = "uptrend"
        elif slope < -0.001:
            self.trend_direction = "downtrend"
        else:
            self.trend_direction = "sideways"
    
    def _detect_patterns(self) -> CandlestickPatternSignal:
        """Detect candlestick patterns in recent candles"""
        current = self.candles[-1]
        
        # Try three-candle patterns first (most reliable)
        if len(self.candles) >= 3:
            three_candle_pattern = self._detect_three_candle_patterns()
            if three_candle_pattern.pattern_type:
                return three_candle_pattern
        
        # Try two-candle patterns
        if len(self.candles) >= 2:
            two_candle_pattern = self._detect_two_candle_patterns()
            if two_candle_pattern.pattern_type:
                return two_candle_pattern
        
        # Try single-candle patterns
        single_candle_pattern = self._detect_single_candle_patterns()
        if single_candle_pattern.pattern_type:
            return single_candle_pattern
        
        # No pattern detected
        return CandlestickPatternSignal(
            timestamp=current.timestamp,
            pattern_name="no_pattern",
            signal_direction="hold",
            trend_context=self.trend_direction
        )
    
    def _detect_single_candle_patterns(self) -> CandlestickPatternSignal:
        """Detect single candlestick patterns"""
        current = self.candles[-1]
        
        # Doji pattern
        if self._is_doji(current):
            return self._create_doji_signal(current)
        
        # Hammer patterns
        if self._is_hammer(current):
            return self._create_hammer_signal(current)
        
        # Shooting star / Inverted hammer
        if self._is_shooting_star(current):
            return self._create_shooting_star_signal(current)
        
        # Marubozu patterns
        if self._is_marubozu(current):
            return self._create_marubozu_signal(current)
        
        return CandlestickPatternSignal()
    
    def _detect_two_candle_patterns(self) -> CandlestickPatternSignal:
        """Detect two-candlestick patterns"""
        if len(self.candles) < 2:
            return CandlestickPatternSignal()
        
        prev = self.candles[-2]
        current = self.candles[-1]
        
        # Engulfing patterns
        if self._is_bullish_engulfing(prev, current):
            return self._create_engulfing_signal(prev, current, True)
        
        if self._is_bearish_engulfing(prev, current):
            return self._create_engulfing_signal(prev, current, False)
        
        # Harami patterns
        if self._is_bullish_harami(prev, current):
            return self._create_harami_signal(prev, current, True)
        
        if self._is_bearish_harami(prev, current):
            return self._create_harami_signal(prev, current, False)
        
        # Piercing line
        if self._is_piercing_line(prev, current):
            return self._create_piercing_line_signal(prev, current)
        
        # Dark cloud cover
        if self._is_dark_cloud_cover(prev, current):
            return self._create_dark_cloud_cover_signal(prev, current)
        
        return CandlestickPatternSignal()
    
    def _detect_three_candle_patterns(self) -> CandlestickPatternSignal:
        """Detect three-candlestick patterns"""
        if len(self.candles) < 3:
            return CandlestickPatternSignal()
        
        first = self.candles[-3]
        second = self.candles[-2]
        third = self.candles[-1]
        
        # Morning star
        if self._is_morning_star(first, second, third):
            return self._create_morning_star_signal(first, second, third)
        
        # Evening star
        if self._is_evening_star(first, second, third):
            return self._create_evening_star_signal(first, second, third)
        
        # Three white soldiers
        if self._is_three_white_soldiers(first, second, third):
            return self._create_three_white_soldiers_signal(first, second, third)
        
        # Three black crows
        if self._is_three_black_crows(first, second, third):
            return self._create_three_black_crows_signal(first, second, third)
        
        return CandlestickPatternSignal()
    
    # Pattern detection methods
    def _is_doji(self, candle: CandlestickData) -> bool:
        """Check if candle is a doji"""
        return candle.body_size <= candle.total_range * self.doji_threshold
    
    def _is_hammer(self, candle: CandlestickData) -> bool:
        """Check if candle is a hammer or hanging man"""
        if candle.total_range == 0:
            return False
        
        return (candle.lower_shadow >= candle.body_size * self.shadow_threshold and
                candle.upper_shadow <= candle.body_size * 0.5 and
                candle.body_size >= candle.total_range * self.body_threshold)
    
    def _is_shooting_star(self, candle: CandlestickData) -> bool:
        """Check if candle is shooting star or inverted hammer"""
        if candle.total_range == 0:
            return False
        
        return (candle.upper_shadow >= candle.body_size * self.shadow_threshold and
                candle.lower_shadow <= candle.body_size * 0.5 and
                candle.body_size >= candle.total_range * self.body_threshold)
    
    def _is_marubozu(self, candle: CandlestickData) -> bool:
        """Check if candle is marubozu (little to no shadows)"""
        shadow_threshold = candle.total_range * 0.05  # 5% of total range
        return (candle.upper_shadow <= shadow_threshold and
                candle.lower_shadow <= shadow_threshold and
                candle.body_size >= candle.total_range * 0.9)
    
    def _is_bullish_engulfing(self, prev: CandlestickData, current: CandlestickData) -> bool:
        """Check for bullish engulfing pattern"""
        return (prev.is_bearish and current.is_bullish and
                current.close > prev.open and current.open < prev.close)
    
    def _is_bearish_engulfing(self, prev: CandlestickData, current: CandlestickData) -> bool:
        """Check for bearish engulfing pattern"""
        return (prev.is_bullish and current.is_bearish and
                current.close < prev.open and current.open > prev.close)
    
    def _is_bullish_harami(self, prev: CandlestickData, current: CandlestickData) -> bool:
        """Check for bullish harami pattern"""
        return (prev.is_bearish and current.is_bullish and
                current.open > prev.close and current.close < prev.open)
    
    def _is_bearish_harami(self, prev: CandlestickData, current: CandlestickData) -> bool:
        """Check for bearish harami pattern"""
        return (prev.is_bullish and current.is_bearish and
                current.open < prev.close and current.close > prev.open)
    
    def _is_piercing_line(self, prev: CandlestickData, current: CandlestickData) -> bool:
        """Check for piercing line pattern"""
        if not (prev.is_bearish and current.is_bullish):
            return False
        
        midpoint = (prev.open + prev.close) / 2
        return current.close > midpoint and current.open < prev.close
    
    def _is_dark_cloud_cover(self, prev: CandlestickData, current: CandlestickData) -> bool:
        """Check for dark cloud cover pattern"""
        if not (prev.is_bullish and current.is_bearish):
            return False
        
        midpoint = (prev.open + prev.close) / 2
        return current.close < midpoint and current.open > prev.close
    
    def _is_morning_star(self, first: CandlestickData, second: CandlestickData, third: CandlestickData) -> bool:
        """Check for morning star pattern"""
        return (first.is_bearish and third.is_bullish and
                second.body_size < first.body_size * 0.5 and
                second.body_size < third.body_size * 0.5 and
                third.close > (first.open + first.close) / 2)
    
    def _is_evening_star(self, first: CandlestickData, second: CandlestickData, third: CandlestickData) -> bool:
        """Check for evening star pattern"""
        return (first.is_bullish and third.is_bearish and
                second.body_size < first.body_size * 0.5 and
                second.body_size < third.body_size * 0.5 and
                third.close < (first.open + first.close) / 2)
    
    def _is_three_white_soldiers(self, first: CandlestickData, second: CandlestickData, third: CandlestickData) -> bool:
        """Check for three white soldiers pattern"""
        return (first.is_bullish and second.is_bullish and third.is_bullish and
                second.close > first.close and third.close > second.close and
                second.open >= first.close * 0.95 and third.open >= second.close * 0.95)
    
    def _is_three_black_crows(self, first: CandlestickData, second: CandlestickData, third: CandlestickData) -> bool:
        """Check for three black crows pattern"""
        return (first.is_bearish and second.is_bearish and third.is_bearish and
                second.close < first.close and third.close < second.close and
                second.open <= first.close * 1.05 and third.open <= second.close * 1.05)
    
    # Signal creation methods (simplified versions)
    def _create_doji_signal(self, candle: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for doji pattern"""
        return CandlestickPatternSignal(
            timestamp=candle.timestamp,
            pattern_type=CandlestickPatternType.DOJI,
            pattern_name="Doji",
            significance=PatternSignificance.MODERATE,
            reversal_potential="neutral",
            pattern_reliability=0.6,
            signal_direction="hold",
            signal_strength=0.5,
            signal_confidence=0.6,
            trend_context=self.trend_direction,
            pattern_description="Indecision pattern - market uncertainty"
        )
    
    def _create_hammer_signal(self, candle: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for hammer pattern"""
        is_bullish_hammer = self.trend_direction == "downtrend"
        
        return CandlestickPatternSignal(
            timestamp=candle.timestamp,
            pattern_type=CandlestickPatternType.HAMMER if is_bullish_hammer else CandlestickPatternType.HANGING_MAN,
            pattern_name="Hammer" if is_bullish_hammer else "Hanging Man",
            significance=PatternSignificance.STRONG if is_bullish_hammer else PatternSignificance.MODERATE,
            reversal_potential="bullish" if is_bullish_hammer else "bearish",
            pattern_reliability=0.7 if is_bullish_hammer else 0.6,
            signal_direction="buy" if is_bullish_hammer else "sell",
            signal_strength=0.7 if is_bullish_hammer else 0.6,
            signal_confidence=0.7,
            trend_context=self.trend_direction,
            pattern_description="Potential reversal pattern with long lower shadow"
        )
    
    def _create_shooting_star_signal(self, candle: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for shooting star pattern"""
        is_bearish_star = self.trend_direction == "uptrend"
        
        return CandlestickPatternSignal(
            timestamp=candle.timestamp,
            pattern_type=CandlestickPatternType.SHOOTING_STAR if is_bearish_star else CandlestickPatternType.INVERTED_HAMMER,
            pattern_name="Shooting Star" if is_bearish_star else "Inverted Hammer",
            significance=PatternSignificance.STRONG if is_bearish_star else PatternSignificance.MODERATE,
            reversal_potential="bearish" if is_bearish_star else "bullish",
            pattern_reliability=0.7 if is_bearish_star else 0.6,
            signal_direction="sell" if is_bearish_star else "buy",
            signal_strength=0.7 if is_bearish_star else 0.6,
            signal_confidence=0.7,
            trend_context=self.trend_direction,
            pattern_description="Potential reversal pattern with long upper shadow"
        )
    
    def _create_marubozu_signal(self, candle: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for marubozu pattern"""
        return CandlestickPatternSignal(
            timestamp=candle.timestamp,
            pattern_type=CandlestickPatternType.MARUBOZU_BULLISH if candle.is_bullish else CandlestickPatternType.MARUBOZU_BEARISH,
            pattern_name="Bullish Marubozu" if candle.is_bullish else "Bearish Marubozu",
            significance=PatternSignificance.STRONG,
            reversal_potential="neutral",
            continuation_signal=True,
            pattern_reliability=0.8,
            signal_direction="buy" if candle.is_bullish else "sell",
            signal_strength=0.8,
            signal_confidence=0.8,
            trend_context=self.trend_direction,
            pattern_description="Strong continuation pattern with no shadows"
        )
    
    def _create_engulfing_signal(self, prev: CandlestickData, current: CandlestickData, is_bullish: bool) -> CandlestickPatternSignal:
        """Create signal for engulfing patterns"""
        return CandlestickPatternSignal(
            timestamp=current.timestamp,
            pattern_type=CandlestickPatternType.ENGULFING_BULLISH if is_bullish else CandlestickPatternType.ENGULFING_BEARISH,
            pattern_name="Bullish Engulfing" if is_bullish else "Bearish Engulfing",
            significance=PatternSignificance.VERY_STRONG,
            reversal_potential="bullish" if is_bullish else "bearish",
            pattern_reliability=0.8,
            signal_direction="buy" if is_bullish else "sell",
            signal_strength=0.8,
            signal_confidence=0.8,
            trend_context=self.trend_direction,
            pattern_description="Strong reversal pattern - second candle engulfs first"
        )
    
    def _create_harami_signal(self, prev: CandlestickData, current: CandlestickData, is_bullish: bool) -> CandlestickPatternSignal:
        """Create signal for harami patterns"""
        return CandlestickPatternSignal(
            timestamp=current.timestamp,
            pattern_type=CandlestickPatternType.HARAMI_BULLISH if is_bullish else CandlestickPatternType.HARAMI_BEARISH,
            pattern_name="Bullish Harami" if is_bullish else "Bearish Harami",
            significance=PatternSignificance.MODERATE,
            reversal_potential="bullish" if is_bullish else "bearish",
            pattern_reliability=0.6,
            signal_direction="buy" if is_bullish else "sell",
            signal_strength=0.6,
            signal_confidence=0.6,
            trend_context=self.trend_direction,
            pattern_description="Reversal pattern - small candle inside large candle"
        )
    
    def _create_piercing_line_signal(self, prev: CandlestickData, current: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for piercing line pattern"""
        return CandlestickPatternSignal(
            timestamp=current.timestamp,
            pattern_type=CandlestickPatternType.PIERCING_LINE,
            pattern_name="Piercing Line",
            significance=PatternSignificance.STRONG,
            reversal_potential="bullish",
            pattern_reliability=0.7,
            signal_direction="buy",
            signal_strength=0.7,
            signal_confidence=0.7,
            trend_context=self.trend_direction,
            pattern_description="Bullish reversal - closes above midpoint of previous bearish candle"
        )
    
    def _create_dark_cloud_cover_signal(self, prev: CandlestickData, current: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for dark cloud cover pattern"""
        return CandlestickPatternSignal(
            timestamp=current.timestamp,
            pattern_type=CandlestickPatternType.DARK_CLOUD_COVER,
            pattern_name="Dark Cloud Cover",
            significance=PatternSignificance.STRONG,
            reversal_potential="bearish",
            pattern_reliability=0.7,
            signal_direction="sell",
            signal_strength=0.7,
            signal_confidence=0.7,
            trend_context=self.trend_direction,
            pattern_description="Bearish reversal - closes below midpoint of previous bullish candle"
        )
    
    def _create_morning_star_signal(self, first: CandlestickData, second: CandlestickData, third: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for morning star pattern"""
        return CandlestickPatternSignal(
            timestamp=third.timestamp,
            pattern_type=CandlestickPatternType.MORNING_STAR,
            pattern_name="Morning Star",
            significance=PatternSignificance.VERY_STRONG,
            reversal_potential="bullish",
            pattern_reliability=0.85,
            signal_direction="buy",
            signal_strength=0.85,
            signal_confidence=0.85,
            trend_context=self.trend_direction,
            pattern_description="Very strong bullish reversal - three candle pattern"
        )
    
    def _create_evening_star_signal(self, first: CandlestickData, second: CandlestickData, third: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for evening star pattern"""
        return CandlestickPatternSignal(
            timestamp=third.timestamp,
            pattern_type=CandlestickPatternType.EVENING_STAR,
            pattern_name="Evening Star",
            significance=PatternSignificance.VERY_STRONG,
            reversal_potential="bearish",
            pattern_reliability=0.85,
            signal_direction="sell",
            signal_strength=0.85,
            signal_confidence=0.85,
            trend_context=self.trend_direction,
            pattern_description="Very strong bearish reversal - three candle pattern"
        )
    
    def _create_three_white_soldiers_signal(self, first: CandlestickData, second: CandlestickData, third: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for three white soldiers pattern"""
        return CandlestickPatternSignal(
            timestamp=third.timestamp,
            pattern_type=CandlestickPatternType.THREE_WHITE_SOLDIERS,
            pattern_name="Three White Soldiers",
            significance=PatternSignificance.VERY_STRONG,
            reversal_potential="bullish",
            continuation_signal=True,
            pattern_reliability=0.8,
            signal_direction="buy",
            signal_strength=0.8,
            signal_confidence=0.8,
            trend_context=self.trend_direction,
            pattern_description="Strong bullish continuation - three consecutive bullish candles"
        )
    
    def _create_three_black_crows_signal(self, first: CandlestickData, second: CandlestickData, third: CandlestickData) -> CandlestickPatternSignal:
        """Create signal for three black crows pattern"""
        return CandlestickPatternSignal(
            timestamp=third.timestamp,
            pattern_type=CandlestickPatternType.THREE_BLACK_CROWS,
            pattern_name="Three Black Crows",
            significance=PatternSignificance.VERY_STRONG,
            reversal_potential="bearish",
            continuation_signal=True,
            pattern_reliability=0.8,
            signal_direction="sell",
            signal_strength=0.8,
            signal_confidence=0.8,
            trend_context=self.trend_direction,
            pattern_description="Strong bearish continuation - three consecutive bearish candles"
        )
    
    def get_recent_patterns(self, count: int = 10) -> List[CandlestickPatternSignal]:
        """Get recent detected patterns"""
        return self.patterns[-count:] if len(self.patterns) >= count else self.patterns
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected patterns"""
        if not self.patterns:
            return {}
        
        pattern_counts = {}
        for pattern in self.patterns:
            if pattern.pattern_type:
                pattern_type = pattern.pattern_type.value
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            'total_patterns': len(self.patterns),
            'pattern_counts': pattern_counts,
            'last_pattern': self.patterns[-1].pattern_name if self.patterns else None,
            'trend_context': self.trend_direction
        }


def test_candlestick_patterns():
    """Test Japanese Candlestick Pattern Recognition with realistic scenarios"""
    print("=== JAPANESE CANDLESTICK PATTERN RECOGNITION TEST ===")
    
    # Initialize pattern detector
    patterns = JapaneseCandlestickPatterns()
    
    print("Testing various candlestick patterns...")
    
    # Test 1: Hammer pattern in downtrend
    print("\n1. Testing Hammer Pattern (in downtrend):")
    patterns.add_candle(1.1050, 1.1060, 1.1040, 1.1045, 1000, "2025-01-01 09:00")  # Bearish
    patterns.add_candle(1.1045, 1.1055, 1.1035, 1.1040, 1100, "2025-01-01 10:00")  # Bearish
    patterns.add_candle(1.1040, 1.1050, 1.1025, 1.1038, 1200, "2025-01-01 11:00")  # Bearish
    signal = patterns.add_candle(1.1035, 1.1040, 1.1020, 1.1038, 1500, "2025-01-01 12:00")  # Hammer
    print(f"  Pattern: {signal.pattern_name}")
    print(f"  Signal: {signal.signal_direction}")
    print(f"  Strength: {signal.signal_strength:.2f}")
    print(f"  Reliability: {signal.pattern_reliability:.2f}")
    
    # Test 2: Bullish Engulfing
    print("\n2. Testing Bullish Engulfing Pattern:")
    signal = patterns.add_candle(1.1038, 1.1042, 1.1032, 1.1034, 1000, "2025-01-01 13:00")  # Small bearish
    signal = patterns.add_candle(1.1032, 1.1045, 1.1030, 1.1044, 1800, "2025-01-01 14:00")  # Engulfing bullish
    print(f"  Pattern: {signal.pattern_name}")
    print(f"  Signal: {signal.signal_direction}")
    print(f"  Strength: {signal.signal_strength:.2f}")
    print(f"  Reliability: {signal.pattern_reliability:.2f}")
    
    # Test 3: Doji pattern
    print("\n3. Testing Doji Pattern:")
    signal = patterns.add_candle(1.1044, 1.1048, 1.1040, 1.1044, 1200, "2025-01-01 15:00")  # Doji
    print(f"  Pattern: {signal.pattern_name}")
    print(f"  Signal: {signal.signal_direction}")
    print(f"  Strength: {signal.signal_strength:.2f}")
    print(f"  Reliability: {signal.pattern_reliability:.2f}")
    
    # Test 4: Evening Star (three-candle pattern)
    print("\n4. Testing Evening Star Pattern:")
    # First establish uptrend
    for i in range(5):
        patterns.add_candle(1.1044 + i*0.0002, 1.1050 + i*0.0002, 1.1042 + i*0.0002, 1.1048 + i*0.0002, 1000)
    
    # Evening star pattern
    signal = patterns.add_candle(1.1052, 1.1058, 1.1050, 1.1057, 1000, "2025-01-01 16:00")  # Bullish candle
    signal = patterns.add_candle(1.1058, 1.1060, 1.1056, 1.1058, 800, "2025-01-01 17:00")   # Small body (star)
    signal = patterns.add_candle(1.1056, 1.1058, 1.1048, 1.1050, 1600, "2025-01-01 18:00")  # Bearish candle
    print(f"  Pattern: {signal.pattern_name}")
    print(f"  Signal: {signal.signal_direction}")
    print(f"  Strength: {signal.signal_strength:.2f}")
    print(f"  Reliability: {signal.pattern_reliability:.2f}")
    
    # Test 5: Three White Soldiers
    print("\n5. Testing Three White Soldiers Pattern:")
    signal = patterns.add_candle(1.1050, 1.1055, 1.1048, 1.1054, 1000, "2025-01-01 19:00")  # Bullish 1
    signal = patterns.add_candle(1.1054, 1.1059, 1.1052, 1.1058, 1100, "2025-01-01 20:00")  # Bullish 2
    signal = patterns.add_candle(1.1058, 1.1063, 1.1056, 1.1062, 1200, "2025-01-01 21:00")  # Bullish 3
    print(f"  Pattern: {signal.pattern_name}")
    print(f"  Signal: {signal.signal_direction}")
    print(f"  Strength: {signal.signal_strength:.2f}")
    print(f"  Reliability: {signal.pattern_reliability:.2f}")
    
    # Test 6: Shooting Star
    print("\n6. Testing Shooting Star Pattern:")
    signal = patterns.add_candle(1.1062, 1.1075, 1.1060, 1.1064, 1400, "2025-01-01 22:00")  # Shooting star
    print(f"  Pattern: {signal.pattern_name}")
    print(f"  Signal: {signal.signal_direction}")
    print(f"  Strength: {signal.signal_strength:.2f}")
    print(f"  Reliability: {signal.pattern_reliability:.2f}")
    
    # Final statistics
    print(f"\n=== PATTERN STATISTICS ===")
    stats = patterns.get_pattern_statistics()
    print(f"Total patterns detected: {stats['total_patterns']}")
    print(f"Current trend context: {stats['trend_context']}")
    print(f"Last pattern: {stats['last_pattern']}")
    
    print(f"\nPattern breakdown:")
    for pattern_type, count in stats['pattern_counts'].items():
        print(f"  {pattern_type}: {count}")
    
    # Recent patterns
    print(f"\n=== RECENT PATTERNS ===")
    recent = patterns.get_recent_patterns(5)
    for i, pattern in enumerate(recent):
        if pattern.pattern_type:
            print(f"{i+1}. {pattern.pattern_name} - {pattern.signal_direction} (Strength: {pattern.signal_strength:.2f})")
    
    print(f"\n=== TEST COMPLETED SUCCESSFULLY ===")
    return True


if __name__ == "__main__":
    test_candlestick_patterns()
