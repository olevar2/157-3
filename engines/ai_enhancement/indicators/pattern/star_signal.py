"""
Morning Star and Evening Star Candlestick Pattern Implementation
CCI Gold Standard - Production Ready

The Star patterns are three-candle reversal formations:
- Morning Star: Bearish candle + Small candle (star) + Bullish candle (bullish reversal)
- Evening Star: Bullish candle + Small candle (star) + Bearish candle (bearish reversal)

Key Features:
- Three-candle pattern with gap characteristics
- Middle candle (star) has small real body
- Third candle confirms reversal direction
- Gaps between candles enhance pattern reliability
- Context-dependent reversal signals

Market Psychology:
- Morning Star: Bears losing control, bulls taking over
- Evening Star: Bulls exhausted, bears gaining control
- Star candle represents indecision and momentum shift
- Third candle confirms new trend direction

Author: Platform3 AI Enhancement Engine
Version: 1.0.0
Category: Candlestick Pattern Recognition
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from decimal import Decimal
import numpy as np
import pandas as pd
from enum import Enum

from engines.indicator_base import IndicatorBase


class StarType(Enum):
    """Enumeration for different star pattern types."""
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    DOJI_MORNING_STAR = "doji_morning_star"
    DOJI_EVENING_STAR = "doji_evening_star"
    NO_PATTERN = "no_pattern"


@dataclass
class StarSignalResult:
    """
    Result data class for Star pattern detection.
    
    Attributes:
        is_detected: Whether a Star pattern is detected
        star_type: Type of star pattern detected
        pattern_strength: Strength score (0.0 to 1.0)
        first_candle_data: First candle analysis
        star_candle_data: Star (middle) candle analysis
        third_candle_data: Third candle analysis
        gap_analysis: Gap characteristics between candles
        star_characteristics: Analysis of star candle properties
        volume_confirmation: Volume trend analysis if available
        reliability_score: Pattern reliability assessment
        reversal_potential: Potential for trend reversal
        entry_level: Suggested entry price level
        target_levels: Potential target price levels
        stop_loss: Suggested stop-loss level
        pattern_context: Market context analysis
        formation_quality: Quality assessment of pattern formation
        market_conditions: Overall market environment assessment
        signal_direction: Direction of anticipated move (bullish/bearish)
        confirmation_strength: Strength of reversal confirmation
    """
    is_detected: bool
    star_type: StarType
    pattern_strength: float
    first_candle_data: Dict[str, Any]
    star_candle_data: Dict[str, Any]
    third_candle_data: Dict[str, Any]
    gap_analysis: Dict[str, Any]
    star_characteristics: Dict[str, Any]
    volume_confirmation: Optional[Dict[str, Any]]
    reliability_score: float
    reversal_potential: float
    entry_level: Optional[float]
    target_levels: List[float]
    stop_loss: Optional[float]
    pattern_context: str
    formation_quality: float
    market_conditions: str
    signal_direction: str
    confirmation_strength: float


class StarSignal(IndicatorBase):
    """
    Morning Star and Evening Star Pattern Detector
    
    Implements advanced detection for both Morning Star and Evening Star reversal patterns
    with sophisticated validation criteria and market context analysis.
    """
    
    def __init__(self, 
                 min_first_body_ratio: float = 0.6,
                 max_star_body_ratio: float = 0.3,
                 min_third_body_ratio: float = 0.6,
                 min_gap_threshold: float = 0.001,
                 volume_confirmation: bool = True,
                 lookback_periods: int = 20):
        """
        Initialize Star pattern detector.
        
        Args:
            min_first_body_ratio: Minimum body-to-range ratio for first candle
            max_star_body_ratio: Maximum body-to-range ratio for star candle
            min_third_body_ratio: Minimum body-to-range ratio for third candle
            min_gap_threshold: Minimum gap size as ratio of price
            volume_confirmation: Whether to use volume for confirmation
            lookback_periods: Periods to analyze for context
        """
        super().__init__()
        self.min_first_body_ratio = min_first_body_ratio
        self.max_star_body_ratio = max_star_body_ratio
        self.min_third_body_ratio = min_third_body_ratio
        self.min_gap_threshold = min_gap_threshold
        self.volume_confirmation = volume_confirmation
        self.lookback_periods = lookback_periods
        
        # Pattern detection thresholds
        self.min_pattern_strength = 0.60
        self.strong_pattern_threshold = 0.80
        
        # Market context parameters
        self.trend_analysis_periods = 14
        self.volatility_periods = 10
        
    def calculate(self, data: pd.DataFrame) -> StarSignalResult:
        """
        Calculate Star pattern detection.
        
        Args:
            data: OHLCV DataFrame with at least 3 periods
            
        Returns:
            StarSignalResult with comprehensive pattern analysis
        """
        try:
            if len(data) < 3:
                return self._create_no_pattern_result("Insufficient data")
                
            # Get the last three candles
            candles = data.tail(3).copy()
            
            # Basic candle analysis
            candle_analysis = self._analyze_individual_candles(candles)
            
            # Pattern validation and type detection
            pattern_detected, star_type, strength = self._validate_star_pattern(candles, candle_analysis)
            
            if not pattern_detected:
                return self._create_no_pattern_result("Pattern criteria not met")
                
            # Advanced analysis
            gap_analysis = self._analyze_gaps(candles)
            star_characteristics = self._analyze_star_candle(candles, candle_analysis[1])
            volume_data = self._analyze_volume_confirmation(candles) if self.volume_confirmation else None
            
            # Market context analysis
            context_analysis = self._analyze_market_context(data)
            
            # Calculate reliability and targets
            reliability = self._calculate_reliability_score(candles, strength, context_analysis, star_type)
            reversal_potential = self._assess_reversal_potential(candles, context_analysis, star_type)
            
            # Generate trading levels
            entry_level = float(candles.iloc[-1]['close'])
            target_levels = self._calculate_target_levels(candles, star_type)
            stop_loss = self._calculate_stop_loss(candles, star_type)
            
            # Determine signal direction
            signal_direction = "Bullish" if star_type in [StarType.MORNING_STAR, StarType.DOJI_MORNING_STAR] else "Bearish"
            
            # Confirmation strength
            confirmation_strength = self._calculate_confirmation_strength(candles, candle_analysis, star_type)
            
            return StarSignalResult(
                is_detected=True,
                star_type=star_type,
                pattern_strength=strength,
                first_candle_data=candle_analysis[0],
                star_candle_data=candle_analysis[1],
                third_candle_data=candle_analysis[2],
                gap_analysis=gap_analysis,
                star_characteristics=star_characteristics,
                volume_confirmation=volume_data,
                reliability_score=reliability,
                reversal_potential=reversal_potential,
                entry_level=entry_level,
                target_levels=target_levels,
                stop_loss=stop_loss,
                pattern_context=context_analysis['context_description'],
                formation_quality=self._assess_formation_quality(candles, strength, star_type),
                market_conditions=context_analysis['market_conditions'],
                signal_direction=signal_direction,
                confirmation_strength=confirmation_strength
            )
            
        except Exception as e:
            return self._create_error_result(f"Calculation error: {str(e)}")
    
    def _analyze_individual_candles(self, candles: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze each individual candle in the pattern."""
        analysis = []
        
        for i, (_, candle) in enumerate(candles.iterrows()):
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])
            
            # Basic measurements
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            # Ratios
            body_ratio = body_size / total_range if total_range > 0 else 0
            upper_shadow_ratio = upper_shadow / total_range if total_range > 0 else 0
            lower_shadow_ratio = lower_shadow / total_range if total_range > 0 else 0
            
            # Candle characteristics
            is_bullish = close_price > open_price
            is_bearish = close_price < open_price
            is_doji = abs(close_price - open_price) / ((high_price + low_price) / 2) < 0.01
            
            analysis.append({
                'candle_index': i + 1,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'body_size': body_size,
                'total_range': total_range,
                'upper_shadow': upper_shadow,
                'lower_shadow': lower_shadow,
                'body_ratio': body_ratio,
                'upper_shadow_ratio': upper_shadow_ratio,
                'lower_shadow_ratio': lower_shadow_ratio,
                'is_bullish': is_bullish,
                'is_bearish': is_bearish,
                'is_doji': is_doji,
                'candle_strength': body_ratio if not is_doji else (upper_shadow_ratio + lower_shadow_ratio)
            })
            
        return analysis
    
    def _validate_star_pattern(self, candles: pd.DataFrame, 
                              candle_analysis: List[Dict[str, Any]]) -> Tuple[bool, StarType, float]:
        """Validate Star pattern criteria and determine type."""
        first_candle = candle_analysis[0]
        star_candle = candle_analysis[1]
        third_candle = candle_analysis[2]
        
        strength_components = []
        
        # Check for Morning Star pattern
        if (first_candle['is_bearish'] and 
            first_candle['body_ratio'] >= self.min_first_body_ratio and
            star_candle['body_ratio'] <= self.max_star_body_ratio and
            third_candle['is_bullish'] and
            third_candle['body_ratio'] >= self.min_third_body_ratio):
            
            strength_components.append(0.4)  # Base pattern recognition
            
            # Check for gaps
            gap_down = self._check_gap_down(candles.iloc[0], candles.iloc[1])
            gap_up = self._check_gap_up(candles.iloc[1], candles.iloc[2])
            
            if gap_down:
                strength_components.append(0.15)
            if gap_up:
                strength_components.append(0.15)
            
            # Check third candle penetration into first candle
            penetration = self._check_penetration(candles.iloc[0], candles.iloc[2])
            if penetration > 0.5:
                strength_components.append(0.2)
            elif penetration > 0.3:
                strength_components.append(0.1)
            
            # Determine if it's a Doji Morning Star
            if star_candle['is_doji']:
                star_type = StarType.DOJI_MORNING_STAR
                strength_components.append(0.1)  # Doji bonus
            else:
                star_type = StarType.MORNING_STAR
            
            total_strength = sum(strength_components)
            return total_strength >= self.min_pattern_strength, star_type, total_strength
        
        # Check for Evening Star pattern
        elif (first_candle['is_bullish'] and 
              first_candle['body_ratio'] >= self.min_first_body_ratio and
              star_candle['body_ratio'] <= self.max_star_body_ratio and
              third_candle['is_bearish'] and
              third_candle['body_ratio'] >= self.min_third_body_ratio):
            
            strength_components.append(0.4)  # Base pattern recognition
            
            # Check for gaps
            gap_up = self._check_gap_up(candles.iloc[0], candles.iloc[1])
            gap_down = self._check_gap_down(candles.iloc[1], candles.iloc[2])
            
            if gap_up:
                strength_components.append(0.15)
            if gap_down:
                strength_components.append(0.15)
            
            # Check third candle penetration into first candle
            penetration = self._check_penetration(candles.iloc[0], candles.iloc[2])
            if penetration > 0.5:
                strength_components.append(0.2)
            elif penetration > 0.3:
                strength_components.append(0.1)
            
            # Determine if it's a Doji Evening Star
            if star_candle['is_doji']:
                star_type = StarType.DOJI_EVENING_STAR
                strength_components.append(0.1)  # Doji bonus
            else:
                star_type = StarType.EVENING_STAR
            
            total_strength = sum(strength_components)
            return total_strength >= self.min_pattern_strength, star_type, total_strength
        
        return False, StarType.NO_PATTERN, 0.0
    
    def _check_gap_down(self, first_candle: pd.Series, second_candle: pd.Series) -> bool:
        """Check for gap down between two candles."""
        first_low = min(float(first_candle['open']), float(first_candle['close']))
        second_high = max(float(second_candle['open']), float(second_candle['close']))
        
        gap_size = (first_low - second_high) / first_low if first_low > 0 else 0
        return gap_size >= self.min_gap_threshold
    
    def _check_gap_up(self, first_candle: pd.Series, second_candle: pd.Series) -> bool:
        """Check for gap up between two candles."""
        first_high = max(float(first_candle['open']), float(first_candle['close']))
        second_low = min(float(second_candle['open']), float(second_candle['close']))
        
        gap_size = (second_low - first_high) / first_high if first_high > 0 else 0
        return gap_size >= self.min_gap_threshold
    
    def _check_penetration(self, first_candle: pd.Series, third_candle: pd.Series) -> float:
        """Check how much the third candle penetrates into the first candle's body."""
        first_open = float(first_candle['open'])
        first_close = float(first_candle['close'])
        third_close = float(third_candle['close'])
        
        first_body_size = abs(first_close - first_open)
        if first_body_size == 0:
            return 0.0
        
        # For Morning Star: check bullish penetration into bearish candle
        if first_close < first_open:  # First candle is bearish
            penetration = max(0, third_close - first_close) / first_body_size
        # For Evening Star: check bearish penetration into bullish candle
        else:  # First candle is bullish
            penetration = max(0, first_close - third_close) / first_body_size
        
        return min(penetration, 1.0)
    
    def _analyze_gaps(self, candles: pd.DataFrame) -> Dict[str, Any]:
        """Analyze gap characteristics in the pattern."""
        first_candle = candles.iloc[0]
        star_candle = candles.iloc[1]
        third_candle = candles.iloc[2]
        
        # Gap between first and star candle
        gap1_down = self._check_gap_down(first_candle, star_candle)
        gap1_up = self._check_gap_up(first_candle, star_candle)
        
        # Gap between star and third candle
        gap2_down = self._check_gap_down(star_candle, third_candle)
        gap2_up = self._check_gap_up(star_candle, third_candle)
        
        # Calculate gap sizes
        first_body_range = [min(float(first_candle['open']), float(first_candle['close'])),
                           max(float(first_candle['open']), float(first_candle['close']))]
        star_body_range = [min(float(star_candle['open']), float(star_candle['close'])),
                          max(float(star_candle['open']), float(star_candle['close']))]
        third_body_range = [min(float(third_candle['open']), float(third_candle['close'])),
                           max(float(third_candle['open']), float(third_candle['close']))]
        
        return {
            'gap1_down': gap1_down,
            'gap1_up': gap1_up,
            'gap2_down': gap2_down,
            'gap2_up': gap2_up,
            'has_gaps': gap1_down or gap1_up or gap2_down or gap2_up,
            'gap_quality': (gap1_down or gap1_up) and (gap2_down or gap2_up),
            'first_body_range': first_body_range,
            'star_body_range': star_body_range,
            'third_body_range': third_body_range
        }
    
    def _analyze_star_candle(self, candles: pd.DataFrame, star_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characteristics of the star candle."""
        star_candle = candles.iloc[1]
        
        # Position analysis
        first_high = float(candles.iloc[0]['high'])
        first_low = float(candles.iloc[0]['low'])
        third_high = float(candles.iloc[2]['high'])
        third_low = float(candles.iloc[2]['low'])
        
        star_high = float(star_candle['high'])
        star_low = float(star_candle['low'])
        
        # Check if star is isolated (doesn't overlap with adjacent candles)
        isolated_from_first = star_low > first_high or star_high < first_low
        isolated_from_third = star_low > third_high or star_high < third_low
        
        return {
            'body_ratio': star_analysis['body_ratio'],
            'is_doji': star_analysis['is_doji'],
            'upper_shadow_ratio': star_analysis['upper_shadow_ratio'],
            'lower_shadow_ratio': star_analysis['lower_shadow_ratio'],
            'isolated_from_first': isolated_from_first,
            'isolated_from_third': isolated_from_third,
            'fully_isolated': isolated_from_first and isolated_from_third,
            'shadow_balance': abs(star_analysis['upper_shadow_ratio'] - star_analysis['lower_shadow_ratio']),
            'star_quality': (1 - star_analysis['body_ratio']) * (1 - abs(star_analysis['upper_shadow_ratio'] - star_analysis['lower_shadow_ratio']))
        }    
    def _analyze_volume_confirmation(self, candles: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze volume confirmation for the pattern."""
        if 'volume' not in candles.columns:
            return None
            
        volumes = candles['volume'].tolist()
        
        # Volume analysis for star patterns
        # Typically expect decreasing volume on star candle, increasing on confirmation
        star_volume_decrease = volumes[1] < volumes[0]
        confirmation_volume_increase = volumes[2] > volumes[1]
        
        return {
            'volumes': volumes,
            'star_volume_decrease': star_volume_decrease,
            'confirmation_volume_increase': confirmation_volume_increase,
            'volume_pattern_ideal': star_volume_decrease and confirmation_volume_increase,
            'volume_trend_strength': self._calculate_volume_trend_strength(volumes)
        }
    
    def _calculate_volume_trend_strength(self, volumes: List[float]) -> float:
        """Calculate strength of volume trend."""
        if len(volumes) < 3:
            return 0.0
        
        # Ideal pattern: high -> low -> higher
        vol_drop = (volumes[0] - volumes[1]) / volumes[0] if volumes[0] > 0 else 0
        vol_rise = (volumes[2] - volumes[1]) / volumes[1] if volumes[1] > 0 else 0
        
        # Normalize and combine
        trend_strength = (vol_drop * 0.4 + vol_rise * 0.6) if vol_drop > 0 and vol_rise > 0 else 0.2
        return min(trend_strength, 1.0)
    
    def _analyze_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze broader market context for pattern reliability."""
        if len(data) < self.trend_analysis_periods:
            return {
                'context_description': 'Insufficient data for context analysis',
                'market_conditions': 'Unknown',
                'trend_strength': 0.5,
                'volatility_level': 0.5
            }
        
        # Calculate trend context
        recent_data = data.tail(self.trend_analysis_periods)
        closes = recent_data['close'].values
        
        # Simple trend analysis
        trend_slope = (closes[-1] - closes[0]) / len(closes)
        trend_strength = abs(trend_slope) / np.mean(closes)
        
        # Volatility analysis
        volatility_data = data.tail(self.volatility_periods)
        price_changes = volatility_data['close'].pct_change().dropna()
        volatility = price_changes.std() if len(price_changes) > 1 else 0
        
        # Determine market conditions
        if trend_slope > 0.001:
            market_condition = 'Bullish trend - Evening Star reversal potential'
            context_description = 'Pattern appears in uptrend - Potential bearish reversal'
        elif trend_slope < -0.001:
            market_condition = 'Bearish trend - Morning Star reversal potential'
            context_description = 'Pattern appears in downtrend - Potential bullish reversal'
        else:
            market_condition = 'Sideways market - Reversal or continuation'
            context_description = 'Pattern appears in consolidation - Direction unclear'
        
        return {
            'context_description': context_description,
            'market_conditions': market_condition,
            'trend_strength': min(trend_strength * 10, 1.0),
            'volatility_level': min(volatility * 100, 1.0),
            'trend_slope': trend_slope
        }
    
    def _calculate_reliability_score(self, candles: pd.DataFrame, pattern_strength: float,
                                   context: Dict[str, Any], star_type: StarType) -> float:
        """Calculate overall pattern reliability score."""
        # Base reliability from pattern strength
        reliability = pattern_strength * 0.6
        
        # Context adjustment based on star type
        is_morning_star = star_type in [StarType.MORNING_STAR, StarType.DOJI_MORNING_STAR]
        is_evening_star = star_type in [StarType.EVENING_STAR, StarType.DOJI_EVENING_STAR]
        
        if is_morning_star and 'Bearish trend' in context['market_conditions']:
            reliability += 0.2  # Morning star in downtrend is ideal
        elif is_evening_star and 'Bullish trend' in context['market_conditions']:
            reliability += 0.2  # Evening star in uptrend is ideal
        elif 'Sideways' in context['market_conditions']:
            reliability += 0.1  # Moderate reliability in sideways market
        
        # Trend strength adjustment
        reliability += context['trend_strength'] * 0.1
        
        # Volatility adjustment (prefer moderate volatility)
        volatility = context['volatility_level']
        if 0.3 <= volatility <= 0.7:
            reliability += 0.1
        
        return min(reliability, 1.0)
    
    def _assess_reversal_potential(self, candles: pd.DataFrame, 
                                 context: Dict[str, Any], star_type: StarType) -> float:
        """Assess potential for trend reversal."""
        # Base potential from pattern characteristics
        pattern_range = float(candles['high'].max()) - float(candles['low'].min())
        first_candle_range = float(candles.iloc[0]['high']) - float(candles.iloc[0]['low'])
        
        base_potential = min(pattern_range / first_candle_range, 1.0) * 0.5
        
        # Context adjustments
        is_morning_star = star_type in [StarType.MORNING_STAR, StarType.DOJI_MORNING_STAR]
        is_evening_star = star_type in [StarType.EVENING_STAR, StarType.DOJI_EVENING_STAR]
        
        if is_morning_star and 'Bearish trend' in context['market_conditions']:
            base_potential += 0.3  # High reversal potential
        elif is_evening_star and 'Bullish trend' in context['market_conditions']:
            base_potential += 0.3  # High reversal potential
        
        # Trend strength adjustment
        base_potential += context['trend_strength'] * 0.2
        
        return min(base_potential, 1.0)
    
    def _calculate_target_levels(self, candles: pd.DataFrame, star_type: StarType) -> List[float]:
        """Calculate potential target levels."""
        current_close = float(candles.iloc[-1]['close'])
        pattern_range = float(candles['high'].max()) - float(candles['low'].min())
        
        is_bullish = star_type in [StarType.MORNING_STAR, StarType.DOJI_MORNING_STAR]
        
        if is_bullish:
            # Bullish targets
            targets = [
                current_close + pattern_range * 0.5,  # Conservative target
                current_close + pattern_range * 1.0,  # Pattern height target
                current_close + pattern_range * 1.618  # Fibonacci extension
            ]
        else:
            # Bearish targets
            targets = [
                current_close - pattern_range * 0.5,  # Conservative target
                current_close - pattern_range * 1.0,  # Pattern height target
                current_close - pattern_range * 1.618  # Fibonacci extension
            ]
        
        return targets
    
    def _calculate_stop_loss(self, candles: pd.DataFrame, star_type: StarType) -> float:
        """Calculate suggested stop-loss level."""
        is_bullish = star_type in [StarType.MORNING_STAR, StarType.DOJI_MORNING_STAR]
        
        if is_bullish:
            # Stop below pattern low
            pattern_low = float(candles['low'].min())
            stop_buffer = (float(candles.iloc[-1]['close']) - pattern_low) * 0.1
            return pattern_low - stop_buffer
        else:
            # Stop above pattern high
            pattern_high = float(candles['high'].max())
            stop_buffer = (pattern_high - float(candles.iloc[-1]['close'])) * 0.1
            return pattern_high + stop_buffer
    
    def _calculate_confirmation_strength(self, candles: pd.DataFrame, 
                                       candle_analysis: List[Dict[str, Any]], 
                                       star_type: StarType) -> float:
        """Calculate strength of reversal confirmation."""
        third_candle = candle_analysis[2]
        
        # Base confirmation from third candle strength
        confirmation = third_candle['body_ratio'] * 0.6
        
        # Penetration bonus
        penetration = self._check_penetration(candles.iloc[0], candles.iloc[2])
        confirmation += penetration * 0.3
        
        # Range relationship bonus
        third_range = third_candle['total_range']
        first_range = candle_analysis[0]['total_range']
        
        if third_range >= first_range * 0.8:  # Third candle is substantial
            confirmation += 0.1
        
        return min(confirmation, 1.0)
    
    def _assess_formation_quality(self, candles: pd.DataFrame, strength: float, star_type: StarType) -> float:
        """Assess the overall quality of pattern formation."""
        # Base quality from pattern strength
        quality = strength * 0.7
        
        # Star candle isolation bonus
        star_candle = candles.iloc[1]
        first_candle = candles.iloc[0]
        third_candle = candles.iloc[2]
        
        # Check for proper star isolation
        star_high = float(star_candle['high'])
        star_low = float(star_candle['low'])
        
        first_body_top = max(float(first_candle['open']), float(first_candle['close']))
        first_body_bottom = min(float(first_candle['open']), float(first_candle['close']))
        
        third_body_top = max(float(third_candle['open']), float(third_candle['close']))
        third_body_bottom = min(float(third_candle['open']), float(third_candle['close']))
        
        # Isolation quality
        if star_low > max(first_body_top, third_body_top) or star_high < min(first_body_bottom, third_body_bottom):
            quality += 0.2  # Perfect isolation
        elif star_low > min(first_body_top, third_body_top) or star_high < max(first_body_bottom, third_body_bottom):
            quality += 0.1  # Partial isolation
        
        # Size relationship quality
        first_body = abs(float(first_candle['close']) - float(first_candle['open']))
        star_body = abs(float(star_candle['close']) - float(star_candle['open']))
        third_body = abs(float(third_candle['close']) - float(third_candle['open']))
        
        if star_body < min(first_body, third_body) * 0.5:
            quality += 0.1  # Star is appropriately small
        
        return min(quality, 1.0)
    
    def _create_no_pattern_result(self, reason: str) -> StarSignalResult:
        """Create result object when no pattern is detected."""
        return StarSignalResult(
            is_detected=False,
            star_type=StarType.NO_PATTERN,
            pattern_strength=0.0,
            first_candle_data={},
            star_candle_data={},
            third_candle_data={},
            gap_analysis={},
            star_characteristics={},
            volume_confirmation=None,
            reliability_score=0.0,
            reversal_potential=0.0,
            entry_level=None,
            target_levels=[],
            stop_loss=None,
            pattern_context=reason,
            formation_quality=0.0,
            market_conditions='No pattern detected',
            signal_direction='None',
            confirmation_strength=0.0
        )
    
    def _create_error_result(self, error_msg: str) -> StarSignalResult:
        """Create result object when an error occurs."""
        return StarSignalResult(
            is_detected=False,
            star_type=StarType.NO_PATTERN,
            pattern_strength=0.0,
            first_candle_data={},
            star_candle_data={},
            third_candle_data={},
            gap_analysis={},
            star_characteristics={},
            volume_confirmation=None,
            reliability_score=0.0,
            reversal_potential=0.0,
            entry_level=None,
            target_levels=[],
            stop_loss=None,
            pattern_context=f'Error: {error_msg}',
            formation_quality=0.0,
            market_conditions='Error in calculation',
            signal_direction='None',
            confirmation_strength=0.0
        )


# Test and demonstration code
if __name__ == "__main__":
    # Create sample data for Morning Star testing
    morning_star_data = pd.DataFrame({
        'open': [105.0, 102.5, 101.0, 101.2, 103.5],
        'high': [105.5, 103.0, 101.3, 101.5, 104.8],
        'low': [102.0, 100.8, 100.7, 100.9, 103.0],
        'close': [102.2, 101.1, 101.1, 104.2, 104.5],
        'volume': [15000, 8000, 12000, 18000, 20000]
    })
    
    # Create sample data for Evening Star testing
    evening_star_data = pd.DataFrame({
        'open': [98.0, 101.8, 102.2, 102.0, 99.5],
        'high': [98.5, 102.3, 102.5, 102.1, 100.0],
        'low': [97.2, 101.5, 101.8, 99.0, 98.8],
        'close': [101.5, 102.0, 102.0, 99.2, 99.0],
        'volume': [12000, 8000, 15000, 18000, 22000]
    })
    
    # Initialize detector
    detector = StarSignal(
        min_first_body_ratio=0.6,
        max_star_body_ratio=0.3,
        min_third_body_ratio=0.6,
        volume_confirmation=True
    )
    
    print("=== Star Pattern Detection Test ===")
    
    # Test Morning Star
    print("\n--- Morning Star Test ---")
    result = detector.calculate(morning_star_data)
    print(f"Pattern Detected: {result.is_detected}")
    print(f"Star Type: {result.star_type.value if result.star_type else 'None'}")
    print(f"Pattern Strength: {result.pattern_strength:.3f}")
    print(f"Signal Direction: {result.signal_direction}")
    print(f"Reliability Score: {result.reliability_score:.3f}")
    print(f"Reversal Potential: {result.reversal_potential:.3f}")
    print(f"Confirmation Strength: {result.confirmation_strength:.3f}")
    
    if result.is_detected:
        print(f"Entry Level: {result.entry_level}")
        print(f"Target Levels: {result.target_levels}")
        print(f"Stop Loss: {result.stop_loss}")
        print(f"Formation Quality: {result.formation_quality:.3f}")
        
        if result.volume_confirmation:
            print(f"Volume Pattern Ideal: {result.volume_confirmation['volume_pattern_ideal']}")
    
    # Test Evening Star
    print("\n--- Evening Star Test ---")
    result = detector.calculate(evening_star_data)
    print(f"Pattern Detected: {result.is_detected}")
    print(f"Star Type: {result.star_type.value if result.star_type else 'None'}")
    print(f"Pattern Strength: {result.pattern_strength:.3f}")
    print(f"Signal Direction: {result.signal_direction}")
    print(f"Reliability Score: {result.reliability_score:.3f}")
    print(f"Reversal Potential: {result.reversal_potential:.3f}")
    print(f"Confirmation Strength: {result.confirmation_strength:.3f}")
    
    if result.is_detected:
        print(f"Entry Level: {result.entry_level}")
        print(f"Target Levels: {result.target_levels}")
        print(f"Stop Loss: {result.stop_loss}")
        print(f"Formation Quality: {result.formation_quality:.3f}")
        
        if result.volume_confirmation:
            print(f"Volume Pattern Ideal: {result.volume_confirmation['volume_pattern_ideal']}")
    
    print("\n=== Test Complete ===")