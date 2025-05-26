"""
Short-Term Elliott Wave Analysis Engine
Specialized for 3-5 wave structures for quick trades (max 5 days).
Optimized for H4 timeframe with rapid pattern recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio

class WaveType(Enum):
    """Elliott Wave types for short-term patterns"""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    DIAGONAL = "diagonal"
    TRIANGLE = "triangle"
    FLAT = "flat"
    ZIGZAG = "zigzag"

class WaveDirection(Enum):
    """Wave direction"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

@dataclass
class WavePoint:
    """Individual wave point"""
    timestamp: datetime
    price: float
    volume: float
    wave_number: int
    wave_type: WaveType
    confidence: float

@dataclass
class ElliottWavePattern:
    """Complete Elliott Wave pattern"""
    symbol: str
    timeframe: str
    pattern_type: WaveType
    direction: WaveDirection
    waves: List[WavePoint]
    start_time: datetime
    end_time: datetime
    price_range: Tuple[float, float]
    confidence: float
    completion_percentage: float
    next_target: Optional[float]
    invalidation_level: float
    expected_duration: timedelta

@dataclass
class WaveAnalysisResult:
    """Elliott Wave analysis result"""
    symbol: str
    timestamp: datetime
    patterns: List[ElliottWavePattern]
    active_pattern: Optional[ElliottWavePattern]
    signal_strength: float
    trade_recommendation: str
    entry_level: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: float

class ShortTermElliottWaves:
    """
    Short-Term Elliott Wave Analysis Engine
    Specialized for H4 timeframe with 1-5 day maximum patterns
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.min_wave_points = 3
        self.max_wave_points = 5
        self.max_pattern_duration = timedelta(days=5)
        self.min_confidence_threshold = 0.6

        # Wave ratio constraints for short-term patterns
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        self.wave_ratio_tolerance = 0.1

        # Pattern cache for performance
        self.pattern_cache = {}
        self.cache_duration = timedelta(minutes=15)

    async def analyze_waves(self,
                           symbol: str,
                           price_data: pd.DataFrame,
                           timeframe: str = "H4") -> WaveAnalysisResult:
        """
        Analyze Elliott Wave patterns for short-term trading

        Args:
            symbol: Trading symbol
            price_data: OHLCV data
            timeframe: Analysis timeframe

        Returns:
            WaveAnalysisResult with patterns and signals
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{len(price_data)}"
            if self._is_cache_valid(cache_key):
                return self.pattern_cache[cache_key]['result']

            # Identify swing points
            swing_points = await self._identify_swing_points(price_data)

            # Find Elliott Wave patterns
            patterns = await self._find_wave_patterns(symbol, swing_points, timeframe)

            # Analyze active pattern
            active_pattern = self._get_active_pattern(patterns)

            # Generate trading signals
            signal_data = await self._generate_wave_signals(active_pattern, price_data)

            result = WaveAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                patterns=patterns,
                active_pattern=active_pattern,
                signal_strength=signal_data['strength'],
                trade_recommendation=signal_data['recommendation'],
                entry_level=signal_data['entry'],
                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data['take_profit'],
                risk_reward_ratio=signal_data['risk_reward']
            )

            # Cache result
            self.pattern_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }

            return result

        except Exception as e:
            self.logger.error(f"Elliott Wave analysis error for {symbol}: {e}")
            return self._create_empty_result(symbol)

    async def _identify_swing_points(self, price_data: pd.DataFrame) -> List[WavePoint]:
        """Identify swing highs and lows for wave analysis"""
        swing_points = []

        # Use pivot points to identify swings
        high_pivots = self._find_pivot_points(price_data['high'].values, order=3)
        low_pivots = self._find_pivot_points(price_data['low'].values, order=3, find_peaks=False)

        # Combine and sort pivot points
        all_pivots = []

        for i, is_peak in enumerate(high_pivots):
            if is_peak:
                all_pivots.append({
                    'index': i,
                    'price': price_data.iloc[i]['high'],
                    'type': 'high',
                    'timestamp': price_data.index[i],
                    'volume': price_data.iloc[i]['volume']
                })

        for i, is_trough in enumerate(low_pivots):
            if is_trough:
                all_pivots.append({
                    'index': i,
                    'price': price_data.iloc[i]['low'],
                    'type': 'low',
                    'timestamp': price_data.index[i],
                    'volume': price_data.iloc[i]['volume']
                })

        # Sort by timestamp
        all_pivots.sort(key=lambda x: x['index'])

        # Convert to WavePoint objects
        for i, pivot in enumerate(all_pivots):
            swing_points.append(WavePoint(
                timestamp=pivot['timestamp'],
                price=pivot['price'],
                volume=pivot['volume'],
                wave_number=i + 1,
                wave_type=WaveType.IMPULSE,  # Will be refined later
                confidence=0.7  # Base confidence
            ))

        return swing_points

    def _find_pivot_points(self, data: np.ndarray, order: int = 3, find_peaks: bool = True) -> List[bool]:
        """Find pivot points (peaks or troughs) in price data"""
        pivots = [False] * len(data)

        for i in range(order, len(data) - order):
            if find_peaks:
                # Find peaks
                is_peak = all(data[i] >= data[i-j] for j in range(1, order+1)) and \
                         all(data[i] >= data[i+j] for j in range(1, order+1))
                pivots[i] = is_peak
            else:
                # Find troughs
                is_trough = all(data[i] <= data[i-j] for j in range(1, order+1)) and \
                           all(data[i] <= data[i+j] for j in range(1, order+1))
                pivots[i] = is_trough

        return pivots

    async def _find_wave_patterns(self,
                                 symbol: str,
                                 swing_points: List[WavePoint],
                                 timeframe: str) -> List[ElliottWavePattern]:
        """Find Elliott Wave patterns from swing points"""
        patterns = []

        # Look for 3-wave and 5-wave patterns
        for start_idx in range(len(swing_points) - 2):
            for end_idx in range(start_idx + 2, min(start_idx + 6, len(swing_points))):
                wave_sequence = swing_points[start_idx:end_idx + 1]

                # Check if this forms a valid Elliott Wave pattern
                pattern = await self._validate_wave_pattern(symbol, wave_sequence, timeframe)
                if pattern:
                    patterns.append(pattern)

        # Remove overlapping patterns, keep highest confidence
        patterns = self._remove_overlapping_patterns(patterns)

        return patterns

    async def _validate_wave_pattern(self,
                                   symbol: str,
                                   wave_sequence: List[WavePoint],
                                   timeframe: str) -> Optional[ElliottWavePattern]:
        """Validate if wave sequence forms a valid Elliott Wave pattern"""
        if len(wave_sequence) < 3:
            return None

        # Check time constraint (max 5 days)
        duration = wave_sequence[-1].timestamp - wave_sequence[0].timestamp
        if duration > self.max_pattern_duration:
            return None

        # Analyze wave structure
        pattern_type, direction, confidence = self._analyze_wave_structure(wave_sequence)

        if confidence < self.min_confidence_threshold:
            return None

        # Calculate price range
        prices = [point.price for point in wave_sequence]
        price_range = (min(prices), max(prices))

        # Estimate completion and next target
        completion_pct = self._calculate_completion_percentage(wave_sequence, pattern_type)
        next_target = self._calculate_next_target(wave_sequence, pattern_type)
        invalidation_level = self._calculate_invalidation_level(wave_sequence, pattern_type)

        return ElliottWavePattern(
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,
            direction=direction,
            waves=wave_sequence,
            start_time=wave_sequence[0].timestamp,
            end_time=wave_sequence[-1].timestamp,
            price_range=price_range,
            confidence=confidence,
            completion_percentage=completion_pct,
            next_target=next_target,
            invalidation_level=invalidation_level,
            expected_duration=duration
        )

    def _analyze_wave_structure(self, wave_sequence: List[WavePoint]) -> Tuple[WaveType, WaveDirection, float]:
        """Analyze the structure of wave sequence"""
        if len(wave_sequence) == 3:
            return self._analyze_three_wave_pattern(wave_sequence)
        elif len(wave_sequence) == 5:
            return self._analyze_five_wave_pattern(wave_sequence)
        else:
            return WaveType.CORRECTIVE, WaveDirection.SIDEWAYS, 0.5

    def _analyze_three_wave_pattern(self, waves: List[WavePoint]) -> Tuple[WaveType, WaveDirection, float]:
        """Analyze 3-wave corrective pattern (A-B-C)"""
        # Determine overall direction
        if waves[2].price > waves[0].price:
            direction = WaveDirection.UP
        elif waves[2].price < waves[0].price:
            direction = WaveDirection.DOWN
        else:
            direction = WaveDirection.SIDEWAYS

        # Check for zigzag pattern
        wave_a = abs(waves[1].price - waves[0].price)
        wave_c = abs(waves[2].price - waves[1].price)

        # Fibonacci relationships
        c_to_a_ratio = wave_c / wave_a if wave_a > 0 else 1.0

        confidence = 0.6  # Base confidence for 3-wave

        # Check for common Fibonacci ratios
        for fib_ratio in [0.618, 1.0, 1.618]:
            if abs(c_to_a_ratio - fib_ratio) < self.wave_ratio_tolerance:
                confidence += 0.2
                break

        return WaveType.ZIGZAG, direction, min(confidence, 1.0)

    def _analyze_five_wave_pattern(self, waves: List[WavePoint]) -> Tuple[WaveType, WaveDirection, float]:
        """Analyze 5-wave impulse pattern (1-2-3-4-5)"""
        # Determine overall direction
        if waves[4].price > waves[0].price:
            direction = WaveDirection.UP
        elif waves[4].price < waves[0].price:
            direction = WaveDirection.DOWN
        else:
            direction = WaveDirection.SIDEWAYS

        confidence = 0.7  # Base confidence for 5-wave

        # Check Elliott Wave rules
        # Rule 1: Wave 2 never retraces more than 100% of wave 1
        wave_1 = abs(waves[1].price - waves[0].price)
        wave_2 = abs(waves[2].price - waves[1].price)

        if wave_2 < wave_1:
            confidence += 0.1

        # Rule 2: Wave 3 is never the shortest
        wave_3 = abs(waves[3].price - waves[2].price)
        wave_5 = abs(waves[4].price - waves[3].price)

        if wave_3 >= max(wave_1, wave_5):
            confidence += 0.1

        # Rule 3: Wave 4 never enters the price territory of wave 1
        if direction == WaveDirection.UP:
            if waves[3].price > waves[1].price:
                confidence += 0.1
        else:
            if waves[3].price < waves[1].price:
                confidence += 0.1

        return WaveType.IMPULSE, direction, min(confidence, 1.0)

    def _calculate_completion_percentage(self, waves: List[WavePoint], pattern_type: WaveType) -> float:
        """Calculate pattern completion percentage"""
        if pattern_type == WaveType.IMPULSE and len(waves) == 5:
            return 100.0  # 5-wave pattern complete
        elif pattern_type == WaveType.ZIGZAG and len(waves) == 3:
            return 100.0  # 3-wave pattern complete
        else:
            return (len(waves) / 5.0) * 100.0  # Partial completion

    def _calculate_next_target(self, waves: List[WavePoint], pattern_type: WaveType) -> Optional[float]:
        """Calculate next price target based on Elliott Wave projections"""
        if len(waves) < 3:
            return None

        if pattern_type == WaveType.IMPULSE and len(waves) >= 3:
            # Project wave 5 target based on wave 1 and 3
            wave_1_size = abs(waves[1].price - waves[0].price)
            wave_3_end = waves[2].price

            # Common projection: Wave 5 = Wave 1 * 1.618 from wave 4 low
            if len(waves) >= 4:
                target = waves[3].price + (wave_1_size * 1.618)
                return target

        elif pattern_type == WaveType.ZIGZAG:
            # Project C wave target
            wave_a_size = abs(waves[1].price - waves[0].price)
            wave_b_end = waves[1].price

            # Common projection: C = A * 1.0 or 1.618
            target = wave_b_end + (wave_a_size * 1.0)
            return target

        return None

    def _calculate_invalidation_level(self, waves: List[WavePoint], pattern_type: WaveType) -> float:
        """Calculate pattern invalidation level"""
        if pattern_type == WaveType.IMPULSE:
            # Invalidation if price breaks below wave 1 low (for uptrend)
            return min(wave.price for wave in waves[:2])
        else:
            # For corrective patterns, invalidation at pattern start
            return waves[0].price

    def _remove_overlapping_patterns(self, patterns: List[ElliottWavePattern]) -> List[ElliottWavePattern]:
        """Remove overlapping patterns, keeping highest confidence"""
        if not patterns:
            return patterns

        # Sort by confidence descending
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        filtered_patterns = []
        for pattern in patterns:
            # Check if this pattern overlaps with any already selected
            overlaps = False
            for selected in filtered_patterns:
                if self._patterns_overlap(pattern, selected):
                    overlaps = True
                    break

            if not overlaps:
                filtered_patterns.append(pattern)

        return filtered_patterns

    def _patterns_overlap(self, pattern1: ElliottWavePattern, pattern2: ElliottWavePattern) -> bool:
        """Check if two patterns overlap in time"""
        return not (pattern1.end_time <= pattern2.start_time or pattern2.end_time <= pattern1.start_time)

    def _get_active_pattern(self, patterns: List[ElliottWavePattern]) -> Optional[ElliottWavePattern]:
        """Get the most recent active pattern"""
        if not patterns:
            return None

        # Find pattern with most recent end time
        active_pattern = max(patterns, key=lambda p: p.end_time)

        # Check if pattern is still active (within last 4 hours for H4 timeframe)
        time_since_end = datetime.now() - active_pattern.end_time
        if time_since_end <= timedelta(hours=8):  # Allow some buffer
            return active_pattern

        return None

    async def _generate_wave_signals(self,
                                   active_pattern: Optional[ElliottWavePattern],
                                   price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on Elliott Wave analysis"""
        if not active_pattern:
            return {
                'strength': 0.0,
                'recommendation': 'HOLD',
                'entry': None,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward': 0.0
            }

        current_price = price_data.iloc[-1]['close']
        signal_strength = active_pattern.confidence

        # Generate signals based on pattern type and completion
        if active_pattern.pattern_type == WaveType.IMPULSE:
            return self._generate_impulse_signals(active_pattern, current_price)
        elif active_pattern.pattern_type == WaveType.ZIGZAG:
            return self._generate_corrective_signals(active_pattern, current_price)
        else:
            return {
                'strength': signal_strength * 0.5,
                'recommendation': 'HOLD',
                'entry': None,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward': 0.0
            }

    def _generate_impulse_signals(self, pattern: ElliottWavePattern, current_price: float) -> Dict[str, Any]:
        """Generate signals for impulse patterns"""
        if pattern.completion_percentage >= 80:
            # Pattern nearly complete, look for reversal
            if pattern.direction == WaveDirection.UP:
                recommendation = 'SELL'
                entry = current_price * 0.995  # Slight discount
                stop_loss = pattern.next_target or (current_price * 1.02)
                take_profit = pattern.invalidation_level
            else:
                recommendation = 'BUY'
                entry = current_price * 1.005  # Slight premium
                stop_loss = pattern.next_target or (current_price * 0.98)
                take_profit = pattern.invalidation_level
        else:
            # Pattern in progress, trade in direction
            if pattern.direction == WaveDirection.UP:
                recommendation = 'BUY'
                entry = current_price * 1.002
                stop_loss = pattern.invalidation_level
                take_profit = pattern.next_target or (current_price * 1.05)
            else:
                recommendation = 'SELL'
                entry = current_price * 0.998
                stop_loss = pattern.invalidation_level
                take_profit = pattern.next_target or (current_price * 0.95)

        # Calculate risk-reward ratio
        if entry and stop_loss and take_profit:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            risk_reward = reward / risk if risk > 0 else 0.0
        else:
            risk_reward = 0.0

        return {
            'strength': pattern.confidence,
            'recommendation': recommendation,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward
        }

    def _generate_corrective_signals(self, pattern: ElliottWavePattern, current_price: float) -> Dict[str, Any]:
        """Generate signals for corrective patterns"""
        # Corrective patterns suggest continuation of main trend after completion
        if pattern.completion_percentage >= 90:
            # Look for trend resumption
            if pattern.direction == WaveDirection.DOWN:  # Correction in uptrend
                recommendation = 'BUY'
                entry = current_price * 1.002
                stop_loss = pattern.price_range[0] * 0.99  # Below correction low
                take_profit = pattern.price_range[1] * 1.1  # Above correction high
            else:  # Correction in downtrend
                recommendation = 'SELL'
                entry = current_price * 0.998
                stop_loss = pattern.price_range[1] * 1.01  # Above correction high
                take_profit = pattern.price_range[0] * 0.9  # Below correction low
        else:
            recommendation = 'HOLD'
            entry = None
            stop_loss = None
            take_profit = None

        # Calculate risk-reward ratio
        if entry and stop_loss and take_profit:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            risk_reward = reward / risk if risk > 0 else 0.0
        else:
            risk_reward = 0.0

        return {
            'strength': pattern.confidence * 0.8,  # Lower confidence for corrective
            'recommendation': recommendation,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.pattern_cache:
            return False

        cache_time = self.pattern_cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self.cache_duration

    def _create_empty_result(self, symbol: str) -> WaveAnalysisResult:
        """Create empty result for error cases"""
        return WaveAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            patterns=[],
            active_pattern=None,
            signal_strength=0.0,
            trade_recommendation='HOLD',
            entry_level=None,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=0.0
        )
