"""
Gann Pattern Detector
Pattern recognition using Gann methods and geometric analysis.

This module provides comprehensive pattern detection including:
- Gann geometric pattern recognition
- Price-time pattern analysis
- Signal generation from patterns
- Pattern strength assessment

Expected Benefits:
- Advanced pattern recognition
- Enhanced signal generation
- Geometric pattern validation
- Mathematical precision in patterns
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import math


@dataclass
class GannPattern:
    """Gann pattern data structure"""
    pattern_name: str
    pattern_type: str  # 'geometric', 'time_price', 'angle_based'
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    pattern_strength: float  # 0-1 confidence
    completion_percentage: float  # 0-100%
    key_levels: List[float]
    geometric_angles: List[float]


@dataclass
class PatternSignal:
    """Trading signal from pattern"""
    signal_type: str  # 'buy', 'sell', 'hold'
    signal_strength: float  # 0-1
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    time_target: Optional[datetime]
    risk_reward_ratio: float
    pattern_source: str


@dataclass
class PatternResult:
    """Pattern detection result"""
    symbol: str
    timestamp: datetime
    detected_patterns: List[GannPattern]
    active_signals: List[PatternSignal]
    dominant_pattern: Optional[GannPattern]
    pattern_confluence: int  # Number of overlapping patterns
    overall_bias: str  # 'bullish', 'bearish', 'neutral'
    analysis_confidence: float


class GannPatternDetector:
    """
    Gann Pattern Detector for geometric pattern recognition
    Implements W.D. Gann's pattern recognition methods
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Gann pattern definitions
        self.gann_patterns = {
            'gann_square': {
                'angles': [45, 90, 135, 180],
                'min_points': 4,
                'strength_threshold': 0.6
            },
            'gann_triangle': {
                'angles': [30, 60, 90],
                'min_points': 3,
                'strength_threshold': 0.5
            },
            'gann_fan_pattern': {
                'angles': [22.5, 45, 67.5],
                'min_points': 3,
                'strength_threshold': 0.7
            }
        }

        # Performance tracking
        self.detection_count = 0
        self.total_detection_time = 0.0

        self.logger.info("Gann Pattern Detector initialized")

    async def detect_patterns(
        self,
        symbol: str,
        price_data: List[Dict],
        lookback_periods: int = 100
    ) -> PatternResult:
        """
        Detect Gann patterns in price data

        Args:
            symbol: Trading symbol
            price_data: List of OHLC data with timestamp
            lookback_periods: Number of periods to analyze

        Returns:
            PatternResult with detected patterns and signals
        """
        start_time = time.perf_counter()

        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').tail(lookback_periods)

            # Detect various Gann patterns
            detected_patterns = await self._detect_all_patterns(df)

            # Generate trading signals from patterns
            active_signals = await self._generate_pattern_signals(detected_patterns, df)

            # Find dominant pattern
            dominant_pattern = await self._find_dominant_pattern(detected_patterns)

            # Calculate pattern confluence
            pattern_confluence = await self._calculate_pattern_confluence(detected_patterns, df)

            # Determine overall bias
            overall_bias = await self._determine_overall_bias(detected_patterns, active_signals)

            # Calculate analysis confidence
            confidence = await self._calculate_analysis_confidence(detected_patterns, df)

            result = PatternResult(
                symbol=symbol,
                timestamp=datetime.now(),
                detected_patterns=detected_patterns,
                active_signals=active_signals,
                dominant_pattern=dominant_pattern,
                pattern_confluence=pattern_confluence,
                overall_bias=overall_bias,
                analysis_confidence=confidence
            )

            # Update performance metrics
            detection_time = time.perf_counter() - start_time
            self.detection_count += 1
            self.total_detection_time += detection_time

            self.logger.debug(f"Pattern detection completed for {symbol} in {detection_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error detecting patterns for {symbol}: {e}")
            raise

    async def _detect_all_patterns(self, df: pd.DataFrame) -> List[GannPattern]:
        """Detect all Gann patterns in the data"""

        patterns = []

        # Find significant pivot points
        pivot_points = await self._find_pivot_points(df)

        if len(pivot_points) < 3:
            return patterns

        # Detect each pattern type
        for pattern_name, pattern_config in self.gann_patterns.items():
            pattern_instances = await self._detect_pattern_type(
                df, pivot_points, pattern_name, pattern_config
            )
            patterns.extend(pattern_instances)

        # Sort by strength
        patterns.sort(key=lambda x: x.pattern_strength, reverse=True)

        return patterns[:10]  # Keep top 10 patterns

    async def _find_pivot_points(self, df: pd.DataFrame) -> List[Tuple[datetime, float, str]]:
        """Find significant pivot points for pattern analysis"""

        pivot_points = []
        window = 3  # Smaller window for more sensitive detection

        for i in range(window, len(df) - window):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_time = df.iloc[i]['timestamp']

            # Check for swing high
            is_high = all(
                current_high >= df.iloc[j]['high']
                for j in range(i - window, i + window + 1)
                if j != i
            )

            if is_high:
                pivot_points.append((current_time, current_high, 'high'))
                continue

            # Check for swing low
            is_low = all(
                current_low <= df.iloc[j]['low']
                for j in range(i - window, i + window + 1)
                if j != i
            )

            if is_low:
                pivot_points.append((current_time, current_low, 'low'))

        return pivot_points

    async def _detect_pattern_type(
        self,
        df: pd.DataFrame,
        pivot_points: List[Tuple[datetime, float, str]],
        pattern_name: str,
        pattern_config: Dict
    ) -> List[GannPattern]:
        """Detect specific pattern type"""

        patterns = []
        min_points = pattern_config['min_points']
        required_angles = pattern_config['angles']

        # Analyze combinations of pivot points
        for i in range(len(pivot_points) - min_points + 1):
            point_group = pivot_points[i:i + min_points]

            # Calculate angles between points
            angles = await self._calculate_pattern_angles(point_group)

            # Check if angles match pattern requirements
            pattern_match = await self._check_angle_match(angles, required_angles)

            if pattern_match > pattern_config['strength_threshold']:
                # Create pattern
                start_time = point_group[0][0]
                end_time = point_group[-1][0]
                start_price = point_group[0][1]
                end_price = point_group[-1][1]

                key_levels = [point[1] for point in point_group]

                pattern = GannPattern(
                    pattern_name=pattern_name,
                    pattern_type='geometric',
                    start_time=start_time,
                    end_time=end_time,
                    start_price=start_price,
                    end_price=end_price,
                    pattern_strength=pattern_match,
                    completion_percentage=100.0,  # Completed pattern
                    key_levels=key_levels,
                    geometric_angles=angles
                )

                patterns.append(pattern)

        return patterns

    async def _calculate_pattern_angles(
        self,
        points: List[Tuple[datetime, float, str]]
    ) -> List[float]:
        """Calculate angles between pattern points"""

        angles = []

        for i in range(len(points) - 1):
            point1 = points[i]
            point2 = points[i + 1]

            # Calculate time and price differences
            time_diff = (point2[0] - point1[0]).total_seconds() / 3600  # hours
            price_diff = point2[1] - point1[1]

            if time_diff > 0:
                # Calculate angle in degrees
                angle_rad = math.atan(price_diff / time_diff)
                angle_deg = math.degrees(angle_rad)
                angles.append(abs(angle_deg))

        return angles

    async def _check_angle_match(
        self,
        calculated_angles: List[float],
        required_angles: List[float],
        tolerance: float = 5.0
    ) -> float:
        """Check how well calculated angles match required pattern angles"""

        if not calculated_angles or not required_angles:
            return 0.0

        matches = 0
        total_comparisons = 0

        for calc_angle in calculated_angles:
            for req_angle in required_angles:
                total_comparisons += 1
                if abs(calc_angle - req_angle) <= tolerance:
                    matches += 1
                    break  # Found a match for this calculated angle

        return matches / len(calculated_angles) if calculated_angles else 0.0

    async def _generate_pattern_signals(
        self,
        patterns: List[GannPattern],
        df: pd.DataFrame
    ) -> List[PatternSignal]:
        """Generate trading signals from detected patterns"""

        signals = []
        current_price = df.iloc[-1]['close']

        for pattern in patterns:
            if pattern.pattern_strength < 0.5:
                continue

            # Determine signal direction based on pattern
            signal_type = await self._determine_signal_direction(pattern, current_price)

            if signal_type == 'hold':
                continue

            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            stop_loss, take_profit = await self._calculate_signal_levels(pattern, current_price, signal_type)

            # Calculate risk-reward ratio
            risk_reward = await self._calculate_risk_reward(entry_price, stop_loss, take_profit)

            signal = PatternSignal(
                signal_type=signal_type,
                signal_strength=pattern.pattern_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_target=None,  # Would need time analysis
                risk_reward_ratio=risk_reward,
                pattern_source=pattern.pattern_name
            )

            signals.append(signal)

        # Sort by signal strength
        signals.sort(key=lambda x: x.signal_strength, reverse=True)

        return signals[:5]  # Keep top 5 signals

    async def _determine_signal_direction(
        self,
        pattern: GannPattern,
        current_price: float
    ) -> str:
        """Determine signal direction from pattern"""

        # Simple logic based on pattern completion and price position
        if current_price > pattern.end_price:
            return 'buy' if pattern.end_price > pattern.start_price else 'hold'
        elif current_price < pattern.end_price:
            return 'sell' if pattern.end_price < pattern.start_price else 'hold'
        else:
            return 'hold'

    async def _calculate_signal_levels(
        self,
        pattern: GannPattern,
        current_price: float,
        signal_type: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""

        if not pattern.key_levels:
            return None, None

        price_range = max(pattern.key_levels) - min(pattern.key_levels)

        if signal_type == 'buy':
            stop_loss = current_price - (price_range * 0.3)
            take_profit = current_price + (price_range * 0.6)
        elif signal_type == 'sell':
            stop_loss = current_price + (price_range * 0.3)
            take_profit = current_price - (price_range * 0.6)
        else:
            return None, None

        return stop_loss, take_profit

    async def _calculate_risk_reward(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ) -> float:
        """Calculate risk-reward ratio"""

        if not stop_loss or not take_profit:
            return 0.0

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        return reward / risk if risk > 0 else 0.0

    async def _find_dominant_pattern(
        self,
        patterns: List[GannPattern]
    ) -> Optional[GannPattern]:
        """Find the most dominant pattern"""

        if not patterns:
            return None

        # Return pattern with highest strength
        return max(patterns, key=lambda x: x.pattern_strength)

    async def _calculate_pattern_confluence(
        self,
        patterns: List[GannPattern],
        df: pd.DataFrame
    ) -> int:
        """Calculate number of overlapping patterns"""

        if len(patterns) < 2:
            return len(patterns)

        current_time = df.iloc[-1]['timestamp']
        active_patterns = 0

        for pattern in patterns:
            # Check if pattern is currently active (recent)
            time_diff = (current_time - pattern.end_time).days
            if time_diff <= 30:  # Active within last 30 days
                active_patterns += 1

        return active_patterns

    async def _determine_overall_bias(
        self,
        patterns: List[GannPattern],
        signals: List[PatternSignal]
    ) -> str:
        """Determine overall market bias from patterns and signals"""

        if not signals:
            return 'neutral'

        buy_strength = sum(s.signal_strength for s in signals if s.signal_type == 'buy')
        sell_strength = sum(s.signal_strength for s in signals if s.signal_type == 'sell')

        if buy_strength > sell_strength * 1.2:
            return 'bullish'
        elif sell_strength > buy_strength * 1.2:
            return 'bearish'
        else:
            return 'neutral'

    async def _calculate_analysis_confidence(
        self,
        patterns: List[GannPattern],
        df: pd.DataFrame
    ) -> float:
        """Calculate overall confidence in pattern analysis"""

        if not patterns:
            return 0.0

        # Base confidence on pattern strengths
        avg_pattern_strength = sum(p.pattern_strength for p in patterns) / len(patterns)

        # Data quality factor
        data_quality = min(1.0, len(df) / 100)

        # Pattern count factor
        pattern_count_factor = min(1.0, len(patterns) / 5)

        confidence = (avg_pattern_strength * 0.6) + (data_quality * 0.3) + (pattern_count_factor * 0.1)

        return min(1.0, confidence)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the detector"""

        if self.detection_count == 0:
            return {
                'detections_performed': 0,
                'average_detection_time': 0.0,
                'total_detection_time': 0.0
            }

        return {
            'detections_performed': self.detection_count,
            'average_detection_time': self.total_detection_time / self.detection_count,
            'total_detection_time': self.total_detection_time
        }
