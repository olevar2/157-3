"""
Gann Time-Price Analysis
Cycle analysis and time-based predictions using Gann methodology.

This module provides comprehensive time-price analysis including:
- Time cycle detection and analysis
- Price-time relationship calculations
- Future time target predictions
- Cycle strength assessment

Expected Benefits:
- Time-based cycle predictions
- Enhanced timing for entries/exits
- Price-time correlation analysis
- Mathematical precision in timing
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
class CycleAnalysis:
    """Time cycle analysis data"""
    cycle_length: int  # in periods
    cycle_strength: float  # 0-1 confidence
    cycle_type: str  # 'price', 'time', 'combined'
    last_cycle_start: datetime
    next_cycle_target: datetime
    cycle_reliability: float
    historical_accuracy: float


@dataclass
class TimeTarget:
    """Time-based target prediction"""
    target_time: datetime
    target_price: Optional[float]
    confidence: float
    calculation_method: str
    supporting_cycles: List[int]
    target_type: str  # 'reversal', 'continuation', 'neutral'


@dataclass
class TimePriceResult:
    """Time-price analysis result"""
    symbol: str
    timestamp: datetime
    detected_cycles: List[CycleAnalysis]
    time_targets: List[TimeTarget]
    dominant_cycle: Optional[CycleAnalysis]
    next_time_window: Optional[Tuple[datetime, datetime]]
    price_time_correlation: float
    analysis_confidence: float


class GannTimePrice:
    """
    Gann Time-Price Analysis
    Implements W.D. Gann's time cycle methodology
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Gann time cycles (in trading days)
        self.gann_time_cycles = [
            7, 14, 21, 30, 45, 60, 90, 120, 144, 180, 360
        ]

        # Natural cycles
        self.natural_cycles = [
            28,   # Lunar month
            365,  # Solar year
            91,   # Season
            182   # Half year
        ]

        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0

        self.logger.info("Gann Time-Price Analysis initialized")

    async def analyze_time_price(
        self,
        symbol: str,
        price_data: List[Dict],
        analysis_period: int = 252  # 1 year of trading days
    ) -> TimePriceResult:
        """
        Analyze time-price relationships and cycles

        Args:
            symbol: Trading symbol
            price_data: List of OHLC data with timestamp
            analysis_period: Number of periods to analyze

        Returns:
            TimePriceResult with complete time-price analysis
        """
        start_time = time.perf_counter()

        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').tail(analysis_period)

            # Detect time cycles
            detected_cycles = await self._detect_time_cycles(df)

            # Calculate time targets
            time_targets = await self._calculate_time_targets(df, detected_cycles)

            # Find dominant cycle
            dominant_cycle = await self._find_dominant_cycle(detected_cycles)

            # Determine next time window
            next_time_window = await self._calculate_next_time_window(detected_cycles, df)

            # Calculate price-time correlation
            price_time_correlation = await self._calculate_price_time_correlation(df)

            # Calculate analysis confidence
            confidence = await self._calculate_analysis_confidence(detected_cycles, df)

            result = TimePriceResult(
                symbol=symbol,
                timestamp=datetime.now(),
                detected_cycles=detected_cycles,
                time_targets=time_targets,
                dominant_cycle=dominant_cycle,
                next_time_window=next_time_window,
                price_time_correlation=price_time_correlation,
                analysis_confidence=confidence
            )

            # Update performance metrics
            calculation_time = time.perf_counter() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time

            self.logger.debug(f"Time-price analysis completed for {symbol} in {calculation_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in time-price analysis for {symbol}: {e}")
            raise

    async def _detect_time_cycles(self, df: pd.DataFrame) -> List[CycleAnalysis]:
        """Detect time cycles in price data"""

        cycles = []

        # Find significant price turning points
        turning_points = await self._find_turning_points(df)

        if len(turning_points) < 3:
            return cycles

        # Analyze each potential cycle length
        all_cycles = self.gann_time_cycles + self.natural_cycles

        for cycle_length in all_cycles:
            if cycle_length >= len(df):
                continue

            cycle_analysis = await self._analyze_cycle_length(df, turning_points, cycle_length)

            if cycle_analysis and cycle_analysis.cycle_strength > 0.3:
                cycles.append(cycle_analysis)

        # Sort by strength
        cycles.sort(key=lambda x: x.cycle_strength, reverse=True)

        return cycles[:10]  # Keep top 10 cycles

    async def _find_turning_points(self, df: pd.DataFrame) -> List[Tuple[datetime, float, str]]:
        """Find significant turning points in price data"""

        turning_points = []
        window = 5  # Look for peaks/troughs in 5-period windows

        for i in range(window, len(df) - window):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_time = df.iloc[i]['timestamp']

            # Check for peak
            is_peak = True
            for j in range(i - window, i + window + 1):
                if j != i and df.iloc[j]['high'] >= current_high:
                    is_peak = False
                    break

            if is_peak:
                turning_points.append((current_time, current_high, 'peak'))
                continue

            # Check for trough
            is_trough = True
            for j in range(i - window, i + window + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_trough = False
                    break

            if is_trough:
                turning_points.append((current_time, current_low, 'trough'))

        return turning_points

    async def _analyze_cycle_length(
        self,
        df: pd.DataFrame,
        turning_points: List[Tuple[datetime, float, str]],
        cycle_length: int
    ) -> Optional[CycleAnalysis]:
        """Analyze a specific cycle length for validity"""

        if len(turning_points) < 3:
            return None

        # Calculate time differences between turning points
        time_diffs = []
        for i in range(1, len(turning_points)):
            time_diff = (turning_points[i][0] - turning_points[i-1][0]).days
            time_diffs.append(time_diff)

        if not time_diffs:
            return None

        # Check how many time differences are close to the cycle length
        matches = 0
        tolerance = cycle_length * 0.15  # 15% tolerance

        for diff in time_diffs:
            if abs(diff - cycle_length) <= tolerance:
                matches += 1

        if matches == 0:
            return None

        # Calculate cycle strength
        cycle_strength = matches / len(time_diffs)

        if cycle_strength < 0.3:
            return None

        # Find last cycle start
        last_cycle_start = turning_points[-1][0] if turning_points else df.iloc[-1]['timestamp']

        # Calculate next cycle target
        next_cycle_target = last_cycle_start + timedelta(days=cycle_length)

        # Calculate reliability based on consistency
        reliability = min(1.0, cycle_strength * 1.2)

        return CycleAnalysis(
            cycle_length=cycle_length,
            cycle_strength=cycle_strength,
            cycle_type='time',
            last_cycle_start=last_cycle_start,
            next_cycle_target=next_cycle_target,
            cycle_reliability=reliability,
            historical_accuracy=cycle_strength  # Simplified
        )

    async def _calculate_time_targets(
        self,
        df: pd.DataFrame,
        detected_cycles: List[CycleAnalysis]
    ) -> List[TimeTarget]:
        """Calculate time-based targets from detected cycles"""

        targets = []
        current_time = df.iloc[-1]['timestamp']

        for cycle in detected_cycles[:5]:  # Use top 5 cycles
            # Calculate target time
            target_time = cycle.next_cycle_target

            # Estimate target type based on cycle pattern
            target_type = 'reversal'  # Simplified - would need more analysis

            # Calculate confidence
            confidence = cycle.cycle_strength * cycle.cycle_reliability

            target = TimeTarget(
                target_time=target_time,
                target_price=None,  # Would need price projection
                confidence=confidence,
                calculation_method=f"Gann Time Cycle - {cycle.cycle_length} days",
                supporting_cycles=[cycle.cycle_length],
                target_type=target_type
            )

            targets.append(target)

        # Sort by confidence
        targets.sort(key=lambda x: x.confidence, reverse=True)

        return targets

    async def _find_dominant_cycle(
        self,
        detected_cycles: List[CycleAnalysis]
    ) -> Optional[CycleAnalysis]:
        """Find the most dominant cycle"""

        if not detected_cycles:
            return None

        # Return cycle with highest combined score
        best_cycle = None
        best_score = 0

        for cycle in detected_cycles:
            # Score based on strength and reliability
            score = (cycle.cycle_strength * 0.6) + (cycle.cycle_reliability * 0.4)

            if score > best_score:
                best_score = score
                best_cycle = cycle

        return best_cycle

    async def _calculate_next_time_window(
        self,
        detected_cycles: List[CycleAnalysis],
        df: pd.DataFrame
    ) -> Optional[Tuple[datetime, datetime]]:
        """Calculate the next significant time window"""

        if not detected_cycles:
            return None

        # Find the nearest cycle targets
        current_time = df.iloc[-1]['timestamp']
        nearest_targets = []

        for cycle in detected_cycles:
            time_diff = (cycle.next_cycle_target - current_time).days
            if 0 < time_diff <= 90:  # Within next 90 days
                nearest_targets.append((cycle.next_cycle_target, cycle.cycle_strength))

        if not nearest_targets:
            return None

        # Sort by time
        nearest_targets.sort(key=lambda x: x[0])

        # Create window around strongest target
        strongest_target = max(nearest_targets, key=lambda x: x[1])
        target_time = strongest_target[0]

        # Create 3-day window around target
        window_start = target_time - timedelta(days=1)
        window_end = target_time + timedelta(days=2)

        return (window_start, window_end)

    async def _calculate_price_time_correlation(self, df: pd.DataFrame) -> float:
        """Calculate correlation between price movements and time"""

        if len(df) < 20:
            return 0.0

        # Simple correlation calculation
        df['price_change'] = df['close'].pct_change()
        df['time_index'] = range(len(df))

        # Calculate correlation
        correlation = df['price_change'].corr(df['time_index'])

        return abs(correlation) if not pd.isna(correlation) else 0.0

    async def _calculate_analysis_confidence(
        self,
        detected_cycles: List[CycleAnalysis],
        df: pd.DataFrame
    ) -> float:
        """Calculate overall confidence in time-price analysis"""

        if not detected_cycles:
            return 0.0

        # Base confidence on cycle strengths
        avg_cycle_strength = sum(cycle.cycle_strength for cycle in detected_cycles) / len(detected_cycles)

        # Data quality factor
        data_quality = min(1.0, len(df) / 252)  # Normalize to 1 year

        # Number of cycles factor
        cycle_count_factor = min(1.0, len(detected_cycles) / 5)

        confidence = (avg_cycle_strength * 0.6) + (data_quality * 0.3) + (cycle_count_factor * 0.1)

        return min(1.0, confidence)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the analyzer"""

        if self.calculation_count == 0:
            return {
                'calculations_performed': 0,
                'average_calculation_time': 0.0,
                'total_calculation_time': 0.0
            }

        return {
            'calculations_performed': self.calculation_count,
            'average_calculation_time': self.total_calculation_time / self.calculation_count,
            'total_calculation_time': self.total_calculation_time
        }
