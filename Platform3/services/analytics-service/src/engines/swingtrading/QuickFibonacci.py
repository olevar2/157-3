"""
Quick Fibonacci Retracement Engine
Fast retracements for H4 reversals optimized for short-term swing trading.
Specialized for 1-5 day maximum holding periods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio

class FibLevel(Enum):
    """Standard Fibonacci retracement levels"""
    LEVEL_0 = 0.0
    LEVEL_236 = 0.236
    LEVEL_382 = 0.382
    LEVEL_500 = 0.5
    LEVEL_618 = 0.618
    LEVEL_786 = 0.786
    LEVEL_1000 = 1.0

class TrendDirection(Enum):
    """Trend direction for Fibonacci analysis"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"

@dataclass
class FibonacciLevel:
    """Individual Fibonacci level"""
    level: FibLevel
    price: float
    percentage: float
    support_strength: float
    resistance_strength: float
    volume_confirmation: float
    touch_count: int
    last_touch_time: Optional[datetime]

@dataclass
class FibonacciRetracement:
    """Complete Fibonacci retracement analysis"""
    symbol: str
    timeframe: str
    trend_direction: TrendDirection
    swing_high: float
    swing_low: float
    swing_high_time: datetime
    swing_low_time: datetime
    current_price: float
    current_retracement: float
    levels: List[FibonacciLevel]
    key_levels: List[FibonacciLevel]
    next_target: Optional[float]
    invalidation_level: float
    confidence: float

@dataclass
class FibonacciSignal:
    """Fibonacci-based trading signal"""
    symbol: str
    timestamp: datetime
    signal_type: str  # BUY, SELL, HOLD
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    confidence: float
    fibonacci_level: FibLevel
    reasoning: str

class QuickFibonacci:
    """
    Quick Fibonacci Retracement Engine
    Optimized for H4 timeframe with fast reversal detection
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Fibonacci levels for analysis
        self.fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.key_levels = [0.382, 0.5, 0.618]  # Most important levels

        # Parameters for short-term trading
        self.min_swing_size = 0.002  # Minimum 20 pips for major pairs
        self.max_swing_age = timedelta(days=5)  # Max 5 days for swing
        self.level_tolerance = 0.0005  # 5 pips tolerance for level touches

        # Volume confirmation thresholds
        self.volume_multiplier_threshold = 1.5

        # Cache for performance
        self.retracement_cache = {}
        self.cache_duration = timedelta(minutes=15)

    async def analyze_fibonacci_retracements(self,
                                           symbol: str,
                                           price_data: pd.DataFrame,
                                           timeframe: str = "H4") -> FibonacciRetracement:
        """
        Analyze Fibonacci retracements for quick reversal opportunities

        Args:
            symbol: Trading symbol
            price_data: OHLCV data
            timeframe: Analysis timeframe

        Returns:
            FibonacciRetracement with levels and analysis
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{len(price_data)}"
            if self._is_cache_valid(cache_key):
                return self.retracement_cache[cache_key]['result']

            # Find recent swing high and low
            swing_high, swing_low, swing_times = await self._find_recent_swing(price_data)

            if not swing_high or not swing_low:
                return self._create_empty_retracement(symbol, timeframe)

            # Determine trend direction
            trend_direction = self._determine_trend_direction(swing_high, swing_low, swing_times)

            # Calculate Fibonacci levels
            fib_levels = self._calculate_fibonacci_levels(swing_high, swing_low, trend_direction)

            # Analyze level strength and touches
            analyzed_levels = await self._analyze_level_strength(fib_levels, price_data)

            # Identify key levels
            key_levels = self._identify_key_levels(analyzed_levels)

            # Calculate current retracement
            current_price = price_data.iloc[-1]['close']
            current_retracement = self._calculate_current_retracement(
                current_price, swing_high, swing_low, trend_direction
            )

            # Determine next target and invalidation
            next_target = self._calculate_next_target(analyzed_levels, current_retracement, trend_direction)
            invalidation_level = self._calculate_invalidation_level(swing_high, swing_low, trend_direction)

            # Calculate overall confidence
            confidence = self._calculate_retracement_confidence(analyzed_levels, current_retracement)

            result = FibonacciRetracement(
                symbol=symbol,
                timeframe=timeframe,
                trend_direction=trend_direction,
                swing_high=swing_high,
                swing_low=swing_low,
                swing_high_time=swing_times['high'],
                swing_low_time=swing_times['low'],
                current_price=current_price,
                current_retracement=current_retracement,
                levels=analyzed_levels,
                key_levels=key_levels,
                next_target=next_target,
                invalidation_level=invalidation_level,
                confidence=confidence
            )

            # Cache result
            self.retracement_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }

            return result

        except Exception as e:
            self.logger.error(f"Fibonacci retracement analysis error for {symbol}: {e}")
            return self._create_empty_retracement(symbol, timeframe)

    async def generate_fibonacci_signals(self, retracement: FibonacciRetracement) -> List[FibonacciSignal]:
        """Generate trading signals based on Fibonacci analysis"""
        signals = []

        if retracement.confidence < 0.6:
            return signals

        current_price = retracement.current_price

        # Check for signals at key Fibonacci levels
        for level in retracement.key_levels:
            signal = await self._check_level_signal(retracement, level, current_price)
            if signal:
                signals.append(signal)

        return signals

    async def _find_recent_swing(self, price_data: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Dict]:
        """Find the most recent significant swing high and low"""
        # Look back maximum 5 days (120 H4 candles)
        lookback_periods = min(120, len(price_data))
        recent_data = price_data.tail(lookback_periods)

        # Find swing highs and lows using pivot points
        swing_high = None
        swing_low = None
        swing_high_time = None
        swing_low_time = None

        # Simple approach: find highest high and lowest low in recent period
        high_idx = recent_data['high'].idxmax()
        low_idx = recent_data['low'].idxmin()

        swing_high = recent_data.loc[high_idx, 'high']
        swing_low = recent_data.loc[low_idx, 'low']
        swing_high_time = high_idx
        swing_low_time = low_idx

        # Validate swing size
        swing_size = abs(swing_high - swing_low) / swing_low
        if swing_size < self.min_swing_size:
            return None, None, {}

        return swing_high, swing_low, {'high': swing_high_time, 'low': swing_low_time}

    def _determine_trend_direction(self, swing_high: float, swing_low: float, swing_times: Dict) -> TrendDirection:
        """Determine the primary trend direction"""
        # If swing high came after swing low, it's an uptrend retracement
        if swing_times['high'] > swing_times['low']:
            return TrendDirection.UPTREND
        else:
            return TrendDirection.DOWNTREND

    def _calculate_fibonacci_levels(self,
                                  swing_high: float,
                                  swing_low: float,
                                  trend_direction: TrendDirection) -> List[FibonacciLevel]:
        """Calculate Fibonacci retracement levels"""
        levels = []
        swing_range = swing_high - swing_low

        for fib_ratio in self.fib_levels:
            if trend_direction == TrendDirection.UPTREND:
                # For uptrend, retracement levels go down from swing high
                price = swing_high - (swing_range * fib_ratio)
            else:
                # For downtrend, retracement levels go up from swing low
                price = swing_low + (swing_range * fib_ratio)

            level = FibonacciLevel(
                level=FibLevel(fib_ratio),
                price=price,
                percentage=fib_ratio * 100,
                support_strength=0.0,
                resistance_strength=0.0,
                volume_confirmation=0.0,
                touch_count=0,
                last_touch_time=None
            )
            levels.append(level)

        return levels

    async def _analyze_level_strength(self,
                                    fib_levels: List[FibonacciLevel],
                                    price_data: pd.DataFrame) -> List[FibonacciLevel]:
        """Analyze the strength of each Fibonacci level"""
        analyzed_levels = []

        for level in fib_levels:
            # Count touches and analyze volume
            touches, last_touch, avg_volume = self._count_level_touches(level.price, price_data)

            # Calculate support/resistance strength
            support_strength = self._calculate_support_strength(level.price, price_data)
            resistance_strength = self._calculate_resistance_strength(level.price, price_data)

            # Volume confirmation
            volume_confirmation = self._calculate_volume_confirmation(level.price, price_data)

            # Update level with analysis
            level.touch_count = touches
            level.last_touch_time = last_touch
            level.support_strength = support_strength
            level.resistance_strength = resistance_strength
            level.volume_confirmation = volume_confirmation

            analyzed_levels.append(level)

        return analyzed_levels

    def _count_level_touches(self, level_price: float, price_data: pd.DataFrame) -> Tuple[int, Optional[datetime], float]:
        """Count how many times price touched a Fibonacci level"""
        touches = 0
        last_touch = None
        volume_sum = 0.0

        for idx, row in price_data.iterrows():
            # Check if price touched the level (within tolerance)
            if (row['low'] <= level_price + self.level_tolerance and
                row['high'] >= level_price - self.level_tolerance):
                touches += 1
                last_touch = idx
                volume_sum += row['volume']

        avg_volume = volume_sum / touches if touches > 0 else 0.0
        return touches, last_touch, avg_volume

    def _calculate_support_strength(self, level_price: float, price_data: pd.DataFrame) -> float:
        """Calculate support strength at a Fibonacci level"""
        support_strength = 0.0

        for idx, row in price_data.iterrows():
            # Check for bounces from support
            if (row['low'] <= level_price + self.level_tolerance and
                row['close'] > level_price):
                support_strength += 1.0

        # Normalize by data length
        return support_strength / len(price_data)

    def _calculate_resistance_strength(self, level_price: float, price_data: pd.DataFrame) -> float:
        """Calculate resistance strength at a Fibonacci level"""
        resistance_strength = 0.0

        for idx, row in price_data.iterrows():
            # Check for rejections from resistance
            if (row['high'] >= level_price - self.level_tolerance and
                row['close'] < level_price):
                resistance_strength += 1.0

        # Normalize by data length
        return resistance_strength / len(price_data)

    def _calculate_volume_confirmation(self, level_price: float, price_data: pd.DataFrame) -> float:
        """Calculate volume confirmation at Fibonacci level"""
        level_volumes = []
        avg_volume = price_data['volume'].mean()

        for idx, row in price_data.iterrows():
            if (row['low'] <= level_price + self.level_tolerance and
                row['high'] >= level_price - self.level_tolerance):
                level_volumes.append(row['volume'])

        if not level_volumes:
            return 0.0

        avg_level_volume = np.mean(level_volumes)
        return avg_level_volume / avg_volume if avg_volume > 0 else 0.0

    def _identify_key_levels(self, analyzed_levels: List[FibonacciLevel]) -> List[FibonacciLevel]:
        """Identify the most important Fibonacci levels"""
        key_levels = []

        for level in analyzed_levels:
            # Key levels are 38.2%, 50%, and 61.8%
            if level.level in [FibLevel.LEVEL_382, FibLevel.LEVEL_500, FibLevel.LEVEL_618]:
                key_levels.append(level)

        # Sort by combined strength (support + resistance + volume)
        key_levels.sort(
            key=lambda l: l.support_strength + l.resistance_strength + l.volume_confirmation,
            reverse=True
        )

        return key_levels

    def _calculate_current_retracement(self,
                                     current_price: float,
                                     swing_high: float,
                                     swing_low: float,
                                     trend_direction: TrendDirection) -> float:
        """Calculate current retracement percentage"""
        swing_range = swing_high - swing_low

        if trend_direction == TrendDirection.UPTREND:
            # Retracement from swing high
            retracement = (swing_high - current_price) / swing_range
        else:
            # Retracement from swing low
            retracement = (current_price - swing_low) / swing_range

        return max(0.0, min(1.0, retracement))  # Clamp between 0 and 1

    def _calculate_next_target(self,
                             levels: List[FibonacciLevel],
                             current_retracement: float,
                             trend_direction: TrendDirection) -> Optional[float]:
        """Calculate next Fibonacci target level"""
        # Find the next significant level based on current retracement
        target_levels = []

        for level in levels:
            level_ratio = level.level.value

            if trend_direction == TrendDirection.UPTREND:
                # In uptrend retracement, look for next support level
                if level_ratio > current_retracement and level.support_strength > 0.1:
                    target_levels.append((level_ratio, level.price))
            else:
                # In downtrend retracement, look for next resistance level
                if level_ratio > current_retracement and level.resistance_strength > 0.1:
                    target_levels.append((level_ratio, level.price))

        if target_levels:
            # Return the closest level
            target_levels.sort(key=lambda x: x[0])
            return target_levels[0][1]

        return None

    def _calculate_invalidation_level(self,
                                    swing_high: float,
                                    swing_low: float,
                                    trend_direction: TrendDirection) -> float:
        """Calculate pattern invalidation level"""
        if trend_direction == TrendDirection.UPTREND:
            # Invalidation below swing low
            return swing_low * 0.995
        else:
            # Invalidation above swing high
            return swing_high * 1.005

    def _calculate_retracement_confidence(self,
                                        levels: List[FibonacciLevel],
                                        current_retracement: float) -> float:
        """Calculate overall confidence in Fibonacci analysis"""
        confidence = 0.5  # Base confidence

        # Boost confidence for key level proximity
        for level in levels:
            if level.level in [FibLevel.LEVEL_382, FibLevel.LEVEL_500, FibLevel.LEVEL_618]:
                level_distance = abs(current_retracement - level.level.value)
                if level_distance < 0.05:  # Within 5% of key level
                    confidence += 0.2 * (level.support_strength + level.resistance_strength)

        # Boost confidence for high touch counts
        max_touches = max([level.touch_count for level in levels], default=0)
        if max_touches >= 3:
            confidence += 0.1

        # Boost confidence for volume confirmation
        avg_volume_conf = np.mean([level.volume_confirmation for level in levels])
        if avg_volume_conf > 1.2:
            confidence += 0.1

        return min(confidence, 1.0)

    async def _check_level_signal(self,
                                retracement: FibonacciRetracement,
                                level: FibonacciLevel,
                                current_price: float) -> Optional[FibonacciSignal]:
        """Check for trading signals at a Fibonacci level"""
        # Check if price is near the level
        price_distance = abs(current_price - level.price) / current_price
        if price_distance > 0.002:  # More than 20 pips away
            return None

        # Determine signal type based on trend and level strength
        if retracement.trend_direction == TrendDirection.UPTREND:
            # In uptrend retracement, look for buy signals at support
            if level.support_strength > 0.2 and current_price <= level.price:
                return self._create_buy_signal(retracement, level, current_price)
        else:
            # In downtrend retracement, look for sell signals at resistance
            if level.resistance_strength > 0.2 and current_price >= level.price:
                return self._create_sell_signal(retracement, level, current_price)

        return None

    def _create_buy_signal(self,
                          retracement: FibonacciRetracement,
                          level: FibonacciLevel,
                          current_price: float) -> FibonacciSignal:
        """Create a buy signal at Fibonacci support"""
        entry_price = level.price * 1.001  # Slight premium for market entry
        stop_loss = retracement.invalidation_level

        # Target next resistance level or swing high
        if retracement.next_target:
            take_profit = retracement.next_target
        else:
            take_profit = retracement.swing_high * 0.95  # Conservative target

        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward = reward / risk if risk > 0 else 0.0

        # Calculate confidence
        confidence = (level.support_strength + level.volume_confirmation) * retracement.confidence

        return FibonacciSignal(
            symbol=retracement.symbol,
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            confidence=confidence,
            fibonacci_level=level.level,
            reasoning=f"Buy at {level.percentage}% Fibonacci support with {level.touch_count} touches"
        )

    def _create_sell_signal(self,
                           retracement: FibonacciRetracement,
                           level: FibonacciLevel,
                           current_price: float) -> FibonacciSignal:
        """Create a sell signal at Fibonacci resistance"""
        entry_price = level.price * 0.999  # Slight discount for market entry
        stop_loss = retracement.invalidation_level

        # Target next support level or swing low
        if retracement.next_target:
            take_profit = retracement.next_target
        else:
            take_profit = retracement.swing_low * 1.05  # Conservative target

        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(entry_price - take_profit)
        risk_reward = reward / risk if risk > 0 else 0.0

        # Calculate confidence
        confidence = (level.resistance_strength + level.volume_confirmation) * retracement.confidence

        return FibonacciSignal(
            symbol=retracement.symbol,
            timestamp=datetime.now(),
            signal_type='SELL',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            confidence=confidence,
            fibonacci_level=level.level,
            reasoning=f"Sell at {level.percentage}% Fibonacci resistance with {level.touch_count} touches"
        )

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.retracement_cache:
            return False

        cache_time = self.retracement_cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self.cache_duration

    def _create_empty_retracement(self, symbol: str, timeframe: str) -> FibonacciRetracement:
        """Create empty retracement for error cases"""
        return FibonacciRetracement(
            symbol=symbol,
            timeframe=timeframe,
            trend_direction=TrendDirection.SIDEWAYS,
            swing_high=0.0,
            swing_low=0.0,
            swing_high_time=datetime.now(),
            swing_low_time=datetime.now(),
            current_price=0.0,
            current_retracement=0.0,
            levels=[],
            key_levels=[],
            next_target=None,
            invalidation_level=0.0,
            confidence=0.0
        )
