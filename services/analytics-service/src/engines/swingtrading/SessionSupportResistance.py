"""
Session-Based Support and Resistance Engine
Identifies key support/resistance levels based on trading sessions (Asian/London/NY).
Optimized for H4 timeframe and short-term swing trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from enum import Enum
import logging
import asyncio
import pytz

class TradingSession(Enum):
    """Major forex trading sessions"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "london_ny_overlap"

class LevelType(Enum):
    """Support/Resistance level types"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    PIVOT = "pivot"

class LevelStrength(Enum):
    """Level strength classification"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class SRLevel:
    """Support/Resistance level"""
    price: float
    level_type: LevelType
    strength: LevelStrength
    session: TradingSession
    formation_time: datetime
    touch_count: int
    last_touch_time: Optional[datetime]
    volume_confirmation: float
    break_count: int
    confidence: float
    age_hours: float

@dataclass
class SessionAnalysis:
    """Analysis for a specific trading session"""
    session: TradingSession
    start_time: datetime
    end_time: datetime
    session_high: float
    session_low: float
    session_range: float
    volume_profile: Dict[str, float]
    key_levels: List[SRLevel]
    breakout_probability: float

@dataclass
class SRAnalysisResult:
    """Complete Support/Resistance analysis result"""
    symbol: str
    timestamp: datetime
    timeframe: str
    current_price: float
    session_analyses: List[SessionAnalysis]
    all_levels: List[SRLevel]
    nearest_support: Optional[SRLevel]
    nearest_resistance: Optional[SRLevel]
    key_levels: List[SRLevel]
    trading_signals: List[Dict[str, Any]]

class SessionSupportResistance:
    """
    Session-Based Support and Resistance Engine
    Analyzes S/R levels based on major forex trading sessions
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Session time definitions (UTC)
        self.session_times = {
            TradingSession.ASIAN: (time(22, 0), time(7, 0)),  # 22:00-07:00 UTC
            TradingSession.LONDON: (time(7, 0), time(16, 0)),  # 07:00-16:00 UTC
            TradingSession.NEW_YORK: (time(13, 0), time(22, 0)),  # 13:00-22:00 UTC
            TradingSession.OVERLAP_LONDON_NY: (time(13, 0), time(16, 0))  # 13:00-16:00 UTC
        }

        # Level detection parameters
        self.min_level_strength = 0.0005  # 5 pips minimum for major pairs
        self.level_tolerance = 0.0003  # 3 pips tolerance for level touches
        self.min_touch_count = 2
        self.max_level_age_hours = 120  # 5 days maximum

        # Volume analysis parameters
        self.volume_multiplier_threshold = 1.5

        # Cache for performance
        self.sr_cache = {}
        self.cache_duration = timedelta(minutes=30)

    async def analyze_session_sr_levels(self,
                                      symbol: str,
                                      price_data: pd.DataFrame,
                                      timeframe: str = "H4") -> SRAnalysisResult:
        """
        Analyze support/resistance levels based on trading sessions

        Args:
            symbol: Trading symbol
            price_data: OHLCV data with datetime index
            timeframe: Analysis timeframe

        Returns:
            SRAnalysisResult with session-based S/R analysis
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{len(price_data)}"
            if self._is_cache_valid(cache_key):
                return self.sr_cache[cache_key]['result']

            # Ensure datetime index
            if not isinstance(price_data.index, pd.DatetimeIndex):
                price_data.index = pd.to_datetime(price_data.index)

            # Analyze each trading session
            session_analyses = []
            all_levels = []

            for session in TradingSession:
                session_analysis = await self._analyze_session(session, price_data)
                if session_analysis:
                    session_analyses.append(session_analysis)
                    all_levels.extend(session_analysis.key_levels)

            # Remove duplicate and overlapping levels
            all_levels = self._consolidate_levels(all_levels)

            # Find nearest support and resistance
            current_price = price_data.iloc[-1]['close']
            nearest_support, nearest_resistance = self._find_nearest_levels(current_price, all_levels)

            # Identify key levels (strongest levels)
            key_levels = self._identify_key_levels(all_levels)

            # Generate trading signals
            trading_signals = await self._generate_sr_signals(current_price, key_levels, nearest_support, nearest_resistance)

            result = SRAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                timeframe=timeframe,
                current_price=current_price,
                session_analyses=session_analyses,
                all_levels=all_levels,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                key_levels=key_levels,
                trading_signals=trading_signals
            )

            # Cache result
            self.sr_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }

            return result

        except Exception as e:
            self.logger.error(f"Session S/R analysis error for {symbol}: {e}")
            return self._create_empty_result(symbol, timeframe)

    async def _analyze_session(self, session: TradingSession, price_data: pd.DataFrame) -> Optional[SessionAnalysis]:
        """Analyze support/resistance levels for a specific session"""
        # Get session data
        session_data = self._filter_session_data(session, price_data)

        if len(session_data) < 10:  # Need minimum data
            return None

        # Calculate session statistics
        session_high = session_data['high'].max()
        session_low = session_data['low'].min()
        session_range = session_high - session_low

        # Calculate volume profile
        volume_profile = self._calculate_volume_profile(session_data)

        # Identify key levels within session
        key_levels = await self._identify_session_levels(session, session_data)

        # Calculate breakout probability
        breakout_probability = self._calculate_breakout_probability(session_data)

        return SessionAnalysis(
            session=session,
            start_time=session_data.index[0],
            end_time=session_data.index[-1],
            session_high=session_high,
            session_low=session_low,
            session_range=session_range,
            volume_profile=volume_profile,
            key_levels=key_levels,
            breakout_probability=breakout_probability
        )

    def _filter_session_data(self, session: TradingSession, price_data: pd.DataFrame) -> pd.DataFrame:
        """Filter price data for specific trading session"""
        session_start, session_end = self.session_times[session]

        # Handle sessions that cross midnight
        if session_start > session_end:  # Asian session case
            mask = (price_data.index.time >= session_start) | (price_data.index.time <= session_end)
        else:
            mask = (price_data.index.time >= session_start) & (price_data.index.time <= session_end)

        return price_data[mask]

    def _calculate_volume_profile(self, session_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile for session"""
        if 'volume' not in session_data.columns:
            return {'total': 0.0, 'average': 0.0, 'peak': 0.0}

        total_volume = session_data['volume'].sum()
        avg_volume = session_data['volume'].mean()
        peak_volume = session_data['volume'].max()

        return {
            'total': total_volume,
            'average': avg_volume,
            'peak': peak_volume
        }

    async def _identify_session_levels(self, session: TradingSession, session_data: pd.DataFrame) -> List[SRLevel]:
        """Identify key support/resistance levels within a session"""
        levels = []

        # Find pivot points
        pivot_highs = self._find_pivot_points(session_data['high'].values, order=2)
        pivot_lows = self._find_pivot_points(session_data['low'].values, order=2, find_peaks=False)

        # Process pivot highs (potential resistance)
        for i, is_pivot in enumerate(pivot_highs):
            if is_pivot and i < len(session_data):
                level = await self._create_sr_level(
                    price=session_data.iloc[i]['high'],
                    level_type=LevelType.RESISTANCE,
                    session=session,
                    formation_time=session_data.index[i],
                    session_data=session_data
                )
                if level:
                    levels.append(level)

        # Process pivot lows (potential support)
        for i, is_pivot in enumerate(pivot_lows):
            if is_pivot and i < len(session_data):
                level = await self._create_sr_level(
                    price=session_data.iloc[i]['low'],
                    level_type=LevelType.SUPPORT,
                    session=session,
                    formation_time=session_data.index[i],
                    session_data=session_data
                )
                if level:
                    levels.append(level)

        # Add session high/low as levels
        session_high_level = await self._create_sr_level(
            price=session_data['high'].max(),
            level_type=LevelType.RESISTANCE,
            session=session,
            formation_time=session_data['high'].idxmax(),
            session_data=session_data
        )
        if session_high_level:
            levels.append(session_high_level)

        session_low_level = await self._create_sr_level(
            price=session_data['low'].min(),
            level_type=LevelType.SUPPORT,
            session=session,
            formation_time=session_data['low'].idxmin(),
            session_data=session_data
        )
        if session_low_level:
            levels.append(session_low_level)

        return levels

    def _find_pivot_points(self, data: np.ndarray, order: int = 2, find_peaks: bool = True) -> List[bool]:
        """Find pivot points in price data"""
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

    async def _create_sr_level(self,
                             price: float,
                             level_type: LevelType,
                             session: TradingSession,
                             formation_time: datetime,
                             session_data: pd.DataFrame) -> Optional[SRLevel]:
        """Create a support/resistance level with analysis"""
        # Count touches and analyze strength
        touch_count, last_touch, break_count = self._analyze_level_touches(price, session_data)

        if touch_count < self.min_touch_count:
            return None

        # Calculate volume confirmation
        volume_confirmation = self._calculate_level_volume_confirmation(price, session_data)

        # Determine strength
        strength = self._calculate_level_strength(touch_count, volume_confirmation, break_count)

        # Calculate age
        age_hours = (datetime.now() - formation_time).total_seconds() / 3600

        if age_hours > self.max_level_age_hours:
            return None

        # Calculate confidence
        confidence = self._calculate_level_confidence(touch_count, volume_confirmation, break_count, age_hours)

        return SRLevel(
            price=price,
            level_type=level_type,
            strength=strength,
            session=session,
            formation_time=formation_time,
            touch_count=touch_count,
            last_touch_time=last_touch,
            volume_confirmation=volume_confirmation,
            break_count=break_count,
            confidence=confidence,
            age_hours=age_hours
        )

    def _analyze_level_touches(self, level_price: float, session_data: pd.DataFrame) -> Tuple[int, Optional[datetime], int]:
        """Analyze how many times price touched a level and how many times it broke"""
        touch_count = 0
        break_count = 0
        last_touch = None

        for idx, row in session_data.iterrows():
            # Check if price touched the level
            if (row['low'] <= level_price + self.level_tolerance and
                row['high'] >= level_price - self.level_tolerance):
                touch_count += 1
                last_touch = idx

                # Check if it was a break (close beyond level)
                if row['close'] > level_price + self.level_tolerance or row['close'] < level_price - self.level_tolerance:
                    break_count += 1

        return touch_count, last_touch, break_count

    def _calculate_level_volume_confirmation(self, level_price: float, session_data: pd.DataFrame) -> float:
        """Calculate volume confirmation at a level"""
        if 'volume' not in session_data.columns:
            return 0.5  # Default neutral confirmation

        level_volumes = []
        avg_volume = session_data['volume'].mean()

        for idx, row in session_data.iterrows():
            if (row['low'] <= level_price + self.level_tolerance and
                row['high'] >= level_price - self.level_tolerance):
                level_volumes.append(row['volume'])

        if not level_volumes:
            return 0.0

        avg_level_volume = np.mean(level_volumes)
        return avg_level_volume / avg_volume if avg_volume > 0 else 0.0

    def _calculate_level_strength(self, touch_count: int, volume_confirmation: float, break_count: int) -> LevelStrength:
        """Calculate the strength of a support/resistance level"""
        # Base strength from touch count
        strength_score = touch_count * 0.3

        # Volume confirmation boost
        strength_score += volume_confirmation * 0.4

        # Penalty for breaks
        strength_score -= break_count * 0.2

        # Classify strength
        if strength_score >= 2.0:
            return LevelStrength.VERY_STRONG
        elif strength_score >= 1.5:
            return LevelStrength.STRONG
        elif strength_score >= 1.0:
            return LevelStrength.MODERATE
        else:
            return LevelStrength.WEAK

    def _calculate_level_confidence(self, touch_count: int, volume_confirmation: float, break_count: int, age_hours: float) -> float:
        """Calculate confidence in a support/resistance level"""
        confidence = 0.5  # Base confidence

        # Touch count boost
        confidence += min(touch_count * 0.1, 0.3)

        # Volume confirmation boost
        confidence += min(volume_confirmation * 0.2, 0.2)

        # Break penalty
        confidence -= min(break_count * 0.1, 0.2)

        # Age penalty (older levels less reliable)
        age_penalty = min(age_hours / self.max_level_age_hours * 0.1, 0.1)
        confidence -= age_penalty

        return max(0.0, min(1.0, confidence))

    def _calculate_breakout_probability(self, session_data: pd.DataFrame) -> float:
        """Calculate probability of breakout from session range"""
        if len(session_data) < 5:
            return 0.5

        # Analyze price action near session boundaries
        session_high = session_data['high'].max()
        session_low = session_data['low'].min()
        session_range = session_high - session_low

        if session_range == 0:
            return 0.5

        # Count touches near boundaries
        high_touches = 0
        low_touches = 0
        tolerance = session_range * 0.1  # 10% of range

        for idx, row in session_data.iterrows():
            if row['high'] >= session_high - tolerance:
                high_touches += 1
            if row['low'] <= session_low + tolerance:
                low_touches += 1

        # More touches = higher breakout probability
        total_touches = high_touches + low_touches
        breakout_prob = min(total_touches / len(session_data), 0.8)

        return breakout_prob

    def _consolidate_levels(self, all_levels: List[SRLevel]) -> List[SRLevel]:
        """Remove duplicate and overlapping levels"""
        if not all_levels:
            return all_levels

        # Sort by price
        all_levels.sort(key=lambda l: l.price)

        consolidated = []
        for level in all_levels:
            # Check if this level is too close to an existing one
            is_duplicate = False
            for existing in consolidated:
                price_diff = abs(level.price - existing.price) / existing.price
                if price_diff < 0.001:  # Within 10 pips
                    # Keep the stronger level
                    if level.confidence > existing.confidence:
                        consolidated.remove(existing)
                        consolidated.append(level)
                    is_duplicate = True
                    break

            if not is_duplicate:
                consolidated.append(level)

        return consolidated

    def _find_nearest_levels(self, current_price: float, all_levels: List[SRLevel]) -> Tuple[Optional[SRLevel], Optional[SRLevel]]:
        """Find nearest support and resistance levels"""
        nearest_support = None
        nearest_resistance = None

        for level in all_levels:
            if level.level_type == LevelType.SUPPORT and level.price < current_price:
                if not nearest_support or level.price > nearest_support.price:
                    nearest_support = level
            elif level.level_type == LevelType.RESISTANCE and level.price > current_price:
                if not nearest_resistance or level.price < nearest_resistance.price:
                    nearest_resistance = level

        return nearest_support, nearest_resistance

    def _identify_key_levels(self, all_levels: List[SRLevel]) -> List[SRLevel]:
        """Identify the most important support/resistance levels"""
        # Filter for strong levels with high confidence
        key_levels = [
            level for level in all_levels
            if level.strength in [LevelStrength.STRONG, LevelStrength.VERY_STRONG] and level.confidence >= 0.7
        ]

        # Sort by confidence
        key_levels.sort(key=lambda l: l.confidence, reverse=True)

        # Return top 10 levels
        return key_levels[:10]

    async def _generate_sr_signals(self,
                                 current_price: float,
                                 key_levels: List[SRLevel],
                                 nearest_support: Optional[SRLevel],
                                 nearest_resistance: Optional[SRLevel]) -> List[Dict[str, Any]]:
        """Generate trading signals based on S/R analysis"""
        signals = []

        # Signal at nearest support
        if nearest_support and nearest_support.confidence >= 0.7:
            distance_to_support = abs(current_price - nearest_support.price) / current_price
            if distance_to_support <= 0.002:  # Within 20 pips
                signals.append({
                    'type': 'BUY',
                    'reason': 'Near strong support',
                    'entry': nearest_support.price * 1.001,
                    'stop_loss': nearest_support.price * 0.995,
                    'take_profit': nearest_resistance.price * 0.99 if nearest_resistance else current_price * 1.02,
                    'confidence': nearest_support.confidence,
                    'session': nearest_support.session.value
                })

        # Signal at nearest resistance
        if nearest_resistance and nearest_resistance.confidence >= 0.7:
            distance_to_resistance = abs(current_price - nearest_resistance.price) / current_price
            if distance_to_resistance <= 0.002:  # Within 20 pips
                signals.append({
                    'type': 'SELL',
                    'reason': 'Near strong resistance',
                    'entry': nearest_resistance.price * 0.999,
                    'stop_loss': nearest_resistance.price * 1.005,
                    'take_profit': nearest_support.price * 1.01 if nearest_support else current_price * 0.98,
                    'confidence': nearest_resistance.confidence,
                    'session': nearest_resistance.session.value
                })

        # Breakout signals
        for level in key_levels:
            if level.strength == LevelStrength.VERY_STRONG:
                distance = abs(current_price - level.price) / current_price
                if distance <= 0.001:  # Very close to strong level
                    if level.level_type == LevelType.RESISTANCE:
                        signals.append({
                            'type': 'BUY_BREAKOUT',
                            'reason': 'Potential resistance breakout',
                            'entry': level.price * 1.002,
                            'stop_loss': level.price * 0.998,
                            'take_profit': level.price * 1.02,
                            'confidence': level.confidence * 0.8,  # Lower confidence for breakouts
                            'session': level.session.value
                        })
                    else:  # SUPPORT
                        signals.append({
                            'type': 'SELL_BREAKDOWN',
                            'reason': 'Potential support breakdown',
                            'entry': level.price * 0.998,
                            'stop_loss': level.price * 1.002,
                            'take_profit': level.price * 0.98,
                            'confidence': level.confidence * 0.8,
                            'session': level.session.value
                        })

        return signals

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.sr_cache:
            return False

        cache_time = self.sr_cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self.cache_duration

    def _create_empty_result(self, symbol: str, timeframe: str) -> SRAnalysisResult:
        """Create empty result for error cases"""
        return SRAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            timeframe=timeframe,
            current_price=0.0,
            session_analyses=[],
            all_levels=[],
            nearest_support=None,
            nearest_resistance=None,
            key_levels=[],
            trading_signals=[]
        )
