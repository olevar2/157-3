"""
Session Support Resistance Module
Session-based support and resistance levels for swing trading
Optimized for Asian/London/NY session-specific level detection.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import deque
import statistics


@dataclass
class SessionInfo:
    """Trading session information"""
    name: str
    start_hour: int  # UTC hour
    end_hour: int    # UTC hour
    timezone_name: str
    volatility_profile: str  # 'high', 'medium', 'low'
    typical_range_pips: int


@dataclass
class SupportResistanceLevel:
    """Support or resistance level"""
    price: float
    level_type: str  # 'support', 'resistance'
    session: str
    strength: float  # 0-1 based on touches and holds
    touches: int
    last_test_time: float
    creation_time: float
    timeframe_validity: List[str]
    break_probability: float


@dataclass
class SessionRange:
    """Session price range"""
    session_name: str
    session_date: str
    high: float
    low: float
    open: float
    close: float
    range_pips: float
    volatility: float
    volume_profile: str  # 'high', 'medium', 'low'


@dataclass
class LevelBreakout:
    """Support/resistance level breakout"""
    level: SupportResistanceLevel
    breakout_type: str  # 'bullish_break', 'bearish_break', 'false_break'
    breakout_time: float
    breakout_strength: float
    volume_confirmation: bool
    target_projection: float


@dataclass
class SessionSupportResistanceResult:
    """Session-based support/resistance analysis result"""
    symbol: str
    timestamp: float
    current_session: str
    session_ranges: Dict[str, SessionRange]
    support_levels: List[SupportResistanceLevel]
    resistance_levels: List[SupportResistanceLevel]
    level_breakouts: List[LevelBreakout]
    session_bias: str  # 'bullish', 'bearish', 'neutral'
    key_levels: Dict[str, float]
    execution_zones: List[Tuple[float, float]]


class SessionSupportResistance:
    """
    Session Support Resistance Engine for Swing Trading
    Provides session-based support/resistance level detection for swing entries
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # Define major trading sessions
        self.trading_sessions = {
            'Asian': SessionInfo('Asian', 0, 9, 'Asia/Tokyo', 'medium', 60),
            'London': SessionInfo('London', 8, 17, 'Europe/London', 'high', 100),
            'NewYork': SessionInfo('NewYork', 13, 22, 'America/New_York', 'high', 90),
            'Overlap_London_NY': SessionInfo('Overlap_London_NY', 13, 17, 'UTC', 'high', 120)
        }
        
        # Level detection parameters
        self.min_level_strength = 0.6
        self.min_touches = 2
        self.level_proximity_pips = 10  # Levels within 10 pips are considered same
        self.max_level_age_hours = 168  # 1 week
        
        # Breakout detection parameters
        self.breakout_confirmation_pips = 15
        self.false_break_retest_periods = 5
        
        # Historical level storage
        self.session_levels: Dict[str, List[SupportResistanceLevel]] = {}
        self.session_ranges_cache: Dict[str, deque] = {}

    async def initialize(self) -> bool:
        """Initialize the Session Support Resistance engine"""
        try:
            self.logger.info("Initializing Session Support Resistance Engine...")
            
            # Test level detection with sample data
            test_data = self._generate_test_data()
            test_result = await self._detect_session_levels(test_data)
            
            if test_result and len(test_result) > 0:
                self.ready = True
                self.logger.info("✅ Session Support Resistance Engine initialized")
                return True
            else:
                raise Exception("Session level detection test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Session Support Resistance Engine initialization failed: {e}")
            return False

    async def analyze_session_levels(self, symbol: str, price_data: List[Dict], 
                                   volume_data: Optional[List[Dict]] = None,
                                   timeframe: str = 'H4') -> SessionSupportResistanceResult:
        """
        Analyze session-based support and resistance levels
        
        Args:
            symbol: Currency pair symbol
            price_data: List of OHLC data dictionaries
            volume_data: Optional volume data
            timeframe: Chart timeframe (default H4)
            
        Returns:
            SessionSupportResistanceResult with session level analysis
        """
        if not self.ready:
            raise Exception("Session Support Resistance Engine not initialized")
            
        if len(price_data) < 48:  # Minimum 2 days of H4 data
            raise Exception("Insufficient data for session analysis (minimum 48 periods)")
            
        try:
            start_time = time.time()
            
            # Extract price data
            closes = [float(data.get('close', 0)) for data in price_data]
            highs = [float(data.get('high', 0)) for data in price_data]
            lows = [float(data.get('low', 0)) for data in price_data]
            timestamps = [float(data.get('timestamp', time.time())) for data in price_data]
            
            # Determine current session
            current_session = self._get_current_session(time.time())
            
            # Calculate session ranges
            session_ranges = await self._calculate_session_ranges(symbol, price_data)
            
            # Detect support and resistance levels
            support_levels, resistance_levels = await self._detect_session_levels(price_data)
            
            # Detect level breakouts
            level_breakouts = await self._detect_level_breakouts(
                price_data, support_levels + resistance_levels, volume_data
            )
            
            # Determine session bias
            session_bias = await self._determine_session_bias(session_ranges, closes[-1])
            
            # Identify key levels for trading
            key_levels = await self._identify_key_levels(support_levels, resistance_levels, closes[-1])
            
            # Define execution zones
            execution_zones = await self._define_execution_zones(
                support_levels, resistance_levels, closes[-1]
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Session level analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return SessionSupportResistanceResult(
                symbol=symbol,
                timestamp=time.time(),
                current_session=current_session,
                session_ranges=session_ranges,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                level_breakouts=level_breakouts,
                session_bias=session_bias,
                key_levels=key_levels,
                execution_zones=execution_zones
            )
            
        except Exception as e:
            self.logger.error(f"Session level analysis failed for {symbol}: {e}")
            raise

    def _get_current_session(self, timestamp: float) -> str:
        """Determine the current trading session"""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        current_hour = dt.hour
        
        # Check for session overlaps first
        if 13 <= current_hour < 17:  # London-NY overlap
            return 'Overlap_London_NY'
        elif 8 <= current_hour < 17:  # London session
            return 'London'
        elif 13 <= current_hour < 22:  # NY session
            return 'NewYork'
        elif current_hour >= 0 and current_hour < 9:  # Asian session
            return 'Asian'
        else:
            return 'Asian'  # Default to Asian for other hours

    async def _calculate_session_ranges(self, symbol: str, 
                                      price_data: List[Dict]) -> Dict[str, SessionRange]:
        """Calculate price ranges for each trading session"""
        session_ranges = {}
        
        # Group data by session
        session_data = self._group_data_by_session(price_data)
        
        for session_name, data_points in session_data.items():
            if not data_points:
                continue
                
            highs = [float(d.get('high', 0)) for d in data_points]
            lows = [float(d.get('low', 0)) for d in data_points]
            closes = [float(d.get('close', 0)) for d in data_points]
            
            session_high = max(highs)
            session_low = min(lows)
            session_open = float(data_points[0].get('open', closes[0]))
            session_close = closes[-1]
            
            # Calculate range in pips (assuming 4-digit quotes)
            range_pips = (session_high - session_low) * 10000
            
            # Calculate volatility
            volatility = np.std(closes) if len(closes) > 1 else 0.0
            
            # Determine volume profile (simplified)
            volume_profile = 'medium'  # Default, would use actual volume data
            
            session_ranges[session_name] = SessionRange(
                session_name=session_name,
                session_date=datetime.fromtimestamp(data_points[0]['timestamp']).strftime('%Y-%m-%d'),
                high=session_high,
                low=session_low,
                open=session_open,
                close=session_close,
                range_pips=range_pips,
                volatility=volatility,
                volume_profile=volume_profile
            )
        
        return session_ranges

    def _group_data_by_session(self, price_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group price data by trading session"""
        session_data = {session: [] for session in self.trading_sessions.keys()}
        
        for data_point in price_data:
            timestamp = float(data_point.get('timestamp', time.time()))
            session = self._get_current_session(timestamp)
            session_data[session].append(data_point)
        
        return session_data

    async def _detect_session_levels(self, price_data: List[Dict]) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Detect support and resistance levels from session data"""
        support_levels = []
        resistance_levels = []
        
        # Group data by session
        session_data = self._group_data_by_session(price_data)
        
        for session_name, data_points in session_data.items():
            if len(data_points) < 6:  # Need minimum data for level detection
                continue
                
            highs = [float(d.get('high', 0)) for d in data_points]
            lows = [float(d.get('low', 0)) for d in data_points]
            timestamps = [float(d.get('timestamp', time.time())) for d in data_points]
            
            # Detect resistance levels (session highs that held)
            session_resistances = await self._find_resistance_levels(
                highs, timestamps, session_name
            )
            resistance_levels.extend(session_resistances)
            
            # Detect support levels (session lows that held)
            session_supports = await self._find_support_levels(
                lows, timestamps, session_name
            )
            support_levels.extend(session_supports)
        
        # Remove duplicate levels that are too close
        support_levels = self._remove_duplicate_levels(support_levels)
        resistance_levels = self._remove_duplicate_levels(resistance_levels)
        
        return support_levels, resistance_levels

    async def _find_resistance_levels(self, highs: List[float], timestamps: List[float], 
                                    session: str) -> List[SupportResistanceLevel]:
        """Find resistance levels from session highs"""
        resistance_levels = []
        
        # Find local maxima
        for i in range(2, len(highs) - 2):
            if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
                highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                
                # Count how many times this level was tested
                touches = self._count_level_touches(highs, highs[i], 'resistance')
                
                if touches >= self.min_touches:
                    strength = min(touches / 5.0, 1.0)  # Normalize to 0-1
                    
                    resistance_levels.append(SupportResistanceLevel(
                        price=highs[i],
                        level_type='resistance',
                        session=session,
                        strength=strength,
                        touches=touches,
                        last_test_time=timestamps[i],
                        creation_time=timestamps[i],
                        timeframe_validity=['H4', 'H1'],
                        break_probability=1.0 - strength
                    ))
        
        return resistance_levels

    async def _find_support_levels(self, lows: List[float], timestamps: List[float], 
                                 session: str) -> List[SupportResistanceLevel]:
        """Find support levels from session lows"""
        support_levels = []
        
        # Find local minima
        for i in range(2, len(lows) - 2):
            if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
                lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                
                # Count how many times this level was tested
                touches = self._count_level_touches(lows, lows[i], 'support')
                
                if touches >= self.min_touches:
                    strength = min(touches / 5.0, 1.0)  # Normalize to 0-1
                    
                    support_levels.append(SupportResistanceLevel(
                        price=lows[i],
                        level_type='support',
                        session=session,
                        strength=strength,
                        touches=touches,
                        last_test_time=timestamps[i],
                        creation_time=timestamps[i],
                        timeframe_validity=['H4', 'H1'],
                        break_probability=1.0 - strength
                    ))
        
        return support_levels

    def _count_level_touches(self, prices: List[float], level_price: float, 
                           level_type: str) -> int:
        """Count how many times a level was touched/tested"""
        touches = 0
        tolerance = level_price * 0.001  # 0.1% tolerance
        
        for price in prices:
            if level_type == 'resistance':
                if abs(price - level_price) <= tolerance and price <= level_price:
                    touches += 1
            else:  # support
                if abs(price - level_price) <= tolerance and price >= level_price:
                    touches += 1
        
        return touches

    def _remove_duplicate_levels(self, levels: List[SupportResistanceLevel]) -> List[SupportResistanceLevel]:
        """Remove levels that are too close to each other"""
        if not levels:
            return levels
            
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x.price)
        unique_levels = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if this level is too close to any existing level
            is_duplicate = False
            for existing_level in unique_levels:
                price_diff_pips = abs(level.price - existing_level.price) * 10000
                if price_diff_pips <= self.level_proximity_pips:
                    # Keep the stronger level
                    if level.strength > existing_level.strength:
                        unique_levels.remove(existing_level)
                        unique_levels.append(level)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_levels.append(level)
        
        return unique_levels

    async def _detect_level_breakouts(self, price_data: List[Dict], 
                                    levels: List[SupportResistanceLevel],
                                    volume_data: Optional[List[Dict]]) -> List[LevelBreakout]:
        """Detect breakouts of support/resistance levels"""
        breakouts = []
        
        if len(price_data) < 10:
            return breakouts
            
        recent_data = price_data[-10:]  # Look at recent 10 periods
        
        for level in levels:
            for i, data_point in enumerate(recent_data):
                high = float(data_point.get('high', 0))
                low = float(data_point.get('low', 0))
                close = float(data_point.get('close', 0))
                timestamp = float(data_point.get('timestamp', time.time()))
                
                # Check for resistance breakout
                if level.level_type == 'resistance' and close > level.price:
                    breakout_strength = (close - level.price) / level.price
                    
                    # Confirm it's not a false break
                    is_false_break = await self._check_false_breakout(
                        recent_data[i:], level, 'bullish'
                    )
                    
                    if not is_false_break:
                        target_projection = level.price + (level.price - level.price * 0.99)  # 1% projection
                        
                        breakouts.append(LevelBreakout(
                            level=level,
                            breakout_type='bullish_break',
                            breakout_time=timestamp,
                            breakout_strength=breakout_strength,
                            volume_confirmation=True,  # Simplified
                            target_projection=target_projection
                        ))
                
                # Check for support breakout
                elif level.level_type == 'support' and close < level.price:
                    breakout_strength = (level.price - close) / level.price
                    
                    is_false_break = await self._check_false_breakout(
                        recent_data[i:], level, 'bearish'
                    )
                    
                    if not is_false_break:
                        target_projection = level.price - (level.price * 0.01)  # 1% projection
                        
                        breakouts.append(LevelBreakout(
                            level=level,
                            breakout_type='bearish_break',
                            breakout_time=timestamp,
                            breakout_strength=breakout_strength,
                            volume_confirmation=True,  # Simplified
                            target_projection=target_projection
                        ))
        
        return breakouts

    async def _check_false_breakout(self, subsequent_data: List[Dict], 
                                  level: SupportResistanceLevel, direction: str) -> bool:
        """Check if a breakout is likely a false break"""
        if len(subsequent_data) < 3:
            return False  # Not enough data to confirm
            
        # Check if price returned to the level within next few periods
        for data_point in subsequent_data[1:4]:  # Check next 3 periods
            close = float(data_point.get('close', 0))
            
            if direction == 'bullish' and close <= level.price:
                return True  # Price came back below resistance
            elif direction == 'bearish' and close >= level.price:
                return True  # Price came back above support
        
        return False

    async def _determine_session_bias(self, session_ranges: Dict[str, SessionRange], 
                                    current_price: float) -> str:
        """Determine the overall session bias"""
        if not session_ranges:
            return 'neutral'
            
        # Get the most recent session
        latest_session = max(session_ranges.values(), key=lambda x: x.session_date)
        
        # Determine bias based on price position within session range
        session_mid = (latest_session.high + latest_session.low) / 2
        
        if current_price > session_mid * 1.002:  # Above midpoint + buffer
            return 'bullish'
        elif current_price < session_mid * 0.998:  # Below midpoint - buffer
            return 'bearish'
        else:
            return 'neutral'

    async def _identify_key_levels(self, support_levels: List[SupportResistanceLevel],
                                 resistance_levels: List[SupportResistanceLevel],
                                 current_price: float) -> Dict[str, float]:
        """Identify key support and resistance levels for trading"""
        key_levels = {}
        
        # Find nearest support below current price
        supports_below = [s for s in support_levels if s.price < current_price]
        if supports_below:
            nearest_support = max(supports_below, key=lambda x: x.price)
            key_levels['nearest_support'] = nearest_support.price
        
        # Find nearest resistance above current price
        resistances_above = [r for r in resistance_levels if r.price > current_price]
        if resistances_above:
            nearest_resistance = min(resistances_above, key=lambda x: x.price)
            key_levels['nearest_resistance'] = nearest_resistance.price
        
        # Find strongest levels
        all_levels = support_levels + resistance_levels
        if all_levels:
            strongest_level = max(all_levels, key=lambda x: x.strength)
            key_levels['strongest_level'] = strongest_level.price
        
        return key_levels

    async def _define_execution_zones(self, support_levels: List[SupportResistanceLevel],
                                    resistance_levels: List[SupportResistanceLevel],
                                    current_price: float) -> List[Tuple[float, float]]:
        """Define execution zones around key levels"""
        execution_zones = []
        
        # Create zones around strong levels
        strong_levels = [l for l in support_levels + resistance_levels if l.strength > 0.7]
        
        for level in strong_levels:
            # Create a zone around the level (±5 pips)
            zone_buffer = level.price * 0.0005  # 5 pips for major pairs
            zone_low = level.price - zone_buffer
            zone_high = level.price + zone_buffer
            execution_zones.append((zone_low, zone_high))
        
        return execution_zones

    def _generate_test_data(self) -> List[Dict]:
        """Generate test data for initialization"""
        test_data = []
        base_price = 1.1000
        
        # Create data with session-like patterns
        for i in range(72):  # 3 days of H4 data
            session_hour = (i * 4) % 24
            
            # Simulate session volatility
            if 8 <= session_hour < 17:  # London session
                volatility = 0.003
            elif 13 <= session_hour < 22:  # NY session
                volatility = 0.0025
            else:  # Asian session
                volatility = 0.0015
            
            price_change = (np.random.random() - 0.5) * volatility
            price = base_price + price_change + (i * 0.0001)  # Slight uptrend
            
            test_data.append({
                'timestamp': time.time() - (72 - i) * 3600 * 4,  # H4 intervals
                'open': price,
                'high': price + volatility * 0.3,
                'low': price - volatility * 0.3,
                'close': price,
                'volume': 1000
            })
            
        return test_data
