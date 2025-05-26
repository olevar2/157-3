"""
Volume Profiles Module
Session-based volume profiles for scalping and day trading
Optimized for volume-based breakout validation and key level identification.
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
class VolumeNode:
    """Volume profile node at specific price level"""
    price_level: float
    volume: float
    percentage: float
    buy_volume: float
    sell_volume: float
    delta: float  # buy_volume - sell_volume
    trades_count: int


@dataclass
class ValueArea:
    """Value area definition"""
    value_area_high: float
    value_area_low: float
    point_of_control: float
    value_area_volume: float
    value_area_percentage: float
    total_volume: float


@dataclass
class SessionProfile:
    """Session-based volume profile"""
    session_name: str
    session_start: float
    session_end: float
    volume_nodes: List[VolumeNode]
    value_area: ValueArea
    session_high: float
    session_low: float
    session_volume: float
    profile_type: str  # 'balanced', 'trending', 'rotational'


@dataclass
class ProfileBreakout:
    """Volume profile breakout signal"""
    breakout_type: str  # 'value_area_high', 'value_area_low', 'poc_break'
    breakout_level: float
    volume_confirmation: bool
    breakout_strength: float
    target_projection: float
    confidence: float


@dataclass
class VolumeProfileResult:
    """Volume profile analysis result"""
    symbol: str
    timestamp: float
    timeframe: str
    current_session_profile: SessionProfile
    historical_profiles: List[SessionProfile]
    profile_breakouts: List[ProfileBreakout]
    key_volume_levels: Dict[str, float]
    volume_distribution: str  # 'normal', 'skewed_high', 'skewed_low', 'bimodal'
    trading_signals: List[Dict[str, Union[str, float]]]


class VolumeProfiles:
    """
    Volume Profiles Engine for Day Trading
    Provides session-based volume profiles for breakout validation
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # Profile construction parameters
        self.price_levels = 50  # Number of price levels for profile
        self.value_area_percentage = 0.70  # 70% value area
        self.min_session_volume = 1000  # Minimum volume for valid session
        
        # Trading sessions
        self.trading_sessions = {
            'Asian': {'start': 0, 'end': 9},      # UTC hours
            'London': {'start': 8, 'end': 17},
            'NewYork': {'start': 13, 'end': 22},
            'Overlap': {'start': 13, 'end': 17}   # London-NY overlap
        }
        
        # Breakout detection parameters
        self.breakout_volume_threshold = 1.5  # Volume confirmation threshold
        self.poc_break_threshold = 0.0010     # 10 pips for major pairs
        
        # Performance optimization
        self.profile_cache: Dict[str, List[SessionProfile]] = {}
        self.level_cache: Dict[str, Dict[float, VolumeNode]] = {}

    async def initialize(self) -> bool:
        """Initialize the Volume Profiles engine"""
        try:
            self.logger.info("Initializing Volume Profiles Engine...")
            
            # Test volume profile construction with sample data
            test_data = self._generate_test_data()
            test_result = await self._build_volume_profile(test_data)
            
            if test_result and test_result.volume_nodes:
                self.ready = True
                self.logger.info("✅ Volume Profiles Engine initialized")
                return True
            else:
                raise Exception("Volume profile construction test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Volume Profiles Engine initialization failed: {e}")
            return False

    async def analyze_volume_profiles(self, symbol: str, price_data: List[Dict], 
                                    volume_data: List[Dict],
                                    timeframe: str = 'M15') -> VolumeProfileResult:
        """
        Analyze session-based volume profiles
        
        Args:
            symbol: Currency pair symbol
            price_data: List of OHLC data dictionaries
            volume_data: List of volume data dictionaries
            timeframe: Chart timeframe (M15-H1)
            
        Returns:
            VolumeProfileResult with volume profile analysis
        """
        if not self.ready:
            raise Exception("Volume Profiles Engine not initialized")
            
        if len(price_data) < 20 or len(volume_data) < 20:
            raise Exception("Insufficient data for volume profile analysis (minimum 20 periods)")
            
        try:
            start_time = time.time()
            
            # Group data by trading sessions
            session_data = await self._group_data_by_sessions(price_data, volume_data)
            
            # Build current session profile
            current_session_profile = await self._build_current_session_profile(session_data)
            
            # Build historical session profiles
            historical_profiles = await self._build_historical_profiles(session_data)
            
            # Detect profile breakouts
            profile_breakouts = await self._detect_profile_breakouts(
                current_session_profile, historical_profiles, price_data[-1]
            )
            
            # Identify key volume levels
            key_volume_levels = await self._identify_key_volume_levels(
                current_session_profile, historical_profiles
            )
            
            # Analyze volume distribution
            volume_distribution = await self._analyze_volume_distribution(current_session_profile)
            
            # Generate trading signals
            trading_signals = await self._generate_trading_signals(
                symbol, current_session_profile, profile_breakouts, timeframe
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Volume profile analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return VolumeProfileResult(
                symbol=symbol,
                timestamp=time.time(),
                timeframe=timeframe,
                current_session_profile=current_session_profile,
                historical_profiles=historical_profiles,
                profile_breakouts=profile_breakouts,
                key_volume_levels=key_volume_levels,
                volume_distribution=volume_distribution,
                trading_signals=trading_signals
            )
            
        except Exception as e:
            self.logger.error(f"Volume profile analysis failed for {symbol}: {e}")
            raise

    async def _group_data_by_sessions(self, price_data: List[Dict], 
                                    volume_data: List[Dict]) -> Dict[str, List[Tuple[Dict, Dict]]]:
        """Group price and volume data by trading sessions"""
        session_data = {session: [] for session in self.trading_sessions.keys()}
        
        # Ensure data alignment
        min_length = min(len(price_data), len(volume_data))
        
        for i in range(min_length):
            price_bar = price_data[i]
            volume_bar = volume_data[i]
            
            timestamp = float(price_bar.get('timestamp', time.time()))
            session = self._get_trading_session(timestamp)
            
            if session:
                session_data[session].append((price_bar, volume_bar))
        
        return session_data

    def _get_trading_session(self, timestamp: float) -> Optional[str]:
        """Determine trading session for given timestamp"""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        hour = dt.hour
        
        # Check for overlap first
        if 13 <= hour < 17:
            return 'Overlap'
        elif 8 <= hour < 17:
            return 'London'
        elif 13 <= hour < 22:
            return 'NewYork'
        elif 0 <= hour < 9:
            return 'Asian'
        
        return None

    async def _build_current_session_profile(self, session_data: Dict[str, List[Tuple[Dict, Dict]]]) -> SessionProfile:
        """Build volume profile for current session"""
        # Get the most recent session with data
        current_session = None
        current_data = []
        
        for session_name, data in session_data.items():
            if data and len(data) > len(current_data):
                current_session = session_name
                current_data = data
        
        if not current_data:
            # Return empty profile
            return SessionProfile(
                session_name='Unknown',
                session_start=time.time(),
                session_end=time.time(),
                volume_nodes=[],
                value_area=ValueArea(0, 0, 0, 0, 0, 0),
                session_high=0,
                session_low=0,
                session_volume=0,
                profile_type='balanced'
            )
        
        return await self._build_volume_profile(current_data, current_session)

    async def _build_volume_profile(self, session_data: List[Tuple[Dict, Dict]], 
                                  session_name: str = 'Current') -> SessionProfile:
        """Build volume profile from session data"""
        if not session_data:
            return SessionProfile(
                session_name=session_name,
                session_start=time.time(),
                session_end=time.time(),
                volume_nodes=[],
                value_area=ValueArea(0, 0, 0, 0, 0, 0),
                session_high=0,
                session_low=0,
                session_volume=0,
                profile_type='balanced'
            )
        
        # Extract price and volume data
        prices = []
        volumes = []
        timestamps = []
        
        for price_bar, volume_bar in session_data:
            high = float(price_bar.get('high', 0))
            low = float(price_bar.get('low', 0))
            close = float(price_bar.get('close', 0))
            volume = float(volume_bar.get('volume', 0))
            timestamp = float(price_bar.get('timestamp', time.time()))
            
            prices.extend([high, low, close])
            volumes.extend([volume/3, volume/3, volume/3])  # Distribute volume
            timestamps.append(timestamp)
        
        if not prices:
            return SessionProfile(
                session_name=session_name,
                session_start=time.time(),
                session_end=time.time(),
                volume_nodes=[],
                value_area=ValueArea(0, 0, 0, 0, 0, 0),
                session_high=0,
                session_low=0,
                session_volume=0,
                profile_type='balanced'
            )
        
        # Calculate session boundaries
        session_high = max(prices)
        session_low = min(prices)
        session_volume = sum(volumes)
        session_start = min(timestamps)
        session_end = max(timestamps)
        
        # Create price levels
        price_range = session_high - session_low
        if price_range == 0:
            price_range = session_high * 0.001  # 0.1% range if no movement
        
        level_size = price_range / self.price_levels
        volume_nodes = []
        
        # Build volume profile
        for i in range(self.price_levels):
            level_price = session_low + (i * level_size)
            level_volume = 0
            level_buy_volume = 0
            level_sell_volume = 0
            trades_count = 0
            
            # Accumulate volume at this price level
            for j, (price_bar, volume_bar) in enumerate(session_data):
                high = float(price_bar.get('high', 0))
                low = float(price_bar.get('low', 0))
                close = float(price_bar.get('close', 0))
                open_price = float(price_bar.get('open', close))
                volume = float(volume_bar.get('volume', 0))
                
                # Check if this price level intersects with the bar
                if low <= level_price <= high:
                    # Distribute volume proportionally
                    bar_range = high - low
                    if bar_range > 0:
                        volume_at_level = volume * (level_size / bar_range)
                    else:
                        volume_at_level = volume
                    
                    level_volume += volume_at_level
                    trades_count += 1
                    
                    # Estimate buy/sell volume based on close position
                    if close > open_price:
                        level_buy_volume += volume_at_level * 0.6
                        level_sell_volume += volume_at_level * 0.4
                    elif close < open_price:
                        level_buy_volume += volume_at_level * 0.4
                        level_sell_volume += volume_at_level * 0.6
                    else:
                        level_buy_volume += volume_at_level * 0.5
                        level_sell_volume += volume_at_level * 0.5
            
            if level_volume > 0:
                volume_nodes.append(VolumeNode(
                    price_level=level_price,
                    volume=level_volume,
                    percentage=(level_volume / session_volume) * 100,
                    buy_volume=level_buy_volume,
                    sell_volume=level_sell_volume,
                    delta=level_buy_volume - level_sell_volume,
                    trades_count=trades_count
                ))
        
        # Calculate value area
        value_area = await self._calculate_value_area(volume_nodes, session_volume)
        
        # Determine profile type
        profile_type = await self._determine_profile_type(volume_nodes, value_area)
        
        return SessionProfile(
            session_name=session_name,
            session_start=session_start,
            session_end=session_end,
            volume_nodes=volume_nodes,
            value_area=value_area,
            session_high=session_high,
            session_low=session_low,
            session_volume=session_volume,
            profile_type=profile_type
        )

    async def _calculate_value_area(self, volume_nodes: List[VolumeNode], 
                                  total_volume: float) -> ValueArea:
        """Calculate value area (70% of volume)"""
        if not volume_nodes or total_volume == 0:
            return ValueArea(0, 0, 0, 0, 0, 0)
        
        # Find Point of Control (highest volume node)
        poc_node = max(volume_nodes, key=lambda x: x.volume)
        poc_price = poc_node.price_level
        
        # Sort nodes by volume (descending)
        sorted_nodes = sorted(volume_nodes, key=lambda x: x.volume, reverse=True)
        
        # Calculate value area
        target_volume = total_volume * self.value_area_percentage
        accumulated_volume = 0
        value_area_nodes = []
        
        for node in sorted_nodes:
            accumulated_volume += node.volume
            value_area_nodes.append(node)
            
            if accumulated_volume >= target_volume:
                break
        
        # Find value area high and low
        if value_area_nodes:
            value_area_high = max(node.price_level for node in value_area_nodes)
            value_area_low = min(node.price_level for node in value_area_nodes)
        else:
            value_area_high = poc_price
            value_area_low = poc_price
        
        return ValueArea(
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            point_of_control=poc_price,
            value_area_volume=accumulated_volume,
            value_area_percentage=(accumulated_volume / total_volume) * 100,
            total_volume=total_volume
        )

    async def _determine_profile_type(self, volume_nodes: List[VolumeNode], 
                                    value_area: ValueArea) -> str:
        """Determine the type of volume profile"""
        if not volume_nodes:
            return 'balanced'
        
        # Calculate profile characteristics
        total_volume = sum(node.volume for node in volume_nodes)
        
        # Check for balanced profile (value area in middle third)
        price_range = max(node.price_level for node in volume_nodes) - min(node.price_level for node in volume_nodes)
        if price_range == 0:
            return 'balanced'
        
        va_center = (value_area.value_area_high + value_area.value_area_low) / 2
        profile_center = (max(node.price_level for node in volume_nodes) + min(node.price_level for node in volume_nodes)) / 2
        
        center_deviation = abs(va_center - profile_center) / price_range
        
        if center_deviation < 0.2:  # Value area in center 20%
            return 'balanced'
        elif va_center > profile_center:
            return 'trending'  # Value area shifted up
        else:
            return 'rotational'  # Value area shifted down

    async def _build_historical_profiles(self, session_data: Dict[str, List[Tuple[Dict, Dict]]]) -> List[SessionProfile]:
        """Build historical session profiles"""
        historical_profiles = []
        
        for session_name, data in session_data.items():
            if data and len(data) >= 5:  # Minimum data for meaningful profile
                profile = await self._build_volume_profile(data, session_name)
                if profile.session_volume >= self.min_session_volume:
                    historical_profiles.append(profile)
        
        return historical_profiles

    async def _detect_profile_breakouts(self, current_profile: SessionProfile,
                                      historical_profiles: List[SessionProfile],
                                      current_price_data: Dict) -> List[ProfileBreakout]:
        """Detect volume profile breakouts"""
        breakouts = []
        
        if not current_profile.volume_nodes:
            return breakouts
        
        current_price = float(current_price_data.get('close', 0))
        current_volume = float(current_price_data.get('volume', 0))
        
        # Check for value area breakouts
        va = current_profile.value_area
        
        # Value Area High breakout
        if current_price > va.value_area_high:
            volume_confirmation = current_volume > (current_profile.session_volume / len(current_profile.volume_nodes)) * self.breakout_volume_threshold
            breakout_strength = (current_price - va.value_area_high) / va.value_area_high
            
            breakouts.append(ProfileBreakout(
                breakout_type='value_area_high',
                breakout_level=va.value_area_high,
                volume_confirmation=volume_confirmation,
                breakout_strength=breakout_strength,
                target_projection=va.value_area_high + (va.value_area_high - va.value_area_low),
                confidence=0.8 if volume_confirmation else 0.6
            ))
        
        # Value Area Low breakout
        elif current_price < va.value_area_low:
            volume_confirmation = current_volume > (current_profile.session_volume / len(current_profile.volume_nodes)) * self.breakout_volume_threshold
            breakout_strength = (va.value_area_low - current_price) / va.value_area_low
            
            breakouts.append(ProfileBreakout(
                breakout_type='value_area_low',
                breakout_level=va.value_area_low,
                volume_confirmation=volume_confirmation,
                breakout_strength=breakout_strength,
                target_projection=va.value_area_low - (va.value_area_high - va.value_area_low),
                confidence=0.8 if volume_confirmation else 0.6
            ))
        
        # Point of Control breakout
        poc_distance = abs(current_price - va.point_of_control)
        if poc_distance > self.poc_break_threshold:
            breakouts.append(ProfileBreakout(
                breakout_type='poc_break',
                breakout_level=va.point_of_control,
                volume_confirmation=True,  # Simplified
                breakout_strength=poc_distance / va.point_of_control,
                target_projection=current_price + (current_price - va.point_of_control),
                confidence=0.7
            ))
        
        return breakouts

    async def _identify_key_volume_levels(self, current_profile: SessionProfile,
                                        historical_profiles: List[SessionProfile]) -> Dict[str, float]:
        """Identify key volume levels from profiles"""
        key_levels = {}
        
        # Current session levels
        if current_profile.value_area:
            key_levels['current_poc'] = current_profile.value_area.point_of_control
            key_levels['current_va_high'] = current_profile.value_area.value_area_high
            key_levels['current_va_low'] = current_profile.value_area.value_area_low
        
        # Historical high volume levels
        all_nodes = []
        for profile in historical_profiles:
            all_nodes.extend(profile.volume_nodes)
        
        if all_nodes:
            # Find highest volume nodes across all sessions
            sorted_nodes = sorted(all_nodes, key=lambda x: x.volume, reverse=True)
            
            if len(sorted_nodes) >= 1:
                key_levels['highest_volume_level'] = sorted_nodes[0].price_level
            if len(sorted_nodes) >= 3:
                key_levels['third_highest_volume'] = sorted_nodes[2].price_level
        
        return key_levels

    async def _analyze_volume_distribution(self, profile: SessionProfile) -> str:
        """Analyze volume distribution pattern"""
        if not profile.volume_nodes:
            return 'normal'
        
        # Calculate distribution characteristics
        volumes = [node.volume for node in profile.volume_nodes]
        prices = [node.price_level for node in profile.volume_nodes]
        
        # Find peaks in volume distribution
        peaks = []
        for i in range(1, len(volumes) - 1):
            if volumes[i] > volumes[i-1] and volumes[i] > volumes[i+1]:
                peaks.append((prices[i], volumes[i]))
        
        if len(peaks) >= 2:
            return 'bimodal'
        elif len(peaks) == 1:
            # Check if peak is in upper or lower half
            peak_price = peaks[0][0]
            price_range = max(prices) - min(prices)
            mid_price = min(prices) + price_range / 2
            
            if peak_price > mid_price + price_range * 0.2:
                return 'skewed_high'
            elif peak_price < mid_price - price_range * 0.2:
                return 'skewed_low'
            else:
                return 'normal'
        else:
            return 'normal'

    async def _generate_trading_signals(self, symbol: str, profile: SessionProfile,
                                      breakouts: List[ProfileBreakout],
                                      timeframe: str) -> List[Dict[str, Union[str, float]]]:
        """Generate trading signals based on volume profile analysis"""
        signals = []
        
        # Breakout signals
        for breakout in breakouts:
            if breakout.confidence > 0.7:
                signal_direction = 'buy' if breakout.breakout_type in ['value_area_high', 'poc_break'] and breakout.breakout_strength > 0 else 'sell'
                
                signals.append({
                    'type': 'volume_profile_breakout',
                    'signal': signal_direction,
                    'confidence': breakout.confidence,
                    'breakout_level': breakout.breakout_level,
                    'target': breakout.target_projection,
                    'volume_confirmation': breakout.volume_confirmation,
                    'timeframe': timeframe,
                    'breakout_type': breakout.breakout_type
                })
        
        # Value area reversion signals
        if profile.value_area and profile.profile_type == 'balanced':
            signals.append({
                'type': 'value_area_reversion',
                'signal': 'range_trade',
                'confidence': 0.6,
                'va_high': profile.value_area.value_area_high,
                'va_low': profile.value_area.value_area_low,
                'poc': profile.value_area.point_of_control,
                'timeframe': timeframe,
                'strategy': 'buy_va_low_sell_va_high'
            })
        
        return signals

    def _generate_test_data(self) -> List[Tuple[Dict, Dict]]:
        """Generate test data for initialization"""
        test_data = []
        base_price = 1.1000
        
        for i in range(30):
            # Create price movement
            price_change = (np.random.random() - 0.5) * 0.002
            open_price = base_price + price_change
            high_price = open_price + np.random.uniform(0.0002, 0.0008)
            low_price = open_price - np.random.uniform(0.0002, 0.0008)
            close_price = low_price + (high_price - low_price) * np.random.random()
            
            # Create volume
            volume = np.random.uniform(800, 1500)
            
            price_data = {
                'timestamp': time.time() - (30 - i) * 900,  # M15 intervals
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            }
            
            volume_data = {
                'timestamp': time.time() - (30 - i) * 900,
                'volume': volume
            }
            
            test_data.append((price_data, volume_data))
            
        return test_data
