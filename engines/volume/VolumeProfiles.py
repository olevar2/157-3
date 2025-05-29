"""
<<<<<<< HEAD
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

=======
Volume Profiles for Session-Based Analysis
Analyzes volume distribution across price levels for different trading sessions.

This module creates volume profiles to identify:
- High Volume Nodes (HVN) - areas of high trading activity
- Low Volume Nodes (LVN) - areas of low trading activity  
- Point of Control (POC) - price level with highest volume
- Value Area - price range containing 70% of volume
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Trading session types"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    OVERLAP_ASIAN_LONDON = "overlap_asian_london"
    ALL_SESSIONS = "all_sessions"

class VolumeNodeType(Enum):
    """Volume node classification"""
    HIGH_VOLUME_NODE = "hvn"
    LOW_VOLUME_NODE = "lvn"
    POINT_OF_CONTROL = "poc"
    VALUE_AREA_HIGH = "vah"
    VALUE_AREA_LOW = "val"

@dataclass
class VolumeNode:
    """Individual volume node"""
    price_level: float
    volume: float
    volume_percentage: float
    node_type: VolumeNodeType
    session: TradingSession
    significance: float  # 0-1 scale
>>>>>>> 5e659b3064c215382ffc9ef1f13510cbfdd547a7

@dataclass
class ValueArea:
    """Value area definition"""
<<<<<<< HEAD
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
=======
    high: float
    low: float
    volume_percentage: float
    session: TradingSession
    poc_price: float
    poc_volume: float

@dataclass
class VolumeProfile:
    """Complete volume profile for a session"""
    session: TradingSession
    start_time: datetime
    end_time: datetime
    price_levels: List[float]
    volumes: List[float]
    volume_nodes: List[VolumeNode]
    point_of_control: VolumeNode
    value_area: ValueArea
    high_volume_nodes: List[VolumeNode]
    low_volume_nodes: List[VolumeNode]
    total_volume: float
    price_range: Tuple[float, float]

@dataclass
class VolumeProfileAnalysisResult:
    """Complete volume profile analysis result"""
    symbol: str
    analysis_time: datetime
    session_profiles: Dict[TradingSession, VolumeProfile]
    current_session: TradingSession
    key_levels: List[float]
    support_levels: List[float]
    resistance_levels: List[float]
    trading_opportunities: List[str]
    session_comparison: Dict[str, float]
    recommendations: List[str]

class VolumeProfiles:
    """
    Volume Profile analyzer for session-based trading.
    
    Creates volume profiles for different trading sessions to identify:
    - Key support/resistance levels
    - High/low volume areas
    - Session-specific trading patterns
    - Value area shifts between sessions
    """
    
    def __init__(self, price_bins: int = 50, value_area_percentage: float = 0.70):
        """
        Initialize volume profile analyzer.
        
        Args:
            price_bins: Number of price bins for volume distribution
            value_area_percentage: Percentage of volume for value area (default 70%)
        """
        self.price_bins = price_bins
        self.value_area_percentage = value_area_percentage
        
    def analyze_volume_profiles(self, data: pd.DataFrame, symbol: str) -> VolumeProfileAnalysisResult:
        """
        Perform complete volume profile analysis.
        
        Args:
            data: OHLCV data with timestamp column
            symbol: Trading symbol
            
        Returns:
            VolumeProfileAnalysisResult with session-based analysis
        """
        try:
            # Validate input data
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Add session information
            data = self._add_session_info(data)
            
            # Create volume profiles for each session
            session_profiles = {}
            for session in TradingSession:
                if session == TradingSession.ALL_SESSIONS:
                    session_data = data
                else:
                    session_data = data[data['session'] == session.value]
                
                if len(session_data) > 0:
                    profile = self._create_volume_profile(session_data, session)
                    session_profiles[session] = profile
            
            # Determine current session
            current_session = self._determine_current_session()
            
            # Extract key levels from all profiles
            key_levels = self._extract_key_levels(session_profiles)
            support_levels = self._identify_support_levels(session_profiles)
            resistance_levels = self._identify_resistance_levels(session_profiles)
            
            # Generate trading opportunities
            trading_opportunities = self._identify_trading_opportunities(session_profiles, current_session)
            
            # Compare sessions
            session_comparison = self._compare_sessions(session_profiles)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(session_profiles, current_session, key_levels)
            
            return VolumeProfileAnalysisResult(
                symbol=symbol,
                analysis_time=datetime.now(),
                session_profiles=session_profiles,
                current_session=current_session,
                key_levels=key_levels,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trading_opportunities=trading_opportunities,
                session_comparison=session_comparison,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Volume profile analysis failed for {symbol}: {e}")
            raise
    
    def _add_session_info(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trading session information to data"""
        data = data.copy()
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Extract hour in UTC
        data['hour_utc'] = data['timestamp'].dt.hour
        
        # Define session hours (UTC)
        def classify_session(hour):
            if 0 <= hour < 8:  # Asian session (Tokyo)
                return TradingSession.ASIAN.value
            elif 8 <= hour < 16:  # London session
                if 13 <= hour < 16:  # London-NY overlap
                    return TradingSession.OVERLAP_LONDON_NY.value
                elif 8 <= hour < 9:  # Asian-London overlap
                    return TradingSession.OVERLAP_ASIAN_LONDON.value
                else:
                    return TradingSession.LONDON.value
            else:  # 16 <= hour < 24, New York session
                return TradingSession.NEW_YORK.value
        
        data['session'] = data['hour_utc'].apply(classify_session)
        return data
    
    def _create_volume_profile(self, session_data: pd.DataFrame, session: TradingSession) -> VolumeProfile:
        """Create volume profile for a trading session"""
        if len(session_data) == 0:
            return None
        
        # Calculate price range
        price_min = session_data['low'].min()
        price_max = session_data['high'].max()
        price_range = (price_min, price_max)
        
        # Create price bins
        price_levels = np.linspace(price_min, price_max, self.price_bins)
        volumes = np.zeros(self.price_bins)
        
        # Distribute volume across price levels
        for _, row in session_data.iterrows():
            # Simple volume distribution: assume uniform distribution within OHLC range
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']
            
            # Find bins that overlap with this bar's range
            overlapping_bins = np.where((price_levels >= bar_low) & (price_levels <= bar_high))[0]
            
            if len(overlapping_bins) > 0:
                volume_per_bin = bar_volume / len(overlapping_bins)
                volumes[overlapping_bins] += volume_per_bin
        
        # Calculate total volume
        total_volume = volumes.sum()
        
        # Create volume nodes
        volume_nodes = []
        for i, (price, volume) in enumerate(zip(price_levels, volumes)):
            volume_percentage = (volume / total_volume * 100) if total_volume > 0 else 0
            
            # Classify node type
            node_type = self._classify_volume_node(volume, volumes, i)
            significance = volume / np.max(volumes) if np.max(volumes) > 0 else 0
            
            node = VolumeNode(
                price_level=price,
                volume=volume,
                volume_percentage=volume_percentage,
                node_type=node_type,
                session=session,
                significance=significance
            )
            volume_nodes.append(node)
        
        # Find Point of Control (highest volume)
        poc_index = np.argmax(volumes)
        point_of_control = volume_nodes[poc_index]
        point_of_control.node_type = VolumeNodeType.POINT_OF_CONTROL
        
        # Calculate Value Area
        value_area = self._calculate_value_area(price_levels, volumes, poc_index, session)
        
        # Identify high and low volume nodes
        high_volume_nodes = [node for node in volume_nodes 
                           if node.significance > 0.7 and node.node_type != VolumeNodeType.POINT_OF_CONTROL]
        low_volume_nodes = [node for node in volume_nodes if node.significance < 0.3]
        
        # Session time range
        start_time = session_data['timestamp'].min()
        end_time = session_data['timestamp'].max()
        
        return VolumeProfile(
            session=session,
            start_time=start_time,
            end_time=end_time,
            price_levels=price_levels.tolist(),
            volumes=volumes.tolist(),
            volume_nodes=volume_nodes,
            point_of_control=point_of_control,
            value_area=value_area,
            high_volume_nodes=high_volume_nodes,
            low_volume_nodes=low_volume_nodes,
            total_volume=total_volume,
            price_range=price_range
        )
    
    def _classify_volume_node(self, volume: float, all_volumes: np.ndarray, index: int) -> VolumeNodeType:
        """Classify volume node type"""
        max_volume = np.max(all_volumes)
        volume_ratio = volume / max_volume if max_volume > 0 else 0
        
        if volume_ratio > 0.8:
            return VolumeNodeType.HIGH_VOLUME_NODE
        elif volume_ratio < 0.2:
            return VolumeNodeType.LOW_VOLUME_NODE
        else:
            return VolumeNodeType.HIGH_VOLUME_NODE if volume_ratio > 0.5 else VolumeNodeType.LOW_VOLUME_NODE
    
    def _calculate_value_area(self, price_levels: np.ndarray, volumes: np.ndarray, 
                            poc_index: int, session: TradingSession) -> ValueArea:
        """Calculate value area containing specified percentage of volume"""
        total_volume = np.sum(volumes)
        target_volume = total_volume * self.value_area_percentage
        
        # Start from POC and expand outward
        value_area_volume = volumes[poc_index]
        upper_index = poc_index
        lower_index = poc_index
        
        while value_area_volume < target_volume:
            # Determine which direction to expand
            upper_volume = volumes[upper_index + 1] if upper_index + 1 < len(volumes) else 0
            lower_volume = volumes[lower_index - 1] if lower_index - 1 >= 0 else 0
            
            if upper_volume >= lower_volume and upper_index + 1 < len(volumes):
                upper_index += 1
                value_area_volume += upper_volume
            elif lower_index - 1 >= 0:
                lower_index -= 1
                value_area_volume += lower_volume
            else:
                break
        
        value_area_high = price_levels[upper_index]
        value_area_low = price_levels[lower_index]
        poc_price = price_levels[poc_index]
        poc_volume = volumes[poc_index]
        
        return ValueArea(
            high=value_area_high,
            low=value_area_low,
            volume_percentage=self.value_area_percentage * 100,
            session=session,
            poc_price=poc_price,
            poc_volume=poc_volume
        )
    
    def _determine_current_session(self) -> TradingSession:
        """Determine current trading session based on UTC time"""
        current_hour = datetime.utcnow().hour
        
        if 0 <= current_hour < 8:
            return TradingSession.ASIAN
        elif 8 <= current_hour < 16:
            if 13 <= current_hour < 16:
                return TradingSession.OVERLAP_LONDON_NY
            elif 8 <= current_hour < 9:
                return TradingSession.OVERLAP_ASIAN_LONDON
            else:
                return TradingSession.LONDON
        else:
            return TradingSession.NEW_YORK
    
    def _extract_key_levels(self, session_profiles: Dict[TradingSession, VolumeProfile]) -> List[float]:
        """Extract key price levels from all session profiles"""
        key_levels = []
        
        for profile in session_profiles.values():
            if profile:
                # Add POC levels
                key_levels.append(profile.point_of_control.price_level)
                
                # Add value area boundaries
                key_levels.append(profile.value_area.high)
                key_levels.append(profile.value_area.low)
                
                # Add high volume nodes
                for node in profile.high_volume_nodes:
                    key_levels.append(node.price_level)
        
        # Remove duplicates and sort
        key_levels = sorted(list(set(key_levels)))
        return key_levels
    
    def _identify_support_levels(self, session_profiles: Dict[TradingSession, VolumeProfile]) -> List[float]:
        """Identify potential support levels"""
        support_levels = []
        
        for profile in session_profiles.values():
            if profile:
                # Value area low as support
                support_levels.append(profile.value_area.low)
                
                # High volume nodes below current price as support
                for node in profile.high_volume_nodes:
                    support_levels.append(node.price_level)
        
        return sorted(list(set(support_levels)))
    
    def _identify_resistance_levels(self, session_profiles: Dict[TradingSession, VolumeProfile]) -> List[float]:
        """Identify potential resistance levels"""
        resistance_levels = []
        
        for profile in session_profiles.values():
            if profile:
                # Value area high as resistance
                resistance_levels.append(profile.value_area.high)
                
                # High volume nodes above current price as resistance
                for node in profile.high_volume_nodes:
                    resistance_levels.append(node.price_level)
        
        return sorted(list(set(resistance_levels)))
    
    def _identify_trading_opportunities(self, session_profiles: Dict[TradingSession, VolumeProfile], 
                                     current_session: TradingSession) -> List[str]:
        """Identify trading opportunities based on volume profiles"""
        opportunities = []
        
        current_profile = session_profiles.get(current_session)
        if not current_profile:
            return opportunities
        
        # Look for low volume areas (potential breakout zones)
        for node in current_profile.low_volume_nodes:
            opportunities.append(f"Low volume area at {node.price_level:.5f} - potential breakout zone")
        
        # Look for value area boundaries
        opportunities.append(f"Value area: {current_profile.value_area.low:.5f} - {current_profile.value_area.high:.5f}")
        opportunities.append(f"POC at {current_profile.point_of_control.price_level:.5f} - key level")
        
        return opportunities
    
    def _compare_sessions(self, session_profiles: Dict[TradingSession, VolumeProfile]) -> Dict[str, float]:
        """Compare volume characteristics across sessions"""
        comparison = {}
        
        volumes = []
        pocs = []
        value_area_ranges = []
        
        for session, profile in session_profiles.items():
            if profile and session != TradingSession.ALL_SESSIONS:
                volumes.append(profile.total_volume)
                pocs.append(profile.point_of_control.price_level)
                value_area_ranges.append(profile.value_area.high - profile.value_area.low)
        
        if volumes:
            comparison['avg_volume'] = np.mean(volumes)
            comparison['volume_std'] = np.std(volumes)
            comparison['poc_range'] = max(pocs) - min(pocs) if pocs else 0
            comparison['avg_value_area_range'] = np.mean(value_area_ranges)
        
        return comparison
    
    def _generate_recommendations(self, session_profiles: Dict[TradingSession, VolumeProfile],
                                current_session: TradingSession, key_levels: List[float]) -> List[str]:
        """Generate trading recommendations based on volume profile analysis"""
        recommendations = []
        
        current_profile = session_profiles.get(current_session)
        if not current_profile:
            recommendations.append("Insufficient data for current session analysis")
            return recommendations
        
        # POC-based recommendations
        recommendations.append(f"Key level: POC at {current_profile.point_of_control.price_level:.5f}")
        recommendations.append(f"Value area: {current_profile.value_area.low:.5f} - {current_profile.value_area.high:.5f}")
        
        # Session-specific recommendations
        if current_session == TradingSession.ASIAN:
            recommendations.append("Asian session: Look for range-bound trading within value area")
        elif current_session == TradingSession.LONDON:
            recommendations.append("London session: Watch for breakouts from Asian range")
        elif current_session == TradingSession.NEW_YORK:
            recommendations.append("NY session: Look for trend continuation or reversal")
        elif current_session in [TradingSession.OVERLAP_LONDON_NY, TradingSession.OVERLAP_ASIAN_LONDON]:
            recommendations.append("Session overlap: Increased volatility expected")
        
        # Volume-based recommendations
        if len(current_profile.low_volume_nodes) > 3:
            recommendations.append("Multiple low volume areas - potential for quick moves through these levels")
        
        if len(current_profile.high_volume_nodes) > 2:
            recommendations.append("Strong volume support/resistance levels identified")
        
        return recommendations
>>>>>>> 5e659b3064c215382ffc9ef1f13510cbfdd547a7
