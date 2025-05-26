"""
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

@dataclass
class ValueArea:
    """Value area definition"""
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
