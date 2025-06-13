# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Volume Breakout Detector - Volume Surge Analysis
Platform3 - Humanitarian Trading System

The Volume Breakout Detector identifies significant volume surges that accompany
price breakouts, validating technical breakouts and distinguishing between sustainable
price movements and false breakouts based on volume analysis.

Key Features:
- Volume surge detection with adaptive thresholds
- Historical volatility-adjusted breakout analysis
- Volume-confirmed price breakouts
- False breakout identification
- Volume climax detection
- Exhaustion move recognition

Humanitarian Mission: Enhance breakout trading accuracy by filtering false signals,
improving entry timing and reducing losses from failed breakouts to maximize
capital efficiency for humanitarian trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from engines.indicator_base import IndicatorSignal, TechnicalIndicator, ServiceError
import logging

logger = logging.getLogger(__name__)

@dataclass
class VolumeBreakoutSignal(IndicatorSignal):
    """Volume breakout signal with detailed breakout analysis"""
    volume_surge_ratio: float = 0.0  # Volume relative to average
    breakout_direction: str = "none"  # "up", "down", "none"
    breakout_strength: float = 0.0  # 0-1 scale
    confirmation_level: float = 0.0  # 0-1 scale
    is_false_breakout: bool = False
    is_exhaustion_move: bool = False
    
    def __post_init__(self):
        """Auto-populate required base fields if missing"""
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = datetime.now()
        if not hasattr(self, 'indicator_name') or self.indicator_name is None:
            self.indicator_name = "VolumeBreakoutDetector"
        if not hasattr(self, 'signal_type') or self.signal_type is None:
            self.signal_type = "volume"


class VolumeBreakoutDetector(TechnicalIndicator):
    """
    VolumeBreakoutDetector analyzes volume and price data to identify breakouts,
    validate them with volume confirmation, and detect false breakouts
    """
    
    def __init__(self, config: dict = None, volume_threshold: float = 1.5, 
                 price_threshold_stdev: float = 1.0,
                 lookback_period: int = 20):
        """
        Initialize the VolumeBreakoutDetector with configurable parameters
        
        Parameters:
        -----------
        config: dict
            Configuration dictionary
        volume_threshold: float
            Multiplier for average volume to consider as surge
        price_threshold_stdev: float
            Number of standard deviations for price movement to be considered breakout
        lookback_period: int
            Period for calculating averages and standard deviations
        """
        super().__init__(config)
        self.logger.info(f"VolumeBreakoutDetector initialized with volume_threshold={volume_threshold}")
        self.volume_threshold = volume_threshold
        self.price_threshold_stdev = price_threshold_stdev
        self.lookback_period = lookback_period
    
    def detect_volume_surge(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect volume surges relative to recent average volume
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data with volume
            
        Returns:
        --------
        Dict[str, float]
            Volume surge metrics
        """
        try:
            if len(data) < self.lookback_period + 1:
                return {'surge_ratio': 0.0, 'is_surge': False}
                
            # Calculate average volume over lookback period (excluding current bar)
            avg_volume = data['volume'].iloc[-(self.lookback_period+1):-1].mean()
            current_volume = data['volume'].iloc[-1]
            
            if avg_volume <= 0:
                return {'surge_ratio': 0.0, 'is_surge': False}
                
            # Calculate ratio
            surge_ratio = current_volume / avg_volume
            
            # Determine if surge threshold is met
            is_surge = surge_ratio >= self.volume_threshold
            
            return {'surge_ratio': surge_ratio, 'is_surge': is_surge}
            
        except Exception as e:
            self.logger.error(f"Error detecting volume surge: {str(e)}")
            return {'surge_ratio': 0.0, 'is_surge': False}
    
    def detect_price_breakout(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect price breakouts based on volatility
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data with OHLC prices
            
        Returns:
        --------
        Dict[str, Any]
            Breakout detection metrics
        """
        try:
            if len(data) < self.lookback_period + 1:
                return {
                    'direction': 'none',
                    'strength': 0.0,
                    'is_breakout': False
                }
                
            # Calculate price range statistics
            lookback_data = data.iloc[-(self.lookback_period+1):-1]
            current_bar = data.iloc[-1]
            
            # Calculate the average range and standard deviation
            ranges = lookback_data['high'] - lookback_data['low']
            avg_range = ranges.mean()
            range_stdev = ranges.std()
            
            # Calculate key price levels
            prev_high = lookback_data['high'].max()
            prev_low = lookback_data['low'].min()
            
            # Check for breakout above previous high
            if current_bar['high'] > prev_high:
                # Calculate breakout strength based on how far price moved beyond previous high
                breakout_distance = (current_bar['high'] - prev_high) / avg_range
                
                return {
                    'direction': 'up',
                    'strength': min(1.0, breakout_distance / 3),  # Normalize to [0,1]
                    'is_breakout': breakout_distance >= self.price_threshold_stdev,
                    'level_broken': prev_high
                }
            
            # Check for breakout below previous low
            elif current_bar['low'] < prev_low:
                # Calculate breakout strength based on how far price moved beyond previous low
                breakout_distance = (prev_low - current_bar['low']) / avg_range
                
                return {
                    'direction': 'down',
                    'strength': min(1.0, breakout_distance / 3),  # Normalize to [0,1]
                    'is_breakout': breakout_distance >= self.price_threshold_stdev,
                    'level_broken': prev_low
                }
            
            # No breakout detected
            return {
                'direction': 'none',
                'strength': 0.0,
                'is_breakout': False
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting price breakout: {str(e)}")
            return {'direction': 'none', 'strength': 0.0, 'is_breakout': False}
    
    def check_false_breakout(self, data: pd.DataFrame, breakout_info: Dict[str, Any]) -> bool:
        """
        Check for signs that a breakout might be false (price failing to follow through)
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data
        breakout_info: Dict[str, Any]
            Breakout detection results
            
        Returns:
        --------
        bool
            True if likely a false breakout
        """
        try:
            if len(data) < 3 or breakout_info['direction'] == 'none':
                return False
                
            current_bar = data.iloc[-1]
            prev_bar = data.iloc[-2]
            
            # Criteria for false breakout:
            # 1. Price closes back inside the broken level on same bar (failure to hold)
            # 2. High volume on breakout, but price ends near open
            
            level_broken = breakout_info.get('level_broken', None)
            
            if level_broken is not None:
                # For upside breakout
                if breakout_info['direction'] == 'up':
                    # Check for close back below broken level
                    if current_bar['close'] < level_broken:
                        return True
                        
                    # Check for weak close far from high
                    bar_range = current_bar['high'] - current_bar['low']
                    if bar_range > 0:
                        close_position = (current_bar['close'] - current_bar['low']) / bar_range
                        if close_position < 0.3:  # Close in bottom 30% of range
                            return True
                
                # For downside breakout
                elif breakout_info['direction'] == 'down':
                    # Check for close back above broken level
                    if current_bar['close'] > level_broken:
                        return True
                        
                    # Check for weak close far from low
                    bar_range = current_bar['high'] - current_bar['low']
                    if bar_range > 0:
                        close_position = (current_bar['close'] - current_bar['low']) / bar_range
                        if close_position > 0.7:  # Close in top 30% of range
                            return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking for false breakout: {str(e)}")
            return False
    
    def detect_exhaustion_move(self, data: pd.DataFrame, volume_info: Dict[str, float]) -> bool:
        """
        Detect potential exhaustion moves (climactic volume with reversal)
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data
        volume_info: Dict[str, float]
            Volume detection results
            
        Returns:
        --------
        bool
            True if likely an exhaustion move
        """
        try:
            if len(data) < 3 or not volume_info['is_surge']:
                return False
                
            # Multiple criteria for exhaustion move:
            # 1. Extreme volume (3x+ average)
            # 2. Wide range bar
            # 3. Price reverses direction on same bar
            
            current_bar = data.iloc[-1]
            prev_bars = data.iloc[-4:-1]
            
            # Check for extreme volume
            if volume_info['surge_ratio'] < 3.0:
                return False
                
            # Check for wide range bar
            avg_range = (prev_bars['high'] - prev_bars['low']).mean()
            current_range = current_bar['high'] - current_bar['low']
            
            if current_range < avg_range * 1.5:
                return False
                
            # Check for price reversal patterns
            
            # For uptrend exhaustion (current bar is up, but closes in lower half)
            if current_bar['close'] > current_bar['open'] and prev_bars['close'].iloc[-1] > prev_bars['open'].iloc[-1]:
                if (current_bar['close'] - current_bar['low']) / (current_bar['high'] - current_bar['low']) < 0.5:
                    # Closes in bottom half of range despite up bar
                    return True
            
            # For downtrend exhaustion (current bar is down, but closes in upper half)
            if current_bar['close'] < current_bar['open'] and prev_bars['close'].iloc[-1] < prev_bars['open'].iloc[-1]:
                if (current_bar['close'] - current_bar['low']) / (current_bar['high'] - current_bar['low']) > 0.5:
                    # Closes in upper half of range despite down bar
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting exhaustion move: {str(e)}")
            return False
    
    def calculate(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> VolumeBreakoutSignal:
        """
        Calculate volume breakout metrics from market data
        
        Parameters:
        -----------
        data: Union[pd.DataFrame, Dict[str, Any]]
            Market data with OHLCV information
            
        Returns:
        --------
        VolumeBreakoutSignal
            Comprehensive volume breakout analysis
        """
        try:
            # Convert dict to DataFrame if necessary
            if isinstance(data, dict) and 'ohlcv' in data:
                data = data['ohlcv']
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Invalid input data format, expected DataFrame")
                
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns in data: {required_columns}")
                
            if len(data) < self.lookback_period + 1:
                raise ValueError(f"Insufficient data: need at least {self.lookback_period + 1} bars")
                
            # Detect volume surge
            volume_info = self.detect_volume_surge(data)
            
            # Detect price breakout
            breakout_info = self.detect_price_breakout(data)
            
            # Check for false breakout
            is_false = self.check_false_breakout(data, breakout_info)
            
            # Check for exhaustion move
            is_exhaustion = self.detect_exhaustion_move(data, volume_info)
            
            # Calculate confirmation level
            # Higher confirmation when volume surge ratio matches breakout strength
            confirmation = 0.0
            if breakout_info['is_breakout'] and volume_info['is_surge']:
                # Base confirmation on relative alignment of volume and price movement
                confirmation = min(1.0, (volume_info['surge_ratio'] / self.volume_threshold) * 
                                  (breakout_info['strength'] / 0.5))
                
                # Reduce confirmation if false breakout signals are present
                if is_false:
                    confirmation *= 0.3
                    
                # Reduce confirmation if exhaustion move signals are present
                if is_exhaustion:
                    confirmation *= 0.5
                    
            # Create signal with buy/sell recommendation
            signal_direction = "neutral"
            signal_strength = 0.0
            
            # Generate signals based on breakout direction and confirmation
            if breakout_info['is_breakout'] and not is_false and confirmation > 0.5:
                if breakout_info['direction'] == 'up':
                    signal_direction = "buy"
                    signal_strength = min(1.0, confirmation * breakout_info['strength'] * 1.5)
                elif breakout_info['direction'] == 'down':
                    signal_direction = "sell"
                    signal_strength = min(1.0, confirmation * breakout_info['strength'] * 1.5)
            
            # Create and return the volume breakout signal
            return VolumeBreakoutSignal(
                timestamp=data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now(),
                indicator_name="VolumeBreakoutDetector",
                signal_type="volume",
                strength=signal_strength,
                confidence=confirmation,
                volume_surge_ratio=volume_info['surge_ratio'],
                breakout_direction=breakout_info['direction'],
                breakout_strength=breakout_info['strength'],
                confirmation_level=confirmation,
                is_false_breakout=is_false,
                is_exhaustion_move=is_exhaustion,
                metadata={
                    "signal_direction": signal_direction,
                    "volume_threshold": self.volume_threshold,
                    "lookback_period": self.lookback_period
                }
            )
        
        except Exception as e:
            self.logger.error(f"Error in VolumeBreakoutDetector calculation: {str(e)}")
            raise ServiceError(f"Calculation failed: {str(e)}")
    
    def generate_signal(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate trading signals based on volume breakout analysis
        
        Parameters:
        -----------
        data: Union[pd.DataFrame, Dict[str, Any]]
            Market data for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Trading signal with direction, strength and analysis
        """
        signal = self.calculate(data)
        
        return {
            "direction": signal.signal,
            "strength": signal.strength,
            "timestamp": signal.timestamp,
            "metadata": {
                "volume_surge_ratio": signal.volume_surge_ratio,
                "breakout_direction": signal.breakout_direction,
                "breakout_strength": signal.breakout_strength,
                "confirmation_level": signal.confirmation_level,
                "is_false_breakout": signal.is_false_breakout,
                "is_exhaustion_move": signal.is_exhaustion_move
            }
        }