#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fibonacci Fan Indicator - Advanced Geometric Trading Tool
Platform3 Phase 3 - Enhanced Fibonacci Analysis

The Fibonacci Fan draws trend lines from a significant swing point using Fibonacci ratios.
It's used for:
- Dynamic support and resistance levels
- Trend line analysis
- Price action validation
- Entry and exit timing
- Trend strength assessment
"""

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import math
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FanLine:
    """Fibonacci Fan line data structure"""
    ratio: float                    # Fibonacci ratio (0.382, 0.5, 0.618, etc.)
    slope: float                   # Line slope
    intercept: float               # Y-intercept
    angle_degrees: float           # Angle in degrees
    start_point: Tuple[float, float]   # (time_index, price)
    end_point: Tuple[float, float]     # (time_index, price)
    support_touches: int           # Number of support touches
    resistance_touches: int        # Number of resistance touches
    strength: float                # Line strength (0-1)
    last_interaction: Optional[datetime]  # Last price interaction

@dataclass
class FanZone:
    """Fibonacci Fan zone between two lines"""
    upper_line: FanLine
    lower_line: FanLine
    zone_strength: float
    price_range: Tuple[float, float]
    zone_type: str                 # 'support_zone', 'resistance_zone', 'channel'
    breakout_probability: float

class FibonacciFanIndicator:
    """
    Advanced Fibonacci Fan Indicator with Dynamic Analysis
    
    Features:
    - Multiple Fibonacci ratios (23.6%, 38.2%, 50%, 61.8%, 78.6%)
    - Dynamic fan line calculation
    - Support/resistance strength analysis
    - Fan zone identification
    - Breakout probability assessment
    - Multi-timeframe consistency
    """
    
    def __init__(self, ratios: List[float] = None):
        """Initialize Fibonacci Fan indicator with Platform3 framework"""
        self.logger = Platform3Logger(self.__class__.__name__)
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
        
        # Standard Fibonacci ratios
        self.ratios = ratios or [0.236, 0.382, 0.500, 0.618, 0.786]
        
        self.logger.info(f"Fibonacci Fan initialized with ratios: {self.ratios}")        
    async def calculate(self, data: Union[np.ndarray, pd.DataFrame], 
                       swing_high_idx: int, swing_low_idx: int,
                       projection_periods: int = 50) -> Optional[Dict[str, Any]]:
        """
        Calculate Fibonacci Fan lines and analysis
        
        Args:
            data: Price data (OHLC DataFrame or close price array)
            swing_high_idx: Index of swing high point
            swing_low_idx: Index of swing low point
            projection_periods: Number of periods to project fan lines
            
        Returns:
            Dictionary containing fan analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.debug("Starting Fibonacci Fan calculation")
            
            # Validate and prepare data
            price_data, high_data, low_data = self._prepare_data(data)
            if price_data is None:
                raise ServiceError("Invalid price data", "INVALID_DATA")
            
            # Validate swing points
            if not self._validate_swing_points(swing_high_idx, swing_low_idx, len(price_data)):
                raise ServiceError("Invalid swing points", "INVALID_SWING_POINTS")
            
            # Calculate fan lines
            fan_lines = await self._calculate_fan_lines(
                price_data, high_data, low_data, 
                swing_high_idx, swing_low_idx, projection_periods
            )
            
            # Analyze line interactions
            line_analysis = await self._analyze_line_interactions(
                price_data, high_data, low_data, fan_lines
            )
            
            # Identify fan zones
            fan_zones = await self._identify_fan_zones(fan_lines, price_data)
            
            # Calculate current analysis
            current_analysis = await self._analyze_current_position(
                price_data, fan_lines, fan_zones
            )
            
            # Compile results
            result = {
                'fan_lines': [self._fan_line_to_dict(line) for line in fan_lines],
                'fan_zones': [self._fan_zone_to_dict(zone) for zone in fan_zones],
                'line_analysis': line_analysis,
                'current_analysis': current_analysis,
                'swing_points': {
                    'high_index': swing_high_idx,
                    'low_index': swing_low_idx,
                    'high_price': float(price_data[swing_high_idx]),
                    'low_price': float(price_data[swing_low_idx])
                },
                'projection_periods': projection_periods,
                'calculation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Fibonacci Fan calculation completed in {result['calculation_time']:.4f}s")
            return result
            
        except ServiceError as e:
            self.logger.error(f"Service error in Fibonacci Fan calculation: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in Fibonacci Fan calculation: {e}")
            self.error_system.handle_error(e, self.__class__.__name__)
            return None    
    async def _calculate_fan_lines(self, price_data: np.ndarray, high_data: np.ndarray, 
                                 low_data: np.ndarray, swing_high_idx: int, 
                                 swing_low_idx: int, projection_periods: int) -> List[FanLine]:
        """Calculate Fibonacci Fan lines from swing points"""
        
        fan_lines = []
        
        # Determine trend direction
        is_bullish = swing_low_idx < swing_high_idx
        
        if is_bullish:
            # Bullish fan: from low to high
            start_idx = swing_low_idx
            end_idx = swing_high_idx
            start_price = price_data[swing_low_idx]
            end_price = price_data[swing_high_idx]
        else:
            # Bearish fan: from high to low
            start_idx = swing_high_idx
            end_idx = swing_low_idx
            start_price = price_data[swing_high_idx]
            end_price = price_data[swing_low_idx]
        
        # Calculate base vector
        time_diff = end_idx - start_idx
        price_diff = end_price - start_price
        
        # Create fan lines for each Fibonacci ratio
        for ratio in self.ratios:
            # Calculate fan line slope
            adjusted_price_diff = price_diff * ratio
            slope = adjusted_price_diff / time_diff if time_diff != 0 else 0
            
            # Calculate intercept (y = mx + b, so b = y - mx)
            intercept = start_price - (slope * start_idx)
            
            # Calculate angle in degrees
            angle_radians = math.atan(slope) if slope != 0 else 0
            angle_degrees = math.degrees(angle_radians)
            
            # Calculate end point for projection
            projection_end_idx = end_idx + projection_periods
            projection_end_price = slope * projection_end_idx + intercept
            
            # Analyze line interactions with historical data
            support_touches, resistance_touches, strength = self._analyze_line_strength(
                price_data, high_data, low_data, slope, intercept, start_idx, end_idx
            )
            
            fan_line = FanLine(
                ratio=ratio,
                slope=slope,
                intercept=intercept,
                angle_degrees=angle_degrees,
                start_point=(start_idx, start_price),
                end_point=(projection_end_idx, projection_end_price),
                support_touches=support_touches,
                resistance_touches=resistance_touches,
                strength=strength,
                last_interaction=None
            )
            
            fan_lines.append(fan_line)
        
        return fan_lines    
    def _analyze_line_strength(self, price_data: np.ndarray, high_data: np.ndarray,
                              low_data: np.ndarray, slope: float, intercept: float,
                              start_idx: int, end_idx: int) -> Tuple[int, int, float]:
        """Analyze how well a line acts as support or resistance"""
        
        support_touches = 0
        resistance_touches = 0
        total_interactions = 0
        
        # Define tolerance for line interaction (percentage of average price)
        avg_price = np.mean(price_data[start_idx:end_idx+1])
        tolerance = avg_price * 0.002  # 0.2% tolerance
        
        for i in range(start_idx, min(end_idx + 1, len(price_data))):
            line_price = slope * i + intercept
            
            # Check for support (low near line, then price bounces up)
            if abs(low_data[i] - line_price) <= tolerance:
                # Look ahead to see if price bounced up
                bounce_periods = min(5, len(price_data) - i - 1)
                if bounce_periods > 0:
                    future_highs = high_data[i+1:i+1+bounce_periods]
                    if len(future_highs) > 0 and np.max(future_highs) > line_price + tolerance:
                        support_touches += 1
                        total_interactions += 1
            
            # Check for resistance (high near line, then price falls)
            elif abs(high_data[i] - line_price) <= tolerance:
                # Look ahead to see if price fell
                decline_periods = min(5, len(price_data) - i - 1)
                if decline_periods > 0:
                    future_lows = low_data[i+1:i+1+decline_periods]
                    if len(future_lows) > 0 and np.min(future_lows) < line_price - tolerance:
                        resistance_touches += 1
                        total_interactions += 1
        
        # Calculate line strength based on interactions
        if total_interactions == 0:
            strength = 0.1  # Minimal strength for untested lines
        else:
            # Strength increases with number of successful interactions
            strength = min(1.0, total_interactions / 10.0)  # Max strength at 10 interactions
        
        return support_touches, resistance_touches, strength
    
    async def _analyze_line_interactions(self, price_data: np.ndarray, high_data: np.ndarray,
                                       low_data: np.ndarray, fan_lines: List[FanLine]) -> Dict[str, Any]:
        """Analyze interactions between price and fan lines"""
        
        current_idx = len(price_data) - 1
        current_price = price_data[current_idx]
        
        # Find closest lines
        line_distances = []
        for line in fan_lines:
            line_price = line.slope * current_idx + line.intercept
            distance = abs(current_price - line_price)
            line_distances.append({
                'ratio': line.ratio,
                'distance': distance,
                'line_price': line_price,
                'is_above': current_price > line_price
            })
        
        # Sort by distance
        line_distances.sort(key=lambda x: x['distance'])
        closest_line = line_distances[0] if line_distances else None
        
        # Analyze overall fan interaction
        total_support_touches = sum(line.support_touches for line in fan_lines)
        total_resistance_touches = sum(line.resistance_touches for line in fan_lines)
        avg_strength = np.mean([line.strength for line in fan_lines]) if fan_lines else 0
        
        return {
            'closest_line': closest_line,
            'total_support_touches': total_support_touches,
            'total_resistance_touches': total_resistance_touches,
            'average_line_strength': float(avg_strength),
            'fan_reliability': min(1.0, (total_support_touches + total_resistance_touches) / 20.0)
        }    
    async def _identify_fan_zones(self, fan_lines: List[FanLine], price_data: np.ndarray) -> List[FanZone]:
        """Identify zones between fan lines that act as channels"""
        
        fan_zones = []
        current_idx = len(price_data) - 1
        
        # Sort fan lines by current price level
        current_line_prices = []
        for line in fan_lines:
            current_price = line.slope * current_idx + line.intercept
            current_line_prices.append((line, current_price))
        
        current_line_prices.sort(key=lambda x: x[1])
        
        # Create zones between adjacent lines
        for i in range(len(current_line_prices) - 1):
            lower_line, lower_price = current_line_prices[i]
            upper_line, upper_price = current_line_prices[i + 1]
            
            # Calculate zone strength (average of both lines)
            zone_strength = (lower_line.strength + upper_line.strength) / 2
            
            # Determine zone type based on line interactions
            if lower_line.support_touches > lower_line.resistance_touches and \
               upper_line.resistance_touches > upper_line.support_touches:
                zone_type = 'channel'
            elif lower_line.support_touches > 0 or upper_line.support_touches > 0:
                zone_type = 'support_zone'
            elif lower_line.resistance_touches > 0 or upper_line.resistance_touches > 0:
                zone_type = 'resistance_zone'
            else:
                zone_type = 'neutral_zone'
            
            # Calculate breakout probability
            total_touches = (lower_line.support_touches + lower_line.resistance_touches + 
                           upper_line.support_touches + upper_line.resistance_touches)
            breakout_probability = max(0.1, min(0.9, total_touches / 15.0))
            
            fan_zone = FanZone(
                upper_line=upper_line,
                lower_line=lower_line,
                zone_strength=zone_strength,
                price_range=(lower_price, upper_price),
                zone_type=zone_type,
                breakout_probability=breakout_probability
            )
            
            fan_zones.append(fan_zone)
        
        return fan_zones
    
    async def _analyze_current_position(self, price_data: np.ndarray, 
                                      fan_lines: List[FanLine], 
                                      fan_zones: List[FanZone]) -> Dict[str, Any]:
        """Analyze current price position relative to fan lines and zones"""
        
        current_idx = len(price_data) - 1
        current_price = price_data[current_idx]
        
        # Find current position relative to lines
        line_positions = []
        for line in fan_lines:
            line_price = line.slope * current_idx + line.intercept
            position = 'above' if current_price > line_price else 'below'
            distance_pct = abs(current_price - line_price) / current_price * 100
            
            line_positions.append({
                'ratio': line.ratio,
                'position': position,
                'distance_pct': distance_pct,
                'line_strength': line.strength
            })
        
        # Find current zone
        current_zone = None
        for zone in fan_zones:
            if zone.price_range[0] <= current_price <= zone.price_range[1]:
                current_zone = zone
                break
        
        # Generate trading signals
        signals = self._generate_trading_signals(current_price, line_positions, current_zone)
        
        return {
            'current_price': float(current_price),
            'line_positions': line_positions,
            'current_zone': self._fan_zone_to_dict(current_zone) if current_zone else None,
            'trading_signals': signals
        }    
    def _generate_trading_signals(self, current_price: float, line_positions: List[Dict], 
                                current_zone: Optional[FanZone]) -> Dict[str, Any]:
        """Generate trading signals based on fan analysis"""
        
        signals = {
            'primary_signal': 'neutral',
            'signal_strength': 0.0,
            'support_levels': [],
            'resistance_levels': [],
            'breakout_targets': []
        }
        
        # Identify nearby support and resistance levels
        for pos in line_positions:
            if pos['distance_pct'] < 2.0:  # Within 2% of current price
                if pos['position'] == 'below':
                    signals['support_levels'].append({
                        'ratio': pos['ratio'],
                        'distance_pct': pos['distance_pct'],
                        'strength': pos['line_strength']
                    })
                else:
                    signals['resistance_levels'].append({
                        'ratio': pos['ratio'],
                        'distance_pct': pos['distance_pct'],
                        'strength': pos['line_strength']
                    })
        
        # Generate primary signal
        if current_zone:
            if current_zone.zone_type == 'support_zone':
                signals['primary_signal'] = 'bullish_support'
                signals['signal_strength'] = current_zone.zone_strength
            elif current_zone.zone_type == 'resistance_zone':
                signals['primary_signal'] = 'bearish_resistance'
                signals['signal_strength'] = current_zone.zone_strength
            elif current_zone.zone_type == 'channel':
                signals['primary_signal'] = 'range_bound'
                signals['signal_strength'] = current_zone.zone_strength
        
        # Add breakout targets
        if signals['support_levels']:
            strongest_support = max(signals['support_levels'], key=lambda x: x['strength'])
            signals['breakout_targets'].append({
                'direction': 'bearish',
                'target_ratio': strongest_support['ratio'],
                'probability': strongest_support['strength']
            })
        
        if signals['resistance_levels']:
            strongest_resistance = max(signals['resistance_levels'], key=lambda x: x['strength'])
            signals['breakout_targets'].append({
                'direction': 'bullish',
                'target_ratio': strongest_resistance['ratio'],
                'probability': strongest_resistance['strength']
            })
        
        return signals
    
    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare and validate input data"""
        try:
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    price_data = data['close'].values
                    high_data = data['high'].values if 'high' in data.columns else price_data
                    low_data = data['low'].values if 'low' in data.columns else price_data
                else:
                    # Assume first column is price data
                    price_data = data.iloc[:, 0].values
                    high_data = low_data = price_data
            elif isinstance(data, np.ndarray):
                price_data = data
                high_data = low_data = price_data
            else:
                return None, None, None
            
            return price_data.astype(float), high_data.astype(float), low_data.astype(float)
            
        except Exception as e:
            self.logger.error(f"Data preparation error: {e}")
            return None, None, None    
    def _validate_swing_points(self, high_idx: int, low_idx: int, data_length: int) -> bool:
        """Validate swing point indices"""
        return (0 <= high_idx < data_length and 
                0 <= low_idx < data_length and 
                high_idx != low_idx)
    
    def _fan_line_to_dict(self, fan_line: FanLine) -> Dict[str, Any]:
        """Convert FanLine to dictionary for JSON serialization"""
        return {
            'ratio': fan_line.ratio,
            'slope': fan_line.slope,
            'intercept': fan_line.intercept,
            'angle_degrees': fan_line.angle_degrees,
            'start_point': fan_line.start_point,
            'end_point': fan_line.end_point,
            'support_touches': fan_line.support_touches,
            'resistance_touches': fan_line.resistance_touches,
            'strength': fan_line.strength,
            'last_interaction': fan_line.last_interaction.isoformat() if fan_line.last_interaction else None
        }
    
    def _fan_zone_to_dict(self, fan_zone: FanZone) -> Dict[str, Any]:
        """Convert FanZone to dictionary for JSON serialization"""
        return {
            'upper_line_ratio': fan_zone.upper_line.ratio,
            'lower_line_ratio': fan_zone.lower_line.ratio,
            'zone_strength': fan_zone.zone_strength,
            'price_range': fan_zone.price_range,
            'zone_type': fan_zone.zone_type,
            'breakout_probability': fan_zone.breakout_probability
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get Fibonacci Fan indicator parameters"""
        return {
            'indicator_name': 'Fibonacci Fan',
            'version': '1.0.0',
            'fibonacci_ratios': self.ratios,
            'features': [
                'Dynamic fan line calculation',
                'Support/resistance analysis',
                'Fan zone identification',
                'Trading signal generation',
                'Breakout probability assessment',
                'Multi-ratio analysis'
            ]
        }
    
    def calculate(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """
        Standard calculate method for BaseIndicator compatibility

        Args:
            data: Market data in dict format with high, low, close arrays
                 or list of OHLC dictionaries, or pandas DataFrame

        Returns:
            Dict containing Fibonacci Fan analysis
        """
        start_time = time.time()

        try:
            # Convert data to standard format
            if isinstance(data, pd.DataFrame):
                high_values = data['high'].values
                low_values = data['low'].values
                close_values = data['close'].values
            elif isinstance(data, dict):
                high_values = np.array(data.get('high', []))
                low_values = np.array(data.get('low', []))
                close_values = np.array(data.get('close', []))
            else:
                # Assume list of dicts
                high_values = np.array([d.get('high', 0) for d in data])
                low_values = np.array([d.get('low', 0) for d in data])
                close_values = np.array([d.get('close', 0) for d in data])

            if len(high_values) < 10 or len(low_values) < 10 or len(close_values) < 10:
                return {"error": "Insufficient data for Fibonacci Fan analysis (need at least 10 periods)"}

            # Find swing points
            swing_high_idx = np.argmax(high_values)
            swing_low_idx = np.argmin(low_values)

            # Determine trend direction
            is_bullish = swing_low_idx < swing_high_idx
            trend_direction = "bullish" if is_bullish else "bearish"

            if is_bullish:
                start_idx = swing_low_idx
                end_idx = swing_high_idx
                start_price = low_values[swing_low_idx]
                end_price = high_values[swing_high_idx]
            else:
                start_idx = swing_high_idx
                end_idx = swing_low_idx
                start_price = high_values[swing_high_idx]
                end_price = low_values[swing_low_idx]

            # Calculate base measurements
            time_diff = abs(end_idx - start_idx)
            price_diff = end_price - start_price
            current_price = close_values[-1]

            if time_diff == 0:
                return {"error": "Invalid swing points for fan calculation"}

            # Calculate fan lines
            fan_lines = []
            for ratio in self.ratios:
                # Calculate adjusted price difference
                adjusted_price_diff = price_diff * ratio
                slope = adjusted_price_diff / time_diff
                intercept = start_price - (slope * start_idx)
                
                # Calculate angle
                angle_degrees = math.degrees(math.atan(slope)) if slope != 0 else 0

                # Project end point
                end_point_idx = len(close_values) - 1
                end_point_price = slope * end_point_idx + intercept

                fan_line_info = {
                    "ratio": ratio,
                    "slope": round(slope, 6),
                    "intercept": round(intercept, 5),
                    "angle_degrees": round(angle_degrees, 2),
                    "start_point": [start_idx, round(start_price, 5)],
                    "end_point": [end_point_idx, round(end_point_price, 5)],
                    "current_level": round(slope * (len(close_values) - 1) + intercept, 5),
                    "distance_from_current": round(abs((slope * (len(close_values) - 1) + intercept) - current_price), 5)
                }
                fan_lines.append(fan_line_info)

            # Find closest fan line to current price
            closest_line = min(fan_lines, key=lambda x: x["distance_from_current"])
            
            # Performance tracking
            calculation_time = time.time() - start_time

            result = {
                "symbol": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "trend_direction": trend_direction,
                "swing_points": {
                    "high": [swing_high_idx, round(float(high_values[swing_high_idx]), 5)],
                    "low": [swing_low_idx, round(float(low_values[swing_low_idx]), 5)]
                },
                "current_price": round(current_price, 5),
                "fibonacci_ratios": self.ratios,
                "fan_lines": fan_lines,
                "closest_fan_line": closest_line,
                "price_difference": round(price_diff, 5),
                "time_difference": time_diff,
                "total_fan_lines": len(fan_lines),
                "calculation_time_ms": round(calculation_time * 1000, 2)
            }

            self.logger.info(f"Fibonacci Fan analysis calculated successfully in {calculation_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Fan analysis: {e}")
            return {"error": str(e)}

# Export for Platform3 integration
__all__ = ['FibonacciFanIndicator', 'FanLine', 'FanZone']