#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gann Grid Indicator - Advanced Geometric Trading Tool
Platform3 Phase 3 - Enhanced Gann Analysis

The Gann Grid creates a matrix of support and resistance levels based on W.D. Gann's 
geometric principles and mathematical ratios. It's used for:
- Multi-dimensional support/resistance analysis
- Price and time coordinate system
- Geometric pattern recognition
- Sacred ratio calculations
- Natural market structure identification
"""

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

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
class GridLine:
    """Gann Grid line data structure"""
    line_type: str              # 'horizontal', 'vertical', 'diagonal'
    price_level: float          # Price coordinate
    time_coordinate: int        # Time coordinate (index)
    ratio: float               # Gann ratio (1, 2, 3, 4, 8, etc.)
    angle_degrees: float       # Line angle in degrees
    strength: float            # Line strength (0-1)
    touches: int              # Number of price touches
    last_touch: Optional[datetime]  # Last interaction time

@dataclass
class GridNode:
    """Gann Grid intersection node"""
    x_coordinate: int          # Time coordinate
    y_coordinate: float        # Price coordinate
    node_type: str            # 'cardinal', 'diagonal', 'standard'
    confluence_strength: float # Strength of confluence at this point
    horizontal_line: Optional[GridLine]
    vertical_line: Optional[GridLine]
    diagonal_lines: List[GridLine]

@dataclass
class GridZone:
    """Gann Grid zone between lines"""
    zone_id: str
    boundaries: Dict[str, float]  # {'top', 'bottom', 'left', 'right'}
    zone_strength: float
    price_range: Tuple[float, float]
    time_range: Tuple[int, int]
    zone_type: str            # 'support', 'resistance', 'channel', 'reversal'

class GannGridIndicator:
    """
    Advanced Gann Grid Indicator with Sacred Geometry
    
    Features:
    - Multi-dimensional grid construction
    - Cardinal and diagonal line systems
    - Sacred ratio integration (1, 2, 3, 4, 8 series)
    - Dynamic support/resistance matrices
    - Grid node confluence analysis
    - Time-price coordinate mapping
    - Natural number progression
    """
    
    def __init__(self, grid_size: int = 9, base_ratio: float = 1.0):
        """Initialize Gann Grid indicator with Platform3 framework"""
        self.logger = Platform3Logger(self.__class__.__name__)
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
        
        self.grid_size = max(3, grid_size)  # Minimum 3x3 grid
        self.base_ratio = base_ratio
        
        # Gann's sacred ratios
        self.gann_ratios = [1/8, 1/4, 1/3, 1/2, 1, 2, 3, 4, 8]
        
        # Grid line angles (degrees)
        self.cardinal_angles = [0, 90, 180, 270]  # Cardinal directions
        self.diagonal_angles = [45, 135, 225, 315]  # Diagonal directions
        self.gann_angles = [15, 30, 45, 60, 75]  # Additional Gann angles
        
        self.logger.info(f"Gann Grid initialized - Size: {self.grid_size}x{self.grid_size}, Base Ratio: {self.base_ratio}")
        
    async def calculate(self, data: Union[np.ndarray, pd.DataFrame], 
                       pivot_high: float, pivot_low: float,
                       pivot_time_idx: int, projection_periods: int = 100) -> Optional[Dict[str, Any]]:
        """
        Calculate Gann Grid analysis
        
        Args:
            data: Price data (OHLC DataFrame or close price array)
            pivot_high: Reference high price for grid construction
            pivot_low: Reference low price for grid construction
            pivot_time_idx: Time index of the pivot point
            projection_periods: Number of periods to project grid
            
        Returns:
            Dictionary containing grid analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.debug("Starting Gann Grid calculation")
            
            # Validate and prepare data
            price_data, high_data, low_data = self._prepare_data(data)
            if price_data is None:
                raise ServiceError("Invalid price data", "INVALID_DATA")
            
            # Validate pivot points
            if not self._validate_pivot_data(pivot_high, pivot_low, pivot_time_idx, len(price_data)):
                raise ServiceError("Invalid pivot data", "INVALID_PIVOT")
            
            # Calculate grid parameters
            grid_params = await self._calculate_grid_parameters(
                pivot_high, pivot_low, pivot_time_idx, projection_periods
            )
            
            # Construct grid lines
            grid_lines = await self._construct_grid_lines(
                price_data, high_data, low_data, grid_params
            )
            
            # Identify grid nodes (intersections)
            grid_nodes = await self._identify_grid_nodes(grid_lines, grid_params)
            
            # Analyze grid zones
            grid_zones = await self._analyze_grid_zones(grid_lines, grid_nodes, price_data)
            
            # Calculate current analysis
            current_analysis = await self._analyze_current_position(
                price_data, grid_lines, grid_nodes, grid_zones
            )
            
            # Compile results
            result = {
                'grid_lines': [self._grid_line_to_dict(line) for line in grid_lines],
                'grid_nodes': [self._grid_node_to_dict(node) for node in grid_nodes],
                'grid_zones': [self._grid_zone_to_dict(zone) for zone in grid_zones],
                'grid_parameters': grid_params,
                'current_analysis': current_analysis,
                'pivot_data': {
                    'high': pivot_high,
                    'low': pivot_low,
                    'time_index': pivot_time_idx,
                    'price_range': pivot_high - pivot_low
                },
                'calculation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Gann Grid calculation completed in {result['calculation_time']:.4f}s")
            return result
            
        except ServiceError as e:
            self.logger.error(f"Service error in Gann Grid calculation: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in Gann Grid calculation: {e}")
            self.error_system.handle_error(e, self.__class__.__name__)
            return None    
    async def _calculate_grid_parameters(self, pivot_high: float, pivot_low: float,
                                       pivot_time_idx: int, projection_periods: int) -> Dict[str, Any]:
        """Calculate grid construction parameters"""
        
        # Base grid measurements
        price_range = pivot_high - pivot_low
        time_range = projection_periods
        
        # Calculate grid spacing using sacred ratios
        price_unit = price_range / self.grid_size
        time_unit = time_range / self.grid_size
        
        # Price levels for horizontal lines
        price_levels = []
        for i in range(self.grid_size + 1):
            # Use Gann's square root progression
            level = pivot_low + (price_range * (i / self.grid_size))
            price_levels.append(level)
        
        # Add additional levels using sacred ratios
        for ratio in self.gann_ratios:
            additional_level = pivot_low + (price_range * ratio)
            if pivot_low <= additional_level <= pivot_high:
                price_levels.append(additional_level)
        
        # Remove duplicates and sort
        price_levels = sorted(list(set(price_levels)))
        
        # Time coordinates for vertical lines
        time_coordinates = []
        for i in range(self.grid_size + 1):
            coord = pivot_time_idx + int(time_range * (i / self.grid_size))
            time_coordinates.append(coord)
        
        return {
            'price_range': price_range,
            'time_range': time_range,
            'price_unit': price_unit,
            'time_unit': time_unit,
            'price_levels': price_levels,
            'time_coordinates': time_coordinates,
            'pivot_center': (pivot_time_idx, (pivot_high + pivot_low) / 2),
            'grid_origin': (pivot_time_idx, pivot_low)
        }
    
    async def _construct_grid_lines(self, price_data: np.ndarray, high_data: np.ndarray,
                                  low_data: np.ndarray, grid_params: Dict[str, Any]) -> List[GridLine]:
        """Construct all grid lines (horizontal, vertical, diagonal)"""
        
        grid_lines = []
        
        # Horizontal lines (price levels)
        for price_level in grid_params['price_levels']:
            touches = self._count_line_touches(price_data, high_data, low_data, price_level, 'horizontal')
            strength = self._calculate_line_strength(touches, 'horizontal')
            
            grid_line = GridLine(
                line_type='horizontal',
                price_level=price_level,
                time_coordinate=0,  # Spans all time
                ratio=self._find_gann_ratio(price_level, grid_params),
                angle_degrees=0.0,
                strength=strength,
                touches=touches,
                last_touch=None
            )
            grid_lines.append(grid_line)
        
        # Vertical lines (time levels)
        for time_coord in grid_params['time_coordinates']:
            if time_coord < len(price_data):
                price_at_time = price_data[time_coord]
                touches = 1  # Always touches once at the time coordinate
                strength = 0.5  # Base strength for time lines
                
                grid_line = GridLine(
                    line_type='vertical',
                    price_level=price_at_time,
                    time_coordinate=time_coord,
                    ratio=1.0,
                    angle_degrees=90.0,
                    strength=strength,
                    touches=touches,
                    last_touch=None
                )
                grid_lines.append(grid_line)
        
        # Diagonal lines (Gann angles)
        diagonal_lines = await self._construct_diagonal_lines(
            price_data, grid_params
        )
        grid_lines.extend(diagonal_lines)
        
        return grid_lines    
    async def _construct_diagonal_lines(self, price_data: np.ndarray, 
                                      grid_params: Dict[str, Any]) -> List[GridLine]:
        """Construct diagonal Gann angle lines"""
        
        diagonal_lines = []
        pivot_time, pivot_price = grid_params['pivot_center']
        price_range = grid_params['price_range']
        time_range = grid_params['time_range']
        
        # Calculate Gann angle slopes
        for angle in self.gann_angles:
            # Convert angle to slope (rise/run)
            angle_rad = math.radians(angle)
            slope = math.tan(angle_rad)
            
            # Adjust slope for price/time scaling
            price_time_ratio = price_range / time_range if time_range > 0 else 1
            adjusted_slope = slope * price_time_ratio
            
            # Create ascending diagonal line
            grid_line_up = GridLine(
                line_type='diagonal',
                price_level=pivot_price,  # Starting price
                time_coordinate=pivot_time,  # Starting time
                ratio=slope,
                angle_degrees=angle,
                strength=0.6,  # Base strength for diagonal lines
                touches=0,
                last_touch=None
            )
            diagonal_lines.append(grid_line_up)
            
            # Create descending diagonal line (negative angle)
            grid_line_down = GridLine(
                line_type='diagonal',
                price_level=pivot_price,
                time_coordinate=pivot_time,
                ratio=-slope,
                angle_degrees=360 - angle,  # Complementary angle
                strength=0.6,
                touches=0,
                last_touch=None
            )
            diagonal_lines.append(grid_line_down)
        
        return diagonal_lines
    
    async def _identify_grid_nodes(self, grid_lines: List[GridLine], 
                                 grid_params: Dict[str, Any]) -> List[GridNode]:
        """Identify intersection points (nodes) in the grid"""
        
        grid_nodes = []
        horizontal_lines = [line for line in grid_lines if line.line_type == 'horizontal']
        vertical_lines = [line for line in grid_lines if line.line_type == 'vertical']
        diagonal_lines = [line for line in grid_lines if line.line_type == 'diagonal']
        
        # Create nodes at horizontal-vertical intersections
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                # Calculate confluence strength
                confluence = (h_line.strength + v_line.strength) / 2
                
                # Find intersecting diagonal lines
                intersecting_diagonals = []
                for d_line in diagonal_lines:
                    if self._lines_intersect_at_node(h_line, v_line, d_line, grid_params):
                        intersecting_diagonals.append(d_line)
                        confluence += d_line.strength * 0.3  # Diagonal bonus
                
                # Determine node type
                node_type = 'standard'
                if len(intersecting_diagonals) >= 2:
                    node_type = 'diagonal'
                elif confluence > 0.8:
                    node_type = 'cardinal'
                
                grid_node = GridNode(
                    x_coordinate=v_line.time_coordinate,
                    y_coordinate=h_line.price_level,
                    node_type=node_type,
                    confluence_strength=min(1.0, confluence),
                    horizontal_line=h_line,
                    vertical_line=v_line,
                    diagonal_lines=intersecting_diagonals
                )
                grid_nodes.append(grid_node)
        
        return grid_nodes    
    async def _analyze_grid_zones(self, grid_lines: List[GridLine], grid_nodes: List[GridNode],
                                price_data: np.ndarray) -> List[GridZone]:
        """Analyze zones between grid lines"""
        
        grid_zones = []
        horizontal_lines = sorted([line for line in grid_lines if line.line_type == 'horizontal'],
                                key=lambda x: x.price_level)
        vertical_lines = sorted([line for line in grid_lines if line.line_type == 'vertical'],
                               key=lambda x: x.time_coordinate)
        
        # Create zones between adjacent horizontal and vertical lines
        for i in range(len(horizontal_lines) - 1):
            for j in range(len(vertical_lines) - 1):
                bottom_line = horizontal_lines[i]
                top_line = horizontal_lines[i + 1]
                left_line = vertical_lines[j]
                right_line = vertical_lines[j + 1]
                
                # Calculate zone boundaries
                boundaries = {
                    'top': top_line.price_level,
                    'bottom': bottom_line.price_level,
                    'left': left_line.time_coordinate,
                    'right': right_line.time_coordinate
                }
                
                # Calculate zone strength based on boundary line strengths
                zone_strength = (bottom_line.strength + top_line.strength + 
                               left_line.strength + right_line.strength) / 4
                
                # Determine zone type based on price action
                zone_type = self._classify_zone_type(boundaries, price_data)
                
                # Find nodes within this zone
                zone_nodes = [node for node in grid_nodes 
                             if (boundaries['left'] <= node.x_coordinate <= boundaries['right'] and
                                 boundaries['bottom'] <= node.y_coordinate <= boundaries['top'])]
                
                # Adjust zone strength based on contained nodes
                for node in zone_nodes:
                    zone_strength += node.confluence_strength * 0.1
                
                grid_zone = GridZone(
                    zone_id=f"zone_{i}_{j}",
                    boundaries=boundaries,
                    zone_strength=min(1.0, zone_strength),
                    price_range=(boundaries['bottom'], boundaries['top']),
                    time_range=(boundaries['left'], boundaries['right']),
                    zone_type=zone_type
                )
                grid_zones.append(grid_zone)
        
        return grid_zones
    
    async def _analyze_current_position(self, price_data: np.ndarray, grid_lines: List[GridLine],
                                      grid_nodes: List[GridNode], grid_zones: List[GridZone]) -> Dict[str, Any]:
        """Analyze current price position relative to the grid"""
        
        if len(price_data) == 0:
            return {}
        
        current_price = price_data[-1]
        current_time_idx = len(price_data) - 1
        
        # Find nearest grid lines
        nearest_support = None
        nearest_resistance = None
        
        horizontal_lines = [line for line in grid_lines if line.line_type == 'horizontal']
        for line in horizontal_lines:
            if line.price_level <= current_price:
                if nearest_support is None or line.price_level > nearest_support.price_level:
                    nearest_support = line
            elif line.price_level > current_price:
                if nearest_resistance is None or line.price_level < nearest_resistance.price_level:
                    nearest_resistance = line
        
        # Find current zone
        current_zone = None
        for zone in grid_zones:
            if (zone.boundaries['bottom'] <= current_price <= zone.boundaries['top'] and
                zone.boundaries['left'] <= current_time_idx <= zone.boundaries['right']):
                current_zone = zone
                break
        
        # Find nearest high-confluence nodes
        nearest_nodes = sorted(grid_nodes, 
                             key=lambda node: abs(node.y_coordinate - current_price))[:5]
        
        # Calculate grid position percentile
        all_price_levels = [line.price_level for line in horizontal_lines]
        if all_price_levels:
            min_price = min(all_price_levels)
            max_price = max(all_price_levels)
            position_percentile = (current_price - min_price) / (max_price - min_price) if max_price > min_price else 0.5
        else:
            position_percentile = 0.5
        
        # Generate trading signals based on grid position
        trading_signals = self._generate_grid_trading_signals(
            current_price, nearest_support, nearest_resistance, current_zone, nearest_nodes
        )
        
        return {
            'current_price': current_price,
            'current_time_index': current_time_idx,
            'nearest_support': self._grid_line_to_dict(nearest_support) if nearest_support else None,
            'nearest_resistance': self._grid_line_to_dict(nearest_resistance) if nearest_resistance else None,
            'current_zone': self._grid_zone_to_dict(current_zone) if current_zone else None,
            'nearest_nodes': [self._grid_node_to_dict(node) for node in nearest_nodes],
            'position_percentile': position_percentile,
            'trading_signals': trading_signals
        }    
    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare and validate input data"""
        try:
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    price_data = data['close'].values
                elif 'Close' in data.columns:
                    price_data = data['Close'].values
                else:
                    price_data = data.iloc[:, -1].values  # Use last column
                
                # Extract high and low if available
                high_data = data['high'].values if 'high' in data.columns else price_data
                low_data = data['low'].values if 'low' in data.columns else price_data
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    price_data = data
                    high_data = price_data
                    low_data = price_data
                else:
                    price_data = data[:, -1]  # Use last column as close
                    high_data = data[:, 1] if data.shape[1] > 1 else price_data  # High
                    low_data = data[:, 2] if data.shape[1] > 2 else price_data   # Low
            else:
                return None, None, None
            
            # Validate data
            if len(price_data) < 10:
                return None, None, None
            
            return price_data.astype(float), high_data.astype(float), low_data.astype(float)
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return None, None, None
    
    def _validate_pivot_data(self, pivot_high: float, pivot_low: float, 
                           pivot_time_idx: int, data_length: int) -> bool:
        """Validate pivot point data"""
        try:
            # Check for valid numeric values
            if not all(isinstance(x, (int, float)) for x in [pivot_high, pivot_low, pivot_time_idx]):
                return False
            
            # Check price relationship
            if pivot_high <= pivot_low:
                return False
            
            # Check time index bounds
            if pivot_time_idx < 0 or pivot_time_idx >= data_length:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _count_line_touches(self, price_data: np.ndarray, high_data: np.ndarray, 
                          low_data: np.ndarray, level: float, line_type: str) -> int:
        """Count how many times price touched a grid line"""
        try:
            touches = 0
            tolerance = (np.max(price_data) - np.min(price_data)) * 0.002  # 0.2% tolerance
            
            if line_type == 'horizontal':
                # Check high and low data for touches
                high_touches = np.sum(np.abs(high_data - level) <= tolerance)
                low_touches = np.sum(np.abs(low_data - level) <= tolerance)
                touches = int(high_touches + low_touches)
            
            return touches
            
        except Exception:
            return 0
    
    def _calculate_line_strength(self, touches: int, line_type: str) -> float:
        """Calculate the strength of a grid line based on interactions"""
        try:
            base_strength = 0.3
            
            if line_type == 'horizontal':
                # More touches = stronger line
                touch_bonus = min(0.6, touches * 0.1)
                return base_strength + touch_bonus
            elif line_type == 'vertical':
                return 0.4  # Time lines have moderate strength
            elif line_type == 'diagonal':
                return 0.5  # Gann angles have good strength
            
            return base_strength
            
        except Exception:
            return 0.3
    
    def _find_gann_ratio(self, price_level: float, grid_params: Dict[str, Any]) -> float:
        """Find the closest Gann ratio for a price level"""
        try:
            price_range = grid_params['price_range']
            pivot_low = grid_params['price_levels'][0]  # Assuming first is lowest
            
            if price_range <= 0:
                return 1.0
            
            # Calculate ratio relative to price range
            level_ratio = (price_level - pivot_low) / price_range
            
            # Find closest Gann ratio
            closest_ratio = min(self.gann_ratios, key=lambda x: abs(x - level_ratio))
            return closest_ratio
            
        except Exception:
            return 1.0    
    def _lines_intersect_at_node(self, h_line: GridLine, v_line: GridLine, 
                               d_line: GridLine, grid_params: Dict[str, Any]) -> bool:
        """Check if diagonal line intersects at the horizontal-vertical node"""
        try:
            if d_line.line_type != 'diagonal':
                return False
            
            # Calculate diagonal line equation: y = mx + b
            # where m is the slope (ratio) and b is y-intercept
            x_node = v_line.time_coordinate
            y_node = h_line.price_level
            
            # Calculate where diagonal line would be at this time coordinate
            time_diff = x_node - d_line.time_coordinate
            price_diff = time_diff * d_line.ratio
            diagonal_price_at_node = d_line.price_level + price_diff
            
            # Check if diagonal line passes close to the node
            tolerance = grid_params['price_range'] * 0.01  # 1% tolerance
            return abs(diagonal_price_at_node - y_node) <= tolerance
            
        except Exception:
            return False
    
    def _classify_zone_type(self, boundaries: Dict[str, float], price_data: np.ndarray) -> str:
        """Classify zone type based on price action within boundaries"""
        try:
            # Find price data within time boundaries
            left_idx = max(0, int(boundaries['left']))
            right_idx = min(len(price_data), int(boundaries['right']))
            
            if left_idx >= right_idx:
                return 'standard'
            
            zone_prices = price_data[left_idx:right_idx]
            zone_high = np.max(zone_prices)
            zone_low = np.min(zone_prices)
            
            top_level = boundaries['top']
            bottom_level = boundaries['bottom']
            
            # Classify based on where price spent most time
            upper_third = bottom_level + (top_level - bottom_level) * 2/3
            lower_third = bottom_level + (top_level - bottom_level) * 1/3
            
            upper_time = np.sum(zone_prices >= upper_third)
            lower_time = np.sum(zone_prices <= lower_third)
            
            if upper_time > len(zone_prices) * 0.6:
                return 'resistance'
            elif lower_time > len(zone_prices) * 0.6:
                return 'support'
            elif zone_high >= top_level * 0.95 and zone_low <= bottom_level * 1.05:
                return 'channel'
            else:
                return 'reversal'
                
        except Exception:
            return 'standard'
    
    def _generate_grid_trading_signals(self, current_price: float, nearest_support: Optional[GridLine],
                                     nearest_resistance: Optional[GridLine], current_zone: Optional[GridZone],
                                     nearest_nodes: List[GridNode]) -> Dict[str, Any]:
        """Generate trading signals based on grid analysis"""
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'support_levels': [],
                'resistance_levels': [],
                'key_nodes': [],
                'zone_analysis': {}
            }
            
            # Support/Resistance analysis
            if nearest_support:
                support_distance = (current_price - nearest_support.price_level) / current_price
                signals['support_levels'].append({
                    'level': nearest_support.price_level,
                    'strength': nearest_support.strength,
                    'distance_pct': support_distance * 100
                })
                
                # Generate buy signal if near strong support
                if support_distance < 0.02 and nearest_support.strength > 0.7:
                    signals['primary_signal'] = 'buy'
                    signals['signal_strength'] = nearest_support.strength
            
            if nearest_resistance:
                resistance_distance = (nearest_resistance.price_level - current_price) / current_price
                signals['resistance_levels'].append({
                    'level': nearest_resistance.price_level,
                    'strength': nearest_resistance.strength,
                    'distance_pct': resistance_distance * 100
                })
                
                # Generate sell signal if near strong resistance
                if resistance_distance < 0.02 and nearest_resistance.strength > 0.7:
                    signals['primary_signal'] = 'sell'
                    signals['signal_strength'] = nearest_resistance.strength
            
            # Node confluence analysis
            for node in nearest_nodes:
                if node.confluence_strength > 0.8:
                    node_distance = abs(node.y_coordinate - current_price) / current_price
                    if node_distance < 0.01:  # Very close to high-confluence node
                        signals['key_nodes'].append({
                            'price': node.y_coordinate,
                            'confluence': node.confluence_strength,
                            'type': node.node_type,
                            'distance_pct': node_distance * 100
                        })
            
            # Zone analysis
            if current_zone:
                zone_position = ((current_price - current_zone.boundaries['bottom']) / 
                               (current_zone.boundaries['top'] - current_zone.boundaries['bottom']))
                
                signals['zone_analysis'] = {
                    'zone_type': current_zone.zone_type,
                    'zone_strength': current_zone.zone_strength,
                    'position_in_zone': zone_position,
                    'zone_id': current_zone.zone_id
                }
                
                # Refine signals based on zone
                if current_zone.zone_type == 'support' and zone_position < 0.3:
                    signals['primary_signal'] = 'buy'
                    signals['signal_strength'] = max(signals['signal_strength'], current_zone.zone_strength)
                elif current_zone.zone_type == 'resistance' and zone_position > 0.7:
                    signals['primary_signal'] = 'sell'
                    signals['signal_strength'] = max(signals['signal_strength'], current_zone.zone_strength)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}    
    def _grid_line_to_dict(self, line: GridLine) -> Dict[str, Any]:
        """Convert GridLine object to dictionary"""
        try:
            return {
                'line_type': line.line_type,
                'price_level': float(line.price_level),
                'time_coordinate': int(line.time_coordinate),
                'ratio': float(line.ratio),
                'angle_degrees': float(line.angle_degrees),
                'strength': float(line.strength),
                'touches': int(line.touches),
                'last_touch': line.last_touch.isoformat() if line.last_touch else None
            }
        except Exception:
            return {}
    
    def _grid_node_to_dict(self, node: GridNode) -> Dict[str, Any]:
        """Convert GridNode object to dictionary"""
        try:
            return {
                'x_coordinate': int(node.x_coordinate),
                'y_coordinate': float(node.y_coordinate),
                'node_type': node.node_type,
                'confluence_strength': float(node.confluence_strength),
                'horizontal_line': self._grid_line_to_dict(node.horizontal_line) if node.horizontal_line else None,
                'vertical_line': self._grid_line_to_dict(node.vertical_line) if node.vertical_line else None,
                'diagonal_lines': [self._grid_line_to_dict(line) for line in node.diagonal_lines]
            }
        except Exception:
            return {}
    
    def _grid_zone_to_dict(self, zone: GridZone) -> Dict[str, Any]:
        """Convert GridZone object to dictionary"""
        try:
            return {
                'zone_id': zone.zone_id,
                'boundaries': {k: float(v) for k, v in zone.boundaries.items()},
                'zone_strength': float(zone.zone_strength),
                'price_range': [float(zone.price_range[0]), float(zone.price_range[1])],
                'time_range': [int(zone.time_range[0]), int(zone.time_range[1])],
                'zone_type': zone.zone_type
            }
        except Exception:
            return {}
    
    def calculate_grid_sync(self, data: Union[np.ndarray, pd.DataFrame], 
                          pivot_high: float, pivot_low: float,
                          pivot_time_idx: int, projection_periods: int = 100) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for grid calculation"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.calculate(data, pivot_high, pivot_low, pivot_time_idx, projection_periods)
            )
            loop.close()
            return result
        except Exception as e:
            self.logger.error(f"Error in synchronous grid calculation: {e}")
            return None
    
    def get_current_grid_status(self, data: Union[np.ndarray, pd.DataFrame], 
                              pivot_high: float, pivot_low: float,
                              pivot_time_idx: int) -> Dict[str, Any]:
        """Get current status relative to the Gann Grid"""
        try:
            result = self.calculate_grid_sync(data, pivot_high, pivot_low, pivot_time_idx, 50)
            if result and 'current_analysis' in result:
                return result['current_analysis']
            return {'status': 'error', 'message': 'Unable to calculate grid status'}
        except Exception as e:
            self.logger.error(f"Error getting current grid status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def find_next_grid_levels(self, data: Union[np.ndarray, pd.DataFrame], 
                            pivot_high: float, pivot_low: float,
                            pivot_time_idx: int) -> Dict[str, List[float]]:
        """Find next support and resistance levels from the grid"""
        try:
            result = self.calculate_grid_sync(data, pivot_high, pivot_low, pivot_time_idx, 50)
            if not result or 'current_analysis' not in result:
                return {'support_levels': [], 'resistance_levels': []}
            
            analysis = result['current_analysis']
            support_levels = [level['level'] for level in analysis.get('support_levels', [])]
            resistance_levels = [level['level'] for level in analysis.get('resistance_levels', [])]
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
        except Exception as e:
            self.logger.error(f"Error finding next grid levels: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def validate_grid_setup(self, data: Union[np.ndarray, pd.DataFrame], 
                          pivot_high: float, pivot_low: float,
                          pivot_time_idx: int) -> Dict[str, Any]:
        """Validate if the grid setup is valid for the given data"""
        try:
            price_data, _, _ = self._prepare_data(data)
            if price_data is None:
                return {'valid': False, 'reason': 'Invalid price data'}
            
            if not self._validate_pivot_data(pivot_high, pivot_low, pivot_time_idx, len(price_data)):
                return {'valid': False, 'reason': 'Invalid pivot data'}
            
            # Check if price range is reasonable
            price_range = pivot_high - pivot_low
            data_range = np.max(price_data) - np.min(price_data)
            
            if price_range < data_range * 0.1:
                return {'valid': False, 'reason': 'Price range too small for effective grid'}
            
            if price_range > data_range * 10:
                return {'valid': False, 'reason': 'Price range too large for data range'}
            
            return {
                'valid': True,
                'price_range': price_range,
                'data_range': data_range,
                'grid_coverage': price_range / data_range
            }
            
        except Exception as e:
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}


# Factory function for easy instantiation
def create_gann_grid_indicator(grid_size: int = 9, base_ratio: float = 1.0) -> GannGridIndicator:
    """
    Factory function to create a Gann Grid indicator instance
    
    Args:
        grid_size: Size of the grid (NxN), minimum 3
        base_ratio: Base ratio for grid calculations
        
    Returns:
        GannGridIndicator instance
    """
    return GannGridIndicator(grid_size=grid_size, base_ratio=base_ratio)


# Example usage and testing
if __name__ == "__main__":
    # This section is for testing purposes only
    import asyncio
    
    async def test_gann_grid():
        """Test the Gann Grid indicator"""
        try:
            # Create sample data
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
            data = pd.DataFrame({
                'date': dates,
                'close': prices,
                'high': prices * 1.02,
                'low': prices * 0.98
            })
            
            # Initialize indicator
            gann_grid = GannGridIndicator(grid_size=9, base_ratio=1.0)
            
            # Define pivot points
            pivot_high = np.max(prices)
            pivot_low = np.min(prices)
            pivot_time_idx = 50
            
            # Calculate grid
            result = await gann_grid.calculate(
                data=data,
                pivot_high=pivot_high,
                pivot_low=pivot_low,
                pivot_time_idx=pivot_time_idx,
                projection_periods=50
            )
            
            if result:
                print("Gann Grid calculation successful!")
                print(f"Grid lines: {len(result['grid_lines'])}")
                print(f"Grid nodes: {len(result['grid_nodes'])}")
                print(f"Grid zones: {len(result['grid_zones'])}")
                print(f"Current signal: {result['current_analysis']['trading_signals']['primary_signal']}")
            else:
                print("Gann Grid calculation failed!")
                
        except Exception as e:
            print(f"Test error: {e}")
    
    # Run test
    # asyncio.run(test_gann_grid())