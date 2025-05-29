"""
Gann Price-Time Relationships Implementation
Advanced geometric analysis of price and time relationships using W.D. Gann's methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import math
from dataclasses import dataclass

@dataclass
class PriceTimeRelationship:
    """Represents a detected price-time relationship."""
    start_price: float
    end_price: float
    start_time: datetime
    end_time: datetime
    time_units: int
    price_change: float
    angle: float
    relationship_type: str
    strength: float

@dataclass
class GannSquareLevel:
    """Represents a level from Gann square calculations."""
    price: float
    time_target: datetime
    square_position: Tuple[int, int]
    significance: float
    level_type: str

class PriceTimeRelationships:
    """
    W.D. Gann Price-Time Relationships Analysis Engine
    
    Implements Gann's geometric price-time analysis including:
    - 45-degree angle calculations (1x1 line)
    - Geometric angle relationships (1x2, 2x1, 1x3, 3x1, etc.)
    - Square of price and time calculations
    - Natural resistance and support from geometric relationships
    - Time equals price analysis
    - Proportional relationships between price moves and time
    """
    
    def __init__(self, 
                 base_angles: List[float] = None,
                 time_units: str = 'days',
                 min_relationship_strength: float = 0.6,
                 max_lookback: int = 252):
        """
        Initialize Price-Time Relationships analyzer.
        
        Args:
            base_angles: List of base angles to analyze (default: Gann's primary angles)
            time_units: Time unit for calculations ('days', 'hours', 'minutes')
            min_relationship_strength: Minimum strength for relationship validation
            max_lookback: Maximum periods to look back for relationship analysis
        """
        # Gann's primary angles (degrees)
        self.base_angles = base_angles or [45, 26.25, 18.75, 15, 7.5, 3.75, 71.25, 75, 82.5, 86.25]
        self.time_units = time_units
        self.min_relationship_strength = min_relationship_strength
        self.max_lookback = max_lookback
        
        # Gann's geometric ratios
        self.gann_ratios = [1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2]
        
        # Price-time angle relationships
        self.angle_relationships = {
            45: (1, 1),    # 1x1 - Most important
            26.25: (1, 2), # 1x2
            18.75: (1, 3), # 1x3
            15: (1, 4),    # 1x4
            7.5: (1, 8),   # 1x8
            3.75: (1, 16), # 1x16
            71.25: (2, 1), # 2x1
            75: (3, 1),    # 3x1
            82.5: (4, 1),  # 4x1
            86.25: (8, 1)  # 8x1
        }
        
        # Detected relationships
        self.relationships: List[PriceTimeRelationship] = []
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict:
        """
        Calculate Price-Time relationships (alias for analyze method for compatibility)
        
        Args:
            data: OHLCV DataFrame with datetime index
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with price-time relationship analysis results
        """
        price_column = kwargs.get('price_column', 'close')
        volume_column = kwargs.get('volume_column', 'volume')
        return self.analyze(data, price_column, volume_column)
        
    def analyze(self, 
                data: pd.DataFrame,
                price_column: str = 'close',
                volume_column: str = 'volume') -> Dict:
        """
        Comprehensive Price-Time relationships analysis.
        
        Args:
            data: OHLCV DataFrame with datetime index
            price_column: Column name for price analysis
            volume_column: Column name for volume analysis
            
        Returns:
            Dictionary containing price-time analysis results
        """
        if len(data) < 20:
            return self._empty_result()
        
        results = {
            'timestamp': data.index[-1],
            'current_price': data[price_column].iloc[-1],
            'gann_angles': {},
            'price_time_squares': {},
            'geometric_relationships': {},
            'time_price_equality': {},
            'support_resistance_levels': [],
            'future_projections': [],
            'angle_intersections': [],
            'square_levels': [],
            'relationship_strength': 0.0,
            'dominant_angle': None
        }
        
        # Analyze Gann angles
        results['gann_angles'] = self._analyze_gann_angles(data, price_column)
        
        # Calculate price-time squares
        results['price_time_squares'] = self._calculate_price_time_squares(data, price_column)
        
        # Analyze geometric relationships
        results['geometric_relationships'] = self._analyze_geometric_relationships(data, price_column)
        
        # Check time equals price conditions
        results['time_price_equality'] = self._analyze_time_price_equality(data, price_column)
        
        # Calculate support/resistance levels
        results['support_resistance_levels'] = self._calculate_support_resistance_levels(data, price_column)
        
        # Project future levels
        results['future_projections'] = self._project_future_levels(data, price_column)
        
        # Find angle intersections
        results['angle_intersections'] = self._find_angle_intersections(data, price_column)
        
        # Calculate square-based levels
        results['square_levels'] = self._calculate_square_levels(data, price_column)
        
        # Calculate overall relationship strength
        results['relationship_strength'] = self._calculate_relationship_strength()
        
        # Determine dominant angle
        results['dominant_angle'] = self._find_dominant_angle(results['gann_angles'])
        
        return results
    
    def _analyze_gann_angles(self, data: pd.DataFrame, price_column: str) -> Dict:
        """Analyze Gann angle relationships."""
        gann_analysis = {}
        prices = data[price_column].values
        timestamps = data.index
        
        # Find significant swing points
        swing_highs, swing_lows = self._find_swing_points(prices)
        
        # Analyze angles from each significant point
        for angle in self.base_angles:
            price_ratio, time_ratio = self.angle_relationships[angle]
            
            angle_data = {
                'angle_degrees': angle,
                'price_time_ratio': f"{price_ratio}x{time_ratio}",
                'current_level': None,
                'trend_lines': [],
                'strength': 0.0,
                'support_levels': [],
                'resistance_levels': []
            }
            
            # Calculate angle lines from swing points
            angle_lines = self._calculate_angle_lines(
                swing_highs + swing_lows, 
                prices, 
                timestamps, 
                angle
            )
            
            angle_data['trend_lines'] = angle_lines
            
            # Calculate current level for this angle
            if angle_lines:
                current_level = self._get_current_angle_level(angle_lines, timestamps[-1], prices[-1])
                angle_data['current_level'] = current_level
            
            # Identify support/resistance from this angle
            support_levels, resistance_levels = self._identify_sr_from_angle(angle_lines, prices)
            angle_data['support_levels'] = support_levels
            angle_data['resistance_levels'] = resistance_levels
            
            # Calculate angle strength
            angle_data['strength'] = self._calculate_angle_strength(angle_lines, prices)
            
            gann_analysis[f'angle_{angle}'] = angle_data
        
        return gann_analysis
    
    def _calculate_price_time_squares(self, data: pd.DataFrame, price_column: str) -> Dict:
        """Calculate price-time square relationships."""
        squares_analysis = {}
        
        current_price = data[price_column].iloc[-1]
        current_time = data.index[-1]
        
        # Find significant price levels
        price_high = data[price_column].max()
        price_low = data[price_column].min()
        price_range = price_high - price_low
        
        # Calculate time span
        time_span = (data.index[-1] - data.index[0]).days
        
        # Analyze different square relationships
        square_sizes = [9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361]
        
        for square_size in square_sizes:
            if square_size > time_span:
                continue
            
            square_root = int(math.sqrt(square_size))
            
            # Calculate price unit for this square
            price_unit = price_range / square_root if square_root > 0 else 0
            
            # Calculate square levels
            square_levels = []
            for i in range(square_root + 1):
                level = price_low + (i * price_unit)
                square_levels.append(level)
            
            # Calculate time projections
            time_unit = timedelta(days=square_size / square_root) if square_root > 0 else timedelta(days=1)
            time_projections = []
            for i in range(square_root + 1):
                proj_time = current_time + (i * time_unit)
                time_projections.append(proj_time)
            
            squares_analysis[f'square_{square_size}'] = {
                'square_size': square_size,
                'square_root': square_root,
                'price_unit': price_unit,
                'time_unit_days': time_unit.days,
                'price_levels': square_levels,
                'time_projections': time_projections,
                'current_square_position': self._get_square_position(current_price, price_low, price_unit, square_root)
            }
        
        return squares_analysis
    
    def _analyze_geometric_relationships(self, data: pd.DataFrame, price_column: str) -> Dict:
        """Analyze geometric price-time relationships."""
        geometric_analysis = {}
        
        # Find major price swings
        major_swings = self._find_major_swings(data, price_column)
        
        for i, swing in enumerate(major_swings[:-1]):
            next_swing = major_swings[i + 1]
            
            # Calculate price and time relationships
            price_change = abs(next_swing['price'] - swing['price'])
            time_change = (next_swing['time'] - swing['time']).days
            
            if time_change == 0:
                continue
            
            # Calculate geometric ratios
            price_time_ratio = price_change / time_change if time_change > 0 else 0
            
            # Check against Gann ratios
            closest_gann_ratio = min(self.gann_ratios, key=lambda x: abs(x - price_time_ratio))
            ratio_deviation = abs(price_time_ratio - closest_gann_ratio)
            
            # Create relationship
            relationship = PriceTimeRelationship(
                start_price=swing['price'],
                end_price=next_swing['price'],
                start_time=swing['time'],
                end_time=next_swing['time'],
                time_units=time_change,
                price_change=price_change,
                angle=math.degrees(math.atan(price_time_ratio)) if price_time_ratio > 0 else 0,
                relationship_type=self._classify_relationship(price_time_ratio),
                strength=max(0, 1 - (ratio_deviation / closest_gann_ratio)) if closest_gann_ratio > 0 else 0
            )
            
            if relationship.strength >= self.min_relationship_strength:
                self.relationships.append(relationship)
        
        # Summarize geometric relationships
        geometric_analysis = {
            'total_relationships': len(self.relationships),
            'strong_relationships': len([r for r in self.relationships if r.strength >= 0.8]),
            'dominant_ratios': self._find_dominant_ratios(),
            'average_strength': np.mean([r.strength for r in self.relationships]) if self.relationships else 0,
            'relationship_types': self._count_relationship_types()
        }
        
        return geometric_analysis
    
    def _analyze_time_price_equality(self, data: pd.DataFrame, price_column: str) -> Dict:
        """Analyze where time equals price in various forms."""
        equality_analysis = {}
        
        current_price = data[price_column].iloc[-1]
        start_time = data.index[0]
        current_time = data.index[-1]
        time_elapsed = (current_time - start_time).days
        
        # Direct time equals price
        equality_analysis['direct_equality'] = {
            'time_days': time_elapsed,
            'current_price': current_price,
            'equality_ratio': current_price / time_elapsed if time_elapsed > 0 else 0,
            'is_equal': abs(current_price - time_elapsed) < (current_price * 0.01)  # Within 1%
        }
        
        # Time equals price range
        price_range = data[price_column].max() - data[price_column].min()
        equality_analysis['range_equality'] = {
            'price_range': price_range,
            'time_range': time_elapsed,
            'range_ratio': price_range / time_elapsed if time_elapsed > 0 else 0,
            'is_equal': abs(price_range - time_elapsed) < (price_range * 0.05)  # Within 5%
        }
        
        # Square root relationships
        price_sqrt = math.sqrt(current_price)
        time_sqrt = math.sqrt(time_elapsed)
        equality_analysis['sqrt_equality'] = {
            'price_sqrt': price_sqrt,
            'time_sqrt': time_sqrt,
            'sqrt_ratio': price_sqrt / time_sqrt if time_sqrt > 0 else 0,
            'is_equal': abs(price_sqrt - time_sqrt) < 1
        }
        
        return equality_analysis
    
    def _calculate_support_resistance_levels(self, data: pd.DataFrame, price_column: str) -> List[Dict]:
        """Calculate support and resistance levels from price-time relationships."""
        sr_levels = []
        current_price = data[price_column].iloc[-1]
        current_time = data.index[-1]
        
        # From angle relationships
        for relationship in self.relationships:
            if relationship.strength >= self.min_relationship_strength:
                # Project angle forward
                time_diff = (current_time - relationship.end_time).days
                price_projection = relationship.end_price + (
                    math.tan(math.radians(relationship.angle)) * time_diff
                )
                
                level_type = 'resistance' if price_projection > current_price else 'support'
                
                sr_levels.append({
                    'price': price_projection,
                    'type': level_type,
                    'strength': relationship.strength,
                    'source': f"angle_{relationship.angle:.1f}",
                    'distance_percent': abs(price_projection - current_price) / current_price
                })
        
        # From Gann ratios
        for ratio in self.gann_ratios:
            level_up = current_price * (1 + ratio)
            level_down = current_price * (1 - ratio)
            
            sr_levels.extend([
                {
                    'price': level_up,
                    'type': 'resistance',
                    'strength': self._calculate_ratio_strength(ratio),
                    'source': f"gann_ratio_{ratio}",
                    'distance_percent': ratio
                },
                {
                    'price': level_down,
                    'type': 'support',
                    'strength': self._calculate_ratio_strength(ratio),
                    'source': f"gann_ratio_{ratio}",
                    'distance_percent': ratio
                }
            ])
        
        # Sort by proximity to current price
        sr_levels.sort(key=lambda x: x['distance_percent'])
        
        return sr_levels[:20]  # Return top 20 levels
    
    def _project_future_levels(self, data: pd.DataFrame, price_column: str) -> List[Dict]:
        """Project future price levels based on price-time relationships."""
        projections = []
        current_price = data[price_column].iloc[-1]
        current_time = data.index[-1]
        
        # Project using strongest relationships
        strong_relationships = [r for r in self.relationships if r.strength >= 0.8]
        
        for relationship in strong_relationships[:5]:  # Top 5 strongest
            for days_ahead in [7, 14, 30, 60, 90]:
                future_time = current_time + timedelta(days=days_ahead)
                
                # Project using relationship angle
                time_diff = days_ahead
                price_projection = current_price + (
                    math.tan(math.radians(relationship.angle)) * time_diff
                )
                
                projections.append({
                    'date': future_time,
                    'days_ahead': days_ahead,
                    'projected_price': price_projection,
                    'relationship_angle': relationship.angle,
                    'confidence': relationship.strength,
                    'projection_type': 'angle_projection'
                })
        
        # Project using time cycles
        for cycle_days in [30, 60, 90, 144, 180, 365]:
            if cycle_days <= len(data):
                # Find price at cycle periods ago
                cycle_index = -min(cycle_days, len(data))
                cycle_price = data[price_column].iloc[cycle_index]
                cycle_time = data.index[cycle_index]
                
                # Calculate cycle relationship
                price_change = current_price - cycle_price
                time_change = (current_time - cycle_time).days
                
                if time_change > 0:
                    # Project forward
                    future_time = current_time + timedelta(days=cycle_days)
                    projected_change = price_change * (cycle_days / time_change)
                    projected_price = current_price + projected_change
                    
                    projections.append({
                        'date': future_time,
                        'days_ahead': cycle_days,
                        'projected_price': projected_price,
                        'cycle_period': cycle_days,
                        'confidence': 0.7,  # Moderate confidence for cycle projections
                        'projection_type': 'cycle_projection'
                    })
        
        # Sort by date
        projections.sort(key=lambda x: x['date'])
        
        return projections
    
    def _find_angle_intersections(self, data: pd.DataFrame, price_column: str) -> List[Dict]:
        """Find intersections between different Gann angles."""
        intersections = []
        current_time = data.index[-1]
        
        # Find swing points for angle calculations
        prices = data[price_column].values
        swing_highs, swing_lows = self._find_swing_points(prices)
        all_swings = swing_highs + swing_lows
        
        # Calculate angle lines from each swing point
        angle_lines = {}
        for angle in self.base_angles:
            angle_lines[angle] = self._calculate_angle_lines(
                all_swings, prices, data.index, angle
            )
        
        # Find intersections between angle lines
        for angle1 in self.base_angles:
            for angle2 in self.base_angles:
                if angle1 >= angle2:
                    continue
                
                lines1 = angle_lines[angle1]
                lines2 = angle_lines[angle2]
                
                for line1 in lines1:
                    for line2 in lines2:
                        intersection = self._calculate_line_intersection(line1, line2)
                        
                        if intersection and intersection['time'] > current_time:
                            # Future intersection
                            days_ahead = (intersection['time'] - current_time).days
                            
                            if days_ahead <= 90:  # Within next 90 days
                                intersections.append({
                                    'time': intersection['time'],
                                    'price': intersection['price'],
                                    'days_ahead': days_ahead,
                                    'angle1': angle1,
                                    'angle2': angle2,
                                    'significance': self._calculate_intersection_significance(angle1, angle2),
                                    'intersection_type': 'angle_intersection'
                                })
        
        # Sort by significance
        intersections.sort(key=lambda x: x['significance'], reverse=True)
        
        return intersections[:10]  # Return top 10 intersections
    
    def _calculate_square_levels(self, data: pd.DataFrame, price_column: str) -> List[GannSquareLevel]:
        """Calculate price levels based on Gann squares."""
        square_levels = []
        current_price = data[price_column].iloc[-1]
        current_time = data.index[-1]
        
        # Calculate square of current price
        price_square = int(math.sqrt(current_price)) ** 2
        next_square = (int(math.sqrt(current_price)) + 1) ** 2
        prev_square = max(1, int(math.sqrt(current_price)) - 1) ** 2
        
        # Calculate levels from price squares
        for square_price in [prev_square, price_square, next_square]:
            square_root = int(math.sqrt(square_price))
            
            # Calculate fractional levels within square
            for fraction in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
                level_price = square_price * fraction
                
                if level_price > 0:
                    # Calculate time target (square of time)
                    time_offset = square_root * fraction
                    time_target = current_time + timedelta(days=int(time_offset))
                    
                    significance = self._calculate_square_significance(square_price, fraction)
                    
                    square_level = GannSquareLevel(
                        price=level_price,
                        time_target=time_target,
                        square_position=(square_root, int(square_root * fraction)),
                        significance=significance,
                        level_type='square_level'
                    )
                    
                    square_levels.append(square_level)
        
        # Sort by significance
        square_levels.sort(key=lambda x: x.significance, reverse=True)
        
        return square_levels[:15]  # Return top 15 levels
    
    def _find_swing_points(self, prices: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """Find significant swing highs and lows."""
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(prices) - 2):
            # Check for swing high
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                swing_highs.append({'index': i, 'price': prices[i]})
            
            # Check for swing low
            elif (prices[i] < prices[i-1] and prices[i] < prices[i+1] and
                  prices[i] < prices[i-2] and prices[i] < prices[i+2]):
                swing_lows.append({'index': i, 'price': prices[i]})
        
        return swing_highs, swing_lows
    
    def _find_major_swings(self, data: pd.DataFrame, price_column: str) -> List[Dict]:
        """Find major price swings for relationship analysis."""
        prices = data[price_column].values
        timestamps = data.index
        
        # Find all swing points
        swing_highs, swing_lows = self._find_swing_points(prices)
        
        # Combine and sort by index
        all_swings = []
        for swing in swing_highs:
            all_swings.append({
                'index': swing['index'],
                'price': swing['price'],
                'time': timestamps[swing['index']],
                'type': 'high'
            })
        
        for swing in swing_lows:
            all_swings.append({
                'index': swing['index'],
                'price': swing['price'],
                'time': timestamps[swing['index']],
                'type': 'low'
            })
        
        all_swings.sort(key=lambda x: x['index'])
        
        # Filter for major swings (significant price moves)
        major_swings = []
        if all_swings:
            major_swings.append(all_swings[0])
            
            for swing in all_swings[1:]:
                last_swing = major_swings[-1]
                price_change_pct = abs(swing['price'] - last_swing['price']) / last_swing['price']
                
                if price_change_pct >= 0.05:  # At least 5% move
                    major_swings.append(swing)
        
        return major_swings
    
    def _calculate_angle_lines(self, swing_points: List[Dict], prices: np.ndarray, 
                             timestamps: pd.DatetimeIndex, angle: float) -> List[Dict]:
        """Calculate angle lines from swing points."""
        angle_lines = []
        
        for swing in swing_points:
            if swing['index'] < len(prices) - 10:  # Need some future data
                # Calculate angle slope
                slope = math.tan(math.radians(angle))
                
                # Create line equation: price = start_price + slope * time_diff
                line = {
                    'start_index': swing['index'],
                    'start_price': swing['price'],
                    'start_time': timestamps[swing['index']],
                    'slope': slope,
                    'angle': angle,
                    'swing_type': swing.get('type', 'unknown')
                }
                
                angle_lines.append(line)
        
        return angle_lines
    
    def _get_current_angle_level(self, angle_lines: List[Dict], current_time: datetime, 
                                current_price: float) -> Optional[float]:
        """Get current price level for angle lines."""
        if not angle_lines:
            return None
        
        # Find most recent angle line
        recent_line = max(angle_lines, key=lambda x: x['start_time'])
        
        # Calculate time difference in days
        time_diff = (current_time - recent_line['start_time']).days
        
        # Calculate price level
        price_level = recent_line['start_price'] + (recent_line['slope'] * time_diff)
        
        return price_level
    
    def _identify_sr_from_angle(self, angle_lines: List[Dict], prices: np.ndarray) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels from angle lines."""
        support_levels = []
        resistance_levels = []
        
        current_price = prices[-1]
        
        for line in angle_lines:
            # Calculate current level for this line
            if line['start_index'] < len(prices):
                time_elapsed = len(prices) - line['start_index']
                level = line['start_price'] + (line['slope'] * time_elapsed)
                
                if level < current_price:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)
        
        return support_levels, resistance_levels
    
    def _calculate_angle_strength(self, angle_lines: List[Dict], prices: np.ndarray) -> float:
        """Calculate strength of angle relationships."""
        if not angle_lines:
            return 0.0
        
        total_strength = 0.0
        valid_lines = 0
        
        for line in angle_lines:
            # Test how well the angle line predicts price movement
            start_idx = line['start_index']
            if start_idx >= len(prices) - 10:
                continue
            
            # Test prediction accuracy over next 10 periods
            test_length = min(10, len(prices) - start_idx - 1)
            prediction_errors = []
            
            for i in range(1, test_length + 1):
                predicted_price = line['start_price'] + (line['slope'] * i)
                actual_price = prices[start_idx + i]
                
                if predicted_price > 0:
                    error = abs(predicted_price - actual_price) / predicted_price
                    prediction_errors.append(error)
            
            if prediction_errors:
                avg_error = np.mean(prediction_errors)
                line_strength = max(0, 1 - avg_error)
                total_strength += line_strength
                valid_lines += 1
        
        return total_strength / valid_lines if valid_lines > 0 else 0.0
    
    def _get_square_position(self, price: float, base_price: float, 
                           price_unit: float, square_root: int) -> Tuple[int, int]:
        """Get position within Gann square."""
        if price_unit <= 0:
            return (0, 0)
        
        relative_price = price - base_price
        position = int(relative_price / price_unit)
        
        row = position // square_root
        col = position % square_root
        
        return (row, col)
    
    def _classify_relationship(self, price_time_ratio: float) -> str:
        """Classify price-time relationship type."""
        if price_time_ratio >= 2.0:
            return "steep_uptrend"
        elif price_time_ratio >= 1.0:
            return "uptrend"
        elif price_time_ratio >= 0.5:
            return "moderate_trend"
        elif price_time_ratio >= 0.25:
            return "sideways"
        else:
            return "ranging"
    
    def _find_dominant_ratios(self) -> List[Tuple[float, int]]:
        """Find dominant price-time ratios."""
        ratio_counts = {}
        
        for relationship in self.relationships:
            ratio = relationship.price_change / relationship.time_units if relationship.time_units > 0 else 0
            
            # Round to nearest Gann ratio
            closest_ratio = min(self.gann_ratios, key=lambda x: abs(x - ratio))
            
            if closest_ratio not in ratio_counts:
                ratio_counts[closest_ratio] = 0
            ratio_counts[closest_ratio] += 1
        
        # Sort by frequency
        sorted_ratios = sorted(ratio_counts.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_ratios[:5]  # Top 5 ratios
    
    def _count_relationship_types(self) -> Dict[str, int]:
        """Count different relationship types."""
        type_counts = {}
        
        for relationship in self.relationships:
            rel_type = relationship.relationship_type
            if rel_type not in type_counts:
                type_counts[rel_type] = 0
            type_counts[rel_type] += 1
        
        return type_counts
    
    def _calculate_ratio_strength(self, ratio: float) -> float:
        """Calculate strength of Gann ratio."""
        # Key Gann ratios have higher strength
        key_ratios = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        if ratio in key_ratios:
            return 0.9
        elif ratio in self.gann_ratios:
            return 0.7
        else:
            return 0.5
    
    def _calculate_line_intersection(self, line1: Dict, line2: Dict) -> Optional[Dict]:
        """Calculate intersection point between two angle lines."""
        # Both lines: price = start_price + slope * time_diff
        # Intersection when: start_price1 + slope1 * t = start_price2 + slope2 * t
        
        slope_diff = line1['slope'] - line2['slope']
        if abs(slope_diff) < 1e-10:  # Parallel lines
            return None
        
        # Calculate time difference where lines intersect
        price_diff = line2['start_price'] - line1['start_price']
        time_diff = price_diff / slope_diff
        
        # Calculate intersection price
        intersection_price = line1['start_price'] + (line1['slope'] * time_diff)
        
        # Calculate intersection time
        intersection_time = line1['start_time'] + timedelta(days=time_diff)
        
        return {
            'time': intersection_time,
            'price': intersection_price,
            'time_diff': time_diff
        }
    
    def _calculate_intersection_significance(self, angle1: float, angle2: float) -> float:
        """Calculate significance of angle intersection."""
        # 45-degree angle intersections are most significant
        significance = 0.5
        
        if 45 in [angle1, angle2]:
            significance += 0.3
        
        # Complementary angles (sum to 90) are significant
        if abs((angle1 + angle2) - 90) < 5:
            significance += 0.2
        
        return min(1.0, significance)
    
    def _calculate_square_significance(self, square_price: float, fraction: float) -> float:
        """Calculate significance of square level."""
        significance = 0.5
        
        # Key fractions are more significant
        key_fractions = [0.25, 0.5, 0.75, 1.0]
        if fraction in key_fractions:
            significance += 0.3
        
        # Perfect squares are more significant
        square_root = int(math.sqrt(square_price))
        if square_root ** 2 == square_price:
            significance += 0.2
        
        return min(1.0, significance)
    
    def _calculate_relationship_strength(self) -> float:
        """Calculate overall relationship strength."""
        if not self.relationships:
            return 0.0
        
        total_strength = sum(r.strength for r in self.relationships)
        return total_strength / len(self.relationships)
    
    def _find_dominant_angle(self, gann_angles: Dict) -> Optional[float]:
        """Find the dominant Gann angle."""
        if not gann_angles:
            return None
        
        best_angle = None
        best_strength = 0
        
        for angle_key, angle_data in gann_angles.items():
            if angle_data['strength'] > best_strength:
                best_strength = angle_data['strength']
                best_angle = angle_data['angle_degrees']
        
        return best_angle
    
    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'timestamp': None,
            'current_price': 0,
            'gann_angles': {},
            'price_time_squares': {},
            'geometric_relationships': {},
            'time_price_equality': {},
            'support_resistance_levels': [],
            'future_projections': [],
            'angle_intersections': [],
            'square_levels': [],
            'relationship_strength': 0.0,
            'dominant_angle': None
        }
    
    def get_analysis_summary(self, analysis_result: Dict) -> str:
        """Generate human-readable analysis summary."""
        if not analysis_result or analysis_result['relationship_strength'] == 0:
            return "No significant price-time relationships detected."
        
        summary_parts = []
        
        # Overall strength
        strength = analysis_result['relationship_strength']
        summary_parts.append(f"Relationship strength: {strength:.1%}")
        
        # Dominant angle
        dominant_angle = analysis_result['dominant_angle']
        if dominant_angle:
            summary_parts.append(f"Dominant angle: {dominant_angle}°")
        
        # Support/resistance levels
        sr_levels = analysis_result['support_resistance_levels']
        if sr_levels:
            nearest_support = min([l for l in sr_levels if l['type'] == 'support'], 
                                key=lambda x: x['distance_percent'], default=None)
            nearest_resistance = min([l for l in sr_levels if l['type'] == 'resistance'], 
                                   key=lambda x: x['distance_percent'], default=None)
            
            if nearest_support:
                summary_parts.append(f"Nearest support: {nearest_support['price']:.2f}")
            if nearest_resistance:
                summary_parts.append(f"Nearest resistance: {nearest_resistance['price']:.2f}")
        
        # Future projections
        projections = analysis_result['future_projections']
        if projections:
            summary_parts.append(f"Future projections: {len(projections)} levels")
        
        return " | ".join(summary_parts)

    def generate_signal(self, current_result, historical_results=None):
        """
        Generate trading signal based on Price-Time relationship analysis
        
        Args:
            current_result: Current calculation result
            historical_results: Historical results (optional)
            
        Returns:
            Trading signal based on price-time relationship analysis
        """
        if not current_result or 'relationships' not in current_result:
            return None
            
        from ..indicator_base import IndicatorSignal, SignalType
        
        relationships = current_result['relationships']
        geometric_analysis = current_result.get('geometric_analysis', {})
        
        if not relationships:
            return IndicatorSignal(
                signal_type=SignalType.NEUTRAL,
                strength=0.3,
                message="No active price-time relationships",
                timestamp=pd.Timestamp.now(),
                price_level=None,
                confidence=0.3
            )
        
        # Analyze relationship strengths and generate signal
        strong_relationships = [r for r in relationships if r.get('strength', 0) > 0.7]
        moderate_relationships = [r for r in relationships if 0.4 <= r.get('strength', 0) <= 0.7]
        
        # Determine signal based on relationship types and strengths
        if len(strong_relationships) >= 2:
            # Multiple strong relationships suggest significant level
            signal_type = SignalType.STRONG_BUY if geometric_analysis.get('trend_direction', 0) > 0 else SignalType.STRONG_SELL
            strength = 0.8
            message = f"Strong price-time convergence: {len(strong_relationships)} relationships"
        elif len(strong_relationships) == 1 or len(moderate_relationships) >= 2:
            signal_type = SignalType.BUY if geometric_analysis.get('trend_direction', 0) > 0 else SignalType.SELL
            strength = 0.6
            message = f"Price-time relationship detected: {len(strong_relationships + moderate_relationships)} levels"
        else:
            signal_type = SignalType.NEUTRAL
            strength = 0.4
            message = "Weak price-time relationships"
        
        return IndicatorSignal(
            signal_type=signal_type,
            strength=strength,
            message=message,
            timestamp=current_result.get('timestamp', pd.Timestamp.now()),
            price_level=current_result.get('current_price', None),
            confidence=strength
        )
