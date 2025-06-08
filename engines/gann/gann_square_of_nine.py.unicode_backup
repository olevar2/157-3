"""
GANN SQUARE OF NINE - Sacred Geometry Price and Time Calculator
Platform3 Advanced Gann Analysis Engine

This module implements W.D. Gann's famous Square of Nine methodology for calculating
price and time relationships using sacred geometric principles.

Features:
- Complete Square of Nine price calculations
- Time square calculations for turning point prediction
- Cardinal cross analysis (0°, 90°, 180°, 270°)
- Diagonal cross analysis (45°, 135°, 225°, 315°)
- Support and resistance level identification from squares
- Natural price progression calculations
- Harmonic price relationships
- Multi-level square analysis for complex patterns

Square of Nine Theory:
- Price squares represent natural support/resistance levels
- Cardinal crosses mark major turning points
- Diagonal crosses indicate secondary reversal levels
- Natural number progression follows market rhythm
- Time squares predict temporal reversal points

Trading Applications:
- Precise support/resistance identification
- Price target calculation
- Time-based reversal prediction
- Natural market rhythm analysis
- Entry/exit optimization
- Risk management level setting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import math
from datetime import datetime, timedelta
from ..indicator_base import IndicatorBase

class SquareLevel(Enum):
    """Square of Nine level types"""
    CARDINAL_CROSS = "cardinal_cross"      # 0°, 90°, 180°, 270°
    DIAGONAL_CROSS = "diagonal_cross"      # 45°, 135°, 225°, 315°
    INTERMEDIATE = "intermediate"          # Other significant angles
    NATURAL_NUMBER = "natural_number"      # Perfect squares (1, 4, 9, 16, 25...)

class SquareDirection(Enum):
    """Direction around the square"""
    CLOCKWISE = "clockwise"
    COUNTERCLOCKWISE = "counterclockwise"

@dataclass
class SquarePoint:
    """Individual point on Square of Nine"""
    value: float
    level: int  # Which square level (1st, 2nd, 3rd, etc.)
    angle: float  # Angle in degrees (0-360)
    square_type: SquareLevel
    strength: float  # 0.0 to 1.0
    is_support: bool
    is_resistance: bool
    historical_touches: int

@dataclass
class TimeSquare:
    """Time-based Square of Nine calculation"""
    base_date: datetime
    square_periods: List[int]  # Time periods in days
    next_turn_dates: List[datetime]
    current_phase: float  # 0.0 to 1.0 in current square
    harmonic_resonance: float

@dataclass
class SquareAnalysis:
    """Complete Square of Nine analysis"""
    price_squares: List[SquarePoint]
    time_squares: TimeSquare
    current_square_level: int
    dominant_angle: float
    support_levels: List[float]
    resistance_levels: List[float]
    price_targets: Dict[str, float]
    natural_progression: List[float]

class GannSquareOfNine(IndicatorBase):
    """
    Advanced Gann Square of Nine Calculator
    
    Implements W.D. Gann's Square of Nine methodology for identifying natural
    price and time relationships using sacred geometric principles.    """
    
    def __init__(self, 
                 square_levels: int = 5,
                 min_price_move: float = 0.001,
                 time_analysis: bool = True,
                 harmonic_analysis: bool = True):
        """
        Initialize Square of Nine calculator
        
        Args:
            square_levels: Number of square levels to calculate (default 5)
            min_price_move: Minimum price movement to consider (0.1%)
            time_analysis: Enable time-based square calculations
            harmonic_analysis: Enable harmonic relationship analysis
        """
        from ..indicator_base import IndicatorType, TimeFrame
        
        super().__init__(
            name="GannSquareOfNine",
            indicator_type=IndicatorType.GANN,
            timeframe=TimeFrame.H1,
            lookback_periods=50,
            parameters={
                'square_levels': square_levels,
                'min_price_move': min_price_move,
                'time_analysis': time_analysis,
                'harmonic_analysis': harmonic_analysis
            }
        )
        
        self.square_levels = square_levels
        self.min_price_move = min_price_move
        self.time_analysis = time_analysis
        self.harmonic_analysis = harmonic_analysis
        
        # Current analysis results
        self.current_analysis = None
        
        # Cardinal and diagonal angles (in degrees)
        self.cardinal_angles = [0, 90, 180, 270]
        self.diagonal_angles = [45, 135, 225, 315]
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, 
                  data: pd.DataFrame, 
                  base_price: Optional[float] = None,
                  base_date: Optional[datetime] = None,
                  **kwargs) -> Dict:
        """
        Calculate Square of Nine analysis
        
        Args:
            data: Price data DataFrame with OHLC columns
            base_price: Base price for square calculation (default: significant low)
            base_date: Base date for time squares (default: data start)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with Square of Nine analysis results
        """
        try:
            if len(data) < 10:
                raise ValueError("Insufficient data for Square of Nine analysis")
            
            # Determine base price
            if base_price is None:
                base_price = self._find_significant_base_price(data)
            
            # Determine base date for time analysis
            if base_date is None and self.time_analysis:
                base_date = data.index[0] if hasattr(data.index, 'to_pydatetime') else datetime.now()
            
            # Calculate price squares
            price_squares = self._calculate_price_squares(base_price, data.iloc[-1]['close'])
            
            # Analyze price interaction with squares
            self._analyze_price_interactions(price_squares, data)
            
            # Calculate time squares if enabled
            time_squares = None
            if self.time_analysis and base_date:
                time_squares = self._calculate_time_squares(base_date, data)
            
            # Identify support and resistance levels
            support_levels, resistance_levels = self._identify_square_levels(price_squares, data.iloc[-1]['close'])
            
            # Calculate price targets
            price_targets = self._calculate_square_targets(price_squares, data.iloc[-1]['close'])
            
            # Generate natural progression
            natural_progression = self._calculate_natural_progression(base_price, data.iloc[-1]['close'])
            
            # Determine current square level and angle
            current_square_level, dominant_angle = self._analyze_current_position(
                base_price, data.iloc[-1]['close'])
            
            # Create analysis result
            self.current_analysis = SquareAnalysis(
                price_squares=price_squares,
                time_squares=time_squares,
                current_square_level=current_square_level,
                dominant_angle=dominant_angle,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                price_targets=price_targets,
                natural_progression=natural_progression
            )
            
            result = {
                'price_squares': [self._square_point_to_dict(sq) for sq in price_squares],
                'current_position': {
                    'square_level': current_square_level,
                    'angle': dominant_angle,
                    'base_price': base_price,
                    'current_price': data.iloc[-1]['close']
                },
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'price_targets': price_targets,
                'natural_progression': natural_progression,
                'total_squares': len(price_squares),
                'active_squares': len([sq for sq in price_squares if sq.strength > 0.3])
            }
            
            # Add time analysis if enabled
            if time_squares:
                result['time_squares'] = {
                    'base_date': base_date.isoformat() if base_date else None,
                    'square_periods': time_squares.square_periods,
                    'next_turn_dates': [dt.isoformat() for dt in time_squares.next_turn_dates],
                    'current_phase': time_squares.current_phase,
                    'harmonic_resonance': time_squares.harmonic_resonance
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Square of Nine: {e}")
            return self._get_default_result(f"Calculation error: {e}")
    
    def _find_significant_base_price(self, data: pd.DataFrame) -> float:
        """Find significant price level to use as base for square calculations"""
        try:
            # Look for significant lows in the data
            lows = data['low'].values
            
            # Find the lowest low in recent history
            lookback = min(50, len(data))
            recent_lows = lows[-lookback:]
            
            # Find lowest point
            min_low = np.min(recent_lows)
            
            # Check if this low has been significant (tested multiple times)
            low_touches = 0
            tolerance = min_low * 0.02  # 2% tolerance
            
            for low in lows:
                if abs(low - min_low) <= tolerance:
                    low_touches += 1
            
            # If significant low found, use it; otherwise use a clean number
            if low_touches >= 2:
                return min_low
            else:
                # Use nearest clean number (round to significant digits)
                return self._round_to_clean_number(min_low)
                
        except Exception as e:
            self.logger.error(f"Error finding base price: {e}")
            return data['low'].min()
    
    def _round_to_clean_number(self, price: float) -> float:
        """Round price to clean number suitable for square calculations"""
        # Determine the order of magnitude
        magnitude = 10 ** math.floor(math.log10(abs(price)))
        
        # Round to nearest clean number
        clean_multipliers = [1, 2, 5, 10]
        
        best_price = price
        min_distance = float('inf')
        
        for multiplier in clean_multipliers:
            candidate = magnitude * multiplier
            distance = abs(price - candidate)
            if distance < min_distance:
                min_distance = distance
                best_price = candidate
        
        return best_price
    
    def _calculate_price_squares(self, base_price: float, current_price: float) -> List[SquarePoint]:
        """Calculate price levels from Square of Nine methodology"""
        squares = []
        
        try:
            # Calculate multiple square levels around base price
            for level in range(1, self.square_levels + 1):
                # Calculate the square root relationships
                base_sqrt = math.sqrt(base_price)
                
                # Calculate cardinal cross points for this level
                for angle in self.cardinal_angles:
                    price = self._calculate_square_price(base_price, level, angle)
                    if price > 0:
                        squares.append(SquarePoint(
                            value=price,
                            level=level,
                            angle=angle,
                            square_type=SquareLevel.CARDINAL_CROSS,
                            strength=0.0,  # Will be calculated later
                            is_support=False,
                            is_resistance=False,
                            historical_touches=0
                        ))
                
                # Calculate diagonal cross points for this level
                for angle in self.diagonal_angles:
                    price = self._calculate_square_price(base_price, level, angle)
                    if price > 0:
                        squares.append(SquarePoint(
                            value=price,
                            level=level,
                            angle=angle,
                            square_type=SquareLevel.DIAGONAL_CROSS,
                            strength=0.0,  # Will be calculated later
                            is_support=False,
                            is_resistance=False,
                            historical_touches=0
                        ))
                
                # Calculate natural number progression
                natural_price = base_price * (level * level)  # Square progression
                if natural_price > 0:
                    squares.append(SquarePoint(
                        value=natural_price,
                        level=level,
                        angle=0.0,  # No specific angle for natural numbers
                        square_type=SquareLevel.NATURAL_NUMBER,
                        strength=0.0,
                        is_support=False,
                        is_resistance=False,
                        historical_touches=0
                    ))
        
        except Exception as e:
            self.logger.error(f"Error calculating price squares: {e}")
        
        return squares
    
    def _calculate_square_price(self, base_price: float, level: int, angle: float) -> float:
        """Calculate price at specific square level and angle"""
        try:
            base_sqrt = math.sqrt(base_price)
            
            # Gann's Square of Nine formula
            # Each level represents one complete revolution around the square
            angle_radians = math.radians(angle)
            
            # Calculate the new square root value
            new_sqrt = base_sqrt + (level * math.cos(angle_radians))
            
            # Ensure positive square root
            if new_sqrt <= 0:
                new_sqrt = base_sqrt + level  # Fallback calculation
            
            # Return squared value
            return new_sqrt * new_sqrt
            
        except Exception as e:
            self.logger.error(f"Error calculating square price: {e}")
            return base_price * (1 + level * 0.1)  # Fallback
    
    def _analyze_price_interactions(self, squares: List[SquarePoint], data: pd.DataFrame) -> None:
        """Analyze how historical prices interact with square levels"""
        
        for square in squares:
            touches = 0
            strength_score = 0.0
            
            tolerance = square.value * 0.01  # 1% tolerance for touches
            
            for _, row in data.iterrows():
                high = row['high']
                low = row['low']
                close = row['close']
                
                # Check if price touched this square level
                if low <= square.value + tolerance and high >= square.value - tolerance:
                    touches += 1
                    
                    # Calculate touch strength
                    if low <= square.value <= high:
                        # Direct touch
                        strength_score += 1.0
                    else:
                        # Near touch
                        distance = min(abs(high - square.value), abs(low - square.value))
                        proximity = max(0, 1.0 - (distance / tolerance))
                        strength_score += proximity
            
            # Update square properties
            square.historical_touches = touches
            square.strength = min(1.0, strength_score / 20.0)  # Normalize
            
            # Determine support/resistance nature
            current_price = data.iloc[-1]['close']
            if square.value < current_price and touches >= 2:
                square.is_support = True
            elif square.value > current_price and touches >= 2:
                square.is_resistance = True
    
    def _calculate_time_squares(self, base_date: datetime, data: pd.DataFrame) -> TimeSquare:
        """Calculate time-based Square of Nine analysis"""
        try:
            current_date = data.index[-1] if hasattr(data.index, 'to_pydatetime') else datetime.now()
            
            # Calculate square periods in days
            square_periods = []
            next_turn_dates = []
            
            # Natural time squares (Gann methodology)
            base_periods = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]  # Perfect squares
            
            for period in base_periods:
                square_periods.append(period)
                next_turn_dates.append(base_date + timedelta(days=period))
            
            # Calculate current phase
            days_elapsed = (current_date - base_date).days
            current_square = 1
            
            for i, period in enumerate(square_periods):
                if days_elapsed <= period:
                    current_square = i + 1
                    break
            
            phase = (days_elapsed % square_periods[current_square - 1]) / square_periods[current_square - 1]
            
            # Calculate harmonic resonance
            harmonic_resonance = self._calculate_harmonic_resonance(days_elapsed, square_periods)
            
            return TimeSquare(
                base_date=base_date,
                square_periods=square_periods,
                next_turn_dates=next_turn_dates,
                current_phase=phase,
                harmonic_resonance=harmonic_resonance
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating time squares: {e}")
            return TimeSquare(
                base_date=base_date,
                square_periods=[],
                next_turn_dates=[],
                current_phase=0.0,
                harmonic_resonance=0.0
            )
    
    def _calculate_harmonic_resonance(self, days_elapsed: int, periods: List[int]) -> float:
        """Calculate harmonic resonance between current time and square periods"""
        resonance_sum = 0.0
        count = 0
        
        for period in periods:
            if period > 0:
                # Calculate how close we are to this harmonic
                ratio = days_elapsed / period
                fractional_part = ratio - math.floor(ratio)
                
                # Resonance is strongest at 0, 0.25, 0.5, 0.75, 1.0
                resonance_points = [0, 0.25, 0.5, 0.75, 1.0]
                min_distance = min(abs(fractional_part - point) for point in resonance_points)
                
                resonance = 1.0 - (min_distance * 4)  # Scale to 0-1
                resonance_sum += max(0, resonance)
                count += 1
        
        return resonance_sum / count if count > 0 else 0.0
    
    def _identify_square_levels(self, 
                              squares: List[SquarePoint], 
                              current_price: float) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels from squares"""
        support_levels = []
        resistance_levels = []
        
        # Filter squares by strength and type
        strong_squares = [sq for sq in squares if sq.strength > 0.3]
        
        for square in strong_squares:
            if square.is_support:
                support_levels.append(square.value)
            elif square.is_resistance:
                resistance_levels.append(square.value)
        
        # Sort levels
        support_levels.sort(reverse=True)  # Highest support first
        resistance_levels.sort()  # Lowest resistance first
        
        return support_levels[:5], resistance_levels[:5]
    
    def _calculate_square_targets(self, 
                                squares: List[SquarePoint], 
                                current_price: float) -> Dict[str, float]:
        """Calculate price targets based on square levels"""
        targets = {}
        
        # Find strongest cardinal cross levels
        cardinal_squares = [sq for sq in squares if sq.square_type == SquareLevel.CARDINAL_CROSS and sq.strength > 0.4]
        cardinal_squares.sort(key=lambda x: x.strength, reverse=True)
        
        if cardinal_squares:
            targets['cardinal_primary'] = cardinal_squares[0].value
        
        # Find strongest diagonal cross levels
        diagonal_squares = [sq for sq in squares if sq.square_type == SquareLevel.DIAGONAL_CROSS and sq.strength > 0.4]
        diagonal_squares.sort(key=lambda x: x.strength, reverse=True)
        
        if diagonal_squares:
            targets['diagonal_primary'] = diagonal_squares[0].value
        
        # Find natural number targets
        natural_squares = [sq for sq in squares if sq.square_type == SquareLevel.NATURAL_NUMBER and sq.strength > 0.3]
        natural_squares.sort(key=lambda x: x.strength, reverse=True)
        
        if natural_squares:
            targets['natural_primary'] = natural_squares[0].value
        
        return targets
    
    def _calculate_natural_progression(self, base_price: float, current_price: float) -> List[float]:
        """Calculate natural price progression using square methodology"""
        progression = []
        
        try:
            base_sqrt = math.sqrt(base_price)
            
            # Calculate natural progression levels
            for i in range(1, 11):  # Next 10 levels
                new_sqrt = base_sqrt + i
                new_price = new_sqrt * new_sqrt
                progression.append(new_price)
        
        except Exception as e:
            self.logger.error(f"Error calculating natural progression: {e}")
        
        return progression
    
    def _analyze_current_position(self, base_price: float, current_price: float) -> Tuple[int, float]:
        """Analyze current position in square methodology"""
        try:
            base_sqrt = math.sqrt(base_price)
            current_sqrt = math.sqrt(current_price)
            
            # Determine which square level we're in
            sqrt_diff = current_sqrt - base_sqrt
            square_level = int(sqrt_diff) + 1
            
            # Calculate angle position (simplified)
            fractional_part = sqrt_diff - math.floor(sqrt_diff)
            angle = fractional_part * 360  # Convert to degrees
            
            return square_level, angle
            
        except Exception as e:
            self.logger.error(f"Error analyzing current position: {e}")
            return 1, 0.0
    
    def _square_point_to_dict(self, square: SquarePoint) -> Dict:
        """Convert SquarePoint to dictionary"""
        return {
            'value': square.value,
            'level': square.level,
            'angle': square.angle,
            'square_type': square.square_type.value,
            'strength': square.strength,
            'is_support': square.is_support,
            'is_resistance': square.is_resistance,
            'historical_touches': square.historical_touches
        }
    
    def _get_default_result(self, error_message: str = "") -> Dict:
        """Return default result structure"""
        return {
            'price_squares': [],
            'current_position': {
                'square_level': 1,
                'angle': 0.0,
                'base_price': 0.0,
                'current_price': 0.0
            },
            'support_levels': [],
            'resistance_levels': [],
            'price_targets': {},
            'natural_progression': [],
            'total_squares': 0,
            'active_squares': 0,
            'error': error_message
        }
    
    def get_signal(self, current_price: Optional[float] = None) -> Dict:
        """
        Get trading signal based on Square of Nine analysis
        
        Args:
            current_price: Current market price (optional)
            
        Returns:
            Dictionary with signal information
        """
        if not self.current_analysis:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'reason': 'No Square of Nine analysis available'}
        
        analysis = self.current_analysis
        
        # Use provided price or analysis current price
        if current_price is None:
            current_price = analysis.price_squares[0].value if analysis.price_squares else 0.0
        
        signal = 'NEUTRAL'
        strength = 0.0
        reason = 'No clear square signal'
        
        # Check proximity to cardinal cross levels (strongest signals)
        cardinal_squares = [sq for sq in analysis.price_squares 
                          if sq.square_type == SquareLevel.CARDINAL_CROSS and sq.strength > 0.5]
        
        for square in cardinal_squares:
            distance_pct = abs(current_price - square.value) / current_price
            
            if distance_pct < 0.02:  # Within 2% of cardinal level
                if square.is_support and current_price >= square.value:
                    signal = 'BUY'
                    strength = square.strength
                    reason = f'Near cardinal support at {square.value:.4f} ({square.angle}°)'
                    break
                elif square.is_resistance and current_price <= square.value:
                    signal = 'SELL'
                    strength = square.strength
                    reason = f'Near cardinal resistance at {square.value:.4f} ({square.angle}°)'
                    break
        
        # Check natural number progression
        if signal == 'NEUTRAL':
            for i, nat_price in enumerate(analysis.natural_progression):
                distance_pct = abs(current_price - nat_price) / current_price
                
                if distance_pct < 0.015:  # Within 1.5% of natural level
                    signal = 'BUY' if current_price < nat_price else 'SELL'
                    strength = 0.6
                    reason = f'Near natural square level {nat_price:.4f}'
                    break
        
        return {
            'signal': signal,
            'strength': strength,
            'reason': reason,
            'square_level': analysis.current_square_level,
            'dominant_angle': analysis.dominant_angle,
            'cardinal_levels': len([sq for sq in analysis.price_squares 
                                  if sq.square_type == SquareLevel.CARDINAL_CROSS])
        }

    def generate_signal(self, current_result, historical_results=None):
        """
        Generate trading signal based on Gann Square of 9 analysis
        
        Args:
            current_result: Current calculation result
            historical_results: Historical results (optional)
            
        Returns:
            Trading signal based on Square of 9 analysis
        """
        if not current_result or 'signal' not in current_result:
            return None
            
        signal_type = current_result['signal']
        strength = current_result.get('strength', 0.5)
        
        from ..indicator_base import IndicatorSignal, SignalType
        
        # Map signal types
        signal_mapping = {
            'buy': SignalType.BUY,
            'strong_buy': SignalType.STRONG_BUY,
            'sell': SignalType.SELL, 
            'strong_sell': SignalType.STRONG_SELL,
            'hold': SignalType.HOLD,
            'neutral': SignalType.NEUTRAL
        }
        
        signal_enum = signal_mapping.get(signal_type, SignalType.NEUTRAL)
        
        return IndicatorSignal(
            signal_type=signal_enum,
            strength=strength,
            message=current_result.get('reason', 'Gann Square of 9 analysis'),
            timestamp=current_result.get('timestamp', pd.Timestamp.now()),
            price_level=current_result.get('current_price', None),
            confidence=strength
        )
