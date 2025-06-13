"""
Platform3 Gann Angles Indicator
==============================

Real implementation of Gann Angles indicator for advanced geometric market analysis.
Based on W.D. Gann's mathematical principles for price-time angle relationships.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

# Gann Constants
GANN_ANGLES = {
    "1x1": 45.0,
    "1x2": 26.565,
    "1x3": 18.435, 
    "1x4": 14.036,
    "1x8": 7.125,
    "2x1": 63.435,
    "3x1": 71.565,
    "4x1": 75.964,
    "8x1": 82.875
}


@dataclass
class GannLevel:
    """Represents a Gann analysis level"""
    
    price: float
    angle: float
    level_type: str  # 'support', 'resistance', 'angle_line'
    time_projection: Optional[int] = None
    strength: float = 1.0
    square_position: Optional[Tuple[int, int]] = None


@dataclass
class GannSwingPoint:
    """Represents a Gann swing point"""
    
    index: int
    price: float
    swing_type: str  # 'high' or 'low'
    time_coordinate: float
    price_coordinate: float


@dataclass 
class GannAnalysisResult:
    """Result structure for Gann analysis"""
    
    levels: List[GannLevel]
    current_price: float
    primary_trend: str  # 'up', 'down', 'sideways'
    support_level: Optional[float]
    resistance_level: Optional[float]
    next_time_target: Optional[int]
    signal: str  # 'buy', 'sell', 'hold'
    signal_strength: float
    geometric_pattern: str


class GannAngles:
    """Gann Angles Indicator - Calculate price-time angle relationships"""
    
    def __init__(self, swing_window: int = 20, price_unit: float = 1.0):
        self.swing_window = swing_window
        self.price_unit = price_unit
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: Union[pd.DataFrame, np.ndarray, Dict]) -> Optional[GannAnalysisResult]:
        """Calculate Gann Angles for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                closes = data["close"].values
                highs = data.get("high", closes).values if "high" in data.columns else closes
                lows = data.get("low", closes).values if "low" in data.columns else closes
            elif isinstance(data, dict):
                closes = np.array(data.get("close", []))
                highs = np.array(data.get("high", closes))  
                lows = np.array(data.get("low", closes))
            elif isinstance(data, np.ndarray):
                closes = data.flatten()
                highs = lows = closes
            else:
                return None
                
            if len(closes) < self.swing_window:
                return None
                
            # Find significant swing points
            swing_high_idx = np.argmax(highs[-self.swing_window:]) + len(highs) - self.swing_window
            swing_low_idx = np.argmin(lows[-self.swing_window:]) + len(lows) - self.swing_window
            
            swing_high = highs[swing_high_idx]
            swing_low = lows[swing_low_idx]
            current_price = closes[-1]
            
            # Calculate Gann angle levels
            levels = []
            price_range = swing_high - swing_low
            
            for angle_name, angle_degrees in GANN_ANGLES.items():
                # Calculate price level based on angle and time
                angle_radians = math.radians(angle_degrees)
                time_units = len(closes) - int(max(swing_high_idx, swing_low_idx))
                
                # Calculate projected price using Gann angle
                if swing_high_idx > swing_low_idx:  # Uptrend from low
                    projected_price = swing_low + (time_units * self.price_unit * math.tan(angle_radians))
                else:  # Downtrend from high  
                    projected_price = swing_high - (time_units * self.price_unit * math.tan(angle_radians))
                
                level_type = "support" if projected_price < current_price else "resistance"
                strength = self._calculate_angle_strength(angle_degrees, price_range)
                
                levels.append(GannLevel(
                    price=projected_price,
                    angle=angle_degrees,
                    level_type=level_type,
                    strength=strength
                ))
                
            # Determine trend and signals
            primary_trend = self._determine_trend(closes, swing_high, swing_low)
            support_level = min([level.price for level in levels if level.level_type == "support"], default=None)
            resistance_level = max([level.price for level in levels if level.level_type == "resistance"], default=None)
            
            signal, signal_strength = self._generate_angle_signal(current_price, levels, primary_trend)
            
            return GannAnalysisResult(
                levels=levels,
                current_price=current_price,
                primary_trend=primary_trend,
                support_level=support_level,
                resistance_level=resistance_level,
                next_time_target=None,
                signal=signal,
                signal_strength=signal_strength,
                geometric_pattern="gann_angles"
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Gann Angles: {e}")
            return None
            
    def _calculate_angle_strength(self, angle_degrees: float, price_range: float) -> float:
        """Calculate strength of Gann angle based on degree and range."""
        # 1x1 (45°) is strongest, others weighted by importance
        base_strength = 1.0
        if angle_degrees == 45.0:  # 1x1 line
            return base_strength        elif angle_degrees in [26.565, 63.435]:  # 1x2, 2x1
            return base_strength * 0.8
        elif angle_degrees in [18.435, 71.565]:  # 1x3, 3x1
            return base_strength * 0.6
        else:
            return base_strength * 0.4
            
    def _determine_trend(self, closes: np.ndarray, swing_high: float, swing_low: float) -> str:
        """Determine primary trend direction."""
        current_price = closes[-1]
        
        if current_price > swing_high * 0.95:
            return "up"
        elif current_price < swing_low * 1.05:
            return "down"
        else:
            return "sideways"
            
    def _generate_angle_signal(self, current_price: float, levels: List[GannLevel], trend: str) -> Tuple[str, float]:
        """Generate trading signal based on Gann angle analysis."""
        # Find closest support and resistance
        supports = [level for level in levels if level.level_type == "support" and level.price < current_price]
        resistances = [level for level in levels if level.level_type == "resistance" and level.price > current_price]
        
        if not supports and not resistances:
            return "hold", 0.3
            
        closest_support = max(supports, key=lambda x: x.price) if supports else None
        closest_resistance = min(resistances, key=lambda x: x.price) if resistances else None
        
        # Generate signal based on proximity to levels and trend
        signal_strength = 0.5
        
        if closest_support and abs(current_price - closest_support.price) / current_price < 0.02:
            if trend == "up":
                return "buy", min(0.9, 0.5 + closest_support.strength)
        elif closest_resistance and abs(current_price - closest_resistance.price) / current_price < 0.02:
            if trend == "down":
                return "sell", min(0.9, 0.5 + closest_resistance.strength)
                
        return "hold", signal_strength
