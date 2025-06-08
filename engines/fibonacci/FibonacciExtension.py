# -*- coding: utf-8 -*-
"""
Fibonacci Extension
Extension levels for price target identification and projection analysis.

This module provides comprehensive Fibonacci extension analysis including:
- Standard extension levels (127.2%, 161.8%, 261.8%)
- Custom extension calculations
- Price target identification
- Extension strength assessment

Expected Benefits:
- Precise price target calculations
- Enhanced projection accuracy
- Dynamic extension level analysis
- Mathematical precision in extensions
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ExtensionLevel:
    """Fibonacci extension level data"""
    level_percentage: float  # e.g., 1.618 for 161.8%
    price_level: float
    level_type: str  # 'target', 'resistance', 'support'
    strength: float  # 0-1 confidence
    distance_from_current: float
    probability: float  # Probability of reaching this level


@dataclass
class ExtensionTarget:
    """Extension price target"""
    target_price: float
    target_level: float  # Fibonacci percentage
    confidence: float
    time_estimate: Optional[datetime]
    risk_reward_ratio: float


@dataclass
class ExtensionResult:
    """Fibonacci extension analysis result"""
    symbol: str
    timestamp: datetime
    swing_points: List[Tuple[datetime, float]]  # A, B, C points
    current_price: float
    extension_direction: str  # 'bullish', 'bearish'
    extension_levels: List[ExtensionLevel]
    price_targets: List[ExtensionTarget]
    next_target: Optional[float]
    analysis_confidence: float


class FibonacciExtension:
    """
    Fibonacci Extension Calculator
    Implements comprehensive Fibonacci extension analysis
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Standard Fibonacci extension levels
        self.extension_levels = [
            1.000,   # 100% - Minimum extension
            1.272,   # 127.2%
            1.414,   # 141.4%
            1.618,   # 161.8% - Golden ratio
            2.000,   # 200%
            2.618,   # 261.8%
            3.000,   # 300%
            4.236    # 423.6%
        ]
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        
        self.logger.info("Fibonacci Extension Calculator initialized")

    async def calculate_extension(
        self,
        symbol: str,
        price_data: List[Dict],
        swing_points: Optional[List[Tuple[datetime, float]]] = None
    ) -> ExtensionResult:
        """
        Calculate Fibonacci extension levels
        
        Args:
            symbol: Trading symbol
            price_data: List of OHLC data with timestamp
            swing_points: Optional ABC swing points, auto-detected if None
            
        Returns:
            ExtensionResult with complete extension analysis
        """
        start_time = time.perf_counter()
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Find swing points if not provided (A, B, C pattern)
            if swing_points is None:
                swing_points = await self._find_abc_pattern(df)
            
            if len(swing_points) < 3:
                raise ValueError("Need at least 3 swing points for extension calculation")
            
            current_price = df.iloc[-1]['close']
            
            # Determine extension direction
            extension_direction = await self._determine_extension_direction(swing_points)
            
            # Calculate extension levels
            extension_levels = await self._calculate_extension_levels(
                swing_points, current_price, extension_direction
            )
            
            # Calculate price targets
            price_targets = await self._calculate_price_targets(extension_levels, current_price)
            
            # Find next target
            next_target = await self._find_next_target(extension_levels, current_price)
            
            # Calculate analysis confidence
            confidence = await self._calculate_analysis_confidence(extension_levels, df)
            
            result = ExtensionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                swing_points=swing_points,
                current_price=current_price,
                extension_direction=extension_direction,
                extension_levels=extension_levels,
                price_targets=price_targets,
                next_target=next_target,
                analysis_confidence=confidence
            )
            
            # Update performance metrics
            calculation_time = time.perf_counter() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time
            
            self.logger.debug(f"Fibonacci extension calculated for {symbol} in {calculation_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci extension for {symbol}: {e}")
            raise

    async def _find_abc_pattern(self, df: pd.DataFrame) -> List[Tuple[datetime, float]]:
        """Find ABC pattern for extension calculation"""
        
        # Simplified ABC pattern detection
        lookback = min(30, len(df))
        recent_df = df.tail(lookback)
        
        # Find three significant points
        points = []
        
        # Point A - significant low/high
        max_idx = recent_df['high'].idxmax()
        min_idx = recent_df['low'].idxmin()
        
        if recent_df.loc[max_idx, 'timestamp'] < recent_df.loc[min_idx, 'timestamp']:
            # High then low pattern
            point_a = (recent_df.loc[max_idx, 'timestamp'], recent_df.loc[max_idx, 'high'])
            point_b = (recent_df.loc[min_idx, 'timestamp'], recent_df.loc[min_idx, 'low'])
            
            # Find point C (recent high)
            after_b = recent_df[recent_df['timestamp'] > point_b[0]]
            if not after_b.empty:
                c_idx = after_b['high'].idxmax()
                point_c = (after_b.loc[c_idx, 'timestamp'], after_b.loc[c_idx, 'high'])
                points = [point_a, point_b, point_c]
        else:
            # Low then high pattern
            point_a = (recent_df.loc[min_idx, 'timestamp'], recent_df.loc[min_idx, 'low'])
            point_b = (recent_df.loc[max_idx, 'timestamp'], recent_df.loc[max_idx, 'high'])
            
            # Find point C (recent low)
            after_b = recent_df[recent_df['timestamp'] > point_b[0]]
            if not after_b.empty:
                c_idx = after_b['low'].idxmin()
                point_c = (after_b.loc[c_idx, 'timestamp'], after_b.loc[c_idx, 'low'])
                points = [point_a, point_b, point_c]
        
        return points if len(points) == 3 else []

    async def _determine_extension_direction(
        self,
        swing_points: List[Tuple[datetime, float]]
    ) -> str:
        """Determine extension direction from swing points"""
        
        if len(swing_points) < 3:
            return 'neutral'
        
        point_a, point_b, point_c = swing_points[:3]
        
        # Bullish if A < B and C > A
        if point_a[1] < point_b[1] and point_c[1] > point_a[1]:
            return 'bullish'
        # Bearish if A > B and C < A
        elif point_a[1] > point_b[1] and point_c[1] < point_a[1]:
            return 'bearish'
        else:
            return 'neutral'

    async def _calculate_extension_levels(
        self,
        swing_points: List[Tuple[datetime, float]],
        current_price: float,
        direction: str
    ) -> List[ExtensionLevel]:
        """Calculate Fibonacci extension levels"""
        
        if len(swing_points) < 3:
            return []
        
        point_a, point_b, point_c = swing_points[:3]
        
        # Calculate AB distance
        ab_distance = abs(point_b[1] - point_a[1])
        
        extension_levels = []
        
        for level_pct in self.extension_levels:
            if direction == 'bullish':
                price_level = point_c[1] + (ab_distance * level_pct)
                level_type = 'target' if price_level > current_price else 'support'
            elif direction == 'bearish':
                price_level = point_c[1] - (ab_distance * level_pct)
                level_type = 'target' if price_level < current_price else 'resistance'
            else:
                continue
            
            # Calculate strength (simplified)
            strength = 0.9 if level_pct in [1.272, 1.618, 2.618] else 0.7
            
            # Calculate distance and probability
            distance = abs(price_level - current_price)
            probability = max(0.1, 1.0 - (distance / current_price))
            
            ext_level = ExtensionLevel(
                level_percentage=level_pct,
                price_level=price_level,
                level_type=level_type,
                strength=strength,
                distance_from_current=distance,
                probability=probability
            )
            
            extension_levels.append(ext_level)
        
        # Sort by distance from current price
        extension_levels.sort(key=lambda x: x.distance_from_current)
        
        return extension_levels

    async def _calculate_price_targets(
        self,
        extension_levels: List[ExtensionLevel],
        current_price: float
    ) -> List[ExtensionTarget]:
        """Calculate price targets from extension levels"""
        
        targets = []
        
        for level in extension_levels[:5]:  # Top 5 levels
            if level.level_type == 'target':
                # Calculate risk-reward (simplified)
                risk = current_price * 0.02  # 2% risk assumption
                reward = abs(level.price_level - current_price)
                risk_reward = reward / risk if risk > 0 else 0
                
                target = ExtensionTarget(
                    target_price=level.price_level,
                    target_level=level.level_percentage,
                    confidence=level.strength * level.probability,
                    time_estimate=None,  # Would need time analysis
                    risk_reward_ratio=risk_reward
                )
                
                targets.append(target)
        
        # Sort by confidence
        targets.sort(key=lambda x: x.confidence, reverse=True)
        
        return targets

    async def _find_next_target(
        self,
        extension_levels: List[ExtensionLevel],
        current_price: float
    ) -> Optional[float]:
        """Find the next price target"""
        
        targets = [level for level in extension_levels if level.level_type == 'target']
        
        if not targets:
            return None
        
        # Return closest target
        closest_target = min(targets, key=lambda x: x.distance_from_current)
        return closest_target.price_level

    async def _calculate_analysis_confidence(
        self,
        extension_levels: List[ExtensionLevel],
        df: pd.DataFrame
    ) -> float:
        """Calculate overall confidence in extension analysis"""
        
        if not extension_levels:
            return 0.0
        
        # Base confidence on level strengths
        avg_strength = sum(level.strength for level in extension_levels) / len(extension_levels)
        
        # Data quality factor
        data_quality = min(1.0, len(df) / 50)
        
        confidence = (avg_strength * 0.7) + (data_quality * 0.3)
        
        return min(1.0, confidence)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the calculator"""
        
        if self.calculation_count == 0:
            return {
                'calculations_performed': 0,
                'average_calculation_time': 0.0,
                'total_calculation_time': 0.0
            }
        
        return {
            'calculations_performed': self.calculation_count,
            'average_calculation_time': self.total_calculation_time / self.calculation_count,
            'total_calculation_time': self.total_calculation_time
        }

    def calculate(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """
        Standard calculate method for BaseIndicator compatibility
        
        Args:
            data: Market data in dict format with high, low, close arrays
                 or list of OHLC dictionaries, or pandas DataFrame
        
        Returns:
            Dict containing Fibonacci extension levels and price targets
        """
        start_time = time.time()
        
        try:
            # Convert data to standard format
            if isinstance(data, pd.DataFrame):
                high_values = data['high'].tolist()
                low_values = data['low'].tolist()
                close_values = data['close'].tolist()
            elif isinstance(data, dict):
                high_values = data.get('high', [])
                low_values = data.get('low', [])
                close_values = data.get('close', [])
            else:
                # Assume list of dicts
                high_values = [d.get('high', 0) for d in data]
                low_values = [d.get('low', 0) for d in data]
                close_values = [d.get('close', 0) for d in data]
            
            if len(high_values) < 3 or len(low_values) < 3:
                return {"error": "Insufficient data for Fibonacci extension calculation (need at least 3 points)"}
            
            # Find ABC pattern (simplified approach)
            # A = initial swing point
            # B = retracement point  
            # C = current extension point
            
            # Find significant highs and lows for ABC pattern
            recent_high = max(high_values[-10:]) if len(high_values) >= 10 else max(high_values)
            recent_low = min(low_values[-10:]) if len(low_values) >= 10 else min(low_values)
            current_price = close_values[-1] if close_values else recent_high
            
            # Determine trend and ABC points
            if current_price > recent_low:
                # Bullish extension: A=low, B=high, C=current
                point_a = recent_low
                point_b = recent_high
                point_c = current_price
                trend_direction = "bullish"
            else:
                # Bearish extension: A=high, B=low, C=current
                point_a = recent_high
                point_b = recent_low
                point_c = current_price
                trend_direction = "bearish"
            
            # Calculate base range (A to B)
            base_range = abs(point_b - point_a)
            
            if base_range == 0:
                return {"error": "No base range available for Fibonacci extension calculation"}
            
            # Calculate extension levels
            extension_levels = {}
            price_targets = {}
            
            for level in self.extension_levels:
                if trend_direction == "bullish":
                    # Extensions above point C
                    extension_price = point_c + (base_range * (level - 1.0))
                else:
                    # Extensions below point C
                    extension_price = point_c - (base_range * (level - 1.0))
                
                level_name = f"{level * 100:.1f}%"
                extension_levels[level_name] = round(extension_price, 5)
                
                # Calculate distance and probability
                distance = abs(extension_price - current_price)
                # Simple probability model based on distance
                probability = max(0.1, 1.0 - (distance / base_range))
                
                price_targets[level_name] = {
                    "price": round(extension_price, 5),
                    "distance": round(distance, 5),
                    "probability": round(probability, 3)
                }
            
            # Find next target (closest extension above current price for bullish, below for bearish)
            next_target = None
            min_distance = float('inf')
            
            for level_name, target_info in price_targets.items():
                target_price = target_info["price"]
                if trend_direction == "bullish" and target_price > current_price:
                    distance = target_price - current_price
                    if distance < min_distance:
                        min_distance = distance
                        next_target = {"level": level_name, "price": target_price, "distance": distance}
                elif trend_direction == "bearish" and target_price < current_price:
                    distance = current_price - target_price
                    if distance < min_distance:
                        min_distance = distance
                        next_target = {"level": level_name, "price": target_price, "distance": distance}
            
            # Performance tracking
            calculation_time = time.time() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time
            
            result = {
                "symbol": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "abc_points": {
                    "point_a": point_a,
                    "point_b": point_b,
                    "point_c": point_c
                },
                "current_price": current_price,
                "base_range": base_range,
                "trend_direction": trend_direction,
                "extension_levels": extension_levels,
                "price_targets": price_targets,
                "next_target": next_target,
                "calculation_time_ms": round(calculation_time * 1000, 2),
                "total_calculations": self.calculation_count
            }
            
            self.logger.info(f"Fibonacci extension calculated successfully in {calculation_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci extension: {e}")
            return {"error": str(e)}
