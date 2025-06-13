# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Projection Arc Calculator
Platform3 Phase 3 - Enhanced with Framework Integration
Projection and arc calculations for advanced Fibonacci analysis.

This module provides projection and arc analysis including:
- Fibonacci projections
- Arc calculations
- Advanced geometric analysis
"""
import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework


@dataclass
class FibonacciProjection:
    """Fibonacci projection data"""
    projection_price: float
    projection_time: datetime
    projection_type: str
    confidence: float


@dataclass
class FibonacciArc:
    """Fibonacci arc data"""
    center_point: Tuple[datetime, float]
    radius: float
    arc_levels: List[float]
    arc_strength: float


@dataclass
class ProjectionResult:
    """Projection analysis result"""
    symbol: str
    timestamp: datetime
    projections: List[FibonacciProjection]
    arcs: List[FibonacciArc]
    next_projection: Optional[FibonacciProjection]
    analysis_confidence: float


class ProjectionArcCalculator:
    """Fibonacci Projection and Arc Calculator"""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.fibonacci_ratios = [0.382, 0.618, 1.0, 1.618, 2.618]
        
    async def calculate_projections(
        self,
        symbol: str,
        price_data: List[Dict],
        base_points: List[Tuple[datetime, float]]
    ) -> ProjectionResult:
        """Calculate Fibonacci projections and arcs"""
        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        projections = []
        arcs = []
        
        if len(base_points) >= 2:
            point1, point2 = base_points[:2]
            price_diff = abs(point2[1] - point1[1])
            time_diff = (point2[0] - point1[0]).days
            
            for ratio in self.fibonacci_ratios:
                proj_price = point2[1] + (price_diff * ratio)
                proj_time = point2[0] + timedelta(days=int(time_diff * ratio))
                
                projections.append(FibonacciProjection(
                    projection_price=proj_price,
                    projection_time=proj_time,
                    projection_type=f"Fibonacci_{ratio}",
                    confidence=0.8 if ratio in [0.618, 1.618] else 0.6
                ))
            
            # Create arc
            center = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
            radius = math.sqrt((point2[1] - point1[1])**2 + time_diff**2)
            
            arcs.append(FibonacciArc(
                center_point=center,
                radius=radius,
                arc_levels=self.fibonacci_ratios,
                arc_strength=0.7
            ))
        
        next_proj = projections[0] if projections else None
        
        return ProjectionResult(
            symbol=symbol,
            timestamp=datetime.now(),
            projections=projections,
            arcs=arcs,
            next_projection=next_proj,
            analysis_confidence=0.7 if projections else 0.3
        )

    def calculate(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """
        Standard calculate method for BaseIndicator compatibility

        Args:
            data: Market data in dict format with high, low, close arrays
                 or list of OHLC dictionaries, or pandas DataFrame

        Returns:
            Dict containing Fibonacci projection and arc analysis
        """
        start_time = time.time()

        try:
            # Convert data to standard format
            if isinstance(data, pd.DataFrame):
                high_values = data['high'].tolist()
                low_values = data['low'].tolist()
                close_values = data['close'].tolist()
                if 'timestamp' in data.columns:
                    timestamps = data['timestamp'].tolist()
                else:
                    timestamps = [datetime.now() - timedelta(days=len(data)-i-1) for i in range(len(data))]
            elif isinstance(data, dict):
                high_values = data.get('high', [])
                low_values = data.get('low', [])
                close_values = data.get('close', [])
                timestamps = data.get('timestamp', [])
                if not timestamps:
                    timestamps = [datetime.now() - timedelta(days=len(close_values)-i-1) for i in range(len(close_values))]
            else:
                # Assume list of dicts
                high_values = [d.get('high', 0) for d in data]
                low_values = [d.get('low', 0) for d in data]
                close_values = [d.get('close', 0) for d in data]
                timestamps = [d.get('timestamp', datetime.now()) for d in data]

            if not high_values or not low_values or not close_values:
                return {"error": "Insufficient data for projection analysis"}

            # Find significant swing points for projections
            swing_high_idx = high_values.index(max(high_values))
            swing_low_idx = low_values.index(min(low_values))
            
            # Sort swing points by time
            swings = [
                (timestamps[swing_low_idx], low_values[swing_low_idx]),
                (timestamps[swing_high_idx], high_values[swing_high_idx])
            ]
            swings.sort(key=lambda x: x[0])

            if len(swings) < 2:
                return {"error": "Need at least 2 swing points for projection analysis"}

            point1, point2 = swings[:2]
            price_diff = abs(point2[1] - point1[1])
            time_diff = (point2[0] - point1[0]).days if isinstance(point2[0], datetime) else 1

            # Calculate Fibonacci projections
            projections = []
            for ratio in self.fibonacci_ratios:
                proj_price = point2[1] + (price_diff * ratio)
                if isinstance(point2[0], datetime):
                    proj_time = point2[0] + timedelta(days=int(time_diff * ratio))
                else:
                    proj_time = datetime.now() + timedelta(days=int(ratio))
                
                projections.append({
                    "projection_price": round(proj_price, 5),
                    "projection_time": proj_time.isoformat() if isinstance(proj_time, datetime) else str(proj_time),
                    "projection_ratio": ratio,
                    "projection_type": f"Fibonacci_{ratio}",
                    "confidence": 0.8 if ratio in [0.618, 1.618] else 0.6,
                    "price_distance": round(price_diff * ratio, 5)
                })

            # Calculate Fibonacci arcs
            center_time = point1[0] + (point2[0] - point1[0]) / 2 if isinstance(point1[0], datetime) else datetime.now()
            center_price = (point1[1] + point2[1]) / 2
            radius = math.sqrt(price_diff**2 + time_diff**2)

            arc_info = {
                "center_time": center_time.isoformat() if isinstance(center_time, datetime) else str(center_time),
                "center_price": round(center_price, 5),
                "radius": round(radius, 2),
                "arc_levels": self.fibonacci_ratios,
                "arc_strength": 0.7
            }

            # Performance tracking
            calculation_time = time.time() - start_time

            result = {
                "symbol": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "base_points": [
                    {"time": point1[0].isoformat() if isinstance(point1[0], datetime) else str(point1[0]), "price": point1[1]},
                    {"time": point2[0].isoformat() if isinstance(point2[0], datetime) else str(point2[0]), "price": point2[1]}
                ],
                "price_difference": round(price_diff, 5),
                "time_difference_days": time_diff,
                "fibonacci_projections": projections,
                "fibonacci_arc": arc_info,
                "next_projection": projections[0] if projections else None,
                "total_projections": len(projections),
                "analysis_confidence": 0.7 if projections else 0.3,
                "calculation_time_ms": round(calculation_time * 1000, 2)
            }

            self.logger.info(f"Projection/Arc analysis calculated successfully in {calculation_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating projection/arc analysis: {e}")
            return {"error": str(e)}
