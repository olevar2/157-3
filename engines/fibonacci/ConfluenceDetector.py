# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Confluence Detector
Platform3 Phase 3 - Enhanced with Framework Integration
Confluence area detection and analysis for Fibonacci levels.

This module provides confluence analysis including:
- Multi-level confluence detection
- Confluence strength assessment
- Signal generation from confluence areas
"""

from shared.logging.platform3_logger import Platform3Logger, LogMetadata
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError, BaseService
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import time
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ConfluenceArea:
    """Confluence area data"""
    area_center: float
    area_range: Tuple[float, float]
    strength: float
    contributing_levels: List[str]
    area_type: str  # 'support', 'resistance'


@dataclass
class ConfluenceSignal:
    """Trading signal from confluence"""
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float


@dataclass
class ConfluenceResult:
    """Confluence analysis result"""
    symbol: str
    timestamp: datetime
    confluence_areas: List[ConfluenceArea]
    signals: List[ConfluenceSignal]
    strongest_confluence: Optional[ConfluenceArea]
    analysis_confidence: float


class ConfluenceDetector:
    """Fibonacci Confluence Detector"""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    async def detect_confluence(
        self,
        symbol: str,
        fibonacci_levels: List[float],
        current_price: float
    ) -> ConfluenceResult:
        """Detect confluence areas from multiple Fibonacci levels"""
        confluence_areas = []
        
        # Group levels that are close together
        for i, level1 in enumerate(fibonacci_levels):
            close_levels = [level1]
            for j, level2 in enumerate(fibonacci_levels[i+1:], i+1):
                if abs(level1 - level2) / level1 <= 0.01:  # Within 1%
                    close_levels.append(level2)
            
            if len(close_levels) >= 2:
                area_center = sum(close_levels) / len(close_levels)
                area_range = (min(close_levels), max(close_levels))
                strength = len(close_levels) / len(fibonacci_levels)
                
                confluence_areas.append(ConfluenceArea(
                    area_center=area_center,
                    area_range=area_range,
                    strength=strength,
                    contributing_levels=[f"Level_{i}" for i in range(len(close_levels))],
                    area_type='support' if area_center < current_price else 'resistance'
                ))
        
        strongest = max(confluence_areas, key=lambda x: x.strength) if confluence_areas else None
        
        return ConfluenceResult(
            symbol=symbol,
            timestamp=datetime.now(),
            confluence_areas=confluence_areas,
            signals=[],
            strongest_confluence=strongest,
            analysis_confidence=0.8 if confluence_areas else 0.3
        )

    def calculate(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """
        Standard calculate method for BaseIndicator compatibility

        Args:
            data: Market data in dict format with high, low, close arrays
                 or list of OHLC dictionaries, or pandas DataFrame

        Returns:
            Dict containing Fibonacci confluence analysis
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

            if not high_values or not low_values or not close_values:
                return {"error": "Insufficient data for confluence analysis"}

            # Find swing high and low
            swing_high = max(high_values)
            swing_low = min(low_values)
            current_price = close_values[-1] if close_values else swing_high
            price_range = swing_high - swing_low

            if price_range == 0:
                return {"error": "No price range available for confluence analysis"}

            # Calculate multiple Fibonacci levels from different sources
            fibonacci_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
            
            # Retracement levels
            retracement_levels = []
            for ratio in fibonacci_ratios:
                level = swing_high - (price_range * ratio)
                retracement_levels.append(level)

            # Extension levels (assuming uptrend)
            extension_levels = []
            for ratio in [1.236, 1.382, 1.618]:
                level = swing_high + (price_range * (ratio - 1))
                extension_levels.append(level)

            all_levels = retracement_levels + extension_levels

            # Find confluence areas (levels within 1% of each other)
            confluence_areas = []
            processed_levels = set()

            for i, level1 in enumerate(all_levels):
                if i in processed_levels:
                    continue
                    
                close_levels = [level1]
                close_indices = [i]
                
                for j, level2 in enumerate(all_levels[i+1:], i+1):
                    if j not in processed_levels and abs(level1 - level2) / level1 <= 0.01:  # Within 1%
                        close_levels.append(level2)
                        close_indices.append(j)

                if len(close_levels) >= 2:
                    area_center = sum(close_levels) / len(close_levels)
                    area_range = (min(close_levels), max(close_levels))
                    strength = len(close_levels) / len(all_levels)
                    
                    confluence_areas.append({
                        "area_center": round(area_center, 5),
                        "area_range": [round(area_range[0], 5), round(area_range[1], 5)],
                        "strength": round(strength, 3),
                        "contributing_levels": len(close_levels),
                        "area_type": "support" if area_center < current_price else "resistance",
                        "distance_from_current": round(abs(area_center - current_price), 5)
                    })
                    
                    processed_levels.update(close_indices)

            # Sort by strength
            confluence_areas.sort(key=lambda x: x["strength"], reverse=True)

            # Performance tracking
            calculation_time = time.time() - start_time

            result = {
                "symbol": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "swing_high": swing_high,
                "swing_low": swing_low,
                "current_price": current_price,
                "price_range": price_range,
                "total_fibonacci_levels": len(all_levels),
                "confluence_areas": confluence_areas,
                "strongest_confluence": confluence_areas[0] if confluence_areas else None,
                "total_confluence_areas": len(confluence_areas),
                "analysis_confidence": 0.8 if confluence_areas else 0.3,
                "calculation_time_ms": round(calculation_time * 1000, 2)
            }

            self.logger.info(f"Confluence analysis calculated successfully in {calculation_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating confluence analysis: {e}")
            return {"error": str(e)}
