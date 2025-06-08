#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Gann Angles Time Cycles - Enhanced Trading Engine
Platform3 Phase 3 - Enhanced with Framework Integration
"""

from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time
import math

class GannAnglesTimeCycles:
    """Enhanced Gann Angles Time Cycles with Platform3 framework integration"""
    
    def __init__(self):
        """Initialize with Platform3 framework components"""
        self.logger = Platform3Logger(self.__class__.__name__)
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
        
        # Gann angle constants
        self.gann_angles = [1, 2, 3, 4, 8]  # 1x1, 2x1, 3x1, 4x1, 8x1
        self.time_cycles = [7, 14, 21, 30, 45, 60, 90, 120, 180, 360]  # Common Gann cycles
        
        self.logger.info(f"{self.__class__.__name__} initialized successfully")
        
    async def calculate_gann_angles(self, price_data: np.ndarray, time_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Calculate Gann angles and time cycles
        
        Args:
            price_data: Array of price values
            time_data: Array of time values
            
        Returns:
            Dictionary containing Gann analysis results or None on error
        """
        start_time = time.time()
        
        try:
            self.logger.debug("Starting Gann angles calculation")
            
            # Input validation
            if price_data is None or time_data is None:
                raise ServiceError("Invalid input data", "INVALID_INPUT")
            
            if len(price_data) != len(time_data):
                raise ServiceError("Price and time data length mismatch", "DATA_MISMATCH")
            
            # Calculate Gann levels
            result = await self._calculate_gann_levels(price_data, time_data)
            
            # Performance monitoring
            execution_time = time.time() - start_time
            self.logger.info(f"Gann angles calculation completed in {execution_time:.4f}s")
            
            return result
            
        except ServiceError as e:
            self.logger.error(f"Service error: {e}", extra={"error": e.to_dict()})
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.error_system.handle_error(e, self.__class__.__name__)
            return None
    
    async def _calculate_gann_levels(self, price_data: np.ndarray, time_data: np.ndarray) -> Dict[str, Any]:
        """
        Internal Gann levels calculation
        
        Args:
            price_data: Array of price values
            time_data: Array of time values
            
        Returns:
            Dictionary containing Gann analysis results
        """
        # Find swing highs and lows
        swing_points = self._identify_swing_points(price_data)
        
        # Calculate angle lines
        angle_lines = self._calculate_angle_lines(swing_points, price_data, time_data)
        
        # Calculate time cycles
        time_cycle_projections = self._calculate_time_cycles(time_data, swing_points)
        
        # Calculate price squares
        price_squares = self._calculate_price_squares(price_data)
        
        return {
            "swing_points": swing_points,
            "angle_lines": angle_lines,
            "time_cycles": time_cycle_projections,
            "price_squares": price_squares,
            "timestamp": datetime.now().isoformat(),
            "engine": self.__class__.__name__
        }
    
    def _identify_swing_points(self, price_data: np.ndarray) -> Dict[str, List]:
        """Identify swing highs and lows"""
        try:
            highs = []
            lows = []
            
            # Simple swing point identification
            for i in range(2, len(price_data) - 2):
                # Swing high
                if (price_data[i] > price_data[i-1] and price_data[i] > price_data[i-2] and
                    price_data[i] > price_data[i+1] and price_data[i] > price_data[i+2]):
                    highs.append({"index": i, "price": float(price_data[i])})
                
                # Swing low
                if (price_data[i] < price_data[i-1] and price_data[i] < price_data[i-2] and
                    price_data[i] < price_data[i+1] and price_data[i] < price_data[i+2]):
                    lows.append({"index": i, "price": float(price_data[i])})
            
            return {"highs": highs, "lows": lows}
            
        except Exception as e:
            self.logger.error(f"Error identifying swing points: {e}")
            return {"highs": [], "lows": []}
    
    def _calculate_angle_lines(self, swing_points: Dict, price_data: np.ndarray, time_data: np.ndarray) -> List[Dict]:
        """Calculate Gann angle lines"""
        try:
            angle_lines = []
            
            # Calculate from swing highs
            for high in swing_points["highs"]:
                for angle in self.gann_angles:
                    # Calculate slope for angle
                    slope = 1.0 / angle  # Price per time unit
                    
                    line = {
                        "type": "resistance",
                        "angle": f"1x{angle}",
                        "start_index": high["index"],
                        "start_price": high["price"],
                        "slope": slope,
                        "support_resistance": "resistance"
                    }
                    angle_lines.append(line)
            
            # Calculate from swing lows
            for low in swing_points["lows"]:
                for angle in self.gann_angles:
                    slope = 1.0 / angle
                    
                    line = {
                        "type": "support",
                        "angle": f"1x{angle}",
                        "start_index": low["index"],
                        "start_price": low["price"],
                        "slope": slope,
                        "support_resistance": "support"
                    }
                    angle_lines.append(line)
            
            return angle_lines
            
        except Exception as e:
            self.logger.error(f"Error calculating angle lines: {e}")
            return []
    
    def _calculate_time_cycles(self, time_data: np.ndarray, swing_points: Dict) -> List[Dict]:
        """Calculate Gann time cycle projections"""
        try:
            projections = []
            
            for cycle in self.time_cycles:
                for high in swing_points["highs"]:
                    if high["index"] + cycle < len(time_data):
                        projections.append({
                            "cycle_length": cycle,
                            "from_high": True,
                            "start_index": high["index"],
                            "projection_index": high["index"] + cycle,
                            "significance": "high"
                        })
                
                for low in swing_points["lows"]:
                    if low["index"] + cycle < len(time_data):
                        projections.append({
                            "cycle_length": cycle,
                            "from_high": False,
                            "start_index": low["index"],
                            "projection_index": low["index"] + cycle,
                            "significance": "low"
                        })
            
            return projections
            
        except Exception as e:
            self.logger.error(f"Error calculating time cycles: {e}")
            return []
    
    def _calculate_price_squares(self, price_data: np.ndarray) -> Dict[str, List]:
        """Calculate Gann price squares"""
        try:
            current_price = float(price_data[-1])
            
            # Calculate square roots and squares
            sqrt_price = math.sqrt(current_price)
            
            # Price squares up and down
            squares_up = []
            squares_down = []
            
            for i in range(1, 11):  # Next 10 squares
                next_sqrt = sqrt_price + i
                squares_up.append(float(next_sqrt ** 2))
                
                prev_sqrt = max(0, sqrt_price - i)
                if prev_sqrt > 0:
                    squares_down.append(float(prev_sqrt ** 2))
            
            return {
                "current_price": current_price,
                "current_sqrt": float(sqrt_price),
                "squares_up": squares_up,
                "squares_down": squares_down
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating price squares: {e}")
            return {"current_price": 0, "current_sqrt": 0, "squares_up": [], "squares_down": []}
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get engine parameters"""
        return {
            "engine_name": self.__class__.__name__,
            "version": "3.0.0",
            "framework": "Platform3",
            "gann_angles": self.gann_angles,
            "time_cycles": self.time_cycles
        }
    
    async def validate_input(self, price_data: Any, time_data: Any) -> bool:
        """Validate input data"""
        try:
            if price_data is None or time_data is None:
                return False
            if not isinstance(price_data, np.ndarray) or not isinstance(time_data, np.ndarray):
                return False
            if len(price_data) == 0 or len(time_data) == 0:
                return False
            if len(price_data) != len(time_data):
                return False
            return True
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return False
