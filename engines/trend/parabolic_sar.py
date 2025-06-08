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
ParabolicSar - Enhanced Trading Engine
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

class ParabolicSar:
    """Enhanced ParabolicSar with Platform3 framework integration"""
    
    def __init__(self):
        """Initialize with Platform3 framework components"""
        self.logger = Platform3Logger(self.__class__.__name__)
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
        
        self.logger.info(f"{self.__class__.__name__} initialized successfully")
        
    async def calculate(self, data: Union[np.ndarray, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Calculate Parabolic SAR values with enhanced accuracy
        
        Args:
            data: Input market data - can be:
                  - Dictionary with 'high', 'low', 'close' keys and optional parameters
                  - Legacy np.ndarray (will be processed as close prices)
            
        Returns:
            Dictionary containing calculated values or None on error
        """
        start_time = time.time()
        
        try:
            self.logger.debug("Starting calculation process")
            
            # Input validation
            if data is None or len(data) == 0:
                raise ServiceError("Invalid input data", "INVALID_INPUT")
            
            # Perform calculations
            result = await self._perform_calculation(data)
            
            # Performance monitoring
            execution_time = time.time() - start_time
            self.logger.info(f"Calculation completed in {execution_time:.4f}s")
            
            return result
            
        except ServiceError as e:
            self.logger.error(f"Service error: {e}", extra={"error": e.to_dict()})
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.error_system.handle_error(e, self.__class__.__name__)
            return None
    
    async def _perform_calculation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Parabolic SAR values
        
        Parabolic SAR formula:
        - Uptrend: SAR(t+1) = SAR(t) + AF * (EP - SAR(t))
        - Downtrend: SAR(t+1) = SAR(t) - AF * (SAR(t) - EP)
        
        Where:
        - SAR = Stop and Reverse
        - AF = Acceleration Factor (starts at 0.02, increases by 0.02 each time EP is updated, max 0.20)
        - EP = Extreme Point (highest high in uptrend, lowest low in downtrend)
        
        Args:
            data: Dictionary containing 'high', 'low', 'close' and optional parameters
            
        Returns:
            Dictionary containing Parabolic SAR values and signals
        """
        try:
            # Extract data
            high_prices = np.array(data.get('high', []))
            low_prices = np.array(data.get('low', []))
            close_prices = np.array(data.get('close', []))
            
            # Parameters
            af_start = data.get('af_start', 0.02)
            af_increment = data.get('af_increment', 0.02)
            af_max = data.get('af_max', 0.20)
            
            if len(high_prices) != len(low_prices) or len(high_prices) != len(close_prices):
                raise ValueError("High, low, and close price arrays must have same length")
            
            if len(high_prices) < 2:
                raise ValueError("Need at least 2 data points for Parabolic SAR calculation")
            
            length = len(high_prices)
            sar = np.full(length, np.nan)
            trend = np.full(length, np.nan)  # 1 for uptrend, -1 for downtrend
            af = np.full(length, np.nan)
            ep = np.full(length, np.nan)  # Extreme Point
            
            # Initialize first values
            # Assume starting in uptrend
            sar[0] = low_prices[0]
            trend[0] = 1
            af[0] = af_start
            ep[0] = high_prices[0]
            
            # Calculate SAR for each period
            for i in range(1, length):
                prev_sar = sar[i-1]
                prev_trend = trend[i-1]
                prev_af = af[i-1]
                prev_ep = ep[i-1]
                
                # Calculate new SAR based on previous trend
                if prev_trend == 1:  # Uptrend
                    new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
                    
                    # SAR should not exceed the low of the current or previous period
                    new_sar = min(new_sar, low_prices[i], low_prices[i-1])
                    
                    # Check for trend reversal
                    if low_prices[i] < new_sar:
                        # Trend reversal to downtrend
                        trend[i] = -1
                        sar[i] = prev_ep  # SAR becomes the previous extreme point
                        af[i] = af_start  # Reset acceleration factor
                        ep[i] = low_prices[i]  # New extreme point is current low
                    else:
                        # Continue uptrend
                        trend[i] = 1
                        sar[i] = new_sar
                        
                        # Update extreme point and acceleration factor
                        if high_prices[i] > prev_ep:
                            ep[i] = high_prices[i]
                            af[i] = min(prev_af + af_increment, af_max)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af
                
                else:  # Downtrend
                    new_sar = prev_sar - prev_af * (prev_sar - prev_ep)
                    
                    # SAR should not exceed the high of the current or previous period
                    new_sar = max(new_sar, high_prices[i], high_prices[i-1])
                    
                    # Check for trend reversal
                    if high_prices[i] > new_sar:
                        # Trend reversal to uptrend
                        trend[i] = 1
                        sar[i] = prev_ep  # SAR becomes the previous extreme point
                        af[i] = af_start  # Reset acceleration factor
                        ep[i] = high_prices[i]  # New extreme point is current high
                    else:
                        # Continue downtrend
                        trend[i] = -1
                        sar[i] = new_sar
                        
                        # Update extreme point and acceleration factor
                        if low_prices[i] < prev_ep:
                            ep[i] = low_prices[i]
                            af[i] = min(prev_af + af_increment, af_max)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af
            
            # Generate signals
            signals = self._generate_psar_signals(sar, trend, close_prices)
            
            return {
                "sar": sar.tolist(),
                "trend": trend.tolist(),
                "acceleration_factor": af.tolist(),
                "extreme_point": ep.tolist(),
                "signals": signals,
                "parameters": {
                    "af_start": af_start,
                    "af_increment": af_increment,
                    "af_max": af_max
                },
                "timestamp": datetime.now().isoformat(),
                "engine": self.__class__.__name__
            }
            
        except Exception as e:
            self.logger.error(f"Parabolic SAR calculation error: {e}")
            raise ServiceError(f"Parabolic SAR calculation failed: {str(e)}", "CALCULATION_ERROR")
    
    def _generate_psar_signals(self, sar: np.ndarray, trend: np.ndarray, 
                              close_prices: np.ndarray) -> List[Dict[str, Any]]:
        """Generate trading signals based on Parabolic SAR"""
        signals = []
        
        for i in range(1, len(trend)):
            if np.isnan(trend[i]) or np.isnan(trend[i-1]):
                continue
            
            # Trend reversal signals
            if trend[i] == 1 and trend[i-1] == -1:
                signals.append({
                    "index": i,
                    "signal": "bullish_reversal",
                    "sar": sar[i],
                    "price": close_prices[i],
                    "trend": "uptrend"
                })
            
            elif trend[i] == -1 and trend[i-1] == 1:
                signals.append({
                    "index": i,
                    "signal": "bearish_reversal",
                    "sar": sar[i],
                    "price": close_prices[i],
                    "trend": "downtrend"
                })
            
            # Current trend signals
            elif trend[i] == 1 and close_prices[i] > sar[i]:
                signals.append({
                    "index": i,
                    "signal": "bullish_trend",
                    "sar": sar[i],
                    "price": close_prices[i],
                    "trend": "uptrend"
                })
            
            elif trend[i] == -1 and close_prices[i] < sar[i]:
                signals.append({
                    "index": i,
                    "signal": "bearish_trend",
                    "sar": sar[i],
                    "price": close_prices[i],
                    "trend": "downtrend"
                })
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get engine parameters"""
        return {
            "engine_name": self.__class__.__name__,
            "version": "3.0.0",
            "framework": "Platform3"
        }
    
    async def validate_input(self, data: Any) -> bool:
        """Validate input data"""
        try:
            if data is None:
                return False
            if isinstance(data, np.ndarray) and len(data) == 0:
                return False
            return True
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return False
