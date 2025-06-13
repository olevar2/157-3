#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
AroonIndicator - Enhanced Trading Engine
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

class AroonIndicator:
    """Enhanced AroonIndicator with Platform3 framework integration"""
    
    def __init__(self):
        """Initialize with Platform3 framework components"""
        self.logger = Platform3Logger(self.__class__.__name__)
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
        
        self.logger.info(f"{self.__class__.__name__} initialized successfully")
        
    async def calculate(self, data: Union[np.ndarray, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Calculate Aroon indicator values with enhanced accuracy
        
        Args:
            data: Input market data - can be:
                  - Dictionary with 'high', 'low' keys and optional 'period'
                  - Legacy np.ndarray (will be processed as high prices)
            
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
        Calculate Aroon indicator values
        
        Aroon Up = ((n - periods since highest high) / n) * 100
        Aroon Down = ((n - periods since lowest low) / n) * 100
        Aroon Oscillator = Aroon Up - Aroon Down
        
        Args:
            data: Dictionary containing 'high', 'low', and optionally 'period' keys
            
        Returns:
            Dictionary containing Aroon Up, Aroon Down, and Aroon Oscillator values
        """
        try:
            # Extract data
            high_prices = np.array(data.get('high', []))
            low_prices = np.array(data.get('low', []))
            period = data.get('period', 14)
            
            if len(high_prices) != len(low_prices):
                raise ValueError("High and low price arrays must have same length")
            
            if len(high_prices) < period:
                raise ValueError(f"Not enough data points. Need at least {period}, got {len(high_prices)}")
            
            length = len(high_prices)
            aroon_up = np.full(length, np.nan)
            aroon_down = np.full(length, np.nan)
            aroon_oscillator = np.full(length, np.nan)
            
            # Calculate Aroon for each period
            for i in range(period - 1, length):
                # Get the window of data
                high_window = high_prices[i - period + 1:i + 1]
                low_window = low_prices[i - period + 1:i + 1]
                
                # Find highest high and lowest low positions
                highest_high_pos = np.argmax(high_window)
                lowest_low_pos = np.argmin(low_window)
                
                # Calculate periods since highest high and lowest low
                periods_since_high = period - 1 - highest_high_pos
                periods_since_low = period - 1 - lowest_low_pos
                
                # Calculate Aroon Up and Down
                aroon_up[i] = ((period - periods_since_high) / period) * 100
                aroon_down[i] = ((period - periods_since_low) / period) * 100
                
                # Calculate Aroon Oscillator
                aroon_oscillator[i] = aroon_up[i] - aroon_down[i]
            
            # Generate signals
            signals = self._generate_aroon_signals(aroon_up, aroon_down, aroon_oscillator)
            
            return {
                "aroon_up": aroon_up.tolist(),
                "aroon_down": aroon_down.tolist(),
                "aroon_oscillator": aroon_oscillator.tolist(),
                "signals": signals,
                "period": period,
                "timestamp": datetime.now().isoformat(),
                "engine": self.__class__.__name__
            }
            
        except Exception as e:
            self.logger.error(f"Aroon calculation error: {e}")
            raise ServiceError(f"Aroon calculation failed: {str(e)}", "CALCULATION_ERROR")
    
    def _generate_aroon_signals(self, aroon_up: np.ndarray, aroon_down: np.ndarray, 
                               aroon_oscillator: np.ndarray) -> List[Dict[str, Any]]:
        """Generate trading signals based on Aroon values"""
        signals = []
        
        for i in range(1, len(aroon_up)):
            if np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]):
                continue
            
            # Strong uptrend signal
            if aroon_up[i] > 70 and aroon_down[i] < 30:
                signals.append({
                    "index": i,
                    "signal": "strong_uptrend",
                    "aroon_up": aroon_up[i],
                    "aroon_down": aroon_down[i],
                    "oscillator": aroon_oscillator[i]
                })
            
            # Strong downtrend signal
            elif aroon_down[i] > 70 and aroon_up[i] < 30:
                signals.append({
                    "index": i,
                    "signal": "strong_downtrend",
                    "aroon_up": aroon_up[i],
                    "aroon_down": aroon_down[i],
                    "oscillator": aroon_oscillator[i]
                })
            
            # Crossover signals
            elif (aroon_up[i] > aroon_down[i] and 
                  aroon_up[i-1] <= aroon_down[i-1]):
                signals.append({
                    "index": i,
                    "signal": "bullish_crossover",
                    "aroon_up": aroon_up[i],
                    "aroon_down": aroon_down[i],
                    "oscillator": aroon_oscillator[i]
                })
            
            elif (aroon_down[i] > aroon_up[i] and 
                  aroon_down[i-1] <= aroon_up[i-1]):
                signals.append({
                    "index": i,
                    "signal": "bearish_crossover",
                    "aroon_up": aroon_up[i],
                    "aroon_down": aroon_down[i],
                    "oscillator": aroon_oscillator[i]
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
