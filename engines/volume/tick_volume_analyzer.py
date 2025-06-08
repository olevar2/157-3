# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Tick Volume Analyzer - High-Precision Volume Analysis
Platform3 - Humanitarian Trading System

The Tick Volume Analyzer processes tick-level market data to provide detailed 
intraday volume flow analysis. It analyzes tick-by-tick data to identify volume 
anomalies, buying/selling pressure imbalances, and institutional activity 
through microstructure patterns.

Key Features:
- Tick distribution analysis
- Up/down tick volume ratio tracking
- Tick intensity measurement
- Average tick size calculation
- Delta volume flow visualization
- Real-time volume profile clustering

Humanitarian Mission: Enhance timing precision through tick-level volume analysis
to maximize profit capture efficiency in high-frequency humanitarian trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from engines.indicator_base import IndicatorSignal, TechnicalIndicator, ServiceError
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TickVolumeSignal(IndicatorSignal):
    """Tick volume-specific signal with detailed microstructure analysis"""
    uptick_ratio: float = 0.5
    tick_intensity: float = 0.0
    average_tick_size: float = 0.0
    delta_volume: float = 0.0
    pressure_direction: str = "neutral"  # "buying", "selling", "neutral"
    momentum_quality: float = 0.0  # 0-1 scale


class TickVolumeAnalyzer(TechnicalIndicator):
    """
    TickVolumeAnalyzer processes tick-level data to identify volume flow patterns,
    buying/selling pressure, and smart money footprints in high-frequency trading data.
    """
    
    def __init__(self, config: dict = None, intensity_window: int = 1000, 
                 delta_threshold: float = 0.15,
                 time_window_seconds: int = 3600):
        """
        Initialize the TickVolumeAnalyzer with configurable parameters
        
        Parameters:
        -----------
        config: dict
            Configuration dictionary
        intensity_window: int
            Number of ticks for intensity calculation
        delta_threshold: float
            Threshold to determine significant delta volume (0.0 - 1.0)
        time_window_seconds: int
            Time window in seconds for calculations (e.g. 3600 = 1 hour)
        """
        super().__init__(config)
        self.logger.info(f"TickVolumeAnalyzer initialized with intensity_window={intensity_window}")
        self.intensity_window = intensity_window
        self.delta_threshold = delta_threshold
        self.time_window = time_window_seconds
        self._cached_tick_data = []
    
    def _preprocess_tick_data(self, tick_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess raw tick data into standardized format
        
        Parameters:
        -----------
        tick_data: List[Dict[str, Any]]
            Raw tick data with price, volume, and direction information
            
        Returns:
        --------
        List[Dict[str, Any]]
            Standardized tick data
        """
        try:
            processed_data = []
            
            for tick in tick_data:
                if not all(k in tick for k in ['price', 'volume']):
                    continue
                    
                # Standardize tick format
                standardized_tick = {
                    'price': float(tick['price']),
                    'volume': float(tick['volume']),
                    'timestamp': tick.get('timestamp', datetime.now()),
                    'direction': tick.get('direction', None)
                }
                
                # Infer direction if not provided
                if standardized_tick['direction'] is None and len(processed_data) > 0:
                    prev_price = processed_data[-1]['price']
                    if standardized_tick['price'] > prev_price:
                        standardized_tick['direction'] = 'up'
                    elif standardized_tick['price'] < prev_price:
                        standardized_tick['direction'] = 'down'
                    else:
                        standardized_tick['direction'] = processed_data[-1]['direction']
                elif standardized_tick['direction'] is None:
                    standardized_tick['direction'] = 'neutral'
                
                processed_data.append(standardized_tick)
            
            return processed_data
        except Exception as e:
            self.logger.error(f"Error preprocessing tick data: {str(e)}")
            return []
    
    def analyze_tick_distribution(self, tick_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze tick distribution to identify volume flow patterns
        
        Parameters:
        -----------
        tick_data: List[Dict[str, Any]]
            List of ticks with price, volume, direction
            
        Returns:
        --------
        Dict[str, float]
            Tick distribution analysis metrics
        """
        try:
            # Process the incoming tick data
            processed_ticks = self._preprocess_tick_data(tick_data)
            if not processed_ticks: # Check if the list is empty
                return {
                    'uptick_ratio': 0.5,
                    'tick_intensity': 0.0,
                    'average_tick_size': 0.0,
                    'delta_volume': 0.0
                }
            
            # Extract values for analysis
            uptick_volume = sum(tick['volume'] for tick in processed_ticks 
                               if tick['direction'] == 'up')
            downtick_volume = sum(tick['volume'] for tick in processed_ticks 
                                 if tick['direction'] == 'down')
            total_volume = uptick_volume + downtick_volume
            
            # Handle edge case with zero volume
            if total_volume <= 0:
                return {
                    'uptick_ratio': 0.5,
                    'tick_intensity': 0.0,
                    'average_tick_size': 0.0,
                    'delta_volume': 0.0
                }
            
            # Calculate metrics
            uptick_ratio = uptick_volume / total_volume
            
            # Calculate time range in seconds
            time_values = [tick['timestamp'] for tick in processed_ticks 
                          if isinstance(tick['timestamp'], datetime)]
            
            if len(time_values) >= 2:
                time_range_seconds = (max(time_values) - min(time_values)).total_seconds()
                tick_intensity = len(processed_ticks) / (time_range_seconds or 1.0)
            else:
                tick_intensity = len(processed_ticks) / self.time_window
            
            avg_tick_size = total_volume / len(processed_ticks)
            delta_volume = (uptick_volume - downtick_volume) / total_volume
            
            return {
                'uptick_ratio': uptick_ratio,
                'tick_intensity': tick_intensity,
                'average_tick_size': avg_tick_size,
                'delta_volume': delta_volume
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing tick distribution: {str(e)}")
            return {
                'uptick_ratio': 0.5,
                'tick_intensity': 0.0,
                'average_tick_size': 0.0,
                'delta_volume': 0.0
            }

    def detect_divergences(self, tick_data: List[Dict[str, Any]], 
                          price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect divergences between tick volume and price movement
        
        Parameters:
        -----------
        tick_data: List[Dict[str, Any]]
            List of tick data
        price_data: pd.DataFrame
            Price OHLC data for the same period
            
        Returns:
        --------
        Dict[str, Any]
            Divergence analysis results
        """
        try:
            result = {
                'tick_price_divergence': False,
                'hidden_accumulation': False,
                'hidden_distribution': False,
                'divergence_score': 0.0
            }
            
            if not tick_data or len(price_data) < 2:
                return result
                
            # Calculate price change direction
            price_change = price_data['close'].iloc[-1] - price_data['close'].iloc[0]
            price_direction = "up" if price_change > 0 else "down"
            
            # Calculate tick metrics
            tick_metrics = self.analyze_tick_distribution(tick_data)
            delta_volume = tick_metrics['delta_volume']
            
            # Assess divergence
            tick_direction = "up" if delta_volume > self.delta_threshold else \
                            "down" if delta_volume < -self.delta_threshold else "neutral"
            
            has_divergence = (price_direction == "up" and tick_direction == "down") or \
                             (price_direction == "down" and tick_direction == "up")
            
            # Calculate divergence score
            if has_divergence:
                divergence_score = min(1.0, abs(delta_volume) * 2)
            else:
                divergence_score = 0.0
                
            # Detect hidden accumulation/distribution
            hidden_accumulation = price_direction == "down" and tick_direction == "up"
            hidden_distribution = price_direction == "up" and tick_direction == "down"
            
            return {
                'tick_price_divergence': has_divergence,
                'hidden_accumulation': hidden_accumulation,
                'hidden_distribution': hidden_distribution,
                'divergence_score': divergence_score
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting divergences: {str(e)}")
            return {
                'tick_price_divergence': False,
                'hidden_accumulation': False,
                'hidden_distribution': False,
                'divergence_score': 0.0
            }
    
    def calculate_momentum_quality(self, tick_data: List[Dict[str, Any]]) -> float:
        """
        Calculate momentum quality based on volume and tick consistency
        
        Parameters:
        -----------
        tick_data: List[Dict[str, Any]]
            List of tick data
            
        Returns:
        --------
        float
            Momentum quality score (0-1)
        """
        try:
            if not tick_data or len(tick_data) < 10:
                return 0.0
                
            processed_ticks = self._preprocess_tick_data(tick_data)
            
            # Extract directions and volumes
            directions = [1 if tick['direction'] == 'up' else -1 if tick['direction'] == 'down' else 0 
                         for tick in processed_ticks]
            volumes = [tick['volume'] for tick in processed_ticks]
            
            # Calculate direction consistency (when 3+ ticks move in same direction)
            direction_consistency = 0
            current_direction = 0
            current_streak = 0
            
            for direction in directions:
                if direction == current_direction and direction != 0:
                    current_streak += 1
                    if current_streak >= 3:
                        direction_consistency += 1
                else:
                    current_direction = direction
                    current_streak = 1
                    
            direction_consistency = min(1.0, direction_consistency / (len(directions) / 3))
            
            # Calculate volume consistency (higher volumes on directional moves)
            if len(volumes) <= 1:
                volume_consistency = 0.0
            else:
                directional_volumes = [volumes[i] for i in range(len(directions)) if directions[i] != 0]
                neutral_volumes = [volumes[i] for i in range(len(directions)) if directions[i] == 0]
                
                if not directional_volumes or not neutral_volumes:
                    volume_consistency = 0.5
                else:
                    avg_directional = np.mean(directional_volumes)
                    avg_neutral = np.mean(neutral_volumes) if neutral_volumes else 0
                    
                    volume_consistency = min(1.0, max(0.0, (avg_directional / (avg_neutral + 0.0001) - 0.5) / 2))
            
            # Combine into momentum quality score
            momentum_quality = (direction_consistency * 0.6) + (volume_consistency * 0.4)
            return min(1.0, max(0.0, momentum_quality))
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum quality: {str(e)}")
            return 0.0
    
    def calculate(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> TickVolumeSignal:
        """
        Calculate tick volume metrics from tick data
        
        Parameters:
        -----------
        data: Union[List[Dict[str, Any]], Dict[str, Any]]
            Either a list of tick data or a dict with 'ticks' key containing tick data
            
        Returns:
        --------
        TickVolumeSignal
            Comprehensive tick volume analysis
        """
        try:
            # Extract tick data from input
            if isinstance(data, dict) and 'ticks' in data:
                tick_data = data['ticks']
                price_data = data.get('price_data', None)
            else:
                tick_data = data
                price_data = None
                
            if not tick_data:
                raise ValueError("No tick data provided")
                
            # Preprocess tick data
            processed_ticks = self._preprocess_tick_data(tick_data)
            
            if len(processed_ticks) < 10:
                raise ValueError("Insufficient tick data: need at least 10 ticks")
                
            # Perform tick distribution analysis
            tick_analysis = self.analyze_tick_distribution(processed_ticks)
            
            # Calculate momentum quality
            momentum = self.calculate_momentum_quality(processed_ticks)
            
            # Determine pressure direction based on delta volume
            delta = tick_analysis['delta_volume']
            if delta > self.delta_threshold:
                pressure = "buying"
            elif delta < -self.delta_threshold:
                pressure = "selling"
            else:
                pressure = "neutral"
                
            # Calculate divergence if price data is available
            divergence = {}
            if price_data is not None and isinstance(price_data, pd.DataFrame) and len(price_data) > 0:
                divergence = self.detect_divergences(processed_ticks, price_data)
            
            # Create signal with buy/sell recommendation
            signal_direction = "neutral"
            signal_strength = 0.0
            
            if pressure == "buying" and momentum > 0.6:
                signal_direction = "buy"
                signal_strength = min(1.0, momentum * abs(delta) * 2)
            elif pressure == "selling" and momentum > 0.6:
                signal_direction = "sell"
                signal_strength = min(1.0, momentum * abs(delta) * 2)
                
            # If divergence exists, it might override the signal
            if divergence.get('hidden_accumulation', False) and divergence.get('divergence_score', 0) > 0.7:
                signal_direction = "buy"
                signal_strength = min(1.0, divergence.get('divergence_score', 0) * 1.2)
            elif divergence.get('hidden_distribution', False) and divergence.get('divergence_score', 0) > 0.7:
                signal_direction = "sell"
                signal_strength = min(1.0, divergence.get('divergence_score', 0) * 1.2)
            
            # Create and return the tick volume signal
            return TickVolumeSignal(
                uptick_ratio=tick_analysis['uptick_ratio'],
                tick_intensity=tick_analysis['tick_intensity'],
                average_tick_size=tick_analysis['average_tick_size'],
                delta_volume=tick_analysis['delta_volume'],
                pressure_direction=pressure,
                momentum_quality=momentum,
                signal=signal_direction,
                strength=signal_strength,
                timestamp=processed_ticks[-1]['timestamp'] if isinstance(processed_ticks[-1]['timestamp'], datetime) else None
            )
        
        except Exception as e:
            self.logger.error(f"Error in TickVolumeAnalyzer calculation: {str(e)}")
            raise ServiceError(f"Calculation failed: {str(e)}")
    
    def generate_signal(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate trading signals based on tick volume analysis
        
        Parameters:
        -----------
        data: Union[List[Dict[str, Any]], Dict[str, Any]]
            Tick data for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Trading signal with direction, strength and analysis
        """
        signal = self.calculate(data)
        
        return {
            "direction": signal.signal,
            "strength": signal.strength,
            "timestamp": signal.timestamp,
            "metadata": {
                "uptick_ratio": signal.uptick_ratio,
                "tick_intensity": signal.tick_intensity,
                "average_tick_size": signal.average_tick_size,
                "delta_volume": signal.delta_volume,
                "pressure": signal.pressure_direction,
                "momentum_quality": signal.momentum_quality
            }
        }