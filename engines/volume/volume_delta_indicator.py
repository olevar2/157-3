# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Volume Delta Indicator - Directional Volume Flow Analysis
Platform3 - Humanitarian Trading System

The Volume Delta Indicator tracks the net volume flowing into buying versus selling
by classifying each transaction based on whether it was executed at the bid or ask.
This provides a detailed view of directional pressure in the market and reveals
hidden accumulation or distribution patterns.

Key Features:
- Buy/sell volume classification
- Cumulative delta calculation
- Delta divergence detection
- Volume intensity analysis
- Absorption pattern identification
- Liquidity utilization measurement

Humanitarian Mission: Identify hidden buying/selling pressure to anticipate price movements
before they occur, maximizing profit potential for humanitarian causes.
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
class VolumeDeltaSignal(IndicatorSignal):
    """Volume delta-specific signal with detailed buying/selling pressure analysis"""
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    delta: float = 0.0
    cumulative_delta: float = 0.0
    absorption_ratio: float = 0.0
    pressure_direction: str = "neutral"  # "buying", "selling", "neutral"
    confidence: float = 0.5  # Add confidence field with default value
    
    def __post_init__(self):
        # Auto-populate required base fields if not provided
        if not self.timestamp:
            self.timestamp = datetime.now()
        if not self.indicator_name:
            self.indicator_name = "VolumeDeltaIndicator"
        if not self.signal_type:
            self.signal_type = self.pressure_direction
        if not hasattr(self, 'confidence') or self.confidence is None:
            self.confidence = 0.5  # Default confidence level


class VolumeDeltaIndicator(TechnicalIndicator):
    """
    VolumeDeltaIndicator tracks the net difference between buying and selling volume
    to identify hidden pressure, absorption patterns, and directional bias in markets.
    """
    
    def __init__(self, config: dict = None, delta_threshold: float = 0.15, 
                 lookback_periods: int = 20,
                 absorption_threshold: float = 0.7):
        """
        Initialize the VolumeDeltaIndicator with configurable parameters
        
        Parameters:
        -----------
        config: dict
            Configuration dictionary
        delta_threshold: float
            Threshold to determine significant delta (0.0 - 1.0)
        lookback_periods: int
            Number of periods for historical analysis
        absorption_threshold: float
            Threshold to identify absorption patterns (0.0 - 1.0)
        """
        super().__init__(config)
        self.logger.info(f"VolumeDeltaIndicator initialized with delta_threshold={delta_threshold}")
        self.delta_threshold = delta_threshold
        self.lookback_periods = lookback_periods
        self.absorption_threshold = absorption_threshold
        self._cached_cumulative_delta = 0.0
    
    def classify_volume(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Classify volume as buy or sell based on price action
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data with OHLCV information
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            Buy volume and sell volume series
        """
        try:
            if data.shape[0] < 2:
                return pd.Series(0, index=data.index), pd.Series(0, index=data.index)
            
            # Initialize buy and sell volumes
            buy_volume = pd.Series(0.0, index=data.index)
            sell_volume = pd.Series(0.0, index=data.index)
            
            # Skip the first bar as we need previous data
            for i in range(1, len(data)):
                close = data['close'].iloc[i]
                prev_close = data['close'].iloc[i-1]
                volume = data['volume'].iloc[i]
                
                # Classify based on close price movement
                if close > prev_close:
                    buy_volume.iloc[i] = volume
                elif close < prev_close:
                    sell_volume.iloc[i] = volume
                else:
                    # If no price change, split the volume
                    buy_volume.iloc[i] = volume / 2
                    sell_volume.iloc[i] = volume / 2
            
            return buy_volume, sell_volume
            
        except Exception as e:
            self.logger.error(f"Error classifying volume: {str(e)}")
            return pd.Series(0, index=data.index), pd.Series(0, index=data.index)
    
    def calculate_delta(self, buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
        """
        Calculate delta (buy volume - sell volume) and normalize
        
        Parameters:
        -----------
        buy_volume: pd.Series
            Series of buy volumes
        sell_volume: pd.Series
            Series of sell volumes
            
        Returns:
        --------
        pd.Series
            Delta volume normalized by total volume
        """
        try:
            # Calculate delta as (buy - sell) / (buy + sell)
            total_volume = buy_volume + sell_volume
            delta = (buy_volume - sell_volume) / total_volume.replace(0, 1)  # Avoid division by zero
            
            return delta
            
        except Exception as e:
            self.logger.error(f"Error calculating delta: {str(e)}")
            return pd.Series(0, index=buy_volume.index)
    
    def calculate_absorption_ratio(self, data: pd.DataFrame, 
                                  buy_volume: pd.Series, 
                                  sell_volume: pd.Series) -> pd.Series:
        """
        Calculate absorption ratio (volume vs price movement)
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data
        buy_volume: pd.Series
            Series of buy volumes
        sell_volume: pd.Series
            Series of sell volumes
            
        Returns:
        --------
        pd.Series
            Absorption ratio series
        """
        try:
            if data.shape[0] < 2:
                return pd.Series(0, index=data.index)
            
            # Calculate price range
            high_low_range = data['high'] - data['low']
            
            # Calculate average range for normalization
            avg_range = high_low_range.rolling(10).mean().fillna(high_low_range)
            
            # Calculate normalized price movement relative to volume
            total_volume = buy_volume + sell_volume
            normalized_price_move = high_low_range / avg_range
            
            # Calculate volume vs price movement ratio (higher = more absorption)
            vol_price_ratio = total_volume / (normalized_price_move * total_volume.mean())
            
            # Apply sigmoid function to normalize between 0 and 1
            absorption_ratio = 1 / (1 + np.exp(-0.5 * (vol_price_ratio - 1)))
            
            return absorption_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating absorption ratio: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def detect_divergences(self, data: pd.DataFrame, delta: pd.Series) -> Dict[str, bool]:
        """
        Detect divergences between price and volume delta
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data
        delta: pd.Series
            Delta volume series
            
        Returns:
        --------
        Dict[str, bool]
            Dictionary of detected divergences
        """
        try:
            result = {
                'bullish_divergence': False,
                'bearish_divergence': False
            }
            
            if data.shape[0] < self.lookback_periods:
                return result
                
            # Get the last N periods
            recent_data = data.iloc[-self.lookback_periods:]
            recent_delta = delta.iloc[-self.lookback_periods:]
            
            # Calculate local min/max
            price_min_idx = recent_data['low'].idxmin()
            price_max_idx = recent_data['high'].idxmax()
            delta_min_idx = recent_delta.idxmin()
            delta_max_idx = recent_delta.idxmax()
            
            # Check for bullish divergence
            # Price making lower lows but delta making higher lows
            if price_min_idx > delta_min_idx and recent_data['close'].iloc[-1] < recent_data['close'].iloc[0]:
                result['bullish_divergence'] = True
                
            # Check for bearish divergence
            # Price making higher highs but delta making lower highs
            if price_max_idx > delta_max_idx and recent_data['close'].iloc[-1] > recent_data['close'].iloc[0]:
                result['bearish_divergence'] = True
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting divergences: {str(e)}")
            return {'bullish_divergence': False, 'bearish_divergence': False}
    
    def calculate(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> VolumeDeltaSignal:
        """
        Calculate volume delta metrics from market data
        
        Parameters:
        -----------
        data: Union[pd.DataFrame, Dict[str, Any]]
            Market data with required columns (OHLCV)
            
        Returns:
        --------
        VolumeDeltaSignal
            Comprehensive volume delta analysis
        """
        try:
            # Convert dict to DataFrame if necessary
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            elif isinstance(data, list):
                # Handle list input by converting to DataFrame
                data = pd.DataFrame(data)
            elif not isinstance(data, pd.DataFrame):
                # Convert any other type to DataFrame
                data = pd.DataFrame(data)
                
            if data.shape[0] < 2:
                raise ValueError("Insufficient data: need at least 2 periods")
                
            # Check required fields
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            if not all(field in data.columns for field in required_fields):
                raise ValueError(f"Missing required fields in data: {required_fields}")
                
            # Classify volume as buy or sell
            buy_volume, sell_volume = self.classify_volume(data)
            
            # Calculate delta
            delta = self.calculate_delta(buy_volume, sell_volume)
            
            # Calculate cumulative delta
            cumulative_delta = delta.cumsum()
            
            # Calculate absorption ratio
            absorption = self.calculate_absorption_ratio(data, buy_volume, sell_volume)
            
            # Get latest values
            latest_buy = buy_volume.iloc[-1]
            latest_sell = sell_volume.iloc[-1]
            latest_delta = delta.iloc[-1]
            latest_cum_delta = cumulative_delta.iloc[-1]
            latest_absorption = absorption.iloc[-1]
            
            # Update cached cumulative delta
            self._cached_cumulative_delta = latest_cum_delta
            
            # Determine pressure direction
            if latest_delta > self.delta_threshold:
                pressure = "buying"
            elif latest_delta < -self.delta_threshold:
                pressure = "selling"
            else:
                pressure = "neutral"
                
            # Check for divergences
            divergences = self.detect_divergences(data, delta)
            
            # Create signal with buy/sell recommendation
            signal_direction = "neutral"
            signal_strength = 0.0
            
            # Generate signal based on volume delta, absorption, and divergences
            if pressure == "buying" or divergences['bullish_divergence']:
                if latest_absorption > self.absorption_threshold:
                    # Strong buying with high absorption is bullish
                    signal_direction = "buy"
                    signal_strength = min(1.0, (abs(latest_delta) * 0.5) + (latest_absorption * 0.5))
                    if divergences['bullish_divergence']:
                        signal_strength += 0.2
            elif pressure == "selling" or divergences['bearish_divergence']:
                if latest_absorption > self.absorption_threshold:
                    # Strong selling with high absorption is bearish
                    signal_direction = "sell"
                    signal_strength = min(1.0, (abs(latest_delta) * 0.5) + (latest_absorption * 0.5))
                    if divergences['bearish_divergence']:
                        signal_strength += 0.2
            
            # Cap signal strength at 1.0
            signal_strength = min(1.0, signal_strength)
                
            # Create and return the volume delta signal
            return VolumeDeltaSignal(
                timestamp=data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now(),
                indicator_name="VolumeDeltaIndicator",
                signal_type=signal_direction,
                confidence=signal_strength,  # Add confidence field
                buy_volume=latest_buy,
                sell_volume=latest_sell,
                delta=latest_delta,
                cumulative_delta=latest_cum_delta,
                absorption_ratio=latest_absorption,
                pressure_direction=pressure,
                strength=signal_strength
            )
        
        except Exception as e:
            self.logger.error(f"Error in VolumeDeltaIndicator calculation: {str(e)}")
            raise ServiceError(f"Calculation failed: {str(e)}")
    
    def generate_signal(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate trading signals based on volume delta analysis
        
        Parameters:
        -----------
        data: Union[pd.DataFrame, Dict[str, Any]]
            Market data for analysis
            
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
                "buy_volume": signal.buy_volume,
                "sell_volume": signal.sell_volume,
                "delta": signal.delta,
                "cumulative_delta": signal.cumulative_delta,
                "absorption_ratio": signal.absorption_ratio,
                "pressure": signal.pressure_direction
            }
        }