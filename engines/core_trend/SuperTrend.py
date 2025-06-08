# -*- coding: utf-8 -*-
"""
SuperTrend Indicator

SuperTrend is a trend-following indicator that uses Average True Range (ATR)
to calculate dynamic support and resistance levels. It helps identify trend 
direction and potential reversal points.

Key Features:
- Dynamic support/resistance calculation
- Trend direction identification
- Buy/sell signal generation
- ATR-based volatility adjustment
- Multi-timeframe compatibility

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SuperTrendSignal(Enum):
    """SuperTrend signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD_LONG = "hold_long"
    HOLD_SHORT = "hold_short"
    NEUTRAL = "neutral"

class TrendDirection(Enum):
    """Trend direction"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"

@dataclass
class SuperTrendResult:
    """SuperTrend analysis result"""
    supertrend_value: float
    trend_direction: TrendDirection
    signal: SuperTrendSignal
    atr_value: float
    upper_band: float
    lower_band: float
    strength: float

class SuperTrendData(NamedTuple):
    """SuperTrend data structure"""
    supertrend: np.ndarray
    trend_direction: np.ndarray
    upper_band: np.ndarray
    lower_band: np.ndarray
    atr: np.ndarray

class SuperTrend:
    """
    SuperTrend Indicator Implementation
    
    SuperTrend is calculated using Average True Range (ATR) and a multiplier
    to create dynamic support and resistance levels.
    
    Formula:
    - Basic Upper Band = (High + Low) / 2 + (Multiplier × ATR)
    - Basic Lower Band = (High + Low) / 2 - (Multiplier × ATR)
    - Final Upper Band = Basic UB < Previous Final UB or Previous Close > Previous Final UB ? Basic UB : Previous Final UB
    - Final Lower Band = Basic LB > Previous Final LB or Previous Close < Previous Final LB ? Basic LB : Previous Final LB
    - SuperTrend = Final UB if trend is down, Final LB if trend is up
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        Initialize SuperTrend indicator
        
        Args:
            period: Period for ATR calculation (default 10)
            multiplier: ATR multiplier for band calculation (default 3.0)
        """
        self.period = period
        self.multiplier = multiplier
        
        logger.info(f"✅ SuperTrend initialized (period={period}, multiplier={multiplier})")

    def calculate_atr(self, high: Union[np.ndarray, pd.Series],
                     low: Union[np.ndarray, pd.Series],
                     close: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Calculate Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Array of ATR values
        """
        try:
            if isinstance(high, pd.Series):
                high = high.values
            if isinstance(low, pd.Series):
                low = low.values
            if isinstance(close, pd.Series):
                close = close.values
            
            # Calculate True Range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            # Set first values (no previous close)
            tr2[0] = tr1[0]
            tr3[0] = tr1[0]
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate ATR using Simple Moving Average
            atr = pd.Series(true_range).rolling(window=self.period, min_periods=1).mean().values
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return np.full(len(high), np.nan)

    def calculate_supertrend(self, high: Union[np.ndarray, pd.Series],
                           low: Union[np.ndarray, pd.Series],
                           close: Union[np.ndarray, pd.Series]) -> SuperTrendData:
        """
        Calculate SuperTrend indicator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            SuperTrendData with all calculated values
        """
        try:
            if isinstance(high, pd.Series):
                high = high.values
            if isinstance(low, pd.Series):
                low = low.values
            if isinstance(close, pd.Series):
                close = close.values
            
            length = len(close)
            
            # Calculate ATR
            atr = self.calculate_atr(high, low, close)
            
            # Calculate median price (HL2)
            median_price = (high + low) / 2
            
            # Calculate basic bands
            basic_upper_band = median_price + (self.multiplier * atr)
            basic_lower_band = median_price - (self.multiplier * atr)
            
            # Initialize final bands and supertrend
            final_upper_band = np.full(length, np.nan)
            final_lower_band = np.full(length, np.nan)
            supertrend = np.full(length, np.nan)
            trend_direction = np.full(length, 0)  # 1 for uptrend, -1 for downtrend
            
            # Calculate final bands
            for i in range(length):
                if i == 0:
                    final_upper_band[i] = basic_upper_band[i]
                    final_lower_band[i] = basic_lower_band[i]
                else:
                    # Final Upper Band
                    if (basic_upper_band[i] < final_upper_band[i-1] or 
                        close[i-1] > final_upper_band[i-1]):
                        final_upper_band[i] = basic_upper_band[i]
                    else:
                        final_upper_band[i] = final_upper_band[i-1]
                    
                    # Final Lower Band
                    if (basic_lower_band[i] > final_lower_band[i-1] or 
                        close[i-1] < final_lower_band[i-1]):
                        final_lower_band[i] = basic_lower_band[i]
                    else:
                        final_lower_band[i] = final_lower_band[i-1]
            
            # Calculate SuperTrend and trend direction
            for i in range(length):
                if i == 0:
                    supertrend[i] = final_lower_band[i]
                    trend_direction[i] = 1
                else:
                    # Determine trend direction
                    if (close[i] <= final_lower_band[i]):
                        trend_direction[i] = -1  # Downtrend
                        supertrend[i] = final_upper_band[i]
                    elif (close[i] >= final_upper_band[i]):
                        trend_direction[i] = 1   # Uptrend
                        supertrend[i] = final_lower_band[i]
                    else:
                        # Continue previous trend
                        trend_direction[i] = trend_direction[i-1]
                        if trend_direction[i] == 1:
                            supertrend[i] = final_lower_band[i]
                        else:
                            supertrend[i] = final_upper_band[i]
            
            return SuperTrendData(
                supertrend=supertrend,
                trend_direction=trend_direction,
                upper_band=final_upper_band,
                lower_band=final_lower_band,
                atr=atr
            )
            
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {e}")
            return SuperTrendData(
                supertrend=np.full(len(close), np.nan),
                trend_direction=np.full(len(close), 0),
                upper_band=np.full(len(close), np.nan),
                lower_band=np.full(len(close), np.nan),
                atr=np.full(len(close), np.nan)
            )

    def analyze(self, high: Union[np.ndarray, pd.Series],
               low: Union[np.ndarray, pd.Series],
               close: Union[np.ndarray, pd.Series]) -> SuperTrendResult:
        """
        Analyze SuperTrend and generate signals
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            SuperTrendResult with analysis
        """
        try:
            # Calculate SuperTrend
            st_data = self.calculate_supertrend(high, low, close)
            
            if len(st_data.supertrend) < 2:
                return SuperTrendResult(
                    supertrend_value=np.nan,
                    trend_direction=TrendDirection.SIDEWAYS,
                    signal=SuperTrendSignal.NEUTRAL,
                    atr_value=np.nan,
                    upper_band=np.nan,
                    lower_band=np.nan,
                    strength=0.0
                )
            
            # Get current values
            current_close = close[-1] if isinstance(close, np.ndarray) else close.iloc[-1]
            current_supertrend = st_data.supertrend[-1]
            current_trend = st_data.trend_direction[-1]
            prev_trend = st_data.trend_direction[-2] if len(st_data.trend_direction) > 1 else current_trend
            
            # Determine trend direction
            if current_trend == 1:
                trend_dir = TrendDirection.UPTREND
            elif current_trend == -1:
                trend_dir = TrendDirection.DOWNTREND
            else:
                trend_dir = TrendDirection.SIDEWAYS
            
            # Generate signals
            signal = SuperTrendSignal.NEUTRAL
            if current_trend == 1 and prev_trend == -1:
                signal = SuperTrendSignal.BUY
            elif current_trend == -1 and prev_trend == 1:
                signal = SuperTrendSignal.SELL
            elif current_trend == 1:
                signal = SuperTrendSignal.HOLD_LONG
            elif current_trend == -1:
                signal = SuperTrendSignal.HOLD_SHORT
            
            # Calculate strength based on distance from SuperTrend
            strength = abs(current_close - current_supertrend) / current_supertrend * 100
            
            return SuperTrendResult(
                supertrend_value=current_supertrend,
                trend_direction=trend_dir,
                signal=signal,
                atr_value=st_data.atr[-1],
                upper_band=st_data.upper_band[-1],
                lower_band=st_data.lower_band[-1],
                strength=min(strength, 100.0)  # Cap at 100%
            )
            
        except Exception as e:
            logger.error(f"Error in SuperTrend analysis: {e}")
            return SuperTrendResult(
                supertrend_value=np.nan,
                trend_direction=TrendDirection.SIDEWAYS,
                signal=SuperTrendSignal.NEUTRAL,
                atr_value=np.nan,
                upper_band=np.nan,
                lower_band=np.nan,
                strength=0.0
            )

    def get_signals(self, high: Union[np.ndarray, pd.Series],
                   low: Union[np.ndarray, pd.Series],
                   close: Union[np.ndarray, pd.Series]) -> List[Dict]:
        """
        Get SuperTrend signals for the entire data series
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            List of signal dictionaries
        """
        try:
            st_data = self.calculate_supertrend(high, low, close)
            signals = []
            
            for i in range(1, len(st_data.trend_direction)):
                current_trend = st_data.trend_direction[i]
                prev_trend = st_data.trend_direction[i-1]
                
                if current_trend != prev_trend:
                    if current_trend == 1:
                        signals.append({
                            'index': i,
                            'signal': SuperTrendSignal.BUY.value,
                            'price': close[i] if isinstance(close, np.ndarray) else close.iloc[i],
                            'supertrend': st_data.supertrend[i],
                            'trend': 'uptrend'
                        })
                    elif current_trend == -1:
                        signals.append({
                            'index': i,
                            'signal': SuperTrendSignal.SELL.value,
                            'price': close[i] if isinstance(close, np.ndarray) else close.iloc[i],
                            'supertrend': st_data.supertrend[i],
                            'trend': 'downtrend'
                        })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting SuperTrend signals: {e}")
            return []
