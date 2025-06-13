# -*- coding: utf-8 -*-
"""
Cloud Position Indicator

Analyzes the relative position of price in relation to cloud formations,
particularly useful for Ichimoku analysis and trend continuation patterns.
Provides clear signals for cloud breakouts and position strength.

Key Features:
- Cloud position analysis (above/below/inside)
- Position strength measurement
- Cloud thickness analysis for volatility
- Breakout signal detection
- Trend continuation confirmation
- Multiple timeframe support

Author: Platform3 Analytics Team
Version: 1.0.0
Category: core_trend
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ..ai_enhancement.indicators.base_indicator import StandardIndicatorInterface, IndicatorValidationError

logger = logging.getLogger(__name__)

class CloudPositionSignal(Enum):
    """Cloud position signal types"""
    BULLISH_BREAKOUT = "bullish_breakout"
    BEARISH_BREAKOUT = "bearish_breakout"
    CLOUD_SUPPORT = "cloud_support"
    CLOUD_RESISTANCE = "cloud_resistance"
    INSIDE_CLOUD = "inside_cloud"
    STRONG_ABOVE = "strong_above"
    STRONG_BELOW = "strong_below"
    NEUTRAL = "neutral"

@dataclass
class CloudPositionResult:
    """Cloud position calculation result"""
    position: str  # 'above', 'below', 'inside'
    strength: float  # Position strength (0-100)
    cloud_thickness: float  # Cloud thickness percentage
    signal: CloudPositionSignal
    confidence: float  # Signal confidence (0-100)
    distance_to_cloud: float  # Distance from price to cloud

class CloudPositionIndicator(StandardIndicatorInterface):
    """
    Cloud Position Indicator
    
    Analyzes price position relative to cloud formations for trend analysis.
    Provides comprehensive cloud-based trading signals and position strength.
    """
    
    CATEGORY = "core_trend"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, **kwargs):
        """Initialize Cloud Position Indicator"""
        super().__init__(**kwargs)
        self._setup_defaults()
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata"""
        return {
            'name': 'CloudPositionIndicator',
            'category': self.CATEGORY,
            'version': self.VERSION,
            'author': self.AUTHOR,
            'description': 'Analyzes price position relative to cloud formations',
            'parameters': list(self.parameters.keys()),
            'required_columns': self._get_required_columns(),
            'minimum_data_points': self._get_minimum_data_points()
        }
        
    def _setup_defaults(self):
        """Setup default parameters"""
        default_params = {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_span_b_period': 52,
            'chikou_span': 26,
            'displacement': 26,
            'position_threshold': 2.0,  # % threshold for strong position
            'breakout_threshold': 1.5   # % threshold for breakout signals
        }
        
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
                
        self.validate_parameters()
        
    def validate_parameters(self) -> None:
        """Validate input parameters"""
        tenkan_period = self.parameters.get('tenkan_period', 9)
        kijun_period = self.parameters.get('kijun_period', 26)
        senkou_span_b_period = self.parameters.get('senkou_span_b_period', 52)
        
        if not isinstance(tenkan_period, int) or tenkan_period < 1:
            raise IndicatorValidationError("tenkan_period must be a positive integer")
            
        if not isinstance(kijun_period, int) or kijun_period < 1:
            raise IndicatorValidationError("kijun_period must be a positive integer")
            
        if not isinstance(senkou_span_b_period, int) or senkou_span_b_period < 1:
            raise IndicatorValidationError("senkou_span_b_period must be a positive integer")
            
        if tenkan_period >= kijun_period:
            raise IndicatorValidationError("tenkan_period must be less than kijun_period")
            
    def _get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return ['high', 'low', 'close']
        
    def _get_minimum_data_points(self) -> int:
        """Get minimum required data points"""
        return max(
            self.parameters.get('senkou_span_b_period', 52),
            self.parameters.get('displacement', 26)
        ) + self.parameters.get('displacement', 26)
        
    def _calculate_ichimoku_lines(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Ichimoku cloud lines"""
        high = data['high']
        low = data['low']
        
        tenkan_period = self.parameters.get('tenkan_period')
        kijun_period = self.parameters.get('kijun_period')
        senkou_span_b_period = self.parameters.get('senkou_span_b_period')
        displacement = self.parameters.get('displacement')
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=senkou_span_b_period).max()
        senkou_low = low.rolling(window=senkou_span_b_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b
        }
        
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate Cloud Position indicator
        
        Args:
            data: OHLC data with columns ['high', 'low', 'close']
            
        Returns:
            Series of CloudPositionResult objects
        """
        try:
            # Convert and validate input data
            if isinstance(data, pd.Series):
                # Convert Series to DataFrame
                df = pd.DataFrame({'close': data})
                if 'high' not in df.columns:
                    df['high'] = data
                if 'low' not in df.columns:
                    df['low'] = data
            else:
                df = data.copy()
                
            # Validate required columns
            required_cols = self._get_required_columns()
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise IndicatorValidationError(f"Missing required columns: {missing_cols}")
            
            # Calculate Ichimoku lines
            lines = self._calculate_ichimoku_lines(df)
            
            # Calculate cloud boundaries
            cloud_top = np.maximum(lines['senkou_span_a'], lines['senkou_span_b'])
            cloud_bottom = np.minimum(lines['senkou_span_a'], lines['senkou_span_b'])
            
            # Calculate cloud thickness
            cloud_thickness = ((cloud_top - cloud_bottom) / df['close'] * 100).fillna(0)
            
            # Calculate position metrics
            results = []
            for i in range(len(df)):
                if pd.isna(cloud_top.iloc[i]) or pd.isna(cloud_bottom.iloc[i]):
                    result = CloudPositionResult(
                        position='unknown',
                        strength=0.0,
                        cloud_thickness=0.0,
                        signal=CloudPositionSignal.NEUTRAL,
                        confidence=0.0,
                        distance_to_cloud=0.0
                    )
                else:
                    result = self._calculate_position_metrics(
                        df['close'].iloc[i],
                        cloud_top.iloc[i],
                        cloud_bottom.iloc[i],
                        cloud_thickness.iloc[i],
                        i, df, lines
                    )
                    
                results.append(result)
                
            # Store calculation details for debugging
            self._last_calculation = {
                'cloud_top': cloud_top,
                'cloud_bottom': cloud_bottom,
                'cloud_thickness': cloud_thickness,
                'ichimoku_lines': lines
            }
            
            return pd.Series(results, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating Cloud Position: {e}")
            # Use the original data length for error results
            data_len = len(data) if hasattr(data, '__len__') else 1
            return pd.Series([CloudPositionResult(
                position='error',
                strength=0.0,
                cloud_thickness=0.0,
                signal=CloudPositionSignal.NEUTRAL,
                confidence=0.0,
                distance_to_cloud=0.0
            )] * data_len, index=data.index if hasattr(data, 'index') else range(data_len))
            
    def _calculate_position_metrics(self, price: float, cloud_top: float, 
                                  cloud_bottom: float, thickness: float,
                                  index: int, df: pd.DataFrame, 
                                  lines: Dict[str, pd.Series]) -> CloudPositionResult:
        """Calculate position metrics for a single data point"""
        
        position_threshold = self.parameters.get('position_threshold')
        breakout_threshold = self.parameters.get('breakout_threshold')
        
        # Determine position
        if price > cloud_top:
            position = 'above'
            distance = (price - cloud_top) / price * 100
        elif price < cloud_bottom:
            position = 'below'
            distance = (cloud_bottom - price) / price * 100
        else:
            position = 'inside'
            distance = 0.0
            
        # Calculate strength
        if position == 'above':
            strength = min(100.0, distance * 10)  # Scale distance to strength
        elif position == 'below':
            strength = min(100.0, distance * 10)
        else:
            # Inside cloud - strength based on position within cloud
            cloud_range = cloud_top - cloud_bottom
            if cloud_range > 0:
                relative_position = (price - cloud_bottom) / cloud_range
                strength = 50.0 - abs(relative_position - 0.5) * 100
            else:
                strength = 25.0
                
        # Generate signals
        signal = self._generate_signal(
            position, distance, thickness, index, df, lines
        )
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            position, distance, thickness, strength, signal
        )
        
        return CloudPositionResult(
            position=position,
            strength=strength,
            cloud_thickness=thickness,
            signal=signal,
            confidence=confidence,
            distance_to_cloud=distance
        )
        
    def _generate_signal(self, position: str, distance: float, thickness: float,
                        index: int, df: pd.DataFrame, 
                        lines: Dict[str, pd.Series]) -> CloudPositionSignal:
        """Generate trading signals based on cloud position"""
        
        breakout_threshold = self.parameters.get('breakout_threshold')
        position_threshold = self.parameters.get('position_threshold')
        
        # Check for recent breakouts (within last 3 periods)
        if index >= 3:
            recent_positions = []
            for i in range(max(0, index-3), index):
                if i < len(df):
                    price = df['close'].iloc[i]
                    top = max(lines['senkou_span_a'].iloc[i], lines['senkou_span_b'].iloc[i])
                    bottom = min(lines['senkou_span_a'].iloc[i], lines['senkou_span_b'].iloc[i])
                    
                    if not (pd.isna(top) or pd.isna(bottom)):
                        if price > top:
                            recent_positions.append('above')
                        elif price < bottom:
                            recent_positions.append('below')
                        else:
                            recent_positions.append('inside')
                            
            # Detect breakouts
            if position == 'above' and 'inside' in recent_positions[-2:]:
                if distance >= breakout_threshold:
                    return CloudPositionSignal.BULLISH_BREAKOUT
                    
            if position == 'below' and 'inside' in recent_positions[-2:]:
                if distance >= breakout_threshold:
                    return CloudPositionSignal.BEARISH_BREAKOUT
                    
        # Position-based signals
        if position == 'above':
            if distance >= position_threshold:
                return CloudPositionSignal.STRONG_ABOVE
            else:
                return CloudPositionSignal.CLOUD_SUPPORT
                
        elif position == 'below':
            if distance >= position_threshold:
                return CloudPositionSignal.STRONG_BELOW
            else:
                return CloudPositionSignal.CLOUD_RESISTANCE
                
        else:  # inside cloud
            return CloudPositionSignal.INSIDE_CLOUD
            
    def _calculate_confidence(self, position: str, distance: float, 
                            thickness: float, strength: float, 
                            signal: CloudPositionSignal) -> float:
        """Calculate signal confidence based on multiple factors"""
        
        confidence = 50.0  # Base confidence
        
        # Distance factor
        if distance > 2.0:
            confidence += 20.0
        elif distance > 1.0:
            confidence += 10.0
            
        # Cloud thickness factor (thicker cloud = more reliable)
        if thickness > 3.0:
            confidence += 15.0
        elif thickness > 1.5:
            confidence += 10.0
        elif thickness < 0.5:
            confidence -= 10.0
            
        # Signal type factor
        if signal in [CloudPositionSignal.BULLISH_BREAKOUT, CloudPositionSignal.BEARISH_BREAKOUT]:
            confidence += 15.0
        elif signal in [CloudPositionSignal.STRONG_ABOVE, CloudPositionSignal.STRONG_BELOW]:
            confidence += 10.0
        elif signal == CloudPositionSignal.INSIDE_CLOUD:
            confidence -= 20.0
            
        return max(0.0, min(100.0, confidence))

def get_indicator_class():
    """Export function for indicator registry"""
    return CloudPositionIndicator