# -*- coding: utf-8 -*-
"""
Trend Strength Indicator

Measures the strength and persistence of market trends using multiple
momentum and volatility factors. Provides comprehensive trend analysis
with adaptive thresholds and multi-timeframe strength assessment.

Key Features:
- Trend strength measurement (0-100 scale)
- Trend persistence analysis
- Momentum acceleration detection
- Volatility-adjusted strength
- Adaptive threshold system
- Multi-factor trend scoring

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

class TrendStrengthSignal(Enum):
    """Trend strength signal types"""
    VERY_STRONG_BULLISH = "very_strong_bullish"
    STRONG_BULLISH = "strong_bullish"
    MODERATE_BULLISH = "moderate_bullish"
    WEAK_BULLISH = "weak_bullish"
    NEUTRAL = "neutral"
    WEAK_BEARISH = "weak_bearish"
    MODERATE_BEARISH = "moderate_bearish"
    STRONG_BEARISH = "strong_bearish"
    VERY_STRONG_BEARISH = "very_strong_bearish"

@dataclass
class TrendStrengthResult:
    """Trend strength calculation result"""
    strength: float  # Overall trend strength (0-100)
    direction: str  # 'bullish', 'bearish', 'neutral'
    persistence: float  # Trend persistence score (0-100)
    momentum: float  # Momentum strength (-100 to +100)
    volatility_factor: float  # Volatility adjustment factor
    signal: TrendStrengthSignal
    confidence: float  # Signal confidence (0-100)

class TrendStrengthIndicator(StandardIndicatorInterface):
    """
    Trend Strength Indicator
    
    Comprehensive trend strength analysis combining momentum, persistence,
    and volatility factors for robust trend assessment.
    """
    
    CATEGORY = "core_trend"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, **kwargs):
        """Initialize Trend Strength Indicator"""
        super().__init__(**kwargs)
        self._setup_defaults()
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata"""
        return {
            'name': 'TrendStrengthIndicator', 
            'category': self.CATEGORY,
            'version': self.VERSION,
            'author': self.AUTHOR,
            'description': 'Measures trend strength using momentum and persistence factors',
            'parameters': list(self.parameters.keys()),
            'required_columns': self._get_required_columns(),
            'minimum_data_points': self._get_minimum_data_points()
        }
        
    def _setup_defaults(self):
        """Setup default parameters"""
        default_params = {
            'period': 20,           # Main calculation period
            'momentum_period': 14,  # Momentum calculation period
            'volatility_period': 10, # Volatility calculation period
            'persistence_period': 30, # Persistence analysis period
            'strong_threshold': 70,  # Strong trend threshold
            'weak_threshold': 30,   # Weak trend threshold
            'smoothing_factor': 0.3  # EMA smoothing factor
        }
        
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
                
        self.validate_parameters()
        
    def validate_parameters(self) -> None:
        """Validate input parameters"""
        period = self.parameters.get('period', 20)
        momentum_period = self.parameters.get('momentum_period', 14)
        volatility_period = self.parameters.get('volatility_period', 10)
        
        if not isinstance(period, int) or period < 2:
            raise IndicatorValidationError("period must be an integer >= 2")
            
        if not isinstance(momentum_period, int) or momentum_period < 2:
            raise IndicatorValidationError("momentum_period must be an integer >= 2")
            
        if not isinstance(volatility_period, int) or volatility_period < 2:
            raise IndicatorValidationError("volatility_period must be an integer >= 2")
            
        smoothing_factor = self.parameters.get('smoothing_factor', 0.3)
        if not isinstance(smoothing_factor, (int, float)) or not (0 < smoothing_factor <= 1):
            raise IndicatorValidationError("smoothing_factor must be between 0 and 1")
            
    def _get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return ['high', 'low', 'close', 'volume']
        
    def _get_minimum_data_points(self) -> int:
        """Get minimum required data points"""
        return max(
            self.parameters.get('period', 20),
            self.parameters.get('persistence_period', 30),
            self.parameters.get('momentum_period', 14)
        ) + 10
        
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate Trend Strength indicator
        
        Args:
            data: OHLC data with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Series of TrendStrengthResult objects
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
                if 'volume' not in df.columns:
                    df['volume'] = pd.Series([1000] * len(data), index=data.index)
            else:
                df = data.copy()
                
            # Validate required columns
            required_cols = self._get_required_columns()
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise IndicatorValidationError(f"Missing required columns: {missing_cols}")
            
            # Calculate trend strength components
            momentum_scores = self._calculate_momentum_strength(df)
            persistence_scores = self._calculate_trend_persistence(df)
            volatility_factors = self._calculate_volatility_adjustment(df)
            direction_scores = self._calculate_direction_strength(df)
            
            # Combine into overall strength
            results = []
            for i in range(len(df)):
                if i < self._get_minimum_data_points() - 10:
                    result = TrendStrengthResult(
                        strength=50.0,
                        direction='neutral',
                        persistence=50.0,
                        momentum=0.0,
                        volatility_factor=1.0,
                        signal=TrendStrengthSignal.NEUTRAL,
                        confidence=0.0
                    )
                else:
                    result = self._calculate_strength_metrics(
                        momentum_scores[i],
                        persistence_scores[i],
                        volatility_factors[i],
                        direction_scores[i],
                        i, df
                    )
                    
                results.append(result)
                
            # Store calculation details for debugging
            self._last_calculation = {
                'momentum_scores': momentum_scores,
                'persistence_scores': persistence_scores,
                'volatility_factors': volatility_factors,
                'direction_scores': direction_scores
            }
            
            return pd.Series(results, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating Trend Strength: {e}")
            # Use the original data length for error results  
            data_len = len(data) if hasattr(data, '__len__') else 1
            return pd.Series([TrendStrengthResult(
                strength=50.0,
                direction='error',
                persistence=50.0,
                momentum=0.0,
                volatility_factor=1.0,
                signal=TrendStrengthSignal.NEUTRAL,
                confidence=0.0
            )] * data_len, index=data.index if hasattr(data, 'index') else range(data_len))
            
    def _calculate_momentum_strength(self, df: pd.DataFrame) -> List[float]:
        """Calculate momentum-based strength scores"""
        momentum_period = self.parameters.get('momentum_period')
        
        # ROC (Rate of Change)
        roc = df['close'].pct_change(momentum_period) * 100
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=momentum_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=momentum_period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # MACD momentum
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_momentum = macd - macd_signal
        
        # Combine momentum indicators
        momentum_scores = []
        for i in range(len(df)):
            if pd.isna(roc.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(macd_momentum.iloc[i]):
                momentum_scores.append(0.0)
            else:
                # Normalize ROC to 0-100 scale
                roc_score = np.tanh(roc.iloc[i] / 10) * 50 + 50
                
                # RSI already 0-100
                rsi_score = rsi.iloc[i]
                
                # MACD momentum normalized
                macd_score = np.tanh(macd_momentum.iloc[i] / df['close'].iloc[i] * 1000) * 50 + 50
                
                # Weighted average
                combined_score = (roc_score * 0.4 + rsi_score * 0.4 + macd_score * 0.2)
                momentum_scores.append(combined_score)
                
        return momentum_scores
        
    def _calculate_trend_persistence(self, df: pd.DataFrame) -> List[float]:
        """Calculate trend persistence scores"""
        persistence_period = self.parameters.get('persistence_period')
        
        # Calculate price direction consistency
        price_changes = df['close'].diff().fillna(0)
        
        persistence_scores = []
        for i in range(len(df)):
            if i < persistence_period:
                persistence_scores.append(50.0)
            else:
                # Look at recent price changes
                recent_changes = price_changes.iloc[i-persistence_period+1:i+1]
                
                # Count directional consistency
                positive_changes = (recent_changes > 0).sum()
                negative_changes = (recent_changes < 0).sum()
                
                # Calculate persistence based on directional dominance
                total_changes = len(recent_changes)
                if total_changes > 0:
                    max_direction = max(positive_changes, negative_changes)
                    persistence = (max_direction / total_changes) * 100
                else:
                    persistence = 50.0
                    
                persistence_scores.append(persistence)
                
        return persistence_scores
        
    def _calculate_volatility_adjustment(self, df: pd.DataFrame) -> List[float]:
        """Calculate volatility adjustment factors"""
        volatility_period = self.parameters.get('volatility_period')
        
        # Calculate Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=volatility_period).mean()
        
        # Calculate normalized volatility
        price_volatility = (atr / df['close']) * 100
        
        # Convert to adjustment factor (lower volatility = higher confidence)
        volatility_factors = []
        for i in range(len(df)):
            if pd.isna(price_volatility.iloc[i]):
                volatility_factors.append(1.0)
            else:
                # Invert volatility for adjustment (high vol = low factor)
                vol_factor = 1.0 / (1.0 + price_volatility.iloc[i] / 10)
                volatility_factors.append(vol_factor)
                
        return volatility_factors
        
    def _calculate_direction_strength(self, df: pd.DataFrame) -> List[float]:
        """Calculate directional strength scores"""
        period = self.parameters.get('period')
        
        # ADX-like calculation for trend direction strength
        high_low = df['high'] - df['low']
        high_close = df['high'] - df['close'].shift()
        low_close = df['close'].shift() - df['low']
        
        plus_dm = np.where((high_close > low_close) & (high_close > 0), high_close, 0)
        minus_dm = np.where((low_close > high_close) & (low_close > 0), low_close, 0)
        
        plus_dm_series = pd.Series(plus_dm, index=df.index).rolling(window=period).mean()
        minus_dm_series = pd.Series(minus_dm, index=df.index).rolling(window=period).mean()
        
        direction_scores = []
        for i in range(len(df)):
            if pd.isna(plus_dm_series.iloc[i]) or pd.isna(minus_dm_series.iloc[i]):
                direction_scores.append(0.0)
            else:
                plus_di = plus_dm_series.iloc[i]
                minus_di = minus_dm_series.iloc[i]
                
                # Calculate directional movement index
                if plus_di + minus_di != 0:
                    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
                    # Determine direction
                    if plus_di > minus_di:
                        direction_scores.append(dx)  # Positive for bullish
                    else:
                        direction_scores.append(-dx)  # Negative for bearish
                else:
                    direction_scores.append(0.0)
                    
        return direction_scores
        
    def _calculate_strength_metrics(self, momentum: float, persistence: float,
                                  volatility_factor: float, direction: float,
                                  index: int, df: pd.DataFrame) -> TrendStrengthResult:
        """Calculate final strength metrics"""
        
        # Combine all factors into overall strength
        base_strength = (momentum * 0.4 + persistence * 0.3 + abs(direction) * 0.3)
        
        # Apply volatility adjustment
        adjusted_strength = base_strength * volatility_factor
        
        # Determine direction
        if direction > 10:
            trend_direction = 'bullish'
        elif direction < -10:
            trend_direction = 'bearish'
        else:
            trend_direction = 'neutral'
            
        # Generate signal
        signal = self._generate_strength_signal(adjusted_strength, direction)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            adjusted_strength, persistence, volatility_factor, abs(direction)
        )
        
        return TrendStrengthResult(
            strength=adjusted_strength,
            direction=trend_direction,
            persistence=persistence,
            momentum=direction,
            volatility_factor=volatility_factor,
            signal=signal,
            confidence=confidence
        )
        
    def _generate_strength_signal(self, strength: float, direction: float) -> TrendStrengthSignal:
        """Generate signals based on trend strength and direction"""
        
        strong_threshold = self.parameters.get('strong_threshold')
        weak_threshold = self.parameters.get('weak_threshold')
        
        if direction > 0:  # Bullish direction
            if strength >= 85:
                return TrendStrengthSignal.VERY_STRONG_BULLISH
            elif strength >= strong_threshold:
                return TrendStrengthSignal.STRONG_BULLISH
            elif strength >= 50:
                return TrendStrengthSignal.MODERATE_BULLISH
            elif strength >= weak_threshold:
                return TrendStrengthSignal.WEAK_BULLISH
            else:
                return TrendStrengthSignal.NEUTRAL
                
        elif direction < 0:  # Bearish direction
            if strength >= 85:
                return TrendStrengthSignal.VERY_STRONG_BEARISH
            elif strength >= strong_threshold:
                return TrendStrengthSignal.STRONG_BEARISH
            elif strength >= 50:
                return TrendStrengthSignal.MODERATE_BEARISH
            elif strength >= weak_threshold:
                return TrendStrengthSignal.WEAK_BEARISH
            else:
                return TrendStrengthSignal.NEUTRAL
                
        else:  # Neutral direction
            return TrendStrengthSignal.NEUTRAL
            
    def _calculate_confidence(self, strength: float, persistence: float,
                            volatility_factor: float, direction_strength: float) -> float:
        """Calculate signal confidence"""
        
        confidence = 50.0  # Base confidence
        
        # Strength factor
        if strength > 80:
            confidence += 25.0
        elif strength > 60:
            confidence += 15.0
        elif strength < 30:
            confidence -= 15.0
            
        # Persistence factor
        if persistence > 80:
            confidence += 20.0
        elif persistence > 60:
            confidence += 10.0
        elif persistence < 40:
            confidence -= 10.0
            
        # Volatility factor
        if volatility_factor > 0.8:
            confidence += 15.0
        elif volatility_factor > 0.6:
            confidence += 10.0
        elif volatility_factor < 0.4:
            confidence -= 15.0
            
        # Direction strength factor
        if direction_strength > 50:
            confidence += 10.0
        elif direction_strength < 20:
            confidence -= 10.0
            
        return max(0.0, min(100.0, confidence))

def get_indicator_class():
    """Export function for indicator registry"""
    return TrendStrengthIndicator