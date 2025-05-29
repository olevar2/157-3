"""
Simple Moving Average (SMA) and Exponential Moving Average (EMA) Indicators

Moving averages are fundamental trend-following indicators that smooth price data
to identify the direction of the trend. SMA gives equal weight to all prices in
the period, while EMA gives more weight to recent prices.

Key Features:
- Simple Moving Average (SMA) calculation
- Exponential Moving Average (EMA) calculation
- Multiple moving average crossover signals
- Trend direction identification
- Support and resistance level detection
- Price-to-MA relationship analysis

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

class MASignal(Enum):
    """Moving Average signal types"""
    BULLISH_CROSSOVER = "bullish_crossover"
    BEARISH_CROSSOVER = "bearish_crossover"
    PRICE_ABOVE_MA = "price_above_ma"
    PRICE_BELOW_MA = "price_below_ma"
    UPTREND_CONFIRMED = "uptrend_confirmed"
    DOWNTREND_CONFIRMED = "downtrend_confirmed"
    TREND_REVERSAL = "trend_reversal"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"
    NEUTRAL = "neutral"

class MAType(Enum):
    """Moving Average types"""
    SMA = "sma"
    EMA = "ema"
    WMA = "wma"
    DEMA = "dema"  # Double EMA
    TEMA = "tema"  # Triple EMA

@dataclass
class MAResult:
    """Moving Average analysis result"""
    ma_value: float
    ma_type: str
    period: int
    signal: MASignal
    strength: float
    trend_direction: str
    slope: float
    price_distance: float

class MAData(NamedTuple):
    """Moving Average data structure"""
    values: np.ndarray
    ma_type: str
    period: int

class MovingAverages:
    """
    Comprehensive Moving Average Indicator Suite
    
    Provides various types of moving averages with trend analysis,
    crossover detection, and support/resistance identification.
    """
    
    def __init__(self,
                 periods: List[int] = [10, 20, 50, 200],
                 ma_types: List[MAType] = [MAType.SMA, MAType.EMA]):
        """
        Initialize Moving Averages indicator
        
        Args:
            periods: List of periods for MA calculation
            ma_types: List of MA types to calculate
        """
        self.periods = sorted(periods)
        self.ma_types = ma_types
        
        # Historical data for analysis
        self.ma_history = {}
        self.crossover_history = []
        
        logger.info(f"✅ Moving Averages initialized (periods={periods}, types={[t.value for t in ma_types]})")

    def calculate_sma(self, prices: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average
        
        Args:
            prices: Price data
            period: Period for SMA calculation
            
        Returns:
            Array of SMA values
        """
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < period:
                return np.full(len(prices), np.nan)
            
            # Calculate SMA using pandas rolling window
            sma = pd.Series(prices).rolling(window=period, min_periods=1).mean().values
            
            # Set initial values to NaN where we don't have enough data
            sma[:period-1] = np.nan
            
            return sma
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return np.full(len(prices), np.nan)

    def calculate_ema(self, prices: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average
        
        Args:
            prices: Price data
            period: Period for EMA calculation
            
        Returns:
            Array of EMA values
        """
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < period:
                return np.full(len(prices), np.nan)
            
            # Calculate EMA using pandas ewm
            ema = pd.Series(prices).ewm(span=period, adjust=False).mean().values
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return np.full(len(prices), np.nan)

    def calculate_wma(self, prices: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
        """
        Calculate Weighted Moving Average
        
        Args:
            prices: Price data
            period: Period for WMA calculation
            
        Returns:
            Array of WMA values
        """
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < period:
                return np.full(len(prices), np.nan)
            
            weights = np.arange(1, period + 1)
            wma = np.full(len(prices), np.nan)
            
            for i in range(period - 1, len(prices)):
                window = prices[i - period + 1:i + 1]
                wma[i] = np.average(window, weights=weights)
            
            return wma
            
        except Exception as e:
            logger.error(f"Error calculating WMA: {e}")
            return np.full(len(prices), np.nan)

    def calculate_dema(self, prices: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
        """
        Calculate Double Exponential Moving Average
        
        Args:
            prices: Price data
            period: Period for DEMA calculation
            
        Returns:
            Array of DEMA values
        """
        try:
            ema1 = self.calculate_ema(prices, period)
            ema2 = self.calculate_ema(ema1, period)
            dema = 2 * ema1 - ema2
            
            return dema
            
        except Exception as e:
            logger.error(f"Error calculating DEMA: {e}")
            return np.full(len(prices), np.nan)

    def calculate_tema(self, prices: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
        """
        Calculate Triple Exponential Moving Average
        
        Args:
            prices: Price data
            period: Period for TEMA calculation
            
        Returns:
            Array of TEMA values
        """
        try:
            ema1 = self.calculate_ema(prices, period)
            ema2 = self.calculate_ema(ema1, period)
            ema3 = self.calculate_ema(ema2, period)
            tema = 3 * ema1 - 3 * ema2 + ema3
            
            return tema
            
        except Exception as e:
            logger.error(f"Error calculating TEMA: {e}")
            return np.full(len(prices), np.nan)

    def calculate_ma(self, prices: Union[np.ndarray, pd.Series], 
                    period: int, ma_type: MAType) -> MAData:
        """
        Calculate moving average of specified type
        
        Args:
            prices: Price data
            period: Period for MA calculation
            ma_type: Type of moving average
            
        Returns:
            MAData with calculated values
        """
        try:
            if ma_type == MAType.SMA:
                values = self.calculate_sma(prices, period)
            elif ma_type == MAType.EMA:
                values = self.calculate_ema(prices, period)
            elif ma_type == MAType.WMA:
                values = self.calculate_wma(prices, period)
            elif ma_type == MAType.DEMA:
                values = self.calculate_dema(prices, period)
            elif ma_type == MAType.TEMA:
                values = self.calculate_tema(prices, period)
            else:
                raise ValueError(f"Unknown MA type: {ma_type}")
            
            return MAData(values, ma_type.value, period)
            
        except Exception as e:
            logger.error(f"Error calculating MA: {e}")
            return MAData(np.full(len(prices), np.nan), ma_type.value, period)

    def analyze_single_ma(self, prices: Union[np.ndarray, pd.Series],
                         period: int, ma_type: MAType) -> MAResult:
        """
        Analyze single moving average
        
        Args:
            prices: Price data
            period: Period for MA calculation
            ma_type: Type of moving average
            
        Returns:
            MAResult with analysis
        """
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            # Calculate MA
            ma_data = self.calculate_ma(prices, period, ma_type)
            ma_values = ma_data.values
            
            if len(ma_values) < 2 or np.isnan(ma_values[-1]):
                return MAResult(np.nan, ma_type.value, period, MASignal.NEUTRAL, 
                              0.0, "neutral", 0.0, 0.0)
            
            current_ma = ma_values[-1]
            current_price = prices[-1]
            
            # Calculate slope (trend direction)
            slope = self._calculate_slope(ma_values)
            
            # Determine trend direction
            trend_direction = self._get_trend_direction(slope)
            
            # Calculate price distance from MA
            price_distance = (current_price - current_ma) / current_ma * 100
            
            # Determine signal
            signal = self._determine_single_ma_signal(prices, ma_values, slope)
            
            # Calculate signal strength
            strength = self._calculate_single_ma_strength(prices, ma_values, slope)
            
            return MAResult(
                ma_value=current_ma,
                ma_type=ma_type.value,
                period=period,
                signal=signal,
                strength=strength,
                trend_direction=trend_direction,
                slope=slope,
                price_distance=price_distance
            )
            
        except Exception as e:
            logger.error(f"Error in single MA analysis: {e}")
            return MAResult(np.nan, ma_type.value, period, MASignal.NEUTRAL, 
                          0.0, "neutral", 0.0, 0.0)

    def analyze_ma_crossover(self, prices: Union[np.ndarray, pd.Series],
                           fast_period: int, slow_period: int,
                           ma_type: MAType = MAType.EMA) -> Dict:
        """
        Analyze moving average crossover signals
        
        Args:
            prices: Price data
            fast_period: Period for fast MA
            slow_period: Period for slow MA
            ma_type: Type of moving average
            
        Returns:
            Dictionary with crossover analysis
        """
        try:
            if fast_period >= slow_period:
                raise ValueError("Fast period must be less than slow period")
            
            # Calculate both MAs
            fast_ma = self.calculate_ma(prices, fast_period, ma_type)
            slow_ma = self.calculate_ma(prices, slow_period, ma_type)
            
            # Detect crossovers
            crossover_signal = self._detect_crossover(fast_ma.values, slow_ma.values)
            
            # Calculate crossover strength
            strength = self._calculate_crossover_strength(fast_ma.values, slow_ma.values)
            
            # Get trend confirmation
            trend_confirmed = self._confirm_crossover_trend(fast_ma.values, slow_ma.values)
            
            return {
                'signal': crossover_signal.value,
                'strength': strength,
                'fast_ma': fast_ma.values[-1] if not np.isnan(fast_ma.values[-1]) else None,
                'slow_ma': slow_ma.values[-1] if not np.isnan(slow_ma.values[-1]) else None,
                'fast_period': fast_period,
                'slow_period': slow_period,
                'ma_type': ma_type.value,
                'trend_confirmed': trend_confirmed,
                'fast_above_slow': fast_ma.values[-1] > slow_ma.values[-1] if not np.isnan(fast_ma.values[-1]) else False
            }
            
        except Exception as e:
            logger.error(f"Error in MA crossover analysis: {e}")
            return {'signal': 'neutral', 'strength': 0.0}

    def _calculate_slope(self, ma_values: np.ndarray, periods: int = 5) -> float:
        """Calculate slope of moving average"""
        try:
            if len(ma_values) < periods or np.any(np.isnan(ma_values[-periods:])):
                return 0.0
            
            recent_values = ma_values[-periods:]
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            
            # Normalize slope relative to price level
            avg_price = np.mean(recent_values)
            normalized_slope = (slope / avg_price) * 100 if avg_price != 0 else 0.0
            
            return normalized_slope
            
        except Exception:
            return 0.0

    def _get_trend_direction(self, slope: float) -> str:
        """Determine trend direction from slope"""
        if slope > 0.1:
            return "bullish"
        elif slope < -0.1:
            return "bearish"
        else:
            return "neutral"

    def _determine_single_ma_signal(self, prices: np.ndarray, ma_values: np.ndarray, slope: float) -> MASignal:
        """Determine signal for single MA analysis"""
        try:
            if len(prices) < 2 or len(ma_values) < 2:
                return MASignal.NEUTRAL
            
            current_price = prices[-1]
            prev_price = prices[-2]
            current_ma = ma_values[-1]
            prev_ma = ma_values[-2]
            
            # Check for price crossing MA
            if prev_price <= prev_ma and current_price > current_ma:
                return MASignal.PRICE_ABOVE_MA
            elif prev_price >= prev_ma and current_price < current_ma:
                return MASignal.PRICE_BELOW_MA
            
            # Check for trend confirmation
            if slope > 0.2 and current_price > current_ma:
                return MASignal.UPTREND_CONFIRMED
            elif slope < -0.2 and current_price < current_ma:
                return MASignal.DOWNTREND_CONFIRMED
            
            # Check for potential reversal
            if abs(slope) < 0.05 and len(ma_values) >= 10:
                recent_slope = self._calculate_slope(ma_values[-10:], 10)
                if abs(recent_slope - slope) > 0.1:
                    return MASignal.TREND_REVERSAL
            
            # Check for support/resistance
            price_distance = abs(current_price - current_ma) / current_ma
            if price_distance < 0.002:  # Very close to MA
                if slope > 0:
                    return MASignal.SUPPORT_BOUNCE
                elif slope < 0:
                    return MASignal.RESISTANCE_REJECTION
            
            return MASignal.NEUTRAL
            
        except Exception:
            return MASignal.NEUTRAL

    def _calculate_single_ma_strength(self, prices: np.ndarray, ma_values: np.ndarray, slope: float) -> float:
        """Calculate strength of single MA signal"""
        try:
            if len(prices) < 2 or np.isnan(ma_values[-1]):
                return 0.0
            
            current_price = prices[-1]
            current_ma = ma_values[-1]
            
            # Base strength on slope magnitude
            slope_strength = min(1.0, abs(slope) / 0.5)
            
            # Factor in price distance from MA
            price_distance = abs(current_price - current_ma) / current_ma
            distance_factor = 1.0 - min(1.0, price_distance / 0.05)  # Stronger when closer
            
            # Factor in trend consistency
            if len(ma_values) >= 5:
                recent_slopes = []
                for i in range(3):
                    if len(ma_values) >= 5 + i:
                        recent_slope = self._calculate_slope(ma_values[-(5+i):], 5)
                        recent_slopes.append(recent_slope)
                
                if recent_slopes:
                    slope_consistency = 1.0 - (np.std(recent_slopes) / max(0.1, np.mean(np.abs(recent_slopes))))
                    slope_consistency = max(0.0, min(1.0, slope_consistency))
                else:
                    slope_consistency = 0.5
            else:
                slope_consistency = 0.5
            
            # Combine factors
            strength = (slope_strength + distance_factor + slope_consistency) / 3
            return max(0.0, min(1.0, strength))
            
        except Exception:
            return 0.0

    def _detect_crossover(self, fast_ma: np.ndarray, slow_ma: np.ndarray) -> MASignal:
        """Detect MA crossover signals"""
        try:
            if len(fast_ma) < 2 or len(slow_ma) < 2:
                return MASignal.NEUTRAL
            
            if np.isnan(fast_ma[-1]) or np.isnan(slow_ma[-1]) or \
               np.isnan(fast_ma[-2]) or np.isnan(slow_ma[-2]):
                return MASignal.NEUTRAL
            
            current_fast = fast_ma[-1]
            prev_fast = fast_ma[-2]
            current_slow = slow_ma[-1]
            prev_slow = slow_ma[-2]
            
            # Bullish crossover: fast MA crosses above slow MA
            if prev_fast <= prev_slow and current_fast > current_slow:
                return MASignal.BULLISH_CROSSOVER
            
            # Bearish crossover: fast MA crosses below slow MA
            elif prev_fast >= prev_slow and current_fast < current_slow:
                return MASignal.BEARISH_CROSSOVER
            
            return MASignal.NEUTRAL
            
        except Exception:
            return MASignal.NEUTRAL

    def _calculate_crossover_strength(self, fast_ma: np.ndarray, slow_ma: np.ndarray) -> float:
        """Calculate strength of crossover signal"""
        try:
            if len(fast_ma) < 5 or len(slow_ma) < 5:
                return 0.0
            
            # Calculate separation between MAs
            ma_separation = abs(fast_ma[-1] - slow_ma[-1]) / slow_ma[-1]
            separation_strength = min(1.0, ma_separation / 0.02)
            
            # Calculate momentum of crossover
            fast_momentum = (fast_ma[-1] - fast_ma[-3]) / fast_ma[-3]
            slow_momentum = (slow_ma[-1] - slow_ma[-3]) / slow_ma[-3]
            momentum_diff = abs(fast_momentum - slow_momentum)
            momentum_strength = min(1.0, momentum_diff / 0.01)
            
            # Combine strengths
            strength = (separation_strength + momentum_strength) / 2
            return max(0.0, min(1.0, strength))
            
        except Exception:
            return 0.0

    def _confirm_crossover_trend(self, fast_ma: np.ndarray, slow_ma: np.ndarray) -> bool:
        """Confirm if crossover is supported by trend"""
        try:
            if len(fast_ma) < 10 or len(slow_ma) < 10:
                return False
            
            # Check if both MAs are trending in same direction
            fast_slope = self._calculate_slope(fast_ma[-10:], 10)
            slow_slope = self._calculate_slope(slow_ma[-10:], 10)
            
            # For bullish crossover, both should be trending up
            if fast_ma[-1] > slow_ma[-1]:
                return fast_slope > 0 and slow_slope > 0
            # For bearish crossover, both should be trending down
            else:
                return fast_slope < 0 and slow_slope < 0
                
        except Exception:
            return False

    def get_comprehensive_analysis(self, prices: Union[np.ndarray, pd.Series]) -> Dict:
        """
        Get comprehensive moving average analysis
        
        Returns:
            Dictionary with complete MA analysis
        """
        try:
            results = {
                'single_ma_analysis': {},
                'crossover_analysis': {},
                'trend_summary': {},
                'recommendations': []
            }
            
            # Analyze individual MAs
            for period in self.periods:
                for ma_type in self.ma_types:
                    key = f"{ma_type.value}_{period}"
                    analysis = self.analyze_single_ma(prices, period, ma_type)
                    results['single_ma_analysis'][key] = {
                        'ma_value': analysis.ma_value,
                        'signal': analysis.signal.value,
                        'strength': analysis.strength,
                        'trend_direction': analysis.trend_direction,
                        'slope': analysis.slope,
                        'price_distance': analysis.price_distance
                    }
            
            # Analyze common crossovers
            crossover_pairs = [(10, 20), (20, 50), (50, 200)]
            for fast, slow in crossover_pairs:
                if fast in self.periods and slow in self.periods:
                    for ma_type in self.ma_types:
                        key = f"{ma_type.value}_{fast}_{slow}"
                        crossover = self.analyze_ma_crossover(prices, fast, slow, ma_type)
                        results['crossover_analysis'][key] = crossover
            
            # Generate trend summary
            results['trend_summary'] = self._generate_trend_summary(results['single_ma_analysis'])
            
            # Generate recommendations
            results['recommendations'] = self._generate_ma_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive MA analysis: {e}")
            return {'error': str(e)}

    def _generate_trend_summary(self, single_ma_analysis: Dict) -> Dict:
        """Generate overall trend summary from MA analysis"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            neutral_signals = 0
            
            for analysis in single_ma_analysis.values():
                if analysis['trend_direction'] == 'bullish':
                    bullish_signals += 1
                elif analysis['trend_direction'] == 'bearish':
                    bearish_signals += 1
                else:
                    neutral_signals += 1
            
            total_signals = bullish_signals + bearish_signals + neutral_signals
            
            if total_signals == 0:
                overall_trend = "neutral"
                confidence = 0.0
            else:
                if bullish_signals > bearish_signals:
                    overall_trend = "bullish"
                    confidence = bullish_signals / total_signals
                elif bearish_signals > bullish_signals:
                    overall_trend = "bearish"
                    confidence = bearish_signals / total_signals
                else:
                    overall_trend = "neutral"
                    confidence = neutral_signals / total_signals
            
            return {
                'overall_trend': overall_trend,
                'confidence': confidence,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'neutral_signals': neutral_signals
            }
            
        except Exception:
            return {'overall_trend': 'neutral', 'confidence': 0.0}

    def _generate_ma_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate trading recommendations based on MA analysis"""
        recommendations = []
        
        try:
            trend_summary = analysis_results.get('trend_summary', {})
            crossover_analysis = analysis_results.get('crossover_analysis', {})
            
            # Overall trend recommendations
            overall_trend = trend_summary.get('overall_trend', 'neutral')
            confidence = trend_summary.get('confidence', 0.0)
            
            if overall_trend == 'bullish' and confidence > 0.7:
                recommendations.append("Strong bullish trend confirmed by multiple MAs - Favor long positions")
            elif overall_trend == 'bearish' and confidence > 0.7:
                recommendations.append("Strong bearish trend confirmed by multiple MAs - Favor short positions")
            elif confidence < 0.4:
                recommendations.append("Mixed MA signals - Wait for clearer trend direction")
            
            # Crossover recommendations
            for key, crossover in crossover_analysis.items():
                if crossover.get('signal') == 'bullish_crossover' and crossover.get('strength', 0) > 0.6:
                    recommendations.append(f"Strong buy signal - {key} bullish crossover detected")
                elif crossover.get('signal') == 'bearish_crossover' and crossover.get('strength', 0) > 0.6:
                    recommendations.append(f"Strong sell signal - {key} bearish crossover detected")
            
            return recommendations
            
        except Exception:
            return ["Unable to generate recommendations"]
