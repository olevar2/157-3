"""
Moving Average Convergence Divergence (MACD) Indicator

MACD is a trend-following momentum indicator that shows the relationship between
two moving averages of a security's price. It consists of the MACD line, signal line,
and histogram, providing insights into trend direction and momentum changes.

Key Features:
- MACD line calculation (fast EMA - slow EMA)
- Signal line (EMA of MACD line)
- MACD histogram (MACD line - signal line)
- Crossover signal detection
- Divergence analysis capabilities
- Zero-line crossover detection
- Customizable periods for all components

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

class MACDSignal(Enum):
    """MACD signal types"""
    BULLISH_CROSSOVER = "bullish_crossover"
    BEARISH_CROSSOVER = "bearish_crossover"
    BULLISH_ZERO_CROSS = "bullish_zero_cross"
    BEARISH_ZERO_CROSS = "bearish_zero_cross"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    MOMENTUM_ACCELERATION = "momentum_acceleration"
    MOMENTUM_DECELERATION = "momentum_deceleration"
    NEUTRAL = "neutral"

@dataclass
class MACDResult:
    """MACD calculation result"""
    macd_line: float
    signal_line: float
    histogram: float
    signal: MACDSignal
    strength: float
    trend_direction: str
    divergence_detected: bool

class MACDData(NamedTuple):
    """MACD data structure"""
    macd_line: np.ndarray
    signal_line: np.ndarray
    histogram: np.ndarray

class MACD:
    """
    Moving Average Convergence Divergence (MACD) Technical Indicator
    
    MACD is calculated by subtracting the long-period EMA from the short-period EMA.
    The signal line is an EMA of the MACD line, and the histogram shows the difference
    between the MACD line and signal line.
    """
    
    def __init__(self,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9):
        """
        Initialize MACD indicator
        
        Args:
            fast_period: Period for fast EMA (default: 12)
            slow_period: Period for slow EMA (default: 26)
            signal_period: Period for signal line EMA (default: 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Validation
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        # Historical data for analysis
        self.price_history = []
        self.macd_history = []
        
        logger.info(f"✅ MACD indicator initialized (fast={fast_period}, slow={slow_period}, signal={signal_period})")

    def calculate(self, prices: Union[np.ndarray, pd.Series]) -> MACDData:
        """
        Calculate MACD components
        
        Args:
            prices: Price data (typically closing prices)
            
        Returns:
            MACDData with MACD line, signal line, and histogram
        """
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            if len(prices) < self.slow_period:
                # Return zeros if insufficient data
                zeros = np.zeros(len(prices))
                return MACDData(zeros, zeros, zeros)
            
            # Calculate EMAs
            fast_ema = self._ema(prices, self.fast_period)
            slow_ema = self._ema(prices, self.slow_period)
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line (EMA of MACD line)
            signal_line = self._ema(macd_line, self.signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return MACDData(macd_line, signal_line, histogram)
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            zeros = np.zeros(len(prices))
            return MACDData(zeros, zeros, zeros)

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        try:
            return pd.Series(data).ewm(span=period, adjust=False).mean().values
        except Exception:
            return np.zeros(len(data))

    def analyze(self, prices: Union[np.ndarray, pd.Series],
                highs: Optional[np.ndarray] = None,
                lows: Optional[np.ndarray] = None) -> MACDResult:
        """
        Comprehensive MACD analysis with signal generation
        
        Args:
            prices: Price data (typically closing prices)
            highs: High prices for divergence analysis
            lows: Low prices for divergence analysis
            
        Returns:
            MACDResult with comprehensive analysis
        """
        try:
            # Calculate MACD components
            macd_data = self.calculate(prices)
            
            # Get current values
            current_macd = macd_data.macd_line[-1]
            current_signal = macd_data.signal_line[-1]
            current_histogram = macd_data.histogram[-1]
            
            # Determine signal
            signal = self._determine_signal(macd_data, prices, highs, lows)
            
            # Calculate signal strength
            strength = self._calculate_strength(macd_data)
            
            # Determine trend direction
            trend_direction = self._get_trend_direction(macd_data)
            
            # Check for divergence
            divergence_detected = self._detect_divergence(macd_data, prices, highs, lows)
            
            return MACDResult(
                macd_line=current_macd,
                signal_line=current_signal,
                histogram=current_histogram,
                signal=signal,
                strength=strength,
                trend_direction=trend_direction,
                divergence_detected=divergence_detected
            )
            
        except Exception as e:
            logger.error(f"Error in MACD analysis: {e}")
            return MACDResult(0.0, 0.0, 0.0, MACDSignal.NEUTRAL, 0.0, "neutral", False)

    def _determine_signal(self, macd_data: MACDData, prices: np.ndarray,
                         highs: Optional[np.ndarray] = None,
                         lows: Optional[np.ndarray] = None) -> MACDSignal:
        """Determine MACD signal based on current conditions"""
        try:
            macd_line = macd_data.macd_line
            signal_line = macd_data.signal_line
            histogram = macd_data.histogram
            
            if len(macd_line) < 2:
                return MACDSignal.NEUTRAL
            
            current_macd = macd_line[-1]
            prev_macd = macd_line[-2]
            current_signal = signal_line[-1]
            prev_signal = signal_line[-2]
            current_hist = histogram[-1]
            prev_hist = histogram[-2]
            
            # Check for MACD line and signal line crossovers
            if prev_macd <= prev_signal and current_macd > current_signal:
                return MACDSignal.BULLISH_CROSSOVER
            elif prev_macd >= prev_signal and current_macd < current_signal:
                return MACDSignal.BEARISH_CROSSOVER
            
            # Check for zero line crossovers
            if prev_macd <= 0 and current_macd > 0:
                return MACDSignal.BULLISH_ZERO_CROSS
            elif prev_macd >= 0 and current_macd < 0:
                return MACDSignal.BEARISH_ZERO_CROSS
            
            # Check for divergences
            if self._detect_bullish_divergence(macd_data, prices, lows):
                return MACDSignal.BULLISH_DIVERGENCE
            elif self._detect_bearish_divergence(macd_data, prices, highs):
                return MACDSignal.BEARISH_DIVERGENCE
            
            # Check for momentum changes in histogram
            if len(histogram) >= 3:
                hist_trend = self._get_histogram_trend(histogram[-3:])
                if hist_trend > 0 and current_hist > 0:
                    return MACDSignal.MOMENTUM_ACCELERATION
                elif hist_trend < 0 and current_hist < 0:
                    return MACDSignal.MOMENTUM_DECELERATION
            
            return MACDSignal.NEUTRAL
            
        except Exception:
            return MACDSignal.NEUTRAL

    def _calculate_strength(self, macd_data: MACDData) -> float:
        """Calculate signal strength based on MACD components"""
        try:
            macd_line = macd_data.macd_line
            histogram = macd_data.histogram
            
            if len(macd_line) < 2:
                return 0.0
            
            # Base strength on histogram magnitude and MACD line position
            current_hist = abs(histogram[-1])
            current_macd = abs(macd_line[-1])
            
            # Normalize based on recent volatility
            recent_period = min(20, len(histogram))
            hist_volatility = np.std(histogram[-recent_period:]) if recent_period > 1 else 1.0
            macd_volatility = np.std(macd_line[-recent_period:]) if recent_period > 1 else 1.0
            
            hist_strength = current_hist / max(hist_volatility, 0.001)
            macd_strength = current_macd / max(macd_volatility, 0.001)
            
            # Combine strengths
            combined_strength = (hist_strength + macd_strength) / 2
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, combined_strength / 3.0))
            
        except Exception:
            return 0.0

    def _get_trend_direction(self, macd_data: MACDData) -> str:
        """Determine trend direction based on MACD components"""
        try:
            macd_line = macd_data.macd_line
            signal_line = macd_data.signal_line
            
            if len(macd_line) < 3:
                return "neutral"
            
            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            
            # Primary trend based on MACD line position relative to zero
            if current_macd > 0:
                primary_trend = "bullish"
            elif current_macd < 0:
                primary_trend = "bearish"
            else:
                primary_trend = "neutral"
            
            # Secondary confirmation from MACD vs signal line
            if current_macd > current_signal:
                secondary_trend = "bullish"
            elif current_macd < current_signal:
                secondary_trend = "bearish"
            else:
                secondary_trend = "neutral"
            
            # Combine trends
            if primary_trend == secondary_trend:
                return primary_trend
            else:
                return "mixed"
                
        except Exception:
            return "neutral"

    def _detect_divergence(self, macd_data: MACDData, prices: np.ndarray,
                          highs: Optional[np.ndarray] = None,
                          lows: Optional[np.ndarray] = None) -> bool:
        """Detect any type of divergence"""
        try:
            bullish_div = self._detect_bullish_divergence(macd_data, prices, lows)
            bearish_div = self._detect_bearish_divergence(macd_data, prices, highs)
            return bullish_div or bearish_div
        except Exception:
            return False

    def _detect_bullish_divergence(self, macd_data: MACDData, prices: np.ndarray,
                                  lows: Optional[np.ndarray] = None) -> bool:
        """Detect bullish divergence (price makes lower low, MACD makes higher low)"""
        try:
            if len(prices) < 20:
                return False
            
            price_data = lows if lows is not None else prices
            macd_line = macd_data.macd_line
            
            # Find recent lows
            recent_period = min(20, len(price_data))
            recent_prices = price_data[-recent_period:]
            recent_macd = macd_line[-recent_period:]
            
            # Find local minima
            price_lows = self._find_local_minima(recent_prices)
            macd_lows = self._find_local_minima(recent_macd)
            
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                # Check for divergence pattern
                latest_price_low = price_lows[-1]
                prev_price_low = price_lows[-2]
                latest_macd_low = macd_lows[-1]
                prev_macd_low = macd_lows[-2]
                
                price_lower = recent_prices[latest_price_low] < recent_prices[prev_price_low]
                macd_higher = recent_macd[latest_macd_low] > recent_macd[prev_macd_low]
                
                return price_lower and macd_higher
            
            return False
            
        except Exception:
            return False

    def _detect_bearish_divergence(self, macd_data: MACDData, prices: np.ndarray,
                                  highs: Optional[np.ndarray] = None) -> bool:
        """Detect bearish divergence (price makes higher high, MACD makes lower high)"""
        try:
            if len(prices) < 20:
                return False
            
            price_data = highs if highs is not None else prices
            macd_line = macd_data.macd_line
            
            # Find recent highs
            recent_period = min(20, len(price_data))
            recent_prices = price_data[-recent_period:]
            recent_macd = macd_line[-recent_period:]
            
            # Find local maxima
            price_highs = self._find_local_maxima(recent_prices)
            macd_highs = self._find_local_maxima(recent_macd)
            
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                # Check for divergence pattern
                latest_price_high = price_highs[-1]
                prev_price_high = price_highs[-2]
                latest_macd_high = macd_highs[-1]
                prev_macd_high = macd_highs[-2]
                
                price_higher = recent_prices[latest_price_high] > recent_prices[prev_price_high]
                macd_lower = recent_macd[latest_macd_high] < recent_macd[prev_macd_high]
                
                return price_higher and macd_lower
            
            return False
            
        except Exception:
            return False

    def _find_local_minima(self, data: np.ndarray, window: int = 3) -> List[int]:
        """Find local minima in data"""
        minima = []
        for i in range(window, len(data) - window):
            if all(data[i] <= data[i-j] for j in range(1, window+1)) and \
               all(data[i] <= data[i+j] for j in range(1, window+1)):
                minima.append(i)
        return minima

    def _find_local_maxima(self, data: np.ndarray, window: int = 3) -> List[int]:
        """Find local maxima in data"""
        maxima = []
        for i in range(window, len(data) - window):
            if all(data[i] >= data[i-j] for j in range(1, window+1)) and \
               all(data[i] >= data[i+j] for j in range(1, window+1)):
                maxima.append(i)
        return maxima

    def _get_histogram_trend(self, histogram_values: np.ndarray) -> float:
        """Get trend direction of histogram"""
        try:
            if len(histogram_values) < 2:
                return 0.0
            return np.polyfit(range(len(histogram_values)), histogram_values, 1)[0]
        except Exception:
            return 0.0

    def get_trading_signals(self, prices: Union[np.ndarray, pd.Series],
                           highs: Optional[np.ndarray] = None,
                           lows: Optional[np.ndarray] = None) -> Dict:
        """
        Get comprehensive trading signals based on MACD analysis
        
        Returns:
            Dictionary with trading recommendations
        """
        try:
            analysis = self.analyze(prices, highs, lows)
            macd_data = self.calculate(prices)
            
            signals = {
                'primary_signal': analysis.signal.value,
                'signal_strength': analysis.strength,
                'macd_line': analysis.macd_line,
                'signal_line': analysis.signal_line,
                'histogram': analysis.histogram,
                'trend_direction': analysis.trend_direction,
                'divergence_detected': analysis.divergence_detected,
                'above_zero': analysis.macd_line > 0,
                'bullish_crossover': analysis.macd_line > analysis.signal_line,
                'recommendations': self._generate_recommendations(analysis, macd_data)
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}

    def _generate_recommendations(self, analysis: MACDResult, macd_data: MACDData) -> List[str]:
        """Generate trading recommendations based on MACD analysis"""
        recommendations = []
        
        try:
            if analysis.signal == MACDSignal.BULLISH_CROSSOVER:
                recommendations.append("Strong buy signal - MACD bullish crossover detected")
            elif analysis.signal == MACDSignal.BEARISH_CROSSOVER:
                recommendations.append("Strong sell signal - MACD bearish crossover detected")
            elif analysis.signal == MACDSignal.BULLISH_ZERO_CROSS:
                recommendations.append("Buy signal - MACD crossed above zero line")
            elif analysis.signal == MACDSignal.BEARISH_ZERO_CROSS:
                recommendations.append("Sell signal - MACD crossed below zero line")
            elif analysis.signal == MACDSignal.BULLISH_DIVERGENCE:
                recommendations.append("Strong buy signal - Bullish divergence detected")
            elif analysis.signal == MACDSignal.BEARISH_DIVERGENCE:
                recommendations.append("Strong sell signal - Bearish divergence detected")
            elif analysis.signal == MACDSignal.MOMENTUM_ACCELERATION:
                recommendations.append("Momentum accelerating - Consider adding to positions")
            elif analysis.signal == MACDSignal.MOMENTUM_DECELERATION:
                recommendations.append("Momentum decelerating - Consider reducing positions")
            
            # Additional recommendations based on trend and position
            if analysis.trend_direction == "bullish" and analysis.macd_line > 0:
                recommendations.append("Strong bullish trend confirmed - Favor long positions")
            elif analysis.trend_direction == "bearish" and analysis.macd_line < 0:
                recommendations.append("Strong bearish trend confirmed - Favor short positions")
            elif analysis.trend_direction == "mixed":
                recommendations.append("Mixed signals - Wait for clearer trend confirmation")
            
            # Histogram-based recommendations
            if analysis.histogram > 0 and len(macd_data.histogram) >= 2:
                if macd_data.histogram[-1] > macd_data.histogram[-2]:
                    recommendations.append("Bullish momentum increasing - Good entry opportunity")
            elif analysis.histogram < 0 and len(macd_data.histogram) >= 2:
                if macd_data.histogram[-1] < macd_data.histogram[-2]:
                    recommendations.append("Bearish momentum increasing - Good short entry")
            
            return recommendations
            
        except Exception:
            return ["Unable to generate recommendations"]
