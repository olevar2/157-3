"""
SuperTrend Volatility Indicator
Advanced implementation with adaptive parameters and trend following signals
Optimized for M1-H4 timeframes and trend identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction states"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    TRANSITION = "transition"

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class SuperTrendSignal:
    """SuperTrend signal data structure"""
    timestamp: datetime
    price: float
    supertrend_value: float
    trend_direction: str
    trend_strength: float
    signal_type: str
    signal_strength: str
    confidence: float
    atr_value: float
    support_resistance: float
    timeframe: str
    session: str

class SuperTrend:
    """
    Advanced SuperTrend implementation for forex trading
    Features:
    - Adaptive ATR period and multiplier
    - Trend strength analysis
    - Support/resistance level identification
    - Signal filtering and confirmation
    - Session-aware analysis
    - Multiple timeframe support
    """
    
    def __init__(self, 
                 atr_period: int = 14,
                 atr_multiplier: float = 3.0,
                 adaptive: bool = True,
                 min_multiplier: float = 2.0,
                 max_multiplier: float = 4.0,
                 timeframes: List[str] = None):
        """
        Initialize SuperTrend calculator
        
        Args:
            atr_period: Period for ATR calculation
            atr_multiplier: Base multiplier for ATR
            adaptive: Enable adaptive parameter adjustment
            min_multiplier: Minimum ATR multiplier
            max_multiplier: Maximum ATR multiplier
            timeframes: List of timeframes to analyze
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.adaptive = adaptive
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']
        
        # Signal thresholds
        self.trend_change_threshold = 0.1  # Minimum price movement for trend change
        self.strong_trend_threshold = 1.5  # ATR multiplier for strong trend
        self.consolidation_threshold = 0.5  # ATR multiplier for consolidation
        
        # Performance tracking
        self.signal_history = []
        self.trend_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'trend_accuracy': 0.0,
            'avg_trend_duration': 0.0
        }
        
        logger.info(f"SuperTrend initialized: atr_period={atr_period}, multiplier={atr_multiplier}, "
                   f"adaptive={adaptive}")
    
    def calculate_supertrend(self, 
                           high: Union[pd.Series, np.ndarray],
                           low: Union[pd.Series, np.ndarray],
                           close: Union[pd.Series, np.ndarray],
                           timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate SuperTrend for given OHLC data
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps for session analysis
            
        Returns:
            Dictionary containing SuperTrend calculations
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            
            if len(close_array) < self.atr_period:
                logger.warning(f"Insufficient data: {len(close_array)} < {self.atr_period}")
                return self._empty_result()
            
            # Calculate ATR
            atr_values = self._calculate_atr(high_array, low_array, close_array)
            
            # Calculate adaptive multiplier if enabled
            multipliers = self._calculate_adaptive_multipliers(atr_values, close_array) if self.adaptive else None
            
            # Calculate basic upper and lower bands
            hl2 = (high_array + low_array) / 2  # Median price
            
            if multipliers is not None:
                upper_bands = hl2 + (multipliers * atr_values)
                lower_bands = hl2 - (multipliers * atr_values)
            else:
                upper_bands = hl2 + (self.atr_multiplier * atr_values)
                lower_bands = hl2 - (self.atr_multiplier * atr_values)
            
            # Calculate final SuperTrend values
            supertrend_values, trend_directions = self._calculate_final_supertrend(
                close_array, upper_bands, lower_bands
            )
            
            # Calculate additional metrics
            trend_strength = self._calculate_trend_strength(close_array, supertrend_values, atr_values)
            support_resistance = self._identify_support_resistance(supertrend_values, trend_directions)
            signal_quality = self._assess_signal_quality(close_array, supertrend_values, trend_directions)
            
            result = {
                'supertrend_values': supertrend_values,
                'trend_directions': trend_directions,
                'atr_values': atr_values,
                'upper_bands': upper_bands,
                'lower_bands': lower_bands,
                'trend_strength': trend_strength,
                'support_resistance': support_resistance,
                'signal_quality': signal_quality,
                'multipliers_used': multipliers if multipliers is not None else np.full_like(atr_values, self.atr_multiplier),
                'atr_period_used': self.atr_period
            }
            
            logger.debug(f"SuperTrend calculated: latest_value={supertrend_values[-1]:.5f}, "
                        f"trend={trend_directions[-1]}, strength={trend_strength[-1]:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {str(e)}")
            return self._empty_result()
    
    def generate_signals(self, 
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[SuperTrendSignal]:
        """
        Generate trading signals based on SuperTrend analysis
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            timeframe: Current timeframe
            
        Returns:
            List of SuperTrendSignal objects
        """
        try:
            supertrend_data = self.calculate_supertrend(high, low, close, timestamps)
            if not supertrend_data or 'supertrend_values' not in supertrend_data:
                return []
            
            signals = []
            current_time = datetime.now()
            
            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_supertrend = supertrend_data['supertrend_values'][-1]
            latest_trend = supertrend_data['trend_directions'][-1]
            latest_strength = supertrend_data['trend_strength'][-1]
            latest_atr = supertrend_data['atr_values'][-1]
            latest_support_resistance = supertrend_data['support_resistance'][-1]
            latest_quality = supertrend_data['signal_quality'][-1]
            
            # Check for trend changes (look at previous values)
            if len(supertrend_data['trend_directions']) > 1:
                prev_trend = supertrend_data['trend_directions'][-2]
                trend_changed = latest_trend != prev_trend
            else:
                trend_changed = False
            
            # Determine current session
            session = self._get_current_session(current_time)
            
            # Generate signals based on SuperTrend analysis
            signal_data = self._analyze_supertrend_signals(
                latest_price, latest_supertrend, latest_trend, latest_strength,
                latest_quality, trend_changed
            )
            
            if signal_data['signal_type'] != 'NONE':
                signal = SuperTrendSignal(
                    timestamp=current_time,
                    price=latest_price,
                    supertrend_value=latest_supertrend,
                    trend_direction=latest_trend,
                    trend_strength=latest_strength,
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    atr_value=latest_atr,
                    support_resistance=latest_support_resistance,
                    timeframe=timeframe,
                    session=session
                )
                
                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()
                
                logger.info(f"SuperTrend signal generated: {signal.signal_type} "
                           f"(trend={signal.trend_direction}, strength={signal.signal_strength}, "
                           f"confidence={signal.confidence:.2f})")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating SuperTrend signals: {str(e)}")
            return []
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Average True Range using Wilder's smoothing"""
        try:
            # Calculate True Range
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate ATR using Wilder's smoothing
            atr = np.zeros_like(true_range)
            atr[0] = true_range[0]
            
            for i in range(1, len(true_range)):
                if i < self.atr_period:
                    atr[i] = np.mean(true_range[:i+1])
                else:
                    atr[i] = (atr[i-1] * (self.atr_period - 1) + true_range[i]) / self.atr_period
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return np.ones_like(high) * 0.001
    
    def _calculate_adaptive_multipliers(self, atr_values: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Calculate adaptive ATR multipliers based on market conditions"""
        try:
            multipliers = np.full_like(atr_values, self.atr_multiplier)
            
            for i in range(20, len(atr_values)):  # Need some history for analysis
                # Calculate recent volatility
                recent_returns = np.diff(close_prices[i-10:i+1]) / close_prices[i-10:i]
                volatility = np.std(recent_returns)
                
                # Calculate ATR trend
                atr_trend = (atr_values[i] - np.mean(atr_values[i-10:i])) / np.mean(atr_values[i-10:i])
                
                # Adjust multiplier based on conditions
                base_multiplier = self.atr_multiplier
                
                # High volatility - reduce multiplier for tighter stops
                if volatility > 0.02:
                    multiplier_adj = -0.5
                # Low volatility - increase multiplier to avoid whipsaws
                elif volatility < 0.005:
                    multiplier_adj = 0.5
                else:
                    multiplier_adj = 0.0
                
                # ATR expanding - reduce multiplier
                if atr_trend > 0.2:
                    multiplier_adj -= 0.3
                # ATR contracting - increase multiplier
                elif atr_trend < -0.2:
                    multiplier_adj += 0.3
                
                # Apply adjustment with bounds
                adjusted_multiplier = base_multiplier + multiplier_adj
                multipliers[i] = np.clip(adjusted_multiplier, self.min_multiplier, self.max_multiplier)
            
            return multipliers
            
        except Exception as e:
            logger.error(f"Error calculating adaptive multipliers: {str(e)}")
            return np.full_like(atr_values, self.atr_multiplier)
    
    def _calculate_final_supertrend(self, close_prices: np.ndarray, 
                                   upper_bands: np.ndarray, lower_bands: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Calculate final SuperTrend values and trend directions"""
        try:
            supertrend = np.zeros_like(close_prices)
            trend_directions = []
            
            # Initialize first values
            if close_prices[0] > upper_bands[0]:
                supertrend[0] = lower_bands[0]
                current_trend = TrendDirection.UPTREND.value
            else:
                supertrend[0] = upper_bands[0]
                current_trend = TrendDirection.DOWNTREND.value
            
            trend_directions.append(current_trend)
            
            # Calculate subsequent values
            for i in range(1, len(close_prices)):
                # Calculate basic upper and lower bands with previous values
                basic_upper = upper_bands[i]
                basic_lower = lower_bands[i]
                
                # Adjust bands based on previous values
                final_upper = basic_upper if basic_upper < upper_bands[i-1] or close_prices[i-1] > upper_bands[i-1] else upper_bands[i-1]
                final_lower = basic_lower if basic_lower > lower_bands[i-1] or close_prices[i-1] < lower_bands[i-1] else lower_bands[i-1]
                
                # Determine trend and SuperTrend value
                if supertrend[i-1] == upper_bands[i-1] and close_prices[i] <= final_upper:
                    supertrend[i] = final_upper
                    current_trend = TrendDirection.DOWNTREND.value
                elif supertrend[i-1] == upper_bands[i-1] and close_prices[i] > final_upper:
                    supertrend[i] = final_lower
                    current_trend = TrendDirection.UPTREND.value
                elif supertrend[i-1] == lower_bands[i-1] and close_prices[i] >= final_lower:
                    supertrend[i] = final_lower
                    current_trend = TrendDirection.UPTREND.value
                elif supertrend[i-1] == lower_bands[i-1] and close_prices[i] < final_lower:
                    supertrend[i] = final_upper
                    current_trend = TrendDirection.DOWNTREND.value
                else:
                    # Maintain previous trend
                    supertrend[i] = supertrend[i-1]
                    current_trend = trend_directions[-1]
                
                trend_directions.append(current_trend)
                
                # Update bands for next iteration
                upper_bands[i] = final_upper
                lower_bands[i] = final_lower
            
            return supertrend, trend_directions
            
        except Exception as e:
            logger.error(f"Error calculating final SuperTrend: {str(e)}")
            return np.zeros_like(close_prices), [TrendDirection.SIDEWAYS.value] * len(close_prices)
    
    def _calculate_trend_strength(self, close_prices: np.ndarray, 
                                 supertrend_values: np.ndarray, atr_values: np.ndarray) -> np.ndarray:
        """Calculate trend strength based on price distance from SuperTrend"""
        try:
            trend_strength = np.zeros_like(close_prices)
            
            for i in range(len(close_prices)):
                if atr_values[i] > 0:
                    # Distance from SuperTrend relative to ATR
                    distance = abs(close_prices[i] - supertrend_values[i])
                    strength = distance / atr_values[i]
                    trend_strength[i] = min(3.0, strength)  # Cap at 3.0
                else:
                    trend_strength[i] = 0.5
            
            return trend_strength
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return np.ones_like(close_prices) * 0.5
    
    def _identify_support_resistance(self, supertrend_values: np.ndarray, 
                                   trend_directions: List[str]) -> np.ndarray:
        """Identify support/resistance levels from SuperTrend"""
        try:
            support_resistance = np.copy(supertrend_values)
            
            # SuperTrend acts as dynamic support in uptrend, resistance in downtrend
            for i in range(len(trend_directions)):
                if trend_directions[i] == TrendDirection.UPTREND.value:
                    # SuperTrend is support
                    support_resistance[i] = supertrend_values[i]
                elif trend_directions[i] == TrendDirection.DOWNTREND.value:
                    # SuperTrend is resistance
                    support_resistance[i] = supertrend_values[i]
                else:
                    # Sideways - use previous value
                    if i > 0:
                        support_resistance[i] = support_resistance[i-1]
            
            return support_resistance
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {str(e)}")
            return np.copy(supertrend_values)
    
    def _assess_signal_quality(self, close_prices: np.ndarray, supertrend_values: np.ndarray, 
                              trend_directions: List[str]) -> np.ndarray:
        """Assess signal quality based on trend consistency and price action"""
        try:
            signal_quality = np.zeros_like(close_prices)
            
            for i in range(5, len(close_prices)):  # Need some history
                # Check trend consistency over recent periods
                recent_trends = trend_directions[max(0, i-5):i+1]
                trend_consistency = len(set(recent_trends)) == 1  # All same trend
                
                # Check price momentum
                price_momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                
                # Check distance from SuperTrend
                distance_ratio = abs(close_prices[i] - supertrend_values[i]) / close_prices[i]
                
                # Calculate quality score
                quality = 0.5  # Base quality
                
                if trend_consistency:
                    quality += 0.2
                
                if abs(price_momentum) > 0.01:  # Strong momentum
                    quality += 0.2
                
                if distance_ratio > 0.005:  # Good separation from SuperTrend
                    quality += 0.1
                
                signal_quality[i] = min(1.0, quality)
            
            return signal_quality
            
        except Exception as e:
            logger.error(f"Error assessing signal quality: {str(e)}")
            return np.ones_like(close_prices) * 0.5
    
    def _analyze_supertrend_signals(self, price: float, supertrend: float, trend: str,
                                   strength: float, quality: float, trend_changed: bool) -> Dict:
        """Analyze current SuperTrend conditions and generate signal"""
        try:
            signal_type = 'NONE'
            signal_strength = SignalStrength.WEAK.value
            confidence = 0.0
            
            # Trend change signals (highest priority)
            if trend_changed:
                if trend == TrendDirection.UPTREND.value:
                    signal_type = 'TREND_CHANGE_BULLISH'
                    confidence = min(0.9, 0.7 + quality * 0.2)
                elif trend == TrendDirection.DOWNTREND.value:
                    signal_type = 'TREND_CHANGE_BEARISH'
                    confidence = min(0.9, 0.7 + quality * 0.2)
                
                # Determine signal strength based on trend strength
                if strength > 2.0:
                    signal_strength = SignalStrength.VERY_STRONG.value
                elif strength > 1.5:
                    signal_strength = SignalStrength.STRONG.value
                elif strength > 1.0:
                    signal_strength = SignalStrength.MODERATE.value
                else:
                    signal_strength = SignalStrength.WEAK.value
            
            # Trend continuation signals
            elif trend == TrendDirection.UPTREND.value and price > supertrend:
                if strength > self.strong_trend_threshold:
                    signal_type = 'STRONG_UPTREND_CONTINUATION'
                    signal_strength = SignalStrength.STRONG.value
                    confidence = min(0.85, 0.6 + strength * 0.15 + quality * 0.1)
                else:
                    signal_type = 'UPTREND_CONTINUATION'
                    signal_strength = SignalStrength.MODERATE.value
                    confidence = min(0.75, 0.5 + strength * 0.15 + quality * 0.1)
            
            elif trend == TrendDirection.DOWNTREND.value and price < supertrend:
                if strength > self.strong_trend_threshold:
                    signal_type = 'STRONG_DOWNTREND_CONTINUATION'
                    signal_strength = SignalStrength.STRONG.value
                    confidence = min(0.85, 0.6 + strength * 0.15 + quality * 0.1)
                else:
                    signal_type = 'DOWNTREND_CONTINUATION'
                    signal_strength = SignalStrength.MODERATE.value
                    confidence = min(0.75, 0.5 + strength * 0.15 + quality * 0.1)
            
            # Consolidation signals
            elif strength < self.consolidation_threshold:
                signal_type = 'CONSOLIDATION'
                signal_strength = SignalStrength.WEAK.value
                confidence = 0.4
            
            return {
                'signal_type': signal_type,
                'strength': signal_strength,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SuperTrend signals: {str(e)}")
            return {'signal_type': 'NONE', 'strength': SignalStrength.WEAK.value, 'confidence': 0.0}
    
    def _get_current_session(self, timestamp: datetime) -> str:
        """Determine current trading session"""
        try:
            hour = timestamp.hour
            if 0 <= hour < 8:
                return 'ASIAN'
            elif 8 <= hour < 16:
                return 'LONDON'
            elif 16 <= hour < 24:
                return 'NY'
            else:
                return 'OVERLAP'
        except Exception:
            return 'UNKNOWN'
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            if len(self.signal_history) > 0:
                self.performance_stats['total_signals'] = len(self.signal_history)
                self.performance_stats['avg_confidence'] = np.mean([s.confidence for s in self.signal_history])
                
                # Estimate accuracy based on signal types and confidence
                high_confidence_signals = [s for s in self.signal_history if s.confidence > 0.7]
                trend_change_signals = [s for s in self.signal_history if 'TREND_CHANGE' in s.signal_type]
                
                if high_confidence_signals:
                    self.performance_stats['successful_signals'] = len(high_confidence_signals)
                    self.performance_stats['accuracy'] = len(high_confidence_signals) / len(self.signal_history)
                
                if trend_change_signals:
                    successful_trends = [s for s in trend_change_signals if s.confidence > 0.8]
                    self.performance_stats['trend_accuracy'] = len(successful_trends) / len(trend_change_signals)
                    
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'supertrend_values': np.array([]),
            'trend_directions': [],
            'atr_values': np.array([]),
            'upper_bands': np.array([]),
            'lower_bands': np.array([]),
            'trend_strength': np.array([]),
            'support_resistance': np.array([]),
            'signal_quality': np.array([]),
            'multipliers_used': np.array([]),
            'atr_period_used': self.atr_period
        }
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.signal_history = []
        self.trend_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'trend_accuracy': 0.0,
            'avg_trend_duration': 0.0
        }
        logger.info("SuperTrend performance stats reset")

# Example usage and testing
if __name__ == "__main__":
    # Create sample OHLC data
    np.random.seed(42)
    n_points = 100
    base_price = 100
    
    # Generate realistic OHLC data with trend
    trend = np.linspace(0, 2, n_points)  # Upward trend
    noise = np.random.randn(n_points) * 0.01
    close_prices = base_price + trend + noise
    high_prices = close_prices + np.abs(np.random.randn(n_points) * 0.005)
    low_prices = close_prices - np.abs(np.random.randn(n_points) * 0.005)
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
    
    # Initialize SuperTrend
    st = SuperTrend(atr_period=14, atr_multiplier=3.0, adaptive=True)
    
    # Calculate SuperTrend
    result = st.calculate_supertrend(high_prices, low_prices, close_prices)
    print("SuperTrend calculation completed")
    print(f"Latest SuperTrend: {result['supertrend_values'][-1]:.5f}")
    print(f"Latest trend: {result['trend_directions'][-1]}")
    print(f"Trend strength: {result['trend_strength'][-1]:.2f}")
    
    # Generate signals
    signals = st.generate_signals(high_prices, low_prices, close_prices, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")
    
    # Display performance stats
    stats = st.get_performance_stats()
    print(f"Performance stats: {stats}")
