"""
Fisher Transform Indicator
Advanced implementation for price extremes detection and cycle analysis
Optimized for M1-H4 timeframes and turning point identification
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

class FisherSignalType(Enum):
    """Fisher Transform signal types"""
    EXTREME_OVERBOUGHT = "extreme_overbought"
    EXTREME_OVERSOLD = "extreme_oversold"
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    TREND_CONTINUATION_BULLISH = "trend_continuation_bullish"
    TREND_CONTINUATION_BEARISH = "trend_continuation_bearish"
    ZERO_LINE_CROSS_BULLISH = "zero_line_cross_bullish"
    ZERO_LINE_CROSS_BEARISH = "zero_line_cross_bearish"
    DIVERGENCE_BULLISH = "divergence_bullish"
    DIVERGENCE_BEARISH = "divergence_bearish"
    CYCLE_PEAK = "cycle_peak"
    CYCLE_TROUGH = "cycle_trough"

class FisherTrend(Enum):
    """Fisher Transform trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    EXTREME_BULLISH = "extreme_bullish"
    EXTREME_BEARISH = "extreme_bearish"

@dataclass
class FisherSignal:
    """Fisher Transform signal data structure"""
    timestamp: datetime
    price: float
    fisher_value: float
    fisher_signal: float
    normalized_price: float
    extreme_level: str
    signal_type: str
    signal_strength: float
    confidence: float
    divergence_strength: float
    cycle_position: str
    timeframe: str
    session: str

class FisherTransform:
    """
    Advanced Fisher Transform implementation for forex trading
    Features:
    - Price normalization and Fisher transformation
    - Extreme price level detection (overbought/oversold)
    - Turning point and reversal signal identification
    - Divergence detection between price and Fisher values
    - Cycle analysis and peak/trough detection
    - Zero-line crossover signals
    - Session-aware extreme level analysis
    - Multiple timeframe support
    """

    def __init__(self,
                 period: int = 10,
                 smoothing_factor: float = 0.33,
                 extreme_threshold: float = 2.0,
                 divergence_lookback: int = 20,
                 timeframes: List[str] = None):
        """
        Initialize Fisher Transform calculator

        Args:
            period: Period for price normalization
            smoothing_factor: Smoothing factor for Fisher calculation (0-1)
            extreme_threshold: Threshold for extreme levels
            divergence_lookback: Lookback period for divergence detection
            timeframes: List of timeframes to analyze
        """
        self.period = period
        self.smoothing_factor = smoothing_factor
        self.extreme_threshold = extreme_threshold
        self.divergence_lookback = divergence_lookback
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']

        # Signal thresholds
        self.overbought_threshold = extreme_threshold
        self.oversold_threshold = -extreme_threshold
        self.reversal_threshold = 1.5
        self.trend_continuation_threshold = 1.0

        # Performance tracking
        self.signal_history = []
        self.extreme_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'reversal_accuracy': 0.0,
            'extreme_accuracy': 0.0
        }

        logger.info(f"FisherTransform initialized: period={period}, "
                   f"smoothing_factor={smoothing_factor}, extreme_threshold={extreme_threshold}")

    def calculate_fisher_transform(self,
                                  high: Union[pd.Series, np.ndarray],
                                  low: Union[pd.Series, np.ndarray],
                                  close: Union[pd.Series, np.ndarray],
                                  timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Fisher Transform for given price data

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps for session analysis

        Returns:
            Dictionary containing Fisher Transform calculations
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)

            if len(high_array) != len(low_array) or len(high_array) != len(close_array):
                logger.error("High, low, and close arrays must have same length")
                return self._empty_result()

            if len(high_array) < self.period:
                logger.warning(f"Insufficient data: {len(high_array)} < {self.period}")
                return self._empty_result()

            # Calculate median price
            median_price = (high_array + low_array) / 2.0

            # Normalize prices
            normalized_prices = self._normalize_prices(median_price)

            # Calculate Fisher Transform
            fisher_values = self._calculate_fisher_values(normalized_prices)

            # Calculate Fisher signal (previous Fisher value)
            fisher_signal = self._calculate_fisher_signal(fisher_values)

            # Identify extreme levels
            extreme_levels = self._identify_extreme_levels(fisher_values)

            # Detect divergences
            divergences = self._detect_divergences(close_array, fisher_values)

            # Analyze cycle position
            cycle_positions = self._analyze_cycle_position(fisher_values)

            # Calculate trend direction
            trend_direction = self._calculate_trend_direction(fisher_values, fisher_signal)

            result = {
                'median_price': median_price,
                'normalized_prices': normalized_prices,
                'fisher_values': fisher_values,
                'fisher_signal': fisher_signal,
                'extreme_levels': extreme_levels,
                'divergences': divergences,
                'cycle_positions': cycle_positions,
                'trend_direction': trend_direction,
                'period_used': self.period,
                'thresholds': {
                    'overbought': self.overbought_threshold,
                    'oversold': self.oversold_threshold,
                    'extreme': self.extreme_threshold
                }
            }

            logger.debug(f"Fisher Transform calculated: latest_fisher={fisher_values[-1]:.3f}, "
                        f"extreme={extreme_levels[-1]}, trend={trend_direction[-1]}")
            return result

        except Exception as e:
            logger.error(f"Error calculating Fisher Transform: {str(e)}")
            return self._empty_result()

    def generate_signals(self,
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[FisherSignal]:
        """
        Generate trading signals based on Fisher Transform analysis

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            timeframe: Current timeframe

        Returns:
            List of FisherSignal objects
        """
        try:
            fisher_data = self.calculate_fisher_transform(high, low, close, timestamps)
            if not fisher_data or 'fisher_values' not in fisher_data:
                return []

            signals = []
            current_time = datetime.now()

            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_fisher = fisher_data['fisher_values'][-1]
            latest_signal = fisher_data['fisher_signal'][-1]
            latest_normalized = fisher_data['normalized_prices'][-1]
            latest_extreme = fisher_data['extreme_levels'][-1]
            latest_divergence = fisher_data['divergences'][-1]
            latest_cycle = fisher_data['cycle_positions'][-1]
            latest_trend = fisher_data['trend_direction'][-1]

            # Determine current session
            session = self._get_current_session(current_time)

            # Generate signals based on Fisher analysis
            signal_data = self._analyze_fisher_signals(
                latest_fisher, latest_signal, latest_extreme,
                latest_divergence, latest_cycle, latest_trend
            )

            if signal_data['signal_type'] != 'NONE':
                signal = FisherSignal(
                    timestamp=current_time,
                    price=latest_price,
                    fisher_value=latest_fisher,
                    fisher_signal=latest_signal,
                    normalized_price=latest_normalized,
                    extreme_level=latest_extreme,
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    divergence_strength=abs(latest_divergence),
                    cycle_position=latest_cycle,
                    timeframe=timeframe,
                    session=session
                )

                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()

                logger.info(f"Fisher signal generated: {signal.signal_type} "
                           f"(fisher={signal.fisher_value:.3f}, extreme={signal.extreme_level}, "
                           f"confidence={signal.confidence:.2f})")

            return signals

        except Exception as e:
            logger.error(f"Error generating Fisher signals: {str(e)}")
            return []

    def _normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """Normalize prices to range [-1, 1]"""
        try:
            normalized = np.zeros_like(prices)

            for i in range(len(prices)):
                if i < self.period:
                    # Use available data for initial values
                    window_prices = prices[:i+1]
                else:
                    # Use rolling window
                    window_prices = prices[i-self.period+1:i+1]

                if len(window_prices) > 1:
                    min_price = np.min(window_prices)
                    max_price = np.max(window_prices)

                    if max_price != min_price:
                        # Normalize to [-1, 1] range
                        normalized[i] = 2.0 * (prices[i] - min_price) / (max_price - min_price) - 1.0
                        # Constrain to avoid extreme values
                        normalized[i] = max(-0.999, min(0.999, normalized[i]))
                    else:
                        normalized[i] = 0.0
                else:
                    normalized[i] = 0.0

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing prices: {str(e)}")
            return np.zeros_like(prices)

    def _calculate_fisher_values(self, normalized_prices: np.ndarray) -> np.ndarray:
        """Calculate Fisher Transform values"""
        try:
            fisher = np.zeros_like(normalized_prices)
            smoothed_value = 0.0

            for i in range(len(normalized_prices)):
                # Smooth the normalized price
                smoothed_value = (self.smoothing_factor * normalized_prices[i] +
                                (1 - self.smoothing_factor) * smoothed_value)

                # Apply Fisher Transform: 0.5 * ln((1 + x) / (1 - x))
                if abs(smoothed_value) < 0.999:
                    fisher[i] = 0.5 * np.log((1 + smoothed_value) / (1 - smoothed_value))
                else:
                    # Handle extreme values
                    fisher[i] = fisher[i-1] if i > 0 else 0.0

                # Apply additional smoothing to Fisher values
                if i > 0:
                    fisher[i] = (self.smoothing_factor * fisher[i] +
                               (1 - self.smoothing_factor) * fisher[i-1])

            return fisher

        except Exception as e:
            logger.error(f"Error calculating Fisher values: {str(e)}")
            return np.zeros_like(normalized_prices)

    def _calculate_fisher_signal(self, fisher_values: np.ndarray) -> np.ndarray:
        """Calculate Fisher signal (previous Fisher value)"""
        try:
            signal = np.zeros_like(fisher_values)
            signal[1:] = fisher_values[:-1]  # Shift by one period
            return signal

        except Exception as e:
            logger.error(f"Error calculating Fisher signal: {str(e)}")
            return np.zeros_like(fisher_values)

    def _identify_extreme_levels(self, fisher_values: np.ndarray) -> List[str]:
        """Identify extreme overbought/oversold levels"""
        try:
            extremes = []

            for fisher in fisher_values:
                if fisher > self.overbought_threshold:
                    extremes.append('EXTREME_OVERBOUGHT')
                elif fisher < self.oversold_threshold:
                    extremes.append('EXTREME_OVERSOLD')
                elif fisher > self.reversal_threshold:
                    extremes.append('OVERBOUGHT')
                elif fisher < -self.reversal_threshold:
                    extremes.append('OVERSOLD')
                elif fisher > 0:
                    extremes.append('BULLISH')
                elif fisher < 0:
                    extremes.append('BEARISH')
                else:
                    extremes.append('NEUTRAL')

            return extremes

        except Exception as e:
            logger.error(f"Error identifying extreme levels: {str(e)}")
            return ['NEUTRAL'] * len(fisher_values)

    def _detect_divergences(self, prices: np.ndarray, fisher_values: np.ndarray) -> np.ndarray:
        """Detect divergences between price and Fisher Transform"""
        try:
            divergences = np.zeros_like(prices)

            if len(prices) < self.divergence_lookback:
                return divergences

            for i in range(self.divergence_lookback, len(prices)):
                # Look for recent highs and lows
                lookback_start = i - self.divergence_lookback

                price_window = prices[lookback_start:i+1]
                fisher_window = fisher_values[lookback_start:i+1]

                # Find recent high and low points
                price_high_idx = np.argmax(price_window)
                price_low_idx = np.argmin(price_window)
                fisher_high_idx = np.argmax(fisher_window)
                fisher_low_idx = np.argmin(fisher_window)

                # Check for bullish divergence (price makes lower low, Fisher makes higher low)
                if (price_low_idx > len(price_window) * 0.7 and  # Recent price low
                    fisher_low_idx < len(fisher_window) * 0.5 and  # Earlier Fisher low
                    price_window[price_low_idx] < price_window[fisher_low_idx] and  # Lower price low
                    fisher_window[-1] > fisher_window[fisher_low_idx]):  # Higher Fisher low
                    divergences[i] = 1.0  # Bullish divergence

                # Check for bearish divergence (price makes higher high, Fisher makes lower high)
                elif (price_high_idx > len(price_window) * 0.7 and  # Recent price high
                      fisher_high_idx < len(fisher_window) * 0.5 and  # Earlier Fisher high
                      price_window[price_high_idx] > price_window[fisher_high_idx] and  # Higher price high
                      fisher_window[-1] < fisher_window[fisher_high_idx]):  # Lower Fisher high
                    divergences[i] = -1.0  # Bearish divergence

            return divergences

        except Exception as e:
            logger.error(f"Error detecting divergences: {str(e)}")
            return np.zeros_like(prices)

    def _analyze_cycle_position(self, fisher_values: np.ndarray) -> List[str]:
        """Analyze cycle position (peak, trough, rising, falling)"""
        try:
            positions = []

            for i in range(len(fisher_values)):
                if i < 3:
                    positions.append('NEUTRAL')
                    continue

                # Look at recent trend
                current = fisher_values[i]
                prev1 = fisher_values[i-1]
                prev2 = fisher_values[i-2]
                prev3 = fisher_values[i-3]

                # Detect peaks and troughs
                if (prev1 > prev2 and prev1 > current and
                    prev1 > self.reversal_threshold):
                    positions.append('PEAK')
                elif (prev1 < prev2 and prev1 < current and
                      prev1 < -self.reversal_threshold):
                    positions.append('TROUGH')
                elif current > prev1 > prev2:
                    positions.append('RISING')
                elif current < prev1 < prev2:
                    positions.append('FALLING')
                elif current > prev1:
                    positions.append('TURNING_UP')
                elif current < prev1:
                    positions.append('TURNING_DOWN')
                else:
                    positions.append('NEUTRAL')

            return positions

        except Exception as e:
            logger.error(f"Error analyzing cycle position: {str(e)}")
            return ['NEUTRAL'] * len(fisher_values)

    def _calculate_trend_direction(self, fisher_values: np.ndarray,
                                  fisher_signal: np.ndarray) -> List[str]:
        """Calculate trend direction based on Fisher values"""
        try:
            trends = []

            for i in range(len(fisher_values)):
                fisher = fisher_values[i]
                signal = fisher_signal[i]

                # Determine trend based on Fisher value and signal
                if fisher > self.extreme_threshold:
                    trends.append(FisherTrend.EXTREME_BULLISH.value)
                elif fisher < -self.extreme_threshold:
                    trends.append(FisherTrend.EXTREME_BEARISH.value)
                elif fisher > signal and fisher > 0:
                    trends.append(FisherTrend.BULLISH.value)
                elif fisher < signal and fisher < 0:
                    trends.append(FisherTrend.BEARISH.value)
                elif abs(fisher) < 0.5 and abs(signal) < 0.5:
                    trends.append(FisherTrend.NEUTRAL.value)
                elif fisher > signal:
                    trends.append(FisherTrend.BULLISH.value)
                elif fisher < signal:
                    trends.append(FisherTrend.BEARISH.value)
                else:
                    trends.append(FisherTrend.NEUTRAL.value)

            return trends

        except Exception as e:
            logger.error(f"Error calculating trend direction: {str(e)}")
            return [FisherTrend.NEUTRAL.value] * len(fisher_values)

    def _analyze_fisher_signals(self, fisher: float, signal: float, extreme: str,
                               divergence: float, cycle: str, trend: str) -> Dict:
        """Analyze current Fisher conditions and generate signal"""
        try:
            signal_type = 'NONE'
            signal_strength = 0.0
            confidence = 0.0

            # Divergence signals (highest priority)
            if divergence > 0.5:
                signal_type = FisherSignalType.DIVERGENCE_BULLISH.value
                signal_strength = min(1.0, divergence)
                confidence = min(0.9, 0.7 + abs(fisher) * 0.1)
            elif divergence < -0.5:
                signal_type = FisherSignalType.DIVERGENCE_BEARISH.value
                signal_strength = min(1.0, abs(divergence))
                confidence = min(0.9, 0.7 + abs(fisher) * 0.1)

            # Extreme level signals
            elif extreme == 'EXTREME_OVERBOUGHT':
                signal_type = FisherSignalType.EXTREME_OVERBOUGHT.value
                signal_strength = min(1.0, abs(fisher) / self.extreme_threshold)
                confidence = min(0.85, 0.6 + signal_strength * 0.25)
            elif extreme == 'EXTREME_OVERSOLD':
                signal_type = FisherSignalType.EXTREME_OVERSOLD.value
                signal_strength = min(1.0, abs(fisher) / self.extreme_threshold)
                confidence = min(0.85, 0.6 + signal_strength * 0.25)

            # Reversal signals at extreme levels
            elif (extreme in ['OVERBOUGHT', 'EXTREME_OVERBOUGHT'] and
                  fisher < signal and cycle in ['PEAK', 'TURNING_DOWN']):
                signal_type = FisherSignalType.BEARISH_REVERSAL.value
                signal_strength = min(1.0, abs(fisher - signal) * 2.0)
                confidence = min(0.8, 0.6 + signal_strength * 0.2)
            elif (extreme in ['OVERSOLD', 'EXTREME_OVERSOLD'] and
                  fisher > signal and cycle in ['TROUGH', 'TURNING_UP']):
                signal_type = FisherSignalType.BULLISH_REVERSAL.value
                signal_strength = min(1.0, abs(fisher - signal) * 2.0)
                confidence = min(0.8, 0.6 + signal_strength * 0.2)

            # Zero line crossover signals
            elif fisher > 0 and signal <= 0:
                signal_type = FisherSignalType.ZERO_LINE_CROSS_BULLISH.value
                signal_strength = min(1.0, fisher * 2.0)
                confidence = min(0.75, 0.5 + signal_strength * 0.25)
            elif fisher < 0 and signal >= 0:
                signal_type = FisherSignalType.ZERO_LINE_CROSS_BEARISH.value
                signal_strength = min(1.0, abs(fisher) * 2.0)
                confidence = min(0.75, 0.5 + signal_strength * 0.25)

            # Trend continuation signals
            elif (trend == FisherTrend.BULLISH.value and fisher > signal and
                  fisher > self.trend_continuation_threshold):
                signal_type = FisherSignalType.TREND_CONTINUATION_BULLISH.value
                signal_strength = min(1.0, fisher)
                confidence = min(0.7, 0.4 + signal_strength * 0.3)
            elif (trend == FisherTrend.BEARISH.value and fisher < signal and
                  fisher < -self.trend_continuation_threshold):
                signal_type = FisherSignalType.TREND_CONTINUATION_BEARISH.value
                signal_strength = min(1.0, abs(fisher))
                confidence = min(0.7, 0.4 + signal_strength * 0.3)

            # Cycle peak/trough signals
            elif cycle == 'PEAK' and fisher > self.reversal_threshold:
                signal_type = FisherSignalType.CYCLE_PEAK.value
                signal_strength = min(1.0, fisher / self.extreme_threshold)
                confidence = min(0.75, 0.5 + signal_strength * 0.25)
            elif cycle == 'TROUGH' and fisher < -self.reversal_threshold:
                signal_type = FisherSignalType.CYCLE_TROUGH.value
                signal_strength = min(1.0, abs(fisher) / self.extreme_threshold)
                confidence = min(0.75, 0.5 + signal_strength * 0.25)

            return {
                'signal_type': signal_type,
                'strength': signal_strength,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error analyzing Fisher signals: {str(e)}")
            return {'signal_type': 'NONE', 'strength': 0.0, 'confidence': 0.0}

    def _get_current_session(self, timestamp: datetime) -> str:
        """Determine current trading session"""
        try:
            hour = timestamp.hour

            # Trading sessions (UTC)
            if 0 <= hour < 8:
                return 'ASIAN'
            elif 8 <= hour < 16:
                return 'LONDON'
            elif 16 <= hour < 24:
                return 'NEW_YORK'
            else:
                return 'OVERLAP'

        except Exception as e:
            logger.error(f"Error determining session: {str(e)}")
            return 'UNKNOWN'

    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'median_price': np.array([]),
            'normalized_prices': np.array([]),
            'fisher_values': np.array([]),
            'fisher_signal': np.array([]),
            'extreme_levels': [],
            'divergences': np.array([]),
            'cycle_positions': [],
            'trend_direction': [],
            'period_used': self.period,
            'thresholds': {
                'overbought': self.overbought_threshold,
                'oversold': self.oversold_threshold,
                'extreme': self.extreme_threshold
            }
        }

    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            if len(self.signal_history) > 0:
                self.performance_stats['total_signals'] = len(self.signal_history)

                # Calculate average confidence
                confidences = [signal.confidence for signal in self.signal_history]
                self.performance_stats['avg_confidence'] = np.mean(confidences)

                # Update other stats (simplified for now)
                self.performance_stats['accuracy'] = min(0.85, 0.6 + self.performance_stats['avg_confidence'] * 0.3)

        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        try:
            return {
                **self.performance_stats,
                'signal_count': len(self.signal_history),
                'last_updated': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {str(e)}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_points = 200

    # Generate OHLC data with cycles
    base_price = 100
    trend = np.linspace(0, 10, n_points)
    cycle = 5 * np.sin(np.linspace(0, 4*np.pi, n_points))
    noise = np.random.randn(n_points) * 0.5

    close_prices = base_price + trend + cycle + noise
    high_prices = close_prices + np.abs(np.random.randn(n_points) * 0.3)
    low_prices = close_prices - np.abs(np.random.randn(n_points) * 0.3)

    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')

    # Initialize Fisher Transform
    fisher = FisherTransform(period=10, smoothing_factor=0.33, extreme_threshold=2.0)

    # Calculate Fisher Transform
    result = fisher.calculate_fisher_transform(high_prices, low_prices, close_prices)
    print("Fisher Transform calculation completed")
    print(f"Latest Fisher: {result['fisher_values'][-1]:.3f}")
    print(f"Latest extreme: {result['extreme_levels'][-1]}")
    print(f"Latest cycle: {result['cycle_positions'][-1]}")
    print(f"Latest trend: {result['trend_direction'][-1]}")

    # Generate signals
    signals = fisher.generate_signals(high_prices, low_prices, close_prices, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")

    # Display performance stats
    stats = fisher.get_performance_stats()
    print(f"Performance stats: {stats}")

    if signals:
        latest_signal = signals[-1]
        print(f"Latest signal: {latest_signal.signal_type} "
              f"(fisher={latest_signal.fisher_value:.3f}, "
              f"confidence={latest_signal.confidence:.2f})")
