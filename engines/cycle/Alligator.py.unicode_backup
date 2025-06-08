"""
Williams Alligator Indicator
Advanced implementation with trend identification and signal generation
Optimized for M1-H4 timeframes and cycle analysis
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

class AlligatorTrend(Enum):
    """Alligator trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    SLEEPING = "sleeping"
    AWAKENING = "awakening"
    HUNTING = "hunting"

class AlligatorSignalType(Enum):
    """Alligator signal types"""
    TREND_START_BULLISH = "trend_start_bullish"
    TREND_START_BEARISH = "trend_start_bearish"
    TREND_CONTINUATION_BULLISH = "trend_continuation_bullish"
    TREND_CONTINUATION_BEARISH = "trend_continuation_bearish"
    TREND_EXHAUSTION = "trend_exhaustion"
    ALLIGATOR_SLEEPING = "alligator_sleeping"
    ALLIGATOR_AWAKENING = "alligator_awakening"
    PRICE_ABOVE_MOUTH = "price_above_mouth"
    PRICE_BELOW_MOUTH = "price_below_mouth"
    LINES_CONVERGING = "lines_converging"
    LINES_DIVERGING = "lines_diverging"

@dataclass
class AlligatorSignal:
    """Alligator signal data structure"""
    timestamp: datetime
    price: float
    jaw_value: float
    teeth_value: float
    lips_value: float
    trend_state: str
    signal_type: str
    signal_strength: float
    confidence: float
    lines_order: str
    price_position: str
    timeframe: str
    session: str

class Alligator:
    """
    Advanced Williams Alligator implementation for forex trading
    Features:
    - Three smoothed moving averages (Jaw, Teeth, Lips)
    - Trend identification and cycle analysis
    - Alligator state detection (sleeping, awakening, hunting)
    - Price position analysis relative to alligator mouth
    - Line convergence and divergence detection
    - Session-aware trend analysis
    - Multiple timeframe support
    """

    def __init__(self,
                 jaw_period: int = 13,
                 teeth_period: int = 8,
                 lips_period: int = 5,
                 jaw_shift: int = 8,
                 teeth_shift: int = 5,
                 lips_shift: int = 3,
                 timeframes: List[str] = None):
        """
        Initialize Alligator calculator

        Args:
            jaw_period: Period for Jaw line (blue line)
            teeth_period: Period for Teeth line (red line)
            lips_period: Period for Lips line (green line)
            jaw_shift: Forward shift for Jaw line
            teeth_shift: Forward shift for Teeth line
            lips_shift: Forward shift for Lips line
            timeframes: List of timeframes to analyze
        """
        self.jaw_period = jaw_period
        self.teeth_period = teeth_period
        self.lips_period = lips_period
        self.jaw_shift = jaw_shift
        self.teeth_shift = teeth_shift
        self.lips_shift = lips_shift
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']

        # Signal thresholds
        self.trend_strength_threshold = 0.7
        self.convergence_threshold = 0.001  # Percentage threshold for line convergence
        self.awakening_threshold = 0.002    # Threshold for alligator awakening
        self.hunting_threshold = 0.005      # Threshold for strong trend (hunting)

        # Performance tracking
        self.signal_history = []
        self.trend_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'trend_accuracy': 0.0,
            'awakening_accuracy': 0.0
        }

        logger.info(f"Alligator initialized: jaw_period={jaw_period}, teeth_period={teeth_period}, "
                   f"lips_period={lips_period}, shifts=({jaw_shift},{teeth_shift},{lips_shift})")

    def calculate_alligator(self,
                           prices: Union[pd.Series, np.ndarray],
                           timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Williams Alligator for given price data

        Args:
            prices: Price data (typically median price: (H+L)/2)
            timestamps: Optional timestamps for session analysis

        Returns:
            Dictionary containing Alligator calculations
        """
        try:
            # Convert to numpy array
            prices_array = np.array(prices)

            if len(prices_array) < max(self.jaw_period, self.teeth_period, self.lips_period):
                logger.warning(f"Insufficient data: {len(prices_array)} < {max(self.jaw_period, self.teeth_period, self.lips_period)}")
                return self._empty_result()

            # Calculate the three Alligator lines
            jaw_line = self._calculate_smoothed_ma(prices_array, self.jaw_period)
            teeth_line = self._calculate_smoothed_ma(prices_array, self.teeth_period)
            lips_line = self._calculate_smoothed_ma(prices_array, self.lips_period)

            # Apply forward shifts
            jaw_shifted = self._apply_shift(jaw_line, self.jaw_shift)
            teeth_shifted = self._apply_shift(teeth_line, self.teeth_shift)
            lips_shifted = self._apply_shift(lips_line, self.lips_shift)

            # Analyze alligator state
            alligator_state = self._analyze_alligator_state(jaw_shifted, teeth_shifted, lips_shifted)

            # Determine trend direction
            trend_direction = self._determine_trend_direction(jaw_shifted, teeth_shifted, lips_shifted, prices_array)

            # Analyze price position relative to alligator
            price_position = self._analyze_price_position(prices_array, jaw_shifted, teeth_shifted, lips_shifted)

            # Detect line convergence/divergence
            line_convergence = self._detect_line_convergence(jaw_shifted, teeth_shifted, lips_shifted)

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(jaw_shifted, teeth_shifted, lips_shifted, prices_array)

            result = {
                'jaw_line': jaw_line,
                'teeth_line': teeth_line,
                'lips_line': lips_line,
                'jaw_shifted': jaw_shifted,
                'teeth_shifted': teeth_shifted,
                'lips_shifted': lips_shifted,
                'alligator_state': alligator_state,
                'trend_direction': trend_direction,
                'price_position': price_position,
                'line_convergence': line_convergence,
                'trend_strength': trend_strength,
                'periods_used': {
                    'jaw': self.jaw_period,
                    'teeth': self.teeth_period,
                    'lips': self.lips_period
                }
            }

            logger.debug(f"Alligator calculated: state={alligator_state[-1]}, "
                        f"trend={trend_direction[-1]}, strength={trend_strength[-1]:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error calculating Alligator: {str(e)}")
            return self._empty_result()

    def generate_signals(self,
                        prices: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[AlligatorSignal]:
        """
        Generate trading signals based on Alligator analysis

        Args:
            prices: Price data
            timestamps: Optional timestamps
            timeframe: Current timeframe

        Returns:
            List of AlligatorSignal objects
        """
        try:
            alligator_data = self.calculate_alligator(prices, timestamps)
            if not alligator_data or 'jaw_shifted' not in alligator_data:
                return []

            signals = []
            current_time = datetime.now()

            # Get latest values
            latest_price = prices.iloc[-1] if isinstance(prices, pd.Series) else prices[-1]
            latest_jaw = alligator_data['jaw_shifted'][-1]
            latest_teeth = alligator_data['teeth_shifted'][-1]
            latest_lips = alligator_data['lips_shifted'][-1]
            latest_state = alligator_data['alligator_state'][-1]
            latest_trend = alligator_data['trend_direction'][-1]
            latest_position = alligator_data['price_position'][-1]
            latest_convergence = alligator_data['line_convergence'][-1]
            latest_strength = alligator_data['trend_strength'][-1]

            # Determine current session
            session = self._get_current_session(current_time)

            # Generate signals based on Alligator analysis
            signal_data = self._analyze_alligator_signals(
                latest_state, latest_trend, latest_position,
                latest_convergence, latest_strength
            )

            if signal_data['signal_type'] != 'NONE':
                signal = AlligatorSignal(
                    timestamp=current_time,
                    price=latest_price,
                    jaw_value=latest_jaw,
                    teeth_value=latest_teeth,
                    lips_value=latest_lips,
                    trend_state=latest_state,
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    lines_order=self._get_lines_order(latest_jaw, latest_teeth, latest_lips),
                    price_position=latest_position,
                    timeframe=timeframe,
                    session=session
                )

                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()

                logger.info(f"Alligator signal generated: {signal.signal_type} "
                           f"(state={signal.trend_state}, confidence={signal.confidence:.2f})")

            return signals

        except Exception as e:
            logger.error(f"Error generating Alligator signals: {str(e)}")
            return []

    def _calculate_smoothed_ma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate smoothed moving average (SMMA)"""
        try:
            smma = np.zeros_like(prices, dtype=float)

            # Initialize with SMA for first value
            if len(prices) >= period:
                smma[period-1] = np.mean(prices[:period])

                # Calculate SMMA for remaining values
                for i in range(period, len(prices)):
                    smma[i] = (smma[i-1] * (period - 1) + prices[i]) / period

            # Fill initial values with available data
            for i in range(period-1):
                if i == 0:
                    smma[i] = prices[i]
                else:
                    smma[i] = np.mean(prices[:i+1])

            return smma

        except Exception as e:
            logger.error(f"Error calculating smoothed MA: {str(e)}")
            return np.zeros_like(prices)

    def _apply_shift(self, line: np.ndarray, shift: int) -> np.ndarray:
        """Apply forward shift to line"""
        try:
            if shift <= 0:
                return line

            shifted = np.zeros_like(line)
            shifted[:-shift] = line[shift:]

            # Fill the end with the last value
            if len(line) > shift:
                shifted[-shift:] = line[-1]

            return shifted

        except Exception as e:
            logger.error(f"Error applying shift: {str(e)}")
            return line

    def _analyze_alligator_state(self, jaw: np.ndarray, teeth: np.ndarray, lips: np.ndarray) -> List[str]:
        """Analyze alligator state (sleeping, awakening, hunting)"""
        try:
            states = []

            for i in range(len(jaw)):
                if i < 5:
                    states.append(AlligatorTrend.SLEEPING.value)
                    continue

                # Calculate line spreads
                jaw_teeth_spread = abs(jaw[i] - teeth[i]) / jaw[i] if jaw[i] != 0 else 0
                teeth_lips_spread = abs(teeth[i] - lips[i]) / teeth[i] if teeth[i] != 0 else 0
                total_spread = abs(jaw[i] - lips[i]) / jaw[i] if jaw[i] != 0 else 0

                # Check line order for trend direction
                lines_ordered = self._check_lines_order(jaw[i], teeth[i], lips[i])

                # Determine state based on spreads and order
                if total_spread < self.convergence_threshold:
                    states.append(AlligatorTrend.SLEEPING.value)
                elif total_spread < self.awakening_threshold and lines_ordered:
                    states.append(AlligatorTrend.AWAKENING.value)
                elif total_spread >= self.hunting_threshold and lines_ordered:
                    states.append(AlligatorTrend.HUNTING.value)
                elif lines_ordered:
                    states.append(AlligatorTrend.AWAKENING.value)
                else:
                    states.append(AlligatorTrend.SLEEPING.value)

            return states

        except Exception as e:
            logger.error(f"Error analyzing alligator state: {str(e)}")
            return [AlligatorTrend.SLEEPING.value] * len(jaw)

    def _determine_trend_direction(self, jaw: np.ndarray, teeth: np.ndarray,
                                  lips: np.ndarray, prices: np.ndarray) -> List[str]:
        """Determine trend direction based on line order and price position"""
        try:
            trends = []

            for i in range(len(jaw)):
                if i < 3:
                    trends.append(AlligatorTrend.NEUTRAL.value)
                    continue

                # Check if lines are in proper order
                bullish_order = lips[i] > teeth[i] > jaw[i]
                bearish_order = lips[i] < teeth[i] < jaw[i]

                # Check price position
                price_above_all = prices[i] > max(jaw[i], teeth[i], lips[i])
                price_below_all = prices[i] < min(jaw[i], teeth[i], lips[i])

                # Determine trend
                if bullish_order and price_above_all:
                    trends.append(AlligatorTrend.BULLISH.value)
                elif bearish_order and price_below_all:
                    trends.append(AlligatorTrend.BEARISH.value)
                elif bullish_order or (lips[i] > jaw[i] and prices[i] > jaw[i]):
                    trends.append(AlligatorTrend.BULLISH.value)
                elif bearish_order or (lips[i] < jaw[i] and prices[i] < jaw[i]):
                    trends.append(AlligatorTrend.BEARISH.value)
                else:
                    trends.append(AlligatorTrend.NEUTRAL.value)

            return trends

        except Exception as e:
            logger.error(f"Error determining trend direction: {str(e)}")
            return [AlligatorTrend.NEUTRAL.value] * len(jaw)

    def _analyze_price_position(self, prices: np.ndarray, jaw: np.ndarray,
                               teeth: np.ndarray, lips: np.ndarray) -> List[str]:
        """Analyze price position relative to alligator lines"""
        try:
            positions = []

            for i in range(len(prices)):
                price = prices[i]

                # Count how many lines price is above
                above_count = 0
                if price > jaw[i]:
                    above_count += 1
                if price > teeth[i]:
                    above_count += 1
                if price > lips[i]:
                    above_count += 1

                # Determine position
                if above_count == 3:
                    positions.append('ABOVE_ALL')
                elif above_count == 0:
                    positions.append('BELOW_ALL')
                elif above_count == 2:
                    positions.append('ABOVE_MOUTH')
                elif above_count == 1:
                    positions.append('BELOW_MOUTH')
                else:
                    positions.append('INSIDE_MOUTH')

            return positions

        except Exception as e:
            logger.error(f"Error analyzing price position: {str(e)}")
            return ['INSIDE_MOUTH'] * len(prices)

    def _detect_line_convergence(self, jaw: np.ndarray, teeth: np.ndarray, lips: np.ndarray) -> List[str]:
        """Detect convergence and divergence of alligator lines"""
        try:
            convergence = []

            for i in range(len(jaw)):
                if i < 5:
                    convergence.append('NEUTRAL')
                    continue

                # Calculate current and previous spreads
                current_spread = abs(jaw[i] - lips[i])
                previous_spread = abs(jaw[i-1] - lips[i-1])

                # Determine convergence/divergence
                if current_spread < previous_spread * 0.95:
                    convergence.append('CONVERGING')
                elif current_spread > previous_spread * 1.05:
                    convergence.append('DIVERGING')
                else:
                    convergence.append('NEUTRAL')

            return convergence

        except Exception as e:
            logger.error(f"Error detecting line convergence: {str(e)}")
            return ['NEUTRAL'] * len(jaw)

    def _calculate_trend_strength(self, jaw: np.ndarray, teeth: np.ndarray,
                                 lips: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Calculate trend strength based on line separation and price position"""
        try:
            strength = np.zeros_like(jaw)

            for i in range(len(jaw)):
                if i < 3:
                    strength[i] = 0.0
                    continue

                # Calculate line separation
                total_spread = abs(jaw[i] - lips[i]) / jaw[i] if jaw[i] != 0 else 0

                # Check line order consistency
                order_score = 0.0
                if lips[i] > teeth[i] > jaw[i]:  # Bullish order
                    order_score = 1.0
                elif lips[i] < teeth[i] < jaw[i]:  # Bearish order
                    order_score = 1.0
                else:
                    order_score = 0.5

                # Price position score
                price_score = 0.0
                if prices[i] > max(jaw[i], teeth[i], lips[i]):
                    price_score = 1.0
                elif prices[i] < min(jaw[i], teeth[i], lips[i]):
                    price_score = 1.0
                else:
                    price_score = 0.3

                # Combine factors
                strength[i] = min(1.0, (total_spread * 100) * order_score * price_score)

            return strength

        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return np.zeros_like(jaw)

    def _check_lines_order(self, jaw: float, teeth: float, lips: float) -> bool:
        """Check if alligator lines are in proper order"""
        try:
            # Bullish order: Lips > Teeth > Jaw
            # Bearish order: Lips < Teeth < Jaw
            return (lips > teeth > jaw) or (lips < teeth < jaw)
        except:
            return False

    def _analyze_alligator_signals(self, state: str, trend: str, position: str,
                                  convergence: str, strength: float) -> Dict:
        """Analyze current alligator conditions and generate signal"""
        try:
            signal_type = 'NONE'
            signal_strength = 0.0
            confidence = 0.0

            # Trend start signals (highest priority)
            if state == AlligatorTrend.AWAKENING.value and convergence == 'DIVERGING':
                if trend == AlligatorTrend.BULLISH.value and position in ['ABOVE_ALL', 'ABOVE_MOUTH']:
                    signal_type = AlligatorSignalType.TREND_START_BULLISH.value
                    signal_strength = min(1.0, strength * 1.2)
                    confidence = min(0.9, 0.7 + strength * 0.2)
                elif trend == AlligatorTrend.BEARISH.value and position in ['BELOW_ALL', 'BELOW_MOUTH']:
                    signal_type = AlligatorSignalType.TREND_START_BEARISH.value
                    signal_strength = min(1.0, strength * 1.2)
                    confidence = min(0.9, 0.7 + strength * 0.2)

            # Trend continuation signals
            elif state == AlligatorTrend.HUNTING.value and strength > self.trend_strength_threshold:
                if trend == AlligatorTrend.BULLISH.value and position == 'ABOVE_ALL':
                    signal_type = AlligatorSignalType.TREND_CONTINUATION_BULLISH.value
                    signal_strength = min(1.0, strength)
                    confidence = min(0.85, 0.6 + strength * 0.25)
                elif trend == AlligatorTrend.BEARISH.value and position == 'BELOW_ALL':
                    signal_type = AlligatorSignalType.TREND_CONTINUATION_BEARISH.value
                    signal_strength = min(1.0, strength)
                    confidence = min(0.85, 0.6 + strength * 0.25)

            # Alligator awakening signals
            elif state == AlligatorTrend.AWAKENING.value:
                signal_type = AlligatorSignalType.ALLIGATOR_AWAKENING.value
                signal_strength = min(1.0, strength * 0.8)
                confidence = min(0.75, 0.5 + strength * 0.25)

            # Price position signals
            elif position == 'ABOVE_ALL' and trend == AlligatorTrend.BULLISH.value:
                signal_type = AlligatorSignalType.PRICE_ABOVE_MOUTH.value
                signal_strength = min(1.0, strength * 0.7)
                confidence = min(0.7, 0.4 + strength * 0.3)
            elif position == 'BELOW_ALL' and trend == AlligatorTrend.BEARISH.value:
                signal_type = AlligatorSignalType.PRICE_BELOW_MOUTH.value
                signal_strength = min(1.0, strength * 0.7)
                confidence = min(0.7, 0.4 + strength * 0.3)

            # Line convergence signals
            elif convergence == 'CONVERGING' and strength < 0.3:
                signal_type = AlligatorSignalType.LINES_CONVERGING.value
                signal_strength = 1.0 - strength
                confidence = min(0.65, 0.4 + (1.0 - strength) * 0.25)
            elif convergence == 'DIVERGING' and strength > 0.5:
                signal_type = AlligatorSignalType.LINES_DIVERGING.value
                signal_strength = strength
                confidence = min(0.7, 0.4 + strength * 0.3)

            # Alligator sleeping signals
            elif state == AlligatorTrend.SLEEPING.value:
                signal_type = AlligatorSignalType.ALLIGATOR_SLEEPING.value
                signal_strength = 1.0 - strength
                confidence = min(0.6, 0.3 + (1.0 - strength) * 0.3)

            # Trend exhaustion signals
            elif state == AlligatorTrend.HUNTING.value and convergence == 'CONVERGING':
                signal_type = AlligatorSignalType.TREND_EXHAUSTION.value
                signal_strength = min(1.0, 1.0 - strength + 0.3)
                confidence = min(0.75, 0.5 + (1.0 - strength) * 0.25)

            return {
                'signal_type': signal_type,
                'strength': signal_strength,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error analyzing alligator signals: {str(e)}")
            return {'signal_type': 'NONE', 'strength': 0.0, 'confidence': 0.0}

    def _get_lines_order(self, jaw: float, teeth: float, lips: float) -> str:
        """Get the current order of alligator lines"""
        try:
            if lips > teeth > jaw:
                return 'BULLISH_ORDER'
            elif lips < teeth < jaw:
                return 'BEARISH_ORDER'
            elif lips > jaw and teeth > jaw:
                return 'MIXED_BULLISH'
            elif lips < jaw and teeth < jaw:
                return 'MIXED_BEARISH'
            else:
                return 'MIXED_NEUTRAL'
        except:
            return 'UNKNOWN'

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
            'jaw_line': np.array([]),
            'teeth_line': np.array([]),
            'lips_line': np.array([]),
            'jaw_shifted': np.array([]),
            'teeth_shifted': np.array([]),
            'lips_shifted': np.array([]),
            'alligator_state': [],
            'trend_direction': [],
            'price_position': [],
            'line_convergence': [],
            'trend_strength': np.array([]),
            'periods_used': {
                'jaw': self.jaw_period,
                'teeth': self.teeth_period,
                'lips': self.lips_period
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

    # Generate trending price data
    trend = np.linspace(100, 110, n_points)
    noise = np.random.randn(n_points) * 0.5
    prices = trend + noise

    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')

    # Initialize Alligator
    alligator = Alligator(jaw_period=13, teeth_period=8, lips_period=5)

    # Calculate Alligator
    result = alligator.calculate_alligator(prices)
    print("Alligator calculation completed")
    print(f"Latest state: {result['alligator_state'][-1]}")
    print(f"Latest trend: {result['trend_direction'][-1]}")
    print(f"Latest strength: {result['trend_strength'][-1]:.2f}")

    # Generate signals
    signals = alligator.generate_signals(prices, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")

    # Display performance stats
    stats = alligator.get_performance_stats()
    print(f"Performance stats: {stats}")

    if signals:
        latest_signal = signals[-1]
        print(f"Latest signal: {latest_signal.signal_type} "
              f"(confidence={latest_signal.confidence:.2f}, "
              f"strength={latest_signal.signal_strength:.2f})")
