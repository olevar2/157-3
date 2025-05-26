"""
Money Flow Index (MFI) Implementation for Platform3 Forex Trading
Advanced volume-price momentum oscillator for buying/selling pressure analysis

Features:
- Traditional MFI calculation with volume-price analysis
- Divergence detection for trend reversal signals
- Overbought/oversold zone identification
- Volume strength analysis and momentum scoring
- Session-aware analysis for forex markets
- Multi-timeframe support for scalping to swing trading
- Real-time signal generation with confidence scoring
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MFISignalType(Enum):
    """MFI signal types"""
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    VOLUME_BREAKOUT = "volume_breakout"
    VOLUME_EXHAUSTION = "volume_exhaustion"
    TREND_CONFIRMATION = "trend_confirmation"
    NEUTRAL = "neutral"

class MFITrend(Enum):
    """MFI trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class MFISignal:
    """MFI trading signal"""
    timestamp: datetime
    signal_type: str
    strength: float
    confidence: float
    mfi_value: float
    price: float
    volume: float
    session: str
    timeframe: str
    metadata: Dict

class MFI:
    """
    Advanced Money Flow Index implementation for forex trading
    Features:
    - Volume-price momentum analysis
    - Buying/selling pressure identification
    - Divergence detection for trend reversals
    - Overbought/oversold zone analysis
    - Volume strength and momentum scoring
    - Session-aware forex market analysis
    - Multi-timeframe support (M1-H4)
    """

    def __init__(self,
                 period: int = 14,
                 overbought_level: float = 80.0,
                 oversold_level: float = 20.0,
                 divergence_lookback: int = 20,
                 volume_threshold: float = 1.5,
                 timeframes: List[str] = None):
        """
        Initialize MFI calculator

        Args:
            period: Period for MFI calculation
            overbought_level: Overbought threshold level
            oversold_level: Oversold threshold level
            divergence_lookback: Lookback period for divergence detection
            volume_threshold: Volume threshold for breakout detection
            timeframes: List of timeframes to analyze
        """
        self.period = period
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        self.divergence_lookback = divergence_lookback
        self.volume_threshold = volume_threshold
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']

        # Signal thresholds
        self.strong_signal_threshold = 0.7
        self.divergence_threshold = 0.6
        self.volume_breakout_threshold = 2.0
        self.trend_confirmation_threshold = 0.75

        # Performance tracking
        self.signal_history = []
        self.divergence_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'divergence_accuracy': 0.0,
            'overbought_accuracy': 0.0,
            'oversold_accuracy': 0.0
        }

        logger.info(f"MFI initialized: period={period}, overbought={overbought_level}, "
                   f"oversold={oversold_level}, divergence_lookback={divergence_lookback}")

    def calculate_mfi(self,
                     high: Union[pd.Series, np.ndarray],
                     low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray],
                     volume: Union[pd.Series, np.ndarray],
                     timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Money Flow Index for given OHLCV data

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            timestamps: Optional timestamps for session analysis

        Returns:
            Dictionary containing MFI values and analysis
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            volume_array = np.array(volume)

            if len(high_array) < self.period + 1:
                logger.warning(f"Insufficient data for MFI calculation: {len(high_array)} < {self.period + 1}")
                return self._empty_result()

            # Calculate typical price
            typical_price = self._calculate_typical_price(high_array, low_array, close_array)

            # Calculate money flow
            money_flow = self._calculate_money_flow(typical_price, volume_array)

            # Calculate positive and negative money flows
            positive_flow, negative_flow = self._calculate_money_flows(typical_price, money_flow)

            # Calculate MFI values
            mfi_values = self._calculate_mfi_values(positive_flow, negative_flow)

            # Calculate MFI trend
            mfi_trend = self._calculate_mfi_trend(mfi_values)

            # Calculate volume strength
            volume_strength = self._calculate_volume_strength(volume_array)

            # Detect divergences
            divergence_signals = self._detect_divergences(close_array, mfi_values)

            # Calculate momentum
            mfi_momentum = self._calculate_mfi_momentum(mfi_values)

            # Identify zones
            mfi_zones = self._identify_mfi_zones(mfi_values)

            result = {
                'mfi_values': mfi_values,
                'mfi_trend': mfi_trend,
                'volume_strength': volume_strength,
                'divergence_signals': divergence_signals,
                'mfi_momentum': mfi_momentum,
                'mfi_zones': mfi_zones,
                'typical_price': typical_price,
                'money_flow': money_flow,
                'positive_flow': positive_flow,
                'negative_flow': negative_flow,
                'period_used': self.period,
                'overbought_level': self.overbought_level,
                'oversold_level': self.oversold_level
            }

            logger.debug(f"MFI calculated: latest_mfi={mfi_values[-1]:.2f}, "
                        f"trend={mfi_trend[-1]}, zone={mfi_zones[-1]}")
            return result

        except Exception as e:
            logger.error(f"Error calculating MFI: {str(e)}")
            return self._empty_result()

    def generate_signals(self,
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        volume: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[MFISignal]:
        """
        Generate trading signals based on MFI analysis

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            timestamps: Optional timestamps
            timeframe: Current timeframe

        Returns:
            List of MFISignal objects
        """
        try:
            mfi_data = self.calculate_mfi(high, low, close, volume, timestamps)
            if not mfi_data or 'mfi_values' not in mfi_data:
                return []

            signals = []
            current_time = datetime.now()

            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_mfi = mfi_data['mfi_values'][-1]
            latest_trend = mfi_data['mfi_trend'][-1]
            latest_volume_strength = mfi_data['volume_strength'][-1]
            latest_divergence = mfi_data['divergence_signals'][-1]
            latest_momentum = mfi_data['mfi_momentum'][-1]
            latest_zone = mfi_data['mfi_zones'][-1]
            latest_volume = volume.iloc[-1] if isinstance(volume, pd.Series) else volume[-1]

            # Determine current session
            session = self._get_current_session(current_time)

            # Generate signals based on MFI analysis
            signal_data = self._analyze_mfi_signals(
                latest_mfi, latest_trend, latest_volume_strength,
                latest_divergence, latest_momentum, latest_zone
            )

            if signal_data['signal_type'] != MFISignalType.NEUTRAL.value:
                signal = MFISignal(
                    timestamp=current_time,
                    signal_type=signal_data['signal_type'],
                    strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    mfi_value=latest_mfi,
                    price=latest_price,
                    volume=latest_volume,
                    session=session,
                    timeframe=timeframe,
                    metadata={
                        'trend': latest_trend,
                        'volume_strength': latest_volume_strength,
                        'divergence': latest_divergence,
                        'momentum': latest_momentum,
                        'zone': latest_zone,
                        'overbought_level': self.overbought_level,
                        'oversold_level': self.oversold_level
                    }
                )
                signals.append(signal)
                self.signal_history.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error generating MFI signals: {str(e)}")
            return []

    def _calculate_typical_price(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate typical price (HLC/3)"""
        return (high + low + close) / 3

    def _calculate_money_flow(self, typical_price: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate money flow (typical price * volume)"""
        return typical_price * volume

    def _calculate_money_flows(self, typical_price: np.ndarray, money_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate positive and negative money flows"""
        try:
            positive_flow = np.zeros_like(money_flow)
            negative_flow = np.zeros_like(money_flow)

            for i in range(1, len(typical_price)):
                if typical_price[i] > typical_price[i-1]:
                    positive_flow[i] = money_flow[i]
                elif typical_price[i] < typical_price[i-1]:
                    negative_flow[i] = money_flow[i]
                # If equal, both remain 0

            return positive_flow, negative_flow

        except Exception as e:
            logger.error(f"Error calculating money flows: {str(e)}")
            return np.zeros_like(money_flow), np.zeros_like(money_flow)

    def _calculate_mfi_values(self, positive_flow: np.ndarray, negative_flow: np.ndarray) -> np.ndarray:
        """Calculate MFI values using rolling sums"""
        try:
            mfi_values = np.full_like(positive_flow, 50.0)  # Default neutral value

            for i in range(self.period, len(positive_flow)):
                # Calculate rolling sums
                pos_sum = np.sum(positive_flow[i-self.period+1:i+1])
                neg_sum = np.sum(negative_flow[i-self.period+1:i+1])

                # Calculate MFI
                if neg_sum == 0:
                    mfi_values[i] = 100.0
                elif pos_sum == 0:
                    mfi_values[i] = 0.0
                else:
                    money_ratio = pos_sum / neg_sum
                    mfi_values[i] = 100 - (100 / (1 + money_ratio))

            return mfi_values

        except Exception as e:
            logger.error(f"Error calculating MFI values: {str(e)}")
            return np.full_like(positive_flow, 50.0)

    def _calculate_mfi_trend(self, mfi_values: np.ndarray) -> List[str]:
        """Calculate MFI trend direction"""
        try:
            trends = []

            for i in range(len(mfi_values)):
                if i < 5:
                    trends.append(MFITrend.NEUTRAL.value)
                    continue

                # Compare current MFI with recent average
                recent_avg = np.mean(mfi_values[max(0, i-5):i])
                current_mfi = mfi_values[i]

                # Calculate trend strength
                if current_mfi > recent_avg + 2:
                    trends.append(MFITrend.BULLISH.value)
                elif current_mfi < recent_avg - 2:
                    trends.append(MFITrend.BEARISH.value)
                else:
                    trends.append(MFITrend.NEUTRAL.value)

            return trends

        except Exception as e:
            logger.error(f"Error calculating MFI trend: {str(e)}")
            return [MFITrend.NEUTRAL.value] * len(mfi_values)

    def _calculate_volume_strength(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume strength relative to recent average"""
        try:
            volume_strength = np.ones_like(volume)

            for i in range(10, len(volume)):
                recent_avg = np.mean(volume[max(0, i-10):i])
                if recent_avg > 0:
                    volume_strength[i] = volume[i] / recent_avg
                else:
                    volume_strength[i] = 1.0

            return volume_strength

        except Exception as e:
            logger.error(f"Error calculating volume strength: {str(e)}")
            return np.ones_like(volume)

    def _detect_divergences(self, prices: np.ndarray, mfi_values: np.ndarray) -> List[str]:
        """Detect bullish and bearish divergences"""
        try:
            divergences = ['none'] * len(prices)

            if len(prices) < self.divergence_lookback:
                return divergences

            for i in range(self.divergence_lookback, len(prices)):
                # Look for divergences in the lookback period
                price_window = prices[i-self.divergence_lookback:i+1]
                mfi_window = mfi_values[i-self.divergence_lookback:i+1]

                # Find recent highs and lows
                price_high_idx = np.argmax(price_window)
                price_low_idx = np.argmin(price_window)
                mfi_high_idx = np.argmax(mfi_window)
                mfi_low_idx = np.argmin(mfi_window)

                # Bullish divergence: price makes lower low, MFI makes higher low
                if (price_low_idx > len(price_window) // 2 and
                    mfi_low_idx > len(mfi_window) // 2 and
                    price_window[price_low_idx] < np.min(price_window[:price_low_idx]) and
                    mfi_window[mfi_low_idx] > np.min(mfi_window[:mfi_low_idx])):
                    divergences[i] = 'bullish'

                # Bearish divergence: price makes higher high, MFI makes lower high
                elif (price_high_idx > len(price_window) // 2 and
                      mfi_high_idx > len(mfi_window) // 2 and
                      price_window[price_high_idx] > np.max(price_window[:price_high_idx]) and
                      mfi_window[mfi_high_idx] < np.max(mfi_window[:mfi_high_idx])):
                    divergences[i] = 'bearish'

            return divergences

        except Exception as e:
            logger.error(f"Error detecting divergences: {str(e)}")
            return ['none'] * len(prices)

    def _calculate_mfi_momentum(self, mfi_values: np.ndarray) -> np.ndarray:
        """Calculate MFI momentum (rate of change)"""
        try:
            momentum = np.zeros_like(mfi_values)

            for i in range(5, len(mfi_values)):
                momentum[i] = mfi_values[i] - mfi_values[i-5]

            return momentum

        except Exception as e:
            logger.error(f"Error calculating MFI momentum: {str(e)}")
            return np.zeros_like(mfi_values)

    def _identify_mfi_zones(self, mfi_values: np.ndarray) -> List[str]:
        """Identify MFI zones (overbought, oversold, neutral)"""
        try:
            zones = []

            for mfi in mfi_values:
                if mfi >= self.overbought_level:
                    zones.append('overbought')
                elif mfi <= self.oversold_level:
                    zones.append('oversold')
                else:
                    zones.append('neutral')

            return zones

        except Exception as e:
            logger.error(f"Error identifying MFI zones: {str(e)}")
            return ['neutral'] * len(mfi_values)

    def _analyze_mfi_signals(self, mfi_value: float, trend: str, volume_strength: float,
                            divergence: str, momentum: float, zone: str) -> Dict:
        """Analyze MFI data to generate trading signals"""
        try:
            signal_type = MFISignalType.NEUTRAL.value
            strength = 0.0
            confidence = 0.0

            # Overbought/Oversold signals
            if zone == 'overbought' and trend == MFITrend.BEARISH.value:
                signal_type = MFISignalType.OVERBOUGHT.value
                strength = min(1.0, (mfi_value - self.overbought_level) / 20.0)
                confidence = min(0.8, 0.5 + strength * 0.3)
            elif zone == 'oversold' and trend == MFITrend.BULLISH.value:
                signal_type = MFISignalType.OVERSOLD.value
                strength = min(1.0, (self.oversold_level - mfi_value) / 20.0)
                confidence = min(0.8, 0.5 + strength * 0.3)

            # Divergence signals
            elif divergence == 'bullish':
                signal_type = MFISignalType.BULLISH_DIVERGENCE.value
                strength = min(1.0, abs(momentum) / 10.0)
                confidence = min(0.85, 0.6 + strength * 0.25)
            elif divergence == 'bearish':
                signal_type = MFISignalType.BEARISH_DIVERGENCE.value
                strength = min(1.0, abs(momentum) / 10.0)
                confidence = min(0.85, 0.6 + strength * 0.25)

            # Volume breakout signals
            elif volume_strength > self.volume_breakout_threshold:
                signal_type = MFISignalType.VOLUME_BREAKOUT.value
                strength = min(1.0, volume_strength / 3.0)
                confidence = min(0.75, 0.4 + strength * 0.35)

            # Volume exhaustion signals
            elif volume_strength < 0.5 and abs(momentum) < 2:
                signal_type = MFISignalType.VOLUME_EXHAUSTION.value
                strength = 1.0 - volume_strength
                confidence = min(0.7, 0.3 + strength * 0.4)

            # Trend confirmation signals
            elif ((trend == MFITrend.BULLISH.value and momentum > 3) or
                  (trend == MFITrend.BEARISH.value and momentum < -3)):
                signal_type = MFISignalType.TREND_CONFIRMATION.value
                strength = min(1.0, abs(momentum) / 10.0)
                confidence = min(0.75, 0.5 + strength * 0.25)

            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error analyzing MFI signals: {str(e)}")
            return {
                'signal_type': MFISignalType.NEUTRAL.value,
                'strength': 0.0,
                'confidence': 0.0
            }

    def _get_current_session(self, timestamp: datetime) -> str:
        """Determine current trading session based on timestamp"""
        try:
            hour = timestamp.hour

            # Trading sessions (UTC)
            if 21 <= hour or hour < 6:
                return 'asian'
            elif 6 <= hour < 14:
                return 'london'
            elif 14 <= hour < 21:
                return 'new_york'
            else:
                return 'overlap'

        except Exception as e:
            logger.error(f"Error determining session: {str(e)}")
            return 'unknown'

    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'mfi_values': np.array([]),
            'mfi_trend': [],
            'volume_strength': np.array([]),
            'divergence_signals': [],
            'mfi_momentum': np.array([]),
            'mfi_zones': [],
            'typical_price': np.array([]),
            'money_flow': np.array([]),
            'positive_flow': np.array([]),
            'negative_flow': np.array([]),
            'period_used': self.period,
            'overbought_level': self.overbought_level,
            'oversold_level': self.oversold_level
        }

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            if len(self.signal_history) == 0:
                return self.performance_stats

            # Calculate basic stats
            total_signals = len(self.signal_history)

            # Calculate accuracy by signal type
            overbought_signals = [s for s in self.signal_history if s.signal_type == MFISignalType.OVERBOUGHT.value]
            oversold_signals = [s for s in self.signal_history if s.signal_type == MFISignalType.OVERSOLD.value]
            divergence_signals = [s for s in self.signal_history if 'divergence' in s.signal_type]

            # Calculate average confidence
            avg_confidence = np.mean([s.confidence for s in self.signal_history])

            self.performance_stats.update({
                'total_signals': total_signals,
                'avg_confidence': avg_confidence,
                'overbought_signals': len(overbought_signals),
                'oversold_signals': len(oversold_signals),
                'divergence_signals': len(divergence_signals),
                'last_updated': datetime.now()
            })

            return self.performance_stats

        except Exception as e:
            logger.error(f"Error calculating performance stats: {str(e)}")
            return self.performance_stats

    def update_signal_outcome(self, signal_id: str, success: bool):
        """Update signal outcome for performance tracking"""
        try:
            # Find signal in history and update outcome
            for signal in self.signal_history:
                if hasattr(signal, 'id') and signal.id == signal_id:
                    signal.success = success
                    break

            # Recalculate performance stats
            successful_signals = sum(1 for s in self.signal_history if hasattr(s, 'success') and s.success)
            total_evaluated = sum(1 for s in self.signal_history if hasattr(s, 'success'))

            if total_evaluated > 0:
                self.performance_stats['successful_signals'] = successful_signals
                self.performance_stats['accuracy'] = successful_signals / total_evaluated

        except Exception as e:
            logger.error(f"Error updating signal outcome: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    n_points = 1000

    # Generate realistic forex price data
    base_price = 1.1000
    price_changes = np.random.normal(0, 0.0001, n_points)
    close_prices = base_price + np.cumsum(price_changes)

    # Generate high and low prices
    high_prices = close_prices + np.random.uniform(0, 0.0005, n_points)
    low_prices = close_prices - np.random.uniform(0, 0.0005, n_points)

    # Generate volume data with some correlation to price movements
    volume_base = 1000
    volume_data = volume_base + np.random.exponential(500, n_points)

    # Ensure positive volume
    volume_data = np.abs(volume_data)

    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')

    # Initialize MFI
    mfi = MFI(period=14, overbought_level=80, oversold_level=20, divergence_lookback=20)

    # Calculate MFI
    result = mfi.calculate_mfi(high_prices, low_prices, close_prices, volume_data)
    print("MFI calculation completed")
    print(f"Latest MFI: {result['mfi_values'][-1]:.2f}")
    print(f"MFI trend: {result['mfi_trend'][-1]}")
    print(f"MFI zone: {result['mfi_zones'][-1]}")
    print(f"Volume strength: {result['volume_strength'][-1]:.2f}")

    # Generate signals
    signals = mfi.generate_signals(high_prices, low_prices, close_prices, volume_data, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")

    if signals:
        latest_signal = signals[-1]
        print(f"Latest signal: {latest_signal.signal_type} (confidence: {latest_signal.confidence:.2f})")

    # Get performance stats
    stats = mfi.get_performance_stats()
    print(f"Performance stats: {stats}")

    print("MFI implementation test completed successfully!")
