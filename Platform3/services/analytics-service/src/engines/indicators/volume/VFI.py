"""
Volume Flow Indicator (VFI) Implementation for Platform3 Forex Trading
Advanced volume flow analysis for directional volume measurement

Features:
- Volume flow calculation based on price and volume relationship
- Directional volume analysis for trend confirmation
- Volume accumulation and distribution detection
- Smoothing options for noise reduction
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

class VFISignalType(Enum):
    """VFI signal types"""
    BULLISH_FLOW = "bullish_flow"
    BEARISH_FLOW = "bearish_flow"
    VOLUME_ACCUMULATION = "volume_accumulation"
    VOLUME_DISTRIBUTION = "volume_distribution"
    FLOW_REVERSAL = "flow_reversal"
    FLOW_EXHAUSTION = "flow_exhaustion"
    TREND_CONFIRMATION = "trend_confirmation"
    NEUTRAL = "neutral"

class VFITrend(Enum):
    """VFI trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class VFISignal:
    """VFI trading signal"""
    timestamp: datetime
    signal_type: str
    strength: float
    confidence: float
    vfi_value: float
    price: float
    volume: float
    session: str
    timeframe: str
    metadata: Dict

class VFI:
    """
    Advanced Volume Flow Indicator implementation for forex trading
    Features:
    - Directional volume flow analysis
    - Volume accumulation/distribution detection
    - Flow reversal identification
    - Trend confirmation through volume flow
    - Session-aware forex market analysis
    - Multi-timeframe support (M1-H4)
    """

    def __init__(self,
                 period: int = 130,
                 smoothing_period: int = 3,
                 volume_cutoff: float = 2.5,
                 price_cutoff: float = 0.2,
                 timeframes: List[str] = None):
        """
        Initialize VFI calculator

        Args:
            period: Period for VFI calculation
            smoothing_period: Period for smoothing the VFI
            volume_cutoff: Volume cutoff multiplier for significant volume
            price_cutoff: Price change cutoff for directional flow
            timeframes: List of timeframes to analyze
        """
        self.period = period
        self.smoothing_period = smoothing_period
        self.volume_cutoff = volume_cutoff
        self.price_cutoff = price_cutoff
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']

        # Signal thresholds
        self.strong_flow_threshold = 0.7
        self.reversal_threshold = 0.6
        self.accumulation_threshold = 0.8
        self.trend_confirmation_threshold = 0.75

        # Performance tracking
        self.signal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'flow_accuracy': 0.0,
            'reversal_accuracy': 0.0
        }

        logger.info(f"VFI initialized: period={period}, smoothing={smoothing_period}, "
                   f"volume_cutoff={volume_cutoff}, price_cutoff={price_cutoff}")

    def calculate_vfi(self,
                     high: Union[pd.Series, np.ndarray],
                     low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray],
                     volume: Union[pd.Series, np.ndarray],
                     timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Volume Flow Indicator for given OHLCV data

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            timestamps: Optional timestamps for session analysis

        Returns:
            Dictionary containing VFI values and analysis
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            volume_array = np.array(volume)

            if len(close_array) < self.period + 1:
                logger.warning(f"Insufficient data for VFI calculation: {len(close_array)} < {self.period + 1}")
                return self._empty_result()

            # Calculate typical price
            typical_price = self._calculate_typical_price(high_array, low_array, close_array)

            # Calculate price changes
            price_changes = self._calculate_price_changes(typical_price)

            # Calculate volume cutoff
            volume_cutoff_values = self._calculate_volume_cutoff(volume_array)

            # Calculate raw VFI
            raw_vfi = self._calculate_raw_vfi(price_changes, volume_array, volume_cutoff_values)

            # Calculate smoothed VFI
            vfi_values = self._calculate_smoothed_vfi(raw_vfi)

            # Calculate VFI trend
            vfi_trend = self._calculate_vfi_trend(vfi_values)

            # Calculate flow strength
            flow_strength = self._calculate_flow_strength(vfi_values)

            # Detect flow reversals
            flow_reversals = self._detect_flow_reversals(vfi_values)

            # Calculate accumulation/distribution
            accumulation_distribution = self._calculate_accumulation_distribution(vfi_values, volume_array)

            result = {
                'vfi_values': vfi_values,
                'raw_vfi': raw_vfi,
                'vfi_trend': vfi_trend,
                'flow_strength': flow_strength,
                'flow_reversals': flow_reversals,
                'accumulation_distribution': accumulation_distribution,
                'typical_price': typical_price,
                'price_changes': price_changes,
                'volume_cutoff_values': volume_cutoff_values,
                'period_used': self.period,
                'smoothing_period_used': self.smoothing_period
            }

            logger.debug(f"VFI calculated: latest_vfi={vfi_values[-1]:.4f}, "
                        f"trend={vfi_trend[-1]}, flow_strength={flow_strength[-1]:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error calculating VFI: {str(e)}")
            return self._empty_result()

    def generate_signals(self,
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        volume: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[VFISignal]:
        """
        Generate trading signals based on VFI analysis

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            timestamps: Optional timestamps
            timeframe: Current timeframe

        Returns:
            List of VFISignal objects
        """
        try:
            vfi_data = self.calculate_vfi(high, low, close, volume, timestamps)
            if not vfi_data or 'vfi_values' not in vfi_data:
                return []

            signals = []
            current_time = datetime.now()

            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_vfi = vfi_data['vfi_values'][-1]
            latest_trend = vfi_data['vfi_trend'][-1]
            latest_flow_strength = vfi_data['flow_strength'][-1]
            latest_reversal = vfi_data['flow_reversals'][-1]
            latest_acc_dist = vfi_data['accumulation_distribution'][-1]
            latest_volume = volume.iloc[-1] if isinstance(volume, pd.Series) else volume[-1]

            # Determine current session
            session = self._get_current_session(current_time)

            # Generate signals based on VFI analysis
            signal_data = self._analyze_vfi_signals(
                latest_vfi, latest_trend, latest_flow_strength,
                latest_reversal, latest_acc_dist
            )

            if signal_data['signal_type'] != VFISignalType.NEUTRAL.value:
                signal = VFISignal(
                    timestamp=current_time,
                    signal_type=signal_data['signal_type'],
                    strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    vfi_value=latest_vfi,
                    price=latest_price,
                    volume=latest_volume,
                    session=session,
                    timeframe=timeframe,
                    metadata={
                        'trend': latest_trend,
                        'flow_strength': latest_flow_strength,
                        'reversal': latest_reversal,
                        'accumulation_distribution': latest_acc_dist,
                        'volume_cutoff': self.volume_cutoff,
                        'price_cutoff': self.price_cutoff
                    }
                )
                signals.append(signal)
                self.signal_history.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error generating VFI signals: {str(e)}")
            return []

    def _calculate_typical_price(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate typical price (HLC/3)"""
        return (high + low + close) / 3

    def _calculate_price_changes(self, typical_price: np.ndarray) -> np.ndarray:
        """Calculate price changes between periods"""
        try:
            price_changes = np.zeros_like(typical_price)

            for i in range(1, len(typical_price)):
                if typical_price[i-1] != 0:
                    price_changes[i] = (typical_price[i] - typical_price[i-1]) / typical_price[i-1] * 100

            return price_changes

        except Exception as e:
            logger.error(f"Error calculating price changes: {str(e)}")
            return np.zeros_like(typical_price)

    def _calculate_volume_cutoff(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume cutoff based on moving average"""
        try:
            volume_cutoff = np.zeros_like(volume)

            for i in range(30, len(volume)):
                avg_volume = np.mean(volume[max(0, i-30):i])
                volume_cutoff[i] = avg_volume * self.volume_cutoff

            return volume_cutoff

        except Exception as e:
            logger.error(f"Error calculating volume cutoff: {str(e)}")
            return np.ones_like(volume) * np.mean(volume) * self.volume_cutoff

    def _calculate_raw_vfi(self, price_changes: np.ndarray, volume: np.ndarray,
                          volume_cutoff: np.ndarray) -> np.ndarray:
        """Calculate raw VFI values"""
        try:
            raw_vfi = np.zeros_like(price_changes)

            for i in range(len(price_changes)):
                # Check if volume is significant and price change is meaningful
                if (volume[i] > volume_cutoff[i] and
                    abs(price_changes[i]) > self.price_cutoff):

                    # Determine direction based on price change
                    if price_changes[i] > 0:
                        raw_vfi[i] = volume[i]  # Positive flow
                    else:
                        raw_vfi[i] = -volume[i]  # Negative flow
                else:
                    raw_vfi[i] = 0  # No significant flow

            return raw_vfi

        except Exception as e:
            logger.error(f"Error calculating raw VFI: {str(e)}")
            return np.zeros_like(price_changes)

    def _calculate_smoothed_vfi(self, raw_vfi: np.ndarray) -> np.ndarray:
        """Calculate smoothed VFI using moving average"""
        try:
            smoothed_vfi = np.zeros_like(raw_vfi)

            for i in range(self.period, len(raw_vfi)):
                # Calculate sum over period
                vfi_sum = np.sum(raw_vfi[i-self.period+1:i+1])

                # Apply smoothing
                if i >= self.period + self.smoothing_period - 1:
                    smoothed_vfi[i] = np.mean(
                        [vfi_sum] + [smoothed_vfi[j] for j in range(
                            max(0, i-self.smoothing_period+1), i
                        )]
                    )
                else:
                    smoothed_vfi[i] = vfi_sum

            return smoothed_vfi

        except Exception as e:
            logger.error(f"Error calculating smoothed VFI: {str(e)}")
            return np.zeros_like(raw_vfi)

    def _calculate_vfi_trend(self, vfi_values: np.ndarray) -> List[str]:
        """Calculate VFI trend direction"""
        try:
            trends = []

            for i in range(len(vfi_values)):
                if i < 10:
                    trends.append(VFITrend.NEUTRAL.value)
                    continue

                # Compare current VFI with recent values
                recent_avg = np.mean(vfi_values[max(0, i-10):i])
                current_vfi = vfi_values[i]

                # Determine trend
                if current_vfi > recent_avg * 1.1:
                    trends.append(VFITrend.BULLISH.value)
                elif current_vfi < recent_avg * 0.9:
                    trends.append(VFITrend.BEARISH.value)
                else:
                    trends.append(VFITrend.NEUTRAL.value)

            return trends

        except Exception as e:
            logger.error(f"Error calculating VFI trend: {str(e)}")
            return [VFITrend.NEUTRAL.value] * len(vfi_values)

    def _calculate_flow_strength(self, vfi_values: np.ndarray) -> np.ndarray:
        """Calculate flow strength based on VFI magnitude"""
        try:
            flow_strength = np.zeros_like(vfi_values)

            for i in range(20, len(vfi_values)):
                # Calculate recent VFI range
                recent_vfi = vfi_values[max(0, i-20):i+1]
                vfi_range = np.max(recent_vfi) - np.min(recent_vfi)

                if vfi_range > 0:
                    # Normalize current VFI relative to recent range
                    flow_strength[i] = abs(vfi_values[i]) / vfi_range
                else:
                    flow_strength[i] = 0.0

            return flow_strength

        except Exception as e:
            logger.error(f"Error calculating flow strength: {str(e)}")
            return np.zeros_like(vfi_values)

    def _detect_flow_reversals(self, vfi_values: np.ndarray) -> List[str]:
        """Detect flow reversals in VFI"""
        try:
            reversals = ['none'] * len(vfi_values)

            for i in range(10, len(vfi_values)):
                # Look for sign changes in VFI trend
                recent_vfi = vfi_values[max(0, i-10):i+1]

                # Check for bullish reversal (negative to positive)
                if (vfi_values[i] > 0 and
                    np.mean(recent_vfi[:-1]) < 0 and
                    vfi_values[i] > np.max(recent_vfi[:-1])):
                    reversals[i] = 'bullish'

                # Check for bearish reversal (positive to negative)
                elif (vfi_values[i] < 0 and
                      np.mean(recent_vfi[:-1]) > 0 and
                      vfi_values[i] < np.min(recent_vfi[:-1])):
                    reversals[i] = 'bearish'

            return reversals

        except Exception as e:
            logger.error(f"Error detecting flow reversals: {str(e)}")
            return ['none'] * len(vfi_values)

    def _calculate_accumulation_distribution(self, vfi_values: np.ndarray,
                                           volume: np.ndarray) -> List[str]:
        """Calculate accumulation/distribution based on VFI and volume"""
        try:
            acc_dist = ['neutral'] * len(vfi_values)

            for i in range(20, len(vfi_values)):
                # Calculate recent VFI and volume trends
                recent_vfi = vfi_values[max(0, i-20):i+1]
                recent_volume = volume[max(0, i-20):i+1]

                vfi_trend = np.mean(recent_vfi[-10:]) - np.mean(recent_vfi[:10])
                volume_trend = np.mean(recent_volume[-10:]) / np.mean(recent_volume[:10])

                # Accumulation: positive VFI trend with increasing volume
                if vfi_trend > 0 and volume_trend > 1.1:
                    acc_dist[i] = 'accumulation'

                # Distribution: negative VFI trend with increasing volume
                elif vfi_trend < 0 and volume_trend > 1.1:
                    acc_dist[i] = 'distribution'

            return acc_dist

        except Exception as e:
            logger.error(f"Error calculating accumulation/distribution: {str(e)}")
            return ['neutral'] * len(vfi_values)

    def _analyze_vfi_signals(self, vfi_value: float, trend: str, flow_strength: float,
                            reversal: str, acc_dist: str) -> Dict:
        """Analyze VFI data to generate trading signals"""
        try:
            signal_type = VFISignalType.NEUTRAL.value
            strength = 0.0
            confidence = 0.0

            # Flow direction signals
            if trend == VFITrend.BULLISH.value and flow_strength > self.strong_flow_threshold:
                signal_type = VFISignalType.BULLISH_FLOW.value
                strength = min(1.0, flow_strength)
                confidence = min(0.8, 0.5 + strength * 0.3)
            elif trend == VFITrend.BEARISH.value and flow_strength > self.strong_flow_threshold:
                signal_type = VFISignalType.BEARISH_FLOW.value
                strength = min(1.0, flow_strength)
                confidence = min(0.8, 0.5 + strength * 0.3)

            # Flow reversal signals
            elif reversal == 'bullish':
                signal_type = VFISignalType.FLOW_REVERSAL.value
                strength = min(1.0, flow_strength * 1.5)
                confidence = min(0.85, 0.6 + strength * 0.25)
            elif reversal == 'bearish':
                signal_type = VFISignalType.FLOW_REVERSAL.value
                strength = min(1.0, flow_strength * 1.5)
                confidence = min(0.85, 0.6 + strength * 0.25)

            # Accumulation/Distribution signals
            elif acc_dist == 'accumulation':
                signal_type = VFISignalType.VOLUME_ACCUMULATION.value
                strength = min(1.0, flow_strength * 1.2)
                confidence = min(0.75, 0.5 + strength * 0.25)
            elif acc_dist == 'distribution':
                signal_type = VFISignalType.VOLUME_DISTRIBUTION.value
                strength = min(1.0, flow_strength * 1.2)
                confidence = min(0.75, 0.5 + strength * 0.25)

            # Flow exhaustion signals
            elif flow_strength < 0.3 and abs(vfi_value) < 0.1:
                signal_type = VFISignalType.FLOW_EXHAUSTION.value
                strength = 1.0 - flow_strength
                confidence = min(0.7, 0.4 + strength * 0.3)

            # Trend confirmation signals
            elif ((trend == VFITrend.BULLISH.value and vfi_value > 0) or
                  (trend == VFITrend.BEARISH.value and vfi_value < 0)):
                signal_type = VFISignalType.TREND_CONFIRMATION.value
                strength = min(1.0, flow_strength * 0.8)
                confidence = min(0.75, 0.5 + strength * 0.25)

            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error analyzing VFI signals: {str(e)}")
            return {
                'signal_type': VFISignalType.NEUTRAL.value,
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
            'vfi_values': np.array([]),
            'raw_vfi': np.array([]),
            'vfi_trend': [],
            'flow_strength': np.array([]),
            'flow_reversals': [],
            'accumulation_distribution': [],
            'typical_price': np.array([]),
            'price_changes': np.array([]),
            'volume_cutoff_values': np.array([]),
            'period_used': self.period,
            'smoothing_period_used': self.smoothing_period
        }

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            if len(self.signal_history) == 0:
                return self.performance_stats

            # Calculate basic stats
            total_signals = len(self.signal_history)

            # Calculate accuracy by signal type
            flow_signals = [s for s in self.signal_history if 'flow' in s.signal_type]
            reversal_signals = [s for s in self.signal_history if 'reversal' in s.signal_type]
            accumulation_signals = [s for s in self.signal_history if 'accumulation' in s.signal_type or 'distribution' in s.signal_type]

            # Calculate average confidence
            avg_confidence = np.mean([s.confidence for s in self.signal_history])

            self.performance_stats.update({
                'total_signals': total_signals,
                'avg_confidence': avg_confidence,
                'flow_signals': len(flow_signals),
                'reversal_signals': len(reversal_signals),
                'accumulation_signals': len(accumulation_signals),
                'last_updated': datetime.now()
            })

            return self.performance_stats

        except Exception as e:
            logger.error(f"Error calculating performance stats: {str(e)}")
            return self.performance_stats


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

    # Initialize VFI
    vfi = VFI(period=130, smoothing_period=3, volume_cutoff=2.5, price_cutoff=0.2)

    # Calculate VFI
    result = vfi.calculate_vfi(high_prices, low_prices, close_prices, volume_data)
    print("VFI calculation completed")
    print(f"Latest VFI: {result['vfi_values'][-1]:.4f}")
    print(f"VFI trend: {result['vfi_trend'][-1]}")
    print(f"Flow strength: {result['flow_strength'][-1]:.2f}")
    print(f"Accumulation/Distribution: {result['accumulation_distribution'][-1]}")

    # Generate signals
    signals = vfi.generate_signals(high_prices, low_prices, close_prices, volume_data, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")

    if signals:
        latest_signal = signals[-1]
        print(f"Latest signal: {latest_signal.signal_type} (confidence: {latest_signal.confidence:.2f})")

    # Get performance stats
    stats = vfi.get_performance_stats()
    print(f"Performance stats: {stats}")

    print("VFI implementation test completed successfully!")
