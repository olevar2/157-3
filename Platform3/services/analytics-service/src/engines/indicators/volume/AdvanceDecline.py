"""
Advance/Decline Line Implementation for Platform3 Forex Trading
Market breadth indicator for analyzing overall market sentiment and strength

Features:
- Advance/Decline ratio calculation for market breadth
- Cumulative advance/decline line for trend analysis
- Market sentiment scoring and momentum analysis
- Divergence detection for trend reversal signals
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

class ADSignalType(Enum):
    """Advance/Decline signal types"""
    BULLISH_BREADTH = "bullish_breadth"
    BEARISH_BREADTH = "bearish_breadth"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    BREADTH_EXPANSION = "breadth_expansion"
    BREADTH_CONTRACTION = "breadth_contraction"
    MOMENTUM_SHIFT = "momentum_shift"
    NEUTRAL = "neutral"

class ADTrend(Enum):
    """Advance/Decline trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class ADSignal:
    """Advance/Decline trading signal"""
    timestamp: datetime
    signal_type: str
    strength: float
    confidence: float
    ad_line: float
    ad_ratio: float
    price: float
    session: str
    timeframe: str
    metadata: Dict

class AdvanceDecline:
    """
    Advanced Advance/Decline Line implementation for forex trading
    Features:
    - Market breadth analysis through advance/decline ratios
    - Cumulative advance/decline line calculation
    - Market sentiment and momentum scoring
    - Divergence detection for trend reversals
    - Session-aware forex market analysis
    - Multi-timeframe support (M1-H4)
    """

    def __init__(self,
                 lookback_period: int = 20,
                 smoothing_period: int = 5,
                 divergence_lookback: int = 30,
                 breadth_threshold: float = 0.6,
                 timeframes: List[str] = None):
        """
        Initialize Advance/Decline calculator

        Args:
            lookback_period: Period for calculating advance/decline ratios
            smoothing_period: Period for smoothing the A/D line
            divergence_lookback: Lookback period for divergence detection
            breadth_threshold: Threshold for significant breadth signals
            timeframes: List of timeframes to analyze
        """
        self.lookback_period = lookback_period
        self.smoothing_period = smoothing_period
        self.divergence_lookback = divergence_lookback
        self.breadth_threshold = breadth_threshold
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']

        # Signal thresholds
        self.strong_breadth_threshold = 0.7
        self.divergence_threshold = 0.6
        self.momentum_threshold = 0.8
        self.expansion_threshold = 1.5

        # Performance tracking
        self.signal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'breadth_accuracy': 0.0,
            'divergence_accuracy': 0.0
        }

        logger.info(f"AdvanceDecline initialized: lookback={lookback_period}, "
                   f"smoothing={smoothing_period}, divergence_lookback={divergence_lookback}")

    def calculate_advance_decline(self,
                                 high: Union[pd.Series, np.ndarray],
                                 low: Union[pd.Series, np.ndarray],
                                 close: Union[pd.Series, np.ndarray],
                                 volume: Union[pd.Series, np.ndarray],
                                 timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Advance/Decline Line for given OHLCV data

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            timestamps: Optional timestamps for session analysis

        Returns:
            Dictionary containing A/D values and analysis
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            volume_array = np.array(volume)

            if len(close_array) < self.lookback_period + 1:
                logger.warning(f"Insufficient data for A/D calculation: {len(close_array)} < {self.lookback_period + 1}")
                return self._empty_result()

            # Calculate price changes
            price_changes = self._calculate_price_changes(close_array)

            # Calculate advances and declines
            advances, declines = self._calculate_advances_declines(price_changes, volume_array)

            # Calculate advance/decline ratios
            ad_ratios = self._calculate_ad_ratios(advances, declines)

            # Calculate cumulative A/D line
            ad_line = self._calculate_ad_line(advances, declines)

            # Calculate smoothed A/D line
            ad_line_smoothed = self._calculate_smoothed_ad_line(ad_line)

            # Calculate A/D trend
            ad_trend = self._calculate_ad_trend(ad_line_smoothed)

            # Calculate breadth momentum
            breadth_momentum = self._calculate_breadth_momentum(ad_ratios)

            # Detect divergences
            divergence_signals = self._detect_divergences(close_array, ad_line_smoothed)

            # Calculate market sentiment
            market_sentiment = self._calculate_market_sentiment(ad_ratios, breadth_momentum)

            result = {
                'ad_line': ad_line,
                'ad_line_smoothed': ad_line_smoothed,
                'ad_ratios': ad_ratios,
                'advances': advances,
                'declines': declines,
                'ad_trend': ad_trend,
                'breadth_momentum': breadth_momentum,
                'divergence_signals': divergence_signals,
                'market_sentiment': market_sentiment,
                'price_changes': price_changes,
                'lookback_period_used': self.lookback_period,
                'smoothing_period_used': self.smoothing_period
            }

            logger.debug(f"A/D calculated: latest_ad_line={ad_line[-1]:.2f}, "
                        f"ad_ratio={ad_ratios[-1]:.2f}, trend={ad_trend[-1]}")
            return result

        except Exception as e:
            logger.error(f"Error calculating A/D: {str(e)}")
            return self._empty_result()

    def generate_signals(self,
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        volume: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[ADSignal]:
        """
        Generate trading signals based on A/D analysis

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            timestamps: Optional timestamps
            timeframe: Current timeframe

        Returns:
            List of ADSignal objects
        """
        try:
            ad_data = self.calculate_advance_decline(high, low, close, volume, timestamps)
            if not ad_data or 'ad_line' not in ad_data:
                return []

            signals = []
            current_time = datetime.now()

            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_ad_line = ad_data['ad_line'][-1]
            latest_ad_ratio = ad_data['ad_ratios'][-1]
            latest_trend = ad_data['ad_trend'][-1]
            latest_momentum = ad_data['breadth_momentum'][-1]
            latest_divergence = ad_data['divergence_signals'][-1]
            latest_sentiment = ad_data['market_sentiment'][-1]

            # Determine current session
            session = self._get_current_session(current_time)

            # Generate signals based on A/D analysis
            signal_data = self._analyze_ad_signals(
                latest_ad_line, latest_ad_ratio, latest_trend,
                latest_momentum, latest_divergence, latest_sentiment
            )

            if signal_data['signal_type'] != ADSignalType.NEUTRAL.value:
                signal = ADSignal(
                    timestamp=current_time,
                    signal_type=signal_data['signal_type'],
                    strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    ad_line=latest_ad_line,
                    ad_ratio=latest_ad_ratio,
                    price=latest_price,
                    session=session,
                    timeframe=timeframe,
                    metadata={
                        'trend': latest_trend,
                        'momentum': latest_momentum,
                        'divergence': latest_divergence,
                        'sentiment': latest_sentiment,
                        'breadth_threshold': self.breadth_threshold
                    }
                )
                signals.append(signal)
                self.signal_history.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error generating A/D signals: {str(e)}")
            return []

    def _calculate_price_changes(self, close: np.ndarray) -> np.ndarray:
        """Calculate price changes between periods"""
        try:
            price_changes = np.zeros_like(close)

            for i in range(1, len(close)):
                price_changes[i] = close[i] - close[i-1]

            return price_changes

        except Exception as e:
            logger.error(f"Error calculating price changes: {str(e)}")
            return np.zeros_like(close)

    def _calculate_advances_declines(self, price_changes: np.ndarray,
                                   volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate advances and declines based on price changes and volume"""
        try:
            advances = np.zeros_like(price_changes)
            declines = np.zeros_like(price_changes)

            for i in range(len(price_changes)):
                if price_changes[i] > 0:
                    advances[i] = volume[i]  # Volume-weighted advance
                elif price_changes[i] < 0:
                    declines[i] = volume[i]  # Volume-weighted decline
                # If no change, both remain 0

            return advances, declines

        except Exception as e:
            logger.error(f"Error calculating advances/declines: {str(e)}")
            return np.zeros_like(price_changes), np.zeros_like(price_changes)

    def _calculate_ad_ratios(self, advances: np.ndarray, declines: np.ndarray) -> np.ndarray:
        """Calculate advance/decline ratios"""
        try:
            ad_ratios = np.zeros_like(advances)

            for i in range(self.lookback_period, len(advances)):
                # Calculate rolling sums
                advance_sum = np.sum(advances[i-self.lookback_period+1:i+1])
                decline_sum = np.sum(declines[i-self.lookback_period+1:i+1])

                # Calculate ratio
                if decline_sum > 0:
                    ad_ratios[i] = advance_sum / decline_sum
                elif advance_sum > 0:
                    ad_ratios[i] = 2.0  # High ratio when no declines
                else:
                    ad_ratios[i] = 1.0  # Neutral when no activity

            return ad_ratios

        except Exception as e:
            logger.error(f"Error calculating A/D ratios: {str(e)}")
            return np.ones_like(advances)

    def _calculate_ad_line(self, advances: np.ndarray, declines: np.ndarray) -> np.ndarray:
        """Calculate cumulative advance/decline line"""
        try:
            ad_line = np.zeros_like(advances)

            for i in range(1, len(advances)):
                # Cumulative sum of (advances - declines)
                ad_line[i] = ad_line[i-1] + (advances[i] - declines[i])

            return ad_line

        except Exception as e:
            logger.error(f"Error calculating A/D line: {str(e)}")
            return np.zeros_like(advances)

    def _calculate_smoothed_ad_line(self, ad_line: np.ndarray) -> np.ndarray:
        """Calculate smoothed A/D line using moving average"""
        try:
            smoothed_ad_line = np.zeros_like(ad_line)

            for i in range(self.smoothing_period, len(ad_line)):
                smoothed_ad_line[i] = np.mean(ad_line[i-self.smoothing_period+1:i+1])

            return smoothed_ad_line

        except Exception as e:
            logger.error(f"Error calculating smoothed A/D line: {str(e)}")
            return ad_line.copy()

    def _calculate_ad_trend(self, ad_line_smoothed: np.ndarray) -> List[str]:
        """Calculate A/D trend direction"""
        try:
            trends = []

            for i in range(len(ad_line_smoothed)):
                if i < 10:
                    trends.append(ADTrend.NEUTRAL.value)
                    continue

                # Compare current A/D with recent average
                recent_avg = np.mean(ad_line_smoothed[max(0, i-10):i])
                current_ad = ad_line_smoothed[i]

                # Determine trend
                if current_ad > recent_avg * 1.05:
                    trends.append(ADTrend.BULLISH.value)
                elif current_ad < recent_avg * 0.95:
                    trends.append(ADTrend.BEARISH.value)
                else:
                    trends.append(ADTrend.NEUTRAL.value)

            return trends

        except Exception as e:
            logger.error(f"Error calculating A/D trend: {str(e)}")
            return [ADTrend.NEUTRAL.value] * len(ad_line_smoothed)

    def _calculate_breadth_momentum(self, ad_ratios: np.ndarray) -> np.ndarray:
        """Calculate breadth momentum based on A/D ratio changes"""
        try:
            momentum = np.zeros_like(ad_ratios)

            for i in range(5, len(ad_ratios)):
                # Calculate rate of change in A/D ratio
                if ad_ratios[i-5] != 0:
                    momentum[i] = (ad_ratios[i] - ad_ratios[i-5]) / ad_ratios[i-5]
                else:
                    momentum[i] = 0.0

            return momentum

        except Exception as e:
            logger.error(f"Error calculating breadth momentum: {str(e)}")
            return np.zeros_like(ad_ratios)

    def _detect_divergences(self, prices: np.ndarray, ad_line: np.ndarray) -> List[str]:
        """Detect bullish and bearish divergences between price and A/D line"""
        try:
            divergences = ['none'] * len(prices)

            if len(prices) < self.divergence_lookback:
                return divergences

            for i in range(self.divergence_lookback, len(prices)):
                # Look for divergences in the lookback period
                price_window = prices[i-self.divergence_lookback:i+1]
                ad_window = ad_line[i-self.divergence_lookback:i+1]

                # Find recent highs and lows
                price_high_idx = np.argmax(price_window)
                price_low_idx = np.argmin(price_window)
                ad_high_idx = np.argmax(ad_window)
                ad_low_idx = np.argmin(ad_window)

                # Bullish divergence: price makes lower low, A/D makes higher low
                if (price_low_idx > len(price_window) // 2 and
                    ad_low_idx > len(ad_window) // 2 and
                    price_window[price_low_idx] < np.min(price_window[:price_low_idx]) and
                    ad_window[ad_low_idx] > np.min(ad_window[:ad_low_idx])):
                    divergences[i] = 'bullish'

                # Bearish divergence: price makes higher high, A/D makes lower high
                elif (price_high_idx > len(price_window) // 2 and
                      ad_high_idx > len(ad_window) // 2 and
                      price_window[price_high_idx] > np.max(price_window[:price_high_idx]) and
                      ad_window[ad_high_idx] < np.max(ad_window[:ad_high_idx])):
                    divergences[i] = 'bearish'

            return divergences

        except Exception as e:
            logger.error(f"Error detecting divergences: {str(e)}")
            return ['none'] * len(prices)

    def _calculate_market_sentiment(self, ad_ratios: np.ndarray,
                                  breadth_momentum: np.ndarray) -> List[str]:
        """Calculate market sentiment based on A/D ratios and momentum"""
        try:
            sentiment = ['neutral'] * len(ad_ratios)

            for i in range(len(ad_ratios)):
                ratio = ad_ratios[i]
                momentum = breadth_momentum[i]

                # Strong bullish sentiment
                if ratio > 1.5 and momentum > 0.1:
                    sentiment[i] = 'very_bullish'
                elif ratio > 1.2 and momentum > 0.05:
                    sentiment[i] = 'bullish'

                # Strong bearish sentiment
                elif ratio < 0.5 and momentum < -0.1:
                    sentiment[i] = 'very_bearish'
                elif ratio < 0.8 and momentum < -0.05:
                    sentiment[i] = 'bearish'

                # Neutral sentiment
                else:
                    sentiment[i] = 'neutral'

            return sentiment

        except Exception as e:
            logger.error(f"Error calculating market sentiment: {str(e)}")
            return ['neutral'] * len(ad_ratios)

    def _analyze_ad_signals(self, ad_line: float, ad_ratio: float, trend: str,
                           momentum: float, divergence: str, sentiment: str) -> Dict:
        """Analyze A/D data to generate trading signals"""
        try:
            signal_type = ADSignalType.NEUTRAL.value
            strength = 0.0
            confidence = 0.0

            # Breadth signals based on A/D ratio
            if ad_ratio > 1.5 and trend == ADTrend.BULLISH.value:
                signal_type = ADSignalType.BULLISH_BREADTH.value
                strength = min(1.0, (ad_ratio - 1.0) / 2.0)
                confidence = min(0.8, 0.5 + strength * 0.3)
            elif ad_ratio < 0.5 and trend == ADTrend.BEARISH.value:
                signal_type = ADSignalType.BEARISH_BREADTH.value
                strength = min(1.0, (1.0 - ad_ratio) / 0.5)
                confidence = min(0.8, 0.5 + strength * 0.3)

            # Divergence signals
            elif divergence == 'bullish':
                signal_type = ADSignalType.BULLISH_DIVERGENCE.value
                strength = min(1.0, abs(momentum) * 5.0)
                confidence = min(0.85, 0.6 + strength * 0.25)
            elif divergence == 'bearish':
                signal_type = ADSignalType.BEARISH_DIVERGENCE.value
                strength = min(1.0, abs(momentum) * 5.0)
                confidence = min(0.85, 0.6 + strength * 0.25)

            # Breadth expansion/contraction signals
            elif abs(momentum) > 0.2:
                if momentum > 0:
                    signal_type = ADSignalType.BREADTH_EXPANSION.value
                else:
                    signal_type = ADSignalType.BREADTH_CONTRACTION.value
                strength = min(1.0, abs(momentum) * 3.0)
                confidence = min(0.75, 0.4 + strength * 0.35)

            # Momentum shift signals
            elif abs(momentum) > 0.1 and sentiment in ['very_bullish', 'very_bearish']:
                signal_type = ADSignalType.MOMENTUM_SHIFT.value
                strength = min(1.0, abs(momentum) * 5.0)
                confidence = min(0.7, 0.5 + strength * 0.2)

            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error analyzing A/D signals: {str(e)}")
            return {
                'signal_type': ADSignalType.NEUTRAL.value,
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
            'ad_line': np.array([]),
            'ad_line_smoothed': np.array([]),
            'ad_ratios': np.array([]),
            'advances': np.array([]),
            'declines': np.array([]),
            'ad_trend': [],
            'breadth_momentum': np.array([]),
            'divergence_signals': [],
            'market_sentiment': [],
            'price_changes': np.array([]),
            'lookback_period_used': self.lookback_period,
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
            breadth_signals = [s for s in self.signal_history if 'breadth' in s.signal_type]
            divergence_signals = [s for s in self.signal_history if 'divergence' in s.signal_type]
            momentum_signals = [s for s in self.signal_history if 'momentum' in s.signal_type or 'expansion' in s.signal_type or 'contraction' in s.signal_type]

            # Calculate average confidence
            avg_confidence = np.mean([s.confidence for s in self.signal_history])

            self.performance_stats.update({
                'total_signals': total_signals,
                'avg_confidence': avg_confidence,
                'breadth_signals': len(breadth_signals),
                'divergence_signals': len(divergence_signals),
                'momentum_signals': len(momentum_signals),
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

    # Initialize Advance/Decline
    ad = AdvanceDecline(lookback_period=20, smoothing_period=5, divergence_lookback=30)

    # Calculate A/D
    result = ad.calculate_advance_decline(high_prices, low_prices, close_prices, volume_data)
    print("Advance/Decline calculation completed")
    print(f"Latest A/D Line: {result['ad_line'][-1]:.2f}")
    print(f"Latest A/D Ratio: {result['ad_ratios'][-1]:.2f}")
    print(f"A/D trend: {result['ad_trend'][-1]}")
    print(f"Market sentiment: {result['market_sentiment'][-1]}")

    # Generate signals
    signals = ad.generate_signals(high_prices, low_prices, close_prices, volume_data, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")

    if signals:
        latest_signal = signals[-1]
        print(f"Latest signal: {latest_signal.signal_type} (confidence: {latest_signal.confidence:.2f})")

    # Get performance stats
    stats = ad.get_performance_stats()
    print(f"Performance stats: {stats}")

    print("Advance/Decline implementation test completed successfully!")
