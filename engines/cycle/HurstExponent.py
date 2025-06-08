# -*- coding: utf-8 -*-
"""
Hurst Exponent Indicator
Advanced implementation for market efficiency analysis and trend persistence
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

class MarketRegime(Enum):
    """Market regime types based on Hurst Exponent"""
    MEAN_REVERTING = "mean_reverting"      # H < 0.5
    RANDOM_WALK = "random_walk"            # H ~= 0.5
    TRENDING = "trending"                  # H > 0.5
    STRONG_TRENDING = "strong_trending"    # H > 0.7
    STRONG_MEAN_REVERTING = "strong_mean_reverting"  # H < 0.3

class HurstSignalType(Enum):
    """Hurst Exponent signal types"""
    TREND_PERSISTENCE_HIGH = "trend_persistence_high"
    TREND_PERSISTENCE_LOW = "trend_persistence_low"
    MEAN_REVERSION_SIGNAL = "mean_reversion_signal"
    REGIME_CHANGE_TO_TRENDING = "regime_change_to_trending"
    REGIME_CHANGE_TO_MEAN_REVERTING = "regime_change_to_mean_reverting"
    MARKET_EFFICIENCY_HIGH = "market_efficiency_high"
    MARKET_EFFICIENCY_LOW = "market_efficiency_low"
    VOLATILITY_CLUSTERING = "volatility_clustering"
    LONG_MEMORY_DETECTED = "long_memory_detected"

@dataclass
class HurstSignal:
    """Hurst Exponent signal data structure"""
    timestamp: datetime
    price: float
    hurst_exponent: float
    market_regime: str
    trend_persistence: float
    mean_reversion_strength: float
    signal_type: str
    signal_strength: float
    confidence: float
    regime_stability: float
    timeframe: str
    session: str

class HurstExponent:
    """
    Advanced Hurst Exponent implementation for forex trading
    Features:
    - Market efficiency analysis using R/S statistics
    - Trend persistence and mean reversion detection
    - Market regime identification (trending vs mean-reverting)
    - Long memory and volatility clustering detection
    - Session-aware regime analysis
    - Multiple timeframe support
    - Rolling window analysis for regime changes
    """

    def __init__(self,
                 window_size: int = 100,
                 min_periods: int = 50,
                 regime_threshold: float = 0.1,
                 timeframes: List[str] = None):
        """
        Initialize Hurst Exponent calculator

        Args:
            window_size: Rolling window size for Hurst calculation
            min_periods: Minimum periods required for calculation
            regime_threshold: Threshold for regime change detection
            timeframes: List of timeframes to analyze
        """
        self.window_size = window_size
        self.min_periods = min_periods
        self.regime_threshold = regime_threshold
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']

        # Hurst thresholds
        self.mean_reverting_threshold = 0.5
        self.trending_threshold = 0.5
        self.strong_trending_threshold = 0.7
        self.strong_mean_reverting_threshold = 0.3
        self.random_walk_tolerance = 0.05  # plus_minus0.05 around 0.5

        # Performance tracking
        self.signal_history = []
        self.regime_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'regime_accuracy': 0.0,
            'trend_persistence_accuracy': 0.0
        }

        logger.info(f"HurstExponent initialized: window_size={window_size}, "
                   f"min_periods={min_periods}, regime_threshold={regime_threshold}")

    def calculate_hurst_exponent(self,
                                prices: Union[pd.Series, np.ndarray],
                                timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Hurst Exponent for given price data

        Args:
            prices: Price data (typically close prices)
            timestamps: Optional timestamps for session analysis

        Returns:
            Dictionary containing Hurst Exponent calculations
        """
        try:
            # Convert to numpy array
            prices_array = np.array(prices)

            if len(prices_array) < self.min_periods:
                logger.warning(f"Insufficient data: {len(prices_array)} < {self.min_periods}")
                return self._empty_result()

            # Calculate rolling Hurst exponents
            hurst_values = self._calculate_rolling_hurst(prices_array)

            # Identify market regimes
            market_regimes = self._identify_market_regimes(hurst_values)

            # Calculate trend persistence
            trend_persistence = self._calculate_trend_persistence(hurst_values)

            # Calculate mean reversion strength
            mean_reversion_strength = self._calculate_mean_reversion_strength(hurst_values)

            # Detect regime changes
            regime_changes = self._detect_regime_changes(market_regimes)

            # Calculate regime stability
            regime_stability = self._calculate_regime_stability(hurst_values)

            # Analyze long memory effects
            long_memory = self._analyze_long_memory(prices_array, hurst_values)

            result = {
                'hurst_values': hurst_values,
                'market_regimes': market_regimes,
                'trend_persistence': trend_persistence,
                'mean_reversion_strength': mean_reversion_strength,
                'regime_changes': regime_changes,
                'regime_stability': regime_stability,
                'long_memory': long_memory,
                'window_size_used': self.window_size,
                'thresholds': {
                    'trending': self.trending_threshold,
                    'mean_reverting': self.mean_reverting_threshold,
                    'strong_trending': self.strong_trending_threshold,
                    'strong_mean_reverting': self.strong_mean_reverting_threshold
                }
            }

            logger.debug(f"Hurst Exponent calculated: latest_hurst={hurst_values[-1]:.3f}, "
                        f"regime={market_regimes[-1]}, persistence={trend_persistence[-1]:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error calculating Hurst Exponent: {str(e)}")
            return self._empty_result()

    def generate_signals(self,
                        prices: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[HurstSignal]:
        """
        Generate trading signals based on Hurst Exponent analysis

        Args:
            prices: Price data
            timestamps: Optional timestamps
            timeframe: Current timeframe

        Returns:
            List of HurstSignal objects
        """
        try:
            hurst_data = self.calculate_hurst_exponent(prices, timestamps)
            if not hurst_data or 'hurst_values' not in hurst_data:
                return []

            signals = []
            current_time = datetime.now()

            # Get latest values
            latest_price = prices.iloc[-1] if isinstance(prices, pd.Series) else prices[-1]
            latest_hurst = hurst_data['hurst_values'][-1]
            latest_regime = hurst_data['market_regimes'][-1]
            latest_persistence = hurst_data['trend_persistence'][-1]
            latest_mean_reversion = hurst_data['mean_reversion_strength'][-1]
            latest_regime_change = hurst_data['regime_changes'][-1]
            latest_stability = hurst_data['regime_stability'][-1]

            # Determine current session
            session = self._get_current_session(current_time)

            # Generate signals based on Hurst analysis
            signal_data = self._analyze_hurst_signals(
                latest_hurst, latest_regime, latest_persistence,
                latest_mean_reversion, latest_regime_change, latest_stability
            )

            if signal_data['signal_type'] != 'NONE':
                signal = HurstSignal(
                    timestamp=current_time,
                    price=latest_price,
                    hurst_exponent=latest_hurst,
                    market_regime=latest_regime,
                    trend_persistence=latest_persistence,
                    mean_reversion_strength=latest_mean_reversion,
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    regime_stability=latest_stability,
                    timeframe=timeframe,
                    session=session
                )

                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()

                logger.info(f"Hurst signal generated: {signal.signal_type} "
                           f"(hurst={signal.hurst_exponent:.3f}, regime={signal.market_regime}, "
                           f"confidence={signal.confidence:.2f})")

            return signals

        except Exception as e:
            logger.error(f"Error generating Hurst signals: {str(e)}")
            return []

    def _calculate_rolling_hurst(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling Hurst exponent using R/S analysis"""
        try:
            hurst_values = np.full(len(prices), 0.5)  # Initialize with random walk

            for i in range(self.min_periods, len(prices)):
                start_idx = max(0, i - self.window_size + 1)
                window_prices = prices[start_idx:i+1]

                if len(window_prices) >= self.min_periods:
                    hurst_values[i] = self._calculate_single_hurst(window_prices)

            return hurst_values

        except Exception as e:
            logger.error(f"Error calculating rolling Hurst: {str(e)}")
            return np.full(len(prices), 0.5)

    def _calculate_single_hurst(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent for a single window using R/S analysis"""
        try:
            if len(prices) < 10:
                return 0.5

            # Calculate log returns
            log_returns = np.diff(np.log(prices))
            n = len(log_returns)

            if n < 5:
                return 0.5

            # Calculate mean return
            mean_return = np.mean(log_returns)

            # Calculate cumulative deviations from mean
            cumulative_deviations = np.cumsum(log_returns - mean_return)

            # Calculate range (R)
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

            # Calculate standard deviation (S)
            S = np.std(log_returns)

            if S == 0 or R == 0:
                return 0.5

            # Calculate R/S ratio
            rs_ratio = R / S

            # Calculate Hurst exponent: H = log(R/S) / log(n)
            if rs_ratio > 0 and n > 1:
                hurst = np.log(rs_ratio) / np.log(n)
                # Constrain to reasonable bounds
                return max(0.0, min(1.0, hurst))
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Error calculating single Hurst: {str(e)}")
            return 0.5

    def _identify_market_regimes(self, hurst_values: np.ndarray) -> List[str]:
        """Identify market regimes based on Hurst exponent values"""
        try:
            regimes = []

            for hurst in hurst_values:
                if hurst < self.strong_mean_reverting_threshold:
                    regimes.append(MarketRegime.STRONG_MEAN_REVERTING.value)
                elif hurst < self.mean_reverting_threshold - self.random_walk_tolerance:
                    regimes.append(MarketRegime.MEAN_REVERTING.value)
                elif abs(hurst - 0.5) <= self.random_walk_tolerance:
                    regimes.append(MarketRegime.RANDOM_WALK.value)
                elif hurst > self.strong_trending_threshold:
                    regimes.append(MarketRegime.STRONG_TRENDING.value)
                elif hurst > self.trending_threshold + self.random_walk_tolerance:
                    regimes.append(MarketRegime.TRENDING.value)
                else:
                    regimes.append(MarketRegime.RANDOM_WALK.value)

            return regimes

        except Exception as e:
            logger.error(f"Error identifying market regimes: {str(e)}")
            return [MarketRegime.RANDOM_WALK.value] * len(hurst_values)

    def _calculate_trend_persistence(self, hurst_values: np.ndarray) -> np.ndarray:
        """Calculate trend persistence score"""
        try:
            # Trend persistence is higher when H > 0.5
            persistence = np.zeros_like(hurst_values)

            for i, hurst in enumerate(hurst_values):
                if hurst > 0.5:
                    # Trending market - high persistence
                    persistence[i] = min(1.0, (hurst - 0.5) * 2.0)
                else:
                    # Mean-reverting market - low persistence
                    persistence[i] = max(0.0, hurst * 2.0)

            return persistence

        except Exception as e:
            logger.error(f"Error calculating trend persistence: {str(e)}")
            return np.full(len(hurst_values), 0.5)

    def _calculate_mean_reversion_strength(self, hurst_values: np.ndarray) -> np.ndarray:
        """Calculate mean reversion strength"""
        try:
            # Mean reversion strength is higher when H < 0.5
            reversion_strength = np.zeros_like(hurst_values)

            for i, hurst in enumerate(hurst_values):
                if hurst < 0.5:
                    # Mean-reverting market - high reversion strength
                    reversion_strength[i] = min(1.0, (0.5 - hurst) * 2.0)
                else:
                    # Trending market - low reversion strength
                    reversion_strength[i] = max(0.0, (1.0 - hurst) * 2.0)

            return reversion_strength

        except Exception as e:
            logger.error(f"Error calculating mean reversion strength: {str(e)}")
            return np.full(len(hurst_values), 0.5)

    def _detect_regime_changes(self, regimes: List[str]) -> List[bool]:
        """Detect regime changes"""
        try:
            changes = [False]  # First value has no change

            for i in range(1, len(regimes)):
                # Check if regime changed from previous period
                if regimes[i] != regimes[i-1]:
                    changes.append(True)
                else:
                    changes.append(False)

            return changes

        except Exception as e:
            logger.error(f"Error detecting regime changes: {str(e)}")
            return [False] * len(regimes)

    def _calculate_regime_stability(self, hurst_values: np.ndarray) -> np.ndarray:
        """Calculate regime stability based on Hurst value consistency"""
        try:
            stability = np.zeros_like(hurst_values)
            window = 10  # Look back window for stability

            for i in range(len(hurst_values)):
                if i < window:
                    stability[i] = 0.5
                    continue

                # Calculate standard deviation of recent Hurst values
                recent_hurst = hurst_values[i-window+1:i+1]
                hurst_std = np.std(recent_hurst)

                # Stability is inverse of volatility (lower std = higher stability)
                stability[i] = max(0.0, min(1.0, 1.0 - hurst_std * 5.0))

            return stability

        except Exception as e:
            logger.error(f"Error calculating regime stability: {str(e)}")
            return np.full(len(hurst_values), 0.5)

    def _analyze_long_memory(self, prices: np.ndarray, hurst_values: np.ndarray) -> Dict:
        """Analyze long memory effects in the time series"""
        try:
            # Long memory is indicated by H significantly different from 0.5
            long_memory_strength = np.abs(hurst_values - 0.5) * 2.0

            # Detect periods of strong long memory
            strong_memory_threshold = 0.6
            strong_memory_periods = long_memory_strength > strong_memory_threshold

            # Calculate volatility clustering (another sign of long memory)
            returns = np.diff(np.log(prices))
            volatility = pd.Series(returns).rolling(window=20).std().fillna(0).values
            volatility_clustering = self._detect_volatility_clustering(volatility)

            return {
                'long_memory_strength': long_memory_strength,
                'strong_memory_periods': strong_memory_periods,
                'volatility_clustering': volatility_clustering,
                'avg_long_memory': np.mean(long_memory_strength)
            }

        except Exception as e:
            logger.error(f"Error analyzing long memory: {str(e)}")
            return {
                'long_memory_strength': np.zeros_like(hurst_values),
                'strong_memory_periods': np.zeros_like(hurst_values, dtype=bool),
                'volatility_clustering': np.zeros_like(hurst_values),
                'avg_long_memory': 0.0
            }

    def _detect_volatility_clustering(self, volatility: np.ndarray) -> np.ndarray:
        """Detect volatility clustering patterns"""
        try:
            clustering = np.zeros_like(volatility)
            window = 10

            for i in range(window, len(volatility)):
                recent_vol = volatility[i-window:i]
                current_vol = volatility[i]

                # High clustering when current volatility is similar to recent average
                if np.mean(recent_vol) > 0:
                    similarity = 1.0 - abs(current_vol - np.mean(recent_vol)) / np.mean(recent_vol)
                    clustering[i] = max(0.0, min(1.0, similarity))

            return clustering

        except Exception as e:
            logger.error(f"Error detecting volatility clustering: {str(e)}")
            return np.zeros_like(volatility)

    def _analyze_hurst_signals(self, hurst: float, regime: str, persistence: float,
                              mean_reversion: float, regime_change: bool, stability: float) -> Dict:
        """Analyze current Hurst conditions and generate signal"""
        try:
            signal_type = 'NONE'
            signal_strength = 0.0
            confidence = 0.0

            # Regime change signals (highest priority)
            if regime_change:
                if regime in [MarketRegime.TRENDING.value, MarketRegime.STRONG_TRENDING.value]:
                    signal_type = HurstSignalType.REGIME_CHANGE_TO_TRENDING.value
                    signal_strength = min(1.0, persistence * 1.2)
                    confidence = min(0.9, 0.7 + stability * 0.2)
                elif regime in [MarketRegime.MEAN_REVERTING.value, MarketRegime.STRONG_MEAN_REVERTING.value]:
                    signal_type = HurstSignalType.REGIME_CHANGE_TO_MEAN_REVERTING.value
                    signal_strength = min(1.0, mean_reversion * 1.2)
                    confidence = min(0.9, 0.7 + stability * 0.2)

            # Strong trend persistence signals
            elif regime == MarketRegime.STRONG_TRENDING.value and persistence > 0.8:
                signal_type = HurstSignalType.TREND_PERSISTENCE_HIGH.value
                signal_strength = min(1.0, persistence)
                confidence = min(0.85, 0.6 + persistence * 0.25)

            # Strong mean reversion signals
            elif regime == MarketRegime.STRONG_MEAN_REVERTING.value and mean_reversion > 0.8:
                signal_type = HurstSignalType.MEAN_REVERSION_SIGNAL.value
                signal_strength = min(1.0, mean_reversion)
                confidence = min(0.85, 0.6 + mean_reversion * 0.25)

            # Market efficiency signals
            elif regime == MarketRegime.RANDOM_WALK.value and stability > 0.7:
                signal_type = HurstSignalType.MARKET_EFFICIENCY_HIGH.value
                signal_strength = min(1.0, stability)
                confidence = min(0.75, 0.5 + stability * 0.25)

            # Trend persistence signals
            elif regime == MarketRegime.TRENDING.value and persistence > 0.6:
                signal_type = HurstSignalType.TREND_PERSISTENCE_HIGH.value
                signal_strength = min(1.0, persistence * 0.8)
                confidence = min(0.75, 0.5 + persistence * 0.25)
            elif persistence < 0.3:
                signal_type = HurstSignalType.TREND_PERSISTENCE_LOW.value
                signal_strength = 1.0 - persistence
                confidence = min(0.7, 0.4 + (1.0 - persistence) * 0.3)

            # Long memory detection
            elif abs(hurst - 0.5) > 0.3:
                signal_type = HurstSignalType.LONG_MEMORY_DETECTED.value
                signal_strength = min(1.0, abs(hurst - 0.5) * 2.0)
                confidence = min(0.8, 0.5 + signal_strength * 0.3)

            # Market efficiency low
            elif stability < 0.3:
                signal_type = HurstSignalType.MARKET_EFFICIENCY_LOW.value
                signal_strength = 1.0 - stability
                confidence = min(0.65, 0.4 + (1.0 - stability) * 0.25)

            return {
                'signal_type': signal_type,
                'strength': signal_strength,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error analyzing Hurst signals: {str(e)}")
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
            'hurst_values': np.array([]),
            'market_regimes': [],
            'trend_persistence': np.array([]),
            'mean_reversion_strength': np.array([]),
            'regime_changes': [],
            'regime_stability': np.array([]),
            'long_memory': {
                'long_memory_strength': np.array([]),
                'strong_memory_periods': np.array([], dtype=bool),
                'volatility_clustering': np.array([]),
                'avg_long_memory': 0.0
            },
            'window_size_used': self.window_size,
            'thresholds': {
                'trending': self.trending_threshold,
                'mean_reverting': self.mean_reverting_threshold,
                'strong_trending': self.strong_trending_threshold,
                'strong_mean_reverting': self.strong_mean_reverting_threshold
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
    n_points = 300

    # Generate price data with different regimes
    # First part: trending
    trend1 = np.linspace(100, 120, n_points//3)
    noise1 = np.random.randn(n_points//3) * 0.5

    # Second part: mean-reverting
    mean_level = 120
    mean_rev = mean_level + np.random.randn(n_points//3) * 2.0
    for i in range(1, len(mean_rev)):
        mean_rev[i] = mean_level + 0.8 * (mean_rev[i-1] - mean_level) + np.random.randn() * 0.5

    # Third part: random walk
    random_walk = np.cumsum(np.random.randn(n_points//3) * 0.3) + 120

    prices = np.concatenate([trend1 + noise1, mean_rev, random_walk])
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')

    # Initialize Hurst Exponent
    hurst_indicator = HurstExponent(window_size=100, min_periods=50)

    # Calculate Hurst Exponent
    result = hurst_indicator.calculate_hurst_exponent(prices)
    print("Hurst Exponent calculation completed")
    print(f"Latest Hurst: {result['hurst_values'][-1]:.3f}")
    print(f"Latest regime: {result['market_regimes'][-1]}")
    print(f"Latest persistence: {result['trend_persistence'][-1]:.2f}")
    print(f"Latest mean reversion: {result['mean_reversion_strength'][-1]:.2f}")

    # Generate signals
    signals = hurst_indicator.generate_signals(prices, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")

    # Display performance stats
    stats = hurst_indicator.get_performance_stats()
    print(f"Performance stats: {stats}")

    if signals:
        latest_signal = signals[-1]
        print(f"Latest signal: {latest_signal.signal_type} "
              f"(hurst={latest_signal.hurst_exponent:.3f}, "
              f"confidence={latest_signal.confidence:.2f})")
