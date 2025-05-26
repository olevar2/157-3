"""
On-Balance Volume (OBV) Indicator
Advanced implementation with trend confirmation and divergence detection
Optimized for M1-H4 timeframes and volume analysis
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

class OBVTrend(Enum):
    """OBV trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    DIVERGENCE_BULLISH = "divergence_bullish"
    DIVERGENCE_BEARISH = "divergence_bearish"

class OBVSignalType(Enum):
    """OBV signal types"""
    TREND_CONFIRMATION_BULLISH = "trend_confirmation_bullish"
    TREND_CONFIRMATION_BEARISH = "trend_confirmation_bearish"
    DIVERGENCE_BULLISH = "divergence_bullish"
    DIVERGENCE_BEARISH = "divergence_bearish"
    VOLUME_BREAKOUT = "volume_breakout"
    VOLUME_EXHAUSTION = "volume_exhaustion"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"

@dataclass
class OBVSignal:
    """OBV signal data structure"""
    timestamp: datetime
    price: float
    obv_value: float
    obv_trend: str
    volume_strength: float
    divergence_strength: float
    signal_type: str
    signal_strength: float
    confidence: float
    volume_ratio: float
    timeframe: str
    session: str

class OBV:
    """
    Advanced On-Balance Volume implementation for forex trading
    Features:
    - Cumulative volume analysis based on price direction
    - Trend confirmation and divergence detection
    - Volume strength and momentum analysis
    - Accumulation/distribution identification
    - Session-aware volume analysis
    - Multiple timeframe support
    """
    
    def __init__(self, 
                 smoothing_period: int = 10,
                 divergence_lookback: int = 20,
                 volume_threshold: float = 1.5,
                 timeframes: List[str] = None):
        """
        Initialize OBV calculator
        
        Args:
            smoothing_period: Period for OBV smoothing
            divergence_lookback: Lookback period for divergence detection
            volume_threshold: Volume threshold for breakout detection
            timeframes: List of timeframes to analyze
        """
        self.smoothing_period = smoothing_period
        self.divergence_lookback = divergence_lookback
        self.volume_threshold = volume_threshold
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']
        
        # Signal thresholds
        self.trend_confirmation_threshold = 0.7
        self.divergence_threshold = 0.6
        self.volume_breakout_threshold = 2.0
        self.accumulation_threshold = 0.8
        
        # Performance tracking
        self.signal_history = []
        self.divergence_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'divergence_accuracy': 0.0,
            'trend_confirmation_accuracy': 0.0
        }
        
        logger.info(f"OBV initialized: smoothing_period={smoothing_period}, "
                   f"divergence_lookback={divergence_lookback}, volume_threshold={volume_threshold}")
    
    def calculate_obv(self, 
                     close: Union[pd.Series, np.ndarray],
                     volume: Union[pd.Series, np.ndarray],
                     timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate On-Balance Volume for given price and volume data
        
        Args:
            close: Close prices
            volume: Volume data
            timestamps: Optional timestamps for session analysis
            
        Returns:
            Dictionary containing OBV calculations
        """
        try:
            # Convert to numpy arrays
            close_array = np.array(close)
            volume_array = np.array(volume)
            
            if len(close_array) != len(volume_array):
                logger.error("Price and volume arrays must have same length")
                return self._empty_result()
            
            if len(close_array) < 2:
                logger.warning(f"Insufficient data: {len(close_array)} < 2")
                return self._empty_result()
            
            # Calculate OBV
            obv_values = self._calculate_obv_values(close_array, volume_array)
            
            # Calculate smoothed OBV
            obv_smoothed = self._calculate_smoothed_obv(obv_values)
            
            # Calculate OBV trend
            obv_trend = self._calculate_obv_trend(obv_smoothed)
            
            # Calculate volume strength
            volume_strength = self._calculate_volume_strength(volume_array)
            
            # Detect divergences
            divergence_signals = self._detect_divergences(close_array, obv_smoothed)
            
            # Calculate volume ratios
            volume_ratios = self._calculate_volume_ratios(volume_array)
            
            # Identify accumulation/distribution
            accumulation_distribution = self._identify_accumulation_distribution(
                close_array, obv_values, volume_strength
            )
            
            result = {
                'obv_values': obv_values,
                'obv_smoothed': obv_smoothed,
                'obv_trend': obv_trend,
                'volume_strength': volume_strength,
                'divergence_signals': divergence_signals,
                'volume_ratios': volume_ratios,
                'accumulation_distribution': accumulation_distribution,
                'smoothing_period_used': self.smoothing_period,
                'divergence_lookback_used': self.divergence_lookback
            }
            
            logger.debug(f"OBV calculated: latest_obv={obv_values[-1]:.0f}, "
                        f"trend={obv_trend[-1]}, volume_strength={volume_strength[-1]:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return self._empty_result()
    
    def generate_signals(self, 
                        close: Union[pd.Series, np.ndarray],
                        volume: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[OBVSignal]:
        """
        Generate trading signals based on OBV analysis
        
        Args:
            close: Close prices
            volume: Volume data
            timestamps: Optional timestamps
            timeframe: Current timeframe
            
        Returns:
            List of OBVSignal objects
        """
        try:
            obv_data = self.calculate_obv(close, volume, timestamps)
            if not obv_data or 'obv_values' not in obv_data:
                return []
            
            signals = []
            current_time = datetime.now()
            
            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_obv = obv_data['obv_values'][-1]
            latest_trend = obv_data['obv_trend'][-1]
            latest_volume_strength = obv_data['volume_strength'][-1]
            latest_divergence = obv_data['divergence_signals'][-1]
            latest_volume_ratio = obv_data['volume_ratios'][-1]
            latest_acc_dist = obv_data['accumulation_distribution'][-1]
            
            # Determine current session
            session = self._get_current_session(current_time)
            
            # Generate signals based on OBV analysis
            signal_data = self._analyze_obv_signals(
                latest_obv, latest_trend, latest_volume_strength,
                latest_divergence, latest_volume_ratio, latest_acc_dist
            )
            
            if signal_data['signal_type'] != 'NONE':
                signal = OBVSignal(
                    timestamp=current_time,
                    price=latest_price,
                    obv_value=latest_obv,
                    obv_trend=latest_trend,
                    volume_strength=latest_volume_strength,
                    divergence_strength=abs(latest_divergence),
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    volume_ratio=latest_volume_ratio,
                    timeframe=timeframe,
                    session=session
                )
                
                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()
                
                logger.info(f"OBV signal generated: {signal.signal_type} "
                           f"(obv={signal.obv_value:.0f}, trend={signal.obv_trend}, "
                           f"confidence={signal.confidence:.2f})")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating OBV signals: {str(e)}")
            return []
    
    def _calculate_obv_values(self, close_prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate raw OBV values"""
        try:
            obv = np.zeros_like(close_prices, dtype=float)
            obv[0] = volume[0]  # Initialize with first volume
            
            for i in range(1, len(close_prices)):
                if close_prices[i] > close_prices[i-1]:
                    # Price up - add volume
                    obv[i] = obv[i-1] + volume[i]
                elif close_prices[i] < close_prices[i-1]:
                    # Price down - subtract volume
                    obv[i] = obv[i-1] - volume[i]
                else:
                    # Price unchanged - keep same OBV
                    obv[i] = obv[i-1]
            
            return obv
            
        except Exception as e:
            logger.error(f"Error calculating OBV values: {str(e)}")
            return np.zeros_like(close_prices)
    
    def _calculate_smoothed_obv(self, obv_values: np.ndarray) -> np.ndarray:
        """Calculate smoothed OBV using moving average"""
        try:
            if self.smoothing_period <= 1:
                return obv_values
            
            return pd.Series(obv_values).rolling(
                window=self.smoothing_period, min_periods=1
            ).mean().values
            
        except Exception as e:
            logger.error(f"Error calculating smoothed OBV: {str(e)}")
            return obv_values
    
    def _calculate_obv_trend(self, obv_smoothed: np.ndarray) -> List[str]:
        """Calculate OBV trend direction"""
        try:
            trends = []
            
            for i in range(len(obv_smoothed)):
                if i < 5:
                    trends.append(OBVTrend.NEUTRAL.value)
                    continue
                
                # Compare current OBV with recent average
                recent_avg = np.mean(obv_smoothed[max(0, i-5):i])
                current_obv = obv_smoothed[i]
                
                # Calculate trend strength
                if current_obv > recent_avg * 1.02:
                    trends.append(OBVTrend.BULLISH.value)
                elif current_obv < recent_avg * 0.98:
                    trends.append(OBVTrend.BEARISH.value)
                else:
                    trends.append(OBVTrend.NEUTRAL.value)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating OBV trend: {str(e)}")
            return [OBVTrend.NEUTRAL.value] * len(obv_smoothed)
    
    def _calculate_volume_strength(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume strength relative to recent average"""
        try:
            volume_strength = np.zeros_like(volume, dtype=float)
            
            for i in range(len(volume)):
                if i < 10:
                    # Use available data for initial values
                    avg_volume = np.mean(volume[:i+1])
                else:
                    # Use rolling 10-period average
                    avg_volume = np.mean(volume[i-9:i+1])
                
                if avg_volume > 0:
                    volume_strength[i] = volume[i] / avg_volume
                else:
                    volume_strength[i] = 1.0
            
            return volume_strength
            
        except Exception as e:
            logger.error(f"Error calculating volume strength: {str(e)}")
            return np.ones_like(volume)
    
    def _detect_divergences(self, prices: np.ndarray, obv_smoothed: np.ndarray) -> np.ndarray:
        """Detect divergences between price and OBV"""
        try:
            divergences = np.zeros_like(prices)
            
            if len(prices) < self.divergence_lookback:
                return divergences
            
            for i in range(self.divergence_lookback, len(prices)):
                # Look for recent highs and lows
                lookback_start = i - self.divergence_lookback
                
                price_window = prices[lookback_start:i+1]
                obv_window = obv_smoothed[lookback_start:i+1]
                
                # Find recent high and low points
                price_high_idx = np.argmax(price_window)
                price_low_idx = np.argmin(price_window)
                obv_high_idx = np.argmax(obv_window)
                obv_low_idx = np.argmin(obv_window)
                
                # Check for bullish divergence (price makes lower low, OBV makes higher low)
                if (price_low_idx > len(price_window) * 0.7 and  # Recent price low
                    obv_low_idx < len(obv_window) * 0.5 and  # Earlier OBV low
                    price_window[price_low_idx] < price_window[obv_low_idx] and  # Lower price low
                    obv_window[-1] > obv_window[obv_low_idx]):  # Higher OBV low
                    divergences[i] = 1.0  # Bullish divergence
                
                # Check for bearish divergence (price makes higher high, OBV makes lower high)
                elif (price_high_idx > len(price_window) * 0.7 and  # Recent price high
                      obv_high_idx < len(obv_window) * 0.5 and  # Earlier OBV high
                      price_window[price_high_idx] > price_window[obv_high_idx] and  # Higher price high
                      obv_window[-1] < obv_window[obv_high_idx]):  # Lower OBV high
                    divergences[i] = -1.0  # Bearish divergence
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting divergences: {str(e)}")
            return np.zeros_like(prices)
    
    def _calculate_volume_ratios(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume ratios for breakout detection"""
        try:
            volume_ratios = np.zeros_like(volume, dtype=float)
            
            for i in range(len(volume)):
                if i < 20:
                    # Use available data for initial values
                    avg_volume = np.mean(volume[:i+1])
                else:
                    # Use rolling 20-period average
                    avg_volume = np.mean(volume[i-19:i+1])
                
                if avg_volume > 0:
                    volume_ratios[i] = volume[i] / avg_volume
                else:
                    volume_ratios[i] = 1.0
            
            return volume_ratios
            
        except Exception as e:
            logger.error(f"Error calculating volume ratios: {str(e)}")
            return np.ones_like(volume)
    
    def _identify_accumulation_distribution(self, prices: np.ndarray, obv_values: np.ndarray, 
                                          volume_strength: np.ndarray) -> List[str]:
        """Identify accumulation and distribution phases"""
        try:
            acc_dist = []
            
            for i in range(len(prices)):
                if i < 10:
                    acc_dist.append('NEUTRAL')
                    continue
                
                # Calculate recent price and OBV trends
                price_trend = (prices[i] - prices[i-10]) / prices[i-10]
                obv_trend = (obv_values[i] - obv_values[i-10]) / abs(obv_values[i-10]) if obv_values[i-10] != 0 else 0
                avg_volume_strength = np.mean(volume_strength[max(0, i-5):i+1])
                
                # Accumulation: Rising OBV with stable/rising prices and strong volume
                if (obv_trend > 0.05 and price_trend > -0.02 and 
                    avg_volume_strength > self.accumulation_threshold):
                    acc_dist.append('ACCUMULATION')
                
                # Distribution: Falling OBV with stable/falling prices and strong volume
                elif (obv_trend < -0.05 and price_trend < 0.02 and 
                      avg_volume_strength > self.accumulation_threshold):
                    acc_dist.append('DISTRIBUTION')
                
                # Neutral conditions
                else:
                    acc_dist.append('NEUTRAL')
            
            return acc_dist
            
        except Exception as e:
            logger.error(f"Error identifying accumulation/distribution: {str(e)}")
            return ['NEUTRAL'] * len(prices)
    
    def _analyze_obv_signals(self, obv_value: float, trend: str, volume_strength: float,
                            divergence: float, volume_ratio: float, acc_dist: str) -> Dict:
        """Analyze current OBV conditions and generate signal"""
        try:
            signal_type = 'NONE'
            strength = 0.0
            confidence = 0.0
            
            # Divergence signals (highest priority)
            if divergence > 0.5:
                signal_type = OBVSignalType.DIVERGENCE_BULLISH.value
                strength = min(1.0, divergence)
                confidence = min(0.9, 0.7 + volume_strength * 0.1 + (volume_ratio / 3.0) * 0.1)
            elif divergence < -0.5:
                signal_type = OBVSignalType.DIVERGENCE_BEARISH.value
                strength = min(1.0, abs(divergence))
                confidence = min(0.9, 0.7 + volume_strength * 0.1 + (volume_ratio / 3.0) * 0.1)
            
            # Volume breakout signals
            elif volume_ratio > self.volume_breakout_threshold:
                signal_type = OBVSignalType.VOLUME_BREAKOUT.value
                strength = min(1.0, volume_ratio / 3.0)
                confidence = min(0.85, 0.6 + strength * 0.25)
            
            # Trend confirmation signals
            elif trend == OBVTrend.BULLISH.value and volume_strength > self.trend_confirmation_threshold:
                signal_type = OBVSignalType.TREND_CONFIRMATION_BULLISH.value
                strength = min(1.0, volume_strength)
                confidence = min(0.8, 0.5 + volume_strength * 0.2 + (volume_ratio / 2.0) * 0.1)
            elif trend == OBVTrend.BEARISH.value and volume_strength > self.trend_confirmation_threshold:
                signal_type = OBVSignalType.TREND_CONFIRMATION_BEARISH.value
                strength = min(1.0, volume_strength)
                confidence = min(0.8, 0.5 + volume_strength * 0.2 + (volume_ratio / 2.0) * 0.1)
            
            # Accumulation/Distribution signals
            elif acc_dist == 'ACCUMULATION':
                signal_type = OBVSignalType.ACCUMULATION.value
                strength = min(1.0, volume_strength)
                confidence = min(0.75, 0.5 + volume_strength * 0.25)
            elif acc_dist == 'DISTRIBUTION':
                signal_type = OBVSignalType.DISTRIBUTION.value
                strength = min(1.0, volume_strength)
                confidence = min(0.75, 0.5 + volume_strength * 0.25)
            
            # Volume exhaustion signals
            elif volume_strength < 0.5 and volume_ratio < 0.7:
                signal_type = OBVSignalType.VOLUME_EXHAUSTION.value
                strength = 1.0 - volume_strength
                confidence = min(0.7, 0.4 + strength * 0.3)
            
            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OBV signals: {str(e)}")
            return {'signal_type': 'NONE', 'strength': 0.0, 'confidence': 0.0}
    
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
                divergence_signals = [s for s in self.signal_history if 'divergence' in s.signal_type.lower()]
                trend_confirmation_signals = [s for s in self.signal_history if 'confirmation' in s.signal_type.lower()]
                
                if high_confidence_signals:
                    self.performance_stats['successful_signals'] = len(high_confidence_signals)
                    self.performance_stats['accuracy'] = len(high_confidence_signals) / len(self.signal_history)
                
                if divergence_signals:
                    successful_divergences = [s for s in divergence_signals if s.confidence > 0.8]
                    self.performance_stats['divergence_accuracy'] = len(successful_divergences) / len(divergence_signals)
                
                if trend_confirmation_signals:
                    successful_confirmations = [s for s in trend_confirmation_signals if s.confidence > 0.7]
                    self.performance_stats['trend_confirmation_accuracy'] = len(successful_confirmations) / len(trend_confirmation_signals)
                    
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'obv_values': np.array([]),
            'obv_smoothed': np.array([]),
            'obv_trend': [],
            'volume_strength': np.array([]),
            'divergence_signals': np.array([]),
            'volume_ratios': np.array([]),
            'accumulation_distribution': [],
            'smoothing_period_used': self.smoothing_period,
            'divergence_lookback_used': self.divergence_lookback
        }
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.signal_history = []
        self.divergence_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'divergence_accuracy': 0.0,
            'trend_confirmation_accuracy': 0.0
        }
        logger.info("OBV performance stats reset")

# Example usage and testing
if __name__ == "__main__":
    # Create sample price and volume data
    np.random.seed(42)
    n_points = 100
    base_price = 100
    
    # Generate realistic price data with trends
    close_prices = np.zeros(n_points)
    volume_data = np.zeros(n_points)
    
    for i in range(n_points):
        if i < 40:
            # Uptrend with increasing volume
            close_prices[i] = base_price + i * 0.05 + np.random.randn() * 0.02
            volume_data[i] = 1000 + i * 10 + np.random.randn() * 100
        elif i < 70:
            # Downtrend with high volume
            close_prices[i] = close_prices[39] - (i-40) * 0.03 + np.random.randn() * 0.02
            volume_data[i] = 1400 + np.random.randn() * 200
        else:
            # Consolidation with decreasing volume
            close_prices[i] = close_prices[69] + np.random.randn() * 0.01
            volume_data[i] = 800 + np.random.randn() * 100
    
    # Ensure positive volume
    volume_data = np.abs(volume_data)
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
    
    # Initialize OBV
    obv = OBV(smoothing_period=10, divergence_lookback=20, volume_threshold=1.5)
    
    # Calculate OBV
    result = obv.calculate_obv(close_prices, volume_data)
    print("OBV calculation completed")
    print(f"Latest OBV: {result['obv_values'][-1]:.0f}")
    print(f"OBV trend: {result['obv_trend'][-1]}")
    print(f"Volume strength: {result['volume_strength'][-1]:.2f}")
    print(f"Accumulation/Distribution: {result['accumulation_distribution'][-1]}")
    
    # Generate signals
    signals = obv.generate_signals(close_prices, volume_data, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")
    
    # Display performance stats
    stats = obv.get_performance_stats()
    print(f"Performance stats: {stats}")
