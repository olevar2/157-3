"""
Parabolic SAR (Stop and Reverse) Indicator
Advanced implementation with adaptive acceleration and trend following
Optimized for M1-H4 timeframes and dynamic stop-loss management
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

class SARTrend(Enum):
    """SAR trend directions"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    REVERSAL = "reversal"

class SARSignalType(Enum):
    """SAR signal types"""
    TREND_REVERSAL_BULLISH = "trend_reversal_bullish"
    TREND_REVERSAL_BEARISH = "trend_reversal_bearish"
    TREND_CONTINUATION_BULLISH = "trend_continuation_bullish"
    TREND_CONTINUATION_BEARISH = "trend_continuation_bearish"
    STOP_LOSS_ADJUSTMENT = "stop_loss_adjustment"
    ACCELERATION_INCREASE = "acceleration_increase"

@dataclass
class ParabolicSARSignal:
    """Parabolic SAR signal data structure"""
    timestamp: datetime
    price: float
    sar_value: float
    trend_direction: str
    acceleration_factor: float
    extreme_point: float
    signal_type: str
    signal_strength: float
    confidence: float
    stop_distance: float
    risk_reward_ratio: float
    timeframe: str
    session: str

class ParabolicSAR:
    """
    Advanced Parabolic SAR implementation for forex trading
    Features:
    - Adaptive acceleration factor adjustment
    - Dynamic stop-loss calculation
    - Trend reversal detection
    - Risk-reward ratio analysis
    - Session-aware analysis
    - Multiple timeframe support
    """
    
    def __init__(self, 
                 initial_af: float = 0.02,
                 max_af: float = 0.2,
                 af_increment: float = 0.02,
                 adaptive: bool = True,
                 timeframes: List[str] = None):
        """
        Initialize Parabolic SAR calculator
        
        Args:
            initial_af: Initial acceleration factor
            max_af: Maximum acceleration factor
            af_increment: Acceleration factor increment
            adaptive: Enable adaptive parameter adjustment
            timeframes: List of timeframes to analyze
        """
        self.initial_af = initial_af
        self.max_af = max_af
        self.af_increment = af_increment
        self.adaptive = adaptive
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']
        
        # Signal thresholds
        self.reversal_threshold = 0.001  # Minimum price movement for reversal
        self.strong_trend_threshold = 0.1  # AF threshold for strong trend
        self.risk_reward_min = 1.5  # Minimum risk-reward ratio
        
        # Performance tracking
        self.signal_history = []
        self.reversal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'reversal_accuracy': 0.0,
            'avg_risk_reward': 0.0
        }
        
        logger.info(f"ParabolicSAR initialized: initial_af={initial_af}, max_af={max_af}, "
                   f"increment={af_increment}, adaptive={adaptive}")
    
    def calculate_sar(self, 
                     high: Union[pd.Series, np.ndarray],
                     low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray],
                     timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Parabolic SAR for given OHLC data
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps for session analysis
            
        Returns:
            Dictionary containing SAR calculations
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            
            if len(close_array) < 3:
                logger.warning(f"Insufficient data: {len(close_array)} < 3")
                return self._empty_result()
            
            # Initialize arrays
            n = len(close_array)
            sar = np.zeros(n)
            af = np.zeros(n)
            ep = np.zeros(n)  # Extreme Point
            trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
            
            # Initialize first values
            sar[0] = low_array[0]
            af[0] = self.initial_af
            ep[0] = high_array[0]
            trend[0] = 1  # Start with uptrend
            
            # Calculate SAR values
            for i in range(1, n):
                prev_sar = sar[i-1]
                prev_af = af[i-1]
                prev_ep = ep[i-1]
                prev_trend = trend[i-1]
                
                # Calculate new SAR
                new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # Determine trend and adjust SAR
                if prev_trend == 1:  # Previous uptrend
                    # Check for trend reversal
                    if low_array[i] <= new_sar:
                        # Trend reversal to downtrend
                        trend[i] = -1
                        sar[i] = prev_ep  # SAR becomes previous extreme point
                        af[i] = self.initial_af
                        ep[i] = low_array[i]
                    else:
                        # Continue uptrend
                        trend[i] = 1
                        sar[i] = max(new_sar, max(low_array[i-1], low_array[i-2] if i > 1 else low_array[i-1]))
                        
                        # Update extreme point and acceleration factor
                        if high_array[i] > prev_ep:
                            ep[i] = high_array[i]
                            af[i] = min(self.max_af, prev_af + self.af_increment)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af
                
                else:  # Previous downtrend
                    # Check for trend reversal
                    if high_array[i] >= new_sar:
                        # Trend reversal to uptrend
                        trend[i] = 1
                        sar[i] = prev_ep  # SAR becomes previous extreme point
                        af[i] = self.initial_af
                        ep[i] = high_array[i]
                    else:
                        # Continue downtrend
                        trend[i] = -1
                        sar[i] = min(new_sar, min(high_array[i-1], high_array[i-2] if i > 1 else high_array[i-1]))
                        
                        # Update extreme point and acceleration factor
                        if low_array[i] < prev_ep:
                            ep[i] = low_array[i]
                            af[i] = min(self.max_af, prev_af + self.af_increment)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af
                
                # Apply adaptive adjustments if enabled
                if self.adaptive:
                    af[i] = self._apply_adaptive_af(af[i], close_array, i)
            
            # Calculate additional metrics
            trend_directions = ['uptrend' if t == 1 else 'downtrend' for t in trend]
            stop_distances = self._calculate_stop_distances(close_array, sar, trend)
            risk_reward_ratios = self._calculate_risk_reward_ratios(close_array, sar, ep, trend)
            reversal_signals = self._detect_reversals(trend)
            
            result = {
                'sar_values': sar,
                'acceleration_factors': af,
                'extreme_points': ep,
                'trend_directions': trend_directions,
                'trend_numeric': trend,
                'stop_distances': stop_distances,
                'risk_reward_ratios': risk_reward_ratios,
                'reversal_signals': reversal_signals,
                'parameters_used': {
                    'initial_af': self.initial_af,
                    'max_af': self.max_af,
                    'af_increment': self.af_increment
                }
            }
            
            logger.debug(f"Parabolic SAR calculated: latest_sar={sar[-1]:.5f}, "
                        f"trend={trend_directions[-1]}, af={af[-1]:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Parabolic SAR: {str(e)}")
            return self._empty_result()
    
    def generate_signals(self, 
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[ParabolicSARSignal]:
        """
        Generate trading signals based on Parabolic SAR
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            timeframe: Current timeframe
            
        Returns:
            List of ParabolicSARSignal objects
        """
        try:
            sar_data = self.calculate_sar(high, low, close, timestamps)
            if not sar_data or 'sar_values' not in sar_data:
                return []
            
            signals = []
            current_time = datetime.now()
            
            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_sar = sar_data['sar_values'][-1]
            latest_trend = sar_data['trend_directions'][-1]
            latest_af = sar_data['acceleration_factors'][-1]
            latest_ep = sar_data['extreme_points'][-1]
            latest_stop_distance = sar_data['stop_distances'][-1]
            latest_risk_reward = sar_data['risk_reward_ratios'][-1]
            latest_reversal = sar_data['reversal_signals'][-1]
            
            # Determine current session
            session = self._get_current_session(current_time)
            
            # Generate signals based on SAR analysis
            signal_data = self._analyze_sar_signals(
                latest_price, latest_sar, latest_trend, latest_af,
                latest_ep, latest_stop_distance, latest_risk_reward, latest_reversal
            )
            
            if signal_data['signal_type'] != 'NONE':
                signal = ParabolicSARSignal(
                    timestamp=current_time,
                    price=latest_price,
                    sar_value=latest_sar,
                    trend_direction=latest_trend,
                    acceleration_factor=latest_af,
                    extreme_point=latest_ep,
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    stop_distance=latest_stop_distance,
                    risk_reward_ratio=latest_risk_reward,
                    timeframe=timeframe,
                    session=session
                )
                
                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()
                
                logger.info(f"Parabolic SAR signal generated: {signal.signal_type} "
                           f"(trend={signal.trend_direction}, af={signal.acceleration_factor:.4f}, "
                           f"confidence={signal.confidence:.2f})")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Parabolic SAR signals: {str(e)}")
            return []
    
    def _apply_adaptive_af(self, current_af: float, prices: np.ndarray, index: int) -> float:
        """Apply adaptive acceleration factor adjustments"""
        try:
            if index < 10:
                return current_af
            
            # Calculate recent volatility
            recent_returns = np.diff(prices[max(0, index-10):index+1]) / prices[max(0, index-10):index]
            volatility = np.std(recent_returns)
            
            # Adjust AF based on volatility
            if volatility > 0.02:  # High volatility - slower acceleration
                adjustment = -0.005
            elif volatility < 0.005:  # Low volatility - faster acceleration
                adjustment = 0.005
            else:
                adjustment = 0.0
            
            # Apply adjustment with bounds
            adjusted_af = current_af + adjustment
            return np.clip(adjusted_af, self.initial_af, self.max_af)
            
        except Exception as e:
            logger.error(f"Error applying adaptive AF: {str(e)}")
            return current_af
    
    def _calculate_stop_distances(self, prices: np.ndarray, sar: np.ndarray, trend: np.ndarray) -> np.ndarray:
        """Calculate stop-loss distances as percentage of price"""
        try:
            stop_distances = np.zeros_like(prices)
            
            for i in range(len(prices)):
                if prices[i] > 0:
                    distance = abs(prices[i] - sar[i]) / prices[i]
                    stop_distances[i] = distance
                else:
                    stop_distances[i] = 0.01  # Default 1%
            
            return stop_distances
            
        except Exception as e:
            logger.error(f"Error calculating stop distances: {str(e)}")
            return np.ones_like(prices) * 0.01
    
    def _calculate_risk_reward_ratios(self, prices: np.ndarray, sar: np.ndarray, 
                                     ep: np.ndarray, trend: np.ndarray) -> np.ndarray:
        """Calculate risk-reward ratios based on SAR and extreme points"""
        try:
            risk_reward = np.zeros_like(prices)
            
            for i in range(len(prices)):
                # Risk = distance to SAR (stop-loss)
                risk = abs(prices[i] - sar[i])
                
                # Reward = distance to extreme point (potential target)
                reward = abs(ep[i] - prices[i])
                
                # Calculate ratio
                if risk > 0:
                    ratio = reward / risk
                    risk_reward[i] = min(10.0, ratio)  # Cap at 10:1
                else:
                    risk_reward[i] = 1.0
            
            return risk_reward
            
        except Exception as e:
            logger.error(f"Error calculating risk-reward ratios: {str(e)}")
            return np.ones_like(prices)
    
    def _detect_reversals(self, trend: np.ndarray) -> np.ndarray:
        """Detect trend reversals"""
        try:
            reversals = np.zeros_like(trend)
            
            for i in range(1, len(trend)):
                if trend[i] != trend[i-1]:
                    reversals[i] = trend[i]  # 1 for bullish reversal, -1 for bearish
                else:
                    reversals[i] = 0
            
            return reversals
            
        except Exception as e:
            logger.error(f"Error detecting reversals: {str(e)}")
            return np.zeros_like(trend)
    
    def _analyze_sar_signals(self, price: float, sar: float, trend: str, af: float,
                            ep: float, stop_distance: float, risk_reward: float, 
                            reversal: float) -> Dict:
        """Analyze current SAR conditions and generate signal"""
        try:
            signal_type = 'NONE'
            strength = 0.0
            confidence = 0.0
            
            # Trend reversal signals (highest priority)
            if reversal != 0:
                if reversal == 1:  # Bullish reversal
                    signal_type = SARSignalType.TREND_REVERSAL_BULLISH.value
                    strength = min(1.0, af / self.initial_af)
                    confidence = min(0.9, 0.7 + (risk_reward / 5.0) * 0.2)
                elif reversal == -1:  # Bearish reversal
                    signal_type = SARSignalType.TREND_REVERSAL_BEARISH.value
                    strength = min(1.0, af / self.initial_af)
                    confidence = min(0.9, 0.7 + (risk_reward / 5.0) * 0.2)
            
            # Trend continuation signals
            elif trend == SARTrend.UPTREND.value and price > sar:
                if af >= self.strong_trend_threshold:
                    signal_type = SARSignalType.ACCELERATION_INCREASE.value
                    strength = af / self.max_af
                    confidence = min(0.85, 0.6 + strength * 0.25)
                else:
                    signal_type = SARSignalType.TREND_CONTINUATION_BULLISH.value
                    strength = (price - sar) / price
                    confidence = min(0.8, 0.5 + strength * 10 + (risk_reward / 5.0) * 0.15)
            
            elif trend == SARTrend.DOWNTREND.value and price < sar:
                if af >= self.strong_trend_threshold:
                    signal_type = SARSignalType.ACCELERATION_INCREASE.value
                    strength = af / self.max_af
                    confidence = min(0.85, 0.6 + strength * 0.25)
                else:
                    signal_type = SARSignalType.TREND_CONTINUATION_BEARISH.value
                    strength = (sar - price) / price
                    confidence = min(0.8, 0.5 + strength * 10 + (risk_reward / 5.0) * 0.15)
            
            # Stop-loss adjustment signals
            elif stop_distance > 0.02:  # Stop distance > 2%
                signal_type = SARSignalType.STOP_LOSS_ADJUSTMENT.value
                strength = min(1.0, stop_distance / 0.05)  # Normalize to 5%
                confidence = 0.6
            
            # Apply risk-reward filter
            if risk_reward < self.risk_reward_min and signal_type != 'NONE':
                confidence *= 0.7  # Reduce confidence for poor risk-reward
            
            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SAR signals: {str(e)}")
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
                self.performance_stats['avg_risk_reward'] = np.mean([s.risk_reward_ratio for s in self.signal_history])
                
                # Estimate accuracy based on signal types and confidence
                high_confidence_signals = [s for s in self.signal_history if s.confidence > 0.7]
                reversal_signals = [s for s in self.signal_history if 'reversal' in s.signal_type.lower()]
                
                if high_confidence_signals:
                    self.performance_stats['successful_signals'] = len(high_confidence_signals)
                    self.performance_stats['accuracy'] = len(high_confidence_signals) / len(self.signal_history)
                
                if reversal_signals:
                    successful_reversals = [s for s in reversal_signals if s.confidence > 0.8]
                    self.performance_stats['reversal_accuracy'] = len(successful_reversals) / len(reversal_signals)
                    
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'sar_values': np.array([]),
            'acceleration_factors': np.array([]),
            'extreme_points': np.array([]),
            'trend_directions': [],
            'trend_numeric': np.array([]),
            'stop_distances': np.array([]),
            'risk_reward_ratios': np.array([]),
            'reversal_signals': np.array([]),
            'parameters_used': {
                'initial_af': self.initial_af,
                'max_af': self.max_af,
                'af_increment': self.af_increment
            }
        }
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.signal_history = []
        self.reversal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'reversal_accuracy': 0.0,
            'avg_risk_reward': 0.0
        }
        logger.info("Parabolic SAR performance stats reset")

# Example usage and testing
if __name__ == "__main__":
    # Create sample OHLC data with clear trends
    np.random.seed(42)
    n_points = 100
    base_price = 100
    
    # Generate realistic OHLC data with trend changes
    close_prices = np.zeros(n_points)
    for i in range(n_points):
        if i < 40:
            close_prices[i] = base_price + i * 0.05 + np.random.randn() * 0.02  # Strong uptrend
        elif i < 70:
            close_prices[i] = close_prices[39] - (i-40) * 0.03 + np.random.randn() * 0.02  # Downtrend
        else:
            close_prices[i] = close_prices[69] + (i-70) * 0.02 + np.random.randn() * 0.01  # Weak uptrend
    
    high_prices = close_prices + np.abs(np.random.randn(n_points) * 0.01)
    low_prices = close_prices - np.abs(np.random.randn(n_points) * 0.01)
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
    
    # Initialize Parabolic SAR
    psar = ParabolicSAR(initial_af=0.02, max_af=0.2, af_increment=0.02, adaptive=True)
    
    # Calculate SAR
    result = psar.calculate_sar(high_prices, low_prices, close_prices)
    print("Parabolic SAR calculation completed")
    print(f"Latest SAR: {result['sar_values'][-1]:.5f}")
    print(f"Latest trend: {result['trend_directions'][-1]}")
    print(f"Latest AF: {result['acceleration_factors'][-1]:.4f}")
    print(f"Risk-Reward ratio: {result['risk_reward_ratios'][-1]:.2f}")
    
    # Generate signals
    signals = psar.generate_signals(high_prices, low_prices, close_prices, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")
    
    # Display performance stats
    stats = psar.get_performance_stats()
    print(f"Performance stats: {stats}")
