"""
Vortex Indicator (VI) - Volatility and Trend Change Detection
Advanced implementation with trend reversal and momentum analysis
Optimized for M1-H4 timeframes and trend change identification
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

class VortexSignal(Enum):
    """Vortex signal types"""
    BULLISH_CROSSOVER = "bullish_crossover"
    BEARISH_CROSSOVER = "bearish_crossover"
    STRONG_UPTREND = "strong_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    TREND_WEAKENING = "trend_weakening"
    CONSOLIDATION = "consolidation"

@dataclass
class VortexIndicatorSignal:
    """Vortex Indicator signal data structure"""
    timestamp: datetime
    price: float
    vi_plus: float
    vi_minus: float
    vi_difference: float
    trend_strength: float
    signal_type: str
    signal_strength: float
    confidence: float
    momentum_score: float
    timeframe: str
    session: str

class VortexIndicator:
    """
    Advanced Vortex Indicator implementation for forex trading
    Features:
    - Dual vortex movement calculation (VI+ and VI-)
    - Trend change detection through crossovers
    - Momentum and trend strength analysis
    - Signal filtering and confirmation
    - Session-aware analysis
    - Adaptive period adjustment
    """
    
    def __init__(self, 
                 period: int = 14,
                 adaptive: bool = True,
                 min_period: int = 10,
                 max_period: int = 25,
                 timeframes: List[str] = None):
        """
        Initialize Vortex Indicator calculator
        
        Args:
            period: Period for VI calculation
            adaptive: Enable adaptive period adjustment
            min_period: Minimum period for adaptive mode
            max_period: Maximum period for adaptive mode
            timeframes: List of timeframes to analyze
        """
        self.period = period
        self.adaptive = adaptive
        self.min_period = min_period
        self.max_period = max_period
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']
        
        # Signal thresholds
        self.crossover_threshold = 0.02  # Minimum difference for valid crossover
        self.strong_trend_threshold = 1.2  # VI value for strong trend
        self.weak_trend_threshold = 0.95  # VI value for weak trend
        self.consolidation_threshold = 0.05  # VI difference for consolidation
        
        # Performance tracking
        self.signal_history = []
        self.crossover_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'crossover_accuracy': 0.0,
            'trend_accuracy': 0.0
        }
        
        logger.info(f"VortexIndicator initialized: period={period}, adaptive={adaptive}")
    
    def calculate_vortex(self, 
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Vortex Indicator for given OHLC data
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps for session analysis
            
        Returns:
            Dictionary containing Vortex calculations
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            
            if len(close_array) < self.period + 1:
                logger.warning(f"Insufficient data: {len(close_array)} < {self.period + 1}")
                return self._empty_result()
            
            # Calculate adaptive period if enabled
            current_period = self._calculate_adaptive_period(close_array) if self.adaptive else self.period
            
            # Calculate Vortex Movements
            vm_plus, vm_minus = self._calculate_vortex_movements(high_array, low_array, close_array)
            
            # Calculate True Range
            true_range = self._calculate_true_range(high_array, low_array, close_array)
            
            # Calculate Vortex Indicators
            vi_plus, vi_minus = self._calculate_vortex_indicators(vm_plus, vm_minus, true_range, current_period)
            
            # Calculate additional metrics
            vi_difference = vi_plus - vi_minus
            trend_strength = self._calculate_trend_strength(vi_plus, vi_minus)
            momentum_score = self._calculate_momentum_score(vi_plus, vi_minus, vi_difference)
            crossover_signals = self._detect_crossovers(vi_plus, vi_minus)
            
            result = {
                'vi_plus': vi_plus,
                'vi_minus': vi_minus,
                'vi_difference': vi_difference,
                'vm_plus': vm_plus,
                'vm_minus': vm_minus,
                'true_range': true_range,
                'trend_strength': trend_strength,
                'momentum_score': momentum_score,
                'crossover_signals': crossover_signals,
                'period_used': current_period
            }
            
            logger.debug(f"Vortex calculated: VI+={vi_plus[-1]:.4f}, VI-={vi_minus[-1]:.4f}, "
                        f"diff={vi_difference[-1]:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Vortex Indicator: {str(e)}")
            return self._empty_result()
    
    def generate_signals(self, 
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[VortexIndicatorSignal]:
        """
        Generate trading signals based on Vortex Indicator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            timeframe: Current timeframe
            
        Returns:
            List of VortexIndicatorSignal objects
        """
        try:
            vortex_data = self.calculate_vortex(high, low, close, timestamps)
            if not vortex_data or 'vi_plus' not in vortex_data:
                return []
            
            signals = []
            current_time = datetime.now()
            
            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_vi_plus = vortex_data['vi_plus'][-1]
            latest_vi_minus = vortex_data['vi_minus'][-1]
            latest_difference = vortex_data['vi_difference'][-1]
            latest_trend_strength = vortex_data['trend_strength'][-1]
            latest_momentum = vortex_data['momentum_score'][-1]
            latest_crossover = vortex_data['crossover_signals'][-1]
            
            # Determine current session
            session = self._get_current_session(current_time)
            
            # Generate signals based on Vortex analysis
            signal_data = self._analyze_vortex_signals(
                latest_vi_plus, latest_vi_minus, latest_difference,
                latest_trend_strength, latest_momentum, latest_crossover
            )
            
            if signal_data['signal_type'] != 'NONE':
                signal = VortexIndicatorSignal(
                    timestamp=current_time,
                    price=latest_price,
                    vi_plus=latest_vi_plus,
                    vi_minus=latest_vi_minus,
                    vi_difference=latest_difference,
                    trend_strength=latest_trend_strength,
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    momentum_score=latest_momentum,
                    timeframe=timeframe,
                    session=session
                )
                
                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()
                
                logger.info(f"Vortex signal generated: {signal.signal_type} "
                           f"(VI+={signal.vi_plus:.3f}, VI-={signal.vi_minus:.3f}, "
                           f"confidence={signal.confidence:.2f})")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Vortex signals: {str(e)}")
            return []
    
    def _calculate_adaptive_period(self, prices: np.ndarray) -> int:
        """Calculate adaptive period based on market volatility"""
        try:
            if len(prices) < 20:
                return self.period
            
            # Calculate recent volatility
            recent_returns = np.diff(prices[-20:]) / prices[-20:-1]
            volatility = np.std(recent_returns)
            
            # Adjust period based on volatility
            if volatility > 0.02:  # High volatility
                adaptive_period = max(self.min_period, self.period - 3)
            elif volatility < 0.005:  # Low volatility
                adaptive_period = min(self.max_period, self.period + 5)
            else:
                adaptive_period = self.period
            
            return adaptive_period
            
        except Exception as e:
            logger.error(f"Error calculating adaptive period: {str(e)}")
            return self.period
    
    def _calculate_vortex_movements(self, high: np.ndarray, low: np.ndarray, 
                                   close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Vortex Movements (VM+ and VM-)"""
        try:
            # Shift arrays to get previous values
            prev_high = np.roll(high, 1)
            prev_low = np.roll(low, 1)
            prev_close = np.roll(close, 1)
            
            # Set first values to avoid NaN
            prev_high[0] = high[0]
            prev_low[0] = low[0]
            prev_close[0] = close[0]
            
            # Calculate Vortex Movements
            vm_plus = np.abs(high - prev_low)  # Current high - previous low
            vm_minus = np.abs(low - prev_high)  # Current low - previous high
            
            return vm_plus, vm_minus
            
        except Exception as e:
            logger.error(f"Error calculating vortex movements: {str(e)}")
            return np.zeros_like(high), np.zeros_like(high)
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range"""
        try:
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            return true_range
            
        except Exception as e:
            logger.error(f"Error calculating true range: {str(e)}")
            return np.ones_like(high) * 0.001
    
    def _calculate_vortex_indicators(self, vm_plus: np.ndarray, vm_minus: np.ndarray, 
                                   true_range: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Vortex Indicators (VI+ and VI-)"""
        try:
            vi_plus = np.zeros_like(vm_plus)
            vi_minus = np.zeros_like(vm_minus)
            
            for i in range(len(vm_plus)):
                if i < period - 1:
                    # Use available data for initial values
                    sum_vm_plus = np.sum(vm_plus[:i+1])
                    sum_vm_minus = np.sum(vm_minus[:i+1])
                    sum_tr = np.sum(true_range[:i+1])
                else:
                    # Use rolling window
                    sum_vm_plus = np.sum(vm_plus[i-period+1:i+1])
                    sum_vm_minus = np.sum(vm_minus[i-period+1:i+1])
                    sum_tr = np.sum(true_range[i-period+1:i+1])
                
                # Calculate VI+ and VI-
                if sum_tr > 0:
                    vi_plus[i] = sum_vm_plus / sum_tr
                    vi_minus[i] = sum_vm_minus / sum_tr
                else:
                    vi_plus[i] = 1.0
                    vi_minus[i] = 1.0
            
            return vi_plus, vi_minus
            
        except Exception as e:
            logger.error(f"Error calculating vortex indicators: {str(e)}")
            return np.ones_like(vm_plus), np.ones_like(vm_minus)
    
    def _calculate_trend_strength(self, vi_plus: np.ndarray, vi_minus: np.ndarray) -> np.ndarray:
        """Calculate trend strength based on VI values"""
        try:
            trend_strength = np.zeros_like(vi_plus)
            
            for i in range(len(vi_plus)):
                # Trend strength based on dominant VI and separation
                if vi_plus[i] > vi_minus[i]:
                    # Uptrend strength
                    strength = vi_plus[i] + (vi_plus[i] - vi_minus[i])
                else:
                    # Downtrend strength
                    strength = vi_minus[i] + (vi_minus[i] - vi_plus[i])
                
                trend_strength[i] = min(3.0, strength)  # Cap at 3.0
            
            return trend_strength
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return np.ones_like(vi_plus)
    
    def _calculate_momentum_score(self, vi_plus: np.ndarray, vi_minus: np.ndarray, 
                                 vi_difference: np.ndarray) -> np.ndarray:
        """Calculate momentum score based on VI dynamics"""
        try:
            momentum_score = np.zeros_like(vi_plus)
            
            for i in range(1, len(vi_plus)):
                # Current momentum based on VI difference
                current_momentum = abs(vi_difference[i])
                
                # Momentum change (acceleration/deceleration)
                if i > 0:
                    momentum_change = abs(vi_difference[i]) - abs(vi_difference[i-1])
                else:
                    momentum_change = 0
                
                # Combined momentum score
                score = current_momentum + (momentum_change * 0.5)
                momentum_score[i] = max(0, min(2.0, score))  # Bound between 0 and 2
            
            return momentum_score
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {str(e)}")
            return np.ones_like(vi_plus) * 0.5
    
    def _detect_crossovers(self, vi_plus: np.ndarray, vi_minus: np.ndarray) -> np.ndarray:
        """Detect crossovers between VI+ and VI-"""
        try:
            crossovers = np.zeros_like(vi_plus)
            
            for i in range(1, len(vi_plus)):
                prev_diff = vi_plus[i-1] - vi_minus[i-1]
                curr_diff = vi_plus[i] - vi_minus[i]
                
                # Bullish crossover (VI+ crosses above VI-)
                if prev_diff <= 0 and curr_diff > self.crossover_threshold:
                    crossovers[i] = 1.0
                # Bearish crossover (VI+ crosses below VI-)
                elif prev_diff >= 0 and curr_diff < -self.crossover_threshold:
                    crossovers[i] = -1.0
                else:
                    crossovers[i] = 0.0
            
            return crossovers
            
        except Exception as e:
            logger.error(f"Error detecting crossovers: {str(e)}")
            return np.zeros_like(vi_plus)
    
    def _analyze_vortex_signals(self, vi_plus: float, vi_minus: float, vi_difference: float,
                               trend_strength: float, momentum: float, crossover: float) -> Dict:
        """Analyze current Vortex conditions and generate signal"""
        try:
            signal_type = 'NONE'
            strength = 0.0
            confidence = 0.0
            
            # Crossover signals (highest priority)
            if crossover > 0.5:  # Bullish crossover
                signal_type = VortexSignal.BULLISH_CROSSOVER.value
                strength = min(1.0, abs(vi_difference) * 2)
                confidence = min(0.9, 0.7 + momentum * 0.1 + strength * 0.1)
            elif crossover < -0.5:  # Bearish crossover
                signal_type = VortexSignal.BEARISH_CROSSOVER.value
                strength = min(1.0, abs(vi_difference) * 2)
                confidence = min(0.9, 0.7 + momentum * 0.1 + strength * 0.1)
            
            # Strong trend signals
            elif vi_plus > self.strong_trend_threshold and vi_plus > vi_minus:
                signal_type = VortexSignal.STRONG_UPTREND.value
                strength = min(1.0, (vi_plus - 1.0) * 2)
                confidence = min(0.85, 0.6 + strength * 0.15 + momentum * 0.1)
            elif vi_minus > self.strong_trend_threshold and vi_minus > vi_plus:
                signal_type = VortexSignal.STRONG_DOWNTREND.value
                strength = min(1.0, (vi_minus - 1.0) * 2)
                confidence = min(0.85, 0.6 + strength * 0.15 + momentum * 0.1)
            
            # Trend weakening signals
            elif max(vi_plus, vi_minus) < self.weak_trend_threshold:
                signal_type = VortexSignal.TREND_WEAKENING.value
                strength = 1.0 - max(vi_plus, vi_minus)
                confidence = min(0.75, 0.5 + strength * 0.25)
            
            # Consolidation signals
            elif abs(vi_difference) < self.consolidation_threshold:
                signal_type = VortexSignal.CONSOLIDATION.value
                strength = 1.0 - abs(vi_difference) / self.consolidation_threshold
                confidence = min(0.7, 0.4 + strength * 0.3)
            
            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Vortex signals: {str(e)}")
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
                crossover_signals = [s for s in self.signal_history if 'crossover' in s.signal_type.lower()]
                trend_signals = [s for s in self.signal_history if 'trend' in s.signal_type.lower()]
                
                if high_confidence_signals:
                    self.performance_stats['successful_signals'] = len(high_confidence_signals)
                    self.performance_stats['accuracy'] = len(high_confidence_signals) / len(self.signal_history)
                
                if crossover_signals:
                    successful_crossovers = [s for s in crossover_signals if s.confidence > 0.8]
                    self.performance_stats['crossover_accuracy'] = len(successful_crossovers) / len(crossover_signals)
                
                if trend_signals:
                    successful_trends = [s for s in trend_signals if s.confidence > 0.7]
                    self.performance_stats['trend_accuracy'] = len(successful_trends) / len(trend_signals)
                    
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'vi_plus': np.array([]),
            'vi_minus': np.array([]),
            'vi_difference': np.array([]),
            'vm_plus': np.array([]),
            'vm_minus': np.array([]),
            'true_range': np.array([]),
            'trend_strength': np.array([]),
            'momentum_score': np.array([]),
            'crossover_signals': np.array([]),
            'period_used': self.period
        }
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.signal_history = []
        self.crossover_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'crossover_accuracy': 0.0,
            'trend_accuracy': 0.0
        }
        logger.info("Vortex Indicator performance stats reset")

# Example usage and testing
if __name__ == "__main__":
    # Create sample OHLC data
    np.random.seed(42)
    n_points = 100
    base_price = 100
    
    # Generate realistic OHLC data with some trend changes
    trend_changes = [0, 30, 60, 80]  # Points where trend changes
    close_prices = np.zeros(n_points)
    
    for i in range(n_points):
        if i < 30:
            close_prices[i] = base_price + i * 0.02 + np.random.randn() * 0.01  # Uptrend
        elif i < 60:
            close_prices[i] = close_prices[29] - (i-30) * 0.015 + np.random.randn() * 0.01  # Downtrend
        elif i < 80:
            close_prices[i] = close_prices[59] + (i-60) * 0.01 + np.random.randn() * 0.01  # Weak uptrend
        else:
            close_prices[i] = close_prices[79] + np.random.randn() * 0.005  # Consolidation
    
    high_prices = close_prices + np.abs(np.random.randn(n_points) * 0.005)
    low_prices = close_prices - np.abs(np.random.randn(n_points) * 0.005)
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
    
    # Initialize Vortex Indicator
    vi = VortexIndicator(period=14, adaptive=True)
    
    # Calculate Vortex
    result = vi.calculate_vortex(high_prices, low_prices, close_prices)
    print("Vortex Indicator calculation completed")
    print(f"Latest VI+: {result['vi_plus'][-1]:.4f}")
    print(f"Latest VI-: {result['vi_minus'][-1]:.4f}")
    print(f"VI Difference: {result['vi_difference'][-1]:.4f}")
    print(f"Trend strength: {result['trend_strength'][-1]:.2f}")
    
    # Generate signals
    signals = vi.generate_signals(high_prices, low_prices, close_prices, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")
    
    # Display performance stats
    stats = vi.get_performance_stats()
    print(f"Performance stats: {stats}")
