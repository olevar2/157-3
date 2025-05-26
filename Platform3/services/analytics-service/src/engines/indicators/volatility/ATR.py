"""
Average True Range (ATR) Volatility Indicator
Advanced implementation with multiple smoothing methods and volatility analysis
Optimized for M1-H4 timeframes and risk management
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

class ATRSmoothingMethod(Enum):
    """ATR smoothing methods"""
    WILDER = "wilder"
    SMA = "sma"
    EMA = "ema"
    ADAPTIVE = "adaptive"

class VolatilityRegime(Enum):
    """Market volatility regimes"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class ATRSignal:
    """ATR signal data structure"""
    timestamp: datetime
    atr_value: float
    atr_percentage: float
    volatility_regime: str
    trend_strength: float
    breakout_potential: float
    risk_level: str
    signal_type: str
    confidence: float
    timeframe: str
    session: str

class ATR:
    """
    Advanced Average True Range implementation for forex trading
    Features:
    - Multiple smoothing methods (Wilder's, SMA, EMA, Adaptive)
    - Volatility regime classification
    - Breakout potential analysis
    - Risk level assessment
    - Session-aware analysis
    - Real-time signal generation
    """
    
    def __init__(self, 
                 period: int = 14,
                 smoothing_method: ATRSmoothingMethod = ATRSmoothingMethod.WILDER,
                 adaptive: bool = True,
                 timeframes: List[str] = None):
        """
        Initialize ATR calculator
        
        Args:
            period: Period for ATR calculation
            smoothing_method: Method for smoothing ATR values
            adaptive: Enable adaptive period adjustment
            timeframes: List of timeframes to analyze
        """
        self.period = period
        self.smoothing_method = smoothing_method
        self.adaptive = adaptive
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']
        
        # Volatility thresholds (as percentage of price)
        self.low_volatility_threshold = 0.005  # 0.5%
        self.normal_volatility_threshold = 0.015  # 1.5%
        self.high_volatility_threshold = 0.03  # 3.0%
        
        # Signal thresholds
        self.breakout_threshold = 1.5  # ATR multiplier for breakout detection
        self.trend_strength_threshold = 1.2  # ATR multiplier for trend strength
        
        # Performance tracking
        self.signal_history = []
        self.volatility_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'volatility_accuracy': 0.0
        }
        
        logger.info(f"ATR initialized: period={period}, smoothing={smoothing_method.value}, adaptive={adaptive}")
    
    def calculate_atr(self, 
                     high: Union[pd.Series, np.ndarray],
                     low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray],
                     timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Average True Range for given OHLC data
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps for session analysis
            
        Returns:
            Dictionary containing ATR calculations
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            
            if len(high_array) < self.period:
                logger.warning(f"Insufficient data: {len(high_array)} < {self.period}")
                return self._empty_result()
            
            # Calculate True Range
            true_range = self._calculate_true_range(high_array, low_array, close_array)
            
            # Calculate ATR using selected smoothing method
            atr_values = self._calculate_atr_smoothed(true_range)
            
            # Calculate ATR as percentage of price
            atr_percentage = (atr_values / close_array) * 100
            
            # Classify volatility regimes
            volatility_regimes = self._classify_volatility_regimes(atr_percentage)
            
            # Calculate additional metrics
            trend_strength = self._calculate_trend_strength(atr_values, close_array)
            breakout_potential = self._calculate_breakout_potential(atr_values, true_range)
            risk_levels = self._assess_risk_levels(atr_percentage, volatility_regimes)
            
            result = {
                'atr_values': atr_values,
                'atr_percentage': atr_percentage,
                'true_range': true_range,
                'volatility_regimes': volatility_regimes,
                'trend_strength': trend_strength,
                'breakout_potential': breakout_potential,
                'risk_levels': risk_levels,
                'period_used': self.period,
                'smoothing_method': self.smoothing_method.value
            }
            
            logger.debug(f"ATR calculated: latest_atr={atr_values[-1]:.6f}, "
                        f"percentage={atr_percentage[-1]:.3f}%, regime={volatility_regimes[-1]}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return self._empty_result()
    
    def generate_signals(self, 
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[ATRSignal]:
        """
        Generate trading signals based on ATR analysis
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            timeframe: Current timeframe
            
        Returns:
            List of ATRSignal objects
        """
        try:
            atr_data = self.calculate_atr(high, low, close, timestamps)
            if not atr_data or 'atr_values' not in atr_data:
                return []
            
            signals = []
            current_time = datetime.now()
            
            # Get latest values
            latest_atr = atr_data['atr_values'][-1]
            latest_atr_pct = atr_data['atr_percentage'][-1]
            latest_regime = atr_data['volatility_regimes'][-1]
            latest_trend_strength = atr_data['trend_strength'][-1]
            latest_breakout_potential = atr_data['breakout_potential'][-1]
            latest_risk_level = atr_data['risk_levels'][-1]
            
            # Determine current session
            session = self._get_current_session(current_time)
            
            # Generate signals based on ATR analysis
            signal_data = self._analyze_atr_signals(
                latest_atr, latest_atr_pct, latest_regime,
                latest_trend_strength, latest_breakout_potential, latest_risk_level
            )
            
            if signal_data['signal_type'] != 'NONE':
                signal = ATRSignal(
                    timestamp=current_time,
                    atr_value=latest_atr,
                    atr_percentage=latest_atr_pct,
                    volatility_regime=latest_regime,
                    trend_strength=latest_trend_strength,
                    breakout_potential=latest_breakout_potential,
                    risk_level=latest_risk_level,
                    signal_type=signal_data['signal_type'],
                    confidence=signal_data['confidence'],
                    timeframe=timeframe,
                    session=session
                )
                
                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()
                
                logger.info(f"ATR signal generated: {signal.signal_type} "
                           f"(regime={signal.volatility_regime}, confidence={signal.confidence:.2f})")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ATR signals: {str(e)}")
            return []
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range values"""
        try:
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]  # Handle first value
            
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            return true_range
            
        except Exception as e:
            logger.error(f"Error calculating true range: {str(e)}")
            return np.zeros_like(high)
    
    def _calculate_atr_smoothed(self, true_range: np.ndarray) -> np.ndarray:
        """Calculate ATR using selected smoothing method"""
        try:
            if self.smoothing_method == ATRSmoothingMethod.WILDER:
                return self._wilder_smoothing(true_range)
            elif self.smoothing_method == ATRSmoothingMethod.SMA:
                return self._sma_smoothing(true_range)
            elif self.smoothing_method == ATRSmoothingMethod.EMA:
                return self._ema_smoothing(true_range)
            elif self.smoothing_method == ATRSmoothingMethod.ADAPTIVE:
                return self._adaptive_smoothing(true_range)
            else:
                return self._wilder_smoothing(true_range)
                
        except Exception as e:
            logger.error(f"Error in ATR smoothing: {str(e)}")
            return np.zeros_like(true_range)
    
    def _wilder_smoothing(self, true_range: np.ndarray) -> np.ndarray:
        """Wilder's smoothing method (original ATR)"""
        atr = np.zeros_like(true_range)
        atr[0] = true_range[0]
        
        for i in range(1, len(true_range)):
            if i < self.period:
                atr[i] = np.mean(true_range[:i+1])
            else:
                atr[i] = (atr[i-1] * (self.period - 1) + true_range[i]) / self.period
        
        return atr
    
    def _sma_smoothing(self, true_range: np.ndarray) -> np.ndarray:
        """Simple Moving Average smoothing"""
        return pd.Series(true_range).rolling(window=self.period, min_periods=1).mean().values
    
    def _ema_smoothing(self, true_range: np.ndarray) -> np.ndarray:
        """Exponential Moving Average smoothing"""
        alpha = 2.0 / (self.period + 1)
        return pd.Series(true_range).ewm(alpha=alpha, adjust=False).mean().values
    
    def _adaptive_smoothing(self, true_range: np.ndarray) -> np.ndarray:
        """Adaptive smoothing based on market conditions"""
        # Use Wilder's as base, but adjust alpha based on volatility
        atr = np.zeros_like(true_range)
        atr[0] = true_range[0]
        
        for i in range(1, len(true_range)):
            if i < self.period:
                atr[i] = np.mean(true_range[:i+1])
            else:
                # Calculate volatility of recent true range values
                recent_volatility = np.std(true_range[max(0, i-10):i+1])
                avg_volatility = np.mean(true_range[max(0, i-20):i+1])
                
                # Adjust smoothing factor based on volatility
                if recent_volatility > avg_volatility * 1.5:
                    # High volatility - faster adaptation
                    alpha = 2.0 / (self.period * 0.7)
                elif recent_volatility < avg_volatility * 0.5:
                    # Low volatility - slower adaptation
                    alpha = 2.0 / (self.period * 1.3)
                else:
                    # Normal volatility - standard Wilder's
                    alpha = 1.0 / self.period
                
                atr[i] = atr[i-1] + alpha * (true_range[i] - atr[i-1])
        
        return atr
    
    def _classify_volatility_regimes(self, atr_percentage: np.ndarray) -> List[str]:
        """Classify volatility regimes based on ATR percentage"""
        try:
            regimes = []
            for atr_pct in atr_percentage:
                if atr_pct < self.low_volatility_threshold * 100:
                    regimes.append(VolatilityRegime.LOW.value)
                elif atr_pct < self.normal_volatility_threshold * 100:
                    regimes.append(VolatilityRegime.NORMAL.value)
                elif atr_pct < self.high_volatility_threshold * 100:
                    regimes.append(VolatilityRegime.HIGH.value)
                else:
                    regimes.append(VolatilityRegime.EXTREME.value)
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error classifying volatility regimes: {str(e)}")
            return [VolatilityRegime.NORMAL.value] * len(atr_percentage)
    
    def _calculate_trend_strength(self, atr_values: np.ndarray, close_prices: np.ndarray) -> np.ndarray:
        """Calculate trend strength based on ATR and price movement"""
        try:
            trend_strength = np.zeros_like(atr_values)
            
            for i in range(1, len(atr_values)):
                if i < 5:
                    trend_strength[i] = 0.5
                    continue
                
                # Calculate price change over recent periods
                price_change = abs(close_prices[i] - close_prices[i-5])
                avg_atr = np.mean(atr_values[max(0, i-5):i+1])
                
                # Trend strength = price change relative to average ATR
                if avg_atr > 0:
                    strength = min(2.0, price_change / (avg_atr * 5))
                    trend_strength[i] = strength
                else:
                    trend_strength[i] = 0.5
            
            return trend_strength
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return np.ones_like(atr_values) * 0.5
    
    def _calculate_breakout_potential(self, atr_values: np.ndarray, true_range: np.ndarray) -> np.ndarray:
        """Calculate breakout potential based on ATR expansion"""
        try:
            breakout_potential = np.zeros_like(atr_values)
            
            for i in range(self.period, len(atr_values)):
                current_atr = atr_values[i]
                avg_atr = np.mean(atr_values[max(0, i-self.period):i])
                current_tr = true_range[i]
                
                # Breakout potential based on ATR expansion and current TR
                if avg_atr > 0:
                    atr_expansion = current_atr / avg_atr
                    tr_relative = current_tr / current_atr if current_atr > 0 else 1.0
                    
                    potential = min(2.0, atr_expansion * tr_relative)
                    breakout_potential[i] = potential
                else:
                    breakout_potential[i] = 1.0
            
            return breakout_potential
            
        except Exception as e:
            logger.error(f"Error calculating breakout potential: {str(e)}")
            return np.ones_like(atr_values)
    
    def _assess_risk_levels(self, atr_percentage: np.ndarray, volatility_regimes: List[str]) -> List[str]:
        """Assess risk levels based on ATR and volatility regimes"""
        try:
            risk_levels = []
            
            for i, (atr_pct, regime) in enumerate(zip(atr_percentage, volatility_regimes)):
                if regime == VolatilityRegime.LOW.value:
                    risk_levels.append('LOW')
                elif regime == VolatilityRegime.NORMAL.value:
                    risk_levels.append('MEDIUM')
                elif regime == VolatilityRegime.HIGH.value:
                    risk_levels.append('HIGH')
                else:  # EXTREME
                    risk_levels.append('EXTREME')
            
            return risk_levels
            
        except Exception as e:
            logger.error(f"Error assessing risk levels: {str(e)}")
            return ['MEDIUM'] * len(atr_percentage)
    
    def _analyze_atr_signals(self, atr_value: float, atr_pct: float, regime: str,
                            trend_strength: float, breakout_potential: float, 
                            risk_level: str) -> Dict:
        """Analyze current ATR conditions and generate signal"""
        try:
            signal_type = 'NONE'
            confidence = 0.0
            
            # Volatility expansion signals
            if regime == VolatilityRegime.LOW.value and breakout_potential > self.breakout_threshold:
                signal_type = 'VOLATILITY_EXPANSION'
                confidence = min(0.8, 0.4 + (breakout_potential - 1.0) * 0.4)
            
            # High volatility warning
            elif regime == VolatilityRegime.EXTREME.value:
                signal_type = 'HIGH_VOLATILITY_WARNING'
                confidence = 0.9
            
            # Trend strength signals
            elif trend_strength > self.trend_strength_threshold:
                signal_type = 'STRONG_TREND'
                confidence = min(0.85, 0.5 + (trend_strength - 1.0) * 0.35)
            
            # Low volatility consolidation
            elif regime == VolatilityRegime.LOW.value and trend_strength < 0.8:
                signal_type = 'LOW_VOLATILITY_CONSOLIDATION'
                confidence = 0.6
            
            # Normal market conditions
            elif regime == VolatilityRegime.NORMAL.value:
                signal_type = 'NORMAL_CONDITIONS'
                confidence = 0.5
            
            return {
                'signal_type': signal_type,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ATR signals: {str(e)}")
            return {'signal_type': 'NONE', 'confidence': 0.0}
    
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
                if high_confidence_signals:
                    self.performance_stats['successful_signals'] = len(high_confidence_signals)
                    self.performance_stats['accuracy'] = len(high_confidence_signals) / len(self.signal_history)
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'atr_values': np.array([]),
            'atr_percentage': np.array([]),
            'true_range': np.array([]),
            'volatility_regimes': [],
            'trend_strength': np.array([]),
            'breakout_potential': np.array([]),
            'risk_levels': [],
            'period_used': self.period,
            'smoothing_method': self.smoothing_method.value
        }
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.signal_history = []
        self.volatility_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'volatility_accuracy': 0.0
        }
        logger.info("ATR performance stats reset")

# Example usage and testing
if __name__ == "__main__":
    # Create sample OHLC data
    np.random.seed(42)
    n_points = 100
    base_price = 100
    
    # Generate realistic OHLC data
    close_prices = base_price + np.cumsum(np.random.randn(n_points) * 0.01)
    high_prices = close_prices + np.abs(np.random.randn(n_points) * 0.005)
    low_prices = close_prices - np.abs(np.random.randn(n_points) * 0.005)
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
    
    # Initialize ATR
    atr = ATR(period=14, smoothing_method=ATRSmoothingMethod.WILDER, adaptive=True)
    
    # Calculate ATR
    result = atr.calculate_atr(high_prices, low_prices, close_prices)
    print("ATR calculation completed")
    print(f"Latest ATR: {result['atr_values'][-1]:.6f}")
    print(f"Latest ATR %: {result['atr_percentage'][-1]:.3f}%")
    print(f"Volatility regime: {result['volatility_regimes'][-1]}")
    
    # Generate signals
    signals = atr.generate_signals(high_prices, low_prices, close_prices, timestamps, 'M1')
    print(f"Generated {len(signals)} signals")
    
    # Display performance stats
    stats = atr.get_performance_stats()
    print(f"Performance stats: {stats}")
