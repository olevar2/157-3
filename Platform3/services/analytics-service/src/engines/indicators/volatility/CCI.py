"""
Commodity Channel Index (CCI) - Momentum and Volatility Indicator
Advanced implementation with adaptive periods and overbought/oversold detection
Optimized for M1-H4 timeframes and momentum analysis
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

class CCIZone(Enum):
    """CCI zones for signal classification"""
    EXTREME_OVERSOLD = "extreme_oversold"  # < -200
    OVERSOLD = "oversold"  # -200 to -100
    NORMAL_BEARISH = "normal_bearish"  # -100 to 0
    NORMAL_BULLISH = "normal_bullish"  # 0 to 100
    OVERBOUGHT = "overbought"  # 100 to 200
    EXTREME_OVERBOUGHT = "extreme_overbought"  # > 200

class CCISignalType(Enum):
    """CCI signal types"""
    OVERSOLD_REVERSAL = "oversold_reversal"
    OVERBOUGHT_REVERSAL = "overbought_reversal"
    ZERO_LINE_CROSS_BULLISH = "zero_line_cross_bullish"
    ZERO_LINE_CROSS_BEARISH = "zero_line_cross_bearish"
    MOMENTUM_ACCELERATION = "momentum_acceleration"
    MOMENTUM_DECELERATION = "momentum_deceleration"
    EXTREME_READING = "extreme_reading"
    DIVERGENCE_BULLISH = "divergence_bullish"
    DIVERGENCE_BEARISH = "divergence_bearish"

@dataclass
class CCISignal:
    """CCI signal data structure"""
    timestamp: datetime
    price: float
    cci_value: float
    cci_zone: str
    signal_type: str
    signal_strength: float
    confidence: float
    momentum_score: float
    divergence_strength: float
    timeframe: str
    session: str

class CCI:
    """
    Advanced Commodity Channel Index implementation for forex trading
    Features:
    - Adaptive period adjustment based on volatility
    - Multiple zone classification for signal generation
    - Divergence detection between price and CCI
    - Momentum acceleration/deceleration analysis
    - Session-aware analysis
    - Overbought/oversold reversal signals
    """
    
    def __init__(self, 
                 period: int = 20,
                 constant: float = 0.015,
                 adaptive: bool = True,
                 min_period: int = 14,
                 max_period: int = 30,
                 timeframes: List[str] = None):
        """
        Initialize CCI calculator
        
        Args:
            period: Period for CCI calculation
            constant: CCI constant (typically 0.015)
            adaptive: Enable adaptive period adjustment
            min_period: Minimum period for adaptive mode
            max_period: Maximum period for adaptive mode
            timeframes: List of timeframes to analyze
        """
        self.period = period
        self.constant = constant
        self.adaptive = adaptive
        self.min_period = min_period
        self.max_period = max_period
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']
        
        # Signal thresholds
        self.oversold_threshold = -100
        self.overbought_threshold = 100
        self.extreme_oversold_threshold = -200
        self.extreme_overbought_threshold = 200
        self.zero_line_threshold = 5  # Buffer around zero line
        
        # Performance tracking
        self.signal_history = []
        self.divergence_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'reversal_accuracy': 0.0,
            'divergence_accuracy': 0.0
        }
        
        logger.info(f"CCI initialized: period={period}, constant={constant}, adaptive={adaptive}")
    
    def calculate_cci(self, 
                     high: Union[pd.Series, np.ndarray],
                     low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray],
                     timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Commodity Channel Index for given OHLC data
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps for session analysis
            
        Returns:
            Dictionary containing CCI calculations
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            
            if len(close_array) < self.period:
                logger.warning(f"Insufficient data: {len(close_array)} < {self.period}")
                return self._empty_result()
            
            # Calculate adaptive period if enabled
            current_period = self._calculate_adaptive_period(close_array) if self.adaptive else self.period
            
            # Calculate Typical Price (TP)
            typical_price = (high_array + low_array + close_array) / 3
            
            # Calculate Simple Moving Average of Typical Price
            sma_tp = self._calculate_sma(typical_price, current_period)
            
            # Calculate Mean Deviation
            mean_deviation = self._calculate_mean_deviation(typical_price, sma_tp, current_period)
            
            # Calculate CCI
            cci_values = (typical_price - sma_tp) / (self.constant * mean_deviation)
            
            # Handle division by zero
            cci_values = np.where(mean_deviation == 0, 0, cci_values)
            
            # Calculate additional metrics
            cci_zones = self._classify_cci_zones(cci_values)
            momentum_scores = self._calculate_momentum_scores(cci_values)
            divergence_signals = self._detect_divergences(close_array, cci_values)
            zero_line_crosses = self._detect_zero_line_crosses(cci_values)
            
            result = {
                'cci_values': cci_values,
                'typical_price': typical_price,
                'sma_tp': sma_tp,
                'mean_deviation': mean_deviation,
                'cci_zones': cci_zones,
                'momentum_scores': momentum_scores,
                'divergence_signals': divergence_signals,
                'zero_line_crosses': zero_line_crosses,
                'period_used': current_period,
                'constant_used': self.constant
            }
            
            logger.debug(f"CCI calculated: latest_cci={cci_values[-1]:.2f}, "
                        f"zone={cci_zones[-1]}, momentum={momentum_scores[-1]:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating CCI: {str(e)}")
            return self._empty_result()
    
    def generate_signals(self, 
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[CCISignal]:
        """
        Generate trading signals based on CCI analysis
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            timeframe: Current timeframe
            
        Returns:
            List of CCISignal objects
        """
        try:
            cci_data = self.calculate_cci(high, low, close, timestamps)
            if not cci_data or 'cci_values' not in cci_data:
                return []
            
            signals = []
            current_time = datetime.now()
            
            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_cci = cci_data['cci_values'][-1]
            latest_zone = cci_data['cci_zones'][-1]
            latest_momentum = cci_data['momentum_scores'][-1]
            latest_divergence = cci_data['divergence_signals'][-1]
            latest_zero_cross = cci_data['zero_line_crosses'][-1]
            
            # Determine current session
            session = self._get_current_session(current_time)
            
            # Generate signals based on CCI analysis
            signal_data = self._analyze_cci_signals(
                latest_cci, latest_zone, latest_momentum,
                latest_divergence, latest_zero_cross
            )
            
            if signal_data['signal_type'] != 'NONE':
                signal = CCISignal(
                    timestamp=current_time,
                    price=latest_price,
                    cci_value=latest_cci,
                    cci_zone=latest_zone,
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    momentum_score=latest_momentum,
                    divergence_strength=abs(latest_divergence),
                    timeframe=timeframe,
                    session=session
                )
                
                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()
                
                logger.info(f"CCI signal generated: {signal.signal_type} "
                           f"(cci={signal.cci_value:.2f}, zone={signal.cci_zone}, "
                           f"confidence={signal.confidence:.2f})")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating CCI signals: {str(e)}")
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
                adaptive_period = max(self.min_period, self.period - 4)
            elif volatility < 0.005:  # Low volatility
                adaptive_period = min(self.max_period, self.period + 6)
            else:
                adaptive_period = self.period
            
            return adaptive_period
            
        except Exception as e:
            logger.error(f"Error calculating adaptive period: {str(e)}")
            return self.period
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return pd.Series(data).rolling(window=period, min_periods=1).mean().values
    
    def _calculate_mean_deviation(self, typical_price: np.ndarray, sma_tp: np.ndarray, period: int) -> np.ndarray:
        """Calculate Mean Deviation for CCI"""
        try:
            mean_deviation = np.zeros_like(typical_price)
            
            for i in range(len(typical_price)):
                if i < period - 1:
                    # Use available data for initial values
                    start_idx = 0
                    end_idx = i + 1
                else:
                    # Use rolling window
                    start_idx = i - period + 1
                    end_idx = i + 1
                
                # Calculate mean deviation for the period
                deviations = np.abs(typical_price[start_idx:end_idx] - sma_tp[i])
                mean_deviation[i] = np.mean(deviations)
            
            # Avoid division by zero
            mean_deviation = np.where(mean_deviation == 0, 0.001, mean_deviation)
            
            return mean_deviation
            
        except Exception as e:
            logger.error(f"Error calculating mean deviation: {str(e)}")
            return np.ones_like(typical_price) * 0.001
    
    def _classify_cci_zones(self, cci_values: np.ndarray) -> List[str]:
        """Classify CCI values into zones"""
        try:
            zones = []
            
            for cci in cci_values:
                if cci < self.extreme_oversold_threshold:
                    zones.append(CCIZone.EXTREME_OVERSOLD.value)
                elif cci < self.oversold_threshold:
                    zones.append(CCIZone.OVERSOLD.value)
                elif cci < 0:
                    zones.append(CCIZone.NORMAL_BEARISH.value)
                elif cci < self.overbought_threshold:
                    zones.append(CCIZone.NORMAL_BULLISH.value)
                elif cci < self.extreme_overbought_threshold:
                    zones.append(CCIZone.OVERBOUGHT.value)
                else:
                    zones.append(CCIZone.EXTREME_OVERBOUGHT.value)
            
            return zones
            
        except Exception as e:
            logger.error(f"Error classifying CCI zones: {str(e)}")
            return [CCIZone.NORMAL_BULLISH.value] * len(cci_values)
    
    def _calculate_momentum_scores(self, cci_values: np.ndarray) -> np.ndarray:
        """Calculate momentum scores based on CCI changes"""
        try:
            momentum_scores = np.zeros_like(cci_values)
            
            for i in range(1, len(cci_values)):
                # Calculate CCI change
                cci_change = cci_values[i] - cci_values[i-1]
                
                # Calculate momentum over recent periods
                if i >= 5:
                    recent_changes = np.diff(cci_values[max(0, i-5):i+1])
                    momentum = np.mean(recent_changes)
                else:
                    momentum = cci_change
                
                # Normalize momentum score
                momentum_scores[i] = np.tanh(momentum / 50.0)  # Scale and bound between -1 and 1
            
            return momentum_scores
            
        except Exception as e:
            logger.error(f"Error calculating momentum scores: {str(e)}")
            return np.zeros_like(cci_values)
    
    def _detect_divergences(self, prices: np.ndarray, cci_values: np.ndarray) -> np.ndarray:
        """Detect divergences between price and CCI"""
        try:
            divergences = np.zeros_like(prices)
            
            # Need sufficient data for divergence analysis
            if len(prices) < 20:
                return divergences
            
            for i in range(10, len(prices)):
                # Look for recent highs and lows
                price_window = prices[max(0, i-10):i+1]
                cci_window = cci_values[max(0, i-10):i+1]
                
                # Find recent high and low points
                price_high_idx = np.argmax(price_window)
                price_low_idx = np.argmin(price_window)
                cci_high_idx = np.argmax(cci_window)
                cci_low_idx = np.argmin(cci_window)
                
                # Check for bullish divergence (price makes lower low, CCI makes higher low)
                if (price_low_idx == len(price_window) - 1 and  # Recent price low
                    cci_low_idx < len(cci_window) - 3 and  # CCI low was earlier
                    cci_window[-1] > cci_window[cci_low_idx]):  # CCI is higher now
                    divergences[i] = 1.0  # Bullish divergence
                
                # Check for bearish divergence (price makes higher high, CCI makes lower high)
                elif (price_high_idx == len(price_window) - 1 and  # Recent price high
                      cci_high_idx < len(cci_window) - 3 and  # CCI high was earlier
                      cci_window[-1] < cci_window[cci_high_idx]):  # CCI is lower now
                    divergences[i] = -1.0  # Bearish divergence
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error detecting divergences: {str(e)}")
            return np.zeros_like(prices)
    
    def _detect_zero_line_crosses(self, cci_values: np.ndarray) -> np.ndarray:
        """Detect zero line crosses"""
        try:
            crosses = np.zeros_like(cci_values)
            
            for i in range(1, len(cci_values)):
                prev_cci = cci_values[i-1]
                curr_cci = cci_values[i]
                
                # Bullish cross (from below zero to above zero)
                if prev_cci < -self.zero_line_threshold and curr_cci > self.zero_line_threshold:
                    crosses[i] = 1.0
                # Bearish cross (from above zero to below zero)
                elif prev_cci > self.zero_line_threshold and curr_cci < -self.zero_line_threshold:
                    crosses[i] = -1.0
            
            return crosses
            
        except Exception as e:
            logger.error(f"Error detecting zero line crosses: {str(e)}")
            return np.zeros_like(cci_values)
    
    def _analyze_cci_signals(self, cci_value: float, zone: str, momentum: float,
                            divergence: float, zero_cross: float) -> Dict:
        """Analyze current CCI conditions and generate signal"""
        try:
            signal_type = 'NONE'
            strength = 0.0
            confidence = 0.0
            
            # Zero line cross signals (high priority)
            if zero_cross > 0.5:
                signal_type = CCISignalType.ZERO_LINE_CROSS_BULLISH.value
                strength = min(1.0, abs(cci_value) / 50.0)
                confidence = min(0.85, 0.7 + abs(momentum) * 0.15)
            elif zero_cross < -0.5:
                signal_type = CCISignalType.ZERO_LINE_CROSS_BEARISH.value
                strength = min(1.0, abs(cci_value) / 50.0)
                confidence = min(0.85, 0.7 + abs(momentum) * 0.15)
            
            # Divergence signals
            elif divergence > 0.5:
                signal_type = CCISignalType.DIVERGENCE_BULLISH.value
                strength = 0.8
                confidence = min(0.9, 0.75 + abs(momentum) * 0.15)
            elif divergence < -0.5:
                signal_type = CCISignalType.DIVERGENCE_BEARISH.value
                strength = 0.8
                confidence = min(0.9, 0.75 + abs(momentum) * 0.15)
            
            # Overbought/Oversold reversal signals
            elif zone == CCIZone.EXTREME_OVERSOLD.value and momentum > 0.1:
                signal_type = CCISignalType.OVERSOLD_REVERSAL.value
                strength = min(1.0, abs(cci_value) / 300.0)
                confidence = min(0.85, 0.6 + strength * 0.25)
            elif zone == CCIZone.EXTREME_OVERBOUGHT.value and momentum < -0.1:
                signal_type = CCISignalType.OVERBOUGHT_REVERSAL.value
                strength = min(1.0, abs(cci_value) / 300.0)
                confidence = min(0.85, 0.6 + strength * 0.25)
            
            # Momentum signals
            elif abs(momentum) > 0.3:
                if momentum > 0:
                    signal_type = CCISignalType.MOMENTUM_ACCELERATION.value
                else:
                    signal_type = CCISignalType.MOMENTUM_DECELERATION.value
                strength = min(1.0, abs(momentum))
                confidence = min(0.75, 0.5 + abs(momentum) * 0.25)
            
            # Extreme reading signals
            elif abs(cci_value) > 250:
                signal_type = CCISignalType.EXTREME_READING.value
                strength = min(1.0, abs(cci_value) / 400.0)
                confidence = min(0.8, 0.5 + strength * 0.3)
            
            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing CCI signals: {str(e)}")
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
                reversal_signals = [s for s in self.signal_history if 'reversal' in s.signal_type.lower()]
                divergence_signals = [s for s in self.signal_history if 'divergence' in s.signal_type.lower()]
                
                if high_confidence_signals:
                    self.performance_stats['successful_signals'] = len(high_confidence_signals)
                    self.performance_stats['accuracy'] = len(high_confidence_signals) / len(self.signal_history)
                
                if reversal_signals:
                    successful_reversals = [s for s in reversal_signals if s.confidence > 0.8]
                    self.performance_stats['reversal_accuracy'] = len(successful_reversals) / len(reversal_signals)
                
                if divergence_signals:
                    successful_divergences = [s for s in divergence_signals if s.confidence > 0.8]
                    self.performance_stats['divergence_accuracy'] = len(successful_divergences) / len(divergence_signals)
                    
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'cci_values': np.array([]),
            'typical_price': np.array([]),
            'sma_tp': np.array([]),
            'mean_deviation': np.array([]),
            'cci_zones': [],
            'momentum_scores': np.array([]),
            'divergence_signals': np.array([]),
            'zero_line_crosses': np.array([]),
            'period_used': self.period,
            'constant_used': self.constant
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
            'reversal_accuracy': 0.0,
            'divergence_accuracy': 0.0
        }
        logger.info("CCI performance stats reset")

# Example usage and testing
if __name__ == "__main__":
    # Create sample OHLC data with overbought/oversold conditions
    np.random.seed(42)
    n_points = 100
    base_price = 100
    
    # Generate realistic OHLC data with momentum changes
    close_prices = np.zeros(n_points)
    for i in range(n_points):
        if i < 25:
            close_prices[i] = base_price + i * 0.08 + np.random.randn() * 0.02  # Strong uptrend
        elif i < 50:
            close_prices[i] = close_prices[24] + np.random.randn() * 0.01  # Consolidation
        elif i < 75:
            close_prices[i] = close_prices[49] - (i-50) * 0.06 + np.random.randn() * 0.02  # Downtrend
        else:
            close_prices[i] = close_prices[74] + (i-75) * 0.04 + np.random.randn() * 0.01  # Recovery
    
    high_prices = close_prices + np.abs(np.random.randn(n_points) * 0.01)
    low_prices = close_prices - np.abs(np.random.randn(n_points) * 0.01)
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
    
    # Initialize CCI
    cci = CCI(period=20, constant=0.015, adaptive=True)
    
    # Calculate CCI
    result = cci.calculate_cci(high_prices, low_prices, close_prices)
    print("CCI calculation completed")
    print(f"Latest CCI: {result['cci_values'][-1]:.2f}")
    print(f"CCI Zone: {result['cci_zones'][-1]}")
    print(f"Momentum score: {result['momentum_scores'][-1]:.2f}")
    
    # Generate signals
    signals = cci.generate_signals(high_prices, low_prices, close_prices, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")
    
    # Display performance stats
    stats = cci.get_performance_stats()
    print(f"Performance stats: {stats}")
