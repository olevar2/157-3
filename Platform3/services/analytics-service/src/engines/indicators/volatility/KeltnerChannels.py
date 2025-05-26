"""
Keltner Channels Volatility Indicator
Advanced implementation with multiple MA types and ATR-based channel calculation
Optimized for M1-H4 timeframes and breakout detection
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

class MAType(Enum):
    """Moving Average types for Keltner Channels"""
    SMA = "sma"
    EMA = "ema"
    WMA = "wma"
    ADAPTIVE = "adaptive"

class ChannelPosition(Enum):
    """Price position relative to Keltner Channels"""
    BELOW_LOWER = "below_lower"
    LOWER_BAND = "lower_band"
    MIDDLE_ZONE = "middle_zone"
    UPPER_BAND = "upper_band"
    ABOVE_UPPER = "above_upper"

@dataclass
class KeltnerSignal:
    """Keltner Channels signal data structure"""
    timestamp: datetime
    price: float
    upper_channel: float
    middle_line: float
    lower_channel: float
    channel_width: float
    price_position: str
    channel_squeeze: float
    breakout_strength: float
    signal_type: str
    signal_strength: float
    confidence: float
    timeframe: str
    session: str

class KeltnerChannels:
    """
    Advanced Keltner Channels implementation for forex trading
    Features:
    - Multiple moving average types for middle line
    - ATR-based channel calculation
    - Channel squeeze and expansion detection
    - Breakout signal generation
    - Price position analysis
    - Session-aware analysis
    """
    
    def __init__(self, 
                 period: int = 20,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 ma_type: MAType = MAType.EMA,
                 adaptive: bool = True,
                 timeframes: List[str] = None):
        """
        Initialize Keltner Channels calculator
        
        Args:
            period: Period for middle line moving average
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR to create channels
            ma_type: Type of moving average for middle line
            adaptive: Enable adaptive parameter adjustment
            timeframes: List of timeframes to analyze
        """
        self.period = period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.ma_type = ma_type
        self.adaptive = adaptive
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']
        
        # Signal thresholds
        self.squeeze_threshold = 0.8  # Channel width relative to average
        self.expansion_threshold = 1.3  # Channel width relative to average
        self.breakout_threshold = 0.1  # Price distance from channel for breakout
        
        # Performance tracking
        self.signal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'breakout_accuracy': 0.0
        }
        
        logger.info(f"KeltnerChannels initialized: period={period}, atr_period={atr_period}, "
                   f"multiplier={atr_multiplier}, ma_type={ma_type.value}")
    
    def calculate_channels(self, 
                          high: Union[pd.Series, np.ndarray],
                          low: Union[pd.Series, np.ndarray],
                          close: Union[pd.Series, np.ndarray],
                          timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Keltner Channels for given OHLC data
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps for session analysis
            
        Returns:
            Dictionary containing channel calculations
        """
        try:
            # Convert to numpy arrays
            high_array = np.array(high)
            low_array = np.array(low)
            close_array = np.array(close)
            
            if len(close_array) < max(self.period, self.atr_period):
                logger.warning(f"Insufficient data: {len(close_array)} < {max(self.period, self.atr_period)}")
                return self._empty_result()
            
            # Calculate middle line (moving average of close prices)
            middle_line = self._calculate_moving_average(close_array)
            
            # Calculate ATR for channel width
            atr_values = self._calculate_atr(high_array, low_array, close_array)
            
            # Calculate upper and lower channels
            upper_channel = middle_line + (self.atr_multiplier * atr_values)
            lower_channel = middle_line - (self.atr_multiplier * atr_values)
            
            # Calculate channel metrics
            channel_width = (upper_channel - lower_channel) / middle_line
            price_positions = self._determine_price_positions(close_array, upper_channel, 
                                                            middle_line, lower_channel)
            
            # Detect squeeze and expansion
            squeeze_levels = self._detect_channel_squeeze(channel_width)
            expansion_levels = self._detect_channel_expansion(channel_width)
            
            # Calculate breakout strength
            breakout_strength = self._calculate_breakout_strength(close_array, upper_channel, 
                                                                lower_channel, atr_values)
            
            result = {
                'upper_channel': upper_channel,
                'middle_line': middle_line,
                'lower_channel': lower_channel,
                'atr_values': atr_values,
                'channel_width': channel_width,
                'price_positions': price_positions,
                'squeeze_levels': squeeze_levels,
                'expansion_levels': expansion_levels,
                'breakout_strength': breakout_strength,
                'period_used': self.period,
                'atr_period_used': self.atr_period,
                'atr_multiplier_used': self.atr_multiplier
            }
            
            logger.debug(f"Keltner Channels calculated: latest_width={channel_width[-1]:.4f}, "
                        f"position={price_positions[-1]}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Keltner Channels: {str(e)}")
            return self._empty_result()
    
    def generate_signals(self, 
                        high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[KeltnerSignal]:
        """
        Generate trading signals based on Keltner Channels
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            timeframe: Current timeframe
            
        Returns:
            List of KeltnerSignal objects
        """
        try:
            channels_data = self.calculate_channels(high, low, close, timestamps)
            if not channels_data or 'upper_channel' not in channels_data:
                return []
            
            signals = []
            current_time = datetime.now()
            
            # Get latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            latest_upper = channels_data['upper_channel'][-1]
            latest_middle = channels_data['middle_line'][-1]
            latest_lower = channels_data['lower_channel'][-1]
            latest_width = channels_data['channel_width'][-1]
            latest_position = channels_data['price_positions'][-1]
            latest_squeeze = channels_data['squeeze_levels'][-1]
            latest_breakout = channels_data['breakout_strength'][-1]
            
            # Determine current session
            session = self._get_current_session(current_time)
            
            # Generate signals based on channel analysis
            signal_data = self._analyze_channel_signals(
                latest_price, latest_upper, latest_middle, latest_lower,
                latest_width, latest_position, latest_squeeze, latest_breakout
            )
            
            if signal_data['signal_type'] != 'NONE':
                signal = KeltnerSignal(
                    timestamp=current_time,
                    price=latest_price,
                    upper_channel=latest_upper,
                    middle_line=latest_middle,
                    lower_channel=latest_lower,
                    channel_width=latest_width,
                    price_position=latest_position,
                    channel_squeeze=latest_squeeze,
                    breakout_strength=latest_breakout,
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    timeframe=timeframe,
                    session=session
                )
                
                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()
                
                logger.info(f"Keltner Channel signal generated: {signal.signal_type} "
                           f"(position={signal.price_position}, confidence={signal.confidence:.2f})")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Keltner Channel signals: {str(e)}")
            return []
    
    def _calculate_moving_average(self, prices: np.ndarray) -> np.ndarray:
        """Calculate moving average for middle line"""
        try:
            if self.ma_type == MAType.SMA:
                return pd.Series(prices).rolling(window=self.period, min_periods=1).mean().values
            elif self.ma_type == MAType.EMA:
                alpha = 2.0 / (self.period + 1)
                return pd.Series(prices).ewm(alpha=alpha, adjust=False).mean().values
            elif self.ma_type == MAType.WMA:
                return self._calculate_wma(prices)
            elif self.ma_type == MAType.ADAPTIVE:
                return self._calculate_adaptive_ma(prices)
            else:
                return pd.Series(prices).ewm(span=self.period, adjust=False).mean().values
                
        except Exception as e:
            logger.error(f"Error calculating moving average: {str(e)}")
            return np.zeros_like(prices)
    
    def _calculate_wma(self, prices: np.ndarray) -> np.ndarray:
        """Calculate Weighted Moving Average"""
        wma = np.zeros_like(prices)
        weights = np.arange(1, self.period + 1)
        weight_sum = np.sum(weights)
        
        for i in range(len(prices)):
            if i < self.period - 1:
                # Use available data for initial values
                current_weights = weights[:i+1]
                current_weight_sum = np.sum(current_weights)
                wma[i] = np.sum(prices[:i+1] * current_weights) / current_weight_sum
            else:
                wma[i] = np.sum(prices[i-self.period+1:i+1] * weights) / weight_sum
        
        return wma
    
    def _calculate_adaptive_ma(self, prices: np.ndarray) -> np.ndarray:
        """Calculate Adaptive Moving Average based on volatility"""
        ama = np.zeros_like(prices)
        ama[0] = prices[0]
        
        for i in range(1, len(prices)):
            if i < 10:
                # Use EMA for initial values
                alpha = 2.0 / (self.period + 1)
            else:
                # Calculate efficiency ratio
                price_change = abs(prices[i] - prices[max(0, i-10)])
                volatility = np.sum(np.abs(np.diff(prices[max(0, i-10):i+1])))
                
                if volatility > 0:
                    efficiency_ratio = price_change / volatility
                else:
                    efficiency_ratio = 0.5
                
                # Adaptive alpha based on efficiency ratio
                fast_alpha = 2.0 / 3.0  # Fast period = 2
                slow_alpha = 2.0 / 31.0  # Slow period = 30
                alpha = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha) ** 2
            
            ama[i] = ama[i-1] + alpha * (prices[i] - ama[i-1])
        
        return ama
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Average True Range"""
        try:
            # Calculate True Range
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate ATR using Wilder's smoothing
            atr = np.zeros_like(true_range)
            atr[0] = true_range[0]
            
            for i in range(1, len(true_range)):
                if i < self.atr_period:
                    atr[i] = np.mean(true_range[:i+1])
                else:
                    atr[i] = (atr[i-1] * (self.atr_period - 1) + true_range[i]) / self.atr_period
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return np.ones_like(high) * 0.001
    
    def _determine_price_positions(self, prices: np.ndarray, upper: np.ndarray, 
                                  middle: np.ndarray, lower: np.ndarray) -> List[str]:
        """Determine price position relative to channels"""
        try:
            positions = []
            
            for i, price in enumerate(prices):
                if i >= len(upper):
                    positions.append(ChannelPosition.MIDDLE_ZONE.value)
                    continue
                
                upper_val = upper[i]
                middle_val = middle[i]
                lower_val = lower[i]
                
                if price > upper_val * 1.001:  # Small buffer for noise
                    positions.append(ChannelPosition.ABOVE_UPPER.value)
                elif price > upper_val * 0.999:
                    positions.append(ChannelPosition.UPPER_BAND.value)
                elif price < lower_val * 0.999:
                    positions.append(ChannelPosition.BELOW_LOWER.value)
                elif price < lower_val * 1.001:
                    positions.append(ChannelPosition.LOWER_BAND.value)
                else:
                    positions.append(ChannelPosition.MIDDLE_ZONE.value)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error determining price positions: {str(e)}")
            return [ChannelPosition.MIDDLE_ZONE.value] * len(prices)
    
    def _detect_channel_squeeze(self, channel_width: np.ndarray) -> np.ndarray:
        """Detect channel squeeze conditions"""
        try:
            if len(channel_width) < 20:
                return np.zeros_like(channel_width)
            
            # Calculate average channel width over recent periods
            avg_width = np.zeros_like(channel_width)
            for i in range(len(channel_width)):
                lookback = min(20, i + 1)
                avg_width[i] = np.mean(channel_width[max(0, i-lookback+1):i+1])
            
            # Squeeze level (0-1 scale)
            squeeze_levels = np.where(channel_width < avg_width * self.squeeze_threshold, 
                                    1.0 - (channel_width / (avg_width * self.squeeze_threshold)), 0.0)
            
            return np.clip(squeeze_levels, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error detecting channel squeeze: {str(e)}")
            return np.zeros_like(channel_width)
    
    def _detect_channel_expansion(self, channel_width: np.ndarray) -> np.ndarray:
        """Detect channel expansion conditions"""
        try:
            if len(channel_width) < 20:
                return np.zeros_like(channel_width)
            
            # Calculate average channel width over recent periods
            avg_width = np.zeros_like(channel_width)
            for i in range(len(channel_width)):
                lookback = min(20, i + 1)
                avg_width[i] = np.mean(channel_width[max(0, i-lookback+1):i+1])
            
            # Expansion level (0-1 scale)
            expansion_levels = np.where(channel_width > avg_width * self.expansion_threshold,
                                      (channel_width / (avg_width * self.expansion_threshold)) - 1.0, 0.0)
            
            return np.clip(expansion_levels, 0.0, 2.0)
            
        except Exception as e:
            logger.error(f"Error detecting channel expansion: {str(e)}")
            return np.zeros_like(channel_width)
    
    def _calculate_breakout_strength(self, prices: np.ndarray, upper: np.ndarray, 
                                   lower: np.ndarray, atr: np.ndarray) -> np.ndarray:
        """Calculate breakout strength"""
        try:
            breakout_strength = np.zeros_like(prices)
            
            for i in range(len(prices)):
                if i >= len(upper) or atr[i] == 0:
                    continue
                
                price = prices[i]
                upper_val = upper[i]
                lower_val = lower[i]
                atr_val = atr[i]
                
                # Calculate distance from channels relative to ATR
                if price > upper_val:
                    strength = (price - upper_val) / atr_val
                elif price < lower_val:
                    strength = (lower_val - price) / atr_val
                else:
                    strength = 0.0
                
                breakout_strength[i] = min(3.0, strength)  # Cap at 3.0
            
            return breakout_strength
            
        except Exception as e:
            logger.error(f"Error calculating breakout strength: {str(e)}")
            return np.zeros_like(prices)
    
    def _analyze_channel_signals(self, price: float, upper: float, middle: float, 
                               lower: float, width: float, position: str, 
                               squeeze: float, breakout: float) -> Dict:
        """Analyze current channel conditions and generate signal"""
        try:
            signal_type = 'NONE'
            strength = 0.0
            confidence = 0.0
            
            # Breakout signals
            if position == ChannelPosition.ABOVE_UPPER.value and breakout > self.breakout_threshold:
                signal_type = 'BULLISH_BREAKOUT'
                strength = min(1.0, breakout / 2.0)
                confidence = min(0.9, 0.6 + strength * 0.3)
            elif position == ChannelPosition.BELOW_LOWER.value and breakout > self.breakout_threshold:
                signal_type = 'BEARISH_BREAKOUT'
                strength = min(1.0, breakout / 2.0)
                confidence = min(0.9, 0.6 + strength * 0.3)
            
            # Channel bounce signals
            elif position == ChannelPosition.UPPER_BAND.value:
                signal_type = 'RESISTANCE_TEST'
                strength = 0.6
                confidence = 0.7
            elif position == ChannelPosition.LOWER_BAND.value:
                signal_type = 'SUPPORT_TEST'
                strength = 0.6
                confidence = 0.7
            
            # Squeeze signals
            elif squeeze > 0.7:
                signal_type = 'CHANNEL_SQUEEZE'
                strength = squeeze
                confidence = min(0.8, 0.5 + squeeze * 0.3)
            
            # Middle line signals
            elif position == ChannelPosition.MIDDLE_ZONE.value:
                if price > middle:
                    signal_type = 'ABOVE_MIDDLE'
                else:
                    signal_type = 'BELOW_MIDDLE'
                strength = 0.4
                confidence = 0.5
            
            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing channel signals: {str(e)}")
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
                breakout_signals = [s for s in self.signal_history if 'BREAKOUT' in s.signal_type]
                
                if high_confidence_signals:
                    self.performance_stats['successful_signals'] = len(high_confidence_signals)
                    self.performance_stats['accuracy'] = len(high_confidence_signals) / len(self.signal_history)
                
                if breakout_signals:
                    successful_breakouts = [s for s in breakout_signals if s.confidence > 0.8]
                    self.performance_stats['breakout_accuracy'] = len(successful_breakouts) / len(breakout_signals)
                    
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'upper_channel': np.array([]),
            'middle_line': np.array([]),
            'lower_channel': np.array([]),
            'atr_values': np.array([]),
            'channel_width': np.array([]),
            'price_positions': [],
            'squeeze_levels': np.array([]),
            'expansion_levels': np.array([]),
            'breakout_strength': np.array([]),
            'period_used': self.period,
            'atr_period_used': self.atr_period,
            'atr_multiplier_used': self.atr_multiplier
        }
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.signal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'breakout_accuracy': 0.0
        }
        logger.info("Keltner Channels performance stats reset")

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
    
    # Initialize Keltner Channels
    kc = KeltnerChannels(period=20, atr_period=14, atr_multiplier=2.0, ma_type=MAType.EMA)
    
    # Calculate channels
    result = kc.calculate_channels(high_prices, low_prices, close_prices)
    print("Keltner Channels calculation completed")
    print(f"Latest channel width: {result['channel_width'][-1]:.4f}")
    print(f"Price position: {result['price_positions'][-1]}")
    
    # Generate signals
    signals = kc.generate_signals(high_prices, low_prices, close_prices, timestamps, 'M15')
    print(f"Generated {len(signals)} signals")
    
    # Display performance stats
    stats = kc.get_performance_stats()
    print(f"Performance stats: {stats}")
