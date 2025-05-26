"""
Bollinger Bands Volatility Indicator
Advanced implementation with dynamic periods and multiple band configurations
Optimized for M1-H4 timeframes and short-term trading strategies
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

class BandType(Enum):
    """Bollinger Band types"""
    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    SQUEEZE = "squeeze"
    EXPANSION = "expansion"

@dataclass
class BollingerBandSignal:
    """Bollinger Band signal data structure"""
    timestamp: datetime
    price: float
    upper_band: float
    middle_band: float
    lower_band: float
    band_width: float
    percent_b: float
    squeeze_level: float
    signal_type: str
    signal_strength: float
    confidence: float
    timeframe: str
    session: str

class BollingerBands:
    """
    Advanced Bollinger Bands implementation for forex trading
    Features:
    - Dynamic period adjustment based on volatility
    - Multiple standard deviation levels
    - Band squeeze and expansion detection
    - Percent B calculations
    - Session-aware analysis
    - Real-time signal generation
    """
    
    def __init__(self, 
                 period: int = 20,
                 std_dev: float = 2.0,
                 adaptive: bool = True,
                 min_period: int = 10,
                 max_period: int = 50,
                 timeframes: List[str] = None):
        """
        Initialize Bollinger Bands calculator
        
        Args:
            period: Base period for moving average
            std_dev: Standard deviation multiplier
            adaptive: Enable adaptive period adjustment
            min_period: Minimum period for adaptive mode
            max_period: Maximum period for adaptive mode
            timeframes: List of timeframes to analyze
        """
        self.period = period
        self.std_dev = std_dev
        self.adaptive = adaptive
        self.min_period = min_period
        self.max_period = max_period
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']
        
        # Signal thresholds
        self.squeeze_threshold = 0.1  # Band width threshold for squeeze
        self.expansion_threshold = 0.3  # Band width threshold for expansion
        self.oversold_threshold = 0.2  # Percent B oversold level
        self.overbought_threshold = 0.8  # Percent B overbought level
        
        # Performance tracking
        self.signal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0
        }
        
        logger.info(f"BollingerBands initialized: period={period}, std_dev={std_dev}, adaptive={adaptive}")
    
    def calculate_bands(self, 
                       prices: Union[pd.Series, np.ndarray],
                       timestamps: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Bollinger Bands for given price data
        
        Args:
            prices: Price data (typically close prices)
            timestamps: Optional timestamps for session analysis
            
        Returns:
            Dictionary containing band calculations
        """
        try:
            if isinstance(prices, pd.Series):
                prices_array = prices.values
            else:
                prices_array = np.array(prices)
            
            if len(prices_array) < self.period:
                logger.warning(f"Insufficient data: {len(prices_array)} < {self.period}")
                return self._empty_result()
            
            # Calculate adaptive period if enabled
            current_period = self._calculate_adaptive_period(prices_array) if self.adaptive else self.period
            
            # Calculate moving average (middle band)
            middle_band = self._calculate_sma(prices_array, current_period)
            
            # Calculate standard deviation
            std_values = self._calculate_rolling_std(prices_array, current_period)
            
            # Calculate upper and lower bands
            upper_band = middle_band + (self.std_dev * std_values)
            lower_band = middle_band - (self.std_dev * std_values)
            
            # Calculate additional metrics
            band_width = (upper_band - lower_band) / middle_band
            percent_b = (prices_array - lower_band) / (upper_band - lower_band)
            
            # Detect squeeze and expansion
            squeeze_level = self._detect_squeeze(band_width)
            expansion_level = self._detect_expansion(band_width)
            
            result = {
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'band_width': band_width,
                'percent_b': percent_b,
                'squeeze_level': squeeze_level,
                'expansion_level': expansion_level,
                'period_used': current_period,
                'std_dev_used': self.std_dev
            }
            
            logger.debug(f"Bollinger Bands calculated: period={current_period}, latest_width={band_width[-1]:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return self._empty_result()
    
    def generate_signals(self, 
                        prices: Union[pd.Series, np.ndarray],
                        timestamps: Optional[pd.Series] = None,
                        timeframe: str = 'M15') -> List[BollingerBandSignal]:
        """
        Generate trading signals based on Bollinger Bands
        
        Args:
            prices: Price data
            timestamps: Optional timestamps
            timeframe: Current timeframe
            
        Returns:
            List of BollingerBandSignal objects
        """
        try:
            bands_data = self.calculate_bands(prices, timestamps)
            if not bands_data or 'upper_band' not in bands_data:
                return []
            
            signals = []
            current_time = datetime.now()
            
            # Get latest values
            latest_price = prices.iloc[-1] if isinstance(prices, pd.Series) else prices[-1]
            latest_upper = bands_data['upper_band'][-1]
            latest_middle = bands_data['middle_band'][-1]
            latest_lower = bands_data['lower_band'][-1]
            latest_width = bands_data['band_width'][-1]
            latest_percent_b = bands_data['percent_b'][-1]
            latest_squeeze = bands_data['squeeze_level'][-1]
            
            # Determine current session
            session = self._get_current_session(current_time)
            
            # Generate signals based on different conditions
            signal_data = self._analyze_band_signals(
                latest_price, latest_upper, latest_middle, latest_lower,
                latest_width, latest_percent_b, latest_squeeze
            )
            
            if signal_data['signal_type'] != 'NONE':
                signal = BollingerBandSignal(
                    timestamp=current_time,
                    price=latest_price,
                    upper_band=latest_upper,
                    middle_band=latest_middle,
                    lower_band=latest_lower,
                    band_width=latest_width,
                    percent_b=latest_percent_b,
                    squeeze_level=latest_squeeze,
                    signal_type=signal_data['signal_type'],
                    signal_strength=signal_data['strength'],
                    confidence=signal_data['confidence'],
                    timeframe=timeframe,
                    session=session
                )
                
                signals.append(signal)
                self.signal_history.append(signal)
                self._update_performance_stats()
                
                logger.info(f"Bollinger Band signal generated: {signal.signal_type} "
                           f"(strength={signal.signal_strength:.2f}, confidence={signal.confidence:.2f})")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Band signals: {str(e)}")
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
                adaptive_period = max(self.min_period, self.period - 5)
            elif volatility < 0.005:  # Low volatility
                adaptive_period = min(self.max_period, self.period + 10)
            else:
                adaptive_period = self.period
            
            return adaptive_period
            
        except Exception as e:
            logger.error(f"Error calculating adaptive period: {str(e)}")
            return self.period
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(window=period, min_periods=1).mean().values
    
    def _calculate_rolling_std(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation"""
        return pd.Series(prices).rolling(window=period, min_periods=1).std().values
    
    def _detect_squeeze(self, band_width: np.ndarray) -> np.ndarray:
        """Detect Bollinger Band squeeze conditions"""
        try:
            # Calculate squeeze level (0-1 scale)
            avg_width = np.mean(band_width[-20:]) if len(band_width) >= 20 else np.mean(band_width)
            squeeze_level = np.where(band_width < avg_width * self.squeeze_threshold, 1.0, 0.0)
            return squeeze_level
        except Exception:
            return np.zeros_like(band_width)
    
    def _detect_expansion(self, band_width: np.ndarray) -> np.ndarray:
        """Detect Bollinger Band expansion conditions"""
        try:
            # Calculate expansion level (0-1 scale)
            avg_width = np.mean(band_width[-20:]) if len(band_width) >= 20 else np.mean(band_width)
            expansion_level = np.where(band_width > avg_width * self.expansion_threshold, 1.0, 0.0)
            return expansion_level
        except Exception:
            return np.zeros_like(band_width)
    
    def _analyze_band_signals(self, price: float, upper: float, middle: float, 
                             lower: float, width: float, percent_b: float, 
                             squeeze: float) -> Dict:
        """Analyze current conditions and generate signal"""
        try:
            signal_type = 'NONE'
            strength = 0.0
            confidence = 0.0
            
            # Band bounce signals
            if percent_b <= self.oversold_threshold:
                signal_type = 'BUY_BOUNCE'
                strength = (self.oversold_threshold - percent_b) / self.oversold_threshold
                confidence = min(0.9, 0.5 + strength * 0.4)
            elif percent_b >= self.overbought_threshold:
                signal_type = 'SELL_BOUNCE'
                strength = (percent_b - self.overbought_threshold) / (1.0 - self.overbought_threshold)
                confidence = min(0.9, 0.5 + strength * 0.4)
            
            # Squeeze breakout signals
            elif squeeze > 0.5 and abs(price - middle) > width * 0.1:
                if price > middle:
                    signal_type = 'BUY_BREAKOUT'
                else:
                    signal_type = 'SELL_BREAKOUT'
                strength = min(1.0, abs(price - middle) / (width * 0.5))
                confidence = min(0.85, 0.6 + squeeze * 0.25)
            
            # Middle band signals
            elif abs(price - middle) < width * 0.05:
                signal_type = 'MIDDLE_REVERSION'
                strength = 0.3
                confidence = 0.4
            
            return {
                'signal_type': signal_type,
                'strength': strength,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing band signals: {str(e)}")
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
                
                # Simple accuracy estimation based on confidence
                high_confidence_signals = [s for s in self.signal_history if s.confidence > 0.7]
                if high_confidence_signals:
                    self.performance_stats['successful_signals'] = len(high_confidence_signals)
                    self.performance_stats['accuracy'] = len(high_confidence_signals) / len(self.signal_history)
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'upper_band': np.array([]),
            'middle_band': np.array([]),
            'lower_band': np.array([]),
            'band_width': np.array([]),
            'percent_b': np.array([]),
            'squeeze_level': np.array([]),
            'expansion_level': np.array([]),
            'period_used': self.period,
            'std_dev_used': self.std_dev
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
            'avg_confidence': 0.0
        }
        logger.info("Bollinger Bands performance stats reset")

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
    timestamps = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    
    # Initialize Bollinger Bands
    bb = BollingerBands(period=20, std_dev=2.0, adaptive=True)
    
    # Calculate bands
    result = bb.calculate_bands(prices)
    print("Bollinger Bands calculation completed")
    print(f"Latest band width: {result['band_width'][-1]:.4f}")
    
    # Generate signals
    signals = bb.generate_signals(prices, timestamps, 'M1')
    print(f"Generated {len(signals)} signals")
    
    # Display performance stats
    stats = bb.get_performance_stats()
    print(f"Performance stats: {stats}")
