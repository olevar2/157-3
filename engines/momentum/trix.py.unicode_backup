"""
TRIX - Triple Exponential Average Oscillator
============================================

A sophisticated implementation of the TRIX indicator that uses a triple-smoothed
exponential moving average to filter out short-term price noise and identify
longer-term momentum changes. TRIX is particularly effective at identifying
trend reversals and momentum shifts while minimizing false signals.

Key Features:
- Triple exponential smoothing for noise reduction
- Zero-line crossover signals
- Signal line crossovers
- Momentum divergence detection
- Trend change identification
- Rate of change analysis
- Multi-timeframe coordination
- Adaptive signal filtering

Humanitarian Use:
Provides reliable trend reversal signals with minimal noise for maximum profit
generation while maintaining ethical trading practices for humanitarian funding.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

from ..indicator_base import IndicatorBase, IndicatorConfig, IndicatorSignal, SignalStrength, MarketCondition

logger = logging.getLogger(__name__)

class TRIXTrendState(Enum):
    """TRIX trend states"""
    BULLISH_MOMENTUM = "bullish_momentum"
    BEARISH_MOMENTUM = "bearish_momentum"
    MOMENTUM_ACCELERATING = "momentum_accelerating"
    MOMENTUM_DECELERATING = "momentum_decelerating"
    TREND_REVERSAL = "trend_reversal"
    NEUTRAL = "neutral"

class TRIXSignalType(Enum):
    """TRIX signal types"""
    ZERO_LINE_CROSS = "zero_line_cross"
    SIGNAL_LINE_CROSS = "signal_line_cross"
    MOMENTUM_DIVERGENCE = "momentum_divergence"
    TREND_CHANGE = "trend_change"
    ACCELERATION_CHANGE = "acceleration_change"

@dataclass
class TRIXConfig(IndicatorConfig):
    """Configuration for TRIX indicator"""
    # Core TRIX settings
    period: int = 14
    signal_period: int = 9
    price_type: str = "close"  # close, hl2, hlc3, ohlc4
    
    # Signal detection settings
    zero_line_threshold: float = 0.0001  # Minimum threshold for zero-line signals
    divergence_lookback: int = 20
    trend_confirmation_bars: int = 3
    
    # Smoothing and filtering
    additional_smoothing: bool = False
    noise_filter_enabled: bool = True
    adaptive_periods: bool = False
    
    # Advanced features
    momentum_analysis: bool = True
    rate_of_change_analysis: bool = True
    multi_level_signals: bool = True
    histogram_analysis: bool = True

@dataclass
class TRIXMomentum:
    """TRIX momentum analysis"""
    momentum_direction: str  # "increasing", "decreasing", "neutral"
    momentum_strength: float  # 0.0 to 1.0
    acceleration: float  # Rate of momentum change
    velocity: float  # Speed of TRIX movement
    persistence: float  # How long momentum has been in current direction

@dataclass
class TRIXAnalysis:
    """Comprehensive TRIX analysis results"""
    trend_state: TRIXTrendState
    trix_value: float
    trix_signal: float
    trix_histogram: float
    momentum: TRIXMomentum
    zero_line_distance: float
    signal_line_distance: float
    divergence_strength: float
    trend_strength: float
    noise_level: float

class TRIX(IndicatorBase[TRIXConfig]):
    """
    Advanced TRIX (Triple Exponential Average) implementation
    
    TRIX calculation:
    1. First EMA: EMA(price, period)
    2. Second EMA: EMA(first_EMA, period)
    3. Third EMA: EMA(second_EMA, period)
    4. TRIX: Rate of change of third EMA
    5. Signal: EMA(TRIX, signal_period)
    
    The triple smoothing eliminates short-term fluctuations while
    preserving longer-term trend changes.
    """
    
    def __init__(self, config: TRIXConfig):
        super().__init__(config)
        
        # EMA calculations
        self.first_ema: List[float] = []
        self.second_ema: List[float] = []
        self.third_ema: List[float] = []
        
        # TRIX values
        self.trix_values: List[float] = []
        self.trix_signal: List[float] = []
        self.trix_histogram: List[float] = []
        
        # Analysis data
        self.momentum_values: List[float] = []
        self.acceleration_values: List[float] = []
        self.velocity_values: List[float] = []
        
        # Pattern tracking
        self.zero_crossings: List[Tuple[int, str]] = []
        self.signal_crossings: List[Tuple[int, str]] = []
        self.divergence_points: List[Tuple[int, str, float]] = []
        self.trend_changes: List[Tuple[int, str]] = []
        
        # State tracking
        self.momentum_direction_count: int = 0
        self.current_trend: str = "neutral"
        self.noise_level: float = 0.0
        
    def _get_price_value(self, data: Dict[str, float]) -> float:
        """Extract price value based on configured price type"""
        if self.config.price_type == "close":
            return data['close']
        elif self.config.price_type == "hl2":
            return (data['high'] + data['low']) / 2
        elif self.config.price_type == "hlc3":
            return (data['high'] + data['low'] + data['close']) / 3
        elif self.config.price_type == "ohlc4":
            return (data['open'] + data['high'] + data['low'] + data['close']) / 4
        else:
            return data['close']
    
    def _calculate_ema(self, values: List[float], new_value: float, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if not values:
            return new_value
        
        # Handle adaptive periods
        if self.config.adaptive_periods:
            volatility = self._calculate_volatility()
            adjusted_period = max(period * (1 - volatility), period * 0.5)
            alpha = 2 / (adjusted_period + 1)
        else:
            alpha = 2 / (period + 1)
        
        return alpha * new_value + (1 - alpha) * values[-1]
    
    def _calculate_volatility(self) -> float:
        """Calculate price volatility for adaptive periods"""
        if len(self.prices) < 20:
            return 0.0
        
        recent_prices = self.prices[-20:]
        returns = [np.log(recent_prices[i] / recent_prices[i-1]) for i in range(1, len(recent_prices))]
        return np.std(returns) if returns else 0.0
    
    def _calculate_trix(self, third_ema: float) -> Optional[float]:
        """Calculate TRIX as rate of change of third EMA"""
        if len(self.third_ema) < 2:
            return None
        
        prev_third_ema = self.third_ema[-1]
        if prev_third_ema <= 0:
            return None
        
        trix = (third_ema - prev_third_ema) / prev_third_ema
        
        # Apply noise filter if enabled
        if self.config.noise_filter_enabled and abs(trix) < self.config.zero_line_threshold / 10:
            return 0.0
        
        return trix
    
    def _calculate_signal_line(self, trix_value: float) -> float:
        """Calculate TRIX signal line"""
        return self._calculate_ema(self.trix_signal, trix_value, self.config.signal_period)
    
    def _analyze_momentum(self, trix_value: float) -> TRIXMomentum:
        """Analyze TRIX momentum characteristics"""
        if len(self.trix_values) < 5:
            return TRIXMomentum(
                momentum_direction="neutral",
                momentum_strength=0.0,
                acceleration=0.0,
                velocity=0.0,
                persistence=0.0
            )
        
        recent_trix = self.trix_values[-5:]
        
        # Momentum direction
        if trix_value > recent_trix[-2]:
            momentum_direction = "increasing"
            if self.current_trend == "increasing":
                self.momentum_direction_count += 1
            else:
                self.momentum_direction_count = 1
                self.current_trend = "increasing"
        elif trix_value < recent_trix[-2]:
            momentum_direction = "decreasing"
            if self.current_trend == "decreasing":
                self.momentum_direction_count += 1
            else:
                self.momentum_direction_count = 1
                self.current_trend = "decreasing"
        else:
            momentum_direction = "neutral"
            self.momentum_direction_count = 0
        
        # Momentum strength
        trix_range = max(recent_trix) - min(recent_trix)
        momentum_strength = min(abs(trix_value - np.mean(recent_trix)) / trix_range, 1.0) if trix_range > 0 else 0.0
        
        # Acceleration (second derivative)
        acceleration = 0.0
        if len(self.trix_values) >= 3:
            prev_change = recent_trix[-2] - recent_trix[-3]
            current_change = trix_value - recent_trix[-2]
            acceleration = current_change - prev_change
        
        # Velocity (first derivative)
        velocity = trix_value - recent_trix[-2] if len(recent_trix) >= 2 else 0.0
        
        # Persistence
        persistence = min(self.momentum_direction_count / 10, 1.0)
        
        return TRIXMomentum(
            momentum_direction=momentum_direction,
            momentum_strength=momentum_strength,
            acceleration=acceleration,
            velocity=velocity,
            persistence=persistence
        )
    
    def _detect_zero_line_crossover(self, trix_value: float) -> Optional[str]:
        """Detect zero line crossovers"""
        if len(self.trix_values) < 2:
            return None
        
        prev_trix = self.trix_values[-1]
        
        # Bullish crossover
        if prev_trix <= 0 < trix_value and abs(trix_value) > self.config.zero_line_threshold:
            return "bullish"
        
        # Bearish crossover
        elif prev_trix >= 0 > trix_value and abs(trix_value) > self.config.zero_line_threshold:
            return "bearish"
        
        return None
    
    def _detect_signal_line_crossover(self, trix_value: float, signal_value: float) -> Optional[str]:
        """Detect signal line crossovers"""
        if len(self.trix_values) < 2 or len(self.trix_signal) < 2:
            return None
        
        prev_trix = self.trix_values[-1]
        prev_signal = self.trix_signal[-1]
        
        # Bullish crossover
        if prev_trix <= prev_signal < trix_value <= signal_value:
            return "bullish"
        
        # Bearish crossover
        elif prev_trix >= prev_signal > trix_value >= signal_value:
            return "bearish"
        
        return None
    
    def _detect_divergence(self, price: float, trix_value: float) -> Tuple[bool, float]:
        """Detect momentum divergence between price and TRIX"""
        if len(self.prices) < self.config.divergence_lookback or len(self.trix_values) < self.config.divergence_lookback:
            return False, 0.0
        
        lookback = self.config.divergence_lookback
        recent_prices = self.prices[-lookback:]
        recent_trix = self.trix_values[-lookback:]
        
        # Calculate trends
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        trix_trend = np.polyfit(range(len(recent_trix)), recent_trix, 1)[0]
        
        # Divergence detection
        price_direction = 1 if price_trend > 0 else -1
        trix_direction = 1 if trix_trend > 0 else -1
        
        divergence_detected = price_direction != trix_direction
        divergence_strength = abs(price_trend) * abs(trix_trend) if divergence_detected else 0.0
        
        return divergence_detected, divergence_strength
    
    def _determine_trend_state(self, trix_value: float, momentum: TRIXMomentum) -> TRIXTrendState:
        """Determine overall TRIX trend state"""
        if abs(trix_value) < self.config.zero_line_threshold:
            return TRIXTrendState.NEUTRAL
        
        if trix_value > 0:
            if momentum.acceleration > 0:
                return TRIXTrendState.MOMENTUM_ACCELERATING
            elif momentum.momentum_direction == "increasing":
                return TRIXTrendState.BULLISH_MOMENTUM
            else:
                return TRIXTrendState.MOMENTUM_DECELERATING
        else:
            if momentum.acceleration < 0:
                return TRIXTrendState.MOMENTUM_ACCELERATING
            elif momentum.momentum_direction == "decreasing":
                return TRIXTrendState.BEARISH_MOMENTUM
            else:
                return TRIXTrendState.MOMENTUM_DECELERATING
    
    def _calculate_noise_level(self) -> float:
        """Calculate current noise level in TRIX"""
        if len(self.trix_values) < 10:
            return 0.0
        
        recent_trix = self.trix_values[-10:]
        trix_volatility = np.std(recent_trix)
        trix_mean = abs(np.mean(recent_trix))
        
        # Noise level as ratio of volatility to signal
        noise_level = trix_volatility / trix_mean if trix_mean > 0 else 1.0
        return min(noise_level, 1.0)
    
    def update(self, data: Dict[str, float]) -> Optional[IndicatorSignal]:
        """Update TRIX with new market data"""
        try:
            price = self._get_price_value(data)
            self.prices.append(price)
            
            # Calculate triple EMA
            first_ema = self._calculate_ema(self.first_ema, price, self.config.period)
            self.first_ema.append(first_ema)
            
            second_ema = self._calculate_ema(self.second_ema, first_ema, self.config.period)
            self.second_ema.append(second_ema)
            
            third_ema = self._calculate_ema(self.third_ema, second_ema, self.config.period)
            self.third_ema.append(third_ema)
            
            # Calculate TRIX
            trix_value = self._calculate_trix(third_ema)
            if trix_value is None:
                return None
            
            self.trix_values.append(trix_value)
            
            # Calculate signal line
            signal_value = self._calculate_signal_line(trix_value)
            self.trix_signal.append(signal_value)
            
            # Calculate histogram
            histogram = trix_value - signal_value
            self.trix_histogram.append(histogram)
            
            # Analyze momentum
            momentum = self._analyze_momentum(trix_value)
            
            # Detect crossovers
            zero_cross = self._detect_zero_line_crossover(trix_value)
            if zero_cross:
                self.zero_crossings.append((len(self.trix_values) - 1, zero_cross))
            
            signal_cross = self._detect_signal_line_crossover(trix_value, signal_value)
            if signal_cross:
                self.signal_crossings.append((len(self.trix_values) - 1, signal_cross))
            
            # Detect divergence
            divergence_detected, divergence_strength = self._detect_divergence(price, trix_value)
            
            # Calculate distances
            zero_line_distance = abs(trix_value)
            signal_line_distance = abs(histogram)
            
            # Determine trend state
            trend_state = self._determine_trend_state(trix_value, momentum)
            
            # Calculate trend strength
            trend_strength = min(momentum.momentum_strength * momentum.persistence, 1.0)
            
            # Calculate noise level
            self.noise_level = self._calculate_noise_level()
            
            # Create analysis
            analysis = TRIXAnalysis(
                trend_state=trend_state,
                trix_value=trix_value,
                trix_signal=signal_value,
                trix_histogram=histogram,
                momentum=momentum,
                zero_line_distance=zero_line_distance,
                signal_line_distance=signal_line_distance,
                divergence_strength=divergence_strength,
                trend_strength=trend_strength,
                noise_level=self.noise_level
            )
            
            # Generate signals
            return self._generate_signals(analysis, zero_cross, signal_cross, data)
            
        except Exception as e:
            logger.error(f"Error updating TRIX: {e}")
            return None
    
    def _generate_signals(self, analysis: TRIXAnalysis, zero_cross: Optional[str], 
                         signal_cross: Optional[str], data: Dict[str, float]) -> IndicatorSignal:
        """Generate trading signals based on TRIX analysis"""
        signals = []
        signal_strength = SignalStrength.NEUTRAL
        confidence = 0.5
        
        # Zero line crossover signals
        if zero_cross:
            signals.append(f"TRIX {zero_cross} zero-line crossover")
            signal_strength = SignalStrength.STRONG
            confidence += 0.2
            
            # Boost confidence for strong momentum
            if analysis.momentum.momentum_strength > 0.7:
                confidence += 0.1
        
        # Signal line crossover signals
        if signal_cross:
            signals.append(f"TRIX {signal_cross} signal line crossover")
            signal_strength = SignalStrength.MEDIUM
            confidence += 0.15
        
        # Momentum state signals
        if analysis.trend_state == TRIXTrendState.MOMENTUM_ACCELERATING:
            direction = "bullish" if analysis.trix_value > 0 else "bearish"
            signals.append(f"TRIX momentum accelerating {direction}ly")
            signal_strength = SignalStrength.STRONG
            confidence += 0.15
        elif analysis.trend_state == TRIXTrendState.MOMENTUM_DECELERATING:
            signals.append("TRIX momentum decelerating - potential reversal")
            signal_strength = SignalStrength.MEDIUM
            confidence += 0.1
        
        # Divergence signals
        if analysis.divergence_strength > 0.5:
            signals.append("Significant momentum divergence detected by TRIX")
            signal_strength = SignalStrength.STRONG
            confidence += 0.2
        
        # Trend strength signals
        if analysis.trend_strength > 0.8:
            direction = "bullish" if analysis.trix_value > 0 else "bearish"
            signals.append(f"Strong {direction} momentum confirmed by TRIX")
            confidence += 0.1
        
        # Histogram signals
        if self.config.histogram_analysis and len(self.trix_histogram) >= 3:
            recent_hist = self.trix_histogram[-3:]
            if all(recent_hist[i] > recent_hist[i+1] for i in range(len(recent_hist)-1)):
                signals.append("TRIX histogram showing momentum weakening")
                confidence += 0.05
            elif all(recent_hist[i] < recent_hist[i+1] for i in range(len(recent_hist)-1)):
                signals.append("TRIX histogram showing momentum strengthening")
                confidence += 0.05
        
        # Noise level adjustment
        if analysis.noise_level > 0.7:
            signals.append("High noise level detected - exercise caution")
            confidence *= 0.8  # Reduce confidence in noisy conditions
        elif analysis.noise_level < 0.3:
            signals.append("Clean TRIX signal with low noise")
            confidence += 0.05
        
        # Market condition assessment
        market_condition = MarketCondition.TRENDING
        if analysis.trend_state == TRIXTrendState.NEUTRAL:
            market_condition = MarketCondition.RANGING
        elif analysis.noise_level > 0.6:
            market_condition = MarketCondition.VOLATILE
        
        return IndicatorSignal(
            indicator_name=self.name,
            signal_strength=signal_strength,
            confidence=min(confidence, 0.95),
            signals=signals,
            market_condition=market_condition,
            metadata={
                'trix_value': analysis.trix_value,
                'trix_signal': analysis.trix_signal,
                'trix_histogram': analysis.trix_histogram,
                'trend_state': analysis.trend_state.value,
                'momentum_direction': analysis.momentum.momentum_direction,
                'momentum_strength': analysis.momentum.momentum_strength,
                'acceleration': analysis.momentum.acceleration,
                'velocity': analysis.momentum.velocity,
                'persistence': analysis.momentum.persistence,
                'zero_line_distance': analysis.zero_line_distance,
                'signal_line_distance': analysis.signal_line_distance,
                'divergence_strength': analysis.divergence_strength,
                'trend_strength': analysis.trend_strength,
                'noise_level': analysis.noise_level,
                'zero_crossings_count': len(self.zero_crossings),
                'signal_crossings_count': len(self.signal_crossings)
            }
        )
    
    @property
    def name(self) -> str:
        return "TRIX"
    
    def get_momentum_analysis(self) -> Dict[str, Any]:
        """Get detailed momentum analysis"""
        if not self.trix_values:
            return {}
        
        return {
            'current_trix': self.trix_values[-1],
            'current_signal': self.trix_signal[-1] if self.trix_signal else None,
            'current_histogram': self.trix_histogram[-1] if self.trix_histogram else None,
            'momentum_direction': self.current_trend,
            'momentum_persistence': self.momentum_direction_count,
            'noise_level': self.noise_level,
            'zero_crossings': len(self.zero_crossings),
            'signal_crossings': len(self.signal_crossings),
            'trend_changes': len(self.trend_changes)
        }

def test_trix():
    """Test TRIX implementation with realistic market data"""
    config = TRIXConfig(
        period=14,
        signal_period=9,
        price_type="close",
        momentum_analysis=True,
        noise_filter_enabled=True
    )
    
    trix = TRIX(config)
    
    # Generate test data with trend changes
    np.random.seed(42)
    base_price = 1.2000
    
    signals = []
    for i in range(100):
        # Create trend changes at specific points
        if i < 30:
            trend = 0.0001  # Slight uptrend
        elif i < 60:
            trend = -0.0002  # Downtrend
        else:
            trend = 0.0003  # Strong uptrend
        
        noise = np.random.normal(0, 0.0005)
        price = base_price + trend + noise
        
        data = {
            'open': price * 0.9999,
            'high': price * 1.0002,
            'low': price * 0.9998,
            'close': price,
            'volume': 1000000
        }
        
        signal = trix.update(data)
        if signal and signal.signals:
            signals.append((i, signal))
        
        base_price = price
    
    print(f"TRIX Test Results:")
    print(f"Total signals generated: {len(signals)}")
    print(f"Zero line crossings: {len(trix.zero_crossings)}")
    print(f"Signal line crossings: {len(trix.signal_crossings)}")
    
    # Print last few signals
    print("\nRecent signals:")
    for i, signal in signals[-3:]:
        print(f"Bar {i}: {signal.signal_strength.value} - {signal.signals}")
    
    # Test momentum analysis
    momentum_analysis = trix.get_momentum_analysis()
    print(f"\nMomentum Analysis:")
    print(f"Current TRIX: {momentum_analysis.get('current_trix', 0):.6f}")
    print(f"Current Signal: {momentum_analysis.get('current_signal', 0):.6f}")
    print(f"Momentum Direction: {momentum_analysis.get('momentum_direction', 'unknown')}")
    print(f"Noise Level: {momentum_analysis.get('noise_level', 0):.3f}")

if __name__ == "__main__":
    test_trix()
