"""
TRIX (Triple Exponential Average) 
=================================

The TRIX indicator is a momentum oscillator that uses a triple exponential 
moving average to filter out price noise and identify trend changes.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import IndicatorBase


class TRIX(IndicatorBase):
    """
    TRIX (Triple Exponential Average) indicator.
    
    TRIX is calculated by taking the rate of change (1-period percent change) 
    of a triple exponentially smoothed moving average.
    """
    
    def __init__(self, 
                 period: int = 14,
                 signal_period: int = 9):
        """
        Initialize TRIX indicator.
        
        Args:
            period: Period for triple exponential smoothing (typically 14)
            signal_period: Period for signal line EMA (typically 9)
        """
        super().__init__(name="TRIX")
        
        self.period = period
        self.signal_period = signal_period
        
        # Validation
        if period <= 0 or signal_period <= 0:
            raise ValueError("Periods must be positive")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate TRIX values.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            Dictionary containing TRIX values and signals
        """
        try:
            # Validate input data
            required_columns = ['close']
            self._validate_data(data, required_columns)
            
            if len(data) < self.period * 3:
                raise ValueError(f"Insufficient data: need at least {self.period * 3} periods")
            
            close = data['close'].values
            
            # Calculate TRIX
            trix, signal = self._calculate_trix(close)
            
            # Generate signals
            signals = self._generate_signals(trix, signal)
            
            # Calculate additional metrics
            metrics = self._calculate_metrics(trix, signal)
            
            return {
                'trix': trix,
                'signal': signal,
                'histogram': trix - signal,
                'signals': signals,
                'metrics': metrics,
                'interpretation': self._interpret_signals(trix[-1], signal[-1], signals[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating TRIX: {e}")
            raise
    
    def _calculate_trix(self, close: np.ndarray) -> tuple:
        """Calculate TRIX and signal line."""
        # First EMA
        ema1 = self._ema(close, self.period)
        
        # Second EMA
        ema2 = self._ema(ema1, self.period)
        
        # Third EMA
        ema3 = self._ema(ema2, self.period)
        
        # Calculate TRIX (rate of change of triple EMA)
        trix = np.full(len(close), np.nan)
        
        for i in range(1, len(ema3)):
            if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
                trix[i] = ((ema3[i] - ema3[i-1]) / ema3[i-1]) * 10000  # Convert to basis points
        
        # Calculate signal line (EMA of TRIX)
        signal = self._ema(trix, self.signal_period)
        
        return trix, signal
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.full(len(data), np.nan)
        alpha = 2.0 / (period + 1)
        
        # Find first valid value
        start_idx = 0
        while start_idx < len(data) and np.isnan(data[start_idx]):
            start_idx += 1
        
        if start_idx >= len(data):
            return ema
        
        ema[start_idx] = data[start_idx]
        
        for i in range(start_idx + 1, len(data)):
            if not np.isnan(data[i]):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            else:
                ema[i] = ema[i-1]
        
        return ema
    
    def _generate_signals(self, trix: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Generate trading signals based on TRIX."""
        signals = np.zeros(len(trix))
        
        for i in range(1, len(trix)):
            if np.isnan(trix[i]) or np.isnan(signal[i]) or np.isnan(trix[i-1]) or np.isnan(signal[i-1]):
                continue
            
            # Zero line crossovers
            if trix[i-1] <= 0 and trix[i] > 0:
                signals[i] = 1  # Buy signal
            elif trix[i-1] >= 0 and trix[i] < 0:
                signals[i] = -1  # Sell signal
            
            # Signal line crossovers
            elif trix[i-1] <= signal[i-1] and trix[i] > signal[i]:
                signals[i] = 0.5  # Weak buy
            elif trix[i-1] >= signal[i-1] and trix[i] < signal[i]:
                signals[i] = -0.5  # Weak sell
        
        return signals
    
    def _calculate_metrics(self, trix: np.ndarray, signal: np.ndarray) -> Dict:
        """Calculate additional TRIX metrics."""
        valid_trix = trix[~np.isnan(trix)]
        valid_signal = signal[~np.isnan(signal)]
        
        if len(valid_trix) == 0:
            return {}
        
        # Momentum analysis
        positive_pct = np.sum(valid_trix > 0) / len(valid_trix) * 100
        negative_pct = np.sum(valid_trix < 0) / len(valid_trix) * 100
        
        # Histogram analysis
        histogram = trix - signal
        valid_histogram = histogram[~np.isnan(histogram)]
        
        positive_histogram_pct = np.sum(valid_histogram > 0) / len(valid_histogram) * 100 if len(valid_histogram) > 0 else 0
        
        # Recent trend
        recent_trix = valid_trix[-min(5, len(valid_trix)):]
        trend = np.mean(np.diff(recent_trix)) if len(recent_trix) > 1 else 0
        
        # Volatility
        trix_volatility = np.std(valid_trix)
        
        return {
            'current_trix': trix[-1] if not np.isnan(trix[-1]) else None,
            'current_signal': signal[-1] if not np.isnan(signal[-1]) else None,
            'positive_momentum_pct': positive_pct,
            'negative_momentum_pct': negative_pct,
            'positive_histogram_pct': positive_histogram_pct,
            'recent_trend': trend,
            'volatility': trix_volatility,
            'mean_trix': np.mean(valid_trix),
            'max_trix': np.max(valid_trix),
            'min_trix': np.min(valid_trix),
            'signal_strength': abs(trix[-1] - signal[-1]) if not np.isnan(trix[-1]) and not np.isnan(signal[-1]) else 0
        }
    
    def _interpret_signals(self, current_trix: float, current_signal: float, current_signal_value: float) -> str:
        """Provide human-readable interpretation."""
        if np.isnan(current_trix) or np.isnan(current_signal):
            return "Insufficient data for TRIX calculation"
        
        momentum = "BULLISH" if current_trix > 0 else "BEARISH"
        histogram = "above" if current_trix > current_signal else "below"
        
        signal_desc = {
            1: "BUY signal (zero line cross up)",
            0.5: "Weak BUY signal (signal line cross up)",
            -0.5: "Weak SELL signal (signal line cross down)",
            -1: "SELL signal (zero line cross down)",
            0: "No signal"
        }.get(current_signal_value, "No signal")
        
        return f"TRIX: {current_trix:.4f} ({momentum}, {histogram} signal line) - {signal_desc}"


def create_trix(period: int = 14, **kwargs) -> TRIX:
    """Factory function to create TRIX indicator."""
    return TRIX(period=period, **kwargs)
    
    # Momentum direction count and strength calculations would go here
    # if this was part of a class method
    # self.momentum_direction_count = 0
    
    # # Momentum strength
    # trix_range = max(recent_trix) - min(recent_trix)
    # momentum_strength = min(abs(trix_value - np.mean(recent_trix)) / trix_range, 1.0) if trix_range > 0 else 0.0

# The following code appears to be orphaned from a class method and should be removed or fixed
# Commenting out to fix syntax error

# # Acceleration (second derivative)
# acceleration = 0.0
# if len(self.trix_values) >= 3:
#     prev_change = recent_trix[-2] - recent_trix[-3]
#     current_change = trix_value - recent_trix[-2]
#     acceleration = current_change - prev_change
# 
# # Velocity (first derivative)
# velocity = trix_value - recent_trix[-2] if len(recent_trix) >= 2 else 0.0
# 
# # Persistence
# persistence = min(self.momentum_direction_count / 10, 1.0)
# 
# return TRIXMomentum(
#     momentum_direction=momentum_direction,
#     momentum_strength=momentum_strength,
#     acceleration=acceleration,
#     velocity=velocity,
#     persistence=persistence
# )

# The above orphaned code has been commented out to fix syntax errors

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
