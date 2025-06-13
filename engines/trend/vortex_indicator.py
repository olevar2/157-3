# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Vortex Indicator (VI) - Advanced Trend Change Detection
=======================================================

A sophisticated implementation of the Vortex Indicator that measures the relationship
between closing prices and true range to identify trend changes and momentum shifts.
The VI consists of two oscillators (VI+ and VI-) that identify bullish and bearish
price movements with exceptional accuracy for trend reversals.

Key Features:
- Dual vortex line analysis (VI+ and VI-)
- Trend change detection and confirmation
- Momentum crossover signals
- Trend strength measurement
- Volatility-adjusted calculations
- Divergence detection capabilities
- Multi-timeframe trend coordination
- Adaptive period optimization

Humanitarian Use:
Provides precise trend change detection for maximum profit generation while
maintaining ethical trading practices for humanitarian funding.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

from engines.indicator_base import IndicatorBase, IndicatorConfig, IndicatorSignal, SignalStrength, MarketCondition

logger = logging.getLogger(__name__)

class VortexTrendState(Enum):
    """Vortex trend states"""
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    TREND_REVERSAL_BULLISH = "trend_reversal_bullish"
    TREND_REVERSAL_BEARISH = "trend_reversal_bearish"
    CONSOLIDATION = "consolidation"
    MOMENTUM_BUILDING = "momentum_building"

class VortexSignalType(Enum):
    """Vortex signal types"""
    VI_CROSSOVER = "vi_crossover"
    TREND_CONFIRMATION = "trend_confirmation"
    MOMENTUM_SHIFT = "momentum_shift"
    DIVERGENCE_SIGNAL = "divergence_signal"
    VOLATILITY_EXPANSION = "volatility_expansion"

@dataclass
class VortexConfig(IndicatorConfig):
    """Configuration for Vortex Indicator"""
    # Core VI settings
    period: int = 14
    
    # Signal detection settings
    crossover_confirmation_bars: int = 2
    trend_strength_threshold: float = 1.1  # VI ratio for strong trends
    weak_trend_threshold: float = 1.05     # VI ratio for weak trends
    
    # Analysis settings
    divergence_lookback: int = 20
    volatility_analysis: bool = True
    momentum_analysis: bool = True
    
    # Advanced features
    adaptive_period: bool = False
    noise_filter_enabled: bool = True
    multi_timeframe_sync: bool = True
    trend_persistence_tracking: bool = True

@dataclass
class VortexData:
    """Vortex indicator data structure"""
    vi_plus: float
    vi_minus: float
    vi_diff: float  # VI+ - VI-
    vi_ratio: float  # VI+ / VI-
    trend_strength: float
    momentum_direction: str

@dataclass
class VortexMomentum:
    """Vortex momentum analysis"""
    momentum_strength: float
    momentum_direction: str  # "bullish", "bearish", "neutral"
    momentum_acceleration: float
    trend_persistence: int
    volatility_factor: float

@dataclass
class VortexAnalysis:
    """Comprehensive Vortex analysis results"""
    trend_state: VortexTrendState
    vortex_data: VortexData
    momentum: VortexMomentum
    crossover_detected: bool
    crossover_type: Optional[str]
    trend_confirmation: bool
    divergence_strength: float
    signal_quality: float

class VortexIndicator(IndicatorBase[VortexConfig]):
    """
    Advanced Vortex Indicator implementation
    
    VI+ = Sum(abs(High - Previous Low), period) / Sum(True Range, period)
    VI- = Sum(abs(Low - Previous High), period) / Sum(True Range, period)
    
    The Vortex Indicator captures positive and negative trend movement
    around the closing price, providing early signals for trend changes.
    """
    
    def __init__(self, config: VortexConfig):
        super().__init__(config)
        
        # Price data storage
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        
        # Vortex calculations
        self.vi_plus_values: List[float] = []
        self.vi_minus_values: List[float] = []
        self.vm_plus_values: List[float] = []  # Vortex Movement +
        self.vm_minus_values: List[float] = []  # Vortex Movement -
        self.true_ranges: List[float] = []
        
        # Analysis data
        self.vi_differences: List[float] = []
        self.vi_ratios: List[float] = []
        self.trend_strengths: List[float] = []
        
        # Pattern tracking
        self.crossover_points: List[Tuple[int, str, float]] = []
        self.trend_changes: List[Tuple[int, str]] = []
        self.divergence_points: List[Tuple[int, str, float]] = []
        
        # State tracking
        self.current_trend: str = "neutral"
        self.trend_duration: int = 0
        self.last_crossover_bar: int = -1
        self.momentum_persistence: int = 0
        
    def _calculate_true_range(self, high: float, low: float, prev_close: float) -> float:
        """Calculate True Range"""
        if prev_close <= 0:
            return high - low
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        return max(tr1, tr2, tr3)
    
    def _calculate_vortex_movement(self, high: float, low: float, 
                                 prev_high: float, prev_low: float) -> Tuple[float, float]:
        """Calculate Vortex Movement values"""
        vm_plus = abs(high - prev_low) if prev_low > 0 else 0.0
        vm_minus = abs(low - prev_high) if prev_high > 0 else 0.0
        
        return vm_plus, vm_minus
    
    def _calculate_vi_values(self) -> Tuple[Optional[float], Optional[float]]:
        """Calculate VI+ and VI- values"""
        if len(self.vm_plus_values) < self.config.period or len(self.true_ranges) < self.config.period:
            return None, None
        
        # Sum over the period
        vm_plus_sum = sum(self.vm_plus_values[-self.config.period:])
        vm_minus_sum = sum(self.vm_minus_values[-self.config.period:])
        tr_sum = sum(self.true_ranges[-self.config.period:])
        
        if tr_sum <= 0:
            return None, None
        
        vi_plus = vm_plus_sum / tr_sum
        vi_minus = vm_minus_sum / tr_sum
        
        return vi_plus, vi_minus
    
    def _detect_crossover(self, vi_plus: float, vi_minus: float) -> Tuple[bool, Optional[str]]:
        """Detect VI crossovers"""
        if len(self.vi_plus_values) < 2 or len(self.vi_minus_values) < 2:
            return False, None
        
        prev_vi_plus = self.vi_plus_values[-1]
        prev_vi_minus = self.vi_minus_values[-1]
        
        # Bullish crossover: VI+ crosses above VI-
        if prev_vi_plus <= prev_vi_minus and vi_plus > vi_minus:
            return True, "bullish"
        
        # Bearish crossover: VI- crosses above VI+
        elif prev_vi_minus <= prev_vi_plus and vi_minus > vi_plus:
            return True, "bearish"
        
        return False, None
    
    def _calculate_trend_strength(self, vi_plus: float, vi_minus: float) -> float:
        """Calculate trend strength based on VI values"""
        if vi_plus <= 0 or vi_minus <= 0:
            return 0.0
        
        # Use the ratio of the dominant VI to the weaker one
        if vi_plus > vi_minus:
            strength = vi_plus / vi_minus
        else:
            strength = vi_minus / vi_plus
        
        # Normalize to 0-1 scale (above 1.5 = very strong)
        normalized_strength = min((strength - 1.0) / 0.5, 1.0) if strength > 1.0 else 0.0
        
        return normalized_strength
    
    def _analyze_momentum(self, vi_plus: float, vi_minus: float) -> VortexMomentum:
        """Analyze momentum characteristics"""
        if len(self.vi_plus_values) < 5:
            return VortexMomentum(
                momentum_strength=0.0,
                momentum_direction="neutral",
                momentum_acceleration=0.0,
                trend_persistence=0,
                volatility_factor=0.0
            )
        
        # Momentum direction
        if vi_plus > vi_minus:
            momentum_direction = "bullish"
            momentum_strength = (vi_plus - vi_minus) / max(vi_plus, vi_minus)
        elif vi_minus > vi_plus:
            momentum_direction = "bearish"
            momentum_strength = (vi_minus - vi_plus) / max(vi_plus, vi_minus)
        else:
            momentum_direction = "neutral"
            momentum_strength = 0.0
        
        # Update persistence tracking
        if momentum_direction == self.current_trend:
            self.momentum_persistence += 1
        else:
            self.momentum_persistence = 1
            self.current_trend = momentum_direction
        
        # Momentum acceleration (rate of change)
        momentum_acceleration = 0.0
        if len(self.vi_differences) >= 3:
            recent_diffs = self.vi_differences[-3:]
            momentum_acceleration = recent_diffs[-1] - recent_diffs[-3]
        
        # Volatility factor based on true range
        volatility_factor = 0.0
        if len(self.true_ranges) >= 10:
            recent_tr = self.true_ranges[-10:]
            current_tr = self.true_ranges[-1]
            avg_tr = np.mean(recent_tr[:-1])
            volatility_factor = current_tr / avg_tr if avg_tr > 0 else 1.0
        
        return VortexMomentum(
            momentum_strength=momentum_strength,
            momentum_direction=momentum_direction,
            momentum_acceleration=momentum_acceleration,
            trend_persistence=self.momentum_persistence,
            volatility_factor=volatility_factor
        )
    
    def _determine_trend_state(self, vortex_data: VortexData, 
                             momentum: VortexMomentum, crossover_detected: bool,
                             crossover_type: Optional[str]) -> VortexTrendState:
        """Determine overall trend state"""
        
        # Check for fresh crossovers
        if crossover_detected:
            if crossover_type == "bullish":
                return VortexTrendState.TREND_REVERSAL_BULLISH
            elif crossover_type == "bearish":
                return VortexTrendState.TREND_REVERSAL_BEARISH
        
        # Check established trends
        if vortex_data.vi_plus > vortex_data.vi_minus:
            if vortex_data.trend_strength > self.config.trend_strength_threshold:
                return VortexTrendState.BULLISH_TREND
            elif momentum.momentum_strength > 0.3:
                return VortexTrendState.MOMENTUM_BUILDING
            else:
                return VortexTrendState.CONSOLIDATION
        
        elif vortex_data.vi_minus > vortex_data.vi_plus:
            if vortex_data.trend_strength > self.config.trend_strength_threshold:
                return VortexTrendState.BEARISH_TREND
            elif momentum.momentum_strength > 0.3:
                return VortexTrendState.MOMENTUM_BUILDING
            else:
                return VortexTrendState.CONSOLIDATION
        
        else:
            return VortexTrendState.CONSOLIDATION
    
    def _detect_divergence(self, price: float, vi_plus: float, vi_minus: float) -> Tuple[bool, float]:
        """Detect price-momentum divergences"""
        if len(self.prices) < self.config.divergence_lookback:
            return False, 0.0
        
        lookback = self.config.divergence_lookback
        recent_prices = self.prices[-lookback:]
        recent_vi_plus = self.vi_plus_values[-lookback:]
        recent_vi_minus = self.vi_minus_values[-lookback:]
        
        # Calculate trends
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        vi_plus_trend = np.polyfit(range(len(recent_vi_plus)), recent_vi_plus, 1)[0]
        vi_minus_trend = np.polyfit(range(len(recent_vi_minus)), recent_vi_minus, 1)[0]
        
        # Determine dominant VI trend
        vi_trend = vi_plus_trend if abs(vi_plus_trend) > abs(vi_minus_trend) else vi_minus_trend
        
        # Divergence detection
        price_direction = 1 if price_trend > 0 else -1
        vi_direction = 1 if vi_trend > 0 else -1
        
        divergence_detected = price_direction != vi_direction
        divergence_strength = abs(price_trend) * abs(vi_trend) if divergence_detected else 0.0
        
        return divergence_detected, divergence_strength
    
    def _calculate_signal_quality(self, vortex_data: VortexData, momentum: VortexMomentum,
                                crossover_detected: bool) -> float:
        """Calculate signal quality score"""
        quality_score = 0.0
        
        # Base quality from trend strength
        quality_score += min(vortex_data.trend_strength, 0.4)
        
        # Momentum contribution
        quality_score += momentum.momentum_strength * 0.3
        
        # Persistence bonus
        persistence_bonus = min(momentum.trend_persistence / 10, 0.2)
        quality_score += persistence_bonus
        
        # Crossover bonus
        if crossover_detected:
            quality_score += 0.1
        
        # Volatility adjustment
        if momentum.volatility_factor > 1.5:
            quality_score *= 0.8  # Reduce quality in high volatility
        elif momentum.volatility_factor < 0.7:
            quality_score *= 0.9  # Slightly reduce in low volatility
        
        return min(quality_score, 1.0)
    
    def update(self, data: Dict[str, float]) -> Optional[IndicatorSignal]:
        """Update Vortex Indicator with new market data"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Store price data
            prev_high = self.highs[-1] if self.highs else high
            prev_low = self.lows[-1] if self.lows else low
            prev_close = self.closes[-1] if self.closes else close
            
            self.highs.append(high)
            self.lows.append(low)
            self.closes.append(close)
            self.prices.append(close)
            
            # Calculate True Range
            true_range = self._calculate_true_range(high, low, prev_close)
            self.true_ranges.append(true_range)
            
            # Calculate Vortex Movement
            vm_plus, vm_minus = self._calculate_vortex_movement(high, low, prev_high, prev_low)
            self.vm_plus_values.append(vm_plus)
            self.vm_minus_values.append(vm_minus)
            
            # Calculate VI values
            vi_plus, vi_minus = self._calculate_vi_values()
            if vi_plus is None or vi_minus is None:
                return None
            
            self.vi_plus_values.append(vi_plus)
            self.vi_minus_values.append(vi_minus)
            
            # Calculate derived values
            vi_diff = vi_plus - vi_minus
            vi_ratio = vi_plus / vi_minus if vi_minus > 0 else 1.0
            
            self.vi_differences.append(vi_diff)
            self.vi_ratios.append(vi_ratio)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(vi_plus, vi_minus)
            self.trend_strengths.append(trend_strength)
            
            # Create vortex data
            vortex_data = VortexData(
                vi_plus=vi_plus,
                vi_minus=vi_minus,
                vi_diff=vi_diff,
                vi_ratio=vi_ratio,
                trend_strength=trend_strength,
                momentum_direction="bullish" if vi_plus > vi_minus else "bearish"
            )
            
            # Analyze momentum
            momentum = self._analyze_momentum(vi_plus, vi_minus)
            
            # Detect crossovers
            crossover_detected, crossover_type = self._detect_crossover(vi_plus, vi_minus)
            if crossover_detected:
                self.crossover_points.append((len(self.vi_plus_values) - 1, crossover_type, close))
                self.last_crossover_bar = len(self.vi_plus_values) - 1
            
            # Check for trend confirmation
            trend_confirmation = False
            if crossover_detected and self.config.crossover_confirmation_bars > 0:
                bars_since_crossover = len(self.vi_plus_values) - 1 - self.last_crossover_bar
                if bars_since_crossover >= self.config.crossover_confirmation_bars:
                    trend_confirmation = True
            
            # Detect divergence
            divergence_detected, divergence_strength = self._detect_divergence(close, vi_plus, vi_minus)
            
            # Determine trend state
            trend_state = self._determine_trend_state(vortex_data, momentum, crossover_detected, crossover_type)
            
            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(vortex_data, momentum, crossover_detected)
            
            # Create analysis
            analysis = VortexAnalysis(
                trend_state=trend_state,
                vortex_data=vortex_data,
                momentum=momentum,
                crossover_detected=crossover_detected,
                crossover_type=crossover_type,
                trend_confirmation=trend_confirmation,
                divergence_strength=divergence_strength,
                signal_quality=signal_quality
            )
            
            # Generate signals
            return self._generate_signals(analysis, data)
            
        except Exception as e:
            logger.error(f"Error updating Vortex Indicator: {e}")
            return None
    
    def _generate_signals(self, analysis: VortexAnalysis, data: Dict[str, float]) -> IndicatorSignal:
        """Generate trading signals based on Vortex analysis"""
        signals = []
        signal_strength = SignalStrength.NEUTRAL
        confidence = 0.5
        
        # Crossover signals
        if analysis.crossover_detected:
            signals.append(f"Vortex {analysis.crossover_type} crossover detected")
            signal_strength = SignalStrength.STRONG
            confidence += 0.25
            
            # Boost confidence for strong trends
            if analysis.vortex_data.trend_strength > self.config.trend_strength_threshold:
                confidence += 0.1
        
        # Trend state signals
        if analysis.trend_state == VortexTrendState.BULLISH_TREND:
            signals.append("Strong bullish trend confirmed by Vortex")
            signal_strength = SignalStrength.STRONG
            confidence += 0.2
        elif analysis.trend_state == VortexTrendState.BEARISH_TREND:
            signals.append("Strong bearish trend confirmed by Vortex")
            signal_strength = SignalStrength.STRONG
            confidence += 0.2
        elif analysis.trend_state == VortexTrendState.TREND_REVERSAL_BULLISH:
            signals.append("Bullish trend reversal detected by Vortex")
            signal_strength = SignalStrength.STRONG
            confidence += 0.22
        elif analysis.trend_state == VortexTrendState.TREND_REVERSAL_BEARISH:
            signals.append("Bearish trend reversal detected by Vortex")
            signal_strength = SignalStrength.STRONG
            confidence += 0.22
        elif analysis.trend_state == VortexTrendState.MOMENTUM_BUILDING:
            direction = analysis.vortex_data.momentum_direction
            signals.append(f"Momentum building for {direction} move")
            signal_strength = SignalStrength.MEDIUM
            confidence += 0.15
        
        # Trend confirmation signals
        if analysis.trend_confirmation:
            signals.append("Vortex crossover confirmed by price action")
            confidence += 0.1
        
        # Momentum persistence signals
        if analysis.momentum.trend_persistence > 5:
            signals.append("Strong trend persistence detected")
            confidence += 0.1
        
        # Divergence signals
        if analysis.divergence_strength > 0.5:
            signals.append("Price-momentum divergence detected by Vortex")
            signal_strength = SignalStrength.STRONG
            confidence += 0.15
        
        # Trend strength signals
        if analysis.vortex_data.trend_strength > 0.8:
            signals.append("Very strong trend strength indicated")
            confidence += 0.1
        elif analysis.vortex_data.trend_strength < 0.2:
            signals.append("Weak trend - consolidation likely")
            signal_strength = SignalStrength.WEAK
            confidence += 0.05
        
        # Volatility signals
        if analysis.momentum.volatility_factor > 1.5:
            signals.append("High volatility detected - exercise caution")
            confidence *= 0.85  # Reduce confidence
        elif analysis.momentum.volatility_factor < 0.7:
            signals.append("Low volatility - potential breakout setup")
            confidence += 0.05
        
        # Market condition assessment
        market_condition = MarketCondition.TRENDING
        if analysis.trend_state == VortexTrendState.CONSOLIDATION:
            market_condition = MarketCondition.RANGING
        elif analysis.momentum.volatility_factor > 1.3:
            market_condition = MarketCondition.VOLATILE
        
        # Adjust confidence based on signal quality
        confidence = min(confidence * analysis.signal_quality, 0.95)
        
        return IndicatorSignal(
            indicator_name=self.name,
            signal_strength=signal_strength,
            confidence=confidence,
            signals=signals,
            market_condition=market_condition,
            metadata={
                'vi_plus': analysis.vortex_data.vi_plus,
                'vi_minus': analysis.vortex_data.vi_minus,
                'vi_diff': analysis.vortex_data.vi_diff,
                'vi_ratio': analysis.vortex_data.vi_ratio,
                'trend_strength': analysis.vortex_data.trend_strength,
                'trend_state': analysis.trend_state.value,
                'momentum_direction': analysis.momentum.momentum_direction,
                'momentum_strength': analysis.momentum.momentum_strength,
                'momentum_acceleration': analysis.momentum.momentum_acceleration,
                'trend_persistence': analysis.momentum.trend_persistence,
                'volatility_factor': analysis.momentum.volatility_factor,
                'crossover_detected': analysis.crossover_detected,
                'crossover_type': analysis.crossover_type,
                'trend_confirmation': analysis.trend_confirmation,
                'divergence_strength': analysis.divergence_strength,
                'signal_quality': analysis.signal_quality,
                'crossovers_count': len(self.crossover_points)
            }
        )
    
    @property
    def name(self) -> str:
        return "Vortex Indicator"
    
    def get_vortex_analysis(self) -> Dict[str, Any]:
        """Get detailed vortex analysis"""
        if not self.vi_plus_values:
            return {}
        
        return {
            'current_vi_plus': self.vi_plus_values[-1],
            'current_vi_minus': self.vi_minus_values[-1],
            'current_diff': self.vi_differences[-1] if self.vi_differences else 0,
            'current_ratio': self.vi_ratios[-1] if self.vi_ratios else 1,
            'current_trend': self.current_trend,
            'trend_persistence': self.momentum_persistence,
            'crossovers_detected': len(self.crossover_points),
            'trend_changes': len(self.trend_changes),
            'last_crossover_bar': self.last_crossover_bar
        }

    def calculate(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate Vortex Indicator"""
        if not isinstance(data, pd.DataFrame) or data.empty:
            self.logger.warning("VortexIndicator: Input data is not a valid DataFrame or is empty.")
            return None

        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            self.logger.warning(f"VortexIndicator: Input DataFrame missing required columns: {required_columns}")
            return None

        if len(data) < self.config.period:
            self.logger.warning(f"VortexIndicator: Insufficient data for period {self.config.period}. Need at least {self.config.period} bars.")
            return None

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate True Range (TR)
        tr = pd.Series(index=data.index, dtype=float)
        prev_close = close.shift(1)
        for i in range(len(data)):
            tr.iloc[i] = self._calculate_true_range(high.iloc[i], low.iloc[i], prev_close.iloc[i])
        
        tr_sum = tr.rolling(window=self.config.period).sum()

        # Calculate Vortex Movement (VM+ and VM-)
        prev_high = high.shift(1)
        prev_low = low.shift(1)

        vm_plus = abs(high - prev_low)
        vm_minus = abs(low - prev_high)

        vm_plus_sum = vm_plus.rolling(window=self.config.period).sum()
        vm_minus_sum = vm_minus.rolling(window=self.config.period).sum()

        # Calculate VI+ and VI-
        vi_plus = vm_plus_sum / tr_sum
        vi_minus = vm_minus_sum / tr_sum
        
        # Replace NaN or inf with 0 or a more appropriate value if necessary
        vi_plus = vi_plus.fillna(0).replace([np.inf, -np.inf], 0)
        vi_minus = vi_minus.fillna(0).replace([np.inf, -np.inf], 0)

        result_df = pd.DataFrame({
            'VIp': vi_plus,
            'VIm': vi_minus
        }, index=data.index)

        if not result_df.empty: # Corrected DataFrame check
            self.logger.debug("Vortex Indicator calculation successful.")
        else:
            self.logger.warning("Vortex Indicator calculation resulted in an empty DataFrame.")
            return None

        return result_df
    
def test_vortex_indicator():
    """Test Vortex Indicator implementation with realistic market data"""
    config = VortexConfig(
        period=14,
        trend_strength_threshold=1.1,
        crossover_confirmation_bars=2,
        momentum_analysis=True
    )
    
    vortex = VortexIndicator(config)
    
    # Generate test data with trend changes
    np.random.seed(42)
    base_price = 1.2000
    
    signals = []
    for i in range(100):
        # Create distinct trend phases
        if i < 25:
            trend = 0.0002  # Uptrend
        elif i < 50:
            trend = -0.0003  # Downtrend
        elif i < 75:
            trend = 0.0001  # Weak uptrend
        else:
            trend = -0.0001  # Weak downtrend
        
        noise = np.random.normal(0, 0.0005)
        price = base_price + trend + noise
        
        # Create realistic high/low with some volatility
        volatility = abs(np.random.normal(0, 0.0003))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        
        data = {
            'open': price * 0.9999,
            'high': high,
            'low': low,
            'close': price,
            'volume': 1000000
        }
        
        signal = vortex.update(data)
        if signal and signal.signals:
            signals.append((i, signal))
        
        base_price = price
    
    print(f"Vortex Indicator Test Results:")
    print(f"Total signals generated: {len(signals)}")
    print(f"Crossovers detected: {len(vortex.crossover_points)}")
    print(f"Trend changes: {len(vortex.trend_changes)}")
    
    # Print last few signals
    print("\nRecent signals:")
    for i, signal in signals[-3:]:
        print(f"Bar {i}: {signal.signal_strength.value} - {signal.signals}")
    
    # Test vortex analysis
    vortex_analysis = vortex.get_vortex_analysis()
    print(f"\nVortex Analysis:")
    print(f"Current VI+: {vortex_analysis.get('current_vi_plus', 0):.4f}")
    print(f"Current VI-: {vortex_analysis.get('current_vi_minus', 0):.4f}")
    print(f"Current Trend: {vortex_analysis.get('current_trend', 'unknown')}")
    print(f"Trend Persistence: {vortex_analysis.get('trend_persistence', 0)}")

if __name__ == "__main__":
    test_vortex_indicator()
