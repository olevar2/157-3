"""
Swing Trading Momentum Indicators
Momentum indicators optimized for swing trading (H1-H4 timeframes)

Features:
- Multi-day momentum analysis
- Swing high/low detection
- Trend reversal momentum
- Weekly/monthly momentum cycles
- Support/resistance momentum
- Fibonacci retracement momentum
- Long-term momentum sustainability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SwingPhase(Enum):
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    CONSOLIDATION = "consolidation"

class SwingDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class SwingSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class MomentumCycle(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class SwingMomentumResult:
    swing_momentum: float
    trend_momentum: float
    reversal_momentum: float
    cycle_momentum: Dict[MomentumCycle, float]
    swing_phase: SwingPhase
    swing_direction: SwingDirection
    signal: SwingSignal
    confidence: float
    swing_high: float
    swing_low: float
    momentum_sustainability: float
    fibonacci_levels: Dict[str, float]
    timestamp: datetime

@dataclass
class SwingConfig:
    momentum_period: int = 21
    trend_period: int = 50
    reversal_period: int = 14
    swing_detection_period: int = 20
    fibonacci_lookback: int = 100
    cycle_periods: Dict[MomentumCycle, int] = None
    
    def __post_init__(self):
        if self.cycle_periods is None:
            self.cycle_periods = {
                MomentumCycle.DAILY: 24,    # 24 hours
                MomentumCycle.WEEKLY: 168,  # 168 hours (7 days)
                MomentumCycle.MONTHLY: 720  # 720 hours (30 days)
            }

class SwingMomentumIndicator:
    """
    Advanced momentum indicator for swing trading (H1-H4 timeframes)
    """
    
    def __init__(self, config: Optional[SwingConfig] = None):
        self.config = config or SwingConfig()
        
        # Data buffers
        self.price_buffer = []
        self.high_buffer = []
        self.low_buffer = []
        self.volume_buffer = []
        self.timestamp_buffer = []
        
        # Swing point tracking
        self.swing_highs = []
        self.swing_lows = []
        self.pivot_points = []
        
        # Momentum tracking
        self.momentum_history = []
        self.trend_history = []
        self.cycle_data = {cycle: [] for cycle in MomentumCycle}
        
        # Performance tracking
        self.calculation_count = 0
        self.signal_accuracy = 0.0
        self.last_calculation_time = None
        
        logger.info("SwingMomentumIndicator initialized")

    def calculate(self, prices: List[float], highs: List[float], lows: List[float],
                 volumes: List[float], timestamps: List[datetime]) -> SwingMomentumResult:
        """
        Calculate comprehensive swing momentum indicators
        
        Args:
            prices: Close prices
            highs: High prices
            lows: Low prices
            volumes: Volume data
            timestamps: Timestamps for cycle analysis
            
        Returns:
            SwingMomentumResult with all momentum indicators
        """
        try:
            start_time = datetime.now()
            
            # Update buffers
            self._update_buffers(prices, highs, lows, volumes, timestamps)
            
            if len(self.price_buffer) < self.config.momentum_period:
                return self._default_result()
            
            # Calculate core momentum indicators
            swing_momentum = self._calculate_swing_momentum()
            trend_momentum = self._calculate_trend_momentum()
            reversal_momentum = self._calculate_reversal_momentum()
            cycle_momentum = self._calculate_cycle_momentum()
            
            # Detect swing points
            swing_high, swing_low = self._detect_swing_points()
            
            # Calculate Fibonacci levels
            fibonacci_levels = self._calculate_fibonacci_levels(swing_high, swing_low)
            
            # Determine swing characteristics
            swing_phase = self._determine_swing_phase(swing_momentum, trend_momentum, reversal_momentum)
            swing_direction = self._determine_swing_direction(swing_momentum, trend_momentum)
            
            # Calculate momentum sustainability
            momentum_sustainability = self._calculate_momentum_sustainability()
            
            # Generate trading signal
            signal = self._generate_swing_signal(
                swing_momentum, trend_momentum, reversal_momentum, 
                swing_phase, momentum_sustainability
            )
            
            # Calculate confidence
            confidence = self._calculate_signal_confidence(
                swing_momentum, trend_momentum, reversal_momentum, 
                cycle_momentum, momentum_sustainability
            )
            
            result = SwingMomentumResult(
                swing_momentum=swing_momentum,
                trend_momentum=trend_momentum,
                reversal_momentum=reversal_momentum,
                cycle_momentum=cycle_momentum,
                swing_phase=swing_phase,
                swing_direction=swing_direction,
                signal=signal,
                confidence=confidence,
                swing_high=swing_high,
                swing_low=swing_low,
                momentum_sustainability=momentum_sustainability,
                fibonacci_levels=fibonacci_levels,
                timestamp=datetime.now()
            )
            
            # Update performance tracking
            self.calculation_count += 1
            self.last_calculation_time = datetime.now()
            calculation_time = (self.last_calculation_time - start_time).total_seconds() * 1000
            
            # Update history
            self.momentum_history.append(swing_momentum)
            self.trend_history.append(trend_momentum)
            
            # Maintain history size
            if len(self.momentum_history) > 200:
                self.momentum_history = self.momentum_history[-200:]
            if len(self.trend_history) > 200:
                self.trend_history = self.trend_history[-200:]
            
            logger.debug(f"Swing momentum calculated: {signal.value} "
                        f"(confidence: {confidence:.3f}, time: {calculation_time:.1f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating swing momentum: {e}")
            return self._default_result()

    def _update_buffers(self, prices: List[float], highs: List[float], 
                       lows: List[float], volumes: List[float], timestamps: List[datetime]):
        """Update internal data buffers"""
        # Extend buffers with new data
        if isinstance(prices, list):
            self.price_buffer.extend(prices)
            self.high_buffer.extend(highs)
            self.low_buffer.extend(lows)
            self.volume_buffer.extend(volumes)
            self.timestamp_buffer.extend(timestamps)
        else:
            self.price_buffer.append(prices)
            self.high_buffer.append(highs)
            self.low_buffer.append(lows)
            self.volume_buffer.append(volumes)
            self.timestamp_buffer.append(timestamps)
        
        # Maintain buffer size (keep last 500 values for swing analysis)
        max_buffer_size = 500
        if len(self.price_buffer) > max_buffer_size:
            self.price_buffer = self.price_buffer[-max_buffer_size:]
            self.high_buffer = self.high_buffer[-max_buffer_size:]
            self.low_buffer = self.low_buffer[-max_buffer_size:]
            self.volume_buffer = self.volume_buffer[-max_buffer_size:]
            self.timestamp_buffer = self.timestamp_buffer[-max_buffer_size:]

    def _calculate_swing_momentum(self) -> float:
        """Calculate swing-specific momentum"""
        try:
            if len(self.price_buffer) < self.config.momentum_period:
                return 0.0
            
            prices = np.array(self.price_buffer[-self.config.momentum_period:])
            
            # Calculate rate of change over swing period
            roc = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
            
            # Calculate momentum acceleration
            if len(prices) >= 10:
                recent_roc = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
                earlier_roc = (prices[-10] - prices[-20]) / prices[-20] if len(prices) >= 20 and prices[-20] != 0 else 0
                acceleration = recent_roc - earlier_roc
            else:
                acceleration = 0
            
            # Volume-weighted momentum
            volumes = np.array(self.volume_buffer[-self.config.momentum_period:])
            if len(volumes) == len(prices):
                avg_volume = np.mean(volumes)
                current_volume = volumes[-1]
                volume_factor = min(current_volume / avg_volume, 3.0) if avg_volume > 0 else 1.0
            else:
                volume_factor = 1.0
            
            # Combine components
            swing_momentum = (roc * 0.6 + acceleration * 0.4) * volume_factor
            
            # Normalize to -1 to 1 range
            swing_momentum = np.tanh(swing_momentum * 10)
            
            return swing_momentum
            
        except Exception as e:
            logger.error(f"Error calculating swing momentum: {e}")
            return 0.0

    def _calculate_trend_momentum(self) -> float:
        """Calculate long-term trend momentum"""
        try:
            if len(self.price_buffer) < self.config.trend_period:
                return 0.0
            
            prices = np.array(self.price_buffer[-self.config.trend_period:])
            
            # Calculate multiple moving averages for trend analysis
            short_ma = np.mean(prices[-10:])
            medium_ma = np.mean(prices[-20:])
            long_ma = np.mean(prices[-self.config.trend_period:])
            
            # Trend strength based on MA alignment
            if short_ma > medium_ma > long_ma:
                trend_strength = 1.0  # Strong uptrend
            elif short_ma < medium_ma < long_ma:
                trend_strength = -1.0  # Strong downtrend
            else:
                # Calculate partial trend strength
                short_vs_medium = (short_ma - medium_ma) / medium_ma if medium_ma != 0 else 0
                medium_vs_long = (medium_ma - long_ma) / long_ma if long_ma != 0 else 0
                trend_strength = (short_vs_medium + medium_vs_long) / 2
                trend_strength = np.tanh(trend_strength * 100)
            
            # Factor in price position relative to trend
            current_price = prices[-1]
            price_vs_trend = (current_price - long_ma) / long_ma if long_ma != 0 else 0
            
            # Combine trend strength and price position
            trend_momentum = trend_strength * 0.7 + np.tanh(price_vs_trend * 50) * 0.3
            
            return trend_momentum
            
        except Exception as e:
            logger.error(f"Error calculating trend momentum: {e}")
            return 0.0

    def _calculate_reversal_momentum(self) -> float:
        """Calculate momentum for potential trend reversals"""
        try:
            if len(self.price_buffer) < self.config.reversal_period:
                return 0.0
            
            prices = np.array(self.price_buffer[-self.config.reversal_period:])
            highs = np.array(self.high_buffer[-self.config.reversal_period:])
            lows = np.array(self.low_buffer[-self.config.reversal_period:])
            
            # Detect divergence patterns
            price_trend = self._calculate_linear_trend(prices)
            high_trend = self._calculate_linear_trend(highs)
            low_trend = self._calculate_linear_trend(lows)
            
            # Bullish divergence: price making lower lows but momentum improving
            if price_trend < 0 and low_trend < 0:
                # Check if recent momentum is improving
                recent_momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] != 0 else 0
                earlier_momentum = (prices[-5] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] != 0 else 0
                
                if recent_momentum > earlier_momentum:
                    reversal_momentum = abs(recent_momentum - earlier_momentum)
                else:
                    reversal_momentum = 0
            
            # Bearish divergence: price making higher highs but momentum weakening
            elif price_trend > 0 and high_trend > 0:
                recent_momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] != 0 else 0
                earlier_momentum = (prices[-5] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] != 0 else 0
                
                if recent_momentum < earlier_momentum:
                    reversal_momentum = -abs(recent_momentum - earlier_momentum)
                else:
                    reversal_momentum = 0
            else:
                reversal_momentum = 0
            
            # Normalize
            reversal_momentum = np.tanh(reversal_momentum * 100)
            
            return reversal_momentum
            
        except Exception as e:
            logger.error(f"Error calculating reversal momentum: {e}")
            return 0.0

    def _calculate_cycle_momentum(self) -> Dict[MomentumCycle, float]:
        """Calculate momentum for different time cycles"""
        cycle_momentum = {}
        
        try:
            for cycle, period in self.config.cycle_periods.items():
                if len(self.price_buffer) >= period:
                    prices = np.array(self.price_buffer[-period:])
                    
                    # Calculate cycle-specific momentum
                    cycle_roc = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
                    
                    # Factor in cycle volatility
                    cycle_volatility = np.std(np.diff(prices)) / np.mean(prices) if np.mean(prices) != 0 else 0
                    
                    # Adjust momentum by volatility (higher volatility = less reliable momentum)
                    adjusted_momentum = cycle_roc * (1 - min(cycle_volatility * 10, 0.5))
                    
                    cycle_momentum[cycle] = np.tanh(adjusted_momentum * 10)
                else:
                    cycle_momentum[cycle] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating cycle momentum: {e}")
            cycle_momentum = {cycle: 0.0 for cycle in MomentumCycle}
        
        return cycle_momentum

    def _detect_swing_points(self) -> Tuple[float, float]:
        """Detect recent swing high and low points"""
        try:
            if len(self.high_buffer) < self.config.swing_detection_period:
                current_price = self.price_buffer[-1] if self.price_buffer else 0
                return current_price, current_price
            
            highs = np.array(self.high_buffer[-self.config.swing_detection_period:])
            lows = np.array(self.low_buffer[-self.config.swing_detection_period:])
            
            # Find swing high (highest high in recent period)
            swing_high = np.max(highs)
            
            # Find swing low (lowest low in recent period)
            swing_low = np.min(lows)
            
            # Update swing point history
            if len(self.swing_highs) == 0 or swing_high > self.swing_highs[-1]:
                self.swing_highs.append(swing_high)
            
            if len(self.swing_lows) == 0 or swing_low < self.swing_lows[-1]:
                self.swing_lows.append(swing_low)
            
            # Maintain history size
            if len(self.swing_highs) > 50:
                self.swing_highs = self.swing_highs[-50:]
            if len(self.swing_lows) > 50:
                self.swing_lows = self.swing_lows[-50:]
            
            return swing_high, swing_low
            
        except Exception as e:
            logger.error(f"Error detecting swing points: {e}")
            current_price = self.price_buffer[-1] if self.price_buffer else 0
            return current_price, current_price

    def _calculate_fibonacci_levels(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            if swing_high == swing_low:
                return {}
            
            range_size = swing_high - swing_low
            
            fibonacci_ratios = {
                '0.0': 0.0,
                '23.6': 0.236,
                '38.2': 0.382,
                '50.0': 0.5,
                '61.8': 0.618,
                '78.6': 0.786,
                '100.0': 1.0
            }
            
            fibonacci_levels = {}
            for level_name, ratio in fibonacci_ratios.items():
                fibonacci_levels[level_name] = swing_high - (range_size * ratio)
            
            return fibonacci_levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}

    def _calculate_linear_trend(self, data: np.ndarray) -> float:
        """Calculate linear trend slope"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        # Normalize slope relative to data level
        normalized_slope = slope / np.mean(data) if np.mean(data) != 0 else 0
        
        return normalized_slope

    def _determine_swing_phase(self, swing_momentum: float, trend_momentum: float, 
                             reversal_momentum: float) -> SwingPhase:
        """Determine current swing trading phase"""
        # Strong upward momentum = markup phase
        if swing_momentum > 0.5 and trend_momentum > 0.3:
            return SwingPhase.MARKUP
        
        # Strong downward momentum = markdown phase
        elif swing_momentum < -0.5 and trend_momentum < -0.3:
            return SwingPhase.MARKDOWN
        
        # Reversal momentum detected = potential phase change
        elif abs(reversal_momentum) > 0.3:
            if reversal_momentum > 0:
                return SwingPhase.ACCUMULATION
            else:
                return SwingPhase.DISTRIBUTION
        
        # Low momentum = consolidation
        elif abs(swing_momentum) < 0.2 and abs(trend_momentum) < 0.2:
            return SwingPhase.CONSOLIDATION
        
        # Default based on trend direction
        elif trend_momentum > 0:
            return SwingPhase.ACCUMULATION
        else:
            return SwingPhase.DISTRIBUTION

    def _determine_swing_direction(self, swing_momentum: float, trend_momentum: float) -> SwingDirection:
        """Determine overall swing direction"""
        combined_momentum = swing_momentum * 0.6 + trend_momentum * 0.4
        
        if combined_momentum > 0.2:
            return SwingDirection.BULLISH
        elif combined_momentum < -0.2:
            return SwingDirection.BEARISH
        else:
            return SwingDirection.NEUTRAL

    def _calculate_momentum_sustainability(self) -> float:
        """Calculate how sustainable current momentum is"""
        try:
            if len(self.momentum_history) < 10:
                return 0.5
            
            recent_momentum = self.momentum_history[-10:]
            
            # Check momentum consistency
            momentum_std = np.std(recent_momentum)
            momentum_mean = np.mean(recent_momentum)
            
            # Lower standard deviation = more sustainable
            consistency_score = 1.0 / (1.0 + momentum_std)
            
            # Factor in momentum strength
            strength_score = min(abs(momentum_mean), 1.0)
            
            # Combine scores
            sustainability = (consistency_score * 0.6 + strength_score * 0.4)
            
            return sustainability
            
        except Exception as e:
            logger.error(f"Error calculating momentum sustainability: {e}")
            return 0.5

    def _generate_swing_signal(self, swing_momentum: float, trend_momentum: float,
                             reversal_momentum: float, swing_phase: SwingPhase,
                             momentum_sustainability: float) -> SwingSignal:
        """Generate swing trading signal"""
        # Strong bullish conditions
        if (swing_momentum > 0.6 and trend_momentum > 0.4 and 
            swing_phase in [SwingPhase.MARKUP, SwingPhase.ACCUMULATION] and
            momentum_sustainability > 0.7):
            return SwingSignal.STRONG_BUY
        
        # Bullish conditions
        elif (swing_momentum > 0.3 and trend_momentum > 0.2 and
              swing_phase != SwingPhase.DISTRIBUTION):
            return SwingSignal.BUY
        
        # Strong bearish conditions
        elif (swing_momentum < -0.6 and trend_momentum < -0.4 and
              swing_phase in [SwingPhase.MARKDOWN, SwingPhase.DISTRIBUTION] and
              momentum_sustainability > 0.7):
            return SwingSignal.STRONG_SELL
        
        # Bearish conditions
        elif (swing_momentum < -0.3 and trend_momentum < -0.2 and
              swing_phase != SwingPhase.ACCUMULATION):
            return SwingSignal.SELL
        
        # Reversal signals
        elif abs(reversal_momentum) > 0.5:
            if reversal_momentum > 0:
                return SwingSignal.BUY
            else:
                return SwingSignal.SELL
        
        # Default to hold
        return SwingSignal.HOLD

    def _calculate_signal_confidence(self, swing_momentum: float, trend_momentum: float,
                                   reversal_momentum: float, cycle_momentum: Dict[MomentumCycle, float],
                                   momentum_sustainability: float) -> float:
        """Calculate confidence score for the signal"""
        # Base confidence on momentum alignment
        momentum_alignment = abs(swing_momentum + trend_momentum) / 2
        
        # Factor in cycle momentum alignment
        cycle_values = list(cycle_momentum.values())
        if cycle_values:
            cycle_alignment = abs(np.mean(cycle_values))
        else:
            cycle_alignment = 0.0
        
        # Factor in sustainability
        sustainability_factor = momentum_sustainability
        
        # Factor in reversal strength (can increase confidence for reversal signals)
        reversal_factor = min(abs(reversal_momentum), 0.5)
        
        # Combine factors
        confidence = (momentum_alignment * 0.4 + cycle_alignment * 0.2 + 
                     sustainability_factor * 0.3 + reversal_factor * 0.1)
        
        return min(max(confidence, 0.0), 1.0)

    def _default_result(self) -> SwingMomentumResult:
        """Return default result when calculation fails"""
        current_price = self.price_buffer[-1] if self.price_buffer else 0
        
        return SwingMomentumResult(
            swing_momentum=0.0,
            trend_momentum=0.0,
            reversal_momentum=0.0,
            cycle_momentum={cycle: 0.0 for cycle in MomentumCycle},
            swing_phase=SwingPhase.CONSOLIDATION,
            swing_direction=SwingDirection.NEUTRAL,
            signal=SwingSignal.HOLD,
            confidence=0.0,
            swing_high=current_price,
            swing_low=current_price,
            momentum_sustainability=0.5,
            fibonacci_levels={},
            timestamp=datetime.now()
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'calculation_count': self.calculation_count,
            'signal_accuracy': self.signal_accuracy,
            'last_calculation_time': self.last_calculation_time.isoformat() if self.last_calculation_time else None,
            'buffer_sizes': {
                'price_buffer': len(self.price_buffer),
                'high_buffer': len(self.high_buffer),
                'low_buffer': len(self.low_buffer),
                'volume_buffer': len(self.volume_buffer)
            },
            'swing_points': {
                'swing_highs_count': len(self.swing_highs),
                'swing_lows_count': len(self.swing_lows),
                'latest_swing_high': self.swing_highs[-1] if self.swing_highs else None,
                'latest_swing_low': self.swing_lows[-1] if self.swing_lows else None
            },
            'momentum_history_size': len(self.momentum_history),
            'config': {
                'momentum_period': self.config.momentum_period,
                'trend_period': self.config.trend_period,
                'reversal_period': self.config.reversal_period,
                'swing_detection_period': self.config.swing_detection_period
            }
        }

    def reset(self):
        """Reset all buffers and tracking data"""
        self.price_buffer.clear()
        self.high_buffer.clear()
        self.low_buffer.clear()
        self.volume_buffer.clear()
        self.timestamp_buffer.clear()
        
        self.swing_highs.clear()
        self.swing_lows.clear()
        self.pivot_points.clear()
        
        self.momentum_history.clear()
        self.trend_history.clear()
        
        for cycle in MomentumCycle:
            self.cycle_data[cycle].clear()
        
        logger.info("SwingMomentumIndicator reset")
