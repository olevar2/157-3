"""
Scalping Momentum Indicators
Ultra-fast momentum indicators optimized for M1-M5 scalping strategies

Features:
- Ultra-fast RSI calculation for scalping
- Micro-momentum detection for tick-level analysis
- Velocity-based momentum scoring
- Acceleration indicators for trend changes
- Session-aware momentum adjustments
- Real-time momentum strength classification
- Scalping-specific signal generation
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

class MomentumStrength(Enum):
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    NEUTRAL = "neutral"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class MomentumDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"

class ScalpingSignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class MomentumResult:
    rsi: float
    micro_momentum: float
    velocity: float
    acceleration: float
    strength: MomentumStrength
    direction: MomentumDirection
    signal: ScalpingSignal
    confidence: float
    timestamp: datetime

@dataclass
class ScalpingMomentumConfig:
    rsi_period: int = 14
    micro_period: int = 5
    velocity_period: int = 3
    acceleration_period: int = 2
    overbought_level: float = 70.0
    oversold_level: float = 30.0
    strong_momentum_threshold: float = 0.8
    weak_momentum_threshold: float = 0.3

class ScalpingMomentum:
    """
    Advanced momentum indicator suite optimized for scalping (M1-M5)
    """

    def __init__(self, config: Optional[ScalpingMomentumConfig] = None):
        self.config = config or ScalpingMomentumConfig()

        # Data buffers for calculations
        self.price_buffer = []
        self.volume_buffer = []
        self.rsi_buffer = []
        self.momentum_buffer = []

        # Calculation caches
        self.last_rsi = None
        self.last_momentum = None
        self.last_velocity = None
        self.last_acceleration = None

        # Performance tracking
        self.calculation_count = 0
        self.signal_accuracy = 0.0
        self.last_calculation_time = None

        logger.info("ScalpingMomentum initialized")

    def calculate(self, prices: List[float], volumes: Optional[List[float]] = None,
                 timestamps: Optional[List[datetime]] = None) -> MomentumResult:
        """
        Calculate comprehensive momentum indicators for scalping

        Args:
            prices: List of price values (OHLC or close prices)
            volumes: Optional volume data for volume-weighted calculations
            timestamps: Optional timestamps for session-aware adjustments

        Returns:
            MomentumResult with all momentum indicators
        """
        try:
            start_time = datetime.now()

            # Update buffers
            self._update_buffers(prices, volumes)

            if len(self.price_buffer) < self.config.rsi_period:
                return self._default_result()

            # Calculate core momentum indicators
            rsi = self._calculate_ultra_fast_rsi()
            micro_momentum = self._calculate_micro_momentum()
            velocity = self._calculate_momentum_velocity()
            acceleration = self._calculate_momentum_acceleration()

            # Determine momentum characteristics
            strength = self._classify_momentum_strength(rsi, micro_momentum, velocity)
            direction = self._determine_momentum_direction(rsi, micro_momentum, velocity)
            signal = self._generate_scalping_signal(rsi, micro_momentum, velocity, acceleration)
            confidence = self._calculate_signal_confidence(rsi, micro_momentum, velocity, acceleration)

            # Apply session-aware adjustments if timestamps provided
            if timestamps and len(timestamps) > 0:
                signal, confidence = self._apply_session_adjustments(
                    signal, confidence, timestamps[-1]
                )

            result = MomentumResult(
                rsi=rsi,
                micro_momentum=micro_momentum,
                velocity=velocity,
                acceleration=acceleration,
                strength=strength,
                direction=direction,
                signal=signal,
                confidence=confidence,
                timestamp=datetime.now()
            )

            # Update performance tracking
            self.calculation_count += 1
            self.last_calculation_time = datetime.now()
            calculation_time = (self.last_calculation_time - start_time).total_seconds() * 1000

            if calculation_time > 1.0:  # Log if calculation takes > 1ms
                logger.warning(f"Slow momentum calculation: {calculation_time:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"❌ Error calculating scalping momentum: {e}")
            return self._default_result()

    def _update_buffers(self, prices: List[float], volumes: Optional[List[float]] = None):
        """Update internal data buffers"""
        # Add new prices to buffer
        if isinstance(prices, list) and len(prices) > 0:
            self.price_buffer.extend(prices)
        elif isinstance(prices, (int, float)):
            self.price_buffer.append(prices)

        # Add volumes if provided
        if volumes:
            if isinstance(volumes, list):
                self.volume_buffer.extend(volumes)
            else:
                self.volume_buffer.append(volumes)

        # Maintain buffer size (keep last 100 values for efficiency)
        max_buffer_size = 100
        if len(self.price_buffer) > max_buffer_size:
            self.price_buffer = self.price_buffer[-max_buffer_size:]
        if len(self.volume_buffer) > max_buffer_size:
            self.volume_buffer = self.volume_buffer[-max_buffer_size:]

    def _calculate_ultra_fast_rsi(self) -> float:
        """Calculate ultra-fast RSI optimized for scalping"""
        try:
            if len(self.price_buffer) < self.config.rsi_period + 1:
                return 50.0  # Neutral RSI

            # Get price changes
            prices = np.array(self.price_buffer[-self.config.rsi_period-1:])
            price_changes = np.diff(prices)

            # Separate gains and losses
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)

            # Calculate average gains and losses using exponential smoothing for speed
            alpha = 2.0 / (self.config.rsi_period + 1)

            if self.last_rsi is None:
                # Initial calculation
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
            else:
                # Exponential smoothing update
                current_gain = gains[-1] if len(gains) > 0 else 0
                current_loss = losses[-1] if len(losses) > 0 else 0

                # Get previous averages from last calculation
                prev_rs = (100 - self.last_rsi) / self.last_rsi if self.last_rsi != 0 else 1
                prev_avg_gain = prev_rs / (1 + prev_rs) if prev_rs != -1 else 0.5
                prev_avg_loss = 1 / (1 + prev_rs) if prev_rs != -1 else 0.5

                avg_gain = alpha * current_gain + (1 - alpha) * prev_avg_gain
                avg_loss = alpha * current_loss + (1 - alpha) * prev_avg_loss

            # Calculate RSI
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            self.last_rsi = rsi
            return rsi

        except Exception as e:
            logger.error(f"Error calculating ultra-fast RSI: {e}")
            return 50.0

    def _calculate_micro_momentum(self) -> float:
        """Calculate micro-momentum for tick-level analysis"""
        try:
            if len(self.price_buffer) < self.config.micro_period + 1:
                return 0.0

            # Get recent prices
            recent_prices = np.array(self.price_buffer[-self.config.micro_period-1:])

            # Calculate price changes
            price_changes = np.diff(recent_prices)

            # Weight recent changes more heavily
            weights = np.exp(np.linspace(0, 1, len(price_changes)))
            weighted_changes = price_changes * weights

            # Calculate micro-momentum as weighted average of recent changes
            micro_momentum = np.sum(weighted_changes) / np.sum(weights)

            # Normalize to -1 to 1 range
            if len(self.price_buffer) > 20:
                recent_volatility = np.std(self.price_buffer[-20:])
                if recent_volatility > 0:
                    micro_momentum = np.tanh(micro_momentum / recent_volatility)

            return micro_momentum

        except Exception as e:
            logger.error(f"Error calculating micro-momentum: {e}")
            return 0.0

    def _calculate_momentum_velocity(self) -> float:
        """Calculate momentum velocity (rate of momentum change)"""
        try:
            if len(self.momentum_buffer) < self.config.velocity_period:
                # Calculate current momentum and add to buffer
                current_momentum = self._calculate_current_momentum()
                self.momentum_buffer.append(current_momentum)
                return 0.0

            # Calculate velocity as rate of change of momentum
            recent_momentum = np.array(self.momentum_buffer[-self.config.velocity_period:])
            velocity = np.gradient(recent_momentum)[-1]  # Latest velocity

            return velocity

        except Exception as e:
            logger.error(f"Error calculating momentum velocity: {e}")
            return 0.0

    def _calculate_momentum_acceleration(self) -> float:
        """Calculate momentum acceleration (rate of velocity change)"""
        try:
            if len(self.price_buffer) < self.config.acceleration_period + 2:
                return 0.0

            # Calculate recent velocities
            velocities = []
            for i in range(self.config.acceleration_period):
                if len(self.price_buffer) >= i + 3:
                    prices = self.price_buffer[-(i+3):-(i) if i > 0 else None]
                    if len(prices) >= 3:
                        velocity = (prices[-1] - prices[0]) / len(prices)
                        velocities.append(velocity)

            if len(velocities) < 2:
                return 0.0

            # Calculate acceleration as change in velocity
            acceleration = velocities[0] - velocities[-1]

            return acceleration

        except Exception as e:
            logger.error(f"Error calculating momentum acceleration: {e}")
            return 0.0

    def _calculate_current_momentum(self) -> float:
        """Calculate current momentum value"""
        if len(self.price_buffer) < 3:
            return 0.0

        # Simple momentum calculation
        return self.price_buffer[-1] - self.price_buffer[-3]

    def _classify_momentum_strength(self, rsi: float, micro_momentum: float, velocity: float) -> MomentumStrength:
        """Classify overall momentum strength"""
        # Combine indicators for strength assessment
        rsi_strength = abs(rsi - 50) / 50  # 0 to 1
        micro_strength = abs(micro_momentum)
        velocity_strength = abs(velocity)

        # Weighted combination
        combined_strength = (rsi_strength * 0.4 + micro_strength * 0.4 + velocity_strength * 0.2)

        if combined_strength > self.config.strong_momentum_threshold:
            return MomentumStrength.VERY_STRONG
        elif combined_strength > 0.6:
            return MomentumStrength.STRONG
        elif combined_strength > self.config.weak_momentum_threshold:
            return MomentumStrength.NEUTRAL
        elif combined_strength > 0.1:
            return MomentumStrength.WEAK
        else:
            return MomentumStrength.VERY_WEAK

    def _determine_momentum_direction(self, rsi: float, micro_momentum: float, velocity: float) -> MomentumDirection:
        """Determine momentum direction"""
        # Combine directional signals
        rsi_direction = 1 if rsi > 50 else -1
        micro_direction = 1 if micro_momentum > 0 else -1
        velocity_direction = 1 if velocity > 0 else -1

        # Weighted direction score
        direction_score = (rsi_direction * 0.4 + micro_direction * 0.4 + velocity_direction * 0.2)

        if direction_score > 0.3:
            return MomentumDirection.BULLISH
        elif direction_score < -0.3:
            return MomentumDirection.BEARISH
        else:
            return MomentumDirection.SIDEWAYS

    def _generate_scalping_signal(self, rsi: float, micro_momentum: float,
                                velocity: float, acceleration: float) -> ScalpingSignal:
        """Generate scalping-specific trading signal"""
        # Strong buy conditions
        if (rsi < self.config.oversold_level and micro_momentum > 0.5 and
            velocity > 0 and acceleration > 0):
            return ScalpingSignal.STRONG_BUY

        # Buy conditions
        if (rsi < 45 and micro_momentum > 0.2 and velocity > 0):
            return ScalpingSignal.BUY

        # Strong sell conditions
        if (rsi > self.config.overbought_level and micro_momentum < -0.5 and
            velocity < 0 and acceleration < 0):
            return ScalpingSignal.STRONG_SELL

        # Sell conditions
        if (rsi > 55 and micro_momentum < -0.2 and velocity < 0):
            return ScalpingSignal.SELL

        # Default to hold
        return ScalpingSignal.HOLD

    def _calculate_signal_confidence(self, rsi: float, micro_momentum: float,
                                   velocity: float, acceleration: float) -> float:
        """Calculate confidence score for the signal"""
        # Base confidence on indicator alignment
        indicators = [rsi - 50, micro_momentum * 50, velocity * 100, acceleration * 100]

        # Check for alignment (all positive or all negative)
        positive_count = sum(1 for x in indicators if x > 0)
        negative_count = sum(1 for x in indicators if x < 0)

        alignment_ratio = max(positive_count, negative_count) / len(indicators)

        # Base confidence on alignment and strength
        base_confidence = alignment_ratio * 0.7

        # Add strength bonus
        strength_bonus = min(abs(rsi - 50) / 50, 1.0) * 0.3

        confidence = min(base_confidence + strength_bonus, 1.0)

        return confidence

    def _apply_session_adjustments(self, signal: ScalpingSignal, confidence: float,
                                 timestamp: datetime) -> Tuple[ScalpingSignal, float]:
        """Apply session-aware adjustments to signals"""
        hour = timestamp.hour

        # London session (8-16 UTC) - high volatility, boost confidence
        if 8 <= hour < 16:
            confidence = min(confidence * 1.1, 1.0)

        # New York session (13-21 UTC) - high volatility, boost confidence
        elif 13 <= hour < 21:
            confidence = min(confidence * 1.1, 1.0)

        # Asian session (0-8 UTC) - lower volatility, reduce confidence
        elif 0 <= hour < 8:
            confidence = confidence * 0.9

        # Overlap periods - highest volatility, maximum boost
        elif (8 <= hour < 13) or (16 <= hour < 21):
            confidence = min(confidence * 1.2, 1.0)

        return signal, confidence

    def _default_result(self) -> MomentumResult:
        """Return default result when calculation fails"""
        return MomentumResult(
            rsi=50.0,
            micro_momentum=0.0,
            velocity=0.0,
            acceleration=0.0,
            strength=MomentumStrength.NEUTRAL,
            direction=MomentumDirection.SIDEWAYS,
            signal=ScalpingSignal.HOLD,
            confidence=0.0,
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
                'volume_buffer': len(self.volume_buffer),
                'momentum_buffer': len(self.momentum_buffer)
            },
            'config': {
                'rsi_period': self.config.rsi_period,
                'micro_period': self.config.micro_period,
                'velocity_period': self.config.velocity_period,
                'acceleration_period': self.config.acceleration_period
            }
        }

    def reset(self):
        """Reset all buffers and cached values"""
        self.price_buffer.clear()
        self.volume_buffer.clear()
        self.rsi_buffer.clear()
        self.momentum_buffer.clear()

        self.last_rsi = None
        self.last_momentum = None
        self.last_velocity = None
        self.last_acceleration = None

        logger.info("ScalpingMomentum reset")
