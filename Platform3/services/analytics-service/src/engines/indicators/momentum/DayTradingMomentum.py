"""
Day Trading Momentum Indicators
Momentum indicators optimized for intraday trading (M15-H1 timeframes)

Features:
- Session-based momentum analysis
- Intraday trend strength assessment
- Breakout momentum detection
- Volume-weighted momentum calculations
- Multi-timeframe momentum confluence
- Session transition momentum tracking
- Real-time momentum strength classification
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

class TradingSession(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    QUIET = "quiet"

class MomentumPhase(Enum):
    ACCUMULATION = "accumulation"
    BREAKOUT = "breakout"
    CONTINUATION = "continuation"
    EXHAUSTION = "exhaustion"
    REVERSAL = "reversal"

class IntradaySignal(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class DayTradingMomentumResult:
    session_momentum: float
    trend_strength: float
    breakout_probability: float
    volume_momentum: float
    momentum_phase: MomentumPhase
    current_session: TradingSession
    signal: IntradaySignal
    confidence: float
    support_level: float
    resistance_level: float
    timestamp: datetime

@dataclass
class DayTradingConfig:
    rsi_period: int = 21
    momentum_period: int = 14
    volume_period: int = 20
    breakout_threshold: float = 0.7
    trend_strength_period: int = 10
    session_momentum_period: int = 30
    support_resistance_period: int = 50

class DayTradingMomentumIndicator:
    """
    Advanced momentum indicator for day trading (M15-H1 timeframes)
    """
    
    def __init__(self, config: Optional[DayTradingConfig] = None):
        self.config = config or DayTradingConfig()
        
        # Data buffers
        self.price_buffer = []
        self.volume_buffer = []
        self.high_buffer = []
        self.low_buffer = []
        self.timestamp_buffer = []
        
        # Session tracking
        self.session_data = {
            TradingSession.ASIAN: {'high': None, 'low': None, 'volume': 0},
            TradingSession.LONDON: {'high': None, 'low': None, 'volume': 0},
            TradingSession.NEW_YORK: {'high': None, 'low': None, 'volume': 0}
        }
        
        # Momentum tracking
        self.momentum_history = []
        self.breakout_levels = {'resistance': [], 'support': []}
        
        # Performance tracking
        self.calculation_count = 0
        self.signal_accuracy = 0.0
        self.last_calculation_time = None
        
        logger.info("DayTradingMomentumIndicator initialized")

    def calculate(self, prices: List[float], volumes: List[float], 
                 highs: List[float], lows: List[float],
                 timestamps: List[datetime]) -> DayTradingMomentumResult:
        """
        Calculate comprehensive day trading momentum indicators
        
        Args:
            prices: Close prices
            volumes: Volume data
            highs: High prices
            lows: Low prices
            timestamps: Timestamps for session analysis
            
        Returns:
            DayTradingMomentumResult with all momentum indicators
        """
        try:
            start_time = datetime.now()
            
            # Update buffers
            self._update_buffers(prices, volumes, highs, lows, timestamps)
            
            if len(self.price_buffer) < self.config.rsi_period:
                return self._default_result()
            
            # Determine current session
            current_session = self._determine_current_session(timestamps[-1])
            
            # Calculate core momentum indicators
            session_momentum = self._calculate_session_momentum(current_session)
            trend_strength = self._calculate_trend_strength()
            breakout_probability = self._calculate_breakout_probability()
            volume_momentum = self._calculate_volume_momentum()
            
            # Determine momentum phase
            momentum_phase = self._determine_momentum_phase(
                session_momentum, trend_strength, breakout_probability
            )
            
            # Calculate support and resistance levels
            support_level, resistance_level = self._calculate_support_resistance()
            
            # Generate trading signal
            signal = self._generate_intraday_signal(
                session_momentum, trend_strength, breakout_probability, 
                volume_momentum, current_session
            )
            
            # Calculate confidence
            confidence = self._calculate_signal_confidence(
                session_momentum, trend_strength, breakout_probability, 
                volume_momentum, current_session
            )
            
            result = DayTradingMomentumResult(
                session_momentum=session_momentum,
                trend_strength=trend_strength,
                breakout_probability=breakout_probability,
                volume_momentum=volume_momentum,
                momentum_phase=momentum_phase,
                current_session=current_session,
                signal=signal,
                confidence=confidence,
                support_level=support_level,
                resistance_level=resistance_level,
                timestamp=datetime.now()
            )
            
            # Update performance tracking
            self.calculation_count += 1
            self.last_calculation_time = datetime.now()
            calculation_time = (self.last_calculation_time - start_time).total_seconds() * 1000
            
            # Update session data
            self._update_session_data(current_session, highs[-1], lows[-1], volumes[-1])
            
            logger.debug(f"Day trading momentum calculated: {signal.value} "
                        f"(confidence: {confidence:.3f}, time: {calculation_time:.1f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating day trading momentum: {e}")
            return self._default_result()

    def _update_buffers(self, prices: List[float], volumes: List[float],
                       highs: List[float], lows: List[float], timestamps: List[datetime]):
        """Update internal data buffers"""
        # Extend buffers with new data
        if isinstance(prices, list):
            self.price_buffer.extend(prices)
            self.volume_buffer.extend(volumes)
            self.high_buffer.extend(highs)
            self.low_buffer.extend(lows)
            self.timestamp_buffer.extend(timestamps)
        else:
            self.price_buffer.append(prices)
            self.volume_buffer.append(volumes)
            self.high_buffer.append(highs)
            self.low_buffer.append(lows)
            self.timestamp_buffer.append(timestamps)
        
        # Maintain buffer size (keep last 200 values for efficiency)
        max_buffer_size = 200
        if len(self.price_buffer) > max_buffer_size:
            self.price_buffer = self.price_buffer[-max_buffer_size:]
            self.volume_buffer = self.volume_buffer[-max_buffer_size:]
            self.high_buffer = self.high_buffer[-max_buffer_size:]
            self.low_buffer = self.low_buffer[-max_buffer_size:]
            self.timestamp_buffer = self.timestamp_buffer[-max_buffer_size:]

    def _determine_current_session(self, timestamp: datetime) -> TradingSession:
        """Determine current trading session based on UTC time"""
        hour = timestamp.hour
        
        # Asian session: 00:00 - 08:00 UTC
        if 0 <= hour < 8:
            return TradingSession.ASIAN
        
        # London session: 08:00 - 16:00 UTC
        elif 8 <= hour < 16:
            return TradingSession.LONDON
        
        # London-NY overlap: 13:00 - 16:00 UTC
        elif 13 <= hour < 16:
            return TradingSession.OVERLAP_LONDON_NY
        
        # New York session: 13:00 - 21:00 UTC
        elif 13 <= hour < 21:
            return TradingSession.NEW_YORK
        
        # Quiet period
        else:
            return TradingSession.QUIET

    def _calculate_session_momentum(self, current_session: TradingSession) -> float:
        """Calculate momentum specific to current trading session"""
        try:
            if len(self.price_buffer) < self.config.session_momentum_period:
                return 0.0
            
            # Get session-specific data
            session_prices = self.price_buffer[-self.config.session_momentum_period:]
            session_volumes = self.volume_buffer[-self.config.session_momentum_period:]
            
            # Calculate price momentum
            price_change = (session_prices[-1] - session_prices[0]) / session_prices[0]
            
            # Calculate volume-weighted momentum
            avg_volume = np.mean(session_volumes)
            current_volume = session_volumes[-1]
            volume_factor = min(current_volume / avg_volume, 3.0) if avg_volume > 0 else 1.0
            
            # Session-specific adjustments
            session_multipliers = {
                TradingSession.ASIAN: 0.8,      # Lower volatility
                TradingSession.LONDON: 1.2,     # High volatility
                TradingSession.NEW_YORK: 1.1,   # High volatility
                TradingSession.OVERLAP_LONDON_NY: 1.3,  # Highest volatility
                TradingSession.QUIET: 0.6       # Very low volatility
            }
            
            session_multiplier = session_multipliers.get(current_session, 1.0)
            
            # Calculate final session momentum
            session_momentum = price_change * volume_factor * session_multiplier
            
            # Normalize to -1 to 1 range
            session_momentum = np.tanh(session_momentum * 100)
            
            return session_momentum
            
        except Exception as e:
            logger.error(f"Error calculating session momentum: {e}")
            return 0.0

    def _calculate_trend_strength(self) -> float:
        """Calculate intraday trend strength"""
        try:
            if len(self.price_buffer) < self.config.trend_strength_period:
                return 0.0
            
            prices = np.array(self.price_buffer[-self.config.trend_strength_period:])
            
            # Calculate linear regression slope
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            
            # Normalize slope relative to price level
            normalized_slope = slope / prices[-1] if prices[-1] != 0 else 0
            
            # Calculate R-squared for trend consistency
            y_pred = np.polyval([slope, prices[0]], x)
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Combine slope and consistency
            trend_strength = normalized_slope * r_squared * 1000
            
            # Normalize to -1 to 1 range
            trend_strength = np.tanh(trend_strength)
            
            return trend_strength
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0

    def _calculate_breakout_probability(self) -> float:
        """Calculate probability of breakout from current range"""
        try:
            if len(self.price_buffer) < self.config.support_resistance_period:
                return 0.0
            
            # Calculate recent range
            recent_highs = self.high_buffer[-self.config.support_resistance_period:]
            recent_lows = self.low_buffer[-self.config.support_resistance_period:]
            recent_volumes = self.volume_buffer[-self.config.support_resistance_period:]
            
            resistance = max(recent_highs)
            support = min(recent_lows)
            current_price = self.price_buffer[-1]
            
            # Calculate position within range
            if resistance != support:
                range_position = (current_price - support) / (resistance - support)
            else:
                range_position = 0.5
            
            # Calculate volume momentum
            avg_volume = np.mean(recent_volumes)
            current_volume = recent_volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate volatility
            price_changes = np.diff(self.price_buffer[-20:])
            volatility = np.std(price_changes) if len(price_changes) > 1 else 0
            
            # Breakout probability factors
            # Higher probability near resistance/support levels
            level_proximity = min(abs(range_position - 1), abs(range_position)) * 2
            
            # Higher probability with increased volume
            volume_factor = min(volume_ratio, 3.0) / 3.0
            
            # Higher probability with increased volatility
            volatility_factor = min(volatility * 1000, 1.0)
            
            # Combine factors
            breakout_probability = (level_proximity * 0.4 + volume_factor * 0.4 + volatility_factor * 0.2)
            
            return min(breakout_probability, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating breakout probability: {e}")
            return 0.0

    def _calculate_volume_momentum(self) -> float:
        """Calculate volume-based momentum"""
        try:
            if len(self.volume_buffer) < self.config.volume_period:
                return 0.0
            
            volumes = np.array(self.volume_buffer[-self.config.volume_period:])
            prices = np.array(self.price_buffer[-self.config.volume_period:])
            
            # Calculate volume-weighted price changes
            price_changes = np.diff(prices)
            volume_weights = volumes[1:]  # Align with price changes
            
            if len(price_changes) == 0 or np.sum(volume_weights) == 0:
                return 0.0
            
            # Volume-weighted momentum
            weighted_momentum = np.sum(price_changes * volume_weights) / np.sum(volume_weights)
            
            # Normalize relative to average price
            avg_price = np.mean(prices)
            normalized_momentum = weighted_momentum / avg_price if avg_price != 0 else 0
            
            # Scale and bound
            volume_momentum = np.tanh(normalized_momentum * 1000)
            
            return volume_momentum
            
        except Exception as e:
            logger.error(f"Error calculating volume momentum: {e}")
            return 0.0

    def _determine_momentum_phase(self, session_momentum: float, trend_strength: float, 
                                breakout_probability: float) -> MomentumPhase:
        """Determine current momentum phase"""
        # Strong trend with low breakout probability = continuation
        if abs(trend_strength) > 0.6 and breakout_probability < 0.3:
            return MomentumPhase.CONTINUATION
        
        # High breakout probability with building momentum = breakout
        elif breakout_probability > 0.7 and abs(session_momentum) > 0.4:
            return MomentumPhase.BREAKOUT
        
        # Strong momentum but weakening trend = exhaustion
        elif abs(session_momentum) > 0.6 and abs(trend_strength) < 0.3:
            return MomentumPhase.EXHAUSTION
        
        # Negative momentum with trend reversal = reversal
        elif session_momentum * trend_strength < -0.3:
            return MomentumPhase.REVERSAL
        
        # Default to accumulation
        else:
            return MomentumPhase.ACCUMULATION

    def _calculate_support_resistance(self) -> Tuple[float, float]:
        """Calculate dynamic support and resistance levels"""
        try:
            if len(self.price_buffer) < self.config.support_resistance_period:
                current_price = self.price_buffer[-1] if self.price_buffer else 0
                return current_price * 0.999, current_price * 1.001
            
            # Use recent highs and lows
            recent_highs = self.high_buffer[-self.config.support_resistance_period:]
            recent_lows = self.low_buffer[-self.config.support_resistance_period:]
            
            # Calculate pivot points
            resistance_levels = []
            support_levels = []
            
            # Find local maxima and minima
            for i in range(2, len(recent_highs) - 2):
                # Local maximum (resistance)
                if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                    recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                    resistance_levels.append(recent_highs[i])
                
                # Local minimum (support)
                if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                    recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                    support_levels.append(recent_lows[i])
            
            # Get nearest levels
            current_price = self.price_buffer[-1]
            
            if resistance_levels:
                resistance = min([r for r in resistance_levels if r > current_price], 
                               default=max(recent_highs))
            else:
                resistance = max(recent_highs)
            
            if support_levels:
                support = max([s for s in support_levels if s < current_price], 
                            default=min(recent_lows))
            else:
                support = min(recent_lows)
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            current_price = self.price_buffer[-1] if self.price_buffer else 0
            return current_price * 0.999, current_price * 1.001

    def _generate_intraday_signal(self, session_momentum: float, trend_strength: float,
                                breakout_probability: float, volume_momentum: float,
                                current_session: TradingSession) -> IntradaySignal:
        """Generate intraday trading signal"""
        # Strong bullish conditions
        if (session_momentum > 0.6 and trend_strength > 0.5 and 
            volume_momentum > 0.4 and breakout_probability > 0.6):
            return IntradaySignal.STRONG_BUY
        
        # Bullish conditions
        elif (session_momentum > 0.3 and trend_strength > 0.2 and volume_momentum > 0.2):
            return IntradaySignal.BUY
        
        # Strong bearish conditions
        elif (session_momentum < -0.6 and trend_strength < -0.5 and 
              volume_momentum < -0.4 and breakout_probability > 0.6):
            return IntradaySignal.STRONG_SELL
        
        # Bearish conditions
        elif (session_momentum < -0.3 and trend_strength < -0.2 and volume_momentum < -0.2):
            return IntradaySignal.SELL
        
        # Session-specific adjustments
        if current_session == TradingSession.QUIET:
            return IntradaySignal.HOLD  # Avoid trading in quiet sessions
        
        # Default to hold
        return IntradaySignal.HOLD

    def _calculate_signal_confidence(self, session_momentum: float, trend_strength: float,
                                   breakout_probability: float, volume_momentum: float,
                                   current_session: TradingSession) -> float:
        """Calculate confidence score for the signal"""
        # Base confidence on indicator alignment
        indicators = [session_momentum, trend_strength, volume_momentum]
        
        # Check for alignment
        positive_count = sum(1 for x in indicators if x > 0.1)
        negative_count = sum(1 for x in indicators if x < -0.1)
        
        alignment_ratio = max(positive_count, negative_count) / len(indicators)
        
        # Factor in breakout probability
        breakout_factor = breakout_probability
        
        # Session confidence multipliers
        session_multipliers = {
            TradingSession.ASIAN: 0.8,
            TradingSession.LONDON: 1.1,
            TradingSession.NEW_YORK: 1.0,
            TradingSession.OVERLAP_LONDON_NY: 1.2,
            TradingSession.QUIET: 0.5
        }
        
        session_multiplier = session_multipliers.get(current_session, 1.0)
        
        # Calculate final confidence
        base_confidence = alignment_ratio * 0.6 + breakout_factor * 0.4
        final_confidence = base_confidence * session_multiplier
        
        return min(max(final_confidence, 0.0), 1.0)

    def _update_session_data(self, session: TradingSession, high: float, low: float, volume: float):
        """Update session-specific data"""
        if session in self.session_data:
            session_info = self.session_data[session]
            
            # Update session high/low
            if session_info['high'] is None or high > session_info['high']:
                session_info['high'] = high
            
            if session_info['low'] is None or low < session_info['low']:
                session_info['low'] = low
            
            # Accumulate volume
            session_info['volume'] += volume

    def _default_result(self) -> DayTradingMomentumResult:
        """Return default result when calculation fails"""
        current_price = self.price_buffer[-1] if self.price_buffer else 0
        
        return DayTradingMomentumResult(
            session_momentum=0.0,
            trend_strength=0.0,
            breakout_probability=0.0,
            volume_momentum=0.0,
            momentum_phase=MomentumPhase.ACCUMULATION,
            current_session=TradingSession.QUIET,
            signal=IntradaySignal.HOLD,
            confidence=0.0,
            support_level=current_price * 0.999,
            resistance_level=current_price * 1.001,
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
                'high_buffer': len(self.high_buffer),
                'low_buffer': len(self.low_buffer)
            },
            'session_data': {
                session.value: {
                    'high': data['high'],
                    'low': data['low'],
                    'volume': data['volume']
                }
                for session, data in self.session_data.items()
            },
            'config': {
                'rsi_period': self.config.rsi_period,
                'momentum_period': self.config.momentum_period,
                'volume_period': self.config.volume_period,
                'breakout_threshold': self.config.breakout_threshold
            }
        }

    def reset(self):
        """Reset all buffers and session data"""
        self.price_buffer.clear()
        self.volume_buffer.clear()
        self.high_buffer.clear()
        self.low_buffer.clear()
        self.timestamp_buffer.clear()
        
        # Reset session data
        for session in self.session_data:
            self.session_data[session] = {'high': None, 'low': None, 'volume': 0}
        
        self.momentum_history.clear()
        self.breakout_levels = {'resistance': [], 'support': []}
        
        logger.info("DayTradingMomentumIndicator reset")
