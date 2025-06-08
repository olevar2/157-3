"""
Keltner Channels - Advanced Volatility-Based Trend Indicator
============================================================

A sophisticated implementation of Keltner Channels that uses Average True Range (ATR)
to create dynamic support and resistance levels around a moving average. Unlike 
Bollinger Bands which use standard deviation, Keltner Channels use ATR for more
responsive volatility-based channel construction.

Key Features:
- ATR-based volatility channels
- Multiple moving average types (EMA, SMA, WMA)
- Adaptive channel width based on market conditions
- Breakout and squeeze detection
- Trend direction confirmation
- Support/resistance level identification
- Channel expansion/contraction analysis
- Multi-timeframe channel coordination

Humanitarian Use:
Provides precise volatility-based entry and exit signals for maximum profit
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

class KeltnerChannelState(Enum):
    """Keltner Channel states"""
    EXPANDING = "expanding"
    CONTRACTING = "contracting"
    SQUEEZE = "squeeze"
    BREAKOUT_UPPER = "breakout_upper"
    BREAKOUT_LOWER = "breakout_lower"
    RANGING = "ranging"

class KeltnerTrendDirection(Enum):
    """Keltner trend directions"""
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    SIDEWAYS = "sideways"
    TREND_REVERSAL = "trend_reversal"

class MovingAverageType(Enum):
    """Moving average types for Keltner Channels"""
    EMA = "ema"
    SMA = "sma"
    WMA = "wma"
    DEMA = "dema"

@dataclass
class KeltnerConfig(IndicatorConfig):
    """Configuration for Keltner Channels"""
    # Core channel settings
    period: int = 20
    atr_period: int = 10
    atr_multiplier: float = 2.0
    ma_type: MovingAverageType = MovingAverageType.EMA
    
    # Channel analysis settings
    squeeze_threshold: float = 0.7  # Channel width ratio for squeeze detection
    expansion_threshold: float = 1.5  # Channel width ratio for expansion
    breakout_confirmation_bars: int = 2
    
    # Trend analysis settings
    trend_confirmation_period: int = 5
    sideways_threshold: float = 0.5  # Price range as % of channel width
    
    # Advanced features
    adaptive_multiplier: bool = True
    volatility_adjustment: bool = True
    dynamic_period: bool = False
    multi_timeframe_sync: bool = True

@dataclass
class KeltnerChannelData:
    """Keltner Channel data structure"""
    upper_band: float
    middle_line: float
    lower_band: float
    atr_value: float
    channel_width: float
    price_position: float  # Where price is within the channel (0-1)

@dataclass
class KeltnerAnalysis:
    """Comprehensive Keltner Channel analysis"""
    channel_state: KeltnerChannelState
    trend_direction: KeltnerTrendDirection
    channel_data: KeltnerChannelData
    squeeze_intensity: float
    breakout_strength: float
    trend_strength: float
    support_resistance_levels: Dict[str, float]
    volatility_regime: str  # "low", "normal", "high"

class KeltnerChannels(IndicatorBase[KeltnerConfig]):
    """
    Advanced Keltner Channels implementation
    
    Keltner Channels consist of:
    - Middle Line: Moving Average (typically EMA)
    - Upper Band: Middle Line + (ATR × Multiplier)
    - Lower Band: Middle Line - (ATR × Multiplier)
    
    The channels adapt to volatility changes through ATR, providing
    dynamic support and resistance levels for trend analysis.
    """
    
    def __init__(self, config: KeltnerConfig):
        super().__init__(config)
        self.upper_bands: List[float] = []
        self.middle_lines: List[float] = []
        self.lower_bands: List[float] = []
        self.atr_values: List[float] = []
        self.true_ranges: List[float] = []
        self.channel_widths: List[float] = []
        
        # Pattern tracking
        self.squeeze_periods: List[Tuple[int, float]] = []
        self.breakout_points: List[Tuple[int, str, float]] = []
        self.trend_changes: List[Tuple[int, str]] = []
        
        # Analysis state
        self.current_trend: KeltnerTrendDirection = KeltnerTrendDirection.SIDEWAYS
        self.squeeze_duration: int = 0
        self.expansion_duration: int = 0
        self.adaptive_atr_multiplier: float = config.atr_multiplier
        
    def _calculate_true_range(self, high: float, low: float, prev_close: float) -> float:
        """Calculate True Range for ATR"""
        tr1 = high - low
        tr2 = abs(high - prev_close) if prev_close > 0 else 0
        tr3 = abs(low - prev_close) if prev_close > 0 else 0
        
        return max(tr1, tr2, tr3)
    
    def _calculate_atr(self, true_range: float) -> float:
        """Calculate Average True Range"""
        self.true_ranges.append(true_range)
        
        if len(self.true_ranges) < self.config.atr_period:
            return np.mean(self.true_ranges)
        
        # Wilder's smoothing method for ATR
        if self.atr_values:
            prev_atr = self.atr_values[-1]
            atr = ((prev_atr * (self.config.atr_period - 1)) + true_range) / self.config.atr_period
        else:
            atr = np.mean(self.true_ranges[-self.config.atr_period:])
        
        return atr
    
    def _calculate_moving_average(self, price: float) -> float:
        """Calculate moving average based on configured type"""
        if self.config.ma_type == MovingAverageType.SMA:
            return self._calculate_sma(price)
        elif self.config.ma_type == MovingAverageType.EMA:
            return self._calculate_ema(price)
        elif self.config.ma_type == MovingAverageType.WMA:
            return self._calculate_wma(price)
        elif self.config.ma_type == MovingAverageType.DEMA:
            return self._calculate_dema(price)
        else:
            return self._calculate_ema(price)  # Default to EMA
    
    def _calculate_sma(self, price: float) -> float:
        """Calculate Simple Moving Average"""
        self.prices.append(price)
        
        if len(self.prices) < self.config.period:
            return np.mean(self.prices)
        
        return np.mean(self.prices[-self.config.period:])
    
    def _calculate_ema(self, price: float) -> float:
        """Calculate Exponential Moving Average"""
        if not self.middle_lines:
            return price
        
        alpha = 2 / (self.config.period + 1)
        return alpha * price + (1 - alpha) * self.middle_lines[-1]
    
    def _calculate_wma(self, price: float) -> float:
        """Calculate Weighted Moving Average"""
        self.prices.append(price)
        
        if len(self.prices) < self.config.period:
            return np.mean(self.prices)
        
        weights = np.arange(1, self.config.period + 1)
        values = self.prices[-self.config.period:]
        return np.average(values, weights=weights)
    
    def _calculate_dema(self, price: float) -> float:
        """Calculate Double Exponential Moving Average"""
        # This is a simplified DEMA implementation
        ema1 = self._calculate_ema(price)
        
        if not hasattr(self, 'ema2_values'):
            self.ema2_values = []
        
        if not self.ema2_values:
            ema2 = ema1
        else:
            alpha = 2 / (self.config.period + 1)
            ema2 = alpha * ema1 + (1 - alpha) * self.ema2_values[-1]
        
        self.ema2_values.append(ema2)
        return 2 * ema1 - ema2
    
    def _update_adaptive_multiplier(self, volatility_regime: str):
        """Update ATR multiplier based on market conditions"""
        if not self.config.adaptive_multiplier:
            return
        
        base_multiplier = self.config.atr_multiplier
        
        if volatility_regime == "low":
            self.adaptive_atr_multiplier = base_multiplier * 0.8
        elif volatility_regime == "high":
            self.adaptive_atr_multiplier = base_multiplier * 1.2
        else:
            self.adaptive_atr_multiplier = base_multiplier
    
    def _determine_volatility_regime(self, atr_value: float) -> str:
        """Determine current volatility regime"""
        if len(self.atr_values) < 20:
            return "normal"
        
        recent_atr_mean = np.mean(self.atr_values[-20:])
        recent_atr_std = np.std(self.atr_values[-20:])
        
        if atr_value < recent_atr_mean - recent_atr_std:
            return "low"
        elif atr_value > recent_atr_mean + recent_atr_std:
            return "high"
        else:
            return "normal"
    
    def _detect_channel_state(self, channel_width: float) -> KeltnerChannelState:
        """Detect current channel state"""
        if len(self.channel_widths) < 10:
            return KeltnerChannelState.RANGING
        
        avg_width = np.mean(self.channel_widths[-10:])
        width_ratio = channel_width / avg_width if avg_width > 0 else 1.0
        
        # Check for squeeze
        if width_ratio < self.config.squeeze_threshold:
            return KeltnerChannelState.SQUEEZE
        
        # Check for expansion
        elif width_ratio > self.config.expansion_threshold:
            return KeltnerChannelState.EXPANDING
        
        # Check for contraction
        elif len(self.channel_widths) >= 3:
            recent_widths = self.channel_widths[-3:]
            if all(recent_widths[i] > recent_widths[i+1] for i in range(len(recent_widths)-1)):
                return KeltnerChannelState.CONTRACTING
        
        return KeltnerChannelState.RANGING
    
    def _detect_breakout(self, price: float, upper_band: float, lower_band: float) -> Optional[str]:
        """Detect channel breakouts"""
        if price > upper_band:
            return "upper"
        elif price < lower_band:
            return "lower"
        return None
    
    def _analyze_trend_direction(self, middle_line: float, price_position: float) -> KeltnerTrendDirection:
        """Analyze trend direction based on channel position and middle line slope"""
        if len(self.middle_lines) < self.config.trend_confirmation_period:
            return KeltnerTrendDirection.SIDEWAYS
        
        # Calculate middle line slope
        recent_middle = self.middle_lines[-self.config.trend_confirmation_period:]
        slope = np.polyfit(range(len(recent_middle)), recent_middle, 1)[0]
        
        # Analyze price position within channel
        if slope > 0 and price_position > 0.6:
            return KeltnerTrendDirection.BULLISH_TREND
        elif slope < 0 and price_position < 0.4:
            return KeltnerTrendDirection.BEARISH_TREND
        elif abs(slope) < self.config.sideways_threshold:
            return KeltnerTrendDirection.SIDEWAYS
        else:
            # Check for potential trend reversal
            if self.current_trend != KeltnerTrendDirection.SIDEWAYS:
                return KeltnerTrendDirection.TREND_REVERSAL
            return KeltnerTrendDirection.SIDEWAYS
    
    def _calculate_price_position(self, price: float, upper_band: float, 
                                lower_band: float) -> float:
        """Calculate price position within the channel (0 = lower band, 1 = upper band)"""
        if upper_band == lower_band:
            return 0.5
        
        position = (price - lower_band) / (upper_band - lower_band)
        return max(0.0, min(1.0, position))
    
    def _calculate_support_resistance_levels(self, channel_data: KeltnerChannelData) -> Dict[str, float]:
        """Calculate dynamic support and resistance levels"""
        levels = {
            'primary_resistance': channel_data.upper_band,
            'primary_support': channel_data.lower_band,
            'middle_line': channel_data.middle_line
        }
        
        # Add quarter levels for fine-grained analysis
        quarter_range = (channel_data.upper_band - channel_data.lower_band) / 4
        levels['upper_quarter'] = channel_data.middle_line + quarter_range
        levels['lower_quarter'] = channel_data.middle_line - quarter_range
        
        return levels
    
    def update(self, data: Dict[str, float]) -> Optional[IndicatorSignal]:
        """Update Keltner Channels with new market data"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            prev_close = self.prices[-1] if self.prices else close
            true_range = self._calculate_true_range(high, low, prev_close)
            
            # Calculate ATR
            atr_value = self._calculate_atr(true_range)
            self.atr_values.append(atr_value)
            
            # Determine volatility regime and update adaptive multiplier
            volatility_regime = self._determine_volatility_regime(atr_value)
            self._update_adaptive_multiplier(volatility_regime)
            
            # Calculate middle line (moving average)
            middle_line = self._calculate_moving_average(close)
            self.middle_lines.append(middle_line)
            
            # Calculate channel bands
            upper_band = middle_line + (atr_value * self.adaptive_atr_multiplier)
            lower_band = middle_line - (atr_value * self.adaptive_atr_multiplier)
            
            self.upper_bands.append(upper_band)
            self.lower_bands.append(lower_band)
            
            # Calculate channel width
            channel_width = upper_band - lower_band
            self.channel_widths.append(channel_width)
            
            # Calculate price position within channel
            price_position = self._calculate_price_position(close, upper_band, lower_band)
            
            # Create channel data
            channel_data = KeltnerChannelData(
                upper_band=upper_band,
                middle_line=middle_line,
                lower_band=lower_band,
                atr_value=atr_value,
                channel_width=channel_width,
                price_position=price_position
            )
            
            # Detect channel state
            channel_state = self._detect_channel_state(channel_width)
            
            # Detect breakouts
            breakout_direction = self._detect_breakout(close, upper_band, lower_band)
            if breakout_direction:
                self.breakout_points.append((len(self.prices), breakout_direction, close))
            
            # Analyze trend direction
            trend_direction = self._analyze_trend_direction(middle_line, price_position)
            if trend_direction != self.current_trend:
                self.trend_changes.append((len(self.prices), trend_direction.value))
                self.current_trend = trend_direction
            
            # Calculate analysis metrics
            squeeze_intensity = 0.0
            if channel_state == KeltnerChannelState.SQUEEZE:
                self.squeeze_duration += 1
                avg_width = np.mean(self.channel_widths[-20:]) if len(self.channel_widths) >= 20 else channel_width
                squeeze_intensity = 1.0 - (channel_width / avg_width) if avg_width > 0 else 0.0
            else:
                self.squeeze_duration = 0
            
            # Calculate breakout strength
            breakout_strength = 0.0
            if breakout_direction:
                if breakout_direction == "upper":
                    breakout_strength = (close - upper_band) / atr_value if atr_value > 0 else 0
                else:
                    breakout_strength = (lower_band - close) / atr_value if atr_value > 0 else 0
                breakout_strength = min(breakout_strength, 2.0)  # Cap at 2 ATR
            
            # Calculate trend strength
            trend_strength = 0.0
            if len(self.middle_lines) >= 5:
                recent_middle = self.middle_lines[-5:]
                slope = abs(np.polyfit(range(len(recent_middle)), recent_middle, 1)[0])
                trend_strength = min(slope / atr_value, 1.0) if atr_value > 0 else 0
            
            # Calculate support/resistance levels
            support_resistance_levels = self._calculate_support_resistance_levels(channel_data)
            
            # Create analysis
            analysis = KeltnerAnalysis(
                channel_state=channel_state,
                trend_direction=trend_direction,
                channel_data=channel_data,
                squeeze_intensity=squeeze_intensity,
                breakout_strength=breakout_strength,
                trend_strength=trend_strength,
                support_resistance_levels=support_resistance_levels,
                volatility_regime=volatility_regime
            )
            
            # Store prices for next iteration
            self.prices.append(close)
            
            # Generate signals
            return self._generate_signals(analysis, data)
            
        except Exception as e:
            logger.error(f"Error updating Keltner Channels: {e}")
            return None
    
    def _generate_signals(self, analysis: KeltnerAnalysis, data: Dict[str, float]) -> IndicatorSignal:
        """Generate trading signals based on Keltner Channel analysis"""
        signals = []
        signal_strength = SignalStrength.NEUTRAL
        confidence = 0.5
        
        # Squeeze signals (potential breakout setup)
        if analysis.channel_state == KeltnerChannelState.SQUEEZE:
            signals.append(f"Keltner Channel squeeze detected - breakout imminent")
            signal_strength = SignalStrength.MEDIUM
            confidence += 0.1 + (analysis.squeeze_intensity * 0.1)
        
        # Breakout signals
        if analysis.breakout_strength > 0.5:
            direction = "bullish" if data['close'] > analysis.channel_data.upper_band else "bearish"
            signals.append(f"Strong {direction} breakout from Keltner Channel")
            signal_strength = SignalStrength.STRONG
            confidence += 0.2 + min(analysis.breakout_strength * 0.1, 0.15)
        
        # Trend direction signals
        if analysis.trend_direction == KeltnerTrendDirection.BULLISH_TREND:
            signals.append("Keltner Channels confirm bullish trend")
            signal_strength = SignalStrength.MEDIUM
            confidence += 0.1 + (analysis.trend_strength * 0.1)
        elif analysis.trend_direction == KeltnerTrendDirection.BEARISH_TREND:
            signals.append("Keltner Channels confirm bearish trend")
            signal_strength = SignalStrength.MEDIUM
            confidence += 0.1 + (analysis.trend_strength * 0.1)
        elif analysis.trend_direction == KeltnerTrendDirection.TREND_REVERSAL:
            signals.append("Potential trend reversal detected by Keltner Channels")
            signal_strength = SignalStrength.STRONG
            confidence += 0.15
        
        # Channel position signals
        if analysis.channel_data.price_position > 0.9:
            signals.append("Price at upper Keltner Channel - potential resistance")
            confidence += 0.1
        elif analysis.channel_data.price_position < 0.1:
            signals.append("Price at lower Keltner Channel - potential support")
            confidence += 0.1
        elif 0.4 <= analysis.channel_data.price_position <= 0.6:
            signals.append("Price near Keltner middle line - neutral zone")
            confidence += 0.05
        
        # Channel expansion/contraction signals
        if analysis.channel_state == KeltnerChannelState.EXPANDING:
            signals.append("Keltner Channels expanding - increased volatility")
            confidence += 0.08
        elif analysis.channel_state == KeltnerChannelState.CONTRACTING:
            signals.append("Keltner Channels contracting - volatility decreasing")
            confidence += 0.05
        
        # Volatility regime signals
        if analysis.volatility_regime == "high":
            signals.append("High volatility regime - wider channels active")
            confidence += 0.05
        elif analysis.volatility_regime == "low":
            signals.append("Low volatility regime - tighter channels active")
            confidence += 0.05
        
        # Market condition assessment
        market_condition = MarketCondition.TRENDING
        if analysis.channel_state == KeltnerChannelState.SQUEEZE:
            market_condition = MarketCondition.RANGING
        elif analysis.breakout_strength > 0.3:
            market_condition = MarketCondition.VOLATILE
        elif analysis.trend_direction == KeltnerTrendDirection.SIDEWAYS:
            market_condition = MarketCondition.RANGING
        
        return IndicatorSignal(
            indicator_name=self.name,
            signal_strength=signal_strength,
            confidence=min(confidence, 0.95),
            signals=signals,
            market_condition=market_condition,
            metadata={
                'upper_band': analysis.channel_data.upper_band,
                'middle_line': analysis.channel_data.middle_line,
                'lower_band': analysis.channel_data.lower_band,
                'atr_value': analysis.channel_data.atr_value,
                'channel_width': analysis.channel_data.channel_width,
                'price_position': analysis.channel_data.price_position,
                'channel_state': analysis.channel_state.value,
                'trend_direction': analysis.trend_direction.value,
                'squeeze_intensity': analysis.squeeze_intensity,
                'breakout_strength': analysis.breakout_strength,
                'trend_strength': analysis.trend_strength,
                'volatility_regime': analysis.volatility_regime,
                'squeeze_duration': self.squeeze_duration,
                'adaptive_multiplier': self.adaptive_atr_multiplier,
                'support_resistance': analysis.support_resistance_levels
            }
        )
    
    @property
    def name(self) -> str:
        return "Keltner Channels"
    
    def get_channel_levels(self) -> Dict[str, float]:
        """Get current channel levels"""
        if not self.upper_bands:
            return {}
        
        return {
            'upper_band': self.upper_bands[-1],
            'middle_line': self.middle_lines[-1],
            'lower_band': self.lower_bands[-1],
            'atr': self.atr_values[-1],
            'channel_width': self.channel_widths[-1],
            'adaptive_multiplier': self.adaptive_atr_multiplier
        }

def test_keltner_channels():
    """Test Keltner Channels implementation with realistic market data"""
    config = KeltnerConfig(
        period=20,
        atr_period=10,
        atr_multiplier=2.0,
        ma_type=MovingAverageType.EMA,
        adaptive_multiplier=True,
        volatility_adjustment=True
    )
    
    keltner = KeltnerChannels(config)
    
    # Generate test data with volatility patterns
    np.random.seed(42)
    base_price = 1.2000
    
    signals = []
    for i in range(100):
        # Create price movement with volatility clustering
        volatility = 0.0005 if i < 30 else (0.0015 if i < 60 else 0.0008)
        price_change = np.random.normal(0, volatility)
        
        # Add trend component
        if i > 40 and i < 80:
            price_change += 0.0002  # Uptrend
        
        price = base_price + price_change
        high = price * (1 + abs(np.random.normal(0, 0.0003)))
        low = price * (1 - abs(np.random.normal(0, 0.0003)))
        
        data = {
            'open': price * 0.9999,
            'high': high,
            'low': low,
            'close': price,
            'volume': 1000000
        }
        
        signal = keltner.update(data)
        if signal and signal.signals:
            signals.append((i, signal))
        
        base_price = price
    
    print(f"Keltner Channels Test Results:")
    print(f"Total signals generated: {len(signals)}")
    print(f"Squeeze periods detected: {len(keltner.squeeze_periods)}")
    print(f"Breakout points detected: {len(keltner.breakout_points)}")
    print(f"Trend changes detected: {len(keltner.trend_changes)}")
    
    # Print last few signals
    print("\nRecent signals:")
    for i, signal in signals[-3:]:
        print(f"Bar {i}: {signal.signal_strength.value} - {signal.signals}")
    
    # Test channel levels
    channel_levels = keltner.get_channel_levels()
    print(f"\nCurrent Channel Levels:")
    for key, value in channel_levels.items():
        print(f"{key}: {value:.6f}")

if __name__ == "__main__":
    test_keltner_channels()
