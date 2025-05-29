"""
Donchian Channels Implementation
Advanced breakout detection and trend following system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from ..indicator_base import TechnicalIndicator, IndicatorConfig, IndicatorSignal, ValidationResult

@dataclass
class DonchianConfig(IndicatorConfig):
    """Configuration for Donchian Channels"""
    period: int = 20
    middle_period: int = 10  # For middle line calculation
    breakout_confirmation: int = 2  # Periods to confirm breakout
    enable_middle_line: bool = True
    enable_squeeze_detection: bool = True
    enable_trend_analysis: bool = True
    enable_volatility_bands: bool = True
    volatility_multiplier: float = 1.5
    
@dataclass
class DonchianChannels:
    """Donchian Channel values"""
    upper: float = 0.0
    lower: float = 0.0
    middle: float = 0.0
    width: float = 0.0
    position: float = 0.0  # Price position within channel (0-1)

class DonchianChannelIndicator(TechnicalIndicator):
    """
    Donchian Channels
    
    Trend-following indicator that plots the highest high and lowest low
    over a specified period, creating dynamic support and resistance levels.
    
    Features:
    - Upper/Lower channel calculation
    - Middle line (optional)
    - Breakout detection and confirmation
    - Channel squeeze analysis
    - Trend strength measurement
    - Volatility-adjusted signals
    - Position within channel analysis
    """
    
    def __init__(self, config: DonchianConfig):
        super().__init__(config)
        self.config = config
        self.high_values = []
        self.low_values = []
        self.close_values = []
        self.channels = []
        
        # Breakout tracking
        self.breakout_history = []
        self.squeeze_periods = []
        self.trend_direction = "NEUTRAL"
        self.trend_strength = 0
        
        # Volatility tracking
        self.volatility_history = []
        
    def calculate(self, high: float, low: float, close: float, 
                 volume: Optional[float] = None, timestamp: Optional[Any] = None) -> IndicatorSignal:
        """Calculate Donchian Channels with comprehensive analysis"""
        
        # Store price data
        self.high_values.append(high)
        self.low_values.append(low)
        self.close_values.append(close)
        
        if len(self.high_values) < self.config.period:
            return self._create_signal(0, 0, "INSUFFICIENT_DATA", {
                'values_needed': self.config.period - len(self.high_values)
            })
        
        # Keep only required values
        if len(self.high_values) > self.config.period * 2:
            self.high_values = self.high_values[-self.config.period * 2:]
            self.low_values = self.low_values[-self.config.period * 2:]
            self.close_values = self.close_values[-self.config.period * 2:]
        
        # Calculate channel values
        upper_channel = max(self.high_values[-self.config.period:])
        lower_channel = min(self.low_values[-self.config.period:])
        
        # Calculate middle line
        if self.config.enable_middle_line:
            if len(self.high_values) >= self.config.middle_period:
                middle_high = max(self.high_values[-self.config.middle_period:])
                middle_low = min(self.low_values[-self.config.middle_period:])
                middle_channel = (middle_high + middle_low) / 2
            else:
                middle_channel = (upper_channel + lower_channel) / 2
        else:
            middle_channel = (upper_channel + lower_channel) / 2
        
        # Calculate channel metrics
        channel_width = upper_channel - lower_channel
        price_position = ((close - lower_channel) / channel_width 
                         if channel_width > 0 else 0.5)
        
        # Create channel object
        current_channel = DonchianChannels(
            upper=upper_channel,
            lower=lower_channel,
            middle=middle_channel,
            width=channel_width,
            position=price_position
        )
        
        self.channels.append(current_channel)
        
        # Track volatility
        self.volatility_history.append(channel_width)
        
        # Keep history manageable
        if len(self.channels) > 200:
            self.channels = self.channels[-200:]
            self.volatility_history = self.volatility_history[-200:]
        
        # Generate comprehensive signal
        return self._generate_comprehensive_signal(current_channel, high, low, close, volume, timestamp)
    
    def _generate_comprehensive_signal(self, channel: DonchianChannels, high: float, 
                                     low: float, close: float, volume: Optional[float], 
                                     timestamp: Optional[Any]) -> IndicatorSignal:
        """Generate comprehensive Donchian Channel signal"""
        
        signal_strength = 0
        confidence = 0.5
        signals = []
        metadata = {
            'upper_channel': channel.upper,
            'lower_channel': channel.lower,
            'middle_channel': channel.middle,
            'channel_width': channel.width,
            'price_position': channel.position
        }
        
        # 1. Breakout analysis
        breakout_analysis = self._analyze_breakouts(channel, high, low, close)
        signal_strength += breakout_analysis['strength']
        confidence += breakout_analysis['confidence']
        signals.extend(breakout_analysis['signals'])
        metadata.update(breakout_analysis['metadata'])
        
        # 2. Channel position analysis
        position_analysis = self._analyze_channel_position(channel, close)
        signal_strength += position_analysis['strength']
        confidence += position_analysis['confidence']
        signals.extend(position_analysis['signals'])
        metadata.update(position_analysis['metadata'])
        
        # 3. Squeeze detection
        if self.config.enable_squeeze_detection and len(self.channels) >= 10:
            squeeze_analysis = self._analyze_squeeze_patterns(channel)
            signal_strength += squeeze_analysis['strength']
            confidence += squeeze_analysis['confidence']
            signals.extend(squeeze_analysis['signals'])
            metadata.update(squeeze_analysis['metadata'])
        
        # 4. Trend analysis
        if self.config.enable_trend_analysis and len(self.channels) >= 5:
            trend_analysis = self._analyze_trend_strength(channel)
            signal_strength += trend_analysis['strength']
            confidence += trend_analysis['confidence']
            signals.extend(trend_analysis['signals'])
            metadata.update(trend_analysis['metadata'])
        
        # 5. Middle line analysis
        if self.config.enable_middle_line and len(self.channels) >= 3:
            middle_analysis = self._analyze_middle_line_interaction(channel, close)
            signal_strength += middle_analysis['strength']
            confidence += middle_analysis['confidence']
            signals.extend(middle_analysis['signals'])
            metadata.update(middle_analysis['metadata'])
        
        # 6. Volatility analysis
        if self.config.enable_volatility_bands and len(self.volatility_history) >= 10:
            volatility_analysis = self._analyze_volatility_context(channel, volume)
            signal_strength *= volatility_analysis['multiplier']
            confidence += volatility_analysis['confidence_adj']
            signals.extend(volatility_analysis['signals'])
            metadata.update(volatility_analysis['metadata'])
        
        # Normalize values
        signal_strength = max(-1, min(1, signal_strength))
        confidence = max(0, min(1, confidence / 5))  # Normalize multi-component confidence
        
        # Determine overall signal
        if abs(signal_strength) < 0.2:
            overall_signal = "NEUTRAL"
        elif signal_strength > 0:
            overall_signal = "BULLISH"
        else:
            overall_signal = "BEARISH"
        
        metadata['component_signals'] = signals
        metadata['trend_direction'] = self.trend_direction
        metadata['trend_strength'] = self.trend_strength
        
        return self._create_signal(signal_strength, confidence, overall_signal, metadata)
    
    def _analyze_breakouts(self, channel: DonchianChannels, high: float, 
                          low: float, close: float) -> Dict:
        """Analyze channel breakout patterns"""
        strength = 0
        confidence = 0
        signals = []
        metadata = {}
        
        # Current breakout detection
        upper_breakout = high > channel.upper or close > channel.upper
        lower_breakout = low < channel.lower or close < channel.lower
        
        if upper_breakout:
            strength = 0.6
            confidence = 0.7
            signals.append("UPPER_CHANNEL_BREAKOUT")
            metadata['breakout_type'] = 'bullish'
            self.breakout_history.append(('bullish', len(self.channels)))
        elif lower_breakout:
            strength = -0.6
            confidence = 0.7
            signals.append("LOWER_CHANNEL_BREAKOUT")
            metadata['breakout_type'] = 'bearish'
            self.breakout_history.append(('bearish', len(self.channels)))
        
        # Breakout confirmation
        if len(self.channels) >= self.config.breakout_confirmation:
            confirmed_breakout = self._check_breakout_confirmation(channel, close)
            if confirmed_breakout:
                confidence += 0.2
                signals.append("CONFIRMED_BREAKOUT")
                metadata['breakout_confirmed'] = True
        
        # Near breakout conditions
        upper_distance = (channel.upper - close) / channel.width if channel.width > 0 else 0
        lower_distance = (close - channel.lower) / channel.width if channel.width > 0 else 0
        
        if upper_distance < 0.05:  # Within 5% of upper channel
            signals.append("NEAR_UPPER_BREAKOUT")
            strength += 0.2
        elif lower_distance < 0.05:  # Within 5% of lower channel
            signals.append("NEAR_LOWER_BREAKOUT")
            strength -= 0.2
        
        # False breakout detection
        if len(self.breakout_history) >= 2:
            recent_breakouts = self.breakout_history[-2:]
            if (recent_breakouts[0][0] != recent_breakouts[1][0] and 
                recent_breakouts[1][1] - recent_breakouts[0][1] < 5):  # Quick reversal
                signals.append("POTENTIAL_FALSE_BREAKOUT")
                strength *= 0.7  # Reduce confidence
        
        metadata['upper_distance_pct'] = upper_distance * 100
        metadata['lower_distance_pct'] = lower_distance * 100
        metadata['breakout_count'] = len(self.breakout_history)
        
        return {
            'strength': strength,
            'confidence': confidence,
            'signals': signals,
            'metadata': metadata
        }
    
    def _check_breakout_confirmation(self, channel: DonchianChannels, close: float) -> bool:
        """Check if breakout is confirmed over multiple periods"""
        if len(self.channels) < self.config.breakout_confirmation:
            return False
        
        recent_channels = self.channels[-self.config.breakout_confirmation:]
        recent_closes = self.close_values[-self.config.breakout_confirmation:]
        
        # Check for sustained breakout
        upper_breaks = sum(1 for i, ch in enumerate(recent_channels) 
                          if recent_closes[i] > ch.upper)
        lower_breaks = sum(1 for i, ch in enumerate(recent_channels) 
                          if recent_closes[i] < ch.lower)
        
        return upper_breaks >= self.config.breakout_confirmation or lower_breaks >= self.config.breakout_confirmation
    
    def _analyze_channel_position(self, channel: DonchianChannels, close: float) -> Dict:
        """Analyze price position within channel"""
        strength = 0
        confidence = 0
        signals = []
        metadata = {}
        
        position = channel.position
        
        # Position-based signals
        if position > 0.8:
            strength = 0.3
            confidence = 0.5
            signals.append("UPPER_CHANNEL_ZONE")
            metadata['channel_zone'] = 'upper'
        elif position > 0.6:
            signals.append("UPPER_MIDDLE_ZONE")
            metadata['channel_zone'] = 'upper_middle'
        elif position < 0.2:
            strength = -0.3
            confidence = 0.5
            signals.append("LOWER_CHANNEL_ZONE")
            metadata['channel_zone'] = 'lower'
        elif position < 0.4:
            signals.append("LOWER_MIDDLE_ZONE")
            metadata['channel_zone'] = 'lower_middle'
        else:
            signals.append("MIDDLE_CHANNEL_ZONE")
            metadata['channel_zone'] = 'middle'
        
        # Position momentum
        if len(self.channels) >= 3:
            prev_position = self.channels[-2].position
            position_momentum = position - prev_position
            
            if position_momentum > 0.1:
                signals.append("RISING_CHANNEL_POSITION")
                strength += 0.2
            elif position_momentum < -0.1:
                signals.append("FALLING_CHANNEL_POSITION")
                strength -= 0.2
            
            metadata['position_momentum'] = position_momentum
        
        # Channel extremes persistence
        if len(self.channels) >= 5:
            recent_positions = [ch.position for ch in self.channels[-5:]]
            
            if all(p > 0.7 for p in recent_positions):
                signals.append("PERSISTENT_UPPER_ZONE")
                confidence += 0.2
            elif all(p < 0.3 for p in recent_positions):
                signals.append("PERSISTENT_LOWER_ZONE")
                confidence += 0.2
        
        metadata['position_percentile'] = position * 100
        
        return {
            'strength': strength,
            'confidence': confidence,
            'signals': signals,
            'metadata': metadata
        }
    
    def _analyze_squeeze_patterns(self, channel: DonchianChannels) -> Dict:
        """Analyze channel squeeze and expansion patterns"""
        strength = 0
        confidence = 0
        signals = []
        metadata = {}
        
        if len(self.volatility_history) < 10:
            return {'strength': 0, 'confidence': 0, 'signals': [], 'metadata': {}}
        
        # Calculate volatility percentiles
        recent_volatility = np.mean(self.volatility_history[-5:])
        historical_volatility = np.mean(self.volatility_history[-20:]) if len(self.volatility_history) >= 20 else recent_volatility
        
        volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
        
        # Squeeze detection
        if volatility_ratio < 0.7:
            signals.append("CHANNEL_SQUEEZE")
            metadata['squeeze_intensity'] = (0.7 - volatility_ratio) / 0.7
            self.squeeze_periods.append(len(self.channels))
            
            # Potential breakout setup
            strength = 0.1  # Neutral but prepare for breakout
            confidence = 0.4
            
        # Expansion detection
        elif volatility_ratio > 1.3:
            signals.append("CHANNEL_EXPANSION")
            metadata['expansion_intensity'] = (volatility_ratio - 1.3) / 1.3
            
            # Confirm trend direction
            if channel.position > 0.5:
                strength = 0.3
                signals.append("BULLISH_EXPANSION")
            else:
                strength = -0.3
                signals.append("BEARISH_EXPANSION")
            
            confidence = 0.6
        
        # Squeeze breakout
        if (len(self.squeeze_periods) > 0 and 
            len(self.channels) - self.squeeze_periods[-1] <= 3 and
            volatility_ratio > 1.2):
            signals.append("SQUEEZE_BREAKOUT")
            strength *= 1.5  # Amplify signals after squeeze
            confidence += 0.3
        
        # Channel width percentile
        if len(self.volatility_history) >= 20:
            width_percentile = sum(1 for w in self.volatility_history[-20:] if w < channel.width) / 20
            metadata['width_percentile'] = width_percentile * 100
            
            if width_percentile < 0.2:
                signals.append("NARROW_CHANNEL_EXTREME")
            elif width_percentile > 0.8:
                signals.append("WIDE_CHANNEL_EXTREME")
        
        metadata['volatility_ratio'] = volatility_ratio
        metadata['squeeze_periods_count'] = len(self.squeeze_periods)
        
        return {
            'strength': strength,
            'confidence': confidence,
            'signals': signals,
            'metadata': metadata
        }
    
    def _analyze_trend_strength(self, channel: DonchianChannels) -> Dict:
        """Analyze trend strength using channel progression"""
        strength = 0
        confidence = 0
        signals = []
        metadata = {}
        
        if len(self.channels) < 5:
            return {'strength': 0, 'confidence': 0, 'signals': [], 'metadata': {}}
        
        # Channel direction analysis
        recent_channels = self.channels[-5:]
        
        # Upper channel trend
        upper_trend = (recent_channels[-1].upper - recent_channels[0].upper) / len(recent_channels)
        lower_trend = (recent_channels[-1].lower - recent_channels[0].lower) / len(recent_channels)
        middle_trend = (recent_channels[-1].middle - recent_channels[0].middle) / len(recent_channels)
        
        # Overall trend direction
        if upper_trend > 0 and lower_trend > 0 and middle_trend > 0:
            self.trend_direction = "BULLISH"
            trend_strength_value = min(abs(upper_trend), abs(lower_trend), abs(middle_trend))
            strength = 0.4
            confidence = 0.6
            signals.append("STRONG_UPTREND")
        elif upper_trend < 0 and lower_trend < 0 and middle_trend < 0:
            self.trend_direction = "BEARISH"
            trend_strength_value = min(abs(upper_trend), abs(lower_trend), abs(middle_trend))
            strength = -0.4
            confidence = 0.6
            signals.append("STRONG_DOWNTREND")
        else:
            self.trend_direction = "NEUTRAL"
            trend_strength_value = 0
            signals.append("SIDEWAYS_TREND")
        
        self.trend_strength = trend_strength_value
        
        # Trend acceleration
        if len(self.channels) >= 10:
            older_channels = self.channels[-10:-5]
            older_middle_trend = (older_channels[-1].middle - older_channels[0].middle) / len(older_channels)
            
            trend_acceleration = middle_trend - older_middle_trend
            
            if abs(trend_acceleration) > abs(older_middle_trend) * 0.5:
                if trend_acceleration > 0:
                    signals.append("TREND_ACCELERATING_UP")
                    strength += 0.2
                else:
                    signals.append("TREND_ACCELERATING_DOWN")
                    strength -= 0.2
                confidence += 0.2
        
        # Channel slope consistency
        upper_slopes = []
        lower_slopes = []
        
        for i in range(1, len(recent_channels)):
            upper_slopes.append(recent_channels[i].upper - recent_channels[i-1].upper)
            lower_slopes.append(recent_channels[i].lower - recent_channels[i-1].lower)
        
        upper_consistency = 1 - (np.std(upper_slopes) / (np.mean(np.abs(upper_slopes)) + 1e-8))
        lower_consistency = 1 - (np.std(lower_slopes) / (np.mean(np.abs(lower_slopes)) + 1e-8))
        
        avg_consistency = (upper_consistency + lower_consistency) / 2
        
        if avg_consistency > 0.7:
            signals.append("CONSISTENT_TREND")
            confidence += 0.2
        elif avg_consistency < 0.3:
            signals.append("CHOPPY_TREND")
            confidence -= 0.1
        
        metadata['upper_trend'] = upper_trend
        metadata['lower_trend'] = lower_trend
        metadata['middle_trend'] = middle_trend
        metadata['trend_consistency'] = avg_consistency
        
        return {
            'strength': strength,
            'confidence': confidence,
            'signals': signals,
            'metadata': metadata
        }
    
    def _analyze_middle_line_interaction(self, channel: DonchianChannels, close: float) -> Dict:
        """Analyze price interaction with middle line"""
        strength = 0
        confidence = 0
        signals = []
        metadata = {}
        
        if len(self.close_values) < 3:
            return {'strength': 0, 'confidence': 0, 'signals': [], 'metadata': {}}
        
        prev_close = self.close_values[-2]
        middle = channel.middle
        prev_middle = self.channels[-2].middle if len(self.channels) >= 2 else middle
        
        # Middle line crossovers
        if prev_close <= prev_middle and close > middle:
            strength = 0.3
            confidence = 0.5
            signals.append("MIDDLE_LINE_CROSS_UP")
            metadata['middle_cross'] = 'bullish'
        elif prev_close >= prev_middle and close < middle:
            strength = -0.3
            confidence = 0.5
            signals.append("MIDDLE_LINE_CROSS_DOWN")
            metadata['middle_cross'] = 'bearish'
        
        # Middle line as support/resistance
        distance_to_middle = abs(close - middle) / channel.width if channel.width > 0 else 0
        
        if distance_to_middle < 0.02:  # Very close to middle line
            signals.append("AT_MIDDLE_LINE")
            
            # Test of middle line
            if len(self.close_values) >= 5:
                recent_distances = [abs(self.close_values[i] - self.channels[i-1].middle) / self.channels[i-1].width 
                                  for i in range(-5, 0) if i < len(self.channels)]
                
                if any(d > 0.1 for d in recent_distances[:-1]) and distance_to_middle < 0.02:
                    signals.append("MIDDLE_LINE_RETEST")
                    confidence += 0.2
        
        # Middle line slope
        if len(self.channels) >= 3:
            middle_slope = (channel.middle - self.channels[-3].middle) / 3
            
            if middle_slope > 0:
                signals.append("MIDDLE_LINE_RISING")
                if close > middle:
                    strength += 0.1
            elif middle_slope < 0:
                signals.append("MIDDLE_LINE_FALLING")
                if close < middle:
                    strength -= 0.1
        
        metadata['distance_to_middle_pct'] = distance_to_middle * 100
        metadata['middle_line_value'] = middle
        
        return {
            'strength': strength,
            'confidence': confidence,
            'signals': signals,
            'metadata': metadata
        }
    
    def _analyze_volatility_context(self, channel: DonchianChannels, volume: Optional[float]) -> Dict:
        """Analyze channel in volatility context"""
        multiplier = 1.0
        confidence_adj = 0
        signals = []
        metadata = {}
        
        # Relative volatility analysis
        if len(self.volatility_history) >= 20:
            current_volatility = channel.width
            avg_volatility = np.mean(self.volatility_history[-20:])
            volatility_std = np.std(self.volatility_history[-20:])
            
            if volatility_std > 0:
                volatility_zscore = (current_volatility - avg_volatility) / volatility_std
                
                if volatility_zscore > 2:
                    signals.append("EXTREMELY_HIGH_VOLATILITY")
                    multiplier *= 0.8  # Reduce signal strength in extreme volatility
                elif volatility_zscore > 1:
                    signals.append("HIGH_VOLATILITY")
                    multiplier *= 0.9
                elif volatility_zscore < -2:
                    signals.append("EXTREMELY_LOW_VOLATILITY")
                    multiplier *= 1.2  # Enhance signals in low volatility
                elif volatility_zscore < -1:
                    signals.append("LOW_VOLATILITY")
                    multiplier *= 1.1
                
                metadata['volatility_zscore'] = volatility_zscore
            
            metadata['avg_volatility'] = avg_volatility
        
        # Volume confirmation
        if volume is not None and volume > 0:
            # Check for volume spikes during breakouts
            if any(signal in ['UPPER_CHANNEL_BREAKOUT', 'LOWER_CHANNEL_BREAKOUT'] 
                   for signal in signals):
                if volume > 1.5:  # Assume normalized volume
                    multiplier *= 1.2
                    confidence_adj = 0.15
                    signals.append("VOLUME_CONFIRMED_BREAKOUT")
                else:
                    signals.append("LOW_VOLUME_BREAKOUT")
                    multiplier *= 0.8
        
        # Adaptive channel width
        if self.config.enable_volatility_bands and len(self.volatility_history) >= 10:
            volatility_trend = np.polyfit(range(10), self.volatility_history[-10:], 1)[0]
            
            if volatility_trend > 0:
                signals.append("EXPANDING_VOLATILITY")
            elif volatility_trend < 0:
                signals.append("CONTRACTING_VOLATILITY")
            
            metadata['volatility_trend'] = volatility_trend
        
        metadata['volatility_multiplier'] = multiplier
        
        return {
            'multiplier': multiplier,
            'confidence_adj': confidence_adj,
            'signals': signals,
            'metadata': metadata
        }
    
    def get_current_channels(self) -> Optional[DonchianChannels]:
        """Get the current channel values"""
        return self.channels[-1] if self.channels else None
    
    def get_signal_strength(self) -> float:
        """Get current signal strength based on position and breakouts"""
        if not self.channels:
            return 0
        
        current_channel = self.channels[-1]
        
        # Base signal on channel position and recent breakouts
        position_signal = (current_channel.position - 0.5) * 2  # Convert to -1 to 1
        
        # Amplify if recent breakout
        if len(self.breakout_history) > 0:
            last_breakout = self.breakout_history[-1]
            if len(self.channels) - last_breakout[1] <= 3:  # Recent breakout
                if last_breakout[0] == 'bullish':
                    position_signal = max(position_signal, 0.5)
                else:
                    position_signal = min(position_signal, -0.5)
        
        return max(-1, min(1, position_signal))
    
    def validate_data(self, high: float, low: float, close: float, 
                     volume: Optional[float] = None) -> ValidationResult:
        """Validate input data for Donchian Channel calculation"""
        if high < 0 or low < 0 or close < 0:
            return ValidationResult(False, "Price values cannot be negative")
        
        if high < low:
            return ValidationResult(False, "High price cannot be less than low price")
        
        if close < low or close > high:
            return ValidationResult(False, "Close price must be between low and high")
        
        if volume is not None and volume < 0:
            return ValidationResult(False, "Volume cannot be negative")
        
        return ValidationResult(True, "Data validation passed")

def test_donchian_channels():
    """Test Donchian Channels implementation with sample data"""
    config = DonchianConfig(
        period=20, 
        middle_period=10,
        enable_squeeze_detection=True,
        enable_trend_analysis=True,
        enable_volatility_bands=True
    )
    donchian = DonchianChannelIndicator(config)
    
    # Test with sample EURUSD data
    test_data = [
        # (High, Low, Close, Volume) - Trending data
        (1.1050, 1.1000, 1.1025, 1000),
        (1.1075, 1.1020, 1.1060, 1200),
        (1.1080, 1.1030, 1.1070, 1100),
        (1.1090, 1.1040, 1.1085, 1300),
        (1.1100, 1.1050, 1.1095, 1400),
        (1.1120, 1.1070, 1.1110, 1500),
        (1.1140, 1.1090, 1.1130, 1600),
        (1.1160, 1.1110, 1.1150, 1700),
        (1.1180, 1.1130, 1.1170, 1800),
        (1.1200, 1.1150, 1.1190, 1900),
        (1.1220, 1.1170, 1.1210, 2000),
        (1.1240, 1.1190, 1.1230, 2100),
        (1.1260, 1.1210, 1.1250, 2200),
        (1.1280, 1.1230, 1.1270, 2300),
        (1.1300, 1.1250, 1.1290, 2400),
        (1.1320, 1.1270, 1.1310, 2500),
        (1.1340, 1.1290, 1.1330, 2600),
        (1.1360, 1.1310, 1.1350, 2700),
        (1.1380, 1.1330, 1.1370, 2800),
        (1.1400, 1.1350, 1.1390, 2900),
        # Breakout attempt
        (1.1420, 1.1360, 1.1410, 3500),  # Upper breakout
        (1.1440, 1.1380, 1.1430, 3800),
        (1.1460, 1.1400, 1.1450, 4000),
        # Some consolidation
        (1.1450, 1.1390, 1.1420, 3200),
        (1.1440, 1.1380, 1.1410, 3000),
        (1.1430, 1.1370, 1.1400, 2800),
        (1.1425, 1.1375, 1.1395, 2600),
        (1.1420, 1.1370, 1.1390, 2400),
        # Continued breakout
        (1.1460, 1.1400, 1.1450, 4200),
        (1.1480, 1.1420, 1.1470, 4500),
    ]
    
    print("Testing Donchian Channels")
    print("=" * 70)
    
    for i, (high, low, close, volume) in enumerate(test_data):
        signal = donchian.calculate(high, low, close, volume, f"2024-01-{i+1:02d}")
        
        channels = donchian.get_current_channels()
        if channels:
            print(f"Day {i+1:2d}: Upper={channels.upper:7.4f} | Lower={channels.lower:7.4f} | "
                  f"Pos={channels.position:5.2f} | Signal={signal.signal:8s} | "
                  f"Strength={signal.strength:6.2f} | Confidence={signal.confidence:5.2f}")
            
            if signal.metadata.get('component_signals'):
                signals_display = ', '.join(signal.metadata['component_signals'][:2])
                print(f"         Signals: {signals_display}")
    
    final_channels = donchian.get_current_channels()
    if final_channels:
        print(f"\nFinal Channels - Upper: {final_channels.upper:.4f}, "
              f"Lower: {final_channels.lower:.4f}, Middle: {final_channels.middle:.4f}")
        print(f"Channel Width: {final_channels.width:.4f}, Position: {final_channels.position:.2f}")
    print(f"Signal Strength: {donchian.get_signal_strength():.2f}")

if __name__ == "__main__":
    test_donchian_channels()
