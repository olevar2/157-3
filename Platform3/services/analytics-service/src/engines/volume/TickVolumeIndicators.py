"""
Tick Volume Indicators Engine
M1-M5 tick volume analysis for scalping and day trading validation.
Provides real-time volume confirmation for short-term trading entries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio

class VolumeSignal(Enum):
    """Volume signal types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"

class VolumeStrength(Enum):
    """Volume strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TickVolumeMetrics:
    """Tick volume analysis metrics"""
    symbol: str
    timestamp: datetime
    timeframe: str
    current_volume: float
    avg_volume: float
    volume_ratio: float
    volume_trend: str
    volume_signal: VolumeSignal
    volume_strength: VolumeStrength
    tick_count: int
    avg_tick_size: float
    volume_price_trend: str
    accumulation_distribution: float
    money_flow_index: float
    volume_oscillator: float

@dataclass
class VolumeConfirmation:
    """Volume confirmation for price movements"""
    price_move_direction: str
    volume_confirmation: bool
    confirmation_strength: float
    volume_spike: bool
    unusual_activity: bool
    institutional_flow: bool
    retail_flow: bool

@dataclass
class TickVolumeResult:
    """Complete tick volume analysis result"""
    symbol: str
    timestamp: datetime
    timeframe: str
    metrics: TickVolumeMetrics
    confirmation: VolumeConfirmation
    signals: List[Dict[str, Any]]
    alerts: List[str]
    confidence: float

class TickVolumeIndicators:
    """
    Tick Volume Indicators Engine
    Specialized for M1-M5 tick volume analysis and scalping validation
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Volume analysis parameters
        self.volume_lookback_periods = 20
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.unusual_volume_threshold = 3.0  # 3x average volume
        self.tick_size_threshold = 0.0001  # 1 pip for major pairs

        # Moving average periods
        self.short_ma_period = 5
        self.long_ma_period = 20

        # Cache for performance
        self.volume_cache = {}
        self.cache_duration = timedelta(minutes=5)

    async def analyze_tick_volume(self,
                                symbol: str,
                                price_data: pd.DataFrame,
                                timeframe: str = "M5") -> TickVolumeResult:
        """
        Analyze tick volume for scalping and day trading validation

        Args:
            symbol: Trading symbol
            price_data: OHLCV data with tick volume
            timeframe: Analysis timeframe (M1, M5)

        Returns:
            TickVolumeResult with comprehensive volume analysis
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{len(price_data)}"
            if self._is_cache_valid(cache_key):
                return self.volume_cache[cache_key]['result']

            # Validate data
            if 'volume' not in price_data.columns or len(price_data) < self.volume_lookback_periods:
                return self._create_empty_result(symbol, timeframe)

            # Calculate volume metrics
            metrics = await self._calculate_volume_metrics(symbol, price_data, timeframe)

            # Analyze volume confirmation
            confirmation = await self._analyze_volume_confirmation(price_data)

            # Generate volume signals
            signals = await self._generate_volume_signals(metrics, confirmation)

            # Generate alerts
            alerts = self._generate_volume_alerts(metrics, confirmation)

            # Calculate overall confidence
            confidence = self._calculate_volume_confidence(metrics, confirmation)

            result = TickVolumeResult(
                symbol=symbol,
                timestamp=datetime.now(),
                timeframe=timeframe,
                metrics=metrics,
                confirmation=confirmation,
                signals=signals,
                alerts=alerts,
                confidence=confidence
            )

            # Cache result
            self.volume_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }

            return result

        except Exception as e:
            self.logger.error(f"Tick volume analysis error for {symbol}: {e}")
            return self._create_empty_result(symbol, timeframe)

    async def _calculate_volume_metrics(self,
                                      symbol: str,
                                      price_data: pd.DataFrame,
                                      timeframe: str) -> TickVolumeMetrics:
        """Calculate comprehensive volume metrics"""
        current_volume = price_data.iloc[-1]['volume']

        # Calculate average volume
        recent_volumes = price_data['volume'].tail(self.volume_lookback_periods)
        avg_volume = recent_volumes.mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Volume trend analysis
        volume_ma_short = price_data['volume'].rolling(self.short_ma_period).mean()
        volume_ma_long = price_data['volume'].rolling(self.long_ma_period).mean()

        if volume_ma_short.iloc[-1] > volume_ma_long.iloc[-1]:
            volume_trend = "increasing"
        elif volume_ma_short.iloc[-1] < volume_ma_long.iloc[-1]:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"

        # Volume signal classification
        volume_signal = self._classify_volume_signal(price_data)

        # Volume strength
        volume_strength = self._calculate_volume_strength(volume_ratio)

        # Tick analysis
        tick_count = len(price_data)
        avg_tick_size = self._calculate_avg_tick_size(price_data)

        # Volume-price trend analysis
        volume_price_trend = self._analyze_volume_price_trend(price_data)

        # Advanced indicators
        accumulation_distribution = self._calculate_accumulation_distribution(price_data)
        money_flow_index = self._calculate_money_flow_index(price_data)
        volume_oscillator = self._calculate_volume_oscillator(price_data)

        return TickVolumeMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            timeframe=timeframe,
            current_volume=current_volume,
            avg_volume=avg_volume,
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            volume_signal=volume_signal,
            volume_strength=volume_strength,
            tick_count=tick_count,
            avg_tick_size=avg_tick_size,
            volume_price_trend=volume_price_trend,
            accumulation_distribution=accumulation_distribution,
            money_flow_index=money_flow_index,
            volume_oscillator=volume_oscillator
        )

    def _classify_volume_signal(self, price_data: pd.DataFrame) -> VolumeSignal:
        """Classify volume signal based on price and volume relationship"""
        recent_data = price_data.tail(5)

        # Calculate price change and volume change
        price_change = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        volume_trend = recent_data['volume'].iloc[-1] - recent_data['volume'].mean()

        if price_change > 0 and volume_trend > 0:
            return VolumeSignal.BULLISH
        elif price_change < 0 and volume_trend > 0:
            return VolumeSignal.BEARISH
        elif volume_trend > recent_data['volume'].std():
            return VolumeSignal.ACCUMULATION
        elif volume_trend < -recent_data['volume'].std():
            return VolumeSignal.DISTRIBUTION
        else:
            return VolumeSignal.NEUTRAL

    def _calculate_volume_strength(self, volume_ratio: float) -> VolumeStrength:
        """Calculate volume strength based on ratio to average"""
        if volume_ratio >= 3.0:
            return VolumeStrength.VERY_STRONG
        elif volume_ratio >= 2.0:
            return VolumeStrength.STRONG
        elif volume_ratio >= 1.5:
            return VolumeStrength.MODERATE
        else:
            return VolumeStrength.WEAK

    def _calculate_avg_tick_size(self, price_data: pd.DataFrame) -> float:
        """Calculate average tick size"""
        tick_sizes = []

        for i in range(1, len(price_data)):
            prev_close = price_data.iloc[i-1]['close']
            curr_close = price_data.iloc[i]['close']
            tick_size = abs(curr_close - prev_close)
            if tick_size > 0:
                tick_sizes.append(tick_size)

        return np.mean(tick_sizes) if tick_sizes else 0.0

    def _analyze_volume_price_trend(self, price_data: pd.DataFrame) -> str:
        """Analyze volume-price trend relationship"""
        recent_data = price_data.tail(10)

        # Calculate correlation between price and volume
        price_changes = recent_data['close'].diff()
        volume_changes = recent_data['volume'].diff()

        correlation = np.corrcoef(price_changes.dropna(), volume_changes.dropna())[0, 1]

        if np.isnan(correlation):
            return "neutral"
        elif correlation > 0.5:
            return "confirming"
        elif correlation < -0.5:
            return "diverging"
        else:
            return "neutral"

    def _calculate_accumulation_distribution(self, price_data: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution Line"""
        if len(price_data) < 2:
            return 0.0

        ad_line = 0.0

        for i in range(len(price_data)):
            high = price_data.iloc[i]['high']
            low = price_data.iloc[i]['low']
            close = price_data.iloc[i]['close']
            volume = price_data.iloc[i]['volume']

            if high != low:
                money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
                money_flow_volume = money_flow_multiplier * volume
                ad_line += money_flow_volume

        return ad_line

    def _calculate_money_flow_index(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Money Flow Index (MFI)"""
        if len(price_data) < period + 1:
            return 50.0  # Neutral value

        typical_prices = (price_data['high'] + price_data['low'] + price_data['close']) / 3
        money_flows = typical_prices * price_data['volume']

        positive_flows = []
        negative_flows = []

        for i in range(1, len(typical_prices)):
            if typical_prices.iloc[i] > typical_prices.iloc[i-1]:
                positive_flows.append(money_flows.iloc[i])
                negative_flows.append(0)
            elif typical_prices.iloc[i] < typical_prices.iloc[i-1]:
                positive_flows.append(0)
                negative_flows.append(money_flows.iloc[i])
            else:
                positive_flows.append(0)
                negative_flows.append(0)

        positive_flow_sum = sum(positive_flows[-period:])
        negative_flow_sum = sum(negative_flows[-period:])

        if negative_flow_sum == 0:
            return 100.0

        money_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi

    def _calculate_volume_oscillator(self, price_data: pd.DataFrame) -> float:
        """Calculate Volume Oscillator"""
        if len(price_data) < self.long_ma_period:
            return 0.0

        short_ma = price_data['volume'].rolling(self.short_ma_period).mean().iloc[-1]
        long_ma = price_data['volume'].rolling(self.long_ma_period).mean().iloc[-1]

        if long_ma == 0:
            return 0.0

        volume_oscillator = ((short_ma - long_ma) / long_ma) * 100
        return volume_oscillator

    async def _analyze_volume_confirmation(self, price_data: pd.DataFrame) -> VolumeConfirmation:
        """Analyze volume confirmation for price movements"""
        recent_data = price_data.tail(5)

        # Determine price move direction
        price_change = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        if price_change > 0:
            price_move_direction = "up"
        elif price_change < 0:
            price_move_direction = "down"
        else:
            price_move_direction = "sideways"

        # Check volume confirmation
        avg_volume = price_data['volume'].tail(20).mean()
        current_volume = recent_data['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        volume_confirmation = volume_ratio > 1.2  # 20% above average
        confirmation_strength = min(volume_ratio / 2.0, 1.0)  # Normalize to 0-1

        # Check for volume spikes
        volume_spike = volume_ratio >= self.volume_spike_threshold
        unusual_activity = volume_ratio >= self.unusual_volume_threshold

        # Analyze flow patterns (simplified)
        institutional_flow = volume_ratio > 2.5 and abs(price_change) > recent_data['close'].std()
        retail_flow = volume_ratio < 1.5 and abs(price_change) < recent_data['close'].std()

        return VolumeConfirmation(
            price_move_direction=price_move_direction,
            volume_confirmation=volume_confirmation,
            confirmation_strength=confirmation_strength,
            volume_spike=volume_spike,
            unusual_activity=unusual_activity,
            institutional_flow=institutional_flow,
            retail_flow=retail_flow
        )

    async def _generate_volume_signals(self,
                                     metrics: TickVolumeMetrics,
                                     confirmation: VolumeConfirmation) -> List[Dict[str, Any]]:
        """Generate trading signals based on volume analysis"""
        signals = []

        # Volume breakout signal
        if metrics.volume_strength in [VolumeStrength.STRONG, VolumeStrength.VERY_STRONG]:
            if confirmation.volume_confirmation:
                signals.append({
                    'type': 'VOLUME_BREAKOUT',
                    'direction': confirmation.price_move_direction,
                    'strength': metrics.volume_strength.value,
                    'confidence': confirmation.confirmation_strength,
                    'reason': f'Strong volume ({metrics.volume_ratio:.1f}x avg) confirming {confirmation.price_move_direction} move'
                })

        # Accumulation/Distribution signals
        if metrics.volume_signal == VolumeSignal.ACCUMULATION:
            signals.append({
                'type': 'ACCUMULATION',
                'direction': 'bullish',
                'strength': 'moderate',
                'confidence': 0.7,
                'reason': 'Volume accumulation pattern detected'
            })
        elif metrics.volume_signal == VolumeSignal.DISTRIBUTION:
            signals.append({
                'type': 'DISTRIBUTION',
                'direction': 'bearish',
                'strength': 'moderate',
                'confidence': 0.7,
                'reason': 'Volume distribution pattern detected'
            })

        # Money Flow Index signals
        if metrics.money_flow_index > 80:
            signals.append({
                'type': 'OVERBOUGHT',
                'direction': 'bearish',
                'strength': 'moderate',
                'confidence': 0.6,
                'reason': f'Money Flow Index overbought at {metrics.money_flow_index:.1f}'
            })
        elif metrics.money_flow_index < 20:
            signals.append({
                'type': 'OVERSOLD',
                'direction': 'bullish',
                'strength': 'moderate',
                'confidence': 0.6,
                'reason': f'Money Flow Index oversold at {metrics.money_flow_index:.1f}'
            })

        # Volume oscillator signals
        if metrics.volume_oscillator > 20:
            signals.append({
                'type': 'HIGH_VOLUME_MOMENTUM',
                'direction': 'continuation',
                'strength': 'moderate',
                'confidence': 0.6,
                'reason': f'Volume oscillator high at {metrics.volume_oscillator:.1f}%'
            })
        elif metrics.volume_oscillator < -20:
            signals.append({
                'type': 'LOW_VOLUME_MOMENTUM',
                'direction': 'reversal',
                'strength': 'weak',
                'confidence': 0.5,
                'reason': f'Volume oscillator low at {metrics.volume_oscillator:.1f}%'
            })

        return signals

    def _generate_volume_alerts(self,
                              metrics: TickVolumeMetrics,
                              confirmation: VolumeConfirmation) -> List[str]:
        """Generate volume-based alerts"""
        alerts = []

        # Unusual volume alert
        if confirmation.unusual_activity:
            alerts.append(f"UNUSUAL VOLUME: {metrics.volume_ratio:.1f}x average volume detected")

        # Volume spike alert
        if confirmation.volume_spike:
            alerts.append(f"VOLUME SPIKE: {metrics.volume_ratio:.1f}x average volume")

        # Institutional flow alert
        if confirmation.institutional_flow:
            alerts.append("INSTITUTIONAL FLOW: Large volume with significant price movement")

        # Volume divergence alert
        if metrics.volume_price_trend == "diverging":
            alerts.append("VOLUME DIVERGENCE: Price and volume moving in opposite directions")

        # Extreme MFI levels
        if metrics.money_flow_index > 90:
            alerts.append(f"EXTREME OVERBOUGHT: MFI at {metrics.money_flow_index:.1f}")
        elif metrics.money_flow_index < 10:
            alerts.append(f"EXTREME OVERSOLD: MFI at {metrics.money_flow_index:.1f}")

        return alerts

    def _calculate_volume_confidence(self,
                                   metrics: TickVolumeMetrics,
                                   confirmation: VolumeConfirmation) -> float:
        """Calculate overall confidence in volume analysis"""
        confidence = 0.5  # Base confidence

        # Volume strength boost
        if metrics.volume_strength == VolumeStrength.VERY_STRONG:
            confidence += 0.3
        elif metrics.volume_strength == VolumeStrength.STRONG:
            confidence += 0.2
        elif metrics.volume_strength == VolumeStrength.MODERATE:
            confidence += 0.1

        # Confirmation boost
        if confirmation.volume_confirmation:
            confidence += 0.2 * confirmation.confirmation_strength

        # Volume-price trend boost
        if metrics.volume_price_trend == "confirming":
            confidence += 0.1
        elif metrics.volume_price_trend == "diverging":
            confidence -= 0.1

        # Signal clarity boost
        if metrics.volume_signal in [VolumeSignal.BULLISH, VolumeSignal.BEARISH]:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.volume_cache:
            return False

        cache_time = self.volume_cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self.cache_duration

    def _create_empty_result(self, symbol: str, timeframe: str) -> TickVolumeResult:
        """Create empty result for error cases"""
        empty_metrics = TickVolumeMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            timeframe=timeframe,
            current_volume=0.0,
            avg_volume=0.0,
            volume_ratio=1.0,
            volume_trend="stable",
            volume_signal=VolumeSignal.NEUTRAL,
            volume_strength=VolumeStrength.WEAK,
            tick_count=0,
            avg_tick_size=0.0,
            volume_price_trend="neutral",
            accumulation_distribution=0.0,
            money_flow_index=50.0,
            volume_oscillator=0.0
        )

        empty_confirmation = VolumeConfirmation(
            price_move_direction="sideways",
            volume_confirmation=False,
            confirmation_strength=0.0,
            volume_spike=False,
            unusual_activity=False,
            institutional_flow=False,
            retail_flow=False
        )

        return TickVolumeResult(
            symbol=symbol,
            timestamp=datetime.now(),
            timeframe=timeframe,
            metrics=empty_metrics,
            confirmation=empty_confirmation,
            signals=[],
            alerts=[],
            confidence=0.0
        )
