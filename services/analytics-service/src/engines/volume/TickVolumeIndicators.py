"""
Tick Volume Indicators Module
M1-M5 tick volume analysis for scalping and day trading
Optimized for real-time volume confirmation for scalping entries.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import statistics


@dataclass
class VolumeBar:
    """Volume bar data structure"""
    timestamp: float
    volume: float
    price: float
    tick_count: int
    buy_volume: float
    sell_volume: float
    volume_delta: float  # buy_volume - sell_volume


@dataclass
class VolumeIndicator:
    """Volume indicator result"""
    name: str
    value: float
    signal: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1
    timeframe: str
    confidence: float


@dataclass
class VolumeProfile:
    """Volume profile for price levels"""
    price_level: float
    volume: float
    percentage: float
    poc: bool  # Point of Control
    value_area: bool  # Value Area High/Low


@dataclass
class TickVolumeResult:
    """Tick volume analysis result"""
    symbol: str
    timestamp: float
    timeframe: str
    volume_indicators: List[VolumeIndicator]
    volume_profile: List[VolumeProfile]
    volume_trend: str
    volume_strength: float
    scalping_signals: List[Dict[str, Union[str, float]]]
    execution_metrics: Dict[str, float]


class TickVolumeIndicators:
    """
    Tick Volume Indicators Engine for Scalping
    Provides M1-M5 tick volume analysis for scalping entry confirmation
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # Volume analysis parameters
        self.volume_sma_periods = [5, 10, 20]
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.tick_threshold = 1.5  # 1.5x average tick count
        
        # Volume profile parameters
        self.profile_levels = 20  # Number of price levels for volume profile
        self.value_area_percentage = 0.7  # 70% value area
        
        # Scalping signal parameters
        self.min_volume_confirmation = 1.5  # Minimum volume multiplier
        self.volume_divergence_periods = 10
        
        # Performance optimization
        self.volume_cache: Dict[str, deque] = {}
        self.indicator_cache: Dict[str, Dict] = {}

    async def initialize(self) -> bool:
        """Initialize the Tick Volume Indicators engine"""
        try:
            self.logger.info("Initializing Tick Volume Indicators Engine...")
            
            # Test volume analysis with sample data
            test_data = self._generate_test_data()
            test_result = await self._analyze_volume_indicators(test_data)
            
            if test_result and len(test_result) > 0:
                self.ready = True
                self.logger.info("✅ Tick Volume Indicators Engine initialized")
                return True
            else:
                raise Exception("Volume indicator analysis test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Tick Volume Indicators Engine initialization failed: {e}")
            return False

    async def analyze_tick_volume(self, symbol: str, price_data: List[Dict], 
                                volume_data: List[Dict],
                                timeframe: str = 'M1') -> TickVolumeResult:
        """
        Analyze tick volume for scalping confirmation
        
        Args:
            symbol: Currency pair symbol
            price_data: List of OHLC data dictionaries
            volume_data: List of volume data dictionaries
            timeframe: Chart timeframe (M1-M5)
            
        Returns:
            TickVolumeResult with volume analysis
        """
        if not self.ready:
            raise Exception("Tick Volume Indicators Engine not initialized")
            
        if len(price_data) < 20 or len(volume_data) < 20:
            raise Exception("Insufficient data for volume analysis (minimum 20 periods)")
            
        try:
            start_time = time.time()
            
            # Prepare volume bars
            volume_bars = await self._prepare_volume_bars(price_data, volume_data)
            
            # Calculate volume indicators
            volume_indicators = await self._calculate_volume_indicators(volume_bars, timeframe)
            
            # Generate volume profile
            volume_profile = await self._generate_volume_profile(volume_bars)
            
            # Analyze volume trend
            volume_trend, volume_strength = await self._analyze_volume_trend(volume_bars)
            
            # Generate scalping signals
            scalping_signals = await self._generate_scalping_signals(
                symbol, volume_bars, volume_indicators, timeframe
            )
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(
                volume_indicators, scalping_signals
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Volume analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return TickVolumeResult(
                symbol=symbol,
                timestamp=time.time(),
                timeframe=timeframe,
                volume_indicators=volume_indicators,
                volume_profile=volume_profile,
                volume_trend=volume_trend,
                volume_strength=volume_strength,
                scalping_signals=scalping_signals,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Volume analysis failed for {symbol}: {e}")
            raise

    async def _prepare_volume_bars(self, price_data: List[Dict], 
                                 volume_data: List[Dict]) -> List[VolumeBar]:
        """Prepare volume bars from price and volume data"""
        volume_bars = []
        
        # Ensure data alignment
        min_length = min(len(price_data), len(volume_data))
        
        for i in range(min_length):
            price_bar = price_data[i]
            volume_bar = volume_data[i]
            
            timestamp = float(price_bar.get('timestamp', time.time()))
            close_price = float(price_bar.get('close', 0))
            volume = float(volume_bar.get('volume', 0))
            tick_count = int(volume_bar.get('tick_count', volume / 100))  # Estimate if not available
            
            # Estimate buy/sell volume (simplified)
            open_price = float(price_bar.get('open', close_price))
            if close_price > open_price:
                buy_volume = volume * 0.6
                sell_volume = volume * 0.4
            elif close_price < open_price:
                buy_volume = volume * 0.4
                sell_volume = volume * 0.6
            else:
                buy_volume = volume * 0.5
                sell_volume = volume * 0.5
            
            volume_delta = buy_volume - sell_volume
            
            volume_bars.append(VolumeBar(
                timestamp=timestamp,
                volume=volume,
                price=close_price,
                tick_count=tick_count,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                volume_delta=volume_delta
            ))
        
        return volume_bars

    async def _calculate_volume_indicators(self, volume_bars: List[VolumeBar], 
                                         timeframe: str) -> List[VolumeIndicator]:
        """Calculate various volume indicators"""
        indicators = []
        
        if len(volume_bars) < 20:
            return indicators
        
        volumes = [bar.volume for bar in volume_bars]
        volume_deltas = [bar.volume_delta for bar in volume_bars]
        tick_counts = [bar.tick_count for bar in volume_bars]
        
        # Volume Moving Averages
        for period in self.volume_sma_periods:
            if len(volumes) >= period:
                sma = statistics.mean(volumes[-period:])
                current_volume = volumes[-1]
                
                if current_volume > sma * self.volume_spike_threshold:
                    signal = 'bullish'
                    strength = min((current_volume / sma) / self.volume_spike_threshold, 1.0)
                elif current_volume < sma * 0.5:
                    signal = 'bearish'
                    strength = min((sma / current_volume) / 2.0, 1.0)
                else:
                    signal = 'neutral'
                    strength = 0.5
                
                indicators.append(VolumeIndicator(
                    name=f'Volume_SMA_{period}',
                    value=sma,
                    signal=signal,
                    strength=strength,
                    timeframe=timeframe,
                    confidence=0.7
                ))
        
        # Volume Delta Indicator
        if len(volume_deltas) >= 10:
            avg_delta = statistics.mean(volume_deltas[-10:])
            current_delta = volume_deltas[-1]
            
            if current_delta > abs(avg_delta) * 2:
                signal = 'bullish'
                strength = min(abs(current_delta) / abs(avg_delta) / 2, 1.0)
            elif current_delta < -abs(avg_delta) * 2:
                signal = 'bearish'
                strength = min(abs(current_delta) / abs(avg_delta) / 2, 1.0)
            else:
                signal = 'neutral'
                strength = 0.5
            
            indicators.append(VolumeIndicator(
                name='Volume_Delta',
                value=current_delta,
                signal=signal,
                strength=strength,
                timeframe=timeframe,
                confidence=0.8
            ))
        
        # Tick Volume Indicator
        if len(tick_counts) >= 10:
            avg_ticks = statistics.mean(tick_counts[-10:])
            current_ticks = tick_counts[-1]
            
            if current_ticks > avg_ticks * self.tick_threshold:
                signal = 'bullish'
                strength = min((current_ticks / avg_ticks) / self.tick_threshold, 1.0)
            elif current_ticks < avg_ticks * 0.7:
                signal = 'bearish'
                strength = min((avg_ticks / current_ticks) / 1.43, 1.0)
            else:
                signal = 'neutral'
                strength = 0.5
            
            indicators.append(VolumeIndicator(
                name='Tick_Volume',
                value=current_ticks,
                signal=signal,
                strength=strength,
                timeframe=timeframe,
                confidence=0.6
            ))
        
        # Volume Rate of Change
        if len(volumes) >= 5:
            roc_period = 5
            volume_roc = (volumes[-1] - volumes[-roc_period]) / volumes[-roc_period] * 100
            
            if volume_roc > 50:  # 50% increase
                signal = 'bullish'
                strength = min(volume_roc / 100, 1.0)
            elif volume_roc < -30:  # 30% decrease
                signal = 'bearish'
                strength = min(abs(volume_roc) / 100, 1.0)
            else:
                signal = 'neutral'
                strength = 0.5
            
            indicators.append(VolumeIndicator(
                name='Volume_ROC',
                value=volume_roc,
                signal=signal,
                strength=strength,
                timeframe=timeframe,
                confidence=0.7
            ))
        
        return indicators

    async def _generate_volume_profile(self, volume_bars: List[VolumeBar]) -> List[VolumeProfile]:
        """Generate volume profile for price levels"""
        if len(volume_bars) < 10:
            return []
        
        # Get price range
        prices = [bar.price for bar in volume_bars]
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            return []
        
        # Create price levels
        level_size = price_range / self.profile_levels
        volume_profile = []
        
        for i in range(self.profile_levels):
            level_price = min_price + (i * level_size)
            level_volume = 0
            
            # Calculate volume at this price level
            for bar in volume_bars:
                if level_price <= bar.price < level_price + level_size:
                    level_volume += bar.volume
            
            volume_profile.append({
                'price_level': level_price,
                'volume': level_volume
            })
        
        # Calculate total volume
        total_volume = sum(profile['volume'] for profile in volume_profile)
        
        if total_volume == 0:
            return []
        
        # Find Point of Control (highest volume)
        max_volume_level = max(volume_profile, key=lambda x: x['volume'])
        
        # Sort by volume to find value area
        sorted_profile = sorted(volume_profile, key=lambda x: x['volume'], reverse=True)
        value_area_volume = total_volume * self.value_area_percentage
        cumulative_volume = 0
        value_area_levels = []
        
        for level in sorted_profile:
            cumulative_volume += level['volume']
            value_area_levels.append(level['price_level'])
            if cumulative_volume >= value_area_volume:
                break
        
        # Create final volume profile
        final_profile = []
        for profile in volume_profile:
            if profile['volume'] > 0:
                final_profile.append(VolumeProfile(
                    price_level=profile['price_level'],
                    volume=profile['volume'],
                    percentage=(profile['volume'] / total_volume) * 100,
                    poc=(profile['price_level'] == max_volume_level['price_level']),
                    value_area=(profile['price_level'] in value_area_levels)
                ))
        
        return final_profile

    async def _analyze_volume_trend(self, volume_bars: List[VolumeBar]) -> Tuple[str, float]:
        """Analyze overall volume trend"""
        if len(volume_bars) < 10:
            return 'neutral', 0.5
        
        # Calculate volume trend using linear regression
        volumes = [bar.volume for bar in volume_bars[-10:]]
        x_values = list(range(len(volumes)))
        
        try:
            # Simple linear regression
            n = len(volumes)
            sum_x = sum(x_values)
            sum_y = sum(volumes)
            sum_xy = sum(x * y for x, y in zip(x_values, volumes))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend
            if slope > 0.1:
                trend = 'increasing'
                strength = min(slope / 0.5, 1.0)
            elif slope < -0.1:
                trend = 'decreasing'
                strength = min(abs(slope) / 0.5, 1.0)
            else:
                trend = 'neutral'
                strength = 0.5
            
            return trend, max(0.1, min(0.9, strength))
            
        except ZeroDivisionError:
            return 'neutral', 0.5

    async def _generate_scalping_signals(self, symbol: str, volume_bars: List[VolumeBar],
                                       indicators: List[VolumeIndicator],
                                       timeframe: str) -> List[Dict[str, Union[str, float]]]:
        """Generate scalping signals based on volume analysis"""
        signals = []
        
        if len(volume_bars) < 5:
            return signals
        
        current_bar = volume_bars[-1]
        
        # Volume confirmation signals
        volume_spike_indicators = [ind for ind in indicators if 'Volume_SMA' in ind.name and ind.signal == 'bullish']
        volume_delta_bullish = [ind for ind in indicators if ind.name == 'Volume_Delta' and ind.signal == 'bullish']
        
        # Strong volume confirmation for scalping entry
        if len(volume_spike_indicators) >= 2 and volume_delta_bullish:
            avg_confidence = statistics.mean([ind.confidence for ind in volume_spike_indicators + volume_delta_bullish])
            
            signals.append({
                'type': 'volume_confirmation',
                'signal': 'buy',
                'confidence': avg_confidence,
                'volume_strength': statistics.mean([ind.strength for ind in volume_spike_indicators]),
                'entry_reason': 'volume_spike_with_delta',
                'timeframe': timeframe,
                'volume_multiplier': current_bar.volume / statistics.mean([bar.volume for bar in volume_bars[-10:-1]])
            })
        
        # Volume divergence signals
        volume_bearish_indicators = [ind for ind in indicators if ind.signal == 'bearish' and ind.strength > 0.7]
        if volume_bearish_indicators:
            avg_confidence = statistics.mean([ind.confidence for ind in volume_bearish_indicators])
            
            signals.append({
                'type': 'volume_divergence',
                'signal': 'sell',
                'confidence': avg_confidence,
                'volume_strength': statistics.mean([ind.strength for ind in volume_bearish_indicators]),
                'entry_reason': 'volume_exhaustion',
                'timeframe': timeframe,
                'volume_multiplier': current_bar.volume / statistics.mean([bar.volume for bar in volume_bars[-10:-1]])
            })
        
        # Tick volume signals
        tick_indicators = [ind for ind in indicators if ind.name == 'Tick_Volume' and ind.strength > 0.6]
        if tick_indicators:
            tick_ind = tick_indicators[0]
            
            signals.append({
                'type': 'tick_volume',
                'signal': 'buy' if tick_ind.signal == 'bullish' else 'sell',
                'confidence': tick_ind.confidence,
                'volume_strength': tick_ind.strength,
                'entry_reason': 'tick_volume_spike',
                'timeframe': timeframe,
                'tick_multiplier': current_bar.tick_count / statistics.mean([bar.tick_count for bar in volume_bars[-5:-1]])
            })
        
        return signals

    async def _calculate_execution_metrics(self, indicators: List[VolumeIndicator],
                                         signals: List[Dict]) -> Dict[str, float]:
        """Calculate execution metrics for volume analysis"""
        metrics = {
            'indicator_count': len(indicators),
            'signal_count': len(signals),
            'avg_indicator_strength': statistics.mean([ind.strength for ind in indicators]) if indicators else 0.0,
            'avg_signal_confidence': statistics.mean([sig['confidence'] for sig in signals]) if signals else 0.0,
            'bullish_indicators': len([ind for ind in indicators if ind.signal == 'bullish']),
            'bearish_indicators': len([ind for ind in indicators if ind.signal == 'bearish']),
            'volume_confirmation_strength': 0.0
        }
        
        # Calculate volume confirmation strength
        volume_confirmations = [sig for sig in signals if sig['type'] == 'volume_confirmation']
        if volume_confirmations:
            metrics['volume_confirmation_strength'] = statistics.mean([
                sig['volume_strength'] for sig in volume_confirmations
            ])
        
        return metrics

    def _generate_test_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate test data for initialization"""
        price_data = []
        volume_data = []
        base_price = 1.1000
        base_volume = 1000
        
        for i in range(30):
            # Create price movement
            price_change = (np.random.random() - 0.5) * 0.001
            price = base_price + price_change
            
            # Create volume with some spikes
            volume_multiplier = 2.0 if i % 10 == 0 else np.random.uniform(0.5, 1.5)
            volume = base_volume * volume_multiplier
            
            price_data.append({
                'timestamp': time.time() - (30 - i) * 60,  # M1 intervals
                'open': price,
                'high': price + 0.0002,
                'low': price - 0.0002,
                'close': price
            })
            
            volume_data.append({
                'timestamp': time.time() - (30 - i) * 60,
                'volume': volume,
                'tick_count': int(volume / 10)
            })
            
        return price_data, volume_data

    async def _analyze_volume_indicators(self, test_data: Tuple[List[Dict], List[Dict]]) -> List[VolumeIndicator]:
        """Test volume indicator analysis"""
        try:
            price_data, volume_data = test_data
            volume_bars = await self._prepare_volume_bars(price_data, volume_data)
            return await self._calculate_volume_indicators(volume_bars, 'M1')
        except Exception:
            return []
