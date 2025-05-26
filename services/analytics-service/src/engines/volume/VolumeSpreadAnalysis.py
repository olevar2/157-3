"""
Volume Spread Analysis Module
VSA for day trading - analyzing volume and spread relationships
Optimized for smart money flow detection and day trading signals.
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
class VSABar:
    """Volume Spread Analysis bar"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float  # high - low
    close_position: float  # (close - low) / spread
    volume_spread_ratio: float


@dataclass
class VSASignal:
    """VSA signal definition"""
    signal_type: str  # 'accumulation', 'distribution', 'markup', 'markdown'
    strength: str  # 'weak', 'medium', 'strong'
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    volume_characteristic: str
    spread_characteristic: str
    price_action: str


@dataclass
class SmartMoneyActivity:
    """Smart money activity detection"""
    activity_type: str  # 'buying', 'selling', 'neutral'
    intensity: float  # 0-1
    volume_evidence: float
    spread_evidence: float
    price_evidence: float
    overall_confidence: float


@dataclass
class VolumeSpreadResult:
    """Volume Spread Analysis result"""
    symbol: str
    timestamp: float
    timeframe: str
    vsa_signals: List[VSASignal]
    smart_money_activity: SmartMoneyActivity
    market_phase: str  # 'accumulation', 'markup', 'distribution', 'markdown'
    volume_analysis: Dict[str, float]
    spread_analysis: Dict[str, float]
    trading_signals: List[Dict[str, Union[str, float]]]


class VolumeSpreadAnalysis:
    """
    Volume Spread Analysis Engine for Day Trading
    Provides VSA analysis for smart money detection and day trading signals
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # VSA analysis parameters
        self.volume_threshold_high = 2.0  # High volume threshold
        self.volume_threshold_low = 0.5   # Low volume threshold
        self.spread_threshold_high = 1.5  # High spread threshold
        self.spread_threshold_low = 0.7   # Low spread threshold
        
        # Close position thresholds
        self.close_high_threshold = 0.7   # Close in upper 30%
        self.close_low_threshold = 0.3    # Close in lower 30%
        
        # Smart money detection parameters
        self.smart_money_volume_threshold = 1.8
        self.smart_money_confidence_threshold = 0.7
        
        # Performance optimization
        self.vsa_cache: Dict[str, deque] = {}
        self.analysis_cache: Dict[str, Dict] = {}

    async def initialize(self) -> bool:
        """Initialize the Volume Spread Analysis engine"""
        try:
            self.logger.info("Initializing Volume Spread Analysis Engine...")
            
            # Test VSA analysis with sample data
            test_data = self._generate_test_data()
            test_result = await self._analyze_vsa_patterns(test_data)
            
            if test_result and len(test_result) > 0:
                self.ready = True
                self.logger.info("✅ Volume Spread Analysis Engine initialized")
                return True
            else:
                raise Exception("VSA analysis test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Volume Spread Analysis Engine initialization failed: {e}")
            return False

    async def analyze_volume_spread(self, symbol: str, price_data: List[Dict], 
                                  volume_data: List[Dict],
                                  timeframe: str = 'M15') -> VolumeSpreadResult:
        """
        Analyze volume and spread relationships for day trading
        
        Args:
            symbol: Currency pair symbol
            price_data: List of OHLC data dictionaries
            volume_data: List of volume data dictionaries
            timeframe: Chart timeframe (M15-H1)
            
        Returns:
            VolumeSpreadResult with VSA analysis
        """
        if not self.ready:
            raise Exception("Volume Spread Analysis Engine not initialized")
            
        if len(price_data) < 20 or len(volume_data) < 20:
            raise Exception("Insufficient data for VSA analysis (minimum 20 periods)")
            
        try:
            start_time = time.time()
            
            # Prepare VSA bars
            vsa_bars = await self._prepare_vsa_bars(price_data, volume_data)
            
            # Analyze VSA patterns
            vsa_signals = await self._analyze_vsa_patterns(vsa_bars)
            
            # Detect smart money activity
            smart_money_activity = await self._detect_smart_money_activity(vsa_bars)
            
            # Determine market phase
            market_phase = await self._determine_market_phase(vsa_bars, vsa_signals)
            
            # Analyze volume characteristics
            volume_analysis = await self._analyze_volume_characteristics(vsa_bars)
            
            # Analyze spread characteristics
            spread_analysis = await self._analyze_spread_characteristics(vsa_bars)
            
            # Generate trading signals
            trading_signals = await self._generate_trading_signals(
                symbol, vsa_bars, vsa_signals, smart_money_activity, timeframe
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"VSA analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return VolumeSpreadResult(
                symbol=symbol,
                timestamp=time.time(),
                timeframe=timeframe,
                vsa_signals=vsa_signals,
                smart_money_activity=smart_money_activity,
                market_phase=market_phase,
                volume_analysis=volume_analysis,
                spread_analysis=spread_analysis,
                trading_signals=trading_signals
            )
            
        except Exception as e:
            self.logger.error(f"VSA analysis failed for {symbol}: {e}")
            raise

    async def _prepare_vsa_bars(self, price_data: List[Dict], 
                              volume_data: List[Dict]) -> List[VSABar]:
        """Prepare VSA bars from price and volume data"""
        vsa_bars = []
        
        # Ensure data alignment
        min_length = min(len(price_data), len(volume_data))
        
        for i in range(min_length):
            price_bar = price_data[i]
            volume_bar = volume_data[i]
            
            timestamp = float(price_bar.get('timestamp', time.time()))
            open_price = float(price_bar.get('open', 0))
            high_price = float(price_bar.get('high', 0))
            low_price = float(price_bar.get('low', 0))
            close_price = float(price_bar.get('close', 0))
            volume = float(volume_bar.get('volume', 0))
            
            # Calculate spread
            spread = high_price - low_price
            
            # Calculate close position within the bar
            close_position = (close_price - low_price) / spread if spread > 0 else 0.5
            
            # Calculate volume to spread ratio
            volume_spread_ratio = volume / spread if spread > 0 else 0
            
            vsa_bars.append(VSABar(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                spread=spread,
                close_position=close_position,
                volume_spread_ratio=volume_spread_ratio
            ))
        
        return vsa_bars

    async def _analyze_vsa_patterns(self, vsa_bars: List[VSABar]) -> List[VSASignal]:
        """Analyze VSA patterns and generate signals"""
        signals = []
        
        if len(vsa_bars) < 10:
            return signals
        
        # Calculate average volume and spread for comparison
        avg_volume = statistics.mean([bar.volume for bar in vsa_bars[-20:]])
        avg_spread = statistics.mean([bar.spread for bar in vsa_bars[-20:]])
        
        # Analyze recent bars
        for i in range(-5, 0):  # Last 5 bars
            if abs(i) > len(vsa_bars):
                continue
                
            bar = vsa_bars[i]
            
            # Classify volume
            if bar.volume > avg_volume * self.volume_threshold_high:
                volume_char = 'high'
            elif bar.volume < avg_volume * self.volume_threshold_low:
                volume_char = 'low'
            else:
                volume_char = 'average'
            
            # Classify spread
            if bar.spread > avg_spread * self.spread_threshold_high:
                spread_char = 'wide'
            elif bar.spread < avg_spread * self.spread_threshold_low:
                spread_char = 'narrow'
            else:
                spread_char = 'average'
            
            # Classify price action
            if bar.close_position > self.close_high_threshold:
                price_action = 'close_high'
            elif bar.close_position < self.close_low_threshold:
                price_action = 'close_low'
            else:
                price_action = 'close_mid'
            
            # Generate VSA signals based on combinations
            vsa_signal = await self._interpret_vsa_combination(
                volume_char, spread_char, price_action, bar
            )
            
            if vsa_signal:
                signals.append(vsa_signal)
        
        return signals

    async def _interpret_vsa_combination(self, volume_char: str, spread_char: str, 
                                       price_action: str, bar: VSABar) -> Optional[VSASignal]:
        """Interpret VSA combination and generate signal"""
        
        # High Volume + Wide Spread + Close High = Bullish (Professional Buying)
        if volume_char == 'high' and spread_char == 'wide' and price_action == 'close_high':
            return VSASignal(
                signal_type='accumulation',
                strength='strong',
                direction='bullish',
                confidence=0.85,
                volume_characteristic=volume_char,
                spread_characteristic=spread_char,
                price_action=price_action
            )
        
        # High Volume + Wide Spread + Close Low = Bearish (Professional Selling)
        elif volume_char == 'high' and spread_char == 'wide' and price_action == 'close_low':
            return VSASignal(
                signal_type='distribution',
                strength='strong',
                direction='bearish',
                confidence=0.85,
                volume_characteristic=volume_char,
                spread_characteristic=spread_char,
                price_action=price_action
            )
        
        # High Volume + Narrow Spread = Absorption (Smart Money)
        elif volume_char == 'high' and spread_char == 'narrow':
            direction = 'bullish' if price_action == 'close_high' else 'bearish'
            return VSASignal(
                signal_type='accumulation' if direction == 'bullish' else 'distribution',
                strength='medium',
                direction=direction,
                confidence=0.75,
                volume_characteristic=volume_char,
                spread_characteristic=spread_char,
                price_action=price_action
            )
        
        # Low Volume + Wide Spread = Weakness
        elif volume_char == 'low' and spread_char == 'wide':
            return VSASignal(
                signal_type='markdown' if price_action == 'close_low' else 'markup',
                strength='weak',
                direction='bearish' if price_action == 'close_low' else 'neutral',
                confidence=0.6,
                volume_characteristic=volume_char,
                spread_characteristic=spread_char,
                price_action=price_action
            )
        
        # Low Volume + Narrow Spread = No Interest
        elif volume_char == 'low' and spread_char == 'narrow':
            return VSASignal(
                signal_type='accumulation',
                strength='weak',
                direction='neutral',
                confidence=0.4,
                volume_characteristic=volume_char,
                spread_characteristic=spread_char,
                price_action=price_action
            )
        
        # Average combinations
        else:
            return VSASignal(
                signal_type='markup' if bar.close > bar.open else 'markdown',
                strength='medium',
                direction='neutral',
                confidence=0.5,
                volume_characteristic=volume_char,
                spread_characteristic=spread_char,
                price_action=price_action
            )

    async def _detect_smart_money_activity(self, vsa_bars: List[VSABar]) -> SmartMoneyActivity:
        """Detect smart money activity based on VSA principles"""
        if len(vsa_bars) < 10:
            return SmartMoneyActivity(
                activity_type='neutral',
                intensity=0.5,
                volume_evidence=0.5,
                spread_evidence=0.5,
                price_evidence=0.5,
                overall_confidence=0.5
            )
        
        recent_bars = vsa_bars[-10:]
        avg_volume = statistics.mean([bar.volume for bar in vsa_bars[-20:]])
        
        # Volume evidence
        high_volume_bars = [bar for bar in recent_bars if bar.volume > avg_volume * self.smart_money_volume_threshold]
        volume_evidence = len(high_volume_bars) / len(recent_bars)
        
        # Spread evidence (smart money often creates narrow spreads on high volume)
        avg_spread = statistics.mean([bar.spread for bar in vsa_bars[-20:]])
        narrow_spread_high_volume = [
            bar for bar in recent_bars 
            if bar.volume > avg_volume * 1.5 and bar.spread < avg_spread * 0.8
        ]
        spread_evidence = len(narrow_spread_high_volume) / len(recent_bars)
        
        # Price evidence (closes in favorable positions)
        favorable_closes = [
            bar for bar in recent_bars 
            if (bar.close > bar.open and bar.close_position > 0.6) or 
               (bar.close < bar.open and bar.close_position < 0.4)
        ]
        price_evidence = len(favorable_closes) / len(recent_bars)
        
        # Determine activity type
        bullish_evidence = sum([
            bar.close_position > 0.7 and bar.volume > avg_volume * 1.5 
            for bar in recent_bars
        ])
        bearish_evidence = sum([
            bar.close_position < 0.3 and bar.volume > avg_volume * 1.5 
            for bar in recent_bars
        ])
        
        if bullish_evidence > bearish_evidence:
            activity_type = 'buying'
            intensity = bullish_evidence / len(recent_bars)
        elif bearish_evidence > bullish_evidence:
            activity_type = 'selling'
            intensity = bearish_evidence / len(recent_bars)
        else:
            activity_type = 'neutral'
            intensity = 0.5
        
        # Overall confidence
        overall_confidence = (volume_evidence + spread_evidence + price_evidence) / 3
        
        return SmartMoneyActivity(
            activity_type=activity_type,
            intensity=intensity,
            volume_evidence=volume_evidence,
            spread_evidence=spread_evidence,
            price_evidence=price_evidence,
            overall_confidence=overall_confidence
        )

    async def _determine_market_phase(self, vsa_bars: List[VSABar], 
                                    vsa_signals: List[VSASignal]) -> str:
        """Determine current market phase based on VSA analysis"""
        if not vsa_signals:
            return 'neutral'
        
        # Count signal types
        accumulation_signals = [s for s in vsa_signals if s.signal_type == 'accumulation']
        distribution_signals = [s for s in vsa_signals if s.signal_type == 'distribution']
        markup_signals = [s for s in vsa_signals if s.signal_type == 'markup']
        markdown_signals = [s for s in vsa_signals if s.signal_type == 'markdown']
        
        # Determine dominant phase
        phase_counts = {
            'accumulation': len(accumulation_signals),
            'distribution': len(distribution_signals),
            'markup': len(markup_signals),
            'markdown': len(markdown_signals)
        }
        
        dominant_phase = max(phase_counts, key=phase_counts.get)
        
        # Validate with price action
        if len(vsa_bars) >= 10:
            recent_price_change = (vsa_bars[-1].close - vsa_bars[-10].close) / vsa_bars[-10].close
            
            if recent_price_change > 0.01 and dominant_phase in ['accumulation', 'markup']:
                return dominant_phase
            elif recent_price_change < -0.01 and dominant_phase in ['distribution', 'markdown']:
                return dominant_phase
        
        return dominant_phase

    async def _analyze_volume_characteristics(self, vsa_bars: List[VSABar]) -> Dict[str, float]:
        """Analyze volume characteristics"""
        if len(vsa_bars) < 10:
            return {}
        
        volumes = [bar.volume for bar in vsa_bars]
        
        return {
            'avg_volume': statistics.mean(volumes),
            'volume_volatility': statistics.stdev(volumes) if len(volumes) > 1 else 0,
            'volume_trend': self._calculate_trend(volumes),
            'high_volume_ratio': len([v for v in volumes[-10:] if v > statistics.mean(volumes) * 1.5]) / 10,
            'volume_consistency': 1 - (statistics.stdev(volumes[-10:]) / statistics.mean(volumes[-10:])) if len(volumes) >= 10 else 0
        }

    async def _analyze_spread_characteristics(self, vsa_bars: List[VSABar]) -> Dict[str, float]:
        """Analyze spread characteristics"""
        if len(vsa_bars) < 10:
            return {}
        
        spreads = [bar.spread for bar in vsa_bars]
        
        return {
            'avg_spread': statistics.mean(spreads),
            'spread_volatility': statistics.stdev(spreads) if len(spreads) > 1 else 0,
            'spread_trend': self._calculate_trend(spreads),
            'narrow_spread_ratio': len([s for s in spreads[-10:] if s < statistics.mean(spreads) * 0.8]) / 10,
            'spread_efficiency': statistics.mean([bar.volume / bar.spread for bar in vsa_bars[-10:] if bar.spread > 0])
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using simple linear regression"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        try:
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0

    async def _generate_trading_signals(self, symbol: str, vsa_bars: List[VSABar],
                                      vsa_signals: List[VSASignal],
                                      smart_money: SmartMoneyActivity,
                                      timeframe: str) -> List[Dict[str, Union[str, float]]]:
        """Generate trading signals based on VSA analysis"""
        signals = []
        
        if not vsa_signals or len(vsa_bars) < 5:
            return signals
        
        current_bar = vsa_bars[-1]
        
        # Strong accumulation signals
        strong_accumulation = [s for s in vsa_signals if s.signal_type == 'accumulation' and s.strength == 'strong']
        if strong_accumulation and smart_money.activity_type == 'buying':
            signals.append({
                'type': 'vsa_accumulation',
                'signal': 'buy',
                'confidence': statistics.mean([s.confidence for s in strong_accumulation]),
                'smart_money_confidence': smart_money.overall_confidence,
                'entry_reason': 'professional_buying_detected',
                'timeframe': timeframe,
                'volume_strength': smart_money.volume_evidence
            })
        
        # Strong distribution signals
        strong_distribution = [s for s in vsa_signals if s.signal_type == 'distribution' and s.strength == 'strong']
        if strong_distribution and smart_money.activity_type == 'selling':
            signals.append({
                'type': 'vsa_distribution',
                'signal': 'sell',
                'confidence': statistics.mean([s.confidence for s in strong_distribution]),
                'smart_money_confidence': smart_money.overall_confidence,
                'entry_reason': 'professional_selling_detected',
                'timeframe': timeframe,
                'volume_strength': smart_money.volume_evidence
            })
        
        # Volume absorption signals
        if smart_money.spread_evidence > 0.6 and smart_money.volume_evidence > 0.7:
            signals.append({
                'type': 'volume_absorption',
                'signal': 'buy' if smart_money.activity_type == 'buying' else 'sell',
                'confidence': smart_money.overall_confidence,
                'smart_money_confidence': smart_money.overall_confidence,
                'entry_reason': 'volume_absorption_detected',
                'timeframe': timeframe,
                'absorption_strength': smart_money.spread_evidence
            })
        
        return signals

    def _generate_test_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate test data for initialization"""
        price_data = []
        volume_data = []
        base_price = 1.1000
        base_volume = 1000
        
        for i in range(30):
            # Create price movement with varying spreads
            price_change = (np.random.random() - 0.5) * 0.002
            spread_factor = np.random.uniform(0.5, 2.0)
            
            open_price = base_price + price_change
            high_price = open_price + (0.0005 * spread_factor)
            low_price = open_price - (0.0005 * spread_factor)
            close_price = low_price + (high_price - low_price) * np.random.random()
            
            # Create volume with VSA characteristics
            if i % 8 == 0:  # Occasional high volume
                volume = base_volume * np.random.uniform(2.0, 3.0)
            else:
                volume = base_volume * np.random.uniform(0.5, 1.5)
            
            price_data.append({
                'timestamp': time.time() - (30 - i) * 900,  # M15 intervals
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            })
            
            volume_data.append({
                'timestamp': time.time() - (30 - i) * 900,
                'volume': volume
            })
            
        return price_data, volume_data
