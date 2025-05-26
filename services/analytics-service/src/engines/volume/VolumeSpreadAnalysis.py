"""
<<<<<<< HEAD
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
=======
Volume Spread Analysis (VSA) for Day Trading
Analyzes the relationship between volume and price spread for day trading signals.

This module implements VSA techniques for identifying smart money activity
and market manipulation patterns in M15-H1 timeframes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VSASignalType(Enum):
    """Volume Spread Analysis signal types"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    MARKUP = "markup"
    MARKDOWN = "markdown"
    NO_DEMAND = "no_demand"
    NO_SUPPLY = "no_supply"
    EFFORT_RESULT = "effort_result"
    STOPPING_VOLUME = "stopping_volume"

class VolumeStrength(Enum):
    """Volume strength classification"""
    ULTRA_HIGH = "ultra_high"
    HIGH = "high"
    AVERAGE = "average"
    LOW = "low"
    ULTRA_LOW = "ultra_low"

class SpreadSize(Enum):
    """Price spread size classification"""
    ULTRA_WIDE = "ultra_wide"
    WIDE = "wide"
    AVERAGE = "average"
    NARROW = "narrow"
    ULTRA_NARROW = "ultra_narrow"

@dataclass
class VSABar:
    """Individual VSA bar analysis"""
    timestamp: datetime
>>>>>>> 5e659b3064c215382ffc9ef1f13510cbfdd547a7
    open: float
    high: float
    low: float
    close: float
<<<<<<< HEAD
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
=======
    volume: int
    spread: float
    volume_strength: VolumeStrength
    spread_size: SpreadSize
    close_position: float  # Where close is in the range (0-1)
    signal_type: Optional[VSASignalType]
    confidence: float
    smart_money_activity: bool

@dataclass
class VSAAnalysisResult:
    """Complete VSA analysis result"""
    symbol: str
    timeframe: str
    analysis_time: datetime
    bars: List[VSABar]
    current_signal: Optional[VSASignalType]
    signal_confidence: float
    market_phase: str
    smart_money_direction: Optional[str]
    volume_trend: str
    spread_trend: str
    recommendations: List[str]

class VolumeSpreadAnalysis:
    """
    Volume Spread Analysis engine for day trading signals.

    Implements VSA methodology to identify:
    - Smart money accumulation/distribution
    - Market manipulation patterns
    - Supply and demand imbalances
    - Effort vs Result analysis
    """

    def __init__(self, lookback_periods: int = 50):
        """
        Initialize VSA analyzer.

        Args:
            lookback_periods: Number of periods for volume/spread analysis
        """
        self.lookback_periods = lookback_periods
        self.volume_percentiles = {}
        self.spread_percentiles = {}

    def analyze_vsa(self, data: pd.DataFrame, symbol: str, timeframe: str) -> VSAAnalysisResult:
        """
        Perform complete Volume Spread Analysis.

        Args:
            data: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            symbol: Trading symbol
            timeframe: Analysis timeframe

        Returns:
            VSAAnalysisResult with complete analysis
        """
        try:
            # Validate input data
            if len(data) < self.lookback_periods:
                raise ValueError(f"Insufficient data: need {self.lookback_periods}, got {len(data)}")

            # Calculate spreads and volume metrics
            data = self._calculate_metrics(data)

            # Classify volume and spread strength
            self._calculate_percentiles(data)

            # Analyze individual bars
            vsa_bars = []
            for i in range(len(data)):
                bar = self._analyze_bar(data.iloc[i], data.iloc[max(0, i-10):i+1])
                vsa_bars.append(bar)

            # Determine overall market analysis
            current_signal = self._determine_current_signal(vsa_bars[-10:])
            signal_confidence = self._calculate_signal_confidence(vsa_bars[-5:])
            market_phase = self._determine_market_phase(vsa_bars[-20:])
            smart_money_direction = self._analyze_smart_money_direction(vsa_bars[-15:])
            volume_trend = self._analyze_volume_trend(data.tail(20))
            spread_trend = self._analyze_spread_trend(data.tail(20))
            recommendations = self._generate_recommendations(current_signal, market_phase, signal_confidence)

            return VSAAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                analysis_time=datetime.now(),
                bars=vsa_bars,
                current_signal=current_signal,
                signal_confidence=signal_confidence,
                market_phase=market_phase,
                smart_money_direction=smart_money_direction,
                volume_trend=volume_trend,
                spread_trend=spread_trend,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"VSA analysis failed for {symbol}: {e}")
            raise

    def _calculate_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VSA metrics"""
        data = data.copy()

        # Calculate spread (high - low)
        data['spread'] = data['high'] - data['low']

        # Calculate close position in range (0 = low, 1 = high)
        data['close_position'] = (data['close'] - data['low']) / (data['spread'] + 1e-10)

        # Calculate volume moving averages
        data['volume_ma_10'] = data['volume'].rolling(10).mean()
        data['volume_ma_20'] = data['volume'].rolling(20).mean()

        # Calculate spread moving averages
        data['spread_ma_10'] = data['spread'].rolling(10).mean()
        data['spread_ma_20'] = data['spread'].rolling(20).mean()

        # Calculate relative volume
        data['relative_volume'] = data['volume'] / (data['volume_ma_20'] + 1e-10)

        # Calculate relative spread
        data['relative_spread'] = data['spread'] / (data['spread_ma_20'] + 1e-10)

        return data

    def _calculate_percentiles(self, data: pd.DataFrame):
        """Calculate volume and spread percentiles for classification"""
        recent_data = data.tail(self.lookback_periods)

        # Volume percentiles
        self.volume_percentiles = {
            'ultra_high': recent_data['volume'].quantile(0.95),
            'high': recent_data['volume'].quantile(0.80),
            'average': recent_data['volume'].quantile(0.50),
            'low': recent_data['volume'].quantile(0.20),
            'ultra_low': recent_data['volume'].quantile(0.05)
        }

        # Spread percentiles
        self.spread_percentiles = {
            'ultra_wide': recent_data['spread'].quantile(0.95),
            'wide': recent_data['spread'].quantile(0.80),
            'average': recent_data['spread'].quantile(0.50),
            'narrow': recent_data['spread'].quantile(0.20),
            'ultra_narrow': recent_data['spread'].quantile(0.05)
        }

    def _classify_volume_strength(self, volume: float) -> VolumeStrength:
        """Classify volume strength based on percentiles"""
        if volume >= self.volume_percentiles['ultra_high']:
            return VolumeStrength.ULTRA_HIGH
        elif volume >= self.volume_percentiles['high']:
            return VolumeStrength.HIGH
        elif volume >= self.volume_percentiles['low']:
            return VolumeStrength.AVERAGE
        elif volume >= self.volume_percentiles['ultra_low']:
            return VolumeStrength.LOW
        else:
            return VolumeStrength.ULTRA_LOW

    def _classify_spread_size(self, spread: float) -> SpreadSize:
        """Classify spread size based on percentiles"""
        if spread >= self.spread_percentiles['ultra_wide']:
            return SpreadSize.ULTRA_WIDE
        elif spread >= self.spread_percentiles['wide']:
            return SpreadSize.WIDE
        elif spread >= self.spread_percentiles['narrow']:
            return SpreadSize.AVERAGE
        elif spread >= self.spread_percentiles['ultra_narrow']:
            return SpreadSize.NARROW
        else:
            return SpreadSize.ULTRA_NARROW

    def _analyze_bar(self, current_bar: pd.Series, context_data: pd.DataFrame) -> VSABar:
        """Analyze individual bar for VSA signals"""
        volume_strength = self._classify_volume_strength(current_bar['volume'])
        spread_size = self._classify_spread_size(current_bar['spread'])
        close_position = current_bar['close_position']

        # Determine VSA signal type
        signal_type = self._determine_vsa_signal(volume_strength, spread_size, close_position, current_bar)

        # Calculate confidence based on signal clarity
        confidence = self._calculate_bar_confidence(volume_strength, spread_size, close_position, signal_type)

        # Detect smart money activity
        smart_money_activity = self._detect_smart_money_activity(volume_strength, spread_size, close_position)

        return VSABar(
            timestamp=current_bar['timestamp'],
            open=current_bar['open'],
            high=current_bar['high'],
            low=current_bar['low'],
            close=current_bar['close'],
            volume=current_bar['volume'],
            spread=current_bar['spread'],
            volume_strength=volume_strength,
            spread_size=spread_size,
            close_position=close_position,
            signal_type=signal_type,
            confidence=confidence,
            smart_money_activity=smart_money_activity
        )

    def _determine_vsa_signal(self, volume_strength: VolumeStrength, spread_size: SpreadSize,
                             close_position: float, bar_data: pd.Series) -> Optional[VSASignalType]:
        """Determine VSA signal type based on volume, spread, and close position"""

        # High volume + narrow spread = potential accumulation/distribution
        if volume_strength in [VolumeStrength.HIGH, VolumeStrength.ULTRA_HIGH]:
            if spread_size in [SpreadSize.NARROW, SpreadSize.ULTRA_NARROW]:
                if close_position > 0.7:
                    return VSASignalType.ACCUMULATION
                elif close_position < 0.3:
                    return VSASignalType.DISTRIBUTION

        # High volume + wide spread = markup/markdown
        if volume_strength in [VolumeStrength.HIGH, VolumeStrength.ULTRA_HIGH]:
            if spread_size in [SpreadSize.WIDE, SpreadSize.ULTRA_WIDE]:
                if close_position > 0.7:
                    return VSASignalType.MARKUP
                elif close_position < 0.3:
                    return VSASignalType.MARKDOWN

        # Low volume + narrow spread = no demand/supply
        if volume_strength in [VolumeStrength.LOW, VolumeStrength.ULTRA_LOW]:
            if spread_size in [SpreadSize.NARROW, SpreadSize.ULTRA_NARROW]:
                if bar_data['close'] > bar_data['open']:
                    return VSASignalType.NO_SUPPLY
                else:
                    return VSASignalType.NO_DEMAND

        # Ultra high volume = potential stopping volume
        if volume_strength == VolumeStrength.ULTRA_HIGH:
            if close_position < 0.5:  # Close in lower half
                return VSASignalType.STOPPING_VOLUME

        return None

    def _calculate_bar_confidence(self, volume_strength: VolumeStrength, spread_size: SpreadSize,
                                 close_position: float, signal_type: Optional[VSASignalType]) -> float:
        """Calculate confidence score for VSA signal"""
        if signal_type is None:
            return 0.0

        confidence = 0.5  # Base confidence

        # Volume strength contribution
        volume_scores = {
            VolumeStrength.ULTRA_HIGH: 0.3,
            VolumeStrength.HIGH: 0.2,
            VolumeStrength.AVERAGE: 0.1,
            VolumeStrength.LOW: 0.05,
            VolumeStrength.ULTRA_LOW: 0.0
        }
        confidence += volume_scores.get(volume_strength, 0.0)

        # Close position clarity (extreme positions are more reliable)
        if close_position > 0.8 or close_position < 0.2:
            confidence += 0.15
        elif close_position > 0.7 or close_position < 0.3:
            confidence += 0.1

        # Signal type specific adjustments
        if signal_type in [VSASignalType.ACCUMULATION, VSASignalType.DISTRIBUTION]:
            confidence += 0.05  # These are high-confidence signals

        return min(confidence, 1.0)

    def _detect_smart_money_activity(self, volume_strength: VolumeStrength, spread_size: SpreadSize,
                                   close_position: float) -> bool:
        """Detect potential smart money activity"""
        # High volume with controlled price movement suggests smart money
        if volume_strength in [VolumeStrength.HIGH, VolumeStrength.ULTRA_HIGH]:
            if spread_size in [SpreadSize.NARROW, SpreadSize.ULTRA_NARROW]:
                return True

        # Extreme close positions with high volume
        if volume_strength == VolumeStrength.ULTRA_HIGH:
            if close_position > 0.9 or close_position < 0.1:
                return True

        return False

    def _determine_current_signal(self, recent_bars: List[VSABar]) -> Optional[VSASignalType]:
        """Determine current market signal from recent bars"""
        if not recent_bars:
            return None

        # Count signal types in recent bars
        signal_counts = {}
        for bar in recent_bars:
            if bar.signal_type:
                signal_counts[bar.signal_type] = signal_counts.get(bar.signal_type, 0) + 1

        if not signal_counts:
            return None

        # Return most frequent signal
        return max(signal_counts, key=signal_counts.get)

    def _calculate_signal_confidence(self, recent_bars: List[VSABar]) -> float:
        """Calculate overall signal confidence"""
        if not recent_bars:
            return 0.0

        confidences = [bar.confidence for bar in recent_bars if bar.signal_type]
        return np.mean(confidences) if confidences else 0.0

    def _determine_market_phase(self, bars: List[VSABar]) -> str:
        """Determine current market phase"""
        if not bars:
            return "unknown"

        accumulation_count = sum(1 for bar in bars if bar.signal_type == VSASignalType.ACCUMULATION)
        distribution_count = sum(1 for bar in bars if bar.signal_type == VSASignalType.DISTRIBUTION)
        markup_count = sum(1 for bar in bars if bar.signal_type == VSASignalType.MARKUP)
        markdown_count = sum(1 for bar in bars if bar.signal_type == VSASignalType.MARKDOWN)

        if accumulation_count > distribution_count and accumulation_count > 2:
            return "accumulation"
        elif distribution_count > accumulation_count and distribution_count > 2:
            return "distribution"
        elif markup_count > markdown_count and markup_count > 1:
            return "markup"
        elif markdown_count > markup_count and markdown_count > 1:
            return "markdown"
        else:
            return "consolidation"

    def _analyze_smart_money_direction(self, bars: List[VSABar]) -> Optional[str]:
        """Analyze smart money direction"""
        smart_money_bars = [bar for bar in bars if bar.smart_money_activity]

        if not smart_money_bars:
            return None

        bullish_signals = sum(1 for bar in smart_money_bars
                            if bar.signal_type in [VSASignalType.ACCUMULATION, VSASignalType.MARKUP])
        bearish_signals = sum(1 for bar in smart_money_bars
                            if bar.signal_type in [VSASignalType.DISTRIBUTION, VSASignalType.MARKDOWN])

        if bullish_signals > bearish_signals:
            return "bullish"
        elif bearish_signals > bullish_signals:
            return "bearish"
        else:
            return "neutral"

    def _analyze_volume_trend(self, data: pd.DataFrame) -> str:
        """Analyze volume trend"""
        if len(data) < 10:
            return "insufficient_data"

        recent_volume = data['volume'].tail(10).mean()
        older_volume = data['volume'].head(10).mean()

        if recent_volume > older_volume * 1.2:
            return "increasing"
        elif recent_volume < older_volume * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _analyze_spread_trend(self, data: pd.DataFrame) -> str:
        """Analyze spread trend"""
        if len(data) < 10:
            return "insufficient_data"

        recent_spread = data['spread'].tail(10).mean()
        older_spread = data['spread'].head(10).mean()

        if recent_spread > older_spread * 1.2:
            return "widening"
        elif recent_spread < older_spread * 0.8:
            return "narrowing"
        else:
            return "stable"

    def _generate_recommendations(self, signal: Optional[VSASignalType], phase: str, confidence: float) -> List[str]:
        """Generate trading recommendations based on VSA analysis"""
        recommendations = []

        if confidence < 0.3:
            recommendations.append("Low confidence signals - wait for clearer setup")
            return recommendations

        if signal == VSASignalType.ACCUMULATION:
            recommendations.append("Potential accumulation detected - look for long opportunities")
            recommendations.append("Watch for breakout above resistance with volume confirmation")
        elif signal == VSASignalType.DISTRIBUTION:
            recommendations.append("Potential distribution detected - look for short opportunities")
            recommendations.append("Watch for breakdown below support with volume confirmation")
        elif signal == VSASignalType.MARKUP:
            recommendations.append("Markup phase - trend continuation likely")
            recommendations.append("Look for pullback entries in direction of trend")
        elif signal == VSASignalType.MARKDOWN:
            recommendations.append("Markdown phase - downtrend continuation likely")
            recommendations.append("Look for bounce entries to short")
        elif signal == VSASignalType.NO_DEMAND:
            recommendations.append("No demand detected - weakness in uptrend")
            recommendations.append("Consider taking profits on long positions")
        elif signal == VSASignalType.NO_SUPPLY:
            recommendations.append("No supply detected - strength in downtrend")
            recommendations.append("Consider taking profits on short positions")
        elif signal == VSASignalType.STOPPING_VOLUME:
            recommendations.append("Stopping volume detected - potential reversal")
            recommendations.append("Wait for confirmation before entering new positions")

        # Phase-specific recommendations
        if phase == "consolidation":
            recommendations.append("Market in consolidation - trade range or wait for breakout")

        return recommendations
>>>>>>> 5e659b3064c215382ffc9ef1f13510cbfdd547a7
