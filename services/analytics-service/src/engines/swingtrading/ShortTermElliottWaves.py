"""
<<<<<<< HEAD
Short-Term Elliott Waves Module
3-5 wave structures for quick swing trades (H4 focus, max 3-5 days)
Optimized for rapid Elliott wave pattern recognition for swing trading entries.
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
class WavePoint:
    """Elliott wave point definition"""
    index: int
    price: float
    timestamp: float
    wave_type: str  # 'impulse', 'corrective'
    wave_number: int  # 1, 2, 3, 4, 5 for impulse; A, B, C for corrective
    confidence: float


@dataclass
class ElliottWavePattern:
    """Complete Elliott wave pattern"""
    pattern_type: str  # '5-wave-impulse', '3-wave-corrective', 'truncated'
    direction: str  # 'bullish', 'bearish'
    waves: List[WavePoint]
    start_time: float
    end_time: float
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    completion_percentage: float


@dataclass
class ShortTermElliottResult:
    """Short-term Elliott wave analysis result"""
    symbol: str
    timestamp: float
    timeframe: str
    current_wave: Optional[WavePoint]
    active_patterns: List[ElliottWavePattern]
    potential_patterns: List[ElliottWavePattern]
    wave_targets: Dict[str, float]
    fibonacci_levels: Dict[str, float]
    execution_signals: List[Dict[str, Union[str, float]]]


class ShortTermElliottWaves:
    """
    Short-Term Elliott Waves Engine for Swing Trading
    Provides 3-5 wave structure detection for quick swing trades (max 5 days)
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # Elliott wave configuration for short-term trading
        self.min_wave_periods = 8  # Minimum periods for wave recognition
        self.max_wave_periods = 120  # Maximum periods (5 days on H4)
        self.min_wave_size = 0.0015  # Minimum wave size (15 pips for major pairs)
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        
        # Wave validation thresholds
        self.wave_3_min_ratio = 1.0  # Wave 3 must be at least equal to wave 1
        self.wave_5_max_ratio = 1.618  # Wave 5 should not exceed 1.618 * wave 1
        self.correction_max_ratio = 0.786  # Corrections should not exceed 78.6%
        
        # Pattern storage for quick access
        self.active_patterns: Dict[str, List[ElliottWavePattern]] = {}
        self.wave_cache: Dict[str, deque] = {}

    async def initialize(self) -> bool:
        """Initialize the Elliott waves engine"""
        try:
            self.logger.info("Initializing Short-Term Elliott Waves Engine...")
            
            # Test wave detection with sample data
            test_data = self._generate_test_data()
            test_result = await self._detect_waves(test_data)
            
            if test_result:
                self.ready = True
                self.logger.info("✅ Short-Term Elliott Waves Engine initialized")
                return True
            else:
                raise Exception("Wave detection test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Elliott Waves Engine initialization failed: {e}")
            return False

    async def analyze_elliott_waves(self, symbol: str, price_data: List[Dict], 
                                  timeframe: str = 'H4') -> ShortTermElliottResult:
        """
        Analyze Elliott wave patterns for short-term swing trading
        
        Args:
            symbol: Currency pair symbol
            price_data: List of OHLC data dictionaries
            timeframe: Chart timeframe (default H4)
            
        Returns:
            ShortTermElliottResult with wave analysis
        """
        if not self.ready:
            raise Exception("Elliott Waves Engine not initialized")
            
        if len(price_data) < 50:
            raise Exception("Insufficient data for Elliott wave analysis (minimum 50 periods)")
            
        try:
            start_time = time.time()
            
            # Extract price arrays
            closes = [float(data.get('close', 0)) for data in price_data]
            highs = [float(data.get('high', 0)) for data in price_data]
            lows = [float(data.get('low', 0)) for data in price_data]
            timestamps = [float(data.get('timestamp', time.time())) for data in price_data]
            
            # Detect swing points for wave analysis
            swing_points = await self._detect_swing_points(highs, lows, closes, timestamps)
            
            # Identify Elliott wave patterns
            active_patterns = await self._identify_wave_patterns(swing_points, 'active')
            potential_patterns = await self._identify_wave_patterns(swing_points, 'potential')
            
            # Determine current wave position
            current_wave = await self._determine_current_wave(swing_points, active_patterns)
            
            # Calculate wave targets and Fibonacci levels
            wave_targets = await self._calculate_wave_targets(active_patterns, closes[-1])
            fibonacci_levels = await self._calculate_fibonacci_levels(swing_points)
            
            # Generate execution signals
            execution_signals = await self._generate_execution_signals(
                symbol, closes[-1], current_wave, active_patterns, wave_targets
            )
            
            # Cache results for performance
            self.active_patterns[symbol] = active_patterns
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Elliott wave analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return ShortTermElliottResult(
                symbol=symbol,
                timestamp=time.time(),
                timeframe=timeframe,
                current_wave=current_wave,
                active_patterns=active_patterns,
                potential_patterns=potential_patterns,
                wave_targets=wave_targets,
                fibonacci_levels=fibonacci_levels,
                execution_signals=execution_signals
            )
            
        except Exception as e:
            self.logger.error(f"Elliott wave analysis failed for {symbol}: {e}")
            raise

    async def _detect_swing_points(self, highs: List[float], lows: List[float], 
                                 closes: List[float], timestamps: List[float]) -> List[WavePoint]:
        """Detect swing highs and lows for wave analysis"""
        swing_points = []
        lookback = 5  # Periods to look back for swing detection
        
        for i in range(lookback, len(highs) - lookback):
            # Check for swing high
            is_swing_high = all(highs[i] >= highs[j] for j in range(i - lookback, i + lookback + 1) if j != i)
            if is_swing_high:
                swing_points.append(WavePoint(
                    index=i,
                    price=highs[i],
                    timestamp=timestamps[i],
                    wave_type='peak',
                    wave_number=0,
                    confidence=0.8
                ))
            
            # Check for swing low
            is_swing_low = all(lows[i] <= lows[j] for j in range(i - lookback, i + lookback + 1) if j != i)
            if is_swing_low:
                swing_points.append(WavePoint(
                    index=i,
                    price=lows[i],
                    timestamp=timestamps[i],
                    wave_type='trough',
                    wave_number=0,
                    confidence=0.8
                ))
        
        # Sort by index
        swing_points.sort(key=lambda x: x.index)
        return swing_points

    async def _identify_wave_patterns(self, swing_points: List[WavePoint], 
                                    pattern_type: str) -> List[ElliottWavePattern]:
        """Identify Elliott wave patterns from swing points"""
        patterns = []
        
        if len(swing_points) < 5:
            return patterns
            
        # Look for 5-wave impulse patterns
        for i in range(len(swing_points) - 4):
            wave_sequence = swing_points[i:i+5]
            
            # Check if this could be a 5-wave pattern
            if self._validate_5_wave_pattern(wave_sequence):
                pattern = ElliottWavePattern(
                    pattern_type='5-wave-impulse',
                    direction='bullish' if wave_sequence[0].price < wave_sequence[4].price else 'bearish',
                    waves=wave_sequence,
                    start_time=wave_sequence[0].timestamp,
                    end_time=wave_sequence[4].timestamp,
                    confidence=self._calculate_pattern_confidence(wave_sequence),
                    target_price=self._calculate_wave_target(wave_sequence),
                    stop_loss=self._calculate_stop_loss(wave_sequence),
                    completion_percentage=100.0 if pattern_type == 'active' else 80.0
                )
                patterns.append(pattern)
        
        # Look for 3-wave corrective patterns
        for i in range(len(swing_points) - 2):
            wave_sequence = swing_points[i:i+3]
            
            if self._validate_3_wave_pattern(wave_sequence):
                pattern = ElliottWavePattern(
                    pattern_type='3-wave-corrective',
                    direction='corrective',
                    waves=wave_sequence,
                    start_time=wave_sequence[0].timestamp,
                    end_time=wave_sequence[2].timestamp,
                    confidence=self._calculate_pattern_confidence(wave_sequence),
                    target_price=self._calculate_correction_target(wave_sequence),
                    stop_loss=self._calculate_stop_loss(wave_sequence),
                    completion_percentage=100.0 if pattern_type == 'active' else 70.0
                )
                patterns.append(pattern)
        
        return patterns

    def _validate_5_wave_pattern(self, waves: List[WavePoint]) -> bool:
        """Validate if swing points form a valid 5-wave Elliott pattern"""
        if len(waves) != 5:
            return False
            
        # Check alternating pattern (high-low-high-low-high or low-high-low-high-low)
        wave_types = [w.wave_type for w in waves]
        
        # Bullish pattern: trough-peak-trough-peak-trough
        bullish_pattern = ['trough', 'peak', 'trough', 'peak', 'trough']
        # Bearish pattern: peak-trough-peak-trough-peak
        bearish_pattern = ['peak', 'trough', 'peak', 'trough', 'peak']
        
        if wave_types not in [bullish_pattern, bearish_pattern]:
            return False
            
        # Validate Elliott wave rules
        wave1_size = abs(waves[1].price - waves[0].price)
        wave3_size = abs(waves[3].price - waves[2].price)
        wave5_size = abs(waves[4].price - waves[3].price)
        
        # Wave 3 cannot be the shortest
        if wave3_size < wave1_size and wave3_size < wave5_size:
            return False
            
        # Wave 4 should not overlap with wave 1 (in most cases)
        if wave_types == bullish_pattern:
            if waves[3].price <= waves[1].price:  # Wave 4 low below wave 1 high
                return False
        else:
            if waves[3].price >= waves[1].price:  # Wave 4 high above wave 1 low
                return False
                
        return True

    def _validate_3_wave_pattern(self, waves: List[WavePoint]) -> bool:
        """Validate if swing points form a valid 3-wave corrective pattern"""
        if len(waves) != 3:
            return False
            
        # Check alternating pattern
        wave_types = [w.wave_type for w in waves]
        valid_patterns = [
            ['peak', 'trough', 'peak'],
            ['trough', 'peak', 'trough']
        ]
        
        return wave_types in valid_patterns

    def _calculate_pattern_confidence(self, waves: List[WavePoint]) -> float:
        """Calculate confidence score for Elliott wave pattern"""
        base_confidence = 0.6
        
        # Add confidence based on wave relationships
        if len(waves) == 5:
            wave1_size = abs(waves[1].price - waves[0].price)
            wave3_size = abs(waves[3].price - waves[2].price)
            
            # Higher confidence if wave 3 is extended
            if wave3_size > wave1_size * 1.618:
                base_confidence += 0.2
                
        # Add confidence based on time duration
        duration_hours = (waves[-1].timestamp - waves[0].timestamp) / 3600
        if 24 <= duration_hours <= 120:  # 1-5 days ideal for swing trading
            base_confidence += 0.1
            
        return min(base_confidence, 0.95)

    def _calculate_wave_target(self, waves: List[WavePoint]) -> float:
        """Calculate target price for wave completion"""
        if len(waves) < 3:
            return waves[-1].price
            
        # For 5-wave patterns, project wave 5 target
        if len(waves) == 5:
            wave1_size = abs(waves[1].price - waves[0].price)
            if waves[0].price < waves[4].price:  # Bullish
                return waves[3].price + (wave1_size * 1.0)  # Equal to wave 1
            else:  # Bearish
                return waves[3].price - (wave1_size * 1.0)
                
        # For 3-wave patterns, project wave C target
        wave_a_size = abs(waves[1].price - waves[0].price)
        if waves[0].price > waves[2].price:  # Bearish correction
            return waves[1].price - (wave_a_size * 1.0)
        else:  # Bullish correction
            return waves[1].price + (wave_a_size * 1.0)

    def _calculate_stop_loss(self, waves: List[WavePoint]) -> float:
        """Calculate stop loss for wave pattern"""
        if len(waves) >= 2:
            # Stop beyond the last significant swing point
            return waves[-2].price
        return waves[-1].price

    async def _determine_current_wave(self, swing_points: List[WavePoint], 
                                    patterns: List[ElliottWavePattern]) -> Optional[WavePoint]:
        """Determine the current wave position"""
        if not swing_points:
            return None
            
        # Return the most recent swing point as current wave reference
        return swing_points[-1] if swing_points else None

    async def _calculate_wave_targets(self, patterns: List[ElliottWavePattern], 
                                    current_price: float) -> Dict[str, float]:
        """Calculate wave targets based on active patterns"""
        targets = {}
        
        for pattern in patterns:
            if pattern.target_price:
                key = f"{pattern.pattern_type}_{pattern.direction}"
                targets[key] = pattern.target_price
                
        return targets

    async def _calculate_fibonacci_levels(self, swing_points: List[WavePoint]) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        levels = {}
        
        if len(swing_points) >= 2:
            last_swing = swing_points[-1]
            prev_swing = swing_points[-2]
            
            swing_range = abs(last_swing.price - prev_swing.price)
            
            for ratio in self.fibonacci_ratios:
                if last_swing.price > prev_swing.price:  # Uptrend
                    levels[f"fib_{ratio}"] = last_swing.price - (swing_range * ratio)
                else:  # Downtrend
                    levels[f"fib_{ratio}"] = last_swing.price + (swing_range * ratio)
                    
        return levels

    async def _generate_execution_signals(self, symbol: str, current_price: float,
                                        current_wave: Optional[WavePoint],
                                        patterns: List[ElliottWavePattern],
                                        targets: Dict[str, float]) -> List[Dict[str, Union[str, float]]]:
        """Generate execution signals for swing trading"""
        signals = []
        
        for pattern in patterns:
            if pattern.confidence > 0.7:
                signal = {
                    'type': 'elliott_wave',
                    'pattern': pattern.pattern_type,
                    'direction': pattern.direction,
                    'confidence': pattern.confidence,
                    'entry_price': current_price,
                    'target_price': pattern.target_price,
                    'stop_loss': pattern.stop_loss,
                    'risk_reward': abs(pattern.target_price - current_price) / abs(pattern.stop_loss - current_price) if pattern.stop_loss else 0,
                    'timeframe': 'H4',
                    'max_duration_days': 5
                }
                signals.append(signal)
                
        return signals

    def _generate_test_data(self) -> List[Dict]:
        """Generate test data for initialization"""
        test_data = []
        base_price = 1.1000
        
        for i in range(100):
            price = base_price + (np.sin(i * 0.1) * 0.01) + (np.random.random() - 0.5) * 0.005
            test_data.append({
                'timestamp': time.time() - (100 - i) * 3600,
                'open': price,
                'high': price + 0.002,
                'low': price - 0.002,
                'close': price,
                'volume': 1000
            })
            
        return test_data

    async def _detect_waves(self, test_data: List[Dict]) -> bool:
        """Test wave detection functionality"""
        try:
            closes = [data['close'] for data in test_data]
            highs = [data['high'] for data in test_data]
            lows = [data['low'] for data in test_data]
            timestamps = [data['timestamp'] for data in test_data]
            
            swing_points = await self._detect_swing_points(highs, lows, closes, timestamps)
            return len(swing_points) > 0
            
        except Exception:
            return False
=======
Short-Term Elliott Wave Analysis Engine
Specialized for 3-5 wave structures for quick trades (max 5 days).
Optimized for H4 timeframe with rapid pattern recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio

class WaveType(Enum):
    """Elliott Wave types for short-term patterns"""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    DIAGONAL = "diagonal"
    TRIANGLE = "triangle"
    FLAT = "flat"
    ZIGZAG = "zigzag"

class WaveDirection(Enum):
    """Wave direction"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

@dataclass
class WavePoint:
    """Individual wave point"""
    timestamp: datetime
    price: float
    volume: float
    wave_number: int
    wave_type: WaveType
    confidence: float

@dataclass
class ElliottWavePattern:
    """Complete Elliott Wave pattern"""
    symbol: str
    timeframe: str
    pattern_type: WaveType
    direction: WaveDirection
    waves: List[WavePoint]
    start_time: datetime
    end_time: datetime
    price_range: Tuple[float, float]
    confidence: float
    completion_percentage: float
    next_target: Optional[float]
    invalidation_level: float
    expected_duration: timedelta

@dataclass
class WaveAnalysisResult:
    """Elliott Wave analysis result"""
    symbol: str
    timestamp: datetime
    patterns: List[ElliottWavePattern]
    active_pattern: Optional[ElliottWavePattern]
    signal_strength: float
    trade_recommendation: str
    entry_level: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: float

class ShortTermElliottWaves:
    """
    Short-Term Elliott Wave Analysis Engine
    Specialized for H4 timeframe with 1-5 day maximum patterns
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.min_wave_points = 3
        self.max_wave_points = 5
        self.max_pattern_duration = timedelta(days=5)
        self.min_confidence_threshold = 0.6

        # Wave ratio constraints for short-term patterns
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        self.wave_ratio_tolerance = 0.1

        # Pattern cache for performance
        self.pattern_cache = {}
        self.cache_duration = timedelta(minutes=15)

    async def analyze_waves(self,
                           symbol: str,
                           price_data: pd.DataFrame,
                           timeframe: str = "H4") -> WaveAnalysisResult:
        """
        Analyze Elliott Wave patterns for short-term trading

        Args:
            symbol: Trading symbol
            price_data: OHLCV data
            timeframe: Analysis timeframe

        Returns:
            WaveAnalysisResult with patterns and signals
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{len(price_data)}"
            if self._is_cache_valid(cache_key):
                return self.pattern_cache[cache_key]['result']

            # Identify swing points
            swing_points = await self._identify_swing_points(price_data)

            # Find Elliott Wave patterns
            patterns = await self._find_wave_patterns(symbol, swing_points, timeframe)

            # Analyze active pattern
            active_pattern = self._get_active_pattern(patterns)

            # Generate trading signals
            signal_data = await self._generate_wave_signals(active_pattern, price_data)

            result = WaveAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                patterns=patterns,
                active_pattern=active_pattern,
                signal_strength=signal_data['strength'],
                trade_recommendation=signal_data['recommendation'],
                entry_level=signal_data['entry'],
                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data['take_profit'],
                risk_reward_ratio=signal_data['risk_reward']
            )

            # Cache result
            self.pattern_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }

            return result

        except Exception as e:
            self.logger.error(f"Elliott Wave analysis error for {symbol}: {e}")
            return self._create_empty_result(symbol)

    async def _identify_swing_points(self, price_data: pd.DataFrame) -> List[WavePoint]:
        """Identify swing highs and lows for wave analysis"""
        swing_points = []

        # Use pivot points to identify swings
        high_pivots = self._find_pivot_points(price_data['high'].values, order=3)
        low_pivots = self._find_pivot_points(price_data['low'].values, order=3, find_peaks=False)

        # Combine and sort pivot points
        all_pivots = []

        for i, is_peak in enumerate(high_pivots):
            if is_peak:
                all_pivots.append({
                    'index': i,
                    'price': price_data.iloc[i]['high'],
                    'type': 'high',
                    'timestamp': price_data.index[i],
                    'volume': price_data.iloc[i]['volume']
                })

        for i, is_trough in enumerate(low_pivots):
            if is_trough:
                all_pivots.append({
                    'index': i,
                    'price': price_data.iloc[i]['low'],
                    'type': 'low',
                    'timestamp': price_data.index[i],
                    'volume': price_data.iloc[i]['volume']
                })

        # Sort by timestamp
        all_pivots.sort(key=lambda x: x['index'])

        # Convert to WavePoint objects
        for i, pivot in enumerate(all_pivots):
            swing_points.append(WavePoint(
                timestamp=pivot['timestamp'],
                price=pivot['price'],
                volume=pivot['volume'],
                wave_number=i + 1,
                wave_type=WaveType.IMPULSE,  # Will be refined later
                confidence=0.7  # Base confidence
            ))

        return swing_points

    def _find_pivot_points(self, data: np.ndarray, order: int = 3, find_peaks: bool = True) -> List[bool]:
        """Find pivot points (peaks or troughs) in price data"""
        pivots = [False] * len(data)

        for i in range(order, len(data) - order):
            if find_peaks:
                # Find peaks
                is_peak = all(data[i] >= data[i-j] for j in range(1, order+1)) and \
                         all(data[i] >= data[i+j] for j in range(1, order+1))
                pivots[i] = is_peak
            else:
                # Find troughs
                is_trough = all(data[i] <= data[i-j] for j in range(1, order+1)) and \
                           all(data[i] <= data[i+j] for j in range(1, order+1))
                pivots[i] = is_trough

        return pivots

    async def _find_wave_patterns(self,
                                 symbol: str,
                                 swing_points: List[WavePoint],
                                 timeframe: str) -> List[ElliottWavePattern]:
        """Find Elliott Wave patterns from swing points"""
        patterns = []

        # Look for 3-wave and 5-wave patterns
        for start_idx in range(len(swing_points) - 2):
            for end_idx in range(start_idx + 2, min(start_idx + 6, len(swing_points))):
                wave_sequence = swing_points[start_idx:end_idx + 1]

                # Check if this forms a valid Elliott Wave pattern
                pattern = await self._validate_wave_pattern(symbol, wave_sequence, timeframe)
                if pattern:
                    patterns.append(pattern)

        # Remove overlapping patterns, keep highest confidence
        patterns = self._remove_overlapping_patterns(patterns)

        return patterns

    async def _validate_wave_pattern(self,
                                   symbol: str,
                                   wave_sequence: List[WavePoint],
                                   timeframe: str) -> Optional[ElliottWavePattern]:
        """Validate if wave sequence forms a valid Elliott Wave pattern"""
        if len(wave_sequence) < 3:
            return None

        # Check time constraint (max 5 days)
        duration = wave_sequence[-1].timestamp - wave_sequence[0].timestamp
        if duration > self.max_pattern_duration:
            return None

        # Analyze wave structure
        pattern_type, direction, confidence = self._analyze_wave_structure(wave_sequence)

        if confidence < self.min_confidence_threshold:
            return None

        # Calculate price range
        prices = [point.price for point in wave_sequence]
        price_range = (min(prices), max(prices))

        # Estimate completion and next target
        completion_pct = self._calculate_completion_percentage(wave_sequence, pattern_type)
        next_target = self._calculate_next_target(wave_sequence, pattern_type)
        invalidation_level = self._calculate_invalidation_level(wave_sequence, pattern_type)

        return ElliottWavePattern(
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,
            direction=direction,
            waves=wave_sequence,
            start_time=wave_sequence[0].timestamp,
            end_time=wave_sequence[-1].timestamp,
            price_range=price_range,
            confidence=confidence,
            completion_percentage=completion_pct,
            next_target=next_target,
            invalidation_level=invalidation_level,
            expected_duration=duration
        )

    def _analyze_wave_structure(self, wave_sequence: List[WavePoint]) -> Tuple[WaveType, WaveDirection, float]:
        """Analyze the structure of wave sequence"""
        if len(wave_sequence) == 3:
            return self._analyze_three_wave_pattern(wave_sequence)
        elif len(wave_sequence) == 5:
            return self._analyze_five_wave_pattern(wave_sequence)
        else:
            return WaveType.CORRECTIVE, WaveDirection.SIDEWAYS, 0.5

    def _analyze_three_wave_pattern(self, waves: List[WavePoint]) -> Tuple[WaveType, WaveDirection, float]:
        """Analyze 3-wave corrective pattern (A-B-C)"""
        # Determine overall direction
        if waves[2].price > waves[0].price:
            direction = WaveDirection.UP
        elif waves[2].price < waves[0].price:
            direction = WaveDirection.DOWN
        else:
            direction = WaveDirection.SIDEWAYS

        # Check for zigzag pattern
        wave_a = abs(waves[1].price - waves[0].price)
        wave_c = abs(waves[2].price - waves[1].price)

        # Fibonacci relationships
        c_to_a_ratio = wave_c / wave_a if wave_a > 0 else 1.0

        confidence = 0.6  # Base confidence for 3-wave

        # Check for common Fibonacci ratios
        for fib_ratio in [0.618, 1.0, 1.618]:
            if abs(c_to_a_ratio - fib_ratio) < self.wave_ratio_tolerance:
                confidence += 0.2
                break

        return WaveType.ZIGZAG, direction, min(confidence, 1.0)

    def _analyze_five_wave_pattern(self, waves: List[WavePoint]) -> Tuple[WaveType, WaveDirection, float]:
        """Analyze 5-wave impulse pattern (1-2-3-4-5)"""
        # Determine overall direction
        if waves[4].price > waves[0].price:
            direction = WaveDirection.UP
        elif waves[4].price < waves[0].price:
            direction = WaveDirection.DOWN
        else:
            direction = WaveDirection.SIDEWAYS

        confidence = 0.7  # Base confidence for 5-wave

        # Check Elliott Wave rules
        # Rule 1: Wave 2 never retraces more than 100% of wave 1
        wave_1 = abs(waves[1].price - waves[0].price)
        wave_2 = abs(waves[2].price - waves[1].price)

        if wave_2 < wave_1:
            confidence += 0.1

        # Rule 2: Wave 3 is never the shortest
        wave_3 = abs(waves[3].price - waves[2].price)
        wave_5 = abs(waves[4].price - waves[3].price)

        if wave_3 >= max(wave_1, wave_5):
            confidence += 0.1

        # Rule 3: Wave 4 never enters the price territory of wave 1
        if direction == WaveDirection.UP:
            if waves[3].price > waves[1].price:
                confidence += 0.1
        else:
            if waves[3].price < waves[1].price:
                confidence += 0.1

        return WaveType.IMPULSE, direction, min(confidence, 1.0)

    def _calculate_completion_percentage(self, waves: List[WavePoint], pattern_type: WaveType) -> float:
        """Calculate pattern completion percentage"""
        if pattern_type == WaveType.IMPULSE and len(waves) == 5:
            return 100.0  # 5-wave pattern complete
        elif pattern_type == WaveType.ZIGZAG and len(waves) == 3:
            return 100.0  # 3-wave pattern complete
        else:
            return (len(waves) / 5.0) * 100.0  # Partial completion

    def _calculate_next_target(self, waves: List[WavePoint], pattern_type: WaveType) -> Optional[float]:
        """Calculate next price target based on Elliott Wave projections"""
        if len(waves) < 3:
            return None

        if pattern_type == WaveType.IMPULSE and len(waves) >= 3:
            # Project wave 5 target based on wave 1 and 3
            wave_1_size = abs(waves[1].price - waves[0].price)
            wave_3_end = waves[2].price

            # Common projection: Wave 5 = Wave 1 * 1.618 from wave 4 low
            if len(waves) >= 4:
                target = waves[3].price + (wave_1_size * 1.618)
                return target

        elif pattern_type == WaveType.ZIGZAG:
            # Project C wave target
            wave_a_size = abs(waves[1].price - waves[0].price)
            wave_b_end = waves[1].price

            # Common projection: C = A * 1.0 or 1.618
            target = wave_b_end + (wave_a_size * 1.0)
            return target

        return None

    def _calculate_invalidation_level(self, waves: List[WavePoint], pattern_type: WaveType) -> float:
        """Calculate pattern invalidation level"""
        if pattern_type == WaveType.IMPULSE:
            # Invalidation if price breaks below wave 1 low (for uptrend)
            return min(wave.price for wave in waves[:2])
        else:
            # For corrective patterns, invalidation at pattern start
            return waves[0].price

    def _remove_overlapping_patterns(self, patterns: List[ElliottWavePattern]) -> List[ElliottWavePattern]:
        """Remove overlapping patterns, keeping highest confidence"""
        if not patterns:
            return patterns

        # Sort by confidence descending
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        filtered_patterns = []
        for pattern in patterns:
            # Check if this pattern overlaps with any already selected
            overlaps = False
            for selected in filtered_patterns:
                if self._patterns_overlap(pattern, selected):
                    overlaps = True
                    break

            if not overlaps:
                filtered_patterns.append(pattern)

        return filtered_patterns

    def _patterns_overlap(self, pattern1: ElliottWavePattern, pattern2: ElliottWavePattern) -> bool:
        """Check if two patterns overlap in time"""
        return not (pattern1.end_time <= pattern2.start_time or pattern2.end_time <= pattern1.start_time)

    def _get_active_pattern(self, patterns: List[ElliottWavePattern]) -> Optional[ElliottWavePattern]:
        """Get the most recent active pattern"""
        if not patterns:
            return None

        # Find pattern with most recent end time
        active_pattern = max(patterns, key=lambda p: p.end_time)

        # Check if pattern is still active (within last 4 hours for H4 timeframe)
        time_since_end = datetime.now() - active_pattern.end_time
        if time_since_end <= timedelta(hours=8):  # Allow some buffer
            return active_pattern

        return None

    async def _generate_wave_signals(self,
                                   active_pattern: Optional[ElliottWavePattern],
                                   price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on Elliott Wave analysis"""
        if not active_pattern:
            return {
                'strength': 0.0,
                'recommendation': 'HOLD',
                'entry': None,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward': 0.0
            }

        current_price = price_data.iloc[-1]['close']
        signal_strength = active_pattern.confidence

        # Generate signals based on pattern type and completion
        if active_pattern.pattern_type == WaveType.IMPULSE:
            return self._generate_impulse_signals(active_pattern, current_price)
        elif active_pattern.pattern_type == WaveType.ZIGZAG:
            return self._generate_corrective_signals(active_pattern, current_price)
        else:
            return {
                'strength': signal_strength * 0.5,
                'recommendation': 'HOLD',
                'entry': None,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward': 0.0
            }

    def _generate_impulse_signals(self, pattern: ElliottWavePattern, current_price: float) -> Dict[str, Any]:
        """Generate signals for impulse patterns"""
        if pattern.completion_percentage >= 80:
            # Pattern nearly complete, look for reversal
            if pattern.direction == WaveDirection.UP:
                recommendation = 'SELL'
                entry = current_price * 0.995  # Slight discount
                stop_loss = pattern.next_target or (current_price * 1.02)
                take_profit = pattern.invalidation_level
            else:
                recommendation = 'BUY'
                entry = current_price * 1.005  # Slight premium
                stop_loss = pattern.next_target or (current_price * 0.98)
                take_profit = pattern.invalidation_level
        else:
            # Pattern in progress, trade in direction
            if pattern.direction == WaveDirection.UP:
                recommendation = 'BUY'
                entry = current_price * 1.002
                stop_loss = pattern.invalidation_level
                take_profit = pattern.next_target or (current_price * 1.05)
            else:
                recommendation = 'SELL'
                entry = current_price * 0.998
                stop_loss = pattern.invalidation_level
                take_profit = pattern.next_target or (current_price * 0.95)

        # Calculate risk-reward ratio
        if entry and stop_loss and take_profit:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            risk_reward = reward / risk if risk > 0 else 0.0
        else:
            risk_reward = 0.0

        return {
            'strength': pattern.confidence,
            'recommendation': recommendation,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward
        }

    def _generate_corrective_signals(self, pattern: ElliottWavePattern, current_price: float) -> Dict[str, Any]:
        """Generate signals for corrective patterns"""
        # Corrective patterns suggest continuation of main trend after completion
        if pattern.completion_percentage >= 90:
            # Look for trend resumption
            if pattern.direction == WaveDirection.DOWN:  # Correction in uptrend
                recommendation = 'BUY'
                entry = current_price * 1.002
                stop_loss = pattern.price_range[0] * 0.99  # Below correction low
                take_profit = pattern.price_range[1] * 1.1  # Above correction high
            else:  # Correction in downtrend
                recommendation = 'SELL'
                entry = current_price * 0.998
                stop_loss = pattern.price_range[1] * 1.01  # Above correction high
                take_profit = pattern.price_range[0] * 0.9  # Below correction low
        else:
            recommendation = 'HOLD'
            entry = None
            stop_loss = None
            take_profit = None

        # Calculate risk-reward ratio
        if entry and stop_loss and take_profit:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            risk_reward = reward / risk if risk > 0 else 0.0
        else:
            risk_reward = 0.0

        return {
            'strength': pattern.confidence * 0.8,  # Lower confidence for corrective
            'recommendation': recommendation,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.pattern_cache:
            return False

        cache_time = self.pattern_cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self.cache_duration

    def _create_empty_result(self, symbol: str) -> WaveAnalysisResult:
        """Create empty result for error cases"""
        return WaveAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            patterns=[],
            active_pattern=None,
            signal_strength=0.0,
            trade_recommendation='HOLD',
            entry_level=None,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=0.0
        )
>>>>>>> 5e659b3064c215382ffc9ef1f13510cbfdd547a7
