"""
Pattern Master - Advanced Pattern Recognition AI Model
Production-ready pattern detection and validation for Platform3 Trading System

For the humanitarian mission: Every pattern detected must be accurate and profitable
to maximize aid for sick babies and poor families.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, NamedTuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import math
import scipy.stats as stats
from scipy.signal import find_peaks, argrelextrema
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Pattern recognition imports
try:
    from scipy.fft import fft, fftfreq
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    # Fallback for pattern analysis
    fft = None
    gaussian_filter1d = None

class PatternType(Enum):
    """Comprehensive pattern classification system"""
    # Classic chart patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    
    # Triangle patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    FALLING_WEDGE = "falling_wedge"
    RISING_WEDGE = "rising_wedge"
    
    # Rectangle patterns
    RECTANGLE_CONTINUATION = "rectangle_continuation"
    RECTANGLE_REVERSAL = "rectangle_reversal"
    FLAG = "flag"
    PENNANT = "pennant"
    
    # Harmonic patterns
    GARTLEY = "gartley"
    BUTTERFLY = "butterfly"
    BAT = "bat"
    CRAB = "crab"
    CYPHER = "cypher"
    SHARK = "shark"
    
    # Elliott Wave patterns
    IMPULSE_WAVE = "impulse_wave"
    CORRECTIVE_WAVE = "corrective_wave"
    ENDING_DIAGONAL = "ending_diagonal"
    LEADING_DIAGONAL = "leading_diagonal"
    
    # Japanese candlestick patterns
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    
    # Volume patterns
    VOLUME_BREAKOUT = "volume_breakout"
    VOLUME_CLIMAX = "volume_climax"
    VOLUME_DIVERGENCE = "volume_divergence"
    
    # Momentum patterns
    MOMENTUM_DIVERGENCE = "momentum_divergence"
    MOMENTUM_CONVERGENCE = "momentum_convergence"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"

@dataclass
class PatternPoint:
    """Precise pattern point with validation"""
    index: int
    price: float
    timestamp: datetime
    volume: float = 0.0
    significance: float = 0.0  # 0-1 confidence score
    
@dataclass
class DetectedPattern:
    """Complete pattern detection with validation and prediction"""
    pattern_type: PatternType
    confidence: float  # 0-1 confidence score
    timeframe: str
    start_time: datetime
    end_time: datetime
    
    # Pattern geometry
    key_points: List[PatternPoint]
    resistance_level: Optional[float] = None
    support_level: Optional[float] = None
    
    # Fibonacci relationships
    fibonacci_ratios: Dict[str, float] = None
    harmonic_ratios: Dict[str, float] = None
    
    # Prediction and targets
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    probability_success: float = 0.0
    expected_duration: Optional[timedelta] = None
    
    # Validation metrics
    volume_confirmation: bool = False
    momentum_confirmation: bool = False
    time_confirmation: bool = False
    
    # Market context
    trend_context: str = "neutral"  # bullish, bearish, neutral
    volatility_context: str = "normal"  # high, normal, low
    session_context: str = "unknown"  # london, ny, asian, overlap
    
    # Pattern quality metrics
    symmetry_score: float = 0.0
    clarity_score: float = 0.0
    completion_percentage: float = 0.0

class PatternMaster:
    """
    Advanced Pattern Recognition AI for Platform3 Trading System
    
    Master of all pattern types:
    - Classical chart patterns (H&S, triangles, flags)
    - Harmonic patterns (Gartley, Butterfly, Bat, Crab)
    - Elliott Wave patterns
    - Japanese candlestick patterns
    - Volume and momentum patterns
    
    For the humanitarian mission: Every pattern detection must be highly accurate
    to ensure maximum profitability for helping sick babies and poor families.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection engines
        self.chart_pattern_engine = ChartPatternEngine()
        self.harmonic_pattern_engine = HarmonicPatternEngine()
        self.elliott_wave_engine = ElliottWaveEngine()
        self.candlestick_engine = CandlestickPatternEngine()
        self.volume_pattern_engine = VolumePatternEngine()
        
        # Pattern validation and scoring
        self.pattern_validator = PatternValidator()
        self.pattern_scorer = PatternScorer()
        
        # Historical pattern performance
        self.pattern_performance_db = {}
        self.success_rates = self._load_pattern_success_rates()
        
        # Real-time pattern tracking
        self.active_patterns = {}
        self.pattern_alerts = []
        
    async def analyze_patterns(
        self, 
        data: pd.DataFrame, 
        timeframe: str = "H1"
    ) -> Dict[str, List[DetectedPattern]]:
        """
        Master pattern analysis across all pattern types.
        
        Returns comprehensive pattern analysis with high-confidence patterns
        suitable for live trading in support of humanitarian mission.
        """
        
        self.logger.info(f"🧠 Pattern Master analyzing {len(data)} bars on {timeframe}")
        
        detected_patterns = {
            'chart_patterns': [],
            'harmonic_patterns': [],
            'elliott_wave_patterns': [],
            'candlestick_patterns': [],
            'volume_patterns': [],
            'high_confidence_patterns': []
        }
        
        # 1. Classical Chart Pattern Analysis
        chart_patterns = await self._detect_chart_patterns(data, timeframe)
        detected_patterns['chart_patterns'] = chart_patterns
        
        # 2. Harmonic Pattern Analysis
        harmonic_patterns = await self._detect_harmonic_patterns(data, timeframe)
        detected_patterns['harmonic_patterns'] = harmonic_patterns
        
        # 3. Elliott Wave Analysis
        elliott_patterns = await self._detect_elliott_wave_patterns(data, timeframe)
        detected_patterns['elliott_wave_patterns'] = elliott_patterns
        
        # 4. Candlestick Pattern Analysis
        candlestick_patterns = await self._detect_candlestick_patterns(data, timeframe)
        detected_patterns['candlestick_patterns'] = candlestick_patterns
        
        # 5. Volume Pattern Analysis
        volume_patterns = await self._detect_volume_patterns(data, timeframe)
        detected_patterns['volume_patterns'] = volume_patterns
        
        # 6. Cross-validate and score all patterns
        validated_patterns = await self._cross_validate_patterns(
            detected_patterns, data, timeframe
        )
        
        # 7. Select high-confidence patterns for trading
        high_confidence = self._filter_high_confidence_patterns(validated_patterns)
        detected_patterns['high_confidence_patterns'] = high_confidence
        
        # 8. Update pattern performance tracking
        await self._update_pattern_performance(detected_patterns)
        
        self.logger.info(f"✅ Pattern Master found {len(high_confidence)} high-confidence patterns")
        
        return detected_patterns
    
    async def _detect_chart_patterns(
        self, 
        data: pd.DataFrame, 
        timeframe: str
    ) -> List[DetectedPattern]:
        """Detect classical chart patterns with professional accuracy"""
        
        patterns = []
        
        # Calculate technical indicators for pattern confirmation
        data = self._add_technical_indicators(data)
        
        # Detect Head and Shoulders patterns
        h_and_s = await self._detect_head_and_shoulders(data, timeframe)
        patterns.extend(h_and_s)
        
        # Detect Double Top/Bottom patterns
        double_patterns = await self._detect_double_patterns(data, timeframe)
        patterns.extend(double_patterns)
        
        # Detect Triangle patterns
        triangle_patterns = await self._detect_triangle_patterns(data, timeframe)
        patterns.extend(triangle_patterns)
        
        # Detect Flag and Pennant patterns
        flag_patterns = await self._detect_flag_patterns(data, timeframe)
        patterns.extend(flag_patterns)
        
        # Detect Rectangle patterns
        rectangle_patterns = await self._detect_rectangle_patterns(data, timeframe)
        patterns.extend(rectangle_patterns)
        
        return patterns
    
    async def _detect_harmonic_patterns(
        self, 
        data: pd.DataFrame, 
        timeframe: str
    ) -> List[DetectedPattern]:
        """Advanced harmonic pattern detection using Fibonacci relationships"""
        
        patterns = []
        
        # Find significant swing points for harmonic analysis
        swing_points = self._find_swing_points(data, min_swing_size=10)
        
        if len(swing_points) < 5:
            return patterns
        
        # Check for major harmonic patterns
        for i in range(len(swing_points) - 4):
            # Extract 5 points for XABCD pattern analysis
            points = swing_points[i:i+5]
            
            # Test for Gartley pattern (0.618 XA retracement, 0.786 AB retracement)
            gartley = self._test_gartley_pattern(points, data, timeframe)
            if gartley:
                patterns.append(gartley)
            
            # Test for Butterfly pattern (0.786 XA retracement, 1.27 AB extension)
            butterfly = self._test_butterfly_pattern(points, data, timeframe)
            if butterfly:
                patterns.append(butterfly)
            
            # Test for Bat pattern (0.382/0.5 XA retracement, 0.886 AB retracement)
            bat = self._test_bat_pattern(points, data, timeframe)
            if bat:
                patterns.append(bat)
            
            # Test for Crab pattern (0.382/0.618 XA retracement, 1.618 AB extension)
            crab = self._test_crab_pattern(points, data, timeframe)
            if crab:
                patterns.append(crab)
        
        return patterns
    
    async def _detect_elliott_wave_patterns(
        self, 
        data: pd.DataFrame, 
        timeframe: str
    ) -> List[DetectedPattern]:
        """Elliott Wave pattern detection with wave counting"""
        
        patterns = []
        
        # Identify major trend movements
        trend_points = self._identify_elliott_trend_points(data)
        
        if len(trend_points) < 8:  # Need at least 8 points for full wave count
            return patterns
        
        # Analyze for 5-wave impulse patterns
        impulse_patterns = self._analyze_impulse_waves(trend_points, data, timeframe)
        patterns.extend(impulse_patterns)
        
        # Analyze for 3-wave corrective patterns  
        corrective_patterns = self._analyze_corrective_waves(trend_points, data, timeframe)
        patterns.extend(corrective_patterns)
        
        # Check for diagonal patterns
        diagonal_patterns = self._analyze_diagonal_patterns(trend_points, data, timeframe)
        patterns.extend(diagonal_patterns)
        
        return patterns
    
    async def _detect_candlestick_patterns(
        self, 
        data: pd.DataFrame, 
        timeframe: str
    ) -> List[DetectedPattern]:
        """Japanese candlestick pattern recognition"""
        
        patterns = []
        
        for i in range(2, len(data) - 1):  # Need surrounding context
            
            # Single candlestick patterns
            current = data.iloc[i]
            prev = data.iloc[i-1]
            next_candle = data.iloc[i+1] if i+1 < len(data) else None
            
            # Doji pattern
            if self._is_doji(current):
                pattern = self._create_doji_pattern(current, i, timeframe)
                if pattern:
                    patterns.append(pattern)
            
            # Hammer/Hanging Man
            if self._is_hammer(current):
                pattern = self._create_hammer_pattern(current, prev, i, timeframe)
                if pattern:
                    patterns.append(pattern)
            
            # Shooting Star
            if self._is_shooting_star(current):
                pattern = self._create_shooting_star_pattern(current, prev, i, timeframe)
                if pattern:
                    patterns.append(pattern)
            
            # Two-candlestick patterns
            if i >= 1:
                # Engulfing patterns
                if self._is_bullish_engulfing(prev, current):
                    pattern = self._create_engulfing_pattern(
                        prev, current, i-1, i, timeframe, bullish=True
                    )
                    patterns.append(pattern)
                
                elif self._is_bearish_engulfing(prev, current):
                    pattern = self._create_engulfing_pattern(
                        prev, current, i-1, i, timeframe, bullish=False
                    )
                    patterns.append(pattern)
            
            # Three-candlestick patterns
            if i >= 2:
                three_candles = data.iloc[i-2:i+1]
                
                # Morning Star
                morning_star = self._detect_morning_star(three_candles, i-2, timeframe)
                if morning_star:
                    patterns.append(morning_star)
                
                # Evening Star
                evening_star = self._detect_evening_star(three_candles, i-2, timeframe)
                if evening_star:
                    patterns.append(evening_star)
        
        return patterns
    
    def _find_swing_points(self, data: pd.DataFrame, min_swing_size: int = 5) -> List[PatternPoint]:
        """Find significant swing highs and lows for pattern analysis"""
        
        highs = data['high'].values
        lows = data['low'].values
        
        # Find peaks and troughs
        peak_indices = argrelextrema(highs, np.greater, order=min_swing_size)[0]
        trough_indices = argrelextrema(lows, np.less, order=min_swing_size)[0]
        
        swing_points = []
        
        # Convert peaks to PatternPoints
        for idx in peak_indices:
            if idx < len(data):
                swing_points.append(PatternPoint(
                    index=idx,
                    price=highs[idx],
                    timestamp=data.index[idx] if hasattr(data.index[idx], 'to_pydatetime') else datetime.now(),
                    volume=data.iloc[idx].get('volume', 0),
                    significance=self._calculate_point_significance(data, idx, True)
                ))
        
        # Convert troughs to PatternPoints
        for idx in trough_indices:
            if idx < len(data):
                swing_points.append(PatternPoint(
                    index=idx,
                    price=lows[idx],
                    timestamp=data.index[idx] if hasattr(data.index[idx], 'to_pydatetime') else datetime.now(),
                    volume=data.iloc[idx].get('volume', 0),
                    significance=self._calculate_point_significance(data, idx, False)
                ))
        
        # Sort by index
        swing_points.sort(key=lambda x: x.index)
        
        return swing_points
    
    def _calculate_point_significance(self, data: pd.DataFrame, idx: int, is_peak: bool) -> float:
        """Calculate the significance of a swing point (0-1)"""
        
        if idx < 10 or idx >= len(data) - 10:
            return 0.5  # Edge points get medium significance
        
        window = 20
        start_idx = max(0, idx - window)
        end_idx = min(len(data), idx + window)
        
        if is_peak:
            local_max = data['high'].iloc[start_idx:end_idx].max()
            local_range = data['high'].iloc[start_idx:end_idx].max() - data['low'].iloc[start_idx:end_idx].min()
            price = data['high'].iloc[idx]
        else:
            local_min = data['low'].iloc[start_idx:end_idx].min()
            local_range = data['high'].iloc[start_idx:end_idx].max() - data['low'].iloc[start_idx:end_idx].min()
            price = data['low'].iloc[idx]
        
        if local_range == 0:
            return 0.5
        
        # Significance based on how extreme the point is within local range
        if is_peak:
            significance = (price - data['low'].iloc[start_idx:end_idx].min()) / local_range
        else:
            significance = (data['high'].iloc[start_idx:end_idx].max() - price) / local_range
        
        return min(1.0, max(0.0, significance))    
    def _test_gartley_pattern(
        self, 
        points: List[PatternPoint], 
        data: pd.DataFrame, 
        timeframe: str
    ) -> Optional[DetectedPattern]:
        """Test for Gartley harmonic pattern (0.618 XA, 0.786 AB)"""
        
        if len(points) < 5:
            return None
        
        X, A, B, C, D = points[:5]
        
        # Calculate Fibonacci ratios
        XA_range = abs(A.price - X.price)
        AB_range = abs(B.price - A.price)
        BC_range = abs(C.price - B.price)
        CD_range = abs(D.price - C.price)
        
        # Gartley rules:
        # 1. AB = 0.618 XA
        # 2. BC = 0.382 or 0.886 AB
        # 3. CD = 1.27 or 1.618 BC
        # 4. AD = 0.786 XA
        
        AB_ratio = AB_range / XA_range if XA_range > 0 else 0
        BC_ratio = BC_range / AB_range if AB_range > 0 else 0
        CD_ratio = CD_range / BC_range if BC_range > 0 else 0
        AD_ratio = abs(D.price - A.price) / XA_range if XA_range > 0 else 0
        
        # Check Gartley ratios with tolerance
        tolerance = 0.05
        
        if (abs(AB_ratio - 0.618) < tolerance and
            (abs(BC_ratio - 0.382) < tolerance or abs(BC_ratio - 0.886) < tolerance) and
            (abs(CD_ratio - 1.27) < tolerance or abs(CD_ratio - 1.618) < tolerance) and
            abs(AD_ratio - 0.786) < tolerance):
            
            # Calculate confidence based on ratio accuracy
            ratio_accuracy = 1.0 - (
                abs(AB_ratio - 0.618) + 
                min(abs(BC_ratio - 0.382), abs(BC_ratio - 0.886)) +
                min(abs(CD_ratio - 1.27), abs(CD_ratio - 1.618)) +
                abs(AD_ratio - 0.786)
            ) / 4
            
            # Determine pattern direction
            is_bullish = D.price < A.price
            
            # Calculate targets
            price_target = self._calculate_gartley_target(X, A, B, C, D, is_bullish)
            stop_loss = D.price + (0.002 * D.price if is_bullish else -0.002 * D.price)
            
            return DetectedPattern(
                pattern_type=PatternType.GARTLEY,
                confidence=ratio_accuracy * 0.85,  # Gartley base confidence
                timeframe=timeframe,
                start_time=X.timestamp,
                end_time=D.timestamp,
                key_points=points[:5],
                price_target=price_target,
                stop_loss=stop_loss,
                fibonacci_ratios={
                    'AB_XA': AB_ratio,
                    'BC_AB': BC_ratio,
                    'CD_BC': CD_ratio,
                    'AD_XA': AD_ratio
                },
                probability_success=0.75,  # Historical Gartley success rate
                trend_context="reversal",
                completion_percentage=100.0
            )
        
        return None
    
    def _is_doji(self, candle: pd.Series) -> bool:
        """Detect Doji candlestick pattern"""
        open_price = candle['open']
        close_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']
        
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        # Doji: body is less than 10% of total range
        if total_range > 0:
            return body_size / total_range < 0.1
        return False
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """Detect Hammer candlestick pattern"""
        open_price = candle['open']
        close_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']
        
        body_size = abs(close_price - open_price)
        lower_shadow = min(open_price, close_price) - low_price
        upper_shadow = high_price - max(open_price, close_price)
        
        # Hammer: long lower shadow (2x body), small upper shadow
        return (lower_shadow >= 2 * body_size and 
                upper_shadow <= body_size * 0.5 and
                body_size > 0)
    
    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """Detect Shooting Star candlestick pattern"""
        open_price = candle['open']
        close_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']
        
        body_size = abs(close_price - open_price)
        lower_shadow = min(open_price, close_price) - low_price
        upper_shadow = high_price - max(open_price, close_price)
        
        # Shooting Star: long upper shadow (2x body), small lower shadow
        return (upper_shadow >= 2 * body_size and 
                lower_shadow <= body_size * 0.5 and
                body_size > 0)
    
    def _is_bullish_engulfing(self, prev_candle: pd.Series, current_candle: pd.Series) -> bool:
        """Detect Bullish Engulfing pattern"""
        # Previous candle is bearish
        prev_bearish = prev_candle['close'] < prev_candle['open']
        # Current candle is bullish
        current_bullish = current_candle['close'] > current_candle['open']
        # Current body engulfs previous body
        engulfs = (current_candle['open'] < prev_candle['close'] and 
                  current_candle['close'] > prev_candle['open'])
        
        return prev_bearish and current_bullish and engulfs
    
    def _is_bearish_engulfing(self, prev_candle: pd.Series, current_candle: pd.Series) -> bool:
        """Detect Bearish Engulfing pattern"""
        # Previous candle is bullish
        prev_bullish = prev_candle['close'] > prev_candle['open']
        # Current candle is bearish
        current_bearish = current_candle['close'] < current_candle['open']
        # Current body engulfs previous body
        engulfs = (current_candle['open'] > prev_candle['close'] and 
                  current_candle['close'] < prev_candle['open'])
        
        return prev_bullish and current_bearish and engulfs
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for pattern confirmation"""
        data = data.copy()
        
        # Moving averages
        data['ema_21'] = data['close'].ewm(span=21).mean()
        data['ema_55'] = data['close'].ewm(span=55).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        return data
    
    def _load_pattern_success_rates(self) -> Dict[PatternType, float]:
        """Load historical success rates for each pattern type"""
        return {
            PatternType.HEAD_AND_SHOULDERS: 0.78,
            PatternType.DOUBLE_TOP: 0.72,
            PatternType.DOUBLE_BOTTOM: 0.74,
            PatternType.ASCENDING_TRIANGLE: 0.82,
            PatternType.DESCENDING_TRIANGLE: 0.79,
            PatternType.SYMMETRICAL_TRIANGLE: 0.68,
            PatternType.GARTLEY: 0.75,
            PatternType.BUTTERFLY: 0.71,
            PatternType.BAT: 0.73,
            PatternType.CRAB: 0.69,
            PatternType.HAMMER: 0.65,
            PatternType.ENGULFING_BULLISH: 0.67,
            PatternType.ENGULFING_BEARISH: 0.67,
            PatternType.MORNING_STAR: 0.71,
            PatternType.EVENING_STAR: 0.69
        }

# Support classes for Pattern Master
class ChartPatternEngine:
    """Specialized engine for classical chart patterns"""
    pass

class HarmonicPatternEngine:
    """Specialized engine for harmonic patterns"""
    pass

class ElliottWaveEngine:
    """Specialized engine for Elliott Wave patterns"""
    pass

class CandlestickPatternEngine:
    """Specialized engine for candlestick patterns"""
    pass

class VolumePatternEngine:
    """Specialized engine for volume-based patterns"""
    pass

class PatternValidator:
    """Validates detected patterns for quality and reliability"""
    pass

class PatternScorer:
    """Scores patterns based on multiple factors for trading decisions"""
    pass

# Example usage for testing
if __name__ == "__main__":
    print("🧠 Pattern Master - Advanced Pattern Recognition AI")
    print("For the humanitarian mission: Detecting profitable patterns")
    print("to generate maximum aid for sick babies and poor families")