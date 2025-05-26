"""
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
    open: float
    high: float
    low: float
    close: float
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
