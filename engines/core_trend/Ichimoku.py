"""
Ichimoku Cloud Indicator
Comprehensive trend analysis system for forex trading

This module implements the complete Ichimoku Kinko Hyo system with all five components:
Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, and Chikou Span.
Optimized for scalping, day trading, and swing trading strategies.

Features:
- Complete Ichimoku system calculation
- Cloud analysis (Kumo)
- Signal generation and filtering
- Support/resistance level detection
- Trend strength analysis
- Multi-timeframe compatibility

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudPosition(Enum):
    """Price position relative to Ichimoku cloud"""
    ABOVE_CLOUD = "above_cloud"
    IN_CLOUD = "in_cloud"
    BELOW_CLOUD = "below_cloud"

class CloudColor(Enum):
    """Ichimoku cloud color (trend direction)"""
    BULLISH = "bullish"    # Senkou A > Senkou B
    BEARISH = "bearish"    # Senkou A < Senkou B
    NEUTRAL = "neutral"    # Senkou A ≈ Senkou B

class IchimokuSignalType(Enum):
    """Types of Ichimoku signals"""
    TENKAN_KIJUN_CROSS = "tenkan_kijun_cross"
    PRICE_CLOUD_BREAK = "price_cloud_break"
    CHIKOU_CONFIRMATION = "chikou_confirmation"
    CLOUD_TWIST = "cloud_twist"
    STRONG_TREND = "strong_trend"
    NO_SIGNAL = "no_signal"

@dataclass
class IchimokuSignal:
    """Ichimoku signal information"""
    signal_type: IchimokuSignalType
    direction: str  # "bullish", "bearish", "neutral"
    strength: float
    confidence: float
    timestamp: datetime
    cloud_position: CloudPosition
    cloud_color: CloudColor

@dataclass
class IchimokuResult:
    """Complete Ichimoku analysis result"""
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float
    cloud_position: CloudPosition
    cloud_color: CloudColor
    cloud_thickness: float
    signal: IchimokuSignal
    timestamp: datetime

class Ichimoku:
    """
    Ichimoku Kinko Hyo (Equilibrium Chart) Indicator

    Implements the complete Ichimoku system for comprehensive trend analysis.
    Provides cloud analysis, signal generation, and trend strength assessment.
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ):
        """
        Initialize Ichimoku indicator

        Args:
            tenkan_period: Tenkan-sen period (default: 9)
            kijun_period: Kijun-sen period (default: 26)
            senkou_b_period: Senkou Span B period (default: 52)
            displacement: Cloud displacement (default: 26)
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

        # Internal state
        self.tenkan_values = []
        self.kijun_values = []
        self.senkou_a_values = []
        self.senkou_b_values = []
        self.chikou_values = []

        # Performance tracking
        self.calculation_count = 0
        self.signal_count = 0

        logger.info(f"Ichimoku initialized with periods: T={tenkan_period}, K={kijun_period}, B={senkou_b_period}")

    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IchimokuResult:
        """
        Calculate Ichimoku indicator

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array

        Returns:
            IchimokuResult with complete analysis
        """
        try:
            if len(high) < self.senkou_b_period + self.displacement:
                raise ValueError(f"Insufficient data: need at least {self.senkou_b_period + self.displacement} periods")

            # Calculate Tenkan-sen (Conversion Line)
            tenkan_sen = self._calculate_midpoint(high, low, self.tenkan_period)

            # Calculate Kijun-sen (Base Line)
            kijun_sen = self._calculate_midpoint(high, low, self.kijun_period)

            # Calculate Senkou Span A (Leading Span A)
            senkou_span_a = (tenkan_sen + kijun_sen) / 2

            # Calculate Senkou Span B (Leading Span B)
            senkou_span_b = self._calculate_midpoint(high, low, self.senkou_b_period)

            # Calculate Chikou Span (Lagging Span)
            chikou_span = close

            # Get current values
            current_tenkan = tenkan_sen[-1] if len(tenkan_sen) > 0 else 0.0
            current_kijun = kijun_sen[-1] if len(kijun_sen) > 0 else 0.0
            current_senkou_a = senkou_span_a[-1] if len(senkou_span_a) > 0 else 0.0
            current_senkou_b = senkou_span_b[-1] if len(senkou_span_b) > 0 else 0.0
            current_chikou = chikou_span[-self.displacement] if len(chikou_span) >= self.displacement else close[-1]
            current_price = close[-1]

            # Analyze cloud
            cloud_position = self._determine_cloud_position(current_price, current_senkou_a, current_senkou_b)
            cloud_color = self._determine_cloud_color(current_senkou_a, current_senkou_b)
            cloud_thickness = abs(current_senkou_a - current_senkou_b)

            # Generate signal
            signal = self._generate_signal(
                tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b,
                chikou_span, close, cloud_position, cloud_color
            )

            # Update performance tracking
            self.calculation_count += 1
            if signal.signal_type != IchimokuSignalType.NO_SIGNAL:
                self.signal_count += 1

            return IchimokuResult(
                tenkan_sen=current_tenkan,
                kijun_sen=current_kijun,
                senkou_span_a=current_senkou_a,
                senkou_span_b=current_senkou_b,
                chikou_span=current_chikou,
                cloud_position=cloud_position,
                cloud_color=cloud_color,
                cloud_thickness=cloud_thickness,
                signal=signal,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
            raise

    def _calculate_midpoint(self, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """Calculate midpoint (highest high + lowest low) / 2 for given period"""
        if len(high) < period:
            return np.array([])

        midpoints = []
        for i in range(period - 1, len(high)):
            period_high = np.max(high[i - period + 1:i + 1])
            period_low = np.min(low[i - period + 1:i + 1])
            midpoints.append((period_high + period_low) / 2)

        return np.array(midpoints)

    def _determine_cloud_position(self, price: float, senkou_a: float, senkou_b: float) -> CloudPosition:
        """Determine price position relative to cloud"""
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        if price > cloud_top:
            return CloudPosition.ABOVE_CLOUD
        elif price < cloud_bottom:
            return CloudPosition.BELOW_CLOUD
        else:
            return CloudPosition.IN_CLOUD

    def _determine_cloud_color(self, senkou_a: float, senkou_b: float) -> CloudColor:
        """Determine cloud color (trend direction)"""
        diff = abs(senkou_a - senkou_b)

        if diff < 0.0001:  # Very close values
            return CloudColor.NEUTRAL
        elif senkou_a > senkou_b:
            return CloudColor.BULLISH
        else:
            return CloudColor.BEARISH

    def _generate_signal(
        self,
        tenkan_sen: np.ndarray,
        kijun_sen: np.ndarray,
        senkou_span_a: np.ndarray,
        senkou_span_b: np.ndarray,
        chikou_span: np.ndarray,
        close: np.ndarray,
        cloud_position: CloudPosition,
        cloud_color: CloudColor
    ) -> IchimokuSignal:
        """Generate Ichimoku trading signal"""
        try:
            if len(tenkan_sen) < 3 or len(kijun_sen) < 3:
                return IchimokuSignal(
                    signal_type=IchimokuSignalType.NO_SIGNAL,
                    direction="neutral",
                    strength=0.0,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    cloud_position=cloud_position,
                    cloud_color=cloud_color
                )

            current_tenkan = tenkan_sen[-1]
            current_kijun = kijun_sen[-1]
            prev_tenkan = tenkan_sen[-2]
            prev_kijun = kijun_sen[-2]

            current_price = close[-1]
            current_senkou_a = senkou_span_a[-1] if len(senkou_span_a) > 0 else 0.0
            current_senkou_b = senkou_span_b[-1] if len(senkou_span_b) > 0 else 0.0

            signal_type = IchimokuSignalType.NO_SIGNAL
            direction = "neutral"
            strength = 0.0
            confidence = 0.0

            # Check for Tenkan-Kijun cross
            if prev_tenkan <= prev_kijun and current_tenkan > current_kijun:
                signal_type = IchimokuSignalType.TENKAN_KIJUN_CROSS
                direction = "bullish"
                strength = 0.7
                confidence = 0.6
            elif prev_tenkan >= prev_kijun and current_tenkan < current_kijun:
                signal_type = IchimokuSignalType.TENKAN_KIJUN_CROSS
                direction = "bearish"
                strength = 0.7
                confidence = 0.6

            # Check for price cloud break
            elif cloud_position == CloudPosition.ABOVE_CLOUD and cloud_color == CloudColor.BULLISH:
                signal_type = IchimokuSignalType.PRICE_CLOUD_BREAK
                direction = "bullish"
                strength = 0.8
                confidence = 0.7
            elif cloud_position == CloudPosition.BELOW_CLOUD and cloud_color == CloudColor.BEARISH:
                signal_type = IchimokuSignalType.PRICE_CLOUD_BREAK
                direction = "bearish"
                strength = 0.8
                confidence = 0.7

            # Check for cloud twist
            elif len(senkou_span_a) >= 2 and len(senkou_span_b) >= 2:
                prev_senkou_a = senkou_span_a[-2]
                prev_senkou_b = senkou_span_b[-2]

                if (prev_senkou_a <= prev_senkou_b and current_senkou_a > current_senkou_b):
                    signal_type = IchimokuSignalType.CLOUD_TWIST
                    direction = "bullish"
                    strength = 0.6
                    confidence = 0.5
                elif (prev_senkou_a >= prev_senkou_b and current_senkou_a < current_senkou_b):
                    signal_type = IchimokuSignalType.CLOUD_TWIST
                    direction = "bearish"
                    strength = 0.6
                    confidence = 0.5

            # Check for strong trend
            if (cloud_position == CloudPosition.ABOVE_CLOUD and
                cloud_color == CloudColor.BULLISH and
                current_tenkan > current_kijun and
                current_price > current_tenkan):
                signal_type = IchimokuSignalType.STRONG_TREND
                direction = "bullish"
                strength = 0.9
                confidence = 0.8
            elif (cloud_position == CloudPosition.BELOW_CLOUD and
                  cloud_color == CloudColor.BEARISH and
                  current_tenkan < current_kijun and
                  current_price < current_tenkan):
                signal_type = IchimokuSignalType.STRONG_TREND
                direction = "bearish"
                strength = 0.9
                confidence = 0.8

            # Chikou confirmation
            if len(chikou_span) >= self.displacement:
                chikou_price = chikou_span[-self.displacement]
                historical_price = close[-self.displacement] if len(close) >= self.displacement else close[-1]

                if signal_type != IchimokuSignalType.NO_SIGNAL:
                    if direction == "bullish" and chikou_price > historical_price:
                        confidence = min(1.0, confidence + 0.2)
                    elif direction == "bearish" and chikou_price < historical_price:
                        confidence = min(1.0, confidence + 0.2)

            return IchimokuSignal(
                signal_type=signal_type,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timestamp=datetime.now(),
                cloud_position=cloud_position,
                cloud_color=cloud_color
            )

        except Exception as e:
            logger.error(f"Error generating Ichimoku signal: {e}")
            return IchimokuSignal(
                signal_type=IchimokuSignalType.NO_SIGNAL,
                direction="neutral",
                strength=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                cloud_position=cloud_position,
                cloud_color=cloud_color
            )

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'calculation_count': self.calculation_count,
            'signal_count': self.signal_count,
            'signal_rate': self.signal_count / max(1, self.calculation_count),
            'tenkan_period': self.tenkan_period,
            'kijun_period': self.kijun_period,
            'senkou_b_period': self.senkou_b_period,
            'displacement': self.displacement
        }
