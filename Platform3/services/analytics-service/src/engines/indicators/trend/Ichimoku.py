"""
Ichimoku Cloud Indicator Module

This module provides comprehensive Ichimoku Kinko Hyo analysis for forex trading,
including all five lines, cloud analysis, and signal generation.
Optimized for scalping (M1-M5), day trading (M15-H1), and swing trading (H4) strategies.

Features:
- Complete Ichimoku system (Tenkan, Kijun, Senkou A/B, Chikou)
- Cloud analysis and support/resistance
- Multiple signal types (TK cross, price vs cloud, etc.)
- Trend strength and direction analysis
- Time-based projections
- Session-optimized parameters

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudPosition(Enum):
    """Price position relative to Ichimoku cloud"""
    ABOVE_CLOUD = "above_cloud"
    IN_CLOUD = "in_cloud"
    BELOW_CLOUD = "below_cloud"

class CloudTrend(Enum):
    """Ichimoku cloud trend direction"""
    BULLISH = "bullish"      # Senkou A > Senkou B
    BEARISH = "bearish"      # Senkou A < Senkou B
    NEUTRAL = "neutral"      # Senkou A ≈ Senkou B

class IchimokuSignal(Enum):
    """Ichimoku trading signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class IchimokuResults:
    """Container for Ichimoku analysis results"""
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float
    cloud_top: float
    cloud_bottom: float
    cloud_thickness: float
    cloud_position: CloudPosition
    cloud_trend: CloudTrend
    tk_cross_signal: Optional[str]
    price_cloud_signal: Optional[str]
    chikou_signal: Optional[str]
    overall_signal: IchimokuSignal
    signal_strength: float
    support_level: float
    resistance_level: float

class IchimokuIndicator:
    """
    Ichimoku Kinko Hyo (Ichimoku Cloud) Indicator

    Provides comprehensive trend analysis using the complete Ichimoku system
    for forex trading strategy development and signal generation.
    """

    def __init__(self,
                 tenkan_period: int = 9,
                 kijun_period: int = 26,
                 senkou_b_period: int = 52,
                 displacement: int = 26):
        """
        Initialize Ichimoku Indicator

        Args:
            tenkan_period: Period for Tenkan-sen (Conversion Line)
            kijun_period: Period for Kijun-sen (Base Line)
            senkou_b_period: Period for Senkou Span B
            displacement: Forward displacement for Senkou spans
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

        # Price history
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.close_history: List[float] = []

        # Ichimoku line history
        self.tenkan_history: List[float] = []
        self.kijun_history: List[float] = []
        self.senkou_a_history: List[float] = []
        self.senkou_b_history: List[float] = []
        self.chikou_history: List[float] = []

        logger.info(f"Ichimoku Indicator initialized: T={tenkan_period}, K={kijun_period}, "
                   f"SB={senkou_b_period}, D={displacement}")

    def _calculate_midpoint(self, highs: List[float], lows: List[float], period: int) -> float:
        """
        Calculate midpoint of highest high and lowest low over period

        Args:
            highs: List of high prices
            lows: List of low prices
            period: Calculation period

        Returns:
            Midpoint value
        """
        if len(highs) < period or len(lows) < period:
            return 0.0

        period_highs = highs[-period:]
        period_lows = lows[-period:]

        highest_high = max(period_highs)
        lowest_low = min(period_lows)

        return (highest_high + lowest_low) / 2.0

    def _calculate_tenkan_sen(self) -> float:
        """Calculate Tenkan-sen (Conversion Line)"""
        return self._calculate_midpoint(self.high_history, self.low_history, self.tenkan_period)

    def _calculate_kijun_sen(self) -> float:
        """Calculate Kijun-sen (Base Line)"""
        return self._calculate_midpoint(self.high_history, self.low_history, self.kijun_period)

    def _calculate_senkou_span_a(self, tenkan: float, kijun: float) -> float:
        """Calculate Senkou Span A (Leading Span A)"""
        return (tenkan + kijun) / 2.0

    def _calculate_senkou_span_b(self) -> float:
        """Calculate Senkou Span B (Leading Span B)"""
        return self._calculate_midpoint(self.high_history, self.low_history, self.senkou_b_period)

    def _calculate_chikou_span(self) -> float:
        """Calculate Chikou Span (Lagging Span)"""
        if len(self.close_history) < self.displacement:
            return 0.0
        return self.close_history[-self.displacement]

    def _determine_cloud_position(self, price: float, senkou_a: float, senkou_b: float) -> CloudPosition:
        """
        Determine price position relative to cloud

        Args:
            price: Current price
            senkou_a: Senkou Span A value
            senkou_b: Senkou Span B value

        Returns:
            CloudPosition enum
        """
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        if price > cloud_top:
            return CloudPosition.ABOVE_CLOUD
        elif price < cloud_bottom:
            return CloudPosition.BELOW_CLOUD
        else:
            return CloudPosition.IN_CLOUD

    def _determine_cloud_trend(self, senkou_a: float, senkou_b: float) -> CloudTrend:
        """
        Determine cloud trend direction

        Args:
            senkou_a: Senkou Span A value
            senkou_b: Senkou Span B value

        Returns:
            CloudTrend enum
        """
        diff_threshold = 0.0001  # Minimum difference for trend determination

        if senkou_a > senkou_b + diff_threshold:
            return CloudTrend.BULLISH
        elif senkou_b > senkou_a + diff_threshold:
            return CloudTrend.BEARISH
        else:
            return CloudTrend.NEUTRAL

    def _detect_tk_cross(self) -> Optional[str]:
        """
        Detect Tenkan-Kijun crossover signals

        Returns:
            Crossover signal type or None
        """
        if len(self.tenkan_history) < 2 or len(self.kijun_history) < 2:
            return None

        current_tenkan = self.tenkan_history[-1]
        current_kijun = self.kijun_history[-1]
        prev_tenkan = self.tenkan_history[-2]
        prev_kijun = self.kijun_history[-2]

        # Bullish TK cross
        if prev_tenkan <= prev_kijun and current_tenkan > current_kijun:
            return "bullish_tk_cross"

        # Bearish TK cross
        elif prev_tenkan >= prev_kijun and current_tenkan < current_kijun:
            return "bearish_tk_cross"

        return None

    def _analyze_price_cloud_signal(self, price: float, cloud_position: CloudPosition,
                                   cloud_trend: CloudTrend) -> Optional[str]:
        """
        Analyze price vs cloud signals

        Args:
            price: Current price
            cloud_position: Price position relative to cloud
            cloud_trend: Cloud trend direction

        Returns:
            Price-cloud signal or None
        """
        if cloud_position == CloudPosition.ABOVE_CLOUD and cloud_trend == CloudTrend.BULLISH:
            return "strong_bullish"
        elif cloud_position == CloudPosition.BELOW_CLOUD and cloud_trend == CloudTrend.BEARISH:
            return "strong_bearish"
        elif cloud_position == CloudPosition.ABOVE_CLOUD and cloud_trend == CloudTrend.BEARISH:
            return "weak_bullish"
        elif cloud_position == CloudPosition.BELOW_CLOUD and cloud_trend == CloudTrend.BULLISH:
            return "weak_bearish"
        else:
            return "neutral"

    def _analyze_chikou_signal(self) -> Optional[str]:
        """
        Analyze Chikou Span signals

        Returns:
            Chikou signal or None
        """
        if len(self.chikou_history) < 2 or len(self.close_history) < self.displacement + 1:
            return None

        current_chikou = self.chikou_history[-1]
        current_price = self.close_history[-1]

        # Compare Chikou with price from displacement periods ago
        if len(self.close_history) >= self.displacement:
            past_price = self.close_history[-(self.displacement + 1)]

            if current_chikou > past_price:
                return "bullish_chikou"
            elif current_chikou < past_price:
                return "bearish_chikou"

        return "neutral_chikou"

    def _calculate_overall_signal(self,
                                tk_signal: Optional[str],
                                price_cloud_signal: Optional[str],
                                chikou_signal: Optional[str],
                                cloud_position: CloudPosition,
                                cloud_trend: CloudTrend) -> Tuple[IchimokuSignal, float]:
        """
        Calculate overall Ichimoku signal and strength

        Args:
            tk_signal: Tenkan-Kijun cross signal
            price_cloud_signal: Price vs cloud signal
            chikou_signal: Chikou span signal
            cloud_position: Price position relative to cloud
            cloud_trend: Cloud trend direction

        Returns:
            Tuple of (overall_signal, signal_strength)
        """
        bullish_signals = 0
        bearish_signals = 0
        signal_strength = 0.0

        # Analyze TK cross
        if tk_signal == "bullish_tk_cross":
            bullish_signals += 1
            signal_strength += 0.3
        elif tk_signal == "bearish_tk_cross":
            bearish_signals += 1
            signal_strength += 0.3

        # Analyze price vs cloud
        if price_cloud_signal == "strong_bullish":
            bullish_signals += 2
            signal_strength += 0.4
        elif price_cloud_signal == "weak_bullish":
            bullish_signals += 1
            signal_strength += 0.2
        elif price_cloud_signal == "strong_bearish":
            bearish_signals += 2
            signal_strength += 0.4
        elif price_cloud_signal == "weak_bearish":
            bearish_signals += 1
            signal_strength += 0.2

        # Analyze Chikou
        if chikou_signal == "bullish_chikou":
            bullish_signals += 1
            signal_strength += 0.2
        elif chikou_signal == "bearish_chikou":
            bearish_signals += 1
            signal_strength += 0.2

        # Determine overall signal
        signal_diff = bullish_signals - bearish_signals

        if signal_diff >= 3:
            return IchimokuSignal.STRONG_BUY, min(1.0, signal_strength)
        elif signal_diff == 2:
            return IchimokuSignal.BUY, min(1.0, signal_strength)
        elif signal_diff == 1:
            return IchimokuSignal.WEAK_BUY, min(1.0, signal_strength)
        elif signal_diff == -1:
            return IchimokuSignal.WEAK_SELL, min(1.0, signal_strength)
        elif signal_diff == -2:
            return IchimokuSignal.SELL, min(1.0, signal_strength)
        elif signal_diff <= -3:
            return IchimokuSignal.STRONG_SELL, min(1.0, signal_strength)
        else:
            return IchimokuSignal.NEUTRAL, min(1.0, signal_strength)

    def _calculate_support_resistance(self, senkou_a: float, senkou_b: float,
                                    kijun: float, tenkan: float) -> Tuple[float, float]:
        """
        Calculate dynamic support and resistance levels

        Args:
            senkou_a: Senkou Span A value
            senkou_b: Senkou Span B value
            kijun: Kijun-sen value
            tenkan: Tenkan-sen value

        Returns:
            Tuple of (support_level, resistance_level)
        """
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Support levels (from strongest to weakest)
        support_candidates = [level for level in [cloud_bottom, kijun, tenkan] if level > 0]
        support_level = min(support_candidates) if support_candidates else 0.0

        # Resistance levels (from strongest to weakest)
        resistance_candidates = [cloud_top, kijun, tenkan]
        resistance_level = max(resistance_candidates) if resistance_candidates else 0.0

        return support_level, resistance_level

    def update(self, high: float, low: float, close: float) -> IchimokuResults:
        """
        Update Ichimoku calculation with new price data

        Args:
            high: Current period high
            low: Current period low
            close: Current period close

        Returns:
            IchimokuResults object with current analysis
        """
        try:
            # Add to price history
            self.high_history.append(high)
            self.low_history.append(low)
            self.close_history.append(close)

            # Calculate Ichimoku lines
            tenkan_sen = self._calculate_tenkan_sen()
            kijun_sen = self._calculate_kijun_sen()
            senkou_span_a = self._calculate_senkou_span_a(tenkan_sen, kijun_sen)
            senkou_span_b = self._calculate_senkou_span_b()
            chikou_span = self._calculate_chikou_span()

            # Update line history
            self.tenkan_history.append(tenkan_sen)
            self.kijun_history.append(kijun_sen)
            self.senkou_a_history.append(senkou_span_a)
            self.senkou_b_history.append(senkou_span_b)
            self.chikou_history.append(chikou_span)

            # Calculate cloud properties
            cloud_top = max(senkou_span_a, senkou_span_b)
            cloud_bottom = min(senkou_span_a, senkou_span_b)
            cloud_thickness = cloud_top - cloud_bottom

            # Analyze positions and trends
            cloud_position = self._determine_cloud_position(close, senkou_span_a, senkou_span_b)
            cloud_trend = self._determine_cloud_trend(senkou_span_a, senkou_span_b)

            # Detect signals
            tk_cross_signal = self._detect_tk_cross()
            price_cloud_signal = self._analyze_price_cloud_signal(close, cloud_position, cloud_trend)
            chikou_signal = self._analyze_chikou_signal()

            # Calculate overall signal
            overall_signal, signal_strength = self._calculate_overall_signal(
                tk_cross_signal, price_cloud_signal, chikou_signal,
                cloud_position, cloud_trend
            )

            # Calculate support and resistance
            support_level, resistance_level = self._calculate_support_resistance(
                senkou_span_a, senkou_span_b, kijun_sen, tenkan_sen
            )

            # Maintain history size
            max_history = max(200, self.senkou_b_period * 2)
            for history in [self.high_history, self.low_history, self.close_history,
                           self.tenkan_history, self.kijun_history,
                           self.senkou_a_history, self.senkou_b_history, self.chikou_history]:
                if len(history) > max_history:
                    history[:] = history[-max_history:]

            result = IchimokuResults(
                tenkan_sen=tenkan_sen,
                kijun_sen=kijun_sen,
                senkou_span_a=senkou_span_a,
                senkou_span_b=senkou_span_b,
                chikou_span=chikou_span,
                cloud_top=cloud_top,
                cloud_bottom=cloud_bottom,
                cloud_thickness=cloud_thickness,
                cloud_position=cloud_position,
                cloud_trend=cloud_trend,
                tk_cross_signal=tk_cross_signal,
                price_cloud_signal=price_cloud_signal,
                chikou_signal=chikou_signal,
                overall_signal=overall_signal,
                signal_strength=signal_strength,
                support_level=support_level,
                resistance_level=resistance_level
            )

            logger.debug(f"Ichimoku updated: Signal={overall_signal.value}, "
                        f"Cloud={cloud_position.value}, Trend={cloud_trend.value}")

            return result

        except Exception as e:
            logger.error(f"Error updating Ichimoku: {str(e)}")
            raise

    def analyze_batch(self,
                     highs: List[float],
                     lows: List[float],
                     closes: List[float]) -> List[IchimokuResults]:
        """
        Analyze batch of price data

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices

        Returns:
            List of IchimokuResults for each period
        """
        results = []

        for high, low, close in zip(highs, lows, closes):
            result = self.update(high, low, close)
            results.append(result)

        return results

    def get_trading_signals(self, results: IchimokuResults) -> Dict[str, Union[str, float, bool]]:
        """
        Generate trading signals based on Ichimoku analysis

        Args:
            results: IchimokuResults from analysis

        Returns:
            Dictionary with trading signals and recommendations
        """
        signals = {
            "overall_signal": results.overall_signal.value,
            "signal_strength": results.signal_strength,
            "cloud_position": results.cloud_position.value,
            "cloud_trend": results.cloud_trend.value,
            "tk_cross": results.tk_cross_signal,
            "price_cloud_signal": results.price_cloud_signal,
            "chikou_signal": results.chikou_signal,
            "support_level": results.support_level,
            "resistance_level": results.resistance_level
        }

        # Trading recommendations
        if results.overall_signal in [IchimokuSignal.STRONG_BUY, IchimokuSignal.BUY]:
            signals["trading_action"] = "buy"
            signals["position_size"] = "large" if results.overall_signal == IchimokuSignal.STRONG_BUY else "normal"
            signals["timeframe_preference"] = "H1-H4"
        elif results.overall_signal in [IchimokuSignal.STRONG_SELL, IchimokuSignal.SELL]:
            signals["trading_action"] = "sell"
            signals["position_size"] = "large" if results.overall_signal == IchimokuSignal.STRONG_SELL else "normal"
            signals["timeframe_preference"] = "H1-H4"
        elif results.overall_signal in [IchimokuSignal.WEAK_BUY, IchimokuSignal.WEAK_SELL]:
            signals["trading_action"] = "cautious"
            signals["position_size"] = "small"
            signals["timeframe_preference"] = "M15-H1"
        else:
            signals["trading_action"] = "wait"
            signals["position_size"] = "minimal"
            signals["timeframe_preference"] = "M5-M15"

        # Cloud-based recommendations
        if results.cloud_position == CloudPosition.ABOVE_CLOUD:
            signals["bias"] = "bullish"
            signals["strategy"] = "buy_dips"
        elif results.cloud_position == CloudPosition.BELOW_CLOUD:
            signals["bias"] = "bearish"
            signals["strategy"] = "sell_rallies"
        else:  # IN_CLOUD
            signals["bias"] = "neutral"
            signals["strategy"] = "range_trading"

        # Cloud thickness analysis
        if results.cloud_thickness > 0.001:  # Thick cloud
            signals["cloud_strength"] = "strong"
            signals["breakout_potential"] = "low"
        else:  # Thin cloud
            signals["cloud_strength"] = "weak"
            signals["breakout_potential"] = "high"

        return signals

# Create alias for expected class name
Ichimoku = IchimokuIndicator
