# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Accumulation/Distribution Line (A/D Line) - Advanced Volume Flow Analysis
Platform3 - Humanitarian Trading System

The Accumulation/Distribution Line is a volume-based indicator that measures
the cumulative flow of money into and out of a security. It combines price
and volume to show whether a stock is being accumulated or distributed.

Key Features:
- Money flow accumulation tracking
- Volume-weighted price analysis
- Trend confirmation/divergence detection
- Buying/selling pressure measurement
- Momentum validation
- Multi-timeframe coordination

Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
Money Flow Volume = Money Flow Multiplier multiply Volume
A/D Line = Previous A/D Line + Money Flow Volume

Humanitarian Mission: Identify smart money accumulation patterns for optimal
entry timing and maximize profit generation through volume-confirmed signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from engines.indicator_base import IndicatorSignal, TechnicalIndicator
import logging

logger = logging.getLogger(__name__)

@dataclass
class AccumulationDistributionSignal(IndicatorSignal):
    """A/D Line-specific signal with detailed analysis"""
    ad_line_value: float = 0.0
    money_flow_multiplier: float = 0.0
    money_flow_volume: float = 0.0
    trend_direction: str = "neutral"  # "accumulation", "distribution", "neutral"
    trend_strength: str = "moderate"  # "strong", "moderate", "weak"
    momentum_phase: str = "stable"  # "accelerating", "decelerating", "stable"
    divergence_signal: Optional[str] = None  # "bullish_divergence", "bearish_divergence"
    volume_confirmation: str = "confirmed"  # "confirmed", "unconfirmed", "weak"
    buying_pressure: float = 0.0  # -1 to 1 scale
    flow_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize required base fields if not provided"""
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = datetime.now()
        if not hasattr(self, 'indicator_name') or self.indicator_name is None:
            self.indicator_name = "AccumulationDistributionLine"
        if not hasattr(self, 'signal_type') or self.signal_type is None:
            from engines.indicator_base import SignalType
            self.signal_type = SignalType.NEUTRAL

class AccumulationDistributionLine(TechnicalIndicator):
    """
    Advanced Accumulation/Distribution Line Implementation
    
    Combines price action with volume to reveal the underlying buying
    and selling pressure, helping identify accumulation and distribution phases.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, 
                 lookback_period: int = 20,
                 trend_confirmation_period: int = 10,
                 divergence_period: int = 15,
                 volume_threshold: float = 1.2):
        """
        Initialize A/D Line with comprehensive analysis parameters
        
        Args:
            config: Optional configuration dictionary or IndicatorConfig object.
            lookback_period: Period for trend analysis (default 20)
            trend_confirmation_period: Periods to confirm trend changes
            divergence_period: Period for divergence analysis
            volume_threshold: Minimum volume multiplier for significance
        """
        from engines.indicator_base import IndicatorConfig, IndicatorType, TimeFrame

        # If config is a dictionary, convert it to IndicatorConfig
        if isinstance(config, dict):
            # Ensure essential keys are present or provide defaults
            name = config.get('name', "AccumulationDistributionLine")
            indicator_type = config.get('indicator_type', IndicatorType.VOLUME)
            timeframe = config.get('timeframe', TimeFrame.D1)
            lb_periods = config.get('lookback_periods', lookback_period)
            params = config.get('parameters', {})

            # Update parameters from direct arguments if not in config dict
            params.setdefault('lookback_period', lookback_period)
            params.setdefault('trend_confirmation_period', trend_confirmation_period)
            params.setdefault('divergence_period', divergence_period)
            params.setdefault('volume_threshold', volume_threshold)
            
            config_obj = IndicatorConfig(
                name=name,
                indicator_type=indicator_type,
                timeframe=timeframe,
                lookback_periods=lb_periods,
                parameters=params
            )
        elif isinstance(config, IndicatorConfig):
            config_obj = config
        else: # No config provided, create a default one
            config_obj = IndicatorConfig(
                name="AccumulationDistributionLine",
                indicator_type=IndicatorType.VOLUME,
                timeframe=TimeFrame.D1,
                lookback_periods=lookback_period,
                parameters={
                    'lookback_period': lookback_period,
                    'trend_confirmation_period': trend_confirmation_period,
                    'divergence_period': divergence_period,
                    'volume_threshold': volume_threshold
                }
            )
        
        super().__init__(config=config_obj) # Added super call with config
        
        # Assign parameters from the config_obj or defaults
        self.lookback_period = config_obj.parameters.get('lookback_period', lookback_period)
        self.trend_confirmation_period = config_obj.parameters.get('trend_confirmation_period', trend_confirmation_period)
        self.divergence_period = config_obj.parameters.get('divergence_period', divergence_period)
        self.volume_threshold = config_obj.parameters.get('volume_threshold', volume_threshold)
        
        # Historical data storage
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.volumes: List[float] = []
        self.ad_line_values: List[float] = []
        self.money_flow_multipliers: List[float] = []
        self.money_flow_volumes: List[float] = []
        self.signals: List[AccumulationDistributionSignal] = []
        
        # Running A/D Line value
        self.ad_line_cumulative = 0.0
        
    def calculate(self, 
                  data=None,
                  high=None, 
                  low=None, 
                  close=None, 
                  volume=None, 
                  timestamp: Optional[Any] = None) -> AccumulationDistributionSignal:
        """
        Calculate A/D Line with comprehensive volume flow analysis
        """
        try:
            # Extract required params from data if not provided separately
            if data is not None:
                if isinstance(data, dict):
                    high = high if high is not None else data.get('high')
                    low = low if low is not None else data.get('low')
                    close = close if close is not None else data.get('close')
                    volume = volume if volume is not None else data.get('volume')
                elif hasattr(data, 'iloc') and len(data) >= 4:  # DataFrame-like
                    high = high if high is not None else data.iloc[1]  # Assuming OHLCV order
                    low = low if low is not None else data.iloc[2]
                    close = close if close is not None else data.iloc[3]
                    volume = volume if volume is not None else data.iloc[4]
                elif isinstance(data, (list, tuple)) and len(data) >= 4:
                    high = high if high is not None else data[1]  # Assuming OHLCV order
                    low = low if low is not None else data[2]
                    close = close if close is not None else data[3]
                    volume = volume if volume is not None else data[4]
            
            # Validate required parameters
            if any(x is None for x in [high, low, close, volume]):
                raise ValueError("Missing required parameters: high, low, close, and volume are required")
            
            # Convert to float if needed
            high = float(high)
            low = float(low)
            close = float(close)
            volume = float(volume)
            
            # Store current values
            self.highs.append(high)
            self.lows.append(low)
            self.closes.append(close)
            self.volumes.append(volume)
            
            # Calculate Money Flow Multiplier
            if high == low:  # Avoid division by zero
                money_flow_multiplier = 0.0
            else:
                money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
            
            # Calculate Money Flow Volume
            money_flow_volume = money_flow_multiplier * volume
            
            # Update cumulative A/D Line
            self.ad_line_cumulative += money_flow_volume
            
            # Store values
            self.ad_line_values.append(self.ad_line_cumulative)
            self.money_flow_multipliers.append(money_flow_multiplier)
            self.money_flow_volumes.append(money_flow_volume)
            
            # Limit history
            if len(self.closes) > 200:
                self.highs = self.highs[-200:]
                self.lows = self.lows[-200:]
                self.closes = self.closes[-200:]
                self.volumes = self.volumes[-200:]
                self.ad_line_values = self.ad_line_values[-200:]
                self.money_flow_multipliers = self.money_flow_multipliers[-200:]
                self.money_flow_volumes = self.money_flow_volumes[-200:]
            
            # Generate comprehensive signal
            signal = self._generate_signal(high, low, close, volume,
                                           self.ad_line_cumulative, money_flow_multiplier, 
                                           money_flow_volume)
            self.signals.append(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"A/D Line calculation error: {e}")
            return self._create_neutral_signal(close, 0.0, 0.0, 0.0)
    
    def _generate_signal(self, high: float, low: float, close: float, volume: float,
                         ad_line_value: float, money_flow_multiplier: float,
                         money_flow_volume: float) -> AccumulationDistributionSignal:
        """Generate comprehensive A/D Line signal with all analysis components"""
        
        # 1. Trend direction analysis
        trend_direction = self._analyze_trend_direction()
        
        # 2. Trend strength analysis
        trend_strength = self._analyze_trend_strength()
        
        # 3. Momentum phase analysis
        momentum_phase = self._analyze_momentum_phase()
        
        # 4. Divergence analysis
        divergence_signal = self._analyze_divergence()
        
        # 5. Volume confirmation analysis
        volume_confirmation = self._analyze_volume_confirmation(volume)
        
        # 6. Buying pressure analysis
        buying_pressure = self._analyze_buying_pressure(money_flow_multiplier)
        
        # 7. Flow analysis
        flow_analysis = self._analyze_money_flow()
        
        # 8. Calculate overall signal strength and confidence
        strength, confidence = self._calculate_signal_metrics(
            trend_direction, trend_strength, momentum_phase, divergence_signal,
            volume_confirmation, buying_pressure, money_flow_multiplier
        )
        
        # Determine signal type based on analysis
        from engines.indicator_base import SignalType
        if trend_direction == "accumulation" and strength > 0.6:
            signal_type = SignalType.BUY
        elif trend_direction == "distribution" and strength > 0.6:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        return AccumulationDistributionSignal(
            timestamp=datetime.now(),
            indicator_name="AccumulationDistributionLine",
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            ad_line_value=ad_line_value,
            money_flow_multiplier=money_flow_multiplier,
            money_flow_volume=money_flow_volume,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            momentum_phase=momentum_phase,
            divergence_signal=divergence_signal,
            volume_confirmation=volume_confirmation,
            buying_pressure=buying_pressure,
            flow_analysis=flow_analysis,
            metadata={
                "indicator": "AD_Line",
                "lookback_period": self.lookback_period,
                "volume_weighted": True,
                "money_flow_focus": True,
                "analysis_components": {
                    "trend": trend_direction,
                    "strength": trend_strength,
                    "momentum": momentum_phase,
                    "divergence": divergence_signal,
                    "volume_confirmation": volume_confirmation,
                    "buying_pressure": buying_pressure,
                    "flow_metrics": flow_analysis
                }
            }
        )
    
    def _analyze_trend_direction(self) -> str:
        """Analyze A/D Line trend direction"""
        if len(self.ad_line_values) < self.trend_confirmation_period:
            return "neutral"
        
        # Analyze recent A/D Line trend
        recent_values = self.ad_line_values[-self.trend_confirmation_period:]
        
        # Calculate slope
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        # Determine trend based on slope magnitude
        if slope > 0:
            return "accumulation"
        elif slope < 0:
            return "distribution"
        else:
            return "neutral"
    
    def _analyze_trend_strength(self) -> str:
        """Analyze strength of A/D Line trend"""
        if len(self.ad_line_values) < self.lookback_period:
            return "weak"
        
        # Calculate trend strength based on consistency and magnitude
        recent_values = self.ad_line_values[-self.lookback_period:]
        
        # Calculate R-squared to measure trend strength
        x = np.arange(len(recent_values))
        slope, intercept = np.polyfit(x, recent_values, 1)
        predicted = slope * x + intercept
        
        ss_res = np.sum((recent_values - predicted) ** 2)
        ss_tot = np.sum((recent_values - np.mean(recent_values)) ** 2)
        
        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # Classify strength
        if r_squared > 0.7:
            return "strong"
        elif r_squared > 0.4:
            return "moderate"
        else:
            return "weak"
    
    def _analyze_momentum_phase(self) -> str:
        """Analyze A/D Line momentum acceleration/deceleration"""
        if len(self.ad_line_values) < 10:
            return "stable"
        
        # Compare recent rate of change to previous rate of change
        recent_values = self.ad_line_values[-6:]
        previous_values = self.ad_line_values[-10:-4] if len(self.ad_line_values) >= 10 else []
        
        if len(previous_values) < 6:
            return "stable"
        
        # Calculate rate of change for both periods
        recent_slope = np.polyfit(np.arange(len(recent_values)), recent_values, 1)[0]
        previous_slope = np.polyfit(np.arange(len(previous_values)), previous_values, 1)[0]
        
        # Determine acceleration/deceleration
        if abs(recent_slope) > abs(previous_slope) * 1.2:
            return "accelerating"
        elif abs(recent_slope) < abs(previous_slope) * 0.8:
            return "decelerating"
        else:
            return "stable"
    
    def _analyze_divergence(self) -> Optional[str]:
        """Analyze divergence between price and A/D Line"""
        if len(self.closes) < self.divergence_period or len(self.ad_line_values) < self.divergence_period:
            return None
        
        # Analyze trends in both price and A/D Line
        price_data = self.closes[-self.divergence_period:]
        ad_data = self.ad_line_values[-self.divergence_period:]
        
        # Calculate slopes
        x = np.arange(len(price_data))
        price_slope = np.polyfit(x, price_data, 1)[0]
        ad_slope = np.polyfit(x, ad_data, 1)[0]
        
        # Detect significant divergences
        price_trend_up = price_slope > 0
        ad_trend_up = ad_slope > 0
        
        # Check for divergence with minimum significance
        min_price_change = abs(price_data[-1] - price_data[0]) / price_data[0]
        min_ad_change = abs(ad_data[-1] - ad_data[0])
        
        if min_price_change > 0.01 and min_ad_change > abs(np.mean(self.money_flow_volumes[-10:])):
            if price_trend_up and not ad_trend_up:
                return "bearish_divergence"
            elif not price_trend_up and ad_trend_up:
                return "bullish_divergence"
        
        return None
    
    def _analyze_volume_confirmation(self, current_volume: float) -> str:
        """Analyze volume confirmation of A/D Line signals"""
        if len(self.volumes) < 10:
            return "weak"
        
        # Calculate average volume
        avg_volume = np.mean(self.volumes[-10:])
        
        # Check volume relative to average
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        if volume_ratio >= self.volume_threshold:
            return "confirmed"
        elif volume_ratio >= 1.0:
            return "unconfirmed"
        else:
            return "weak"
    
    def _analyze_buying_pressure(self, money_flow_multiplier: float) -> float:
        """Analyze current buying/selling pressure"""
        
        # Money Flow Multiplier ranges from -1 to +1
        # Convert to buying pressure scale
        return money_flow_multiplier
    
    def _analyze_money_flow(self) -> Dict[str, Any]:
        """Analyze money flow patterns and characteristics"""
        
        flow_analysis = {
            "current_flow": 0.0,
            "average_flow": 0.0,
            "flow_consistency": 0.0,
            "accumulation_strength": 0.0,
            "distribution_strength": 0.0
        }
        
        if len(self.money_flow_volumes) < 10:
            return flow_analysis
        
        recent_flows = self.money_flow_volumes[-10:]
        
        # Current flow characteristics
        flow_analysis["current_flow"] = recent_flows[-1]
        flow_analysis["average_flow"] = np.mean(recent_flows)
        
        # Flow consistency (low std = consistent)
        flow_std = np.std(recent_flows)
        flow_mean = abs(np.mean(recent_flows))
        if flow_mean > 0:
            flow_analysis["flow_consistency"] = 1.0 - min(1.0, flow_std / flow_mean)
        
        # Accumulation/distribution strength
        positive_flows = [f for f in recent_flows if f > 0]
        negative_flows = [f for f in recent_flows if f < 0]
        
        if positive_flows:
            flow_analysis["accumulation_strength"] = np.mean(positive_flows) / max(abs(np.mean(recent_flows)), 1)
        
        if negative_flows:
            flow_analysis["distribution_strength"] = abs(np.mean(negative_flows)) / max(abs(np.mean(recent_flows)), 1)
        
        return flow_analysis
    
    def _calculate_signal_metrics(self, trend_direction: str, trend_strength: str,
                                  momentum_phase: str, divergence_signal: Optional[str],
                                  volume_confirmation: str, buying_pressure: float,
                                  money_flow_multiplier: float) -> Tuple[float, float]:
        """Calculate overall signal strength and confidence"""
        
        strength = 0.0
        confidence = 0.0
        
        # 1. Trend direction contribution
        if trend_direction == "accumulation":
            strength += 0.4
            confidence += 0.3
        elif trend_direction == "distribution":
            strength -= 0.4
            confidence += 0.3
        
        # 2. Buying pressure contribution
        strength += buying_pressure * 0.3
        confidence += abs(buying_pressure) * 0.2
        
        # 3. Divergence signals (strong indicator)
        if divergence_signal == "bullish_divergence":
            strength += 0.5
            confidence += 0.4
        elif divergence_signal == "bearish_divergence":
            strength -= 0.5
            confidence += 0.4
        
        # 4. Volume confirmation
        volume_multipliers = {
            "confirmed": 1.3,
            "unconfirmed": 1.0,
            "weak": 0.7
        }
        
        volume_mult = volume_multipliers.get(volume_confirmation, 1.0)
        strength *= volume_mult
        confidence *= volume_mult
        
        # 5. Trend strength adjustment
        strength_multipliers = {
            "strong": 1.2,
            "moderate": 1.0,
            "weak": 0.8
        }
        
        trend_mult = strength_multipliers.get(trend_strength, 1.0)
        strength *= trend_mult
        confidence *= trend_mult
        
        # 6. Momentum phase adjustment
        if momentum_phase == "accelerating":
            strength *= 1.1
            confidence += 0.1
        elif momentum_phase == "decelerating":
            strength *= 0.9
            confidence -= 0.1
        
        # Normalize
        strength = max(-1.0, min(1.0, strength))
        confidence = max(0.0, min(1.0, confidence))
        
        return strength, confidence
    
    def _create_neutral_signal(self, close: float, ad_line_value: float = 0.0,
                               money_flow_multiplier: float = 0.0, 
                               money_flow_volume: float = 0.0) -> AccumulationDistributionSignal:
        """Create neutral signal for insufficient data"""
        from engines.indicator_base import SignalType
        return AccumulationDistributionSignal(
            timestamp=datetime.now(),
            indicator_name="AccumulationDistributionLine",
            signal_type=SignalType.NEUTRAL,
            strength=0.0,
            confidence=0.0,
            ad_line_value=ad_line_value,
            money_flow_multiplier=money_flow_multiplier,
            money_flow_volume=money_flow_volume,
            trend_direction="neutral",
            trend_strength="weak",
            momentum_phase="stable",
            divergence_signal=None,
            volume_confirmation="weak",
            buying_pressure=0.0,
            flow_analysis={},
            metadata={
                "indicator": "AD_Line",
                "status": "insufficient_data"
            }
        )

    def generate_signal(self, high: float, low: float, close: float, volume: float, timestamp: Optional[Any] = None) -> IndicatorSignal:
        """Generate signal using standard interface - delegates to calculate method"""
        return self.calculate(high, low, close, volume, timestamp)

# Test function
def test_accumulation_distribution():
    """Test A/D Line with realistic EURUSD data"""
    print("Testing Accumulation/Distribution Line...")
    
    # Simulate EURUSD price data with accumulation/distribution phases
    np.random.seed(42)
    base_price = 1.1000
    prices = []
    volumes = []
    
    for i in range(100):
        # Create different phases
        if i < 25:
            # Accumulation phase - higher closes, increasing volume
            bias = 0.7  # Favor higher closes
            volume_trend = 1000000 + i * 10000
        elif i < 50:
            # Distribution phase - lower closes, high volume
            bias = 0.3  # Favor lower closes
            volume_trend = 1250000 - (i - 25) * 5000
        elif i < 75:
            # Neutral phase - mixed closes, normal volume
            bias = 0.5  # Neutral
            volume_trend = 1000000
        else:
            # Accumulation phase again
            bias = 0.6
            volume_trend = 1000000 + (i - 75) * 8000
        
        # Generate price movement
        volatility = 0.0005
        price_change = np.random.normal(0, volatility)
        price = max(0.5, base_price + price_change * i * 0.001)
        
        # Generate OHLC with bias
        spread = volatility * 2
        if np.random.random() < bias:
            # Bias toward higher close
            close = price + np.random.uniform(0, spread * 0.7)
            open_price = close - np.random.uniform(0, spread * 0.3)
        else:
            # Bias toward lower close
            close = price - np.random.uniform(0, spread * 0.7)
            open_price = close + np.random.uniform(0, spread * 0.3)
        
        high = max(open_price, close) + np.random.uniform(0, spread * 0.2)
        low = min(open_price, close) - np.random.uniform(0, spread * 0.2)
        
        # Generate volume with trend and noise
        volume = max(500000, volume_trend + np.random.normal(0, 100000))
        
        prices.append((high, low, close))
        volumes.append(volume)
    
    # Test A/D Line
    ad_line = AccumulationDistributionLine(lookback_period=20)
    
    print("\nA/D Line Test Results:")
    print("-" * 60)
    
    for i, ((high, low, close), volume) in enumerate(zip(prices, volumes)):
        signal = ad_line.calculate(high, low, close, volume)
        
        # Print key signals
        if i >= 25 and i % 15 == 0:  # After enough data and every 15th
            print(f"Period {i+1}:")
            print(f"  Price: {close:.5f}")
            print(f"  A/D Line: {signal.ad_line_value:.0f}")
            print(f"  Money Flow Mult: {signal.money_flow_multiplier:.3f}")
            print(f"  Money Flow Vol: {signal.money_flow_volume:.0f}")
            print(f"  Trend: {signal.trend_direction}")
            print(f"  Strength: {signal.trend_strength}")
            print(f"  Momentum: {signal.momentum_phase}")
            print(f"  Divergence: {signal.divergence_signal}")
            print(f"  Volume Confirm: {signal.volume_confirmation}")
            print(f"  Buying Pressure: {signal.buying_pressure:.3f}")
            print(f"  Signal: {signal.strength:.3f} (conf: {signal.confidence:.3f})")
            print()

if __name__ == "__main__":
    test_accumulation_distribution()
