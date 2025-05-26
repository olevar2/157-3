"""
<<<<<<< HEAD
Order Flow Imbalance Module
Bid/ask volume imbalances detection for scalping and day trading
Optimized for order flow imbalance alerts for quick profits.
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
class OrderFlowData:
    """Order flow data structure"""
    timestamp: float
    bid_volume: float
    ask_volume: float
    bid_price: float
    ask_price: float
    spread: float
    total_volume: float
    imbalance_ratio: float  # ask_volume / bid_volume


@dataclass
class ImbalanceSignal:
    """Order flow imbalance signal"""
    signal_type: str  # 'buy_imbalance', 'sell_imbalance', 'balanced'
    imbalance_strength: float  # 0-1
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    duration_periods: int
    volume_intensity: float
    price_impact_probability: float


@dataclass
class FlowPattern:
    """Order flow pattern"""
    pattern_type: str  # 'aggressive_buying', 'aggressive_selling', 'absorption', 'exhaustion'
    start_time: float
    end_time: float
    total_volume: float
    average_imbalance: float
    price_movement: float
    pattern_strength: float


@dataclass
class OrderFlowResult:
    """Order flow imbalance analysis result"""
    symbol: str
    timestamp: float
    timeframe: str
    current_imbalance: ImbalanceSignal
    imbalance_history: List[ImbalanceSignal]
    flow_patterns: List[FlowPattern]
    market_pressure: str  # 'buying_pressure', 'selling_pressure', 'balanced'
    scalping_alerts: List[Dict[str, Union[str, float]]]
    execution_metrics: Dict[str, float]


class OrderFlowImbalance:
    """
    Order Flow Imbalance Engine for Scalping
    Provides bid/ask volume imbalance detection for quick profit opportunities
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # Imbalance detection parameters
        self.imbalance_threshold_strong = 2.0  # 2:1 ratio for strong imbalance
        self.imbalance_threshold_medium = 1.5  # 1.5:1 ratio for medium imbalance
        self.imbalance_duration_min = 3  # Minimum periods for sustained imbalance
        
        # Pattern detection parameters
        self.pattern_min_volume = 1000  # Minimum volume for pattern recognition
        self.pattern_min_duration = 5   # Minimum periods for pattern
        self.absorption_threshold = 0.8  # Threshold for absorption detection
        
        # Alert parameters
        self.alert_confidence_threshold = 0.7
        self.volume_spike_threshold = 1.8
        
        # Performance optimization
        self.flow_cache: Dict[str, deque] = {}
        self.pattern_cache: Dict[str, List[FlowPattern]] = {}

    async def initialize(self) -> bool:
        """Initialize the Order Flow Imbalance engine"""
        try:
            self.logger.info("Initializing Order Flow Imbalance Engine...")
            
            # Test order flow analysis with sample data
            test_data = self._generate_test_data()
            test_result = await self._analyze_order_flow(test_data)
            
            if test_result and len(test_result) > 0:
                self.ready = True
                self.logger.info("✅ Order Flow Imbalance Engine initialized")
                return True
            else:
                raise Exception("Order flow analysis test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Order Flow Imbalance Engine initialization failed: {e}")
            return False

    async def analyze_order_flow_imbalance(self, symbol: str, order_flow_data: List[Dict],
                                         timeframe: str = 'M1') -> OrderFlowResult:
        """
        Analyze order flow imbalances for scalping opportunities
        
        Args:
            symbol: Currency pair symbol
            order_flow_data: List of order flow data dictionaries
            timeframe: Chart timeframe (M1-M5)
            
        Returns:
            OrderFlowResult with imbalance analysis
        """
        if not self.ready:
            raise Exception("Order Flow Imbalance Engine not initialized")
            
        if len(order_flow_data) < 10:
            raise Exception("Insufficient data for order flow analysis (minimum 10 periods)")
            
        try:
            start_time = time.time()
            
            # Prepare order flow data
            flow_data = await self._prepare_order_flow_data(order_flow_data)
            
            # Detect current imbalance
            current_imbalance = await self._detect_current_imbalance(flow_data)
            
            # Analyze imbalance history
            imbalance_history = await self._analyze_imbalance_history(flow_data)
            
            # Detect flow patterns
            flow_patterns = await self._detect_flow_patterns(flow_data)
            
            # Determine market pressure
            market_pressure = await self._determine_market_pressure(flow_data, imbalance_history)
            
            # Generate scalping alerts
            scalping_alerts = await self._generate_scalping_alerts(
                symbol, flow_data, current_imbalance, flow_patterns, timeframe
            )
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(
                flow_data, imbalance_history, scalping_alerts
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Order flow analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return OrderFlowResult(
                symbol=symbol,
                timestamp=time.time(),
                timeframe=timeframe,
                current_imbalance=current_imbalance,
                imbalance_history=imbalance_history,
                flow_patterns=flow_patterns,
                market_pressure=market_pressure,
                scalping_alerts=scalping_alerts,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Order flow analysis failed for {symbol}: {e}")
            raise

    async def _prepare_order_flow_data(self, order_flow_data: List[Dict]) -> List[OrderFlowData]:
        """Prepare order flow data structures"""
        flow_data = []
        
        for data in order_flow_data:
            timestamp = float(data.get('timestamp', time.time()))
            bid_volume = float(data.get('bid_volume', 0))
            ask_volume = float(data.get('ask_volume', 0))
            bid_price = float(data.get('bid_price', 0))
            ask_price = float(data.get('ask_price', 0))
            
            spread = ask_price - bid_price
            total_volume = bid_volume + ask_volume
            
            # Calculate imbalance ratio (avoid division by zero)
            if bid_volume > 0:
                imbalance_ratio = ask_volume / bid_volume
            elif ask_volume > 0:
                imbalance_ratio = float('inf')  # Pure ask volume
            else:
                imbalance_ratio = 1.0  # No volume
            
            flow_data.append(OrderFlowData(
                timestamp=timestamp,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                bid_price=bid_price,
                ask_price=ask_price,
                spread=spread,
                total_volume=total_volume,
                imbalance_ratio=imbalance_ratio
            ))
        
        return flow_data

    async def _detect_current_imbalance(self, flow_data: List[OrderFlowData]) -> ImbalanceSignal:
        """Detect current order flow imbalance"""
        if not flow_data:
            return ImbalanceSignal(
                signal_type='balanced',
                imbalance_strength=0.5,
                direction='neutral',
                confidence=0.5,
                duration_periods=0,
                volume_intensity=0.5,
                price_impact_probability=0.5
            )
        
        current_flow = flow_data[-1]
        
        # Determine signal type and direction
        if current_flow.imbalance_ratio > self.imbalance_threshold_strong:
            signal_type = 'buy_imbalance'
            direction = 'bullish'
            imbalance_strength = min((current_flow.imbalance_ratio - 1) / (self.imbalance_threshold_strong - 1), 1.0)
        elif current_flow.imbalance_ratio < (1 / self.imbalance_threshold_strong):
            signal_type = 'sell_imbalance'
            direction = 'bearish'
            imbalance_strength = min((1 - current_flow.imbalance_ratio) / (1 - 1/self.imbalance_threshold_strong), 1.0)
        else:
            signal_type = 'balanced'
            direction = 'neutral'
            imbalance_strength = 0.5
        
        # Calculate duration of current imbalance
        duration_periods = await self._calculate_imbalance_duration(flow_data, signal_type)
        
        # Calculate volume intensity
        if len(flow_data) >= 10:
            avg_volume = statistics.mean([f.total_volume for f in flow_data[-10:]])
            volume_intensity = min(current_flow.total_volume / avg_volume, 2.0) / 2.0
        else:
            volume_intensity = 0.5
        
        # Calculate confidence
        confidence = (imbalance_strength + volume_intensity + min(duration_periods / 5, 1.0)) / 3
        
        # Calculate price impact probability
        price_impact_probability = confidence * imbalance_strength
        
        return ImbalanceSignal(
            signal_type=signal_type,
            imbalance_strength=imbalance_strength,
            direction=direction,
            confidence=confidence,
            duration_periods=duration_periods,
            volume_intensity=volume_intensity,
            price_impact_probability=price_impact_probability
        )

    async def _calculate_imbalance_duration(self, flow_data: List[OrderFlowData], 
                                          signal_type: str) -> int:
        """Calculate how long the current imbalance has persisted"""
        duration = 0
        
        # Look backwards from current data
        for i in range(len(flow_data) - 1, -1, -1):
            flow = flow_data[i]
            
            current_signal = await self._classify_imbalance(flow)
            
            if current_signal == signal_type:
                duration += 1
            else:
                break
        
        return duration

    async def _classify_imbalance(self, flow: OrderFlowData) -> str:
        """Classify a single flow data point"""
        if flow.imbalance_ratio > self.imbalance_threshold_medium:
            return 'buy_imbalance'
        elif flow.imbalance_ratio < (1 / self.imbalance_threshold_medium):
            return 'sell_imbalance'
        else:
            return 'balanced'

    async def _analyze_imbalance_history(self, flow_data: List[OrderFlowData]) -> List[ImbalanceSignal]:
        """Analyze historical imbalance patterns"""
        history = []
        
        # Analyze last 20 periods or available data
        analysis_periods = min(20, len(flow_data))
        
        for i in range(len(flow_data) - analysis_periods, len(flow_data)):
            if i < 0:
                continue
                
            # Create a subset for analysis
            subset = flow_data[max(0, i-5):i+1]
            
            if subset:
                imbalance = await self._detect_current_imbalance(subset)
                history.append(imbalance)
        
        return history

    async def _detect_flow_patterns(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect order flow patterns"""
        patterns = []
        
        if len(flow_data) < self.pattern_min_duration:
            return patterns
        
        # Detect aggressive buying patterns
        patterns.extend(await self._detect_aggressive_buying(flow_data))
        
        # Detect aggressive selling patterns
        patterns.extend(await self._detect_aggressive_selling(flow_data))
        
        # Detect absorption patterns
        patterns.extend(await self._detect_absorption_patterns(flow_data))
        
        # Detect exhaustion patterns
        patterns.extend(await self._detect_exhaustion_patterns(flow_data))
        
        return patterns

    async def _detect_aggressive_buying(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect aggressive buying patterns"""
        patterns = []
        
        # Look for sustained periods of buy imbalance with increasing volume
        for i in range(len(flow_data) - self.pattern_min_duration):
            window = flow_data[i:i + self.pattern_min_duration]
            
            # Check for sustained buy imbalance
            buy_imbalance_count = sum(1 for f in window if f.imbalance_ratio > self.imbalance_threshold_medium)
            
            if buy_imbalance_count >= self.pattern_min_duration * 0.7:  # 70% of periods
                total_volume = sum(f.total_volume for f in window)
                avg_imbalance = statistics.mean([f.imbalance_ratio for f in window])
                
                if total_volume >= self.pattern_min_volume:
                    # Calculate price movement (simplified)
                    price_movement = (window[-1].ask_price - window[0].bid_price) / window[0].bid_price
                    
                    pattern_strength = min(avg_imbalance / self.imbalance_threshold_strong, 1.0)
                    
                    patterns.append(FlowPattern(
                        pattern_type='aggressive_buying',
                        start_time=window[0].timestamp,
                        end_time=window[-1].timestamp,
                        total_volume=total_volume,
                        average_imbalance=avg_imbalance,
                        price_movement=price_movement,
                        pattern_strength=pattern_strength
                    ))
        
        return patterns

    async def _detect_aggressive_selling(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect aggressive selling patterns"""
        patterns = []
        
        # Look for sustained periods of sell imbalance with increasing volume
        for i in range(len(flow_data) - self.pattern_min_duration):
            window = flow_data[i:i + self.pattern_min_duration]
            
            # Check for sustained sell imbalance
            sell_imbalance_count = sum(1 for f in window if f.imbalance_ratio < (1 / self.imbalance_threshold_medium))
            
            if sell_imbalance_count >= self.pattern_min_duration * 0.7:  # 70% of periods
                total_volume = sum(f.total_volume for f in window)
                avg_imbalance = statistics.mean([1/f.imbalance_ratio if f.imbalance_ratio > 0 else 1 for f in window])
                
                if total_volume >= self.pattern_min_volume:
                    # Calculate price movement (simplified)
                    price_movement = (window[-1].bid_price - window[0].ask_price) / window[0].ask_price
                    
                    pattern_strength = min(avg_imbalance / self.imbalance_threshold_strong, 1.0)
                    
                    patterns.append(FlowPattern(
                        pattern_type='aggressive_selling',
                        start_time=window[0].timestamp,
                        end_time=window[-1].timestamp,
                        total_volume=total_volume,
                        average_imbalance=avg_imbalance,
                        price_movement=price_movement,
                        pattern_strength=pattern_strength
                    ))
        
        return patterns

    async def _detect_absorption_patterns(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect volume absorption patterns"""
        patterns = []
        
        # Look for high volume with minimal price movement
        for i in range(len(flow_data) - self.pattern_min_duration):
            window = flow_data[i:i + self.pattern_min_duration]
            
            total_volume = sum(f.total_volume for f in window)
            avg_volume = statistics.mean([f.total_volume for f in flow_data[max(0, i-10):i]])
            
            if total_volume > avg_volume * self.volume_spike_threshold:
                # Check for minimal price movement
                price_range = max(f.ask_price for f in window) - min(f.bid_price for f in window)
                avg_spread = statistics.mean([f.spread for f in window])
                
                if price_range <= avg_spread * 2:  # Price movement within 2x average spread
                    avg_imbalance = statistics.mean([f.imbalance_ratio for f in window])
                    pattern_strength = min(total_volume / (avg_volume * self.volume_spike_threshold), 1.0)
                    
                    patterns.append(FlowPattern(
                        pattern_type='absorption',
                        start_time=window[0].timestamp,
                        end_time=window[-1].timestamp,
                        total_volume=total_volume,
                        average_imbalance=avg_imbalance,
                        price_movement=price_range,
                        pattern_strength=pattern_strength
                    ))
        
        return patterns

    async def _detect_exhaustion_patterns(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect exhaustion patterns"""
        patterns = []
        
        # Look for decreasing volume with continued imbalance
        for i in range(len(flow_data) - self.pattern_min_duration):
            window = flow_data[i:i + self.pattern_min_duration]
            
            # Check for volume decline
            volumes = [f.total_volume for f in window]
            volume_trend = await self._calculate_trend(volumes)
            
            if volume_trend < -0.1:  # Declining volume
                # Check for sustained imbalance
                imbalances = [f.imbalance_ratio for f in window]
                avg_imbalance = statistics.mean(imbalances)
                
                if avg_imbalance > self.imbalance_threshold_medium or avg_imbalance < (1 / self.imbalance_threshold_medium):
                    total_volume = sum(volumes)
                    price_movement = (window[-1].ask_price - window[0].bid_price) / window[0].bid_price
                    pattern_strength = abs(volume_trend) * 0.5 + (abs(avg_imbalance - 1) * 0.5)
                    
                    patterns.append(FlowPattern(
                        pattern_type='exhaustion',
                        start_time=window[0].timestamp,
                        end_time=window[-1].timestamp,
                        total_volume=total_volume,
                        average_imbalance=avg_imbalance,
                        price_movement=price_movement,
                        pattern_strength=min(pattern_strength, 1.0)
                    ))
        
        return patterns

    async def _calculate_trend(self, values: List[float]) -> float:
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

    async def _determine_market_pressure(self, flow_data: List[OrderFlowData], 
                                       history: List[ImbalanceSignal]) -> str:
        """Determine overall market pressure"""
        if not history:
            return 'balanced'
        
        # Count recent imbalance types
        recent_signals = history[-10:] if len(history) >= 10 else history
        
        buy_pressure = sum(1 for s in recent_signals if s.signal_type == 'buy_imbalance')
        sell_pressure = sum(1 for s in recent_signals if s.signal_type == 'sell_imbalance')
        
        if buy_pressure > sell_pressure * 1.5:
            return 'buying_pressure'
        elif sell_pressure > buy_pressure * 1.5:
            return 'selling_pressure'
        else:
            return 'balanced'

    async def _generate_scalping_alerts(self, symbol: str, flow_data: List[OrderFlowData],
                                      current_imbalance: ImbalanceSignal,
                                      patterns: List[FlowPattern],
                                      timeframe: str) -> List[Dict[str, Union[str, float]]]:
        """Generate scalping alerts based on order flow analysis"""
        alerts = []
        
        # Strong imbalance alerts
        if (current_imbalance.confidence >= self.alert_confidence_threshold and 
            current_imbalance.imbalance_strength > 0.7):
            
            alerts.append({
                'type': 'order_flow_imbalance',
                'signal': 'buy' if current_imbalance.direction == 'bullish' else 'sell',
                'confidence': current_imbalance.confidence,
                'imbalance_strength': current_imbalance.imbalance_strength,
                'duration': current_imbalance.duration_periods,
                'volume_intensity': current_imbalance.volume_intensity,
                'price_impact_probability': current_imbalance.price_impact_probability,
                'timeframe': timeframe,
                'alert_reason': 'strong_order_flow_imbalance'
            })
        
        # Pattern-based alerts
        for pattern in patterns:
            if pattern.pattern_strength > 0.7:
                if pattern.pattern_type == 'aggressive_buying':
                    alerts.append({
                        'type': 'aggressive_buying',
                        'signal': 'buy',
                        'confidence': pattern.pattern_strength,
                        'pattern_duration': (pattern.end_time - pattern.start_time) / 60,  # minutes
                        'total_volume': pattern.total_volume,
                        'timeframe': timeframe,
                        'alert_reason': 'aggressive_buying_detected'
                    })
                elif pattern.pattern_type == 'aggressive_selling':
                    alerts.append({
                        'type': 'aggressive_selling',
                        'signal': 'sell',
                        'confidence': pattern.pattern_strength,
                        'pattern_duration': (pattern.end_time - pattern.start_time) / 60,  # minutes
                        'total_volume': pattern.total_volume,
                        'timeframe': timeframe,
                        'alert_reason': 'aggressive_selling_detected'
                    })
                elif pattern.pattern_type == 'absorption':
                    alerts.append({
                        'type': 'volume_absorption',
                        'signal': 'reversal_pending',
                        'confidence': pattern.pattern_strength,
                        'absorption_volume': pattern.total_volume,
                        'timeframe': timeframe,
                        'alert_reason': 'volume_absorption_detected'
                    })
        
        return alerts

    async def _calculate_execution_metrics(self, flow_data: List[OrderFlowData],
                                         history: List[ImbalanceSignal],
                                         alerts: List[Dict]) -> Dict[str, float]:
        """Calculate execution metrics for order flow analysis"""
        metrics = {
            'total_volume': sum(f.total_volume for f in flow_data),
            'avg_imbalance_ratio': statistics.mean([f.imbalance_ratio for f in flow_data]) if flow_data else 1.0,
            'imbalance_periods': len([s for s in history if s.signal_type != 'balanced']),
            'alert_count': len(alerts),
            'avg_alert_confidence': statistics.mean([a['confidence'] for a in alerts]) if alerts else 0.0,
            'buying_pressure_ratio': len([s for s in history if s.direction == 'bullish']) / len(history) if history else 0.5,
            'selling_pressure_ratio': len([s for s in history if s.direction == 'bearish']) / len(history) if history else 0.5
        }
        
        return metrics

    def _generate_test_data(self) -> List[Dict]:
        """Generate test data for initialization"""
        test_data = []
        base_bid = 1.0998
        base_ask = 1.1002
        
        for i in range(20):
            # Create varying imbalances
            if i % 7 == 0:  # Occasional strong imbalance
                bid_volume = np.random.uniform(500, 1000)
                ask_volume = np.random.uniform(1500, 2500)  # Strong buy imbalance
            elif i % 5 == 0:
                bid_volume = np.random.uniform(1500, 2500)  # Strong sell imbalance
                ask_volume = np.random.uniform(500, 1000)
            else:
                bid_volume = np.random.uniform(800, 1200)
                ask_volume = np.random.uniform(800, 1200)
            
            # Small price movements
            price_change = (np.random.random() - 0.5) * 0.0002
            bid_price = base_bid + price_change
            ask_price = base_ask + price_change
            
            test_data.append({
                'timestamp': time.time() - (20 - i) * 60,  # M1 intervals
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'bid_price': bid_price,
                'ask_price': ask_price
            })
            
        return test_data

    async def _analyze_order_flow(self, test_data: List[Dict]) -> List[ImbalanceSignal]:
        """Test order flow analysis"""
        try:
            flow_data = await self._prepare_order_flow_data(test_data)
            return await self._analyze_imbalance_history(flow_data)
        except Exception:
            return []
=======
Order Flow Imbalance Analysis for Day Trading
Analyzes bid/ask volume imbalances to identify institutional activity and market direction.

This module detects order flow imbalances that can signal:
- Institutional buying/selling pressure
- Market maker activity
- Liquidity gaps and potential price movements
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

class ImbalanceType(Enum):
    """Order flow imbalance types"""
    BUY_IMBALANCE = "buy_imbalance"
    SELL_IMBALANCE = "sell_imbalance"
    BALANCED = "balanced"
    EXTREME_BUY = "extreme_buy"
    EXTREME_SELL = "extreme_sell"

class ImbalanceStrength(Enum):
    """Imbalance strength levels"""
    EXTREME = "extreme"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"

@dataclass
class OrderFlowBar:
    """Order flow analysis for a single bar"""
    timestamp: datetime
    buy_volume: float
    sell_volume: float
    total_volume: float
    buy_ratio: float
    sell_ratio: float
    imbalance_ratio: float
    imbalance_type: ImbalanceType
    imbalance_strength: ImbalanceStrength
    delta: float  # Buy volume - Sell volume
    cumulative_delta: float
    volume_weighted_price: float

@dataclass
class OrderFlowAnalysisResult:
    """Complete order flow analysis result"""
    symbol: str
    timeframe: str
    analysis_time: datetime
    bars: List[OrderFlowBar]
    current_imbalance: ImbalanceType
    imbalance_strength: ImbalanceStrength
    cumulative_delta: float
    delta_trend: str
    institutional_activity: bool
    market_sentiment: str
    volume_profile: Dict[str, float]
    recommendations: List[str]

class OrderFlowImbalance:
    """
    Order Flow Imbalance analyzer for detecting institutional activity.
    
    Analyzes:
    - Bid/Ask volume imbalances
    - Delta (buy volume - sell volume)
    - Cumulative delta trends
    - Volume-weighted price levels
    - Institutional footprint detection
    """
    
    def __init__(self, lookback_periods: int = 20):
        """
        Initialize order flow analyzer.
        
        Args:
            lookback_periods: Number of periods for analysis
        """
        self.lookback_periods = lookback_periods
        
    def analyze_order_flow(self, data: pd.DataFrame, symbol: str, timeframe: str) -> OrderFlowAnalysisResult:
        """
        Perform complete order flow imbalance analysis.
        
        Args:
            data: OHLCV data with bid/ask volume columns
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            OrderFlowAnalysisResult with complete analysis
        """
        try:
            # Validate input data
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Estimate buy/sell volume if not provided
            if 'buy_volume' not in data.columns or 'sell_volume' not in data.columns:
                data = self._estimate_buy_sell_volume(data)
            
            # Calculate order flow metrics
            data = self._calculate_order_flow_metrics(data)
            
            # Analyze individual bars
            order_flow_bars = []
            cumulative_delta = 0.0
            
            for i in range(len(data)):
                bar_data = data.iloc[i]
                cumulative_delta += bar_data['delta']
                
                bar = self._analyze_order_flow_bar(bar_data, cumulative_delta)
                order_flow_bars.append(bar)
            
            # Determine overall analysis
            current_imbalance = self._determine_current_imbalance(order_flow_bars[-10:])
            imbalance_strength = self._calculate_imbalance_strength(order_flow_bars[-5:])
            delta_trend = self._analyze_delta_trend(order_flow_bars[-20:])
            institutional_activity = self._detect_institutional_activity(order_flow_bars[-15:])
            market_sentiment = self._determine_market_sentiment(order_flow_bars[-10:])
            volume_profile = self._calculate_volume_profile(data.tail(50))
            recommendations = self._generate_recommendations(current_imbalance, imbalance_strength, delta_trend)
            
            return OrderFlowAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                analysis_time=datetime.now(),
                bars=order_flow_bars,
                current_imbalance=current_imbalance,
                imbalance_strength=imbalance_strength,
                cumulative_delta=cumulative_delta,
                delta_trend=delta_trend,
                institutional_activity=institutional_activity,
                market_sentiment=market_sentiment,
                volume_profile=volume_profile,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Order flow analysis failed for {symbol}: {e}")
            raise
    
    def _estimate_buy_sell_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """Estimate buy/sell volume from price action"""
        data = data.copy()
        
        # Simple estimation based on close position in range
        data['range'] = data['high'] - data['low']
        data['close_position'] = (data['close'] - data['low']) / (data['range'] + 1e-10)
        
        # Estimate buy/sell volume based on close position and volume
        data['buy_volume'] = data['volume'] * data['close_position']
        data['sell_volume'] = data['volume'] * (1 - data['close_position'])
        
        # Adjust for price change direction
        data['price_change'] = data['close'] - data['open']
        
        # If price went up, assume more buying pressure
        up_mask = data['price_change'] > 0
        data.loc[up_mask, 'buy_volume'] *= 1.2
        data.loc[up_mask, 'sell_volume'] *= 0.8
        
        # If price went down, assume more selling pressure
        down_mask = data['price_change'] < 0
        data.loc[down_mask, 'buy_volume'] *= 0.8
        data.loc[down_mask, 'sell_volume'] *= 1.2
        
        # Ensure volumes sum to total volume
        total_estimated = data['buy_volume'] + data['sell_volume']
        data['buy_volume'] = data['buy_volume'] / total_estimated * data['volume']
        data['sell_volume'] = data['sell_volume'] / total_estimated * data['volume']
        
        return data
    
    def _calculate_order_flow_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow metrics"""
        data = data.copy()
        
        # Calculate ratios
        data['buy_ratio'] = data['buy_volume'] / (data['volume'] + 1e-10)
        data['sell_ratio'] = data['sell_volume'] / (data['volume'] + 1e-10)
        
        # Calculate delta (net buying pressure)
        data['delta'] = data['buy_volume'] - data['sell_volume']
        
        # Calculate imbalance ratio (-1 to 1, where 1 = all buying, -1 = all selling)
        data['imbalance_ratio'] = data['delta'] / (data['volume'] + 1e-10)
        
        # Calculate volume-weighted price
        data['volume_weighted_price'] = (
            (data['high'] + data['low'] + data['close']) / 3 * data['volume']
        )
        
        # Calculate moving averages for trend analysis
        data['delta_ma_5'] = data['delta'].rolling(5).mean()
        data['delta_ma_10'] = data['delta'].rolling(10).mean()
        data['imbalance_ma_5'] = data['imbalance_ratio'].rolling(5).mean()
        
        return data
    
    def _analyze_order_flow_bar(self, bar_data: pd.Series, cumulative_delta: float) -> OrderFlowBar:
        """Analyze individual bar for order flow"""
        
        # Determine imbalance type
        imbalance_type = self._classify_imbalance_type(bar_data['imbalance_ratio'])
        
        # Determine imbalance strength
        imbalance_strength = self._classify_imbalance_strength(abs(bar_data['imbalance_ratio']))
        
        return OrderFlowBar(
            timestamp=bar_data['timestamp'],
            buy_volume=bar_data['buy_volume'],
            sell_volume=bar_data['sell_volume'],
            total_volume=bar_data['volume'],
            buy_ratio=bar_data['buy_ratio'],
            sell_ratio=bar_data['sell_ratio'],
            imbalance_ratio=bar_data['imbalance_ratio'],
            imbalance_type=imbalance_type,
            imbalance_strength=imbalance_strength,
            delta=bar_data['delta'],
            cumulative_delta=cumulative_delta,
            volume_weighted_price=bar_data['volume_weighted_price']
        )
    
    def _classify_imbalance_type(self, imbalance_ratio: float) -> ImbalanceType:
        """Classify imbalance type based on ratio"""
        if imbalance_ratio > 0.6:
            return ImbalanceType.EXTREME_BUY
        elif imbalance_ratio > 0.2:
            return ImbalanceType.BUY_IMBALANCE
        elif imbalance_ratio < -0.6:
            return ImbalanceType.EXTREME_SELL
        elif imbalance_ratio < -0.2:
            return ImbalanceType.SELL_IMBALANCE
        else:
            return ImbalanceType.BALANCED
    
    def _classify_imbalance_strength(self, abs_imbalance_ratio: float) -> ImbalanceStrength:
        """Classify imbalance strength"""
        if abs_imbalance_ratio > 0.7:
            return ImbalanceStrength.EXTREME
        elif abs_imbalance_ratio > 0.5:
            return ImbalanceStrength.STRONG
        elif abs_imbalance_ratio > 0.3:
            return ImbalanceStrength.MODERATE
        elif abs_imbalance_ratio > 0.1:
            return ImbalanceStrength.WEAK
        else:
            return ImbalanceStrength.NEUTRAL
    
    def _determine_current_imbalance(self, recent_bars: List[OrderFlowBar]) -> ImbalanceType:
        """Determine current market imbalance"""
        if not recent_bars:
            return ImbalanceType.BALANCED
        
        # Weight recent bars more heavily
        weighted_imbalance = 0.0
        total_weight = 0.0
        
        for i, bar in enumerate(recent_bars):
            weight = i + 1  # More recent bars get higher weight
            weighted_imbalance += bar.imbalance_ratio * weight
            total_weight += weight
        
        avg_imbalance = weighted_imbalance / total_weight if total_weight > 0 else 0.0
        return self._classify_imbalance_type(avg_imbalance)
    
    def _calculate_imbalance_strength(self, recent_bars: List[OrderFlowBar]) -> ImbalanceStrength:
        """Calculate overall imbalance strength"""
        if not recent_bars:
            return ImbalanceStrength.NEUTRAL
        
        avg_abs_imbalance = np.mean([abs(bar.imbalance_ratio) for bar in recent_bars])
        return self._classify_imbalance_strength(avg_abs_imbalance)
    
    def _analyze_delta_trend(self, bars: List[OrderFlowBar]) -> str:
        """Analyze cumulative delta trend"""
        if len(bars) < 10:
            return "insufficient_data"
        
        recent_delta = np.mean([bar.cumulative_delta for bar in bars[-5:]])
        older_delta = np.mean([bar.cumulative_delta for bar in bars[:5]])
        
        if recent_delta > older_delta * 1.1:
            return "increasing"
        elif recent_delta < older_delta * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_institutional_activity(self, bars: List[OrderFlowBar]) -> bool:
        """Detect potential institutional activity"""
        if not bars:
            return False
        
        # Look for sustained imbalances with high volume
        strong_imbalances = sum(1 for bar in bars 
                              if bar.imbalance_strength in [ImbalanceStrength.STRONG, ImbalanceStrength.EXTREME])
        
        # Look for large delta moves
        large_deltas = sum(1 for bar in bars if abs(bar.delta) > np.std([b.delta for b in bars]) * 2)
        
        return strong_imbalances >= 3 or large_deltas >= 2
    
    def _determine_market_sentiment(self, bars: List[OrderFlowBar]) -> str:
        """Determine overall market sentiment"""
        if not bars:
            return "neutral"
        
        buy_imbalances = sum(1 for bar in bars 
                           if bar.imbalance_type in [ImbalanceType.BUY_IMBALANCE, ImbalanceType.EXTREME_BUY])
        sell_imbalances = sum(1 for bar in bars 
                            if bar.imbalance_type in [ImbalanceType.SELL_IMBALANCE, ImbalanceType.EXTREME_SELL])
        
        if buy_imbalances > sell_imbalances + 2:
            return "bullish"
        elif sell_imbalances > buy_imbalances + 2:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile metrics"""
        if len(data) == 0:
            return {}
        
        total_volume = data['volume'].sum()
        buy_volume = data['buy_volume'].sum()
        sell_volume = data['sell_volume'].sum()
        
        return {
            'total_volume': total_volume,
            'buy_volume_pct': (buy_volume / total_volume * 100) if total_volume > 0 else 0,
            'sell_volume_pct': (sell_volume / total_volume * 100) if total_volume > 0 else 0,
            'avg_imbalance': data['imbalance_ratio'].mean(),
            'max_imbalance': data['imbalance_ratio'].max(),
            'min_imbalance': data['imbalance_ratio'].min()
        }
    
    def _generate_recommendations(self, imbalance: ImbalanceType, strength: ImbalanceStrength, 
                                trend: str) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        if strength == ImbalanceStrength.NEUTRAL:
            recommendations.append("Balanced order flow - wait for clearer signals")
            return recommendations
        
        if imbalance in [ImbalanceType.BUY_IMBALANCE, ImbalanceType.EXTREME_BUY]:
            recommendations.append("Strong buying pressure detected")
            if strength in [ImbalanceStrength.STRONG, ImbalanceStrength.EXTREME]:
                recommendations.append("Consider long positions on pullbacks")
                recommendations.append("Watch for continuation of upward momentum")
        
        elif imbalance in [ImbalanceType.SELL_IMBALANCE, ImbalanceType.EXTREME_SELL]:
            recommendations.append("Strong selling pressure detected")
            if strength in [ImbalanceStrength.STRONG, ImbalanceStrength.EXTREME]:
                recommendations.append("Consider short positions on bounces")
                recommendations.append("Watch for continuation of downward momentum")
        
        # Trend-based recommendations
        if trend == "increasing":
            recommendations.append("Cumulative delta trending up - bullish bias")
        elif trend == "decreasing":
            recommendations.append("Cumulative delta trending down - bearish bias")
        
        return recommendations
>>>>>>> 5e659b3064c215382ffc9ef1f13510cbfdd547a7
