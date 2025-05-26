"""
Smart Money Indicators for Institutional Flow Detection
Analyzes market data to identify institutional trading activity and smart money flow.

This module implements indicators to detect:
- Institutional accumulation/distribution patterns
- Smart money vs retail money activity
- Market maker behavior
- Large order flow and block trades
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

class SmartMoneyActivity(Enum):
    """Smart money activity types"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    MARKUP = "markup"
    MARKDOWN = "markdown"
    MANIPULATION = "manipulation"
    ABSORPTION = "absorption"
    NEUTRAL = "neutral"

class InstitutionalBehavior(Enum):
    """Institutional behavior patterns"""
    AGGRESSIVE_BUYING = "aggressive_buying"
    AGGRESSIVE_SELLING = "aggressive_selling"
    STEALTH_ACCUMULATION = "stealth_accumulation"
    STEALTH_DISTRIBUTION = "stealth_distribution"
    MARKET_MAKING = "market_making"
    STOP_HUNTING = "stop_hunting"
    PASSIVE = "passive"

class FlowStrength(Enum):
    """Smart money flow strength"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"

@dataclass
class SmartMoneySignal:
    """Individual smart money signal"""
    timestamp: datetime
    activity_type: SmartMoneyActivity
    institutional_behavior: InstitutionalBehavior
    flow_strength: FlowStrength
    confidence: float
    volume_ratio: float
    price_efficiency: float
    order_flow_imbalance: float
    market_impact: float

@dataclass
class InstitutionalFootprint:
    """Institutional trading footprint analysis"""
    session: str
    total_institutional_volume: float
    institutional_percentage: float
    avg_trade_size: float
    large_trade_count: int
    stealth_activity_score: float
    manipulation_score: float
    efficiency_ratio: float

@dataclass
class SmartMoneyAnalysisResult:
    """Complete smart money analysis result"""
    symbol: str
    timeframe: str
    analysis_time: datetime
    signals: List[SmartMoneySignal]
    current_activity: SmartMoneyActivity
    institutional_footprint: InstitutionalFootprint
    flow_direction: str
    flow_strength: FlowStrength
    smart_money_index: float
    retail_sentiment: str
    key_insights: List[str]
    trading_implications: List[str]
    recommendations: List[str]

class SmartMoneyIndicators:
    """
    Smart Money Indicators for detecting institutional activity.
    
    Analyzes:
    - Volume patterns and anomalies
    - Price efficiency and market impact
    - Order flow characteristics
    - Stealth trading patterns
    - Market manipulation signals
    """
    
    def __init__(self, lookback_periods: int = 50, large_trade_threshold: float = 2.0):
        """
        Initialize smart money analyzer.
        
        Args:
            lookback_periods: Number of periods for analysis
            large_trade_threshold: Multiplier for identifying large trades (vs average volume)
        """
        self.lookback_periods = lookback_periods
        self.large_trade_threshold = large_trade_threshold
        
    def analyze_smart_money(self, data: pd.DataFrame, symbol: str, timeframe: str) -> SmartMoneyAnalysisResult:
        """
        Perform complete smart money analysis.
        
        Args:
            data: OHLCV data with timestamp column
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            SmartMoneyAnalysisResult with institutional activity analysis
        """
        try:
            # Validate input data
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            
            # Calculate smart money indicators
            data = self._calculate_smart_money_metrics(data)
            
            # Analyze individual signals
            signals = []
            for i in range(len(data)):
                if i >= 10:  # Need some history for analysis
                    signal = self._analyze_smart_money_signal(data.iloc[i], data.iloc[max(0, i-20):i+1])
                    signals.append(signal)
            
            # Determine current activity
            current_activity = self._determine_current_activity(signals[-10:] if len(signals) >= 10 else signals)
            
            # Calculate institutional footprint
            institutional_footprint = self._calculate_institutional_footprint(data.tail(50))
            
            # Analyze flow characteristics
            flow_direction = self._analyze_flow_direction(signals[-20:] if len(signals) >= 20 else signals)
            flow_strength = self._calculate_flow_strength(signals[-10:] if len(signals) >= 10 else signals)
            
            # Calculate smart money index
            smart_money_index = self._calculate_smart_money_index(data.tail(30))
            
            # Determine retail sentiment
            retail_sentiment = self._analyze_retail_sentiment(signals[-15:] if len(signals) >= 15 else signals)
            
            # Generate insights and recommendations
            key_insights = self._generate_key_insights(current_activity, institutional_footprint, smart_money_index)
            trading_implications = self._analyze_trading_implications(current_activity, flow_direction, flow_strength)
            recommendations = self._generate_recommendations(current_activity, flow_strength, smart_money_index)
            
            return SmartMoneyAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                analysis_time=datetime.now(),
                signals=signals,
                current_activity=current_activity,
                institutional_footprint=institutional_footprint,
                flow_direction=flow_direction,
                flow_strength=flow_strength,
                smart_money_index=smart_money_index,
                retail_sentiment=retail_sentiment,
                key_insights=key_insights,
                trading_implications=trading_implications,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Smart money analysis failed for {symbol}: {e}")
            raise
    
    def _calculate_smart_money_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate smart money detection metrics"""
        data = data.copy()
        
        # Calculate basic metrics
        data['price_change'] = data['close'] - data['open']
        data['price_range'] = data['high'] - data['low']
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / (data['volume_ma'] + 1e-10)
        
        # Price efficiency (how much price moved relative to volume)
        data['price_efficiency'] = abs(data['price_change']) / (data['volume'] + 1e-10) * 1000000
        
        # Market impact (price change per unit volume)
        data['market_impact'] = abs(data['price_change']) / (data['volume_ratio'] + 1e-10)
        
        # Order flow estimation
        data['close_position'] = (data['close'] - data['low']) / (data['price_range'] + 1e-10)
        data['buy_pressure'] = data['volume'] * data['close_position']
        data['sell_pressure'] = data['volume'] * (1 - data['close_position'])
        data['order_flow_imbalance'] = (data['buy_pressure'] - data['sell_pressure']) / (data['volume'] + 1e-10)
        
        # Smart money volume detection
        data['large_volume'] = data['volume'] > (data['volume_ma'] * self.large_trade_threshold)
        data['stealth_volume'] = (data['volume'] < data['volume_ma'] * 0.8) & (abs(data['price_change']) > data['price_range'] * 0.3)
        
        # Manipulation detection
        data['price_volatility'] = data['price_range'].rolling(10).std()
        data['volume_volatility'] = data['volume'].rolling(10).std()
        data['manipulation_score'] = (data['price_volatility'] / (data['volume_volatility'] + 1e-10)) * data['volume_ratio']
        
        # Absorption patterns (high volume, low price movement)
        data['absorption'] = (data['volume_ratio'] > 1.5) & (abs(data['price_change']) < data['price_range'] * 0.3)
        
        return data
    
    def _analyze_smart_money_signal(self, current_bar: pd.Series, context_data: pd.DataFrame) -> SmartMoneySignal:
        """Analyze individual bar for smart money signals"""
        
        # Determine activity type
        activity_type = self._classify_activity_type(current_bar, context_data)
        
        # Determine institutional behavior
        institutional_behavior = self._classify_institutional_behavior(current_bar, context_data)
        
        # Calculate flow strength
        flow_strength = self._classify_flow_strength(current_bar)
        
        # Calculate confidence
        confidence = self._calculate_signal_confidence(current_bar, activity_type, institutional_behavior)
        
        return SmartMoneySignal(
            timestamp=current_bar['timestamp'],
            activity_type=activity_type,
            institutional_behavior=institutional_behavior,
            flow_strength=flow_strength,
            confidence=confidence,
            volume_ratio=current_bar['volume_ratio'],
            price_efficiency=current_bar['price_efficiency'],
            order_flow_imbalance=current_bar['order_flow_imbalance'],
            market_impact=current_bar['market_impact']
        )
    
    def _classify_activity_type(self, bar: pd.Series, context: pd.DataFrame) -> SmartMoneyActivity:
        """Classify smart money activity type"""
        
        # High volume with controlled price movement = accumulation/distribution
        if bar['volume_ratio'] > 1.5 and abs(bar['price_change']) < bar['price_range'] * 0.4:
            if bar['order_flow_imbalance'] > 0.2:
                return SmartMoneyActivity.ACCUMULATION
            elif bar['order_flow_imbalance'] < -0.2:
                return SmartMoneyActivity.DISTRIBUTION
            else:
                return SmartMoneyActivity.ABSORPTION
        
        # High volume with significant price movement = markup/markdown
        elif bar['volume_ratio'] > 1.2 and abs(bar['price_change']) > bar['price_range'] * 0.6:
            if bar['price_change'] > 0:
                return SmartMoneyActivity.MARKUP
            else:
                return SmartMoneyActivity.MARKDOWN
        
        # High manipulation score
        elif bar['manipulation_score'] > context['manipulation_score'].quantile(0.8):
            return SmartMoneyActivity.MANIPULATION
        
        else:
            return SmartMoneyActivity.NEUTRAL
    
    def _classify_institutional_behavior(self, bar: pd.Series, context: pd.DataFrame) -> InstitutionalBehavior:
        """Classify institutional behavior pattern"""
        
        # Aggressive patterns (high volume, immediate price impact)
        if bar['volume_ratio'] > 2.0 and bar['market_impact'] > context['market_impact'].quantile(0.8):
            if bar['order_flow_imbalance'] > 0.3:
                return InstitutionalBehavior.AGGRESSIVE_BUYING
            elif bar['order_flow_imbalance'] < -0.3:
                return InstitutionalBehavior.AGGRESSIVE_SELLING
        
        # Stealth patterns (low volume, significant price movement)
        elif bar['stealth_volume'] and abs(bar['price_change']) > bar['price_range'] * 0.4:
            if bar['order_flow_imbalance'] > 0.1:
                return InstitutionalBehavior.STEALTH_ACCUMULATION
            elif bar['order_flow_imbalance'] < -0.1:
                return InstitutionalBehavior.STEALTH_DISTRIBUTION
        
        # Market making (balanced flow, tight spreads)
        elif abs(bar['order_flow_imbalance']) < 0.1 and bar['price_range'] < context['price_range'].quantile(0.3):
            return InstitutionalBehavior.MARKET_MAKING
        
        # Stop hunting (manipulation patterns)
        elif bar['manipulation_score'] > context['manipulation_score'].quantile(0.9):
            return InstitutionalBehavior.STOP_HUNTING
        
        else:
            return InstitutionalBehavior.PASSIVE
    
    def _classify_flow_strength(self, bar: pd.Series) -> FlowStrength:
        """Classify flow strength based on volume and price metrics"""
        strength_score = (bar['volume_ratio'] * 0.4 + 
                         abs(bar['order_flow_imbalance']) * 0.3 + 
                         bar['market_impact'] * 0.3)
        
        if strength_score > 3.0:
            return FlowStrength.VERY_STRONG
        elif strength_score > 2.0:
            return FlowStrength.STRONG
        elif strength_score > 1.0:
            return FlowStrength.MODERATE
        elif strength_score > 0.5:
            return FlowStrength.WEAK
        else:
            return FlowStrength.NEUTRAL
    
    def _calculate_signal_confidence(self, bar: pd.Series, activity: SmartMoneyActivity, 
                                   behavior: InstitutionalBehavior) -> float:
        """Calculate confidence in smart money signal"""
        confidence = 0.5  # Base confidence
        
        # Volume-based confidence
        if bar['volume_ratio'] > 2.0:
            confidence += 0.2
        elif bar['volume_ratio'] > 1.5:
            confidence += 0.1
        
        # Order flow confidence
        if abs(bar['order_flow_imbalance']) > 0.3:
            confidence += 0.15
        elif abs(bar['order_flow_imbalance']) > 0.2:
            confidence += 0.1
        
        # Activity type confidence
        if activity in [SmartMoneyActivity.ACCUMULATION, SmartMoneyActivity.DISTRIBUTION]:
            confidence += 0.1
        
        # Behavior confidence
        if behavior in [InstitutionalBehavior.AGGRESSIVE_BUYING, InstitutionalBehavior.AGGRESSIVE_SELLING]:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _determine_current_activity(self, recent_signals: List[SmartMoneySignal]) -> SmartMoneyActivity:
        """Determine current smart money activity"""
        if not recent_signals:
            return SmartMoneyActivity.NEUTRAL
        
        # Weight recent signals more heavily
        activity_scores = {}
        total_weight = 0
        
        for i, signal in enumerate(recent_signals):
            weight = (i + 1) * signal.confidence  # More recent and confident signals get higher weight
            activity_scores[signal.activity_type] = activity_scores.get(signal.activity_type, 0) + weight
            total_weight += weight
        
        if not activity_scores:
            return SmartMoneyActivity.NEUTRAL
        
        return max(activity_scores, key=activity_scores.get)
    
    def _calculate_institutional_footprint(self, data: pd.DataFrame) -> InstitutionalFootprint:
        """Calculate institutional trading footprint"""
        if len(data) == 0:
            return InstitutionalFootprint("unknown", 0, 0, 0, 0, 0, 0, 0)
        
        # Estimate institutional volume (large trades + stealth trades)
        institutional_volume = data[data['large_volume'] | data['stealth_volume']]['volume'].sum()
        total_volume = data['volume'].sum()
        institutional_percentage = (institutional_volume / total_volume * 100) if total_volume > 0 else 0
        
        # Average trade size
        avg_trade_size = data['volume'].mean()
        
        # Large trade count
        large_trade_count = data['large_volume'].sum()
        
        # Stealth activity score
        stealth_activity_score = data['stealth_volume'].mean() * 100
        
        # Manipulation score
        manipulation_score = data['manipulation_score'].mean()
        
        # Efficiency ratio (price movement per unit volume)
        efficiency_ratio = data['price_efficiency'].mean()
        
        return InstitutionalFootprint(
            session="current",
            total_institutional_volume=institutional_volume,
            institutional_percentage=institutional_percentage,
            avg_trade_size=avg_trade_size,
            large_trade_count=large_trade_count,
            stealth_activity_score=stealth_activity_score,
            manipulation_score=manipulation_score,
            efficiency_ratio=efficiency_ratio
        )
    
    def _analyze_flow_direction(self, signals: List[SmartMoneySignal]) -> str:
        """Analyze overall flow direction"""
        if not signals:
            return "neutral"
        
        bullish_signals = sum(1 for signal in signals 
                            if signal.activity_type in [SmartMoneyActivity.ACCUMULATION, SmartMoneyActivity.MARKUP])
        bearish_signals = sum(1 for signal in signals 
                            if signal.activity_type in [SmartMoneyActivity.DISTRIBUTION, SmartMoneyActivity.MARKDOWN])
        
        if bullish_signals > bearish_signals + 2:
            return "bullish"
        elif bearish_signals > bullish_signals + 2:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_flow_strength(self, signals: List[SmartMoneySignal]) -> FlowStrength:
        """Calculate overall flow strength"""
        if not signals:
            return FlowStrength.NEUTRAL
        
        avg_strength_score = np.mean([
            4 if signal.flow_strength == FlowStrength.VERY_STRONG else
            3 if signal.flow_strength == FlowStrength.STRONG else
            2 if signal.flow_strength == FlowStrength.MODERATE else
            1 if signal.flow_strength == FlowStrength.WEAK else 0
            for signal in signals
        ])
        
        if avg_strength_score >= 3.5:
            return FlowStrength.VERY_STRONG
        elif avg_strength_score >= 2.5:
            return FlowStrength.STRONG
        elif avg_strength_score >= 1.5:
            return FlowStrength.MODERATE
        elif avg_strength_score >= 0.5:
            return FlowStrength.WEAK
        else:
            return FlowStrength.NEUTRAL
    
    def _calculate_smart_money_index(self, data: pd.DataFrame) -> float:
        """Calculate overall smart money index (0-100)"""
        if len(data) == 0:
            return 50.0
        
        # Combine multiple factors
        institutional_factor = min(data['large_volume'].mean() * 100, 30)
        stealth_factor = min(data['stealth_volume'].mean() * 100, 25)
        efficiency_factor = min(data['price_efficiency'].mean() / 10, 25)
        manipulation_factor = min(data['manipulation_score'].mean() / 5, 20)
        
        smart_money_index = institutional_factor + stealth_factor + efficiency_factor + manipulation_factor
        return min(smart_money_index, 100.0)
    
    def _analyze_retail_sentiment(self, signals: List[SmartMoneySignal]) -> str:
        """Analyze retail sentiment (often opposite to smart money)"""
        if not signals:
            return "neutral"
        
        smart_money_bullish = sum(1 for signal in signals 
                                if signal.activity_type in [SmartMoneyActivity.ACCUMULATION, SmartMoneyActivity.MARKUP])
        smart_money_bearish = sum(1 for signal in signals 
                                if signal.activity_type in [SmartMoneyActivity.DISTRIBUTION, SmartMoneyActivity.MARKDOWN])
        
        # Retail often trades opposite to smart money
        if smart_money_bullish > smart_money_bearish + 1:
            return "bearish"  # Retail likely bearish when smart money is bullish
        elif smart_money_bearish > smart_money_bullish + 1:
            return "bullish"  # Retail likely bullish when smart money is bearish
        else:
            return "neutral"
    
    def _generate_key_insights(self, activity: SmartMoneyActivity, footprint: InstitutionalFootprint, 
                             index: float) -> List[str]:
        """Generate key insights from smart money analysis"""
        insights = []
        
        insights.append(f"Smart Money Index: {index:.1f}/100")
        insights.append(f"Current Activity: {activity.value}")
        insights.append(f"Institutional Volume: {footprint.institutional_percentage:.1f}%")
        
        if footprint.stealth_activity_score > 20:
            insights.append("High stealth activity detected - institutions trading quietly")
        
        if footprint.manipulation_score > 2:
            insights.append("Potential market manipulation patterns identified")
        
        if footprint.large_trade_count > 5:
            insights.append("Multiple large trades detected - institutional presence")
        
        return insights
    
    def _analyze_trading_implications(self, activity: SmartMoneyActivity, direction: str, 
                                   strength: FlowStrength) -> List[str]:
        """Analyze trading implications"""
        implications = []
        
        if activity == SmartMoneyActivity.ACCUMULATION:
            implications.append("Smart money accumulating - potential upward pressure")
            implications.append("Look for breakout opportunities above resistance")
        elif activity == SmartMoneyActivity.DISTRIBUTION:
            implications.append("Smart money distributing - potential downward pressure")
            implications.append("Look for breakdown opportunities below support")
        elif activity == SmartMoneyActivity.MANIPULATION:
            implications.append("Market manipulation detected - be cautious of false signals")
            implications.append("Wait for confirmation before entering positions")
        
        if strength in [FlowStrength.STRONG, FlowStrength.VERY_STRONG]:
            implications.append("Strong institutional flow - high probability moves")
        
        return implications
    
    def _generate_recommendations(self, activity: SmartMoneyActivity, strength: FlowStrength, 
                                index: float) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        if index > 70:
            recommendations.append("High smart money activity - follow institutional flow")
        elif index < 30:
            recommendations.append("Low institutional activity - be cautious of retail-driven moves")
        
        if activity == SmartMoneyActivity.ACCUMULATION and strength != FlowStrength.WEAK:
            recommendations.append("Consider long positions on pullbacks")
        elif activity == SmartMoneyActivity.DISTRIBUTION and strength != FlowStrength.WEAK:
            recommendations.append("Consider short positions on bounces")
        elif activity == SmartMoneyActivity.MANIPULATION:
            recommendations.append("Avoid trading until manipulation phase ends")
        
        if strength == FlowStrength.NEUTRAL:
            recommendations.append("Weak signals - wait for stronger confirmation")
        
        return recommendations
