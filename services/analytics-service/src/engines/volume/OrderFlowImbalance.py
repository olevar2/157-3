"""
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
