"""
Market Microstructure Analysis for Platform3
Advanced analysis of market microstructure patterns and dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MicrostructurePattern(Enum):
    """Types of microstructure patterns"""
    HIGH_FREQUENCY_REVERSAL = "HIGH_FREQUENCY_REVERSAL"
    ORDER_FLOW_IMBALANCE = "ORDER_FLOW_IMBALANCE"
    PRICE_IMPACT_DECAY = "PRICE_IMPACT_DECAY"
    VOLATILITY_CLUSTERING = "VOLATILITY_CLUSTERING"
    LIQUIDITY_SHORTAGE = "LIQUIDITY_SHORTAGE"
    MARKET_MAKER_ACTIVITY = "MARKET_MAKER_ACTIVITY"


@dataclass
class MicrostructureSignal:
    """Market microstructure signal"""
    pattern_type: MicrostructurePattern
    intensity: float  # 0.0 to 1.0
    duration: timedelta
    timestamp: datetime
    market_impact: float
    liquidity_score: float
    metadata: Dict[str, Any]


class MarketMicrostructureAnalysis:
    """
    Advanced Market Microstructure Analysis
    
    Analyzes high-frequency market patterns:
    - Order flow analysis
    - Price impact modeling
    - Liquidity dynamics
    - Market maker detection
    - Volatility clustering
    """
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.pattern_history = []
        self.order_flow_buffer = []
        self.volatility_buffer = []
        
        logger.info("MarketMicrostructureAnalysis initialized")
    
    def analyze_microstructure(
        self, 
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame = None,
        order_book: Dict[str, Any] = None
    ) -> List[MicrostructureSignal]:
        """
        Analyze market microstructure patterns
        
        Args:
            price_data: High-frequency price data
            volume_data: Volume data
            order_book: Order book snapshot
            
        Returns:
            List of detected microstructure signals
        """
        signals = []
        
        try:
            # Analyze different microstructure patterns
            signals.extend(self._detect_hf_reversals(price_data))
            signals.extend(self._detect_order_flow_imbalance(price_data, volume_data))
            signals.extend(self._detect_price_impact_decay(price_data))
            signals.extend(self._detect_volatility_clustering(price_data))
            
            if order_book:
                signals.extend(self._detect_liquidity_patterns(order_book))
                signals.extend(self._detect_market_maker_activity(order_book))
            
            # Update history
            self.pattern_history.extend(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in microstructure analysis: {e}")
            return []
    
    def _detect_hf_reversals(self, price_data: pd.DataFrame) -> List[MicrostructureSignal]:
        """Detect high-frequency price reversals"""
        signals = []
        
        try:
            if len(price_data) < 10:
                return signals
            
            # Calculate short-term price changes
            prices = price_data['close'].values[-20:]
            returns = np.diff(prices) / prices[:-1]
            
            # Detect reversals using threshold
            reversal_threshold = 2 * np.std(returns)
            
            for i in range(2, len(returns)):
                if (abs(returns[i]) > reversal_threshold and 
                    np.sign(returns[i]) != np.sign(returns[i-1])):
                    
                    intensity = min(abs(returns[i]) / reversal_threshold, 1.0)
                    
                    signal = MicrostructureSignal(
                        pattern_type=MicrostructurePattern.HIGH_FREQUENCY_REVERSAL,
                        intensity=intensity,
                        duration=timedelta(minutes=1),
                        timestamp=datetime.now(),
                        market_impact=intensity * 0.5,
                        liquidity_score=1.0 - intensity,
                        metadata={
                            "return_magnitude": abs(returns[i]),
                            "threshold": reversal_threshold,
                            "previous_return": returns[i-1]
                        }
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error detecting HF reversals: {e}")
        
        return signals
    
    def _detect_order_flow_imbalance(
        self, 
        price_data: pd.DataFrame, 
        volume_data: pd.DataFrame = None
    ) -> List[MicrostructureSignal]:
        """Detect order flow imbalances"""
        signals = []
        
        try:
            if volume_data is None or len(price_data) < 5:
                return signals
            
            # Calculate volume-weighted price changes
            prices = price_data['close'].values[-10:]
            volumes = volume_data['volume'].values[-10:]
            
            # Compute order flow imbalance proxy
            price_changes = np.diff(prices)
            volume_changes = volumes[1:]
            
            # Calculate imbalance score
            imbalance_score = np.corrcoef(price_changes, volume_changes)[0, 1]
            
            if abs(imbalance_score) > 0.5:  # Significant correlation
                intensity = min(abs(imbalance_score), 1.0)
                
                signal = MicrostructureSignal(
                    pattern_type=MicrostructurePattern.ORDER_FLOW_IMBALANCE,
                    intensity=intensity,
                    duration=timedelta(minutes=5),
                    timestamp=datetime.now(),
                    market_impact=intensity * 0.7,
                    liquidity_score=0.5,
                    metadata={
                        "correlation": imbalance_score,
                        "avg_volume": np.mean(volumes),
                        "price_volatility": np.std(price_changes)
                    }
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error detecting order flow imbalance: {e}")
        
        return signals
    
    def _detect_price_impact_decay(self, price_data: pd.DataFrame) -> List[MicrostructureSignal]:
        """Detect price impact decay patterns"""
        signals = []
        
        try:
            if len(price_data) < 15:
                return signals
            
            prices = price_data['close'].values[-15:]
            
            # Look for impact-decay pattern
            for i in range(5, len(prices) - 5):
                # Check for initial impact
                initial_move = abs(prices[i] - prices[i-1]) / prices[i-1]
                
                if initial_move > 0.001:  # Significant move
                    # Check for decay
                    subsequent_prices = prices[i+1:i+5]
                    decay_rate = self._calculate_decay_rate(prices[i], subsequent_prices)
                    
                    if decay_rate > 0.3:  # Significant decay
                        intensity = min(decay_rate, 1.0)
                        
                        signal = MicrostructureSignal(
                            pattern_type=MicrostructurePattern.PRICE_IMPACT_DECAY,
                            intensity=intensity,
                            duration=timedelta(minutes=3),
                            timestamp=datetime.now(),
                            market_impact=0.3,
                            liquidity_score=decay_rate,
                            metadata={
                                "initial_impact": initial_move,
                                "decay_rate": decay_rate,
                                "recovery_time": len(subsequent_prices)
                            }
                        )
                        signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error detecting price impact decay: {e}")
        
        return signals
    
    def _calculate_decay_rate(self, impact_price: float, subsequent_prices: np.ndarray) -> float:
        """Calculate the rate of price impact decay"""
        try:
            if len(subsequent_prices) == 0:
                return 0.0
            
            # Measure how much price reverts toward pre-impact level
            final_price = subsequent_prices[-1]
            decay_ratio = abs(impact_price - final_price) / impact_price
            
            return min(decay_ratio * 10, 1.0)  # Scale to 0-1
            
        except Exception:
            return 0.0
    
    def _detect_volatility_clustering(self, price_data: pd.DataFrame) -> List[MicrostructureSignal]:
        """Detect volatility clustering patterns"""
        signals = []
        
        try:
            if len(price_data) < 20:
                return signals
            
            prices = price_data['close'].values[-20:]
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate rolling volatility
            window_size = 5
            volatilities = []
            
            for i in range(window_size, len(returns)):
                vol = np.std(returns[i-window_size:i])
                volatilities.append(vol)
            
            if len(volatilities) > 5:
                # Detect clustering using autocorrelation
                volatilities = np.array(volatilities)
                autocorr = np.corrcoef(volatilities[:-1], volatilities[1:])[0, 1]
                
                if autocorr > 0.4:  # Significant clustering
                    intensity = min(autocorr, 1.0)
                    
                    signal = MicrostructureSignal(
                        pattern_type=MicrostructurePattern.VOLATILITY_CLUSTERING,
                        intensity=intensity,
                        duration=timedelta(minutes=10),
                        timestamp=datetime.now(),
                        market_impact=intensity * 0.4,
                        liquidity_score=0.6,
                        metadata={
                            "autocorrelation": autocorr,
                            "avg_volatility": np.mean(volatilities),
                            "vol_std": np.std(volatilities)
                        }
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error detecting volatility clustering: {e}")
        
        return signals
    
    def _detect_liquidity_patterns(self, order_book: Dict[str, Any]) -> List[MicrostructureSignal]:
        """Detect liquidity shortage patterns"""
        signals = []
        
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return signals
            
            # Calculate bid-ask spread
            best_bid = max(bids, key=lambda x: x[0])[0] if bids else 0
            best_ask = min(asks, key=lambda x: x[0])[0] if asks else 0
            
            if best_bid > 0 and best_ask > 0:
                spread = (best_ask - best_bid) / best_bid
                
                # Calculate order book depth
                total_bid_volume = sum(bid[1] for bid in bids[:5])
                total_ask_volume = sum(ask[1] for ask in asks[:5])
                
                # Detect liquidity shortage
                if spread > 0.002 or min(total_bid_volume, total_ask_volume) < 1000:
                    intensity = min(spread * 500, 1.0)  # Scale spread to intensity
                    
                    signal = MicrostructureSignal(
                        pattern_type=MicrostructurePattern.LIQUIDITY_SHORTAGE,
                        intensity=intensity,
                        duration=timedelta(minutes=2),
                        timestamp=datetime.now(),
                        market_impact=intensity * 0.8,
                        liquidity_score=1.0 - intensity,
                        metadata={
                            "spread": spread,
                            "bid_volume": total_bid_volume,
                            "ask_volume": total_ask_volume,
                            "best_bid": best_bid,
                            "best_ask": best_ask
                        }
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error detecting liquidity patterns: {e}")
        
        return signals
    
    def _detect_market_maker_activity(self, order_book: Dict[str, Any]) -> List[MicrostructureSignal]:
        """Detect market maker activity patterns"""
        signals = []
        
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if len(bids) < 3 or len(asks) < 3:
                return signals
            
            # Analyze order size distribution
            bid_sizes = [bid[1] for bid in bids[:10]]
            ask_sizes = [ask[1] for ask in asks[:10]]
            
            # Look for uniform large orders (market maker signature)
            bid_uniformity = 1.0 - (np.std(bid_sizes) / np.mean(bid_sizes)) if np.mean(bid_sizes) > 0 else 0
            ask_uniformity = 1.0 - (np.std(ask_sizes) / np.mean(ask_sizes)) if np.mean(ask_sizes) > 0 else 0
            
            avg_uniformity = (bid_uniformity + ask_uniformity) / 2
            
            if avg_uniformity > 0.7:  # High uniformity suggests market maker
                intensity = min(avg_uniformity, 1.0)
                
                signal = MicrostructureSignal(
                    pattern_type=MicrostructurePattern.MARKET_MAKER_ACTIVITY,
                    intensity=intensity,
                    duration=timedelta(minutes=5),
                    timestamp=datetime.now(),
                    market_impact=0.2,  # Market makers typically reduce impact
                    liquidity_score=intensity,  # High uniformity = better liquidity
                    metadata={
                        "bid_uniformity": bid_uniformity,
                        "ask_uniformity": ask_uniformity,
                        "avg_bid_size": np.mean(bid_sizes),
                        "avg_ask_size": np.mean(ask_sizes)
                    }
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error detecting market maker activity: {e}")
        
        return signals
    
    def get_microstructure_summary(self) -> Dict[str, Any]:
        """Get summary of recent microstructure patterns"""
        if not self.pattern_history:
            return {"total_patterns": 0}
        
        recent_patterns = [
            p for p in self.pattern_history 
            if (datetime.now() - p.timestamp) < timedelta(hours=1)
        ]
        
        pattern_counts = {}
        for pattern in recent_patterns:
            pattern_type = pattern.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            "total_patterns": len(recent_patterns),
            "pattern_distribution": pattern_counts,
            "avg_intensity": np.mean([p.intensity for p in recent_patterns]),
            "avg_market_impact": np.mean([p.market_impact for p in recent_patterns]),
            "avg_liquidity_score": np.mean([p.liquidity_score for p in recent_patterns])
        }


# Global instance
market_microstructure_analysis = MarketMicrostructureAnalysis()
