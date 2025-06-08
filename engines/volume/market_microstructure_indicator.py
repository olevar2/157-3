# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Market Microstructure Indicator - Order Flow Analysis
Platform3 - Humanitarian Trading System

The Market Microstructure Indicator analyzes order book data to reveal insights about
market liquidity, order flow imbalances, and institutional activity. By examining the 
microstructure of markets, it identifies hidden patterns in buying and selling pressure.

Key Features:
- Order book imbalance calculation
- Liquidity cluster detection
- Price impact estimation
- Bid-ask spread analysis
- Order flow momentum tracking
- Institutional footprint detection

Humanitarian Mission: Identify market microstructure inefficiencies to extract
better prices and improve trade execution, enhancing profit generation potential
for humanitarian causes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from engines.indicator_base import IndicatorSignal, TechnicalIndicator, ServiceError
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class MarketMicrostructureSignal(IndicatorSignal):
    """Market microstructure-specific signal with detailed order flow analysis"""
    order_flow_imbalance: float = 0.0
    price_impact: float = 0.0
    liquidity_score: float = 0.0
    bid_ask_spread: float = 0.0
    pressure_direction: str = "neutral"  # "buy", "sell", "neutral"
    institutional_activity_score: float = 0.0


class MarketMicrostructureIndicator(TechnicalIndicator):
    """
    MarketMicrostructureIndicator analyzes order book data to identify
    order flow imbalances and market microstructure patterns that reveal
    buying/selling pressure and institutional activity.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, # Added config
                 imbalance_threshold: float = 0.1, 
                 liquidity_depth: int = 5,
                 institutional_threshold: float = 5.0):
        """
        Initialize the MarketMicrostructureIndicator with configurable parameters
        
        Parameters:
        -----------
        config: Configuration dictionary.
        imbalance_threshold: float
            Threshold to determine significant order flow imbalance (0.0 - 1.0)
        liquidity_depth: int
            Number of price levels to analyze in the order book
        institutional_threshold: float
            Size multiplier to identify institutional orders (vs. retail)
        """
        super().__init__(config=config) # Pass config to super
        self.logger.info(f"MarketMicrostructureIndicator initialized with imbalance_threshold={imbalance_threshold}")
        self.imbalance_threshold = imbalance_threshold
        self.liquidity_depth = liquidity_depth
        self.institutional_threshold = institutional_threshold
    
    def calculate_order_flow_imbalance(self, order_book_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Calculate order flow imbalance from order book data
        
        Parameters:
        -----------
        order_book_data: Dict[str, List[Dict[str, Any]]]
            Order book data with 'bids' and 'asks' lists, each containing
            price level dictionaries with 'price' and 'volume' keys
            
        Returns:
        --------
        Dict[str, Any]
            Order flow imbalance metrics
        """
        try:
            # Validate input
            if not isinstance(order_book_data, dict) or 'bids' not in order_book_data or 'asks' not in order_book_data:
                raise ValueError("Invalid order book data format")
                
            bids = order_book_data['bids']
            asks = order_book_data['asks']
            
            if not bids or not asks:
                return {
                    'order_flow_imbalance': 0.0,
                    'pressure_direction': 'neutral',
                    'bid_ask_spread': 0.0,
                    'top_level_imbalance': 0.0
                }
                
            # Calculate total volume at bid and ask up to liquidity_depth
            bid_depth = min(self.liquidity_depth, len(bids))
            ask_depth = min(self.liquidity_depth, len(asks))
            
            bid_volume = sum(level['volume'] for level in bids[:bid_depth])
            ask_volume = sum(level['volume'] for level in asks[:ask_depth])
            
            # Handle edge case with zero volume
            if bid_volume + ask_volume <= 0:
                return {
                    'order_flow_imbalance': 0.0,
                    'pressure_direction': 'neutral',
                    'bid_ask_spread': 0.0,
                    'top_level_imbalance': 0.0
                }
            
            # Calculate imbalance (-1.0 to 1.0, positive means more bid volume)
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Determine pressure direction
            if imbalance > self.imbalance_threshold:
                pressure = "buy"
            elif imbalance < -self.imbalance_threshold:
                pressure = "sell"
            else:
                pressure = "neutral"
                
            # Calculate bid-ask spread
            if bids and asks:
                best_bid = bids[0]['price']
                best_ask = asks[0]['price']
                bid_ask_spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
            else:
                bid_ask_spread = 0.0
                
            # Calculate top level (best bid/ask) imbalance
            if bids and asks:
                top_bid_vol = bids[0]['volume']
                top_ask_vol = asks[0]['volume']
                top_level_imbalance = (top_bid_vol - top_ask_vol) / (top_bid_vol + top_ask_vol) \
                                    if (top_bid_vol + top_ask_vol) > 0 else 0.0
            else:
                top_level_imbalance = 0.0
            
            return {
                'order_flow_imbalance': imbalance,
                'pressure_direction': pressure,
                'bid_ask_spread': bid_ask_spread,
                'top_level_imbalance': top_level_imbalance
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating order flow imbalance: {str(e)}")
            return {
                'order_flow_imbalance': 0.0,
                'pressure_direction': 'neutral',
                'bid_ask_spread': 0.0,
                'top_level_imbalance': 0.0
            }
    
    def estimate_price_impact(self, order_book_data: Dict[str, List[Dict[str, Any]]],
                             target_volume: float) -> Dict[str, float]:
        """
        Estimate price impact of a trade of given size
        
        Parameters:
        -----------
        order_book_data: Dict[str, List[Dict[str, Any]]]
            Order book data
        target_volume: float
            Target volume to execute
            
        Returns:
        --------
        Dict[str, float]
            Price impact estimates for buy and sell
        """
        try:
            if not isinstance(order_book_data, dict) or 'bids' not in order_book_data or 'asks' not in order_book_data:
                return {'buy_impact': 0.0, 'sell_impact': 0.0}
                
            bids = order_book_data['bids']
            asks = order_book_data['asks']
            
            if not bids or not asks:
                return {'buy_impact': 0.0, 'sell_impact': 0.0}
                
            # Calculate price impact for buy order
            buy_volume_filled = 0.0
            buy_notional = 0.0
            buy_levels_used = 0
            
            for level in asks:
                level_price = level['price']
                level_volume = level['volume']
                
                volume_to_take = min(level_volume, target_volume - buy_volume_filled)
                buy_notional += volume_to_take * level_price
                buy_volume_filled += volume_to_take
                buy_levels_used += 1
                
                if buy_volume_filled >= target_volume:
                    break
                    
            # Calculate price impact for sell order
            sell_volume_filled = 0.0
            sell_notional = 0.0
            sell_levels_used = 0
            
            for level in bids:
                level_price = level['price']
                level_volume = level['volume']
                
                volume_to_take = min(level_volume, target_volume - sell_volume_filled)
                sell_notional += volume_to_take * level_price
                sell_volume_filled += volume_to_take
                sell_levels_used += 1
                
                if sell_volume_filled >= target_volume:
                    break
            
            # Calculate effective prices and impacts
            best_ask = asks[0]['price']
            best_bid = bids[0]['price']
            mid_price = (best_ask + best_bid) / 2
            
            buy_avg_price = buy_notional / buy_volume_filled if buy_volume_filled > 0 else best_ask
            sell_avg_price = sell_notional / sell_volume_filled if sell_volume_filled > 0 else best_bid
            
            buy_impact = (buy_avg_price - best_ask) / mid_price
            sell_impact = (best_bid - sell_avg_price) / mid_price
            
            return {
                'buy_impact': buy_impact,
                'sell_impact': sell_impact,
                'buy_levels_used': buy_levels_used,
                'sell_levels_used': sell_levels_used
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating price impact: {str(e)}")
            return {
                'buy_impact': 0.0,
                'sell_impact': 0.0,
                'buy_levels_used': 0,
                'sell_levels_used': 0
            }
    
    def calculate_liquidity_score(self, order_book_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        Calculate market liquidity score
        
        Parameters:
        -----------
        order_book_data: Dict[str, List[Dict[str, Any]]]
            Order book data
            
        Returns:
        --------
        float
            Liquidity score (0-1 scale, higher is more liquid)
        """
        try:
            if not isinstance(order_book_data, dict) or 'bids' not in order_book_data or 'asks' not in order_book_data:
                return 0.0
                
            bids = order_book_data['bids']
            asks = order_book_data['asks']
            
            if not bids or not asks:
                return 0.0
                
            # Extract prices and volumes
            bid_prices = [level['price'] for level in bids]
            ask_prices = [level['price'] for level in asks]
            bid_volumes = [level['volume'] for level in bids]
            ask_volumes = [level['volume'] for level in asks]
            
            # Calculate metrics
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid_price = (best_bid + best_ask) / 2
            spread = (best_ask - best_bid) / mid_price
            
            # Calculate volume within 0.5% of mid price
            threshold = mid_price * 0.005
            close_bid_volume = sum(vol for price, vol in zip(bid_prices, bid_volumes) 
                                 if mid_price - price <= threshold)
            close_ask_volume = sum(vol for price, vol in zip(ask_prices, ask_volumes) 
                                 if price - mid_price <= threshold)
            
            # Calculate decay factor (how quickly volume decays away from midpoint)
            if len(bid_prices) > 1 and len(ask_prices) > 1:
                bid_decay = (bid_volumes[0] - bid_volumes[-1]) / (bid_prices[-1] - bid_prices[0]) \
                          if (bid_prices[-1] - bid_prices[0]) != 0 else 0
                ask_decay = (ask_volumes[0] - ask_volumes[-1]) / (ask_prices[-1] - ask_prices[0]) \
                          if (ask_prices[-1] - ask_prices[0]) != 0 else 0
                decay_factor = (bid_decay + ask_decay) / 2
            else:
                decay_factor = 0
            
            # Normalize metrics
            norm_spread = max(0, 1 - (spread * 100))  # Lower spread = higher liquidity
            norm_volume = min(1, (close_bid_volume + close_ask_volume) / 1000)  # Normalize depth
            norm_decay = max(0, min(1, 1 - (decay_factor / 100)))  # Lower decay = higher liquidity
            
            # Combine into liquidity score (0-1)
            liquidity_score = (norm_spread * 0.4) + (norm_volume * 0.4) + (norm_decay * 0.2)
            return max(0.0, min(1.0, liquidity_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0.0
    
    def detect_institutional_activity(self, order_book_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        Detect signs of institutional activity in the order book
        
        Parameters:
        -----------
        order_book_data: Dict[str, List[Dict[str, Any]]]
            Order book data
            
        Returns:
        --------
        float
            Institutional activity score (0-1 scale)
        """
        try:
            if not isinstance(order_book_data, dict) or 'bids' not in order_book_data or 'asks' not in order_book_data:
                return 0.0
                
            bids = order_book_data['bids']
            asks = order_book_data['asks']
            
            if not bids or not asks:
                return 0.0
                
            # Calculate average order sizes
            avg_bid_size = np.mean([level['volume'] for level in bids]) if bids else 0
            avg_ask_size = np.mean([level['volume'] for level in asks]) if asks else 0
            avg_size = (avg_bid_size + avg_ask_size) / 2
            
            # Find large orders (potential institutional footprints)
            large_bid_levels = [level for level in bids if level['volume'] > avg_bid_size * self.institutional_threshold]
            large_ask_levels = [level for level in asks if level['volume'] > avg_ask_size * self.institutional_threshold]
            
            large_levels_count = len(large_bid_levels) + len(large_ask_levels)
            total_levels_count = len(bids) + len(asks)
            
            # Calculate clustering (institutional orders often cluster at specific levels)
            bid_diffs = [bids[i]['price'] - bids[i+1]['price'] for i in range(len(bids)-1)] if len(bids) > 1 else []
            ask_diffs = [asks[i+1]['price'] - asks[i]['price'] for i in range(len(asks)-1)] if len(asks) > 1 else []
            
            all_diffs = bid_diffs + ask_diffs
            if all_diffs:
                clustering_factor = np.std(all_diffs) / np.mean(all_diffs) if np.mean(all_diffs) > 0 else 0
                norm_clustering = max(0, min(1, 1 - clustering_factor))  # Lower spread = higher clustering
            else:
                norm_clustering = 0
                
            # Calculate large order proportion
            large_volume_ratio = 0
            if total_levels_count > 0:
                large_volume = sum(level['volume'] for level in large_bid_levels + large_ask_levels)
                total_volume = sum(level['volume'] for level in bids + asks)
                large_volume_ratio = large_volume / total_volume if total_volume > 0 else 0
                
            # Combine into institutional activity score (0-1)
            institutional_score = (large_volume_ratio * 0.6) + (norm_clustering * 0.4) 
            return max(0.0, min(1.0, institutional_score))
            
        except Exception as e:
            self.logger.error(f"Error detecting institutional activity: {str(e)}")
            return 0.0
    
    def calculate(self, data: Dict[str, Any]) -> MarketMicrostructureSignal:
        """
        Calculate market microstructure metrics from order book data
        
        Parameters:
        -----------
        data: Dict[str, Any]
            Market data containing 'order_book' field with order book data
            
        Returns:
        --------
        MarketMicrostructureSignal
            Comprehensive market microstructure analysis
        """
        try:
            # Extract order book data
            if not isinstance(data, dict) or 'order_book' not in data:
                raise ValueError("Missing order book data")
                
            order_book = data['order_book']
            
            # Optional estimate volume for price impact calculation
            estimate_volume = data.get('estimate_volume', 10.0)  # Default trade size to estimate impact
            
            # Calculate order flow imbalance
            imbalance_metrics = self.calculate_order_flow_imbalance(order_book)
            
            # Estimate price impact
            impact_metrics = self.estimate_price_impact(order_book, estimate_volume)
            
            # Calculate liquidity score
            liquidity = self.calculate_liquidity_score(order_book)
            
            # Detect institutional activity
            institutional_score = self.detect_institutional_activity(order_book)
            
            # Average price impact
            avg_impact = (impact_metrics['buy_impact'] + impact_metrics['sell_impact']) / 2
            
            # Create signal with buy/sell recommendation
            signal_direction = "neutral"
            signal_strength = 0.0
            
            # Generate signals based on combined metrics
            imbalance = imbalance_metrics['order_flow_imbalance']
            pressure = imbalance_metrics['pressure_direction']
            
            if pressure == "buy" and institutional_score > 0.5:
                signal_direction = "buy"
                signal_strength = min(1.0, (abs(imbalance) * 0.6 + institutional_score * 0.4))
            elif pressure == "sell" and institutional_score > 0.5:
                signal_direction = "sell"
                signal_strength = min(1.0, (abs(imbalance) * 0.6 + institutional_score * 0.4))
                
            # Create and return the microstructure signal
            return MarketMicrostructureSignal(
                order_flow_imbalance=imbalance_metrics['order_flow_imbalance'],
                price_impact=avg_impact,
                liquidity_score=liquidity,
                bid_ask_spread=imbalance_metrics['bid_ask_spread'],
                pressure_direction=imbalance_metrics['pressure_direction'],
                institutional_activity_score=institutional_score,
                signal=signal_direction,
                strength=signal_strength,
                timestamp=data.get('timestamp', datetime.now())
            )
        
        except Exception as e:
            self.logger.error(f"Error in MarketMicrostructureIndicator calculation: {str(e)}")
            raise ServiceError(f"Calculation failed: {str(e)}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on market microstructure analysis
        
        Parameters:
        -----------
        data: Dict[str, Any]
            Order book data for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Trading signal with direction, strength and analysis
        """
        signal = self.calculate(data)
        
        return {
            "direction": signal.signal,
            "strength": signal.strength,
            "timestamp": signal.timestamp,
            "metadata": {
                "order_flow_imbalance": signal.order_flow_imbalance,
                "price_impact": signal.price_impact,
                "liquidity": signal.liquidity_score,
                "bid_ask_spread": signal.bid_ask_spread,
                "pressure": signal.pressure_direction,
                "institutional_activity": signal.institutional_activity_score
            }
        }