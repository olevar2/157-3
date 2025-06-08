# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Order Flow Sequence Analyzer - Trade Sequence Pattern Recognition
Platform3 - Humanitarian Trading System

The Order Flow Sequence Analyzer identifies significant patterns in sequential trading activity
to detect potential reversals, continuations, or institutional trading footprints before they
become visible in price action.

Key Features:
- Trade sequence pattern recognition
- Aggressor analysis (buy vs sell initiated trades)
- Order flow acceleration/deceleration detection
- Sequential absorption analysis
- Tape reading automation
- Market regime classification

Humanitarian Mission: Extract predictive insights from trade sequencing to anticipate
price movements before they occur, enabling the capture of early entry points for
maximized humanitarian profit generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from engines.indicator_base import IndicatorSignal, TechnicalIndicator, ServiceError
import logging
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class OrderFlowSequenceSignal(IndicatorSignal):
    """Order flow sequence analysis signal with pattern recognition"""
    aggressor_ratio: float = 0.0  # -1.0 to 1.0, positive means buy dominance
    sequence_momentum: float = 0.0  # -1.0 to 1.0, momentum of recent trades
    absorption_detected: bool = False
    absorption_side: str = "none"  # "buy", "sell", "none"
    detected_patterns: List[Dict[str, Any]] = field(default_factory=list)
    regime_classification: str = "neutral"  # "trending", "reversal", "neutral", "choppy"


class OrderFlowSequenceAnalyzer(TechnicalIndicator):
    """
    OrderFlowSequenceAnalyzer examines sequences of trades to identify patterns,
    aggressors, absorption, and market regime characteristics from order flow
    """
    
    def __init__(self, sequence_window: int = 100, 
                pattern_recognition_threshold: float = 0.75,
                momentum_lookback: int = 20):
        """
        Initialize the OrderFlowSequenceAnalyzer with configurable parameters
        
        Parameters:
        -----------
        sequence_window: int
            Number of trades to keep in analysis window
        pattern_recognition_threshold: float
            Confidence threshold for pattern recognition (0-1)
        momentum_lookback: int
            Number of trades for momentum calculation
        """
        super().__init__()
        self.logger.info(f"OrderFlowSequenceAnalyzer initialized with sequence_window={sequence_window}")
        self.sequence_window = sequence_window
        self.pattern_threshold = pattern_recognition_threshold
        self.momentum_lookback = momentum_lookback
        self.trade_sequence = deque(maxlen=sequence_window)
        
        # Define common sequence patterns
        self.patterns = {
            "absorption": {
                "description": "Large orders absorbed without price movement",
                "significance": "Indicates hidden support/resistance level"
            },
            "climax": {
                "description": "Accelerating sequence of same-direction trades",
                "significance": "Potential exhaustion or capitulation"
            },
            "reversal": {
                "description": "Sudden shift from one aggressor to another",
                "significance": "Early indication of trend change"
            },
            "stepping": {
                "description": "Sequential trades at incrementally higher/lower prices",
                "significance": "Strong directional pressure"
            },
            "stalling": {
                "description": "Sequence loses momentum without price follow-through",
                "significance": "Potential trend fatigue"
            }
        }
    
    def analyze_aggressors(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze buy/sell aggressors in trade sequence
        
        Parameters:
        -----------
        trades: List[Dict[str, Any]]
            List of trades with aggressor information
            
        Returns:
        --------
        Dict[str, Any]
            Aggressor analysis metrics
        """
        try:
            if not trades:
                return {
                    'buy_count': 0,
                    'sell_count': 0,
                    'neutral_count': 0,
                    'aggressor_ratio': 0.0,
                    'recent_shift': 0.0
                }
                
            # Count aggressors
            buy_count = sum(1 for trade in trades if trade.get('aggressor', '').lower() == 'buy')
            sell_count = sum(1 for trade in trades if trade.get('aggressor', '').lower() == 'sell')
            neutral_count = len(trades) - buy_count - sell_count
            
            # Calculate aggressor ratio (-1 to 1, positive means buy dominance)
            total_directional = buy_count + sell_count
            aggressor_ratio = 0.0
            
            if total_directional > 0:
                aggressor_ratio = (buy_count - sell_count) / total_directional
                
            # Calculate recent shift (comparing most recent third vs first third)
            recent_shift = 0.0
            
            if len(trades) >= 6:
                first_third = trades[:len(trades)//3]
                last_third = trades[-len(trades)//3:]
                
                first_buy = sum(1 for trade in first_third if trade.get('aggressor', '').lower() == 'buy')
                first_sell = sum(1 for trade in first_third if trade.get('aggressor', '').lower() == 'sell')
                
                last_buy = sum(1 for trade in last_third if trade.get('aggressor', '').lower() == 'buy')
                last_sell = sum(1 for trade in last_third if trade.get('aggressor', '').lower() == 'sell')
                
                first_ratio = (first_buy - first_sell) / len(first_third) if len(first_third) > 0 else 0
                last_ratio = (last_buy - last_sell) / len(last_third) if len(last_third) > 0 else 0
                
                recent_shift = last_ratio - first_ratio
            
            return {
                'buy_count': buy_count,
                'sell_count': sell_count,
                'neutral_count': neutral_count,
                'aggressor_ratio': aggressor_ratio,
                'recent_shift': recent_shift
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing aggressors: {str(e)}")
            return {
                'buy_count': 0,
                'sell_count': 0,
                'neutral_count': 0,
                'aggressor_ratio': 0.0,
                'recent_shift': 0.0
            }
    
    def calculate_sequence_momentum(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate momentum of recent trades in sequence
        
        Parameters:
        -----------
        trades: List[Dict[str, Any]]
            List of recent trades
            
        Returns:
        --------
        float
            Sequence momentum (-1.0 to 1.0)
        """
        try:
            if len(trades) < self.momentum_lookback:
                return 0.0
                
            # Use most recent trades for calculation
            recent_trades = trades[-self.momentum_lookback:]
            
            # Calculate weighted price movement
            price_movements = []
            weights = []
            
            for i in range(1, len(recent_trades)):
                curr_price = recent_trades[i].get('price', 0)
                prev_price = recent_trades[i-1].get('price', 0)
                
                if curr_price > 0 and prev_price > 0:
                    # Calculate price movement direction
                    price_movement = (curr_price - prev_price) / prev_price
                    # More recent trades get higher weights
                    weight = (i / len(recent_trades)) ** 2
                    
                    price_movements.append(price_movement)
                    weights.append(weight)
            
            # Calculate weighted average of price movements
            if price_movements and weights:
                weighted_momentum = sum(m * w for m, w in zip(price_movements, weights)) / sum(weights)
                
                # Normalize to [-1, 1] range with sigmoid-like function
                normalized_momentum = max(-1.0, min(1.0, weighted_momentum * 100))
                return normalized_momentum
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating sequence momentum: {str(e)}")
            return 0.0
    
    def detect_absorption(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect absorption patterns where large trades are absorbed without price movement
        
        Parameters:
        -----------
        trades: List[Dict[str, Any]]
            List of trades
            
        Returns:
        --------
        Dict[str, Any]
            Absorption detection results
        """
        try:
            if len(trades) < 10:
                return {'detected': False, 'side': 'none', 'confidence': 0.0}
                
            # Calculate average trade size
            avg_size = sum(trade.get('volume', 0) for trade in trades) / len(trades)
            
            # Look for large trades that don't move price significantly
            absorption_candidates = []
            
            for i in range(len(trades) - 5):
                # Check for large trade
                if trades[i].get('volume', 0) > avg_size * 2:
                    large_trade = trades[i]
                    trade_price = large_trade.get('price', 0)
                    next_trades = trades[i+1:i+6]  # Look at next 5 trades
                    
                    # Calculate price range after large trade
                    prices = [t.get('price', 0) for t in next_trades]
                    max_price = max(prices) if prices else trade_price
                    min_price = min(prices) if prices else trade_price
                    price_range = (max_price - min_price) / trade_price if trade_price > 0 else 0
                    
                    # If price barely moved despite large trade
                    if price_range < 0.001:  # Less than 0.1% price movement
                        # Determine side of absorption
                        side = large_trade.get('aggressor', '').lower()
                        
                        # Calculate confidence based on trade size and price stability
                        confidence = min(1.0, (large_trade.get('volume', 0) / avg_size) * (1 - price_range * 1000))
                        
                        absorption_candidates.append({
                            'index': i,
                            'trade': large_trade,
                            'side': side,
                            'confidence': confidence
                        })
            
            # If no candidates found
            if not absorption_candidates:
                return {'detected': False, 'side': 'none', 'confidence': 0.0}
                
            # Get most confident candidate
            best_candidate = max(absorption_candidates, key=lambda x: x['confidence'])
            
            return {
                'detected': best_candidate['confidence'] >= self.pattern_threshold,
                'side': best_candidate['side'],
                'confidence': best_candidate['confidence'],
                'trade_volume': best_candidate['trade'].get('volume', 0),
                'trade_price': best_candidate['trade'].get('price', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting absorption: {str(e)}")
            return {'detected': False, 'side': 'none', 'confidence': 0.0}
    
    def recognize_patterns(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Recognize various order flow patterns in trade sequence
        
        Parameters:
        -----------
        trades: List[Dict[str, Any]]
            List of trades
            
        Returns:
        --------
        List[Dict[str, Any]]
            Recognized patterns
        """
        try:
            recognized_patterns = []
            
            if len(trades) < 20:
                return recognized_patterns
                
            # Detect climax pattern (accelerating same-direction trades)
            # Count consecutive trades in same direction
            max_consecutive = 0
            current_consecutive = 1
            current_direction = None
            
            for i in range(1, len(trades)):
                direction = trades[i].get('aggressor', '')
                prev_direction = trades[i-1].get('aggressor', '')
                
                if direction and direction == prev_direction:
                    current_consecutive += 1
                    current_direction = direction
                else:
                    max_consecutive = max(max_consecutive, current_consecutive)
                    current_consecutive = 1
                    
            # Final update
            max_consecutive = max(max_consecutive, current_consecutive)
            
            # If we have a significant run of same-direction trades
            if max_consecutive >= 8:
                recognized_patterns.append({
                    'type': 'climax',
                    'direction': current_direction,
                    'consecutive_count': max_consecutive,
                    'confidence': min(1.0, max_consecutive / 15),
                    'implications': 'Potential exhaustion or capitulation'
                })
            
            # Detect stepping pattern (sequential trades at incrementally higher/lower prices)
            steps_up = 0
            steps_down = 0
            max_steps_up = 0
            max_steps_down = 0
            
            for i in range(1, len(trades)):
                curr_price = trades[i].get('price', 0)
                prev_price = trades[i-1].get('price', 0)
                
                if curr_price > prev_price:
                    steps_up += 1
                    steps_down = 0
                    max_steps_up = max(max_steps_up, steps_up)
                elif curr_price < prev_price:
                    steps_down += 1
                    steps_up = 0
                    max_steps_down = max(max_steps_down, steps_down)
            
            # If we have significant stepping pattern
            if max_steps_up >= 5:
                recognized_patterns.append({
                    'type': 'stepping',
                    'direction': 'up',
                    'step_count': max_steps_up,
                    'confidence': min(1.0, max_steps_up / 10),
                    'implications': 'Strong upward pressure'
                })
            
            if max_steps_down >= 5:
                recognized_patterns.append({
                    'type': 'stepping',
                    'direction': 'down',
                    'step_count': max_steps_down,
                    'confidence': min(1.0, max_steps_down / 10),
                    'implications': 'Strong downward pressure'
                })
            
            # Detect reversal pattern (sudden shift in aggressor)
            buy_aggregated = [0] * (len(trades) // 5)
            sell_aggregated = [0] * (len(trades) // 5)
            
            for i, trade in enumerate(trades):
                bucket = i // 5
                if bucket < len(buy_aggregated):
                    if trade.get('aggressor', '').lower() == 'buy':
                        buy_aggregated[bucket] += 1
                    elif trade.get('aggressor', '').lower() == 'sell':
                        sell_aggregated[bucket] += 1
            
            # Check for dominant buyers turning to sellers or vice versa
            reversal_detected = False
            reversal_confidence = 0.0
            reversal_direction = ''
            
            for i in range(2, len(buy_aggregated)):
                # Buy to sell reversal
                if (buy_aggregated[i-2] > sell_aggregated[i-2] * 2 and
                    buy_aggregated[i-1] > sell_aggregated[i-1] and
                    sell_aggregated[i] > buy_aggregated[i] * 1.5):
                    reversal_detected = True
                    reversal_confidence = min(1.0, sell_aggregated[i] / (buy_aggregated[i] + 1))
                    reversal_direction = 'buy_to_sell'
                    break
                    
                # Sell to buy reversal
                if (sell_aggregated[i-2] > buy_aggregated[i-2] * 2 and
                    sell_aggregated[i-1] > buy_aggregated[i-1] and
                    buy_aggregated[i] > sell_aggregated[i] * 1.5):
                    reversal_detected = True
                    reversal_confidence = min(1.0, buy_aggregated[i] / (sell_aggregated[i] + 1))
                    reversal_direction = 'sell_to_buy'
                    break
            
            if reversal_detected:
                recognized_patterns.append({
                    'type': 'reversal',
                    'direction': reversal_direction,
                    'confidence': reversal_confidence,
                    'implications': 'Early indication of trend change'
                })
                
            # Only include patterns that meet threshold confidence
            return [p for p in recognized_patterns if p['confidence'] >= self.pattern_threshold]
            
        except Exception as e:
            self.logger.error(f"Error recognizing patterns: {str(e)}")
            return []
    
    def classify_market_regime(self, trades: List[Dict[str, Any]], 
                              aggressor_info: Dict[str, Any],
                              momentum: float) -> str:
        """
        Classify market regime based on order flow characteristics
        
        Parameters:
        -----------
        trades: List[Dict[str, Any]]
            List of trades
        aggressor_info: Dict[str, Any]
            Aggressor analysis results
        momentum: float
            Sequence momentum value
            
        Returns:
        --------
        str
            Market regime classification
        """
        try:
            if len(trades) < 20:
                return "neutral"
                
            # Calculate price volatility
            prices = [trade.get('price', 0) for trade in trades if trade.get('price', 0) > 0]
            if not prices:
                return "neutral"
                
            price_range = max(prices) - min(prices)
            avg_price = sum(prices) / len(prices)
            volatility = price_range / avg_price if avg_price > 0 else 0
            
            # Calculate trade direction consistency
            consistency = abs(aggressor_info['aggressor_ratio'])
            
            # Determine regime from combination of factors
            
            # High momentum and high consistency = trending
            if abs(momentum) > 0.7 and consistency > 0.6:
                return "trending"
                
            # Low momentum but significant shift in aggressors = potential reversal
            if abs(momentum) < 0.3 and abs(aggressor_info['recent_shift']) > 0.5:
                return "reversal"
                
            # High volatility but low consistency = choppy
            if volatility > 0.005 and consistency < 0.3:
                return "choppy"
            
            # Default
            return "neutral"
            
        except Exception as e:
            self.logger.error(f"Error classifying market regime: {str(e)}")
            return "neutral"
    
    def calculate(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> OrderFlowSequenceSignal:
        """
        Calculate order flow sequence metrics from trade data
        
        Parameters:
        -----------
        data: Union[List[Dict[str, Any]], Dict[str, Any]]
            List of trades or dict with 'trades' key
            
        Returns:
        --------
        OrderFlowSequenceSignal
            Comprehensive order flow sequence analysis
        """
        try:
            # Extract trade data
            if isinstance(data, dict) and 'trades' in data:
                trades = data['trades']
            else:
                trades = data
                
            if not trades or not isinstance(trades, list):
                raise ValueError("Invalid trade data format")
                
            # Update internal trade sequence
            for trade in trades:
                self.trade_sequence.append(trade)
                
            current_sequence = list(self.trade_sequence)
            
            if len(current_sequence) < 10:
                raise ValueError(f"Insufficient trade data: need at least 10 trades")
                
            # Analyze aggressors
            aggressor_info = self.analyze_aggressors(current_sequence)
            
            # Calculate sequence momentum
            momentum = self.calculate_sequence_momentum(current_sequence)
            
            # Detect absorption
            absorption_info = self.detect_absorption(current_sequence)
            
            # Recognize patterns
            patterns = self.recognize_patterns(current_sequence)
            
            # Classify market regime
            regime = self.classify_market_regime(current_sequence, aggressor_info, momentum)
            
            # Generate signal based on analysis
            signal_direction = "neutral"
            signal_strength = 0.0
            
            # Generate buy signal if:
            # 1. Strong buy aggressor ratio, or
            # 2. Detected buy absorption pattern, or
            # 3. Detected sell-to-buy reversal pattern
            buy_signal_strength = 0.0
            
            if aggressor_info['aggressor_ratio'] > 0.5:
                buy_signal_strength = max(buy_signal_strength, aggressor_info['aggressor_ratio'] - 0.3)
                
            if absorption_info['detected'] and absorption_info['side'] == 'buy':
                buy_signal_strength = max(buy_signal_strength, absorption_info['confidence'] - 0.2)
                
            for pattern in patterns:
                if pattern['type'] == 'reversal' and pattern['direction'] == 'sell_to_buy':
                    buy_signal_strength = max(buy_signal_strength, pattern['confidence'] - 0.2)
            
            # Generate sell signal using similar logic
            sell_signal_strength = 0.0
            
            if aggressor_info['aggressor_ratio'] < -0.5:
                sell_signal_strength = max(sell_signal_strength, abs(aggressor_info['aggressor_ratio']) - 0.3)
                
            if absorption_info['detected'] and absorption_info['side'] == 'sell':
                sell_signal_strength = max(sell_signal_strength, absorption_info['confidence'] - 0.2)
                
            for pattern in patterns:
                if pattern['type'] == 'reversal' and pattern['direction'] == 'buy_to_sell':
                    sell_signal_strength = max(sell_signal_strength, pattern['confidence'] - 0.2)
            
            # Determine final signal
            if buy_signal_strength > 0.3 and buy_signal_strength > sell_signal_strength:
                signal_direction = "buy"
                signal_strength = min(1.0, buy_signal_strength)
            elif sell_signal_strength > 0.3 and sell_signal_strength > buy_signal_strength:
                signal_direction = "sell"
                signal_strength = min(1.0, sell_signal_strength)
                
            # Create timestamp from most recent trade
            timestamp = None
            if current_sequence and 'timestamp' in current_sequence[-1]:
                timestamp = current_sequence[-1]['timestamp']
            
            # Create and return the order flow sequence signal
            return OrderFlowSequenceSignal(
                aggressor_ratio=aggressor_info['aggressor_ratio'],
                sequence_momentum=momentum,
                absorption_detected=absorption_info['detected'],
                absorption_side=absorption_info['side'],
                detected_patterns=patterns,
                regime_classification=regime,
                signal=signal_direction,
                strength=signal_strength,
                timestamp=timestamp
            )
        
        except Exception as e:
            self.logger.error(f"Error in OrderFlowSequenceAnalyzer calculation: {str(e)}")
            raise ServiceError(f"Calculation failed: {str(e)}")
    
    def generate_signal(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate trading signals based on order flow sequence analysis
        
        Parameters:
        -----------
        data: Union[List[Dict[str, Any]], Dict[str, Any]]
            Trade data for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Trading signal with direction, strength and analysis
        """
        signal = self.calculate(data)
        
        patterns_info = [
            {
                "type": pattern["type"],
                "direction": pattern.get("direction", "neutral"),
                "confidence": pattern["confidence"]
            } for pattern in signal.detected_patterns
        ]
        
        return {
            "direction": signal.signal,
            "strength": signal.strength,
            "timestamp": signal.timestamp,
            "metadata": {
                "aggressor_ratio": signal.aggressor_ratio,
                "momentum": signal.sequence_momentum,
                "absorption_detected": signal.absorption_detected,
                "absorption_side": signal.absorption_side,
                "patterns": patterns_info,
                "market_regime": signal.regime_classification
            }
        }