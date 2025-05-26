"""
Advanced Position Sizing System
Intelligent position sizing algorithms for optimal risk-adjusted returns

Features:
- Kelly Criterion optimization
- Volatility-adjusted position sizing
- Risk parity allocation
- Maximum drawdown protection
- Dynamic position scaling
- Multi-timeframe risk assessment
- Session-based adjustments
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import redis
import json
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    FIXED_FRACTIONAL = "fixed_fractional"
    DYNAMIC_SCALING = "dynamic_scaling"
    SESSION_BASED = "session_based"

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class TradingSignal:
    symbol: str
    direction: str  # 'buy' or 'sell'
    confidence: float  # 0.0 to 1.0
    timeframe: str  # 'M1', 'M5', 'M15', 'H1', 'H4'
    expected_return: float
    stop_loss_distance: float
    take_profit_distance: float
    volatility: float
    session: str  # 'asian', 'london', 'ny', 'overlap'
    timestamp: datetime

@dataclass
class PositionSizeResult:
    recommended_size: float
    max_size: float
    min_size: float
    risk_amount: float
    expected_return: float
    risk_reward_ratio: float
    confidence_adjusted_size: float
    volatility_adjusted_size: float
    kelly_optimal_size: float
    sizing_method: SizingMethod
    reasoning: List[str]
    warnings: List[str]

@dataclass
class AccountMetrics:
    account_balance: float
    available_margin: float
    used_margin: float
    unrealized_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    open_positions: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

class AdvancedPositionSizer:
    """
    Advanced position sizing system with multiple algorithms
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Position sizing parameters
        self.sizing_params = {
            'max_risk_per_trade': 0.02,        # 2% max risk per trade
            'max_portfolio_risk': 0.10,        # 10% max portfolio risk
            'kelly_multiplier': 0.25,          # Kelly fraction multiplier
            'volatility_lookback': 20,         # Days for volatility calculation
            'min_position_size': 0.01,         # Minimum lot size
            'max_position_size': 10.0,         # Maximum lot size
            'confidence_threshold': 0.60,      # Minimum confidence for full size
            'session_multipliers': {           # Session-based size adjustments
                'asian': 0.8,
                'london': 1.2,
                'ny': 1.0,
                'overlap': 1.1
            }
        }
        
        # Risk level configurations
        self.risk_configs = {
            RiskLevel.CONSERVATIVE: {
                'max_risk_per_trade': 0.01,
                'kelly_multiplier': 0.15,
                'volatility_multiplier': 0.5
            },
            RiskLevel.MODERATE: {
                'max_risk_per_trade': 0.02,
                'kelly_multiplier': 0.25,
                'volatility_multiplier': 0.75
            },
            RiskLevel.AGGRESSIVE: {
                'max_risk_per_trade': 0.03,
                'kelly_multiplier': 0.35,
                'volatility_multiplier': 1.0
            },
            RiskLevel.MAXIMUM: {
                'max_risk_per_trade': 0.05,
                'kelly_multiplier': 0.50,
                'volatility_multiplier': 1.25
            }
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_calculations': 0,
            'average_calculation_time': 0.0,
            'sizing_method_usage': {},
            'risk_adjustments': 0,
            'warnings_generated': 0
        }
        
        logger.info("AdvancedPositionSizer initialized")

    async def calculate_position_size(self, signal: TradingSignal, account: AccountMetrics,
                                    risk_level: RiskLevel = RiskLevel.MODERATE,
                                    preferred_method: SizingMethod = SizingMethod.KELLY_CRITERION) -> PositionSizeResult:
        """
        Calculate optimal position size using multiple algorithms
        """
        start_time = datetime.now()
        
        try:
            # Get risk configuration
            risk_config = self.risk_configs[risk_level]
            
            # Calculate position sizes using different methods
            kelly_size = await self._calculate_kelly_size(signal, account, risk_config)
            volatility_size = await self._calculate_volatility_adjusted_size(signal, account, risk_config)
            risk_parity_size = await self._calculate_risk_parity_size(signal, account, risk_config)
            fixed_size = await self._calculate_fixed_fractional_size(signal, account, risk_config)
            
            # Apply confidence adjustments
            confidence_adjusted_size = await self._apply_confidence_adjustment(
                kelly_size, signal.confidence
            )
            
            # Apply session-based adjustments
            session_adjusted_size = await self._apply_session_adjustment(
                confidence_adjusted_size, signal.session
            )
            
            # Apply volatility adjustments
            volatility_adjusted_size = await self._apply_volatility_adjustment(
                session_adjusted_size, signal.volatility
            )
            
            # Determine recommended size based on preferred method
            size_options = {
                SizingMethod.KELLY_CRITERION: kelly_size,
                SizingMethod.VOLATILITY_ADJUSTED: volatility_size,
                SizingMethod.RISK_PARITY: risk_parity_size,
                SizingMethod.FIXED_FRACTIONAL: fixed_size,
                SizingMethod.DYNAMIC_SCALING: volatility_adjusted_size
            }
            
            recommended_size = size_options.get(preferred_method, kelly_size)
            
            # Apply final constraints
            final_size, warnings = await self._apply_size_constraints(
                recommended_size, signal, account, risk_config
            )
            
            # Calculate risk metrics
            risk_amount = final_size * signal.stop_loss_distance
            expected_return = final_size * signal.expected_return
            risk_reward_ratio = signal.take_profit_distance / signal.stop_loss_distance if signal.stop_loss_distance > 0 else 0
            
            # Generate reasoning
            reasoning = await self._generate_sizing_reasoning(
                signal, account, risk_config, preferred_method, final_size
            )
            
            result = PositionSizeResult(
                recommended_size=final_size,
                max_size=min(self.sizing_params['max_position_size'], 
                           account.available_margin / (signal.current_price * 100000)),
                min_size=self.sizing_params['min_position_size'],
                risk_amount=risk_amount,
                expected_return=expected_return,
                risk_reward_ratio=risk_reward_ratio,
                confidence_adjusted_size=confidence_adjusted_size,
                volatility_adjusted_size=volatility_adjusted_size,
                kelly_optimal_size=kelly_size,
                sizing_method=preferred_method,
                reasoning=reasoning,
                warnings=warnings
            )
            
            # Update performance stats
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['average_calculation_time'] = (
                (self.performance_stats['average_calculation_time'] * 
                 (self.performance_stats['total_calculations'] - 1) + calculation_time) /
                self.performance_stats['total_calculations']
            )
            
            method_key = preferred_method.value
            self.performance_stats['sizing_method_usage'][method_key] = (
                self.performance_stats['sizing_method_usage'].get(method_key, 0) + 1
            )
            
            if warnings:
                self.performance_stats['warnings_generated'] += len(warnings)
            
            logger.info(f"✅ Position size calculated: {final_size:.2f} lots for {signal.symbol}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return self._default_position_size(signal, account)

    async def _calculate_kelly_size(self, signal: TradingSignal, account: AccountMetrics, 
                                  risk_config: Dict) -> float:
        """Calculate position size using Kelly Criterion"""
        try:
            # Kelly formula: f = (bp - q) / b
            # where f = fraction to bet, b = odds, p = win probability, q = loss probability
            
            win_rate = max(account.win_rate, 0.01)  # Avoid division by zero
            avg_win = max(account.avg_win, 0.01)
            avg_loss = max(abs(account.avg_loss), 0.01)
            
            # Calculate odds (reward/risk ratio)
            odds = avg_win / avg_loss
            
            # Kelly fraction
            kelly_fraction = (odds * win_rate - (1 - win_rate)) / odds
            
            # Apply Kelly multiplier for safety
            kelly_multiplier = risk_config['kelly_multiplier']
            adjusted_kelly = max(0, kelly_fraction * kelly_multiplier)
            
            # Convert to position size
            risk_amount = account.account_balance * adjusted_kelly
            position_size = risk_amount / signal.stop_loss_distance if signal.stop_loss_distance > 0 else 0
            
            return min(position_size, self.sizing_params['max_position_size'])
            
        except Exception as e:
            logger.error(f"Error in Kelly calculation: {e}")
            return self.sizing_params['min_position_size']

    async def _calculate_volatility_adjusted_size(self, signal: TradingSignal, account: AccountMetrics,
                                                risk_config: Dict) -> float:
        """Calculate position size adjusted for volatility"""
        try:
            # Base risk amount
            base_risk = account.account_balance * risk_config['max_risk_per_trade']
            
            # Volatility adjustment
            avg_volatility = 0.01  # Assume 1% average daily volatility
            volatility_ratio = avg_volatility / max(signal.volatility, 0.001)
            volatility_multiplier = risk_config['volatility_multiplier']
            
            # Adjust size based on volatility
            adjusted_risk = base_risk * volatility_ratio * volatility_multiplier
            position_size = adjusted_risk / signal.stop_loss_distance if signal.stop_loss_distance > 0 else 0
            
            return min(position_size, self.sizing_params['max_position_size'])
            
        except Exception as e:
            logger.error(f"Error in volatility adjustment: {e}")
            return self.sizing_params['min_position_size']

    async def _calculate_risk_parity_size(self, signal: TradingSignal, account: AccountMetrics,
                                        risk_config: Dict) -> float:
        """Calculate position size using risk parity approach"""
        try:
            # Equal risk allocation across positions
            max_positions = 10  # Assume max 10 concurrent positions
            risk_per_position = account.account_balance * risk_config['max_risk_per_trade'] / max_positions
            
            position_size = risk_per_position / signal.stop_loss_distance if signal.stop_loss_distance > 0 else 0
            
            return min(position_size, self.sizing_params['max_position_size'])
            
        except Exception as e:
            logger.error(f"Error in risk parity calculation: {e}")
            return self.sizing_params['min_position_size']

    async def _calculate_fixed_fractional_size(self, signal: TradingSignal, account: AccountMetrics,
                                             risk_config: Dict) -> float:
        """Calculate position size using fixed fractional method"""
        try:
            risk_amount = account.account_balance * risk_config['max_risk_per_trade']
            position_size = risk_amount / signal.stop_loss_distance if signal.stop_loss_distance > 0 else 0
            
            return min(position_size, self.sizing_params['max_position_size'])
            
        except Exception as e:
            logger.error(f"Error in fixed fractional calculation: {e}")
            return self.sizing_params['min_position_size']

    async def _apply_confidence_adjustment(self, base_size: float, confidence: float) -> float:
        """Apply confidence-based size adjustment"""
        confidence_threshold = self.sizing_params['confidence_threshold']
        
        if confidence < confidence_threshold:
            # Reduce size for low confidence signals
            confidence_multiplier = confidence / confidence_threshold
            return base_size * confidence_multiplier
        
        return base_size

    async def _apply_session_adjustment(self, base_size: float, session: str) -> float:
        """Apply session-based size adjustment"""
        session_multiplier = self.sizing_params['session_multipliers'].get(session, 1.0)
        return base_size * session_multiplier

    async def _apply_volatility_adjustment(self, base_size: float, volatility: float) -> float:
        """Apply volatility-based size adjustment"""
        avg_volatility = 0.01  # 1% average daily volatility
        
        if volatility > avg_volatility * 2:
            # Reduce size for high volatility
            return base_size * 0.7
        elif volatility < avg_volatility * 0.5:
            # Increase size for low volatility
            return base_size * 1.2
        
        return base_size

    async def _apply_size_constraints(self, size: float, signal: TradingSignal, 
                                    account: AccountMetrics, risk_config: Dict) -> Tuple[float, List[str]]:
        """Apply final size constraints and generate warnings"""
        warnings = []
        
        # Minimum size constraint
        if size < self.sizing_params['min_position_size']:
            size = self.sizing_params['min_position_size']
            warnings.append(f"Position size increased to minimum: {size}")
        
        # Maximum size constraint
        if size > self.sizing_params['max_position_size']:
            size = self.sizing_params['max_position_size']
            warnings.append(f"Position size reduced to maximum: {size}")
        
        # Margin constraint
        required_margin = size * signal.current_price * 100000 * 0.01  # Assuming 1% margin
        if required_margin > account.available_margin:
            size = account.available_margin / (signal.current_price * 100000 * 0.01)
            warnings.append(f"Position size reduced due to margin constraint: {size}")
        
        # Risk constraint
        risk_amount = size * signal.stop_loss_distance
        max_risk = account.account_balance * risk_config['max_risk_per_trade']
        if risk_amount > max_risk:
            size = max_risk / signal.stop_loss_distance
            warnings.append(f"Position size reduced due to risk constraint: {size}")
        
        # Drawdown constraint
        if account.current_drawdown > 0.05:  # 5% drawdown
            size *= 0.5  # Reduce size by 50%
            warnings.append("Position size reduced due to current drawdown")
        
        return max(size, self.sizing_params['min_position_size']), warnings

    async def _generate_sizing_reasoning(self, signal: TradingSignal, account: AccountMetrics,
                                       risk_config: Dict, method: SizingMethod, final_size: float) -> List[str]:
        """Generate reasoning for position sizing decision"""
        reasoning = []
        
        reasoning.append(f"Using {method.value} sizing method")
        reasoning.append(f"Signal confidence: {signal.confidence:.2%}")
        reasoning.append(f"Risk per trade: {risk_config['max_risk_per_trade']:.2%}")
        reasoning.append(f"Stop loss distance: {signal.stop_loss_distance:.5f}")
        reasoning.append(f"Session: {signal.session}")
        reasoning.append(f"Volatility: {signal.volatility:.4f}")
        reasoning.append(f"Final position size: {final_size:.2f} lots")
        
        return reasoning

    def _default_position_size(self, signal: TradingSignal, account: AccountMetrics) -> PositionSizeResult:
        """Return default position size in case of errors"""
        default_size = self.sizing_params['min_position_size']
        
        return PositionSizeResult(
            recommended_size=default_size,
            max_size=self.sizing_params['max_position_size'],
            min_size=self.sizing_params['min_position_size'],
            risk_amount=default_size * signal.stop_loss_distance,
            expected_return=default_size * signal.expected_return,
            risk_reward_ratio=1.0,
            confidence_adjusted_size=default_size,
            volatility_adjusted_size=default_size,
            kelly_optimal_size=default_size,
            sizing_method=SizingMethod.FIXED_FRACTIONAL,
            reasoning=["Default sizing due to calculation error"],
            warnings=["Using default minimum position size"]
        )

    async def get_performance_stats(self) -> Dict:
        """Get position sizing performance statistics"""
        return {
            **self.performance_stats,
            'sizing_parameters': self.sizing_params,
            'risk_configurations': {k.value: v for k, v in self.risk_configs.items()}
        }

    async def update_sizing_parameters(self, new_params: Dict):
        """Update position sizing parameters"""
        self.sizing_params.update(new_params)
        logger.info("Position sizing parameters updated")

    async def optimize_kelly_multiplier(self, historical_performance: List[Dict]) -> float:
        """Optimize Kelly multiplier based on historical performance"""
        try:
            # Analyze historical performance to optimize Kelly multiplier
            returns = [trade['return'] for trade in historical_performance]
            
            if len(returns) < 30:
                return self.sizing_params['kelly_multiplier']
            
            # Calculate optimal Kelly multiplier
            win_rate = len([r for r in returns if r > 0]) / len(returns)
            avg_win = statistics.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
            avg_loss = abs(statistics.mean([r for r in returns if r < 0])) if any(r < 0 for r in returns) else 1
            
            if avg_loss > 0:
                odds = avg_win / avg_loss
                kelly_fraction = (odds * win_rate - (1 - win_rate)) / odds
                optimal_multiplier = min(max(kelly_fraction * 0.25, 0.1), 0.5)
                
                logger.info(f"Optimized Kelly multiplier: {optimal_multiplier:.3f}")
                return optimal_multiplier
            
            return self.sizing_params['kelly_multiplier']
            
        except Exception as e:
            logger.error(f"Error optimizing Kelly multiplier: {e}")
            return self.sizing_params['kelly_multiplier']
