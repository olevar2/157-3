"""
Dynamic Stop-Loss & Take-Profit Mechanism
Advanced risk management with adaptive levels based on market conditions

Features:
- Dynamic stop-loss adjustment based on volatility
- Trailing stop-loss with multiple algorithms
- Take-profit scaling and partial position management
- Market condition-aware level adjustments
- Real-time risk monitoring and alerts
- Integration with technical indicators
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import redis
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StopLossType(Enum):
    """Stop-loss types supported by the system"""
    FIXED = "fixed"
    TRAILING = "trailing"
    VOLATILITY_BASED = "volatility_based"
    ATR_BASED = "atr_based"
    SUPPORT_RESISTANCE = "support_resistance"
    FIBONACCI = "fibonacci"

class TakeProfitType(Enum):
    """Take-profit types supported by the system"""
    FIXED = "fixed"
    SCALING = "scaling"
    RISK_REWARD = "risk_reward"
    FIBONACCI = "fibonacci"
    RESISTANCE_BASED = "resistance_based"

@dataclass
class Position:
    """Trading position data structure"""
    position_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    quantity: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_risk_per_trade: float = 0.02  # 2% of account
    max_daily_risk: float = 0.06  # 6% of account daily
    stop_loss_type: StopLossType = StopLossType.ATR_BASED
    take_profit_type: TakeProfitType = TakeProfitType.RISK_REWARD
    risk_reward_ratio: float = 2.0
    trailing_stop_distance: float = 0.001  # 10 pips for forex
    volatility_multiplier: float = 2.0
    atr_period: int = 14
    fibonacci_levels: List[float] = field(default_factory=lambda: [0.382, 0.618, 1.0, 1.618])

@dataclass
class MarketCondition:
    """Current market condition assessment"""
    volatility: float
    trend_strength: float
    support_level: float
    resistance_level: float
    atr_value: float
    session: str  # 'asian', 'london', 'ny', 'overlap'
    timestamp: datetime

class DynamicLevelManager:
    """
    Advanced dynamic stop-loss and take-profit management system
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.positions: Dict[str, Position] = {}
        self.risk_params = RiskParameters()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        # Performance tracking
        self.performance_stats = {
            'total_adjustments': 0,
            'successful_stops': 0,
            'successful_profits': 0,
            'average_adjustment_time': 0.0,
            'risk_violations': 0
        }
        
        logger.info("DynamicLevelManager initialized")

    async def start(self):
        """Start the dynamic level management system"""
        self.running = True
        logger.info("Starting Dynamic Level Manager...")
        
        # Start background tasks
        asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._update_market_conditions())
        asyncio.create_task(self._performance_monitoring())
        
        logger.info("✅ Dynamic Level Manager started successfully")

    async def stop(self):
        """Stop the dynamic level management system"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Dynamic Level Manager stopped")

    async def add_position(self, position: Position) -> bool:
        """
        Add a new position to risk management
        
        Args:
            position: Position object to manage
            
        Returns:
            bool: Success status
        """
        try:
            # Calculate initial stop-loss and take-profit
            await self._calculate_initial_levels(position)
            
            # Store position
            self.positions[position.position_id] = position
            
            # Cache in Redis for persistence
            await self._cache_position(position)
            
            logger.info(f"✅ Added position {position.position_id} to risk management")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add position {position.position_id}: {e}")
            return False

    async def remove_position(self, position_id: str) -> bool:
        """
        Remove position from risk management
        
        Args:
            position_id: ID of position to remove
            
        Returns:
            bool: Success status
        """
        try:
            if position_id in self.positions:
                del self.positions[position_id]
                
                # Remove from Redis cache
                self.redis_client.delete(f"position:{position_id}")
                
                logger.info(f"✅ Removed position {position_id} from risk management")
                return True
            else:
                logger.warning(f"Position {position_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to remove position {position_id}: {e}")
            return False

    async def update_position_price(self, position_id: str, current_price: float) -> bool:
        """
        Update position with current market price and adjust levels
        
        Args:
            position_id: ID of position to update
            current_price: Current market price
            
        Returns:
            bool: Success status
        """
        try:
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} not found for price update")
                return False
            
            position = self.positions[position_id]
            position.current_price = current_price
            
            # Calculate unrealized P&L
            if position.side == 'buy':
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            
            # Adjust dynamic levels
            await self._adjust_dynamic_levels(position)
            
            # Update cache
            await self._cache_position(position)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update position {position_id} price: {e}")
            return False

    async def _calculate_initial_levels(self, position: Position):
        """Calculate initial stop-loss and take-profit levels"""
        try:
            # Get market conditions
            market_condition = await self._get_market_condition(position.symbol)
            
            # Calculate stop-loss based on type
            if self.risk_params.stop_loss_type == StopLossType.ATR_BASED:
                stop_distance = market_condition.atr_value * self.risk_params.volatility_multiplier
            elif self.risk_params.stop_loss_type == StopLossType.VOLATILITY_BASED:
                stop_distance = market_condition.volatility * self.risk_params.volatility_multiplier
            elif self.risk_params.stop_loss_type == StopLossType.SUPPORT_RESISTANCE:
                if position.side == 'buy':
                    stop_distance = position.entry_price - market_condition.support_level
                else:
                    stop_distance = market_condition.resistance_level - position.entry_price
            else:  # FIXED
                stop_distance = self.risk_params.trailing_stop_distance
            
            # Set stop-loss
            if position.side == 'buy':
                position.stop_loss = position.entry_price - stop_distance
            else:
                position.stop_loss = position.entry_price + stop_distance
            
            # Calculate take-profit based on type
            if self.risk_params.take_profit_type == TakeProfitType.RISK_REWARD:
                profit_distance = stop_distance * self.risk_params.risk_reward_ratio
            elif self.risk_params.take_profit_type == TakeProfitType.FIBONACCI:
                profit_distance = stop_distance * self.risk_params.fibonacci_levels[1]  # 0.618
            elif self.risk_params.take_profit_type == TakeProfitType.RESISTANCE_BASED:
                if position.side == 'buy':
                    profit_distance = market_condition.resistance_level - position.entry_price
                else:
                    profit_distance = position.entry_price - market_condition.support_level
            else:  # FIXED
                profit_distance = stop_distance * 2.0
            
            # Set take-profit
            if position.side == 'buy':
                position.take_profit = position.entry_price + profit_distance
            else:
                position.take_profit = position.entry_price - profit_distance
            
            logger.info(f"Initial levels set for {position.position_id}: SL={position.stop_loss:.5f}, TP={position.take_profit:.5f}")
            
        except Exception as e:
            logger.error(f"Failed to calculate initial levels for {position.position_id}: {e}")

    async def _adjust_dynamic_levels(self, position: Position):
        """Adjust stop-loss and take-profit levels dynamically"""
        try:
            start_time = datetime.now()
            
            # Get current market conditions
            market_condition = await self._get_market_condition(position.symbol)
            
            # Adjust trailing stop-loss
            if self.risk_params.stop_loss_type == StopLossType.TRAILING:
                await self._adjust_trailing_stop(position, market_condition)
            
            # Adjust volatility-based levels
            elif self.risk_params.stop_loss_type == StopLossType.VOLATILITY_BASED:
                await self._adjust_volatility_based_levels(position, market_condition)
            
            # Adjust ATR-based levels
            elif self.risk_params.stop_loss_type == StopLossType.ATR_BASED:
                await self._adjust_atr_based_levels(position, market_condition)
            
            # Check for partial profit taking
            if self.risk_params.take_profit_type == TakeProfitType.SCALING:
                await self._check_partial_profits(position, market_condition)
            
            # Update performance stats
            adjustment_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_stats['total_adjustments'] += 1
            self.performance_stats['average_adjustment_time'] = (
                (self.performance_stats['average_adjustment_time'] * (self.performance_stats['total_adjustments'] - 1) + adjustment_time) /
                self.performance_stats['total_adjustments']
            )
            
        except Exception as e:
            logger.error(f"Failed to adjust dynamic levels for {position.position_id}: {e}")

    async def _adjust_trailing_stop(self, position: Position, market_condition: MarketCondition):
        """Adjust trailing stop-loss"""
        try:
            trailing_distance = self.risk_params.trailing_stop_distance
            
            # Adjust distance based on volatility
            if market_condition.volatility > 0.002:  # High volatility
                trailing_distance *= 1.5
            elif market_condition.volatility < 0.0005:  # Low volatility
                trailing_distance *= 0.7
            
            if position.side == 'buy':
                new_stop = position.current_price - trailing_distance
                if position.stop_loss is None or new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    logger.info(f"Trailing stop adjusted for {position.position_id}: {new_stop:.5f}")
            else:
                new_stop = position.current_price + trailing_distance
                if position.stop_loss is None or new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    logger.info(f"Trailing stop adjusted for {position.position_id}: {new_stop:.5f}")
                    
        except Exception as e:
            logger.error(f"Failed to adjust trailing stop for {position.position_id}: {e}")

    async def _adjust_volatility_based_levels(self, position: Position, market_condition: MarketCondition):
        """Adjust levels based on current volatility"""
        try:
            volatility_factor = market_condition.volatility * self.risk_params.volatility_multiplier
            
            if position.side == 'buy':
                new_stop = position.current_price - volatility_factor
                if position.stop_loss is None or new_stop > position.stop_loss:
                    position.stop_loss = new_stop
            else:
                new_stop = position.current_price + volatility_factor
                if position.stop_loss is None or new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    
        except Exception as e:
            logger.error(f"Failed to adjust volatility-based levels for {position.position_id}: {e}")

    async def _adjust_atr_based_levels(self, position: Position, market_condition: MarketCondition):
        """Adjust levels based on Average True Range"""
        try:
            atr_factor = market_condition.atr_value * self.risk_params.volatility_multiplier
            
            if position.side == 'buy':
                new_stop = position.current_price - atr_factor
                if position.stop_loss is None or new_stop > position.stop_loss:
                    position.stop_loss = new_stop
            else:
                new_stop = position.current_price + atr_factor
                if position.stop_loss is None or new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    
        except Exception as e:
            logger.error(f"Failed to adjust ATR-based levels for {position.position_id}: {e}")

    async def _check_partial_profits(self, position: Position, market_condition: MarketCondition):
        """Check for partial profit taking opportunities"""
        try:
            # Calculate profit percentage
            if position.side == 'buy':
                profit_pct = (position.current_price - position.entry_price) / position.entry_price
            else:
                profit_pct = (position.entry_price - position.current_price) / position.entry_price
            
            # Take partial profits at Fibonacci levels
            for level in self.risk_params.fibonacci_levels:
                if profit_pct >= level * 0.01:  # 1% profit per Fibonacci level
                    # Signal for partial profit taking
                    logger.info(f"Partial profit opportunity for {position.position_id} at {level} level")
                    
        except Exception as e:
            logger.error(f"Failed to check partial profits for {position.position_id}: {e}")

    async def _get_market_condition(self, symbol: str) -> MarketCondition:
        """Get current market conditions for symbol"""
        try:
            # Try to get from cache first
            cached_condition = self.redis_client.get(f"market_condition:{symbol}")
            if cached_condition:
                data = json.loads(cached_condition)
                return MarketCondition(**data)
            
            # Generate mock market condition (replace with actual market data)
            condition = MarketCondition(
                volatility=np.random.uniform(0.0005, 0.003),
                trend_strength=np.random.uniform(0.3, 0.9),
                support_level=1.1000 - np.random.uniform(0.001, 0.01),
                resistance_level=1.1000 + np.random.uniform(0.001, 0.01),
                atr_value=np.random.uniform(0.0008, 0.002),
                session=self._get_current_session(),
                timestamp=datetime.now()
            )
            
            # Cache for 1 minute
            self.redis_client.setex(
                f"market_condition:{symbol}",
                60,
                json.dumps(condition.__dict__, default=str)
            )
            
            return condition
            
        except Exception as e:
            logger.error(f"Failed to get market condition for {symbol}: {e}")
            # Return default condition
            return MarketCondition(
                volatility=0.001,
                trend_strength=0.5,
                support_level=1.0950,
                resistance_level=1.1050,
                atr_value=0.001,
                session='london',
                timestamp=datetime.now()
            )

    def _get_current_session(self) -> str:
        """Determine current trading session"""
        now = datetime.now()
        hour = now.hour
        
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'london'
        elif 16 <= hour < 24:
            return 'ny'
        else:
            return 'overlap'

    async def _cache_position(self, position: Position):
        """Cache position data in Redis"""
        try:
            position_data = {
                'position_id': position.position_id,
                'symbol': position.symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'quantity': position.quantity,
                'entry_time': position.entry_time.isoformat(),
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'trailing_stop': position.trailing_stop,
                'metadata': position.metadata
            }
            
            self.redis_client.setex(
                f"position:{position.position_id}",
                3600,  # 1 hour TTL
                json.dumps(position_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache position {position.position_id}: {e}")

    async def _monitor_positions(self):
        """Background task to monitor all positions"""
        while self.running:
            try:
                for position in self.positions.values():
                    # Check for stop-loss hits
                    if position.stop_loss and self._check_stop_loss_hit(position):
                        logger.warning(f"🛑 Stop-loss hit for position {position.position_id}")
                        self.performance_stats['successful_stops'] += 1
                        await self._trigger_stop_loss(position)
                    
                    # Check for take-profit hits
                    if position.take_profit and self._check_take_profit_hit(position):
                        logger.info(f"🎯 Take-profit hit for position {position.position_id}")
                        self.performance_stats['successful_profits'] += 1
                        await self._trigger_take_profit(position)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(5)

    def _check_stop_loss_hit(self, position: Position) -> bool:
        """Check if stop-loss level has been hit"""
        if position.stop_loss is None:
            return False
        
        if position.side == 'buy':
            return position.current_price <= position.stop_loss
        else:
            return position.current_price >= position.stop_loss

    def _check_take_profit_hit(self, position: Position) -> bool:
        """Check if take-profit level has been hit"""
        if position.take_profit is None:
            return False
        
        if position.side == 'buy':
            return position.current_price >= position.take_profit
        else:
            return position.current_price <= position.take_profit

    async def _trigger_stop_loss(self, position: Position):
        """Trigger stop-loss execution"""
        try:
            # Signal to trading engine to close position
            signal = {
                'action': 'close_position',
                'position_id': position.position_id,
                'reason': 'stop_loss',
                'price': position.stop_loss,
                'timestamp': datetime.now().isoformat()
            }
            
            # Publish to Redis for trading engine
            self.redis_client.publish('trading_signals', json.dumps(signal))
            
            logger.info(f"Stop-loss signal sent for position {position.position_id}")
            
        except Exception as e:
            logger.error(f"Failed to trigger stop-loss for {position.position_id}: {e}")

    async def _trigger_take_profit(self, position: Position):
        """Trigger take-profit execution"""
        try:
            # Signal to trading engine to close position
            signal = {
                'action': 'close_position',
                'position_id': position.position_id,
                'reason': 'take_profit',
                'price': position.take_profit,
                'timestamp': datetime.now().isoformat()
            }
            
            # Publish to Redis for trading engine
            self.redis_client.publish('trading_signals', json.dumps(signal))
            
            logger.info(f"Take-profit signal sent for position {position.position_id}")
            
        except Exception as e:
            logger.error(f"Failed to trigger take-profit for {position.position_id}: {e}")

    async def _update_market_conditions(self):
        """Background task to update market conditions"""
        while self.running:
            try:
                # Update market conditions for all active symbols
                symbols = set(pos.symbol for pos in self.positions.values())
                
                for symbol in symbols:
                    await self._get_market_condition(symbol)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating market conditions: {e}")
                await asyncio.sleep(60)

    async def _performance_monitoring(self):
        """Background task for performance monitoring"""
        while self.running:
            try:
                # Log performance statistics
                logger.info(f"Performance Stats: {self.performance_stats}")
                
                # Cache performance stats
                self.redis_client.setex(
                    'risk_manager_performance',
                    300,  # 5 minutes TTL
                    json.dumps(self.performance_stats)
                )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_stats.copy()

    async def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all managed positions"""
        try:
            total_positions = len(self.positions)
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            positions_by_side = {'buy': 0, 'sell': 0}
            for pos in self.positions.values():
                positions_by_side[pos.side] += 1
            
            return {
                'total_positions': total_positions,
                'total_unrealized_pnl': total_unrealized_pnl,
                'positions_by_side': positions_by_side,
                'performance_stats': self.performance_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get position summary: {e}")
            return {}

# Example usage and testing
async def main():
    """Example usage of DynamicLevelManager"""
    # Initialize manager
    manager = DynamicLevelManager()
    await manager.start()
    
    # Create test position
    test_position = Position(
        position_id="TEST_001",
        symbol="EURUSD",
        side="buy",
        entry_price=1.1000,
        quantity=100000,
        entry_time=datetime.now()
    )
    
    # Add position to management
    await manager.add_position(test_position)
    
    # Simulate price updates
    for i in range(10):
        new_price = 1.1000 + (i * 0.0001)  # Price moving up
        await manager.update_position_price("TEST_001", new_price)
        await asyncio.sleep(1)
    
    # Get summary
    summary = await manager.get_position_summary()
    print(f"Position Summary: {summary}")
    
    # Stop manager
    await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
