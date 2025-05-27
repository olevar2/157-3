"""
Hedging Strategy Manager
Automated hedging strategies for risk mitigation and portfolio protection

Features:
- Currency pair correlation hedging
- Portfolio delta hedging
- Volatility hedging strategies
- Dynamic hedge ratio calculation
- Real-time hedge effectiveness monitoring
- Multiple hedging algorithms
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

class HedgeType(Enum):
    """Types of hedging strategies"""
    CORRELATION = "correlation"
    DELTA_NEUTRAL = "delta_neutral"
    VOLATILITY = "volatility"
    PAIRS_TRADING = "pairs_trading"
    BASKET = "basket"
    DYNAMIC = "dynamic"

class HedgeStatus(Enum):
    """Hedge status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PARTIAL = "partial"
    EXPIRED = "expired"
    FAILED = "failed"

@dataclass
class Position:
    """Trading position for hedging"""
    position_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HedgePosition:
    """Hedge position data"""
    hedge_id: str
    original_position_id: str
    hedge_symbol: str
    hedge_side: str
    hedge_quantity: float
    hedge_price: float
    hedge_ratio: float
    effectiveness: float
    created_at: datetime
    status: HedgeStatus = HedgeStatus.ACTIVE

@dataclass
class CorrelationData:
    """Currency pair correlation data"""
    symbol1: str
    symbol2: str
    correlation: float
    rolling_correlation: float
    confidence: float
    last_updated: datetime

class HedgingStrategyManager:
    """
    Advanced automated hedging strategy manager
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.positions: Dict[str, Position] = {}
        self.hedge_positions: Dict[str, HedgePosition] = {}
        self.correlations: Dict[str, CorrelationData] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        # Hedging parameters
        self.hedge_threshold = 0.7  # Minimum correlation for hedging
        self.max_hedge_ratio = 1.0  # Maximum hedge ratio
        self.rebalance_threshold = 0.1  # Rebalance when effectiveness drops below 90%
        self.correlation_window = 50  # Rolling correlation window
        
        # Performance tracking
        self.performance_stats = {
            'total_hedges': 0,
            'active_hedges': 0,
            'hedge_effectiveness': 0.0,
            'portfolio_variance_reduction': 0.0,
            'hedging_cost': 0.0
        }
        
        logger.info("HedgingStrategyManager initialized")

    async def start(self):
        """Start the hedging strategy manager"""
        self.running = True
        logger.info("Starting Hedging Strategy Manager...")
        
        # Start background tasks
        asyncio.create_task(self._monitor_correlations())
        asyncio.create_task(self._monitor_hedge_effectiveness())
        asyncio.create_task(self._rebalance_hedges())
        
        logger.info("✅ Hedging Strategy Manager started successfully")

    async def stop(self):
        """Stop the hedging strategy manager"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Hedging Strategy Manager stopped")

    async def add_position(self, position: Position) -> bool:
        """Add position and evaluate hedging opportunities"""
        try:
            self.positions[position.position_id] = position
            
            # Evaluate hedging opportunities
            await self._evaluate_hedging_opportunities(position)
            
            logger.info(f"✅ Added position {position.position_id} and evaluated hedging")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add position {position.position_id}: {e}")
            return False

    async def remove_position(self, position_id: str) -> bool:
        """Remove position and close associated hedges"""
        try:
            if position_id in self.positions:
                # Close associated hedges
                await self._close_position_hedges(position_id)
                
                # Remove position
                del self.positions[position_id]
                
                logger.info(f"✅ Removed position {position_id} and closed hedges")
                return True
            else:
                logger.warning(f"Position {position_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to remove position {position_id}: {e}")
            return False

    async def create_correlation_hedge(self, position_id: str, hedge_symbol: str) -> Optional[HedgePosition]:
        """Create correlation-based hedge"""
        try:
            position = self.positions.get(position_id)
            if not position:
                logger.error(f"Position {position_id} not found")
                return None
            
            # Get correlation data
            correlation_key = f"{position.symbol}_{hedge_symbol}"
            correlation_data = await self._get_correlation(position.symbol, hedge_symbol)
            
            if abs(correlation_data.correlation) < self.hedge_threshold:
                logger.warning(f"Correlation too low for hedging: {correlation_data.correlation}")
                return None
            
            # Calculate hedge ratio
            hedge_ratio = min(abs(correlation_data.correlation), self.max_hedge_ratio)
            
            # Calculate hedge quantity
            hedge_quantity = position.quantity * hedge_ratio
            
            # Determine hedge side (opposite for negative correlation)
            if correlation_data.correlation > 0:
                hedge_side = 'sell' if position.side == 'buy' else 'buy'
            else:
                hedge_side = position.side
            
            # Create hedge position
            hedge_position = HedgePosition(
                hedge_id=f"hedge_{position_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                original_position_id=position_id,
                hedge_symbol=hedge_symbol,
                hedge_side=hedge_side,
                hedge_quantity=hedge_quantity,
                hedge_price=await self._get_current_price(hedge_symbol),
                hedge_ratio=hedge_ratio,
                effectiveness=abs(correlation_data.correlation),
                created_at=datetime.now()
            )
            
            self.hedge_positions[hedge_position.hedge_id] = hedge_position
            self.performance_stats['total_hedges'] += 1
            self.performance_stats['active_hedges'] += 1
            
            logger.info(f"✅ Created correlation hedge {hedge_position.hedge_id}")
            return hedge_position
            
        except Exception as e:
            logger.error(f"❌ Failed to create correlation hedge: {e}")
            return None

    async def create_delta_neutral_hedge(self, portfolio_positions: List[str]) -> List[HedgePosition]:
        """Create delta-neutral hedge for portfolio"""
        try:
            hedges = []
            
            # Calculate portfolio delta
            portfolio_delta = await self._calculate_portfolio_delta(portfolio_positions)
            
            if abs(portfolio_delta) < 0.01:  # Already delta neutral
                logger.info("Portfolio already delta neutral")
                return hedges
            
            # Find best hedging instrument
            hedge_symbol = await self._find_best_hedge_instrument(portfolio_positions)
            
            # Calculate hedge quantity to neutralize delta
            hedge_quantity = abs(portfolio_delta)
            hedge_side = 'sell' if portfolio_delta > 0 else 'buy'
            
            # Create hedge
            hedge_position = HedgePosition(
                hedge_id=f"delta_hedge_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                original_position_id="portfolio",
                hedge_symbol=hedge_symbol,
                hedge_side=hedge_side,
                hedge_quantity=hedge_quantity,
                hedge_price=await self._get_current_price(hedge_symbol),
                hedge_ratio=1.0,
                effectiveness=0.95,  # Assume high effectiveness for delta hedging
                created_at=datetime.now()
            )
            
            self.hedge_positions[hedge_position.hedge_id] = hedge_position
            hedges.append(hedge_position)
            
            self.performance_stats['total_hedges'] += 1
            self.performance_stats['active_hedges'] += 1
            
            logger.info(f"✅ Created delta-neutral hedge {hedge_position.hedge_id}")
            return hedges
            
        except Exception as e:
            logger.error(f"❌ Failed to create delta-neutral hedge: {e}")
            return []

    async def _evaluate_hedging_opportunities(self, position: Position):
        """Evaluate hedging opportunities for a position"""
        try:
            # Get correlated instruments
            correlated_symbols = await self._get_correlated_symbols(position.symbol)
            
            for symbol, correlation in correlated_symbols.items():
                if abs(correlation) >= self.hedge_threshold:
                    # Consider creating hedge
                    hedge_position = await self.create_correlation_hedge(position.position_id, symbol)
                    if hedge_position:
                        logger.info(f"Auto-created hedge for {position.position_id} with {symbol}")
                        
        except Exception as e:
            logger.error(f"Failed to evaluate hedging opportunities: {e}")

    async def _get_correlation(self, symbol1: str, symbol2: str) -> CorrelationData:
        """Get correlation data between two symbols"""
        try:
            # Try cache first
            cache_key = f"correlation:{symbol1}:{symbol2}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return CorrelationData(**data)
            
            # Calculate correlation (mock implementation)
            correlation = np.random.uniform(-0.9, 0.9)
            rolling_correlation = correlation + np.random.uniform(-0.1, 0.1)
            
            correlation_data = CorrelationData(
                symbol1=symbol1,
                symbol2=symbol2,
                correlation=correlation,
                rolling_correlation=rolling_correlation,
                confidence=0.85,
                last_updated=datetime.now()
            )
            
            # Cache for 5 minutes
            self.redis_client.setex(
                cache_key,
                300,
                json.dumps(correlation_data.__dict__, default=str)
            )
            
            return correlation_data
            
        except Exception as e:
            logger.error(f"Failed to get correlation for {symbol1}-{symbol2}: {e}")
            return CorrelationData(symbol1, symbol2, 0.0, 0.0, 0.0, datetime.now())

    async def _get_correlated_symbols(self, symbol: str) -> Dict[str, float]:
        """Get symbols correlated with the given symbol"""
        try:
            # Mock correlated symbols (replace with actual correlation matrix)
            correlations = {
                'EURUSD': {'GBPUSD': 0.75, 'AUDUSD': 0.65, 'USDCHF': -0.80},
                'GBPUSD': {'EURUSD': 0.75, 'EURGBP': -0.85, 'AUDUSD': 0.70},
                'USDJPY': {'USDCHF': 0.60, 'EURJPY': 0.85, 'GBPJPY': 0.80},
                'AUDUSD': {'NZDUSD': 0.85, 'EURUSD': 0.65, 'USDCAD': -0.70}
            }
            
            return correlations.get(symbol, {})
            
        except Exception as e:
            logger.error(f"Failed to get correlated symbols for {symbol}: {e}")
            return {}

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (mock implementation)"""
        # Mock price data
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.3000,
            'USDJPY': 110.00,
            'AUDUSD': 0.7500,
            'USDCHF': 0.9200,
            'NZDUSD': 0.7000
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        return base_price + np.random.uniform(-0.01, 0.01)

    async def _calculate_portfolio_delta(self, position_ids: List[str]) -> float:
        """Calculate portfolio delta"""
        try:
            total_delta = 0.0
            
            for position_id in position_ids:
                position = self.positions.get(position_id)
                if position:
                    # Simple delta calculation (1 for buy, -1 for sell)
                    delta = position.quantity if position.side == 'buy' else -position.quantity
                    total_delta += delta
            
            return total_delta
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio delta: {e}")
            return 0.0

    async def _find_best_hedge_instrument(self, position_ids: List[str]) -> str:
        """Find best instrument for hedging portfolio"""
        # Simple implementation - return most liquid pair
        return 'EURUSD'

    async def _close_position_hedges(self, position_id: str):
        """Close all hedges associated with a position"""
        try:
            hedges_to_close = [
                hedge_id for hedge_id, hedge in self.hedge_positions.items()
                if hedge.original_position_id == position_id
            ]
            
            for hedge_id in hedges_to_close:
                hedge = self.hedge_positions[hedge_id]
                hedge.status = HedgeStatus.EXPIRED
                self.performance_stats['active_hedges'] -= 1
                logger.info(f"Closed hedge {hedge_id}")
                
        except Exception as e:
            logger.error(f"Failed to close hedges for position {position_id}: {e}")

    async def _monitor_correlations(self):
        """Monitor correlation changes"""
        while self.running:
            try:
                # Update correlations for active hedges
                for hedge in self.hedge_positions.values():
                    if hedge.status == HedgeStatus.ACTIVE:
                        position = self.positions.get(hedge.original_position_id)
                        if position:
                            correlation_data = await self._get_correlation(
                                position.symbol, hedge.hedge_symbol
                            )
                            hedge.effectiveness = abs(correlation_data.correlation)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error monitoring correlations: {e}")
                await asyncio.sleep(60)

    async def _monitor_hedge_effectiveness(self):
        """Monitor hedge effectiveness"""
        while self.running:
            try:
                total_effectiveness = 0.0
                active_count = 0
                
                for hedge in self.hedge_positions.values():
                    if hedge.status == HedgeStatus.ACTIVE:
                        total_effectiveness += hedge.effectiveness
                        active_count += 1
                
                if active_count > 0:
                    self.performance_stats['hedge_effectiveness'] = total_effectiveness / active_count
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring hedge effectiveness: {e}")
                await asyncio.sleep(30)

    async def _rebalance_hedges(self):
        """Rebalance hedges when effectiveness drops"""
        while self.running:
            try:
                for hedge in self.hedge_positions.values():
                    if (hedge.status == HedgeStatus.ACTIVE and 
                        hedge.effectiveness < (1.0 - self.rebalance_threshold)):
                        
                        logger.info(f"Rebalancing hedge {hedge.hedge_id} due to low effectiveness")
                        # Implement rebalancing logic here
                        
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error rebalancing hedges: {e}")
                await asyncio.sleep(300)

    def get_hedge_summary(self) -> Dict[str, Any]:
        """Get summary of current hedging status"""
        return {
            'total_positions': len(self.positions),
            'active_hedges': self.performance_stats['active_hedges'],
            'hedge_effectiveness': self.performance_stats['hedge_effectiveness'],
            'portfolio_variance_reduction': self.performance_stats['portfolio_variance_reduction'],
            'hedging_cost': self.performance_stats['hedging_cost']
        }
