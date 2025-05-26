#!/usr/bin/env python3
"""
Algorithmic Arbitrage Engine
Advanced arbitrage detection and execution for forex trading platform
Identifies and exploits price discrepancies across multiple data sources and brokers

Author: Platform3 Development Team
Version: 1.0.0
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import statistics
from collections import defaultdict, deque
import threading
import numpy as np

# Third-party imports
import redis
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import pandas as pd

class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    SPATIAL = "spatial"  # Price differences across brokers
    TEMPORAL = "temporal"  # Price differences over time
    TRIANGULAR = "triangular"  # Currency triangle arbitrage
    STATISTICAL = "statistical"  # Statistical arbitrage based on correlations

class OpportunityStatus(Enum):
    """Status of arbitrage opportunity"""
    DETECTED = "detected"
    VALIDATED = "validated"
    EXECUTING = "executing"
    EXECUTED = "executed"
    EXPIRED = "expired"
    FAILED = "failed"

@dataclass
class PriceQuote:
    """Price quote from a data source"""
    source: str
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    volume: float = 0.0
    spread: float = 0.0
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.spread == 0.0:
            self.spread = self.ask - self.bid

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity data structure"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    symbol: str
    source_a: str
    source_b: str
    price_a: PriceQuote
    price_b: PriceQuote
    profit_potential: float  # Expected profit in pips
    profit_percentage: float  # Expected profit as percentage
    confidence: float  # Confidence score 0-1
    risk_score: float  # Risk assessment 0-1
    execution_window: float  # Time window in seconds
    detected_at: datetime
    status: OpportunityStatus = OpportunityStatus.DETECTED
    executed_at: Optional[datetime] = None
    actual_profit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "arbitrage_type": self.arbitrage_type.value,
            "symbol": self.symbol,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "price_a": asdict(self.price_a),
            "price_b": asdict(self.price_b),
            "profit_potential": self.profit_potential,
            "profit_percentage": self.profit_percentage,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "execution_window": self.execution_window,
            "detected_at": self.detected_at.isoformat(),
            "status": self.status.value,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "actual_profit": self.actual_profit
        }

@dataclass
class ArbitrageConfig:
    """Configuration for arbitrage engine"""
    # Minimum profit thresholds
    min_profit_pips: float = 0.5
    min_profit_percentage: float = 0.01  # 0.01%

    # Risk management
    max_position_size: float = 10000.0  # Base currency units
    max_exposure_per_symbol: float = 50000.0
    max_daily_trades: int = 100

    # Timing constraints
    max_execution_delay: float = 2.0  # seconds
    opportunity_timeout: float = 10.0  # seconds

    # Data sources to monitor
    data_sources: List[str] = None
    currency_pairs: List[str] = None

    # Database settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "forex_trading"
    postgres_user: str = "forex_admin"
    postgres_password: str = "ForexSecure2025!"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "RedisSecure2025!"

    # Performance settings
    update_interval: float = 0.1  # 100ms
    max_concurrent_opportunities: int = 50

    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = [
                "broker_a", "broker_b", "broker_c",
                "data_feed_1", "data_feed_2"
            ]

        if self.currency_pairs is None:
            self.currency_pairs = [
                "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
                "USDCAD", "USDCHF", "NZDUSD", "EURGBP"
            ]

class ArbitrageEngine:
    """
    High-performance arbitrage detection and execution engine
    Monitors multiple data sources for price discrepancies
    """

    def __init__(self, config: ArbitrageConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.running = False

        # Initialize database connections
        self._init_connections()

        # Price data storage
        self.price_data: Dict[str, Dict[str, PriceQuote]] = defaultdict(dict)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Active opportunities
        self.active_opportunities: Dict[str, ArbitrageOpportunity] = {}

        # Statistics and performance tracking
        self.stats = {
            "opportunities_detected": 0,
            "opportunities_executed": 0,
            "total_profit": 0.0,
            "total_trades": 0,
            "success_rate": 0.0,
            "avg_profit_per_trade": 0.0,
            "start_time": None,
            "type_stats": defaultdict(int),
            "source_stats": defaultdict(int)
        }

        # Thread safety
        self.lock = threading.RLock()

        # Risk management
        self.daily_trades = 0
        self.daily_reset_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self.position_exposure: Dict[str, float] = defaultdict(float)

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("ArbitrageEngine")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _init_connections(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection pool
            self.pg_pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=8,
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )

            # Redis connection
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=True
            )

            self.logger.info("Database connections initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize connections: {e}")
            raise

    async def start(self):
        """Start the arbitrage engine"""
        self.logger.info("Starting arbitrage engine...")
        self.running = True
        self.stats["start_time"] = time.time()

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_price_feeds()),
            asyncio.create_task(self._detect_opportunities()),
            asyncio.create_task(self._manage_opportunities()),
            asyncio.create_task(self._monitor_performance()),
            asyncio.create_task(self._cleanup_expired_opportunities())
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in arbitrage engine: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the arbitrage engine"""
        self.logger.info("Stopping arbitrage engine...")
        self.running = False
        self._close_connections()

    async def _monitor_price_feeds(self):
        """Monitor price feeds from multiple sources"""
        while self.running:
            try:
                # Simulate price feed monitoring
                # In production, this would connect to actual data feeds
                await self._simulate_price_updates()

                await asyncio.sleep(self.config.update_interval)

            except Exception as e:
                self.logger.error(f"Error monitoring price feeds: {e}")
                await asyncio.sleep(1)

    async def _simulate_price_updates(self):
        """Simulate price updates from multiple sources"""
        # This is a simulation - in production, replace with actual data feed connections
        import random

        base_prices = {
            "EURUSD": 1.1000,
            "GBPUSD": 1.3000,
            "USDJPY": 110.00,
            "AUDUSD": 0.7500
        }

        for symbol in self.config.currency_pairs[:4]:  # Limit for simulation
            for source in self.config.data_sources[:3]:  # Limit for simulation
                if symbol in base_prices:
                    base_price = base_prices[symbol]

                    # Add random variation
                    variation = random.uniform(-0.001, 0.001)
                    bid = base_price + variation
                    ask = bid + random.uniform(0.0001, 0.0005)

                    # Add source-specific bias to create arbitrage opportunities
                    if source == "broker_b":
                        bid += random.uniform(-0.0002, 0.0002)
                        ask += random.uniform(-0.0002, 0.0002)

                    quote = PriceQuote(
                        source=source,
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        timestamp=datetime.now(timezone.utc),
                        volume=random.uniform(100, 1000),
                        latency_ms=random.uniform(1, 10)
                    )

                    await self._update_price_data(quote)

    async def _update_price_data(self, quote: PriceQuote):
        """Update price data and trigger opportunity detection"""
        with self.lock:
            # Store latest price
            self.price_data[quote.symbol][quote.source] = quote

            # Store in history
            self.price_history[f"{quote.symbol}_{quote.source}"].append(quote)

            # Cache in Redis for fast access
            cache_key = f"price:{quote.symbol}:{quote.source}"
            self.redis_client.hset(cache_key, mapping={
                "bid": quote.bid,
                "ask": quote.ask,
                "timestamp": quote.timestamp.isoformat(),
                "spread": quote.spread
            })
            self.redis_client.expire(cache_key, 60)  # 1 minute expiry

    async def _detect_opportunities(self):
        """Detect arbitrage opportunities"""
        while self.running:
            try:
                # Check for spatial arbitrage (price differences across sources)
                await self._detect_spatial_arbitrage()

                # Check for triangular arbitrage
                await self._detect_triangular_arbitrage()

                await asyncio.sleep(0.05)  # 50ms detection cycle

            except Exception as e:
                self.logger.error(f"Error detecting opportunities: {e}")
                await asyncio.sleep(1)

    async def _detect_spatial_arbitrage(self):
        """Detect spatial arbitrage opportunities"""
        with self.lock:
            for symbol in self.config.currency_pairs:
                if symbol in self.price_data:
                    sources = list(self.price_data[symbol].keys())

                    # Compare all source pairs
                    for i in range(len(sources)):
                        for j in range(i + 1, len(sources)):
                            source_a = sources[i]
                            source_b = sources[j]

                            quote_a = self.price_data[symbol][source_a]
                            quote_b = self.price_data[symbol][source_b]

                            # Check if quotes are recent enough
                            now = datetime.now(timezone.utc)
                            if (now - quote_a.timestamp).total_seconds() > 5 or \
                               (now - quote_b.timestamp).total_seconds() > 5:
                                continue

                            # Check for arbitrage opportunity
                            opportunity = self._analyze_spatial_opportunity(quote_a, quote_b)
                            if opportunity:
                                await self._process_opportunity(opportunity)

    def _analyze_spatial_opportunity(self, quote_a: PriceQuote, quote_b: PriceQuote) -> Optional[ArbitrageOpportunity]:
        """Analyze potential spatial arbitrage opportunity"""
        try:
            # Calculate potential profit scenarios
            # Scenario 1: Buy from A, sell to B
            profit_a_to_b = quote_b.bid - quote_a.ask

            # Scenario 2: Buy from B, sell to A
            profit_b_to_a = quote_a.bid - quote_b.ask

            # Choose the more profitable scenario
            if profit_a_to_b > profit_b_to_a:
                profit_pips = profit_a_to_b * 10000  # Convert to pips
                profit_percentage = (profit_a_to_b / quote_a.ask) * 100
                buy_source = quote_a.source
                sell_source = quote_b.source
                buy_price = quote_a
                sell_price = quote_b
            else:
                profit_pips = profit_b_to_a * 10000
                profit_percentage = (profit_b_to_a / quote_b.ask) * 100
                buy_source = quote_b.source
                sell_source = quote_a.source
                buy_price = quote_b
                sell_price = quote_a

            # Check if opportunity meets minimum thresholds
            if (profit_pips >= self.config.min_profit_pips and
                profit_percentage >= self.config.min_profit_percentage):

                # Calculate confidence and risk scores
                confidence = self._calculate_confidence(buy_price, sell_price)
                risk_score = self._calculate_risk_score(buy_price, sell_price)

                # Calculate execution window
                execution_window = self._calculate_execution_window(buy_price, sell_price)

                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"spatial_{quote_a.symbol}_{int(time.time() * 1000)}",
                    arbitrage_type=ArbitrageType.SPATIAL,
                    symbol=quote_a.symbol,
                    source_a=buy_source,
                    source_b=sell_source,
                    price_a=buy_price,
                    price_b=sell_price,
                    profit_potential=profit_pips,
                    profit_percentage=profit_percentage,
                    confidence=confidence,
                    risk_score=risk_score,
                    execution_window=execution_window,
                    detected_at=datetime.now(timezone.utc)
                )

                return opportunity

            return None

        except Exception as e:
            self.logger.error(f"Error analyzing spatial opportunity: {e}")
            return None

    def _calculate_confidence(self, price_a: PriceQuote, price_b: PriceQuote) -> float:
        """Calculate confidence score for opportunity"""
        confidence = 1.0

        # Reduce confidence based on latency
        max_latency = max(price_a.latency_ms, price_b.latency_ms)
        confidence -= min(max_latency / 100, 0.3)  # Max 30% reduction

        # Reduce confidence based on spread width
        avg_spread = (price_a.spread + price_b.spread) / 2
        if avg_spread > 0.0005:  # 0.5 pips
            confidence -= min(avg_spread * 1000, 0.2)  # Max 20% reduction

        # Reduce confidence based on time difference
        time_diff = abs((price_a.timestamp - price_b.timestamp).total_seconds())
        confidence -= min(time_diff / 10, 0.2)  # Max 20% reduction

        return max(confidence, 0.1)  # Minimum 10% confidence

    def _calculate_risk_score(self, price_a: PriceQuote, price_b: PriceQuote) -> float:
        """Calculate risk score for opportunity"""
        risk = 0.0

        # Increase risk based on volatility
        symbol = price_a.symbol
        if symbol in self.price_history:
            recent_prices = [p.mid_price for p in list(self.price_history[f"{symbol}_{price_a.source}"])[-20:]]
            if len(recent_prices) > 5:
                volatility = statistics.stdev(recent_prices)
                risk += min(volatility * 10000, 0.5)  # Max 50% risk from volatility

        # Increase risk based on execution delay potential
        max_latency = max(price_a.latency_ms, price_b.latency_ms)
        risk += min(max_latency / 200, 0.3)  # Max 30% risk from latency

        return min(risk, 1.0)  # Cap at 100%

    def _calculate_execution_window(self, price_a: PriceQuote, price_b: PriceQuote) -> float:
        """Calculate execution window in seconds"""
        base_window = 5.0  # 5 seconds base

        # Reduce window based on volatility and latency
        max_latency = max(price_a.latency_ms, price_b.latency_ms)
        window = base_window - (max_latency / 1000)  # Reduce by latency

        return max(window, 1.0)  # Minimum 1 second

    async def _detect_triangular_arbitrage(self):
        """Detect triangular arbitrage opportunities"""
        # Triangular arbitrage example: EUR/USD, GBP/USD, EUR/GBP
        triangles = [
            ("EURUSD", "GBPUSD", "EURGBP"),
            ("USDJPY", "EURJPY", "EURUSD"),
            ("AUDUSD", "NZDUSD", "AUDNZD")
        ]

        with self.lock:
            for triangle in triangles:
                if all(symbol in self.price_data for symbol in triangle):
                    opportunity = await self._analyze_triangular_opportunity(triangle)
                    if opportunity:
                        await self._process_opportunity(opportunity)

    async def _analyze_triangular_opportunity(self, triangle: Tuple[str, str, str]) -> Optional[ArbitrageOpportunity]:
        """Analyze triangular arbitrage opportunity"""
        try:
            pair1, pair2, pair3 = triangle

            # Get the best quotes for each pair from all sources
            quote1 = self._get_best_quote(pair1)
            quote2 = self._get_best_quote(pair2)
            quote3 = self._get_best_quote(pair3)

            if not all([quote1, quote2, quote3]):
                return None

            # Calculate triangular arbitrage profit
            # This is a simplified calculation - real implementation would need
            # to consider currency directions and cross rates

            # Forward direction
            forward_rate = quote1.mid_price * quote2.mid_price / quote3.mid_price
            forward_profit = abs(forward_rate - 1.0)

            # Reverse direction
            reverse_rate = quote3.mid_price / (quote1.mid_price * quote2.mid_price)
            reverse_profit = abs(reverse_rate - 1.0)

            max_profit = max(forward_profit, reverse_profit)
            profit_pips = max_profit * 10000
            profit_percentage = max_profit * 100

            if (profit_pips >= self.config.min_profit_pips and
                profit_percentage >= self.config.min_profit_percentage):

                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"triangular_{pair1}_{int(time.time() * 1000)}",
                    arbitrage_type=ArbitrageType.TRIANGULAR,
                    symbol=f"{pair1}+{pair2}+{pair3}",
                    source_a="multi_source",
                    source_b="triangular",
                    price_a=quote1,
                    price_b=quote3,
                    profit_potential=profit_pips,
                    profit_percentage=profit_percentage,
                    confidence=0.7,  # Lower confidence for triangular
                    risk_score=0.4,
                    execution_window=3.0,  # Shorter window for triangular
                    detected_at=datetime.now(timezone.utc)
                )

                return opportunity

            return None

        except Exception as e:
            self.logger.error(f"Error analyzing triangular opportunity: {e}")
            return None

    def _get_best_quote(self, symbol: str) -> Optional[PriceQuote]:
        """Get the best quote for a symbol across all sources"""
        if symbol not in self.price_data:
            return None

        quotes = list(self.price_data[symbol].values())
        if not quotes:
            return None

        # Return the quote with the tightest spread
        return min(quotes, key=lambda q: q.spread)

    async def _process_opportunity(self, opportunity: ArbitrageOpportunity):
        """Process detected arbitrage opportunity"""
        try:
            # Check if we already have this opportunity
            if opportunity.opportunity_id in self.active_opportunities:
                return

            # Check risk management constraints
            if not self._check_risk_constraints(opportunity):
                return

            # Validate opportunity
            if await self._validate_opportunity(opportunity):
                opportunity.status = OpportunityStatus.VALIDATED

                # Store opportunity
                self.active_opportunities[opportunity.opportunity_id] = opportunity
                await self._store_opportunity(opportunity)

                # Update statistics
                with self.lock:
                    self.stats["opportunities_detected"] += 1
                    self.stats["type_stats"][opportunity.arbitrage_type.value] += 1

                self.logger.info(
                    f"Arbitrage opportunity detected: {opportunity.symbol} "
                    f"({opportunity.arbitrage_type.value}) - "
                    f"Profit: {opportunity.profit_potential:.2f} pips "
                    f"({opportunity.profit_percentage:.3f}%)"
                )

                # Execute if conditions are met
                if opportunity.confidence > 0.7 and opportunity.risk_score < 0.5:
                    await self._execute_opportunity(opportunity)

        except Exception as e:
            self.logger.error(f"Error processing opportunity: {e}")

    def _check_risk_constraints(self, opportunity: ArbitrageOpportunity) -> bool:
        """Check if opportunity meets risk management constraints"""
        # Check daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            return False

        # Check position exposure
        current_exposure = self.position_exposure.get(opportunity.symbol, 0)
        if current_exposure >= self.config.max_exposure_per_symbol:
            return False

        # Check if we have too many active opportunities
        if len(self.active_opportunities) >= self.config.max_concurrent_opportunities:
            return False

        return True

    async def _validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate arbitrage opportunity with fresh data"""
        try:
            # Get fresh quotes
            fresh_quote_a = await self._get_fresh_quote(opportunity.symbol, opportunity.source_a)
            fresh_quote_b = await self._get_fresh_quote(opportunity.symbol, opportunity.source_b)

            if not fresh_quote_a or not fresh_quote_b:
                return False

            # Recalculate profit with fresh data
            if opportunity.arbitrage_type == ArbitrageType.SPATIAL:
                profit_a_to_b = fresh_quote_b.bid - fresh_quote_a.ask
                profit_b_to_a = fresh_quote_a.bid - fresh_quote_b.ask
                max_profit = max(profit_a_to_b, profit_b_to_a)
                profit_pips = max_profit * 10000

                # Check if opportunity still exists
                return profit_pips >= self.config.min_profit_pips

            return True

        except Exception as e:
            self.logger.error(f"Error validating opportunity: {e}")
            return False

    async def _get_fresh_quote(self, symbol: str, source: str) -> Optional[PriceQuote]:
        """Get fresh quote for validation"""
        # In production, this would fetch real-time data
        # For simulation, return cached data if recent enough
        with self.lock:
            if symbol in self.price_data and source in self.price_data[symbol]:
                quote = self.price_data[symbol][source]
                age = (datetime.now(timezone.utc) - quote.timestamp).total_seconds()
                if age < 2.0:  # Quote is fresh enough
                    return quote

        return None

    async def _execute_opportunity(self, opportunity: ArbitrageOpportunity):
        """Execute arbitrage opportunity"""
        try:
            opportunity.status = OpportunityStatus.EXECUTING

            # Simulate execution (in production, place actual trades)
            execution_success = await self._simulate_execution(opportunity)

            if execution_success:
                opportunity.status = OpportunityStatus.EXECUTED
                opportunity.executed_at = datetime.now(timezone.utc)
                opportunity.actual_profit = opportunity.profit_potential * 0.8  # Simulate slippage

                # Update statistics
                with self.lock:
                    self.stats["opportunities_executed"] += 1
                    self.stats["total_profit"] += opportunity.actual_profit
                    self.stats["total_trades"] += 1
                    self.daily_trades += 1

                    # Update position exposure
                    self.position_exposure[opportunity.symbol] += self.config.max_position_size

                self.logger.info(
                    f"Arbitrage executed: {opportunity.symbol} - "
                    f"Actual profit: {opportunity.actual_profit:.2f} pips"
                )
            else:
                opportunity.status = OpportunityStatus.FAILED

            # Update opportunity in storage
            await self._store_opportunity(opportunity)

        except Exception as e:
            self.logger.error(f"Error executing opportunity: {e}")
            opportunity.status = OpportunityStatus.FAILED

    async def _simulate_execution(self, opportunity: ArbitrageOpportunity) -> bool:
        """Simulate trade execution"""
        # Simulate execution delay and success probability
        await asyncio.sleep(0.1)  # 100ms execution delay

        # Success probability based on confidence and risk
        success_probability = opportunity.confidence * (1 - opportunity.risk_score)

        import random
        return random.random() < success_probability

    async def _manage_opportunities(self):
        """Manage active opportunities"""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                expired_opportunities = []

                with self.lock:
                    for opp_id, opportunity in self.active_opportunities.items():
                        # Check if opportunity has expired
                        age = (current_time - opportunity.detected_at).total_seconds()
                        if age > opportunity.execution_window:
                            opportunity.status = OpportunityStatus.EXPIRED
                            expired_opportunities.append(opp_id)

                # Remove expired opportunities
                for opp_id in expired_opportunities:
                    del self.active_opportunities[opp_id]

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                self.logger.error(f"Error managing opportunities: {e}")
                await asyncio.sleep(1)

    async def _cleanup_expired_opportunities(self):
        """Clean up expired opportunities from database"""
        while self.running:
            try:
                # Reset daily counters at midnight
                now = datetime.now(timezone.utc)
                if now.date() > self.daily_reset_time.date():
                    self.daily_trades = 0
                    self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    self.position_exposure.clear()

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                self.logger.error(f"Error in cleanup: {e}")
                await asyncio.sleep(3600)

    async def _store_opportunity(self, opportunity: ArbitrageOpportunity):
        """Store opportunity in database"""
        try:
            conn = self.pg_pool.getconn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO arbitrage_opportunities
                        (opportunity_id, arbitrage_type, symbol, source_a, source_b,
                         profit_potential, profit_percentage, confidence, risk_score,
                         execution_window, detected_at, status, executed_at, actual_profit)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (opportunity_id)
                        DO UPDATE SET
                            status = EXCLUDED.status,
                            executed_at = EXCLUDED.executed_at,
                            actual_profit = EXCLUDED.actual_profit
                    """, (
                        opportunity.opportunity_id,
                        opportunity.arbitrage_type.value,
                        opportunity.symbol,
                        opportunity.source_a,
                        opportunity.source_b,
                        opportunity.profit_potential,
                        opportunity.profit_percentage,
                        opportunity.confidence,
                        opportunity.risk_score,
                        opportunity.execution_window,
                        opportunity.detected_at,
                        opportunity.status.value,
                        opportunity.executed_at,
                        opportunity.actual_profit
                    ))
                    conn.commit()
            finally:
                self.pg_pool.putconn(conn)

        except Exception as e:
            self.logger.error(f"Error storing opportunity: {e}")

    async def _monitor_performance(self):
        """Monitor and log performance metrics"""
        while self.running:
            await asyncio.sleep(300)  # Report every 5 minutes

            with self.lock:
                if self.stats["start_time"]:
                    runtime = time.time() - self.stats["start_time"]
                    opportunities_per_hour = (self.stats["opportunities_detected"] / runtime) * 3600

                    success_rate = 0
                    if self.stats["opportunities_detected"] > 0:
                        success_rate = (self.stats["opportunities_executed"] / self.stats["opportunities_detected"]) * 100

                    avg_profit = 0
                    if self.stats["total_trades"] > 0:
                        avg_profit = self.stats["total_profit"] / self.stats["total_trades"]

                    self.logger.info(
                        f"Arbitrage Performance: "
                        f"{opportunities_per_hour:.1f} opportunities/hour, "
                        f"Success rate: {success_rate:.1f}%, "
                        f"Total profit: {self.stats['total_profit']:.2f} pips, "
                        f"Avg profit/trade: {avg_profit:.2f} pips, "
                        f"Active opportunities: {len(self.active_opportunities)}"
                    )

    def get_stats(self) -> Dict[str, Any]:
        """Get arbitrage engine statistics"""
        with self.lock:
            stats = self.stats.copy()

            if stats["start_time"]:
                runtime = time.time() - stats["start_time"]
                stats["runtime_hours"] = runtime / 3600
                stats["opportunities_per_hour"] = (stats["opportunities_detected"] / runtime) * 3600 if runtime > 0 else 0

            if stats["opportunities_detected"] > 0:
                stats["success_rate"] = (stats["opportunities_executed"] / stats["opportunities_detected"]) * 100

            if stats["total_trades"] > 0:
                stats["avg_profit_per_trade"] = stats["total_profit"] / stats["total_trades"]

            stats["active_opportunities"] = len(self.active_opportunities)
            stats["daily_trades"] = self.daily_trades

            return stats

    def get_active_opportunities(self) -> List[Dict[str, Any]]:
        """Get list of active opportunities"""
        with self.lock:
            return [opp.to_dict() for opp in self.active_opportunities.values()]

    def _close_connections(self):
        """Close database connections"""
        try:
            if hasattr(self, 'redis_client'):
                self.redis_client.close()

            if hasattr(self, 'pg_pool'):
                self.pg_pool.closeall()

            self.logger.info("Connections closed")

        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")


# Example usage and testing
async def main():
    """Main function for testing arbitrage engine"""
    config = ArbitrageConfig()
    engine = ArbitrageEngine(config)

    # Start the engine
    engine_task = asyncio.create_task(engine.start())

    # Run for a test period
    await asyncio.sleep(30)
    engine.stop()

    # Print statistics
    stats = engine.get_stats()
    print(f"Arbitrage engine statistics: {json.dumps(stats, indent=2)}")

    # Print active opportunities
    opportunities = engine.get_active_opportunities()
    print(f"Active opportunities: {len(opportunities)}")


if __name__ == "__main__":
    asyncio.run(main())
