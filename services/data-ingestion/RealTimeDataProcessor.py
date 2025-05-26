#!/usr/bin/env python3
"""
Real-Time Market Data Processor
High-throughput data ingestion and processing pipeline for forex trading platform
Optimized for scalping, day trading, and swing trading strategies

Author: Platform3 Development Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import signal
import sys

# Third-party imports
import redis
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import kafka
from kafka import KafkaProducer, KafkaConsumer
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np
import pandas as pd

# Configuration
@dataclass
class DataProcessorConfig:
    """Configuration for real-time data processor"""
    # Database connections
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "forex_trading"
    postgres_user: str = "forex_admin"
    postgres_password: str = "ForexSecure2025!"

    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "RedisSecure2025!"
    redis_db: int = 0

    # InfluxDB configuration
    influx_url: str = "http://localhost:8086"
    influx_token: str = "forex-influx-token"
    influx_org: str = "forex-trading"
    influx_bucket: str = "market-data"

    # Kafka configuration
    kafka_brokers: List[str] = None
    kafka_topics: Dict[str, str] = None

    # Performance settings
    batch_size: int = 1000
    flush_interval: float = 0.1  # 100ms
    max_workers: int = 8
    buffer_size: int = 10000

    def __post_init__(self):
        if self.kafka_brokers is None:
            self.kafka_brokers = ["localhost:9092"]
        if self.kafka_topics is None:
            self.kafka_topics = {
                "tick_data": "forex-tick-data",
                "aggregated_data": "forex-aggregated-data",
                "signals": "forex-signals"
            }

@dataclass
class TickData:
    """Tick data structure for forex market data"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    volume: float
    spread: float
    session: str  # Asian, London, NY, Overlap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "volume": self.volume,
            "spread": self.spread,
            "session": self.session
        }

class RealTimeDataProcessor:
    """
    High-performance real-time market data processor
    Handles tick data ingestion, validation, and distribution
    """

    def __init__(self, config: DataProcessorConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.running = False
        self.stats = {
            "processed_ticks": 0,
            "validation_errors": 0,
            "processing_latency": [],
            "start_time": None
        }

        # Initialize connections
        self._init_connections()

        # Processing queues and buffers
        self.tick_queue = queue.Queue(maxsize=config.buffer_size)
        self.batch_buffer = []
        self.last_flush = time.time()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)

        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("RealTimeDataProcessor")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _init_connections(self):
        """Initialize database and messaging connections"""
        try:
            # PostgreSQL connection pool
            self.pg_pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=self.config.max_workers,
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
                db=self.config.redis_db,
                decode_responses=True
            )

            # InfluxDB client
            self.influx_client = influxdb_client.InfluxDBClient(
                url=self.config.influx_url,
                token=self.config.influx_token,
                org=self.config.influx_org
            )
            self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

            # Kafka producer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_brokers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='lz4',
                batch_size=16384,
                linger_ms=10,
                acks='all'
            )

            self.logger.info("All connections initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize connections: {e}")
            raise

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    async def start(self):
        """Start the real-time data processing pipeline"""
        self.logger.info("Starting real-time data processor...")
        self.running = True
        self.stats["start_time"] = time.time()

        # Start processing tasks
        tasks = [
            asyncio.create_task(self._process_tick_queue()),
            asyncio.create_task(self._flush_batches()),
            asyncio.create_task(self._monitor_performance())
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the data processor gracefully"""
        self.logger.info("Stopping data processor...")
        self.running = False

        # Flush remaining data
        if self.batch_buffer:
            self._flush_batch()

        # Close connections
        self._close_connections()

        # Print final statistics
        self._print_final_stats()

    def ingest_tick(self, tick_data: TickData) -> bool:
        """
        Ingest a single tick data point
        Returns True if successfully queued, False if queue is full
        """
        try:
            self.tick_queue.put_nowait(tick_data)
            return True
        except queue.Full:
            self.logger.warning("Tick queue is full, dropping tick data")
            return False

    async def _process_tick_queue(self):
        """Process ticks from the queue"""
        while self.running:
            try:
                # Get tick with timeout
                try:
                    tick = self.tick_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                start_time = time.time()

                # Validate tick data
                if self._validate_tick(tick):
                    # Add to batch buffer
                    self.batch_buffer.append(tick)
                    self.stats["processed_ticks"] += 1

                    # Record processing latency
                    latency = (time.time() - start_time) * 1000  # ms
                    self.stats["processing_latency"].append(latency)

                    # Flush batch if size limit reached
                    if len(self.batch_buffer) >= self.config.batch_size:
                        await self._flush_batch_async()
                else:
                    self.stats["validation_errors"] += 1

                self.tick_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error processing tick: {e}")
                await asyncio.sleep(0.001)  # Brief pause on error

    def _validate_tick(self, tick: TickData) -> bool:
        """Validate tick data quality"""
        try:
            # Basic validation checks
            if not tick.symbol or len(tick.symbol) < 6:
                return False

            if tick.bid <= 0 or tick.ask <= 0:
                return False

            if tick.ask <= tick.bid:
                return False

            if tick.spread < 0:
                return False

            if tick.volume < 0:
                return False

            # Timestamp validation
            if not isinstance(tick.timestamp, datetime):
                return False

            # Session validation
            if tick.session not in ["Asian", "London", "NY", "Overlap"]:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    async def _flush_batches(self):
        """Periodically flush batches based on time interval"""
        while self.running:
            await asyncio.sleep(self.config.flush_interval)

            if (time.time() - self.last_flush) >= self.config.flush_interval:
                if self.batch_buffer:
                    await self._flush_batch_async()

    async def _flush_batch_async(self):
        """Asynchronously flush current batch"""
        if not self.batch_buffer:
            return

        batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.last_flush = time.time()

        # Submit batch processing to thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._process_batch, batch)

    def _flush_batch(self):
        """Synchronously flush current batch"""
        if not self.batch_buffer:
            return

        batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.last_flush = time.time()

        self._process_batch(batch)

    def _process_batch(self, batch: List[TickData]):
        """Process a batch of tick data"""
        try:
            # Store in InfluxDB for time-series analysis
            self._store_influxdb(batch)

            # Cache latest ticks in Redis
            self._cache_redis(batch)

            # Store aggregated data in PostgreSQL
            self._store_postgres(batch)

            # Publish to Kafka for real-time consumers
            self._publish_kafka(batch)

            self.logger.debug(f"Processed batch of {len(batch)} ticks")

        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")

    def _store_influxdb(self, batch: List[TickData]):
        """Store tick data in InfluxDB for time-series analysis"""
        try:
            points = []
            for tick in batch:
                point = {
                    "measurement": "forex_ticks",
                    "tags": {
                        "symbol": tick.symbol,
                        "session": tick.session
                    },
                    "fields": {
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "volume": tick.volume,
                        "spread": tick.spread,
                        "mid_price": (tick.bid + tick.ask) / 2
                    },
                    "time": tick.timestamp
                }
                points.append(point)

            self.influx_write_api.write(
                bucket=self.config.influx_bucket,
                record=points
            )

        except Exception as e:
            self.logger.error(f"InfluxDB storage error: {e}")

    def _cache_redis(self, batch: List[TickData]):
        """Cache latest tick data in Redis for fast access"""
        try:
            pipe = self.redis_client.pipeline()

            for tick in batch:
                # Store latest tick for each symbol
                key = f"tick:{tick.symbol}"
                pipe.hset(key, mapping=tick.to_dict())
                pipe.expire(key, 300)  # 5 minute expiry

                # Store in session-based lists for scalping
                session_key = f"session:{tick.session}:{tick.symbol}"
                pipe.lpush(session_key, json.dumps(tick.to_dict()))
                pipe.ltrim(session_key, 0, 999)  # Keep last 1000 ticks
                pipe.expire(session_key, 3600)  # 1 hour expiry

            pipe.execute()

        except Exception as e:
            self.logger.error(f"Redis caching error: {e}")

    def _store_postgres(self, batch: List[TickData]):
        """Store aggregated data in PostgreSQL"""
        try:
            conn = self.pg_pool.getconn()
            try:
                with conn.cursor() as cursor:
                    # Aggregate ticks by symbol and minute
                    aggregated = self._aggregate_ticks(batch)

                    for agg in aggregated:
                        cursor.execute("""
                            INSERT INTO market_data_1m
                            (symbol, timestamp, open_bid, high_bid, low_bid, close_bid,
                             open_ask, high_ask, low_ask, close_ask, volume, tick_count, session)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (symbol, timestamp)
                            DO UPDATE SET
                                high_bid = GREATEST(market_data_1m.high_bid, EXCLUDED.high_bid),
                                low_bid = LEAST(market_data_1m.low_bid, EXCLUDED.low_bid),
                                close_bid = EXCLUDED.close_bid,
                                high_ask = GREATEST(market_data_1m.high_ask, EXCLUDED.high_ask),
                                low_ask = LEAST(market_data_1m.low_ask, EXCLUDED.low_ask),
                                close_ask = EXCLUDED.close_ask,
                                volume = market_data_1m.volume + EXCLUDED.volume,
                                tick_count = market_data_1m.tick_count + EXCLUDED.tick_count
                        """, agg)

                    conn.commit()
            finally:
                self.pg_pool.putconn(conn)

        except Exception as e:
            self.logger.error(f"PostgreSQL storage error: {e}")

    def _aggregate_ticks(self, batch: List[TickData]) -> List[tuple]:
        """Aggregate tick data into 1-minute OHLC bars"""
        aggregated = {}

        for tick in batch:
            # Round timestamp to minute
            minute_ts = tick.timestamp.replace(second=0, microsecond=0)
            key = (tick.symbol, minute_ts, tick.session)

            if key not in aggregated:
                aggregated[key] = {
                    "open_bid": tick.bid,
                    "high_bid": tick.bid,
                    "low_bid": tick.bid,
                    "close_bid": tick.bid,
                    "open_ask": tick.ask,
                    "high_ask": tick.ask,
                    "low_ask": tick.ask,
                    "close_ask": tick.ask,
                    "volume": tick.volume,
                    "tick_count": 1
                }
            else:
                agg = aggregated[key]
                agg["high_bid"] = max(agg["high_bid"], tick.bid)
                agg["low_bid"] = min(agg["low_bid"], tick.bid)
                agg["close_bid"] = tick.bid
                agg["high_ask"] = max(agg["high_ask"], tick.ask)
                agg["low_ask"] = min(agg["low_ask"], tick.ask)
                agg["close_ask"] = tick.ask
                agg["volume"] += tick.volume
                agg["tick_count"] += 1

        # Convert to tuple format for database insertion
        result = []
        for (symbol, timestamp, session), agg in aggregated.items():
            result.append((
                symbol, timestamp,
                agg["open_bid"], agg["high_bid"], agg["low_bid"], agg["close_bid"],
                agg["open_ask"], agg["high_ask"], agg["low_ask"], agg["close_ask"],
                agg["volume"], agg["tick_count"], session
            ))

        return result

    def _publish_kafka(self, batch: List[TickData]):
        """Publish tick data to Kafka for real-time consumers"""
        try:
            for tick in batch:
                # Publish to tick data topic
                self.kafka_producer.send(
                    self.config.kafka_topics["tick_data"],
                    value=tick.to_dict()
                )

                # Publish aggregated data if needed
                if len(self.batch_buffer) % 100 == 0:  # Every 100 ticks
                    agg_data = {
                        "symbol": tick.symbol,
                        "timestamp": tick.timestamp.isoformat(),
                        "latest_bid": tick.bid,
                        "latest_ask": tick.ask,
                        "session": tick.session,
                        "batch_size": len(batch)
                    }

                    self.kafka_producer.send(
                        self.config.kafka_topics["aggregated_data"],
                        value=agg_data
                    )

            # Ensure messages are sent
            self.kafka_producer.flush()

        except Exception as e:
            self.logger.error(f"Kafka publishing error: {e}")

    async def _monitor_performance(self):
        """Monitor and log performance metrics"""
        while self.running:
            await asyncio.sleep(30)  # Report every 30 seconds

            if self.stats["processed_ticks"] > 0:
                runtime = time.time() - self.stats["start_time"]
                tps = self.stats["processed_ticks"] / runtime

                avg_latency = 0
                if self.stats["processing_latency"]:
                    avg_latency = np.mean(self.stats["processing_latency"][-1000:])  # Last 1000 samples

                self.logger.info(
                    f"Performance: {tps:.2f} ticks/sec, "
                    f"Avg latency: {avg_latency:.2f}ms, "
                    f"Queue size: {self.tick_queue.qsize()}, "
                    f"Validation errors: {self.stats['validation_errors']}"
                )

    def _close_connections(self):
        """Close all database and messaging connections"""
        try:
            if hasattr(self, 'kafka_producer'):
                self.kafka_producer.close()

            if hasattr(self, 'influx_client'):
                self.influx_client.close()

            if hasattr(self, 'redis_client'):
                self.redis_client.close()

            if hasattr(self, 'pg_pool'):
                self.pg_pool.closeall()

            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)

            self.logger.info("All connections closed")

        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")

    def _print_final_stats(self):
        """Print final processing statistics"""
        if self.stats["start_time"]:
            runtime = time.time() - self.stats["start_time"]
            tps = self.stats["processed_ticks"] / runtime if runtime > 0 else 0

            self.logger.info("=== Final Statistics ===")
            self.logger.info(f"Runtime: {runtime:.2f} seconds")
            self.logger.info(f"Total ticks processed: {self.stats['processed_ticks']}")
            self.logger.info(f"Average throughput: {tps:.2f} ticks/second")
            self.logger.info(f"Validation errors: {self.stats['validation_errors']}")

            if self.stats["processing_latency"]:
                avg_latency = np.mean(self.stats["processing_latency"])
                p95_latency = np.percentile(self.stats["processing_latency"], 95)
                self.logger.info(f"Average latency: {avg_latency:.2f}ms")
                self.logger.info(f"95th percentile latency: {p95_latency:.2f}ms")


# Example usage and testing
async def main():
    """Main function for testing the data processor"""
    config = DataProcessorConfig()
    processor = RealTimeDataProcessor(config)

    # Start the processor
    processor_task = asyncio.create_task(processor.start())

    # Simulate tick data ingestion
    async def simulate_ticks():
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        sessions = ["Asian", "London", "NY", "Overlap"]

        for i in range(10000):  # Simulate 10,000 ticks
            symbol = symbols[i % len(symbols)]
            session = sessions[i % len(sessions)]

            tick = TickData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bid=1.1000 + (i % 100) * 0.0001,
                ask=1.1005 + (i % 100) * 0.0001,
                volume=100 + (i % 50),
                spread=0.0005,
                session=session
            )

            processor.ingest_tick(tick)
            await asyncio.sleep(0.001)  # 1ms between ticks

    # Run simulation
    simulation_task = asyncio.create_task(simulate_ticks())

    try:
        await asyncio.gather(processor_task, simulation_task)
    except KeyboardInterrupt:
        processor.stop()


if __name__ == "__main__":
    asyncio.run(main())
