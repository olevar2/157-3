"""
AI Feature Store Pipeline for Forex Trading Platform
Real-time feature computation and storage optimized for scalping/day trading

This pipeline processes tick data and computes features defined in feature-definitions.yaml
Optimized for sub-second latency and high-frequency trading requirements
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import redis
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from kafka import KafkaConsumer, KafkaProducer
from sqlalchemy import create_engine, text
import talib
import json
import time
from concurrent.futures import ThreadPoolExecutor
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for individual features"""
    name: str
    type: str
    description: str
    calculation: str
    timeframes: List[str]
    lag_features: List[int]
    window_functions: List[str]
    data_source: str
    update_frequency: str
    business_value: str

class FeaturePipeline:
    """High-performance feature computation pipeline for forex trading"""
    
    def __init__(self, config_path: str = "feature-definitions.yaml"):
        self.config_path = config_path
        self.features_config = self._load_feature_config()
        
        # Initialize connections
        self.redis_client = self._setup_redis()
        self.postgres_engine = self._setup_postgres()
        self.kafka_consumer = self._setup_kafka_consumer()
        self.kafka_producer = self._setup_kafka_producer()
        
        # Performance optimization
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.feature_cache = {}
        self.calculation_cache = {}
        
        # Session tracking for trading hours
        self.session_tracker = SessionTracker()
        
        logger.info("Feature Pipeline initialized with {} feature categories".format(
            len(self.features_config)))

    def _load_feature_config(self) -> Dict:
        """Load feature definitions from YAML configuration"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config['features']
        except Exception as e:
            logger.error(f"Failed to load feature config: {e}")
            raise

    def _setup_redis(self) -> redis.Redis:
        """Setup Redis connection for feature storage"""
        return redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
            retry_on_timeout=True
        )

    def _setup_postgres(self):
        """Setup PostgreSQL connection for historical data"""
        return create_engine(
            'postgresql://trading_user:trading_pass@localhost:5432/trading_db',
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )

    def _setup_kafka_consumer(self) -> KafkaConsumer:
        """Setup Kafka consumer for real-time tick data"""
        return KafkaConsumer(
            'forex-ticks-m1',
            'forex-ticks-m5',
            'forex-orderflow',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='feature-pipeline',
            enable_auto_commit=True,
            auto_offset_reset='latest'
        )

    def _setup_kafka_producer(self) -> KafkaProducer:
        """Setup Kafka producer for computed features"""
        return KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            compression_type='lz4',
            batch_size=16384,
            linger_ms=1
        )

    async def start_pipeline(self):
        """Start the feature computation pipeline"""
        logger.info("Starting AI Feature Store Pipeline...")
        
        tasks = [
            asyncio.create_task(self._process_real_time_features()),
            asyncio.create_task(self._compute_batch_features()),
            asyncio.create_task(self._monitor_pipeline_health())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

    async def _process_real_time_features(self):
        """Process real-time tick data and compute features"""
        logger.info("Starting real-time feature processing...")
        
        while True:
            try:
                message_batch = self.kafka_consumer.poll(timeout_ms=100, max_records=500)
                
                if message_batch:
                    start_time = time.time()
                    
                    # Process messages in parallel
                    tasks = []
                    for topic_partition, messages in message_batch.items():
                        for message in messages:
                            task = asyncio.create_task(
                                self._process_tick_message(message.value)
                            )
                            tasks.append(task)
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    
                    processing_time = (time.time() - start_time) * 1000
                    logger.debug(f"Processed {len(tasks)} messages in {processing_time:.2f}ms")
                
                await asyncio.sleep(0.001)  # 1ms sleep for high-frequency processing
                
            except Exception as e:
                logger.error(f"Real-time processing error: {e}")
                await asyncio.sleep(0.1)

    async def _process_tick_message(self, tick_data: Dict):
        """Process individual tick message and compute features"""
        try:
            symbol = tick_data.get('symbol')
            timestamp = tick_data.get('timestamp')
            
            if not symbol or not timestamp:
                return
            
            # Compute microstructure features
            await self._compute_microstructure_features(symbol, tick_data)
            
            # Compute price action features
            await self._compute_price_action_features(symbol, tick_data)
            
            # Compute technical indicator features
            await self._compute_technical_features(symbol, tick_data)
            
            # Store features in Redis for real-time serving
            await self._store_real_time_features(symbol, timestamp)
            
        except Exception as e:
            logger.error(f"Tick processing error: {e}")

    async def _compute_microstructure_features(self, symbol: str, tick_data: Dict):
        """Compute market microstructure features"""
        try:
            bid = float(tick_data.get('bid', 0))
            ask = float(tick_data.get('ask', 0))
            
            # Bid-ask spread
            if bid > 0 and ask > 0:
                spread = (ask - bid) * 10000  # Convert to pips
                await self._update_feature_with_lags(
                    f"{symbol}:bid_ask_spread", 
                    spread,
                    self.features_config['microstructure']['bid_ask_spread']
                )
            
            # Order flow imbalance
            bid_volume = float(tick_data.get('bid_volume', 0))
            ask_volume = float(tick_data.get('ask_volume', 0))
            
            if bid_volume + ask_volume > 0:
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                await self._update_feature_with_lags(
                    f"{symbol}:order_flow_imbalance",
                    imbalance,
                    self.features_config['microstructure']['order_flow_imbalance']
                )
            
            # Tick volume (increment counter)
            tick_volume_key = f"{symbol}:tick_volume:m1"
            current_minute = datetime.now().replace(second=0, microsecond=0)
            self.redis_client.hincrby(f"tick_volume:{current_minute}", symbol, 1)
            
        except Exception as e:
            logger.error(f"Microstructure feature error: {e}")

    async def _compute_price_action_features(self, symbol: str, tick_data: Dict):
        """Compute price action features"""
        try:
            price = float(tick_data.get('price', 0))
            if price <= 0:
                return
            
            # Get recent prices for calculations
            recent_prices = await self._get_recent_prices(symbol, 20)
            if len(recent_prices) < 5:
                return
            
            prices_array = np.array(recent_prices + [price])
            
            # Price momentum (rate of change)
            if len(prices_array) >= 2:
                momentum = (prices_array[-1] - prices_array[-2]) / prices_array[-2] * 100
                await self._update_feature_with_lags(
                    f"{symbol}:price_momentum",
                    momentum,
                    self.features_config['price_action']['price_momentum']
                )
            
            # Price volatility (rolling standard deviation)
            if len(prices_array) >= 5:
                volatility = np.std(prices_array[-5:]) / np.mean(prices_array[-5:]) * 100
                await self._update_feature_with_lags(
                    f"{symbol}:price_volatility",
                    volatility,
                    self.features_config['price_action']['price_volatility']
                )
            
            # High-low range
            if len(prices_array) >= 5:
                price_range = (np.max(prices_array[-5:]) - np.min(prices_array[-5:])) * 10000
                await self._update_feature_with_lags(
                    f"{symbol}:high_low_range",
                    price_range,
                    self.features_config['price_action']['high_low_range']
                )
            
        except Exception as e:
            logger.error(f"Price action feature error: {e}")

    async def _compute_technical_features(self, symbol: str, tick_data: Dict):
        """Compute technical indicator features"""
        try:
            # Get sufficient historical data for technical indicators
            historical_data = await self._get_historical_ohlcv(symbol, 50)
            if len(historical_data) < 20:
                return
            
            closes = np.array([float(d['close']) for d in historical_data])
            highs = np.array([float(d['high']) for d in historical_data])
            lows = np.array([float(d['low']) for d in historical_data])
            volumes = np.array([float(d['volume']) for d in historical_data])
            
            # RSI
            rsi = talib.RSI(closes, timeperiod=14)
            if not np.isnan(rsi[-1]):
                await self._update_feature_with_lags(
                    f"{symbol}:rsi_14",
                    float(rsi[-1]),
                    self.features_config['technical_indicators']['rsi_14']
                )
            
            # Moving Averages
            sma_20 = talib.SMA(closes, timeperiod=20)
            if not np.isnan(sma_20[-1]):
                await self._update_feature_with_lags(
                    f"{symbol}:sma_20",
                    float(sma_20[-1]),
                    self.features_config['technical_indicators']['sma_20']
                )
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(closes)
            if not np.isnan(macd[-1]):
                await self._update_feature_with_lags(
                    f"{symbol}:macd",
                    float(macd[-1]),
                    self.features_config['technical_indicators']['macd']
                )
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(closes)
            if not np.isnan(bb_upper[-1]):
                bb_position = (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                await self._update_feature_with_lags(
                    f"{symbol}:bollinger_position",
                    float(bb_position),
                    self.features_config['technical_indicators']['bollinger_position']
                )
            
        except Exception as e:
            logger.error(f"Technical feature error: {e}")

    async def _update_feature_with_lags(self, feature_key: str, value: float, feature_config: FeatureConfig):
        """Update feature value with lag features and window functions"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Store current value
            self.redis_client.hset(f"features:{feature_key}", "current", value)
            self.redis_client.hset(f"features:{feature_key}", "timestamp", timestamp)
            
            # Update lag features
            for lag in feature_config.get('lag_features', []):
                lag_key = f"features:{feature_key}:lag_{lag}"
                current_lag_value = self.redis_client.hget(lag_key, "value")
                if current_lag_value is not None:
                    # Shift lag values
                    for i in range(lag - 1, 0, -1):
                        prev_value = self.redis_client.hget(f"features:{feature_key}:lag_{i}", "value")
                        if prev_value is not None:
                            self.redis_client.hset(f"features:{feature_key}:lag_{i+1}", "value", prev_value)
                
                # Set lag_1 to current value
                self.redis_client.hset(f"features:{feature_key}:lag_1", "value", value)
            
            # Compute window functions
            await self._compute_window_functions(feature_key, value, feature_config)
            
        except Exception as e:
            logger.error(f"Feature update error: {e}")

    async def _compute_window_functions(self, feature_key: str, current_value: float, feature_config: FeatureConfig):
        """Compute rolling window functions for features"""
        try:
            window_functions = feature_config.get('window_functions', [])
            
            for func in window_functions:
                if 'rolling_mean' in func:
                    window_size = int(func.split('_')[-1])
                    await self._compute_rolling_mean(feature_key, current_value, window_size)
                
                elif 'rolling_std' in func:
                    window_size = int(func.split('_')[-1])
                    await self._compute_rolling_std(feature_key, current_value, window_size)
                
                elif 'exponential_mean' in func:
                    alpha = float(func.split('_')[-1])
                    await self._compute_exponential_mean(feature_key, current_value, alpha)
                
                elif 'z_score' in func:
                    await self._compute_z_score(feature_key, current_value)
                    
        except Exception as e:
            logger.error(f"Window function error: {e}")

    async def _compute_rolling_mean(self, feature_key: str, current_value: float, window_size: int):
        """Compute rolling mean for feature"""
        try:
            values_key = f"window:{feature_key}:values"
            
            # Add current value to rolling window
            self.redis_client.lpush(values_key, current_value)
            self.redis_client.ltrim(values_key, 0, window_size - 1)
            
            # Get all values and compute mean
            values = [float(v) for v in self.redis_client.lrange(values_key, 0, -1)]
            if values:
                rolling_mean = sum(values) / len(values)
                self.redis_client.hset(f"features:{feature_key}", f"rolling_mean_{window_size}", rolling_mean)
                
        except Exception as e:
            logger.error(f"Rolling mean error: {e}")

    async def _compute_rolling_std(self, feature_key: str, current_value: float, window_size: int):
        """Compute rolling standard deviation for feature"""
        try:
            values_key = f"window:{feature_key}:values"
            values = [float(v) for v in self.redis_client.lrange(values_key, 0, window_size - 1)]
            
            if len(values) >= 2:
                rolling_std = np.std(values)
                self.redis_client.hset(f"features:{feature_key}", f"rolling_std_{window_size}", rolling_std)
                
        except Exception as e:
            logger.error(f"Rolling std error: {e}")

    async def _compute_exponential_mean(self, feature_key: str, current_value: float, alpha: float):
        """Compute exponential moving average"""
        try:
            ema_key = f"features:{feature_key}:ema_{alpha}"
            current_ema = self.redis_client.hget(ema_key, "value")
            
            if current_ema is None:
                new_ema = current_value
            else:
                new_ema = alpha * current_value + (1 - alpha) * float(current_ema)
            
            self.redis_client.hset(ema_key, "value", new_ema)
            self.redis_client.hset(f"features:{feature_key}", f"exponential_mean_{alpha}", new_ema)
            
        except Exception as e:
            logger.error(f"Exponential mean error: {e}")

    async def _get_recent_prices(self, symbol: str, count: int) -> List[float]:
        """Get recent prices from Redis cache"""
        try:
            prices_key = f"recent_prices:{symbol}"
            price_strings = self.redis_client.lrange(prices_key, 0, count - 1)
            return [float(p) for p in price_strings]
        except:
            return []

    async def _get_historical_ohlcv(self, symbol: str, count: int) -> List[Dict]:
        """Get historical OHLCV data from PostgreSQL"""
        try:
            query = text("""
                SELECT high, low, open, close, volume, timestamp
                FROM market_data_m1 
                WHERE symbol = :symbol 
                ORDER BY timestamp DESC 
                LIMIT :count
            """)
            
            with self.postgres_engine.connect() as conn:
                result = conn.execute(query, symbol=symbol, count=count)
                return [dict(row) for row in result]
        except:
            return []

    async def _store_real_time_features(self, symbol: str, timestamp: str):
        """Store computed features for real-time serving"""
        try:
            # Publish features to Kafka for downstream consumption
            feature_payload = {
                'symbol': symbol,
                'timestamp': timestamp,
                'features': await self._get_current_features(symbol)
            }
            
            self.kafka_producer.send('computed-features', feature_payload)
            
        except Exception as e:
            logger.error(f"Feature storage error: {e}")

    async def _get_current_features(self, symbol: str) -> Dict:
        """Get all current features for a symbol"""
        try:
            features = {}
            
            # Get all feature keys for the symbol
            pattern = f"features:{symbol}:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                feature_data = self.redis_client.hgetall(key)
                feature_name = key.split(':')[-1]
                features[feature_name] = feature_data
            
            return features
            
        except Exception as e:
            logger.error(f"Get features error: {e}")
            return {}

    async def _compute_batch_features(self):
        """Compute batch features periodically"""
        while True:
            try:
                # Run session-based feature computation every minute
                await self._compute_session_features()
                
                # Run correlation features every 5 minutes
                if datetime.now().minute % 5 == 0:
                    await self._compute_correlation_features()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Batch feature error: {e}")
                await asyncio.sleep(60)

    async def _compute_session_features(self):
        """Compute session-based features"""
        try:
            current_session = self.session_tracker.get_current_session()
            
            # Get session statistics from Redis
            for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']:
                session_key = f"session:{current_session}:{symbol}"
                
                # Session volatility
                session_prices = self.redis_client.lrange(f"session_prices:{symbol}", 0, -1)
                if len(session_prices) > 5:
                    prices = [float(p) for p in session_prices]
                    volatility = np.std(prices) / np.mean(prices) * 100
                    self.redis_client.hset(f"features:{symbol}:session_volatility", "current", volatility)
                
                # Session volume
                session_volume = self.redis_client.hget(f"session_volume:{symbol}", current_session)
                if session_volume:
                    self.redis_client.hset(f"features:{symbol}:session_volume", "current", float(session_volume))
                    
        except Exception as e:
            logger.error(f"Session feature error: {e}")

    async def _compute_correlation_features(self):
        """Compute correlation features between currency pairs"""
        try:
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
            
            # Get recent prices for correlation calculation
            price_data = {}
            for pair in major_pairs:
                prices = await self._get_recent_prices(pair, 20)
                if len(prices) >= 20:
                    price_data[pair] = prices
            
            # Compute pairwise correlations
            for i, pair1 in enumerate(major_pairs):
                for pair2 in major_pairs[i+1:]:
                    if pair1 in price_data and pair2 in price_data:
                        correlation = np.corrcoef(price_data[pair1], price_data[pair2])[0, 1]
                        
                        # Store correlation feature
                        corr_key = f"features:correlation:{pair1}_{pair2}"
                        self.redis_client.hset(corr_key, "current", correlation)
                        
        except Exception as e:
            logger.error(f"Correlation feature error: {e}")

    async def _monitor_pipeline_health(self):
        """Monitor pipeline health and performance"""
        while True:
            try:
                # Check Redis connection
                self.redis_client.ping()
                
                # Check Kafka consumer lag
                # Log pipeline statistics
                feature_count = len(self.redis_client.keys("features:*"))
                logger.info(f"Pipeline healthy - {feature_count} features computed")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)


class SessionTracker:
    """Track trading sessions for session-based features"""
    
    def __init__(self):
        self.sessions = {
            'asian': {'start': '21:00', 'end': '06:00'},
            'london': {'start': '07:00', 'end': '16:00'},
            'newyork': {'start': '12:00', 'end': '21:00'}
        }
    
    def get_current_session(self) -> str:
        """Get current trading session"""
        current_hour = datetime.now().hour
        
        if 21 <= current_hour or current_hour < 6:
            return 'asian'
        elif 7 <= current_hour < 16:
            return 'london'
        elif 12 <= current_hour < 21:
            return 'newyork'
        else:
            return 'overlap'


# Main execution
if __name__ == "__main__":
    pipeline = FeaturePipeline()
    asyncio.run(pipeline.start_pipeline())
