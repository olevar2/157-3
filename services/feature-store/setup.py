#!/usr/bin/env python3
"""
Feature Store Setup Script
Initializes the AI Feature Store infrastructure for forex trading
"""

import asyncio
import logging
import redis
import psycopg2
from kafka import KafkaAdminClient, KafkaClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
import yaml
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStoreSetup:
    """Setup and initialize the feature store infrastructure"""
    
    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.postgres_port = int(os.getenv('POSTGRES_PORT', 5432))
        self.postgres_db = os.getenv('POSTGRES_DB', 'trading_db')
        self.postgres_user = os.getenv('POSTGRES_USER', 'trading_user')
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', 'trading_pass')
        self.kafka_brokers = os.getenv('KAFKA_BROKERS', 'localhost:9092').split(',')

    async def setup_all(self):
        """Setup all feature store components"""
        logger.info("Starting Feature Store setup...")
        
        try:
            await self.setup_redis()
            await self.setup_postgres()
            await self.setup_kafka()
            await self.create_feature_schemas()
            await self.validate_setup()
            
            logger.info("Feature Store setup completed successfully!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            sys.exit(1)

    async def setup_redis(self):
        """Setup Redis for feature storage"""
        logger.info("Setting up Redis...")
        
        try:
            # Connect to Redis
            r = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            r.ping()
            
            # Create feature namespaces
            namespaces = [
                'features',
                'window',
                'history', 
                'session',
                'correlation',
                'metadata'
            ]
            
            for namespace in namespaces:
                r.sadd('feature_namespaces', namespace)
            
            # Set up feature expiration policies
            r.config_set('maxmemory-policy', 'allkeys-lru')
            r.config_set('maxmemory', '2gb')
            
            # Create feature indexes for fast lookups
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
            for symbol in symbols:
                r.sadd('active_symbols', symbol)
            
            logger.info("Redis setup completed")
            
        except Exception as e:
            logger.error(f"Redis setup failed: {e}")
            raise

    async def setup_postgres(self):
        """Setup PostgreSQL for historical feature storage"""
        logger.info("Setting up PostgreSQL...")
        
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password
            )
            cur = conn.cursor()
            
            # Create feature history tables
            create_tables_sql = """
            -- Feature history table
            CREATE TABLE IF NOT EXISTS feature_history (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                feature_value DOUBLE PRECISION NOT NULL,
                computation_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                session VARCHAR(20),
                timeframe VARCHAR(10),
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            
            -- Create indexes for fast queries
            CREATE INDEX IF NOT EXISTS idx_feature_history_symbol_feature 
                ON feature_history(symbol, feature_name);
            CREATE INDEX IF NOT EXISTS idx_feature_history_timestamp 
                ON feature_history(computation_timestamp);
            CREATE INDEX IF NOT EXISTS idx_feature_history_session 
                ON feature_history(session);
            
            -- Feature metadata table
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name VARCHAR(100) PRIMARY KEY,
                feature_type VARCHAR(50) NOT NULL,
                description TEXT,
                calculation_method TEXT,
                update_frequency VARCHAR(20),
                data_source VARCHAR(100),
                business_value TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            
            -- Feature statistics table for monitoring
            CREATE TABLE IF NOT EXISTS feature_statistics (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                date DATE NOT NULL,
                min_value DOUBLE PRECISION,
                max_value DOUBLE PRECISION,
                avg_value DOUBLE PRECISION,
                std_value DOUBLE PRECISION,
                count_values BIGINT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(symbol, feature_name, date)
            );
            
            -- Performance metrics table
            CREATE TABLE IF NOT EXISTS pipeline_metrics (
                id BIGSERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata JSONB
            );
            """
            
            cur.execute(create_tables_sql)
            conn.commit()
            
            logger.info("PostgreSQL setup completed")
            
        except Exception as e:
            logger.error(f"PostgreSQL setup failed: {e}")
            raise
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()

    async def setup_kafka(self):
        """Setup Kafka topics for feature streaming"""
        logger.info("Setting up Kafka...")
        
        try:
            # Create Kafka admin client
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.kafka_brokers,
                client_id='feature_store_setup'
            )
            
            # Define topics for feature pipeline
            topics = [
                NewTopic(
                    name='computed-features',
                    num_partitions=8,
                    replication_factor=1,
                    topic_configs={
                        'retention.ms': '86400000',  # 24 hours
                        'compression.type': 'lz4',
                        'cleanup.policy': 'delete'
                    }
                ),
                NewTopic(
                    name='feature-requests',
                    num_partitions=4,
                    replication_factor=1,
                    topic_configs={
                        'retention.ms': '3600000',  # 1 hour
                        'compression.type': 'lz4'
                    }
                ),
                NewTopic(
                    name='feature-errors',
                    num_partitions=2,
                    replication_factor=1,
                    topic_configs={
                        'retention.ms': '604800000',  # 7 days
                        'cleanup.policy': 'compact'
                    }
                ),
                NewTopic(
                    name='pipeline-metrics',
                    num_partitions=2,
                    replication_factor=1,
                    topic_configs={
                        'retention.ms': '259200000',  # 3 days
                        'compression.type': 'gzip'
                    }
                )
            ]
            
            # Create topics
            try:
                admin_client.create_topics(topics, timeout_ms=10000)
                logger.info("Kafka topics created successfully")
            except Exception as e:
                if "TopicExistsError" in str(e):
                    logger.info("Kafka topics already exist")
                else:
                    raise
            
        except Exception as e:
            logger.error(f"Kafka setup failed: {e}")
            raise

    async def create_feature_schemas(self):
        """Create feature schemas in Redis from YAML definitions"""
        logger.info("Creating feature schemas...")
        
        try:
            # Load feature definitions
            with open('feature-definitions.yaml', 'r') as file:
                feature_config = yaml.safe_load(file)
            
            r = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            
            # Store feature definitions in Redis
            for category, features in feature_config['features'].items():
                for feature_name, feature_def in features.items():
                    schema_key = f"schema:{category}:{feature_name}"
                    
                    # Store feature schema
                    r.hset(schema_key, mapping={
                        'type': feature_def.get('type', 'float64'),
                        'description': feature_def.get('description', ''),
                        'calculation': feature_def.get('calculation', ''),
                        'timeframes': ','.join(feature_def.get('timeframes', [])),
                        'lag_features': ','.join(map(str, feature_def.get('lag_features', []))),
                        'window_functions': ','.join(feature_def.get('window_functions', [])),
                        'data_source': feature_def.get('data_source', ''),
                        'update_frequency': feature_def.get('update_frequency', ''),
                        'business_value': feature_def.get('business_value', ''),
                        'category': category
                    })
                    
                    # Add to feature registry
                    r.sadd('feature_registry', f"{category}:{feature_name}")
            
            logger.info("Feature schemas created successfully")
            
        except Exception as e:
            logger.error(f"Feature schema creation failed: {e}")
            raise

    async def validate_setup(self):
        """Validate that all components are working correctly"""
        logger.info("Validating setup...")
        
        try:
            # Test Redis connection
            r = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            r.ping()
            
            # Test PostgreSQL connection
            conn = psycopg2.connect(
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password
            )
            conn.close()
            
            # Test Kafka connection
            client = KafkaClient(bootstrap_servers=self.kafka_brokers)
            client.check_version()
            client.close()
            
            # Validate feature schemas
            feature_count = r.scard('feature_registry')
            if feature_count == 0:
                raise Exception("No features found in registry")
            
            logger.info(f"Validation successful - {feature_count} features registered")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

    def print_summary(self):
        """Print setup summary"""
        print("\n" + "="*60)
        print("AI FEATURE STORE SETUP COMPLETE")
        print("="*60)
        print(f"Redis: {self.redis_host}:{self.redis_port}")
        print(f"PostgreSQL: {self.postgres_host}:{self.postgres_port}")
        print(f"Kafka: {', '.join(self.kafka_brokers)}")
        print("\nNext steps:")
        print("1. Start the feature pipeline: python src/feature-pipeline.py")
        print("2. Start the serving API: npm run dev")
        print("3. Monitor logs for any errors")
        print("4. Test with sample data")
        print("="*60)


async def main():
    """Main setup function"""
    setup = FeatureStoreSetup()
    await setup.setup_all()
    setup.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
