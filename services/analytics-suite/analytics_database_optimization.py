#!/usr/bin/env python3
"""
Analytics Database Optimization System for Platform3

This module provides comprehensive database optimization capabilities for analytics services including:
- Connection pooling and resource management
- Query optimization and caching strategies
- Asynchronous database operations with transaction management
- Performance monitoring and metrics collection
- Intelligent data partitioning and indexing recommendations
- Advanced caching strategies for analytics queries
- Real-time performance monitoring and alerting
"""

import asyncio
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import asyncpg
import redis
import psutil
from abc import ABC, abstractmethod

# Platform3 imports
import sys
import os
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration for analytics optimization"""
    host: str = "localhost"
    port: int = 5432
    database: str = "platform3_analytics"
    user: str = "platform3_user"
    password: str = ""
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 30.0
    command_timeout: float = 60.0
    enable_cache: bool = True
    cache_ttl: int = 300  # 5 minutes
    enable_monitoring: bool = True

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_hash: str
    execution_time: float
    rows_affected: int
    cache_hit: bool
    timestamp: datetime
    query_type: str
    table_names: List[str]
    optimization_suggestions: List[str]

@dataclass
class ConnectionPoolMetrics:
    """Connection pool performance metrics"""
    active_connections: int
    idle_connections: int
    total_connections: int
    queue_size: int
    wait_time: float
    connection_errors: int
    timestamp: datetime

@dataclass
class OptimizationReport:
    """Database optimization performance report"""
    report_id: str
    generated_at: datetime
    total_queries: int
    avg_execution_time: float
    cache_hit_ratio: float
    slow_queries: List[QueryMetrics]
    optimization_recommendations: List[str]
    connection_metrics: ConnectionPoolMetrics
    performance_score: float

class QueryCache:
    """Advanced query caching system with intelligent TTL management"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 300):
        """Initialize query cache"""
        self.default_ttl = default_ttl
        self.redis_client = None
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0,
            'cache_size': 0
        }
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis unavailable, using memory cache: {e}")
    
    def _generate_cache_key(self, query: str, params: tuple = ()) -> str:
        """Generate cache key for query with parameters"""
        query_normalized = ' '.join(query.split()).upper()
        params_str = str(sorted(params)) if params else ""
        cache_input = f"{query_normalized}:{params_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    async def get(self, query: str, params: tuple = ()) -> Optional[Any]:
        """Get cached query result"""
        cache_key = self._generate_cache_key(query, params)
        self.cache_stats['total_queries'] += 1
        
        try:
            # Try Redis first
            if self.redis_client:
                cached_data = self.redis_client.get(f"query_cache:{cache_key}")
                if cached_data:
                    self.cache_stats['hits'] += 1
                    return json.loads(cached_data)
            
            # Fallback to memory cache
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]
                if cached_item['expires_at'] > datetime.now():
                    self.cache_stats['hits'] += 1
                    return cached_item['data']
                else:
                    del self.memory_cache[cache_key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    async def set(self, query: str, result: Any, params: tuple = (), ttl: Optional[int] = None) -> bool:
        """Cache query result"""
        cache_key = self._generate_cache_key(query, params)
        ttl = ttl or self.default_ttl
        
        try:
            # Store in Redis
            if self.redis_client:
                self.redis_client.setex(
                    f"query_cache:{cache_key}",
                    ttl,
                    json.dumps(result, default=str)
                )
            
            # Store in memory cache as backup
            self.memory_cache[cache_key] = {
                'data': result,
                'expires_at': datetime.now() + timedelta(seconds=ttl)
            }
            
            self.cache_stats['cache_size'] = len(self.memory_cache)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        hit_ratio = (self.cache_stats['hits'] / max(self.cache_stats['total_queries'], 1)) * 100
        return {
            **self.cache_stats,
            'hit_ratio': hit_ratio,
            'redis_available': self.redis_client is not None
        }

class ConnectionPoolManager:
    """Advanced database connection pool with performance monitoring"""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize connection pool manager"""
        self.config = config
        self.pool = None
        self.metrics = ConnectionPoolMetrics(
            active_connections=0,
            idle_connections=0,
            total_connections=0,
            queue_size=0,
            wait_time=0.0,
            connection_errors=0,
            timestamp=datetime.now()
        )
        self.connection_history = []
        
    async def initialize(self) -> bool:
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout,
                server_settings={
                    'application_name': 'platform3_analytics',
                    'jit': 'off'  # Disable JIT for consistent performance
                }
            )
            
            logger.info(f"Database pool initialized: {self.config.min_connections}-{self.config.max_connections} connections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            self.metrics.connection_errors += 1
            return False
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool with metrics tracking"""
        start_time = time.time()
        connection = None
        
        try:
            if not self.pool:
                raise Exception("Connection pool not initialized")
            
            connection = await self.pool.acquire()
            wait_time = time.time() - start_time
            
            # Update metrics
            self.metrics.wait_time = wait_time
            self.metrics.active_connections = len(self.pool._holders) - len(self.pool._queue._queue)
            self.metrics.idle_connections = len(self.pool._queue._queue)
            self.metrics.total_connections = len(self.pool._holders)
            self.metrics.timestamp = datetime.now()
            
            yield connection
            
        except Exception as e:
            logger.error(f"Connection acquisition error: {e}")
            self.metrics.connection_errors += 1
            raise
        finally:
            if connection:
                try:
                    await self.pool.release(connection)
                except Exception as e:
                    logger.error(f"Connection release error: {e}")
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

class QueryOptimizer:
    """Intelligent query optimization and analysis system"""
    
    def __init__(self):
        """Initialize query optimizer"""
        self.query_history = []
        self.optimization_rules = self._load_optimization_rules()
        self.index_recommendations = set()
        
    def _load_optimization_rules(self) -> Dict[str, List[str]]:
        """Load query optimization rules"""
        return {
            'SELECT': [
                "Use specific column names instead of SELECT *",
                "Add appropriate WHERE clauses to limit result sets",
                "Consider using LIMIT for large result sets",
                "Use indexes on WHERE clause columns"
            ],
            'INSERT': [
                "Use batch inserts for multiple records",
                "Consider using COPY for bulk data loading",
                "Ensure proper indexing on target table"
            ],
            'UPDATE': [
                "Use specific WHERE clauses to limit affected rows",
                "Consider using indexes on WHERE clause columns",
                "Batch updates when possible"
            ],
            'DELETE': [
                "Use specific WHERE clauses to avoid accidental deletions",
                "Consider using indexes on WHERE clause columns",
                "Use CASCADE carefully with foreign keys"
            ]
        }
    
    def analyze_query(self, query: str, execution_time: float, rows_affected: int) -> QueryMetrics:
        """Analyze query performance and provide optimization suggestions"""
        query_type = self._detect_query_type(query)
        table_names = self._extract_table_names(query)
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Generate optimization suggestions
        suggestions = []
        
        # Performance-based suggestions
        if execution_time > 1.0:
            suggestions.append("Consider adding indexes to improve query performance")
        
        if rows_affected > 10000:
            suggestions.append("Large result set - consider pagination or filtering")
        
        # Query-specific suggestions
        if query_type in self.optimization_rules:
            suggestions.extend(self.optimization_rules[query_type])
        
        # Index recommendations
        if "WHERE" in query.upper() and execution_time > 0.5:
            where_columns = self._extract_where_columns(query)
            for column in where_columns:
                self.index_recommendations.add(f"CREATE INDEX idx_{column} ON {table_names[0]}({column})")
        
        metrics = QueryMetrics(
            query_hash=query_hash,
            execution_time=execution_time,
            rows_affected=rows_affected,
            cache_hit=False,
            timestamp=datetime.now(),
            query_type=query_type,
            table_names=table_names,
            optimization_suggestions=suggestions
        )
        
        self.query_history.append(metrics)
        return metrics
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of SQL query"""
        query_upper = query.upper().strip()
        for query_type in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']:
            if query_upper.startswith(query_type):
                return query_type
        return 'UNKNOWN'
    
    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from query"""
        import re
        
        # Simple table name extraction (can be enhanced)
        patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)',
            r'UPDATE\s+(\w+)'
        ]
        
        tables = set()
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            tables.update(matches)
        
        return list(tables)
    
    def _extract_where_columns(self, query: str) -> List[str]:
        """Extract column names from WHERE clauses"""
        import re
        
        # Simple WHERE column extraction
        where_pattern = r'WHERE\s+.*?(\w+)\s*[=<>!]'
        matches = re.findall(where_pattern, query, re.IGNORECASE)
        return matches
    
    def get_slow_queries(self, threshold: float = 1.0) -> List[QueryMetrics]:
        """Get queries that exceed performance threshold"""
        return [q for q in self.query_history if q.execution_time > threshold]
    
    def get_index_recommendations(self) -> List[str]:
        """Get recommended indexes based on query analysis"""
        return list(self.index_recommendations)

class AnalyticsDatabaseOptimizer:
    """Main analytics database optimization system"""
    
    def __init__(self, config: DatabaseConfig, enable_monitoring: bool = True):
        """Initialize analytics database optimizer"""
        self.config = config
        self.pool_manager = ConnectionPoolManager(config)
        self.query_cache = QueryCache() if config.enable_cache else None
        self.query_optimizer = QueryOptimizer()
        self.enable_monitoring = enable_monitoring
        
        # Performance monitoring
        self.query_metrics = []
        self.performance_history = []
        
        # Platform3 Communication Framework
        self.communication_framework = Platform3CommunicationFramework(
            service_name="analytics-database-optimizer",
            service_port=8003,
            redis_url="redis://localhost:6379",
            consul_host="localhost",
            consul_port=8500
        )
    
    async def initialize(self) -> bool:
        """Initialize the database optimizer"""
        try:
            # Initialize connection pool
            pool_success = await self.pool_manager.initialize()
            if not pool_success:
                return False
            
            # Initialize communication framework
            try:
                self.communication_framework.initialize()
                logger.info("Database optimizer communication framework initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize communication framework: {e}")
            
            # Create optimization tables if they don't exist
            await self._create_optimization_tables()
            
            logger.info("Analytics database optimizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database optimizer: {e}")
            return False
    
    async def _create_optimization_tables(self):
        """Create tables for storing optimization data"""
        create_tables_sql = """
        -- Query performance metrics table
        CREATE TABLE IF NOT EXISTS query_performance_metrics (
            id SERIAL PRIMARY KEY,
            query_hash VARCHAR(32) NOT NULL,
            execution_time FLOAT NOT NULL,
            rows_affected INTEGER NOT NULL,
            cache_hit BOOLEAN DEFAULT FALSE,
            query_type VARCHAR(20),
            timestamp TIMESTAMP DEFAULT NOW(),
            optimization_suggestions TEXT[]
        );
        
        CREATE INDEX IF NOT EXISTS idx_qpm_query_hash ON query_performance_metrics(query_hash);
        CREATE INDEX IF NOT EXISTS idx_qpm_timestamp ON query_performance_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_qpm_execution_time ON query_performance_metrics(execution_time);
        
        -- Connection pool metrics table
        CREATE TABLE IF NOT EXISTS connection_pool_metrics (
            id SERIAL PRIMARY KEY,
            active_connections INTEGER NOT NULL,
            idle_connections INTEGER NOT NULL,
            total_connections INTEGER NOT NULL,
            queue_size INTEGER NOT NULL,
            wait_time FLOAT NOT NULL,
            connection_errors INTEGER DEFAULT 0,
            timestamp TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_cpm_timestamp ON connection_pool_metrics(timestamp);
        
        -- Optimization recommendations table
        CREATE TABLE IF NOT EXISTS optimization_recommendations (
            id SERIAL PRIMARY KEY,
            recommendation_type VARCHAR(50) NOT NULL,
            recommendation_text TEXT NOT NULL,
            priority VARCHAR(20) DEFAULT 'medium',
            implemented BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_or_type ON optimization_recommendations(recommendation_type);
        CREATE INDEX IF NOT EXISTS idx_or_priority ON optimization_recommendations(priority);
        CREATE INDEX IF NOT EXISTS idx_or_implemented ON optimization_recommendations(implemented);
        """
        
        try:
            async with self.pool_manager.get_connection() as conn:
                await conn.execute(create_tables_sql)
                logger.info("Optimization tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create optimization tables: {e}")
    
    async def execute_query(self, query: str, params: tuple = (), cache_ttl: Optional[int] = None) -> Any:
        """Execute optimized query with caching and metrics"""
        start_time = time.time()
        cache_hit = False
        result = None
        
        try:
            # Check cache first
            if self.query_cache:
                cached_result = await self.query_cache.get(query, params)
                if cached_result is not None:
                    cache_hit = True
                    result = cached_result
                    execution_time = time.time() - start_time
                    
                    if self.enable_monitoring:
                        metrics = QueryMetrics(
                            query_hash=hashlib.md5(query.encode()).hexdigest(),
                            execution_time=execution_time,
                            rows_affected=len(result) if isinstance(result, list) else 1,
                            cache_hit=True,
                            timestamp=datetime.now(),
                            query_type=self.query_optimizer._detect_query_type(query),
                            table_names=self.query_optimizer._extract_table_names(query),
                            optimization_suggestions=[]
                        )
                        self.query_metrics.append(metrics)
                    
                    return result
            
            # Execute query if not cached
            async with self.pool_manager.get_connection() as conn:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)
            
            execution_time = time.time() - start_time
            rows_affected = len(result) if result else 0
            
            # Cache result if caching is enabled
            if self.query_cache and not cache_hit:
                await self.query_cache.set(query, result, params, cache_ttl)
            
            # Record metrics
            if self.enable_monitoring:
                metrics = self.query_optimizer.analyze_query(query, execution_time, rows_affected)
                metrics.cache_hit = cache_hit
                self.query_metrics.append(metrics)
                
                # Store metrics in database
                await self._store_query_metrics(metrics)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution error: {e} (took {execution_time:.3f}s)")
            raise
    
    async def _store_query_metrics(self, metrics: QueryMetrics):
        """Store query metrics in database"""
        try:
            insert_sql = """
            INSERT INTO query_performance_metrics 
            (query_hash, execution_time, rows_affected, cache_hit, query_type, optimization_suggestions)
            VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            async with self.pool_manager.get_connection() as conn:
                await conn.execute(
                    insert_sql,
                    metrics.query_hash,
                    metrics.execution_time,
                    metrics.rows_affected,
                    metrics.cache_hit,
                    metrics.query_type,
                    metrics.optimization_suggestions
                )
                
        except Exception as e:
            logger.error(f"Failed to store query metrics: {e}")
    
    async def get_optimization_report(self, hours_back: int = 24) -> OptimizationReport:
        """Generate comprehensive optimization report"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        try:
            # Get query metrics from database
            metrics_sql = """
            SELECT * FROM query_performance_metrics 
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY execution_time DESC
            """
            
            async with self.pool_manager.get_connection() as conn:
                metrics_rows = await conn.fetch(metrics_sql, start_time, end_time)
            
            # Calculate statistics
            total_queries = len(metrics_rows)
            avg_execution_time = sum(row['execution_time'] for row in metrics_rows) / max(total_queries, 1)
            cache_hits = sum(1 for row in metrics_rows if row['cache_hit'])
            cache_hit_ratio = (cache_hits / max(total_queries, 1)) * 100
            
            # Get slow queries
            slow_queries = [
                QueryMetrics(
                    query_hash=row['query_hash'],
                    execution_time=row['execution_time'],
                    rows_affected=row['rows_affected'],
                    cache_hit=row['cache_hit'],
                    timestamp=row['timestamp'],
                    query_type=row['query_type'],
                    table_names=[],
                    optimization_suggestions=row['optimization_suggestions'] or []
                )
                for row in metrics_rows if row['execution_time'] > 1.0
            ]
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(metrics_rows)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(
                avg_execution_time, cache_hit_ratio, len(slow_queries), total_queries
            )
            
            report = OptimizationReport(
                report_id=f"opt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                total_queries=total_queries,
                avg_execution_time=avg_execution_time,
                cache_hit_ratio=cache_hit_ratio,
                slow_queries=slow_queries[:10],  # Top 10 slowest
                optimization_recommendations=recommendations,
                connection_metrics=self.pool_manager.metrics,
                performance_score=performance_score
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate optimization report: {e}")
            # Return empty report
            return OptimizationReport(
                report_id=f"opt_report_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                total_queries=0,
                avg_execution_time=0.0,
                cache_hit_ratio=0.0,
                slow_queries=[],
                optimization_recommendations=["Error generating report - check system health"],
                connection_metrics=self.pool_manager.metrics,
                performance_score=0.0
            )
    
    def _generate_optimization_recommendations(self, metrics_rows: List[Dict]) -> List[str]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []
        
        if not metrics_rows:
            return ["No query data available for recommendations"]
        
        # Analyze execution times
        slow_query_count = sum(1 for row in metrics_rows if row['execution_time'] > 1.0)
        if slow_query_count > len(metrics_rows) * 0.1:  # More than 10% slow queries
            recommendations.append("High number of slow queries detected - consider index optimization")
        
        # Analyze cache hit ratio
        cache_hits = sum(1 for row in metrics_rows if row['cache_hit'])
        cache_hit_ratio = (cache_hits / len(metrics_rows)) * 100
        if cache_hit_ratio < 50:
            recommendations.append("Low cache hit ratio - consider increasing cache TTL or cache size")
        
        # Query type analysis
        query_types = {}
        for row in metrics_rows:
            qt = row.get('query_type', 'UNKNOWN')
            query_types[qt] = query_types.get(qt, 0) + 1
        
        if query_types.get('SELECT', 0) > len(metrics_rows) * 0.8:
            recommendations.append("High SELECT query volume - consider read replicas for scaling")
        
        # Add index recommendations
        index_recommendations = self.query_optimizer.get_index_recommendations()
        if index_recommendations:
            recommendations.extend(index_recommendations[:5])  # Top 5 recommendations
        
        return recommendations
    
    def _calculate_performance_score(self, avg_execution_time: float, cache_hit_ratio: float, 
                                   slow_query_count: int, total_queries: int) -> float:
        """Calculate overall performance score (0-100)"""
        score = 100.0
        
        # Execution time penalty
        if avg_execution_time > 0.1:
            score -= min(30, avg_execution_time * 10)
        
        # Cache hit ratio bonus
        score += (cache_hit_ratio - 50) * 0.3
        
        # Slow query penalty
        if total_queries > 0:
            slow_query_ratio = slow_query_count / total_queries
            score -= slow_query_ratio * 20
        
        return max(0.0, min(100.0, score))
    
    async def optimize_table_indexes(self, table_name: str) -> List[str]:
        """Analyze and suggest index optimizations for a table"""
        try:
            # Get table statistics
            stats_sql = """
            SELECT 
                schemaname,
                tablename,
                attname,
                n_distinct,
                correlation,
                most_common_vals,
                most_common_freqs
            FROM pg_stats 
            WHERE tablename = $1
            """
            
            async with self.pool_manager.get_connection() as conn:
                stats = await conn.fetch(stats_sql, table_name)
            
            # Get current indexes
            indexes_sql = """
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = $1
            """
            
            async with self.pool_manager.get_connection() as conn:
                current_indexes = await conn.fetch(indexes_sql, table_name)
            
            recommendations = []
            
            # Analyze columns for index recommendations
            for stat in stats:
                column = stat['attname']
                n_distinct = stat['n_distinct']
                
                # High cardinality columns are good for indexes
                if n_distinct > 100:
                    existing_index = any(column in idx['indexdef'] for idx in current_indexes)
                    if not existing_index:
                        recommendations.append(
                            f"CREATE INDEX idx_{table_name}_{column} ON {table_name}({column})"
                        )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to analyze table indexes: {e}")
            return []
    
    async def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries"""
        try:
            if self.query_cache and self.query_cache.redis_client:
                # Clean Redis cache
                keys = self.query_cache.redis_client.keys("query_cache:*")
                expired_keys = []
                
                for key in keys:
                    ttl = self.query_cache.redis_client.ttl(key)
                    if ttl <= 0:
                        expired_keys.append(key)
                
                if expired_keys:
                    self.query_cache.redis_client.delete(*expired_keys)
                    logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
            
            # Clean memory cache
            if self.query_cache:
                current_time = datetime.now()
                expired_keys = [
                    key for key, value in self.query_cache.memory_cache.items()
                    if value['expires_at'] < current_time
                ]
                
                for key in expired_keys:
                    del self.query_cache.memory_cache[key]
                
                if expired_keys:
                    logger.info(f"Cleaned {len(expired_keys)} expired memory cache entries")
                    
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time database performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Database metrics
            connection_metrics = asdict(self.pool_manager.metrics)
            
            # Cache metrics
            cache_stats = self.query_cache.get_stats() if self.query_cache else {}
            
            # Recent query metrics
            recent_queries = len([q for q in self.query_metrics 
                                if q.timestamp > datetime.now() - timedelta(minutes=5)])
            
            avg_recent_execution = sum(q.execution_time for q in self.query_metrics[-100:]) / max(len(self.query_metrics[-100:]), 1)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available / 1024 / 1024
                },
                'database': {
                    **connection_metrics,
                    'recent_queries_5min': recent_queries,
                    'avg_recent_execution_time': avg_recent_execution
                },
                'cache': cache_stats,
                'performance_score': self._calculate_performance_score(
                    avg_recent_execution, 
                    cache_stats.get('hit_ratio', 0), 
                    0, 
                    recent_queries
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def close(self):
        """Close database optimizer and cleanup resources"""
        try:
            await self.pool_manager.close()
            logger.info("Analytics database optimizer closed successfully")
        except Exception as e:
            logger.error(f"Error closing database optimizer: {e}")

# Example usage and testing
async def main():
    """Example usage of the analytics database optimizer"""
    config = DatabaseConfig(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        database=os.getenv('DB_NAME', 'platform3_analytics'),
        user=os.getenv('DB_USER', 'platform3_user'),
        password=os.getenv('DB_PASSWORD', ''),
        enable_cache=True,
        enable_monitoring=True
    )
    
    optimizer = AnalyticsDatabaseOptimizer(config)
    
    try:
        # Initialize optimizer
        success = await optimizer.initialize()
        if not success:
            logger.error("Failed to initialize optimizer")
            return
        
        # Example queries
        test_queries = [
            "SELECT COUNT(*) FROM trading_metrics WHERE created_at > NOW() - INTERVAL '1 day'",
            "SELECT symbol, AVG(pnl) FROM trades GROUP BY symbol",
            "SELECT * FROM market_data WHERE timestamp > $1",
        ]
        
        # Execute test queries
        for query in test_queries:
            try:
                if "$1" in query:
                    result = await optimizer.execute_query(query, (datetime.now() - timedelta(hours=1),))
                else:
                    result = await optimizer.execute_query(query)
                logger.info(f"Query executed successfully: {len(result) if result else 0} rows")
            except Exception as e:
                logger.error(f"Query failed: {e}")
        
        # Generate optimization report
        report = await optimizer.get_optimization_report()
        logger.info(f"Optimization report generated: {report.performance_score:.1f} score")
        
        # Get real-time metrics
        metrics = await optimizer.get_real_time_metrics()
        logger.info(f"Real-time metrics: {metrics.get('performance_score', 'N/A')} performance score")
        
        # Clean up cache
        await optimizer.cleanup_cache()
        
    finally:
        await optimizer.close()

if __name__ == "__main__":
    asyncio.run(main())
class QueryPerformanceMetrics:
    """Query performance tracking"""
    query_id: str
    execution_time: float
    rows_processed: int
    cache_hit: bool
    optimization_applied: List[str]
    timestamp: datetime

class AnalyticsDatabaseOptimizer:
    """
    Database optimization layer for analytics queries
    Implements caching, query optimization, and performance monitoring
    """
    
    def __init__(self, database_manager: Platform3DatabaseManager):
        """Initialize the analytics database optimizer"""
        self.db_manager = database_manager
        self.query_cache = {}
        self.cache_timestamps = {}
        self.performance_metrics = []
        
        # Optimization strategies
        self.optimization_strategies = {
            "trading_data": self._optimize_trading_query,
            "market_data": self._optimize_market_query,
            "performance_metrics": self._optimize_performance_query,
            "analytics_history": self._optimize_analytics_query
        }
        
        logger.info("Analytics Database Optimizer initialized")

    async def execute_analytics_query(self, query: AnalyticsQuery) -> pd.DataFrame:
        """
        Execute optimized analytics query with caching and performance monitoring
        """
        start_time = datetime.utcnow()
        cache_key = self._generate_cache_key(query)
        
        try:
            # Check cache first
            cached_result = self._get_cached_result(cache_key, query.cache_ttl)
            if cached_result is not None:
                self._record_performance(query, start_time, len(cached_result), True, ["cache_hit"])
                logger.info(f"Cache hit for query {query.query_id}")
                return cached_result
            
            # Apply query optimization
            optimized_query, optimizations = await self._optimize_query(query)
            
            # Execute query
            result = await self._execute_query(optimized_query)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Record performance
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._record_performance(query, start_time, len(result), False, optimizations)
            
            logger.info(f"Query {query.query_id} executed in {execution_time:.3f}s, {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Error executing analytics query {query.query_id}: {e}")
            raise

    async def get_trading_data(self, 
                              symbol: str = None,
                              start_date: datetime = None,
                              end_date: datetime = None,
                              trade_type: str = None) -> pd.DataFrame:
        """Get optimized trading data for analytics"""
        
        # Set default timeframe if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        query = AnalyticsQuery(
            query_id=f"trading_data_{symbol}_{start_date.date()}_{end_date.date()}",
            query_type="trading_data",
            timeframe=f"{start_date.date()}_to_{end_date.date()}",
            filters={
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "trade_type": trade_type
            },
            optimization_hints=["use_index_symbol", "partition_by_date", "limit_columns"],
            cache_ttl=600  # 10 minutes for trading data
        )
        
        return await self.execute_analytics_query(query)

    async def get_market_data(self,
                             symbol: str,
                             timeframe: str = "1H",
                             start_date: datetime = None,
                             end_date: datetime = None) -> pd.DataFrame:
        """Get optimized market data for analytics"""
        
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        query = AnalyticsQuery(
            query_id=f"market_data_{symbol}_{timeframe}_{start_date.date()}",
            query_type="market_data",
            timeframe=timeframe,
            filters={
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date
            },
            optimization_hints=["use_time_index", "aggregate_by_timeframe", "compress_ohlcv"],
            cache_ttl=300  # 5 minutes for market data
        )
        
        return await self.execute_analytics_query(query)

    async def get_performance_metrics(self,
                                    metric_type: str = None,
                                    time_period: str = "1d") -> pd.DataFrame:
        """Get optimized performance metrics for analytics"""
        
        end_date = datetime.utcnow()
        if time_period == "1d":
            start_date = end_date - timedelta(days=1)
        elif time_period == "1w":
            start_date = end_date - timedelta(weeks=1)
        elif time_period == "1m":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=1)
        
        query = AnalyticsQuery(
            query_id=f"performance_metrics_{metric_type}_{time_period}",
            query_type="performance_metrics",
            timeframe=time_period,
            filters={
                "metric_type": metric_type,
                "start_date": start_date,
                "end_date": end_date
            },
            optimization_hints=["use_metric_index", "aggregate_by_period", "filter_active_only"],
            cache_ttl=180  # 3 minutes for performance metrics
        )
        
        return await self.execute_analytics_query(query)

    async def store_analytics_result(self, 
                                   result_type: str,
                                   data: Dict[str, Any],
                                   metadata: Dict[str, Any] = None) -> bool:
        """Store analytics result with optimization"""
        try:
            # Prepare optimized storage structure
            storage_data = {
                "result_type": result_type,
                "generated_at": datetime.utcnow(),
                "data": data,
                "metadata": metadata or {}
            }
            
            # Use optimized insert strategy
            await self._store_with_optimization(storage_data)
            
            logger.info(f"Analytics result {result_type} stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error storing analytics result {result_type}: {e}")
            return False

    def _generate_cache_key(self, query: AnalyticsQuery) -> str:
        """Generate cache key for query"""
        filter_str = "_".join(f"{k}:{v}" for k, v in sorted(query.filters.items()) if v is not None)
        return f"{query.query_type}_{query.timeframe}_{filter_str}"

    def _get_cached_result(self, cache_key: str, ttl: int) -> Optional[pd.DataFrame]:
        """Get cached result if valid"""
        if cache_key not in self.query_cache:
            return None
        
        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return None
        
        # Check TTL
        if (datetime.utcnow() - cache_time).total_seconds() > ttl:
            # Cache expired
            del self.query_cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None
        
        return self.query_cache[cache_key]

    def _cache_result(self, cache_key: str, result: pd.DataFrame):
        """Cache query result"""
        self.query_cache[cache_key] = result.copy()
        self.cache_timestamps[cache_key] = datetime.utcnow()
        
        # Limit cache size (keep last 100 queries)
        if len(self.query_cache) > 100:
            oldest_key = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
            del self.query_cache[oldest_key]
            del self.cache_timestamps[oldest_key]

    async def _optimize_query(self, query: AnalyticsQuery) -> tuple[Dict[str, Any], List[str]]:
        """Apply query optimization based on type and hints"""
        optimizer = self.optimization_strategies.get(query.query_type)
        if optimizer:
            return await optimizer(query)
        else:
            # Default optimization
            return query.filters, ["default_optimization"]

    async def _optimize_trading_query(self, query: AnalyticsQuery) -> tuple[Dict[str, Any], List[str]]:
        """Optimize trading data query"""
        optimizations = []
        optimized_filters = query.filters.copy()
        
        # Add index hints
        if "use_index_symbol" in query.optimization_hints:
            optimized_filters["index_hint"] = "symbol_date_idx"
            optimizations.append("symbol_index")
        
        # Add date partitioning
        if "partition_by_date" in query.optimization_hints:
            optimized_filters["partition_strategy"] = "date_range"
            optimizations.append("date_partition")
        
        # Limit columns for performance
        if "limit_columns" in query.optimization_hints:
            optimized_filters["select_columns"] = [
                "symbol", "entry_time", "exit_time", "entry_price", 
                "exit_price", "quantity", "pnl", "trade_type"
            ]
            optimizations.append("column_limit")
        
        return optimized_filters, optimizations

    async def _optimize_market_query(self, query: AnalyticsQuery) -> tuple[Dict[str, Any], List[str]]:
        """Optimize market data query"""
        optimizations = []
        optimized_filters = query.filters.copy()
        
        # Time-based indexing
        if "use_time_index" in query.optimization_hints:
            optimized_filters["index_hint"] = "time_symbol_idx"
            optimizations.append("time_index")
        
        # Timeframe aggregation
        if "aggregate_by_timeframe" in query.optimization_hints:
            optimized_filters["aggregation"] = query.filters.get("timeframe", "1H")
            optimizations.append("timeframe_agg")
        
        # OHLCV compression
        if "compress_ohlcv" in query.optimization_hints:
            optimized_filters["compression"] = "ohlcv_only"
            optimizations.append("ohlcv_compress")
        
        return optimized_filters, optimizations

    async def _optimize_performance_query(self, query: AnalyticsQuery) -> tuple[Dict[str, Any], List[str]]:
        """Optimize performance metrics query"""
        optimizations = []
        optimized_filters = query.filters.copy()
        
        # Metric-specific indexing
        if "use_metric_index" in query.optimization_hints:
            optimized_filters["index_hint"] = "metric_type_time_idx"
            optimizations.append("metric_index")
        
        # Period aggregation
        if "aggregate_by_period" in query.optimization_hints:
            optimized_filters["aggregation_period"] = query.timeframe
            optimizations.append("period_agg")
        
        # Filter active metrics only
        if "filter_active_only" in query.optimization_hints:
            optimized_filters["active_only"] = True
            optimizations.append("active_filter")
        
        return optimized_filters, optimizations

    async def _optimize_analytics_query(self, query: AnalyticsQuery) -> tuple[Dict[str, Any], List[str]]:
        """Optimize analytics history query"""
        optimizations = []
        optimized_filters = query.filters.copy()
        
        # Add analytics-specific optimizations
        optimized_filters["result_compression"] = True
        optimizations.append("result_compress")
        
        return optimized_filters, optimizations

    async def _execute_query(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Execute the actual database query"""
        # This would integrate with the actual database
        # For now, return sample data structure
        
        sample_data = {
            "timestamp": [datetime.utcnow() - timedelta(hours=i) for i in range(24)],
            "symbol": ["EURUSD"] * 24,
            "value": [1.1000 + i * 0.001 for i in range(24)],
            "volume": [100000 + i * 1000 for i in range(24)]
        }
        
        return pd.DataFrame(sample_data)

    async def _store_with_optimization(self, data: Dict[str, Any]):
        """Store data with optimization strategies"""
        # Implement optimized storage logic
        # This would integrate with Platform3DatabaseManager
        logger.info(f"Storing {data['result_type']} with optimization")

    def _record_performance(self, 
                          query: AnalyticsQuery, 
                          start_time: datetime, 
                          rows_processed: int,
                          cache_hit: bool, 
                          optimizations: List[str]):
        """Record query performance metrics"""
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        metric = QueryPerformanceMetrics(
            query_id=query.query_id,
            execution_time=execution_time,
            rows_processed=rows_processed,
            cache_hit=cache_hit,
            optimization_applied=optimizations,
            timestamp=datetime.utcnow()
        )
        
        self.performance_metrics.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate database performance report"""
        if not self.performance_metrics:
            return {"message": "No performance data available"}
        
        # Calculate aggregate metrics
        total_queries = len(self.performance_metrics)
        cache_hits = sum(1 for m in self.performance_metrics if m.cache_hit)
        avg_execution_time = sum(m.execution_time for m in self.performance_metrics) / total_queries
        total_rows_processed = sum(m.rows_processed for m in self.performance_metrics)
        
        # Most common optimizations
        optimization_counts = {}
        for metric in self.performance_metrics:
            for opt in metric.optimization_applied:
                optimization_counts[opt] = optimization_counts.get(opt, 0) + 1
        
        return {
            "total_queries": total_queries,
            "cache_hit_rate": (cache_hits / total_queries) * 100,
            "average_execution_time": avg_execution_time,
            "total_rows_processed": total_rows_processed,
            "most_used_optimizations": sorted(optimization_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "performance_trend": "stable",  # Could implement trend analysis
            "generated_at": datetime.utcnow().isoformat()
        }

# Integration with Advanced Analytics Framework
class AnalyticsFrameworkDatabaseIntegration:
    """Integration layer between Analytics Framework and Database Optimizer"""
    
    def __init__(self, analytics_framework, database_optimizer: AnalyticsDatabaseOptimizer):
        """Initialize the integration layer"""
        self.analytics_framework = analytics_framework
        self.db_optimizer = database_optimizer
        
        logger.info("Analytics Framework Database Integration initialized")

    async def enhance_analytics_with_database(self):
        """Enhance analytics framework with database optimization"""
        
        # Inject database optimizer into analytics engines
        for engine_name, engine in self.analytics_framework.engines.items():
            if hasattr(engine, 'set_database_optimizer'):
                engine.set_database_optimizer(self.db_optimizer)
                logger.info(f"Database optimizer injected into {engine_name}")
        
        # Set up automatic data collection for analytics
        await self._setup_automatic_data_collection()
        
        logger.info("Analytics framework enhanced with database optimization")

    async def _setup_automatic_data_collection(self):
        """Set up automatic data collection for analytics"""
        # This would set up scheduled tasks to collect data
        logger.info("Automatic data collection configured")

# Usage example
async def main():
    """Example usage of database optimization integration"""
    
    # Initialize database manager (would be actual Platform3DatabaseManager)
    db_manager = Platform3DatabaseManager()
    
    # Initialize database optimizer
    db_optimizer = AnalyticsDatabaseOptimizer(db_manager)
    
    # Test optimized queries
    trading_data = await db_optimizer.get_trading_data(
        symbol="EURUSD",
        start_date=datetime.utcnow() - timedelta(days=7)
    )
    
    print(f"Retrieved {len(trading_data)} trading records")
    
    # Get performance report
    performance_report = db_optimizer.get_performance_report()
    print(f"Database performance: {performance_report}")

if __name__ == "__main__":
    asyncio.run(main())
