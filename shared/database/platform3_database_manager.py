#!/usr/bin/env python3
"""
Platform3 Database Transaction and Performance Optimization Framework
Comprehensive database optimization following Platform3 patterns with connection pooling,
transaction management, query optimization, and health monitoring.
"""

import os
import sys
import asyncio
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from pathlib import Path
import logging

# Add shared modules to path
from shared.logging.platform3_logger import Platform3Logger, LogMetadata, get_logger
from shared.error_handling.platform3_error_system import BaseService, ServiceError, DatabaseError

try:
    import psycopg2
    from psycopg2 import pool, sql, extras
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "platform3"
    user: str = "postgres"
    password: str = "password"
    
    # Connection pool settings
    min_connections: int = 2
    max_connections: int = 20
    idle_timeout: int = 30000  # milliseconds
    connection_timeout: int = 2000  # milliseconds
    
    # Performance settings
    enable_ssl: bool = False
    enable_prepared_statements: bool = True
    enable_query_cache: bool = True
    cache_ttl: int = 300  # seconds
    
    # Health check settings
    health_check_interval: int = 30  # seconds
    max_retry_attempts: int = 3
    retry_delay: int = 1  # seconds


@dataclass
class QueryCacheEntry:
    """Query cache entry with metadata"""
    result: Any
    timestamp: float
    query_hash: str
    execution_time: float
    hit_count: int = 0


@dataclass
class PerformanceMetrics:
    """Database performance metrics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_query_time: float = 0.0
    max_query_time: float = 0.0
    min_query_time: float = float('inf')
    active_connections: int = 0
    total_connections_created: int = 0
    
    def add_query_time(self, execution_time: float):
        """Add query execution time to metrics"""
        self.total_queries += 1
        self.avg_query_time = ((self.avg_query_time * (self.total_queries - 1)) + execution_time) / self.total_queries
        self.max_query_time = max(self.max_query_time, execution_time)
        self.min_query_time = min(self.min_query_time, execution_time)


class Platform3DatabaseManager(BaseService):
    """
    Comprehensive database manager for Platform3 with transaction management,
    connection pooling, query optimization, and performance monitoring.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        super().__init__(service_name="database_manager")
        
        self.config = config or DatabaseConfig()
        self.logger = get_logger("platform3.database")
        
        # Connection pool
        self.pool: Optional[pool.ThreadedConnectionPool] = None
        self.is_connected: bool = False
        
        # Query cache
        self.query_cache: Dict[str, QueryCacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.metrics_lock = threading.RLock()
        
        # Health monitoring
        self._health_check_task: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
        
        # Prepared statements cache
        self.prepared_statements: Dict[str, str] = {}
    
    async def initialize(self) -> bool:
        """Initialize database connection pool and health monitoring"""
        try:
            if not HAS_PSYCOPG2:
                self.logger.warning("psycopg2 not available, using fallback database manager")
                return await self._initialize_fallback()
            
            # Create connection pool
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                connect_timeout=self.config.connection_timeout // 1000,
                options=f"-c idle_in_transaction_session_timeout={self.config.idle_timeout}"
            )
            
            # Test connection
            await self._test_connection()
            self.is_connected = True
            
            # Start health monitoring            self._start_health_monitoring()
            
            self.logger.info(
                "Database manager initialized successfully",
                meta={
                    "operation": "database_initialize",
                    "min_connections": self.config.min_connections,
                    "max_connections": self.config.max_connections
                }
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize database manager: {str(e)}"
            self.logger.error(error_msg, meta={
                "error_type": "database_initialization_error",
                "error_details": str(e)
            })
            
            self.emit_error(DatabaseError(
                message=error_msg,
                error_code="DB_INIT_ERROR",
                service_context="database_manager"
            ))
            
            return False
    
    async def _initialize_fallback(self) -> bool:
        """Initialize fallback database manager (SQLite)"""
        if not HAS_SQLITE:
            self.logger.error("No database backend available")
            return False
        
        self.logger.info("Using SQLite fallback database")
        self.is_connected = True
        return True
    
    async def _test_connection(self):
        """Test database connection"""
        conn = None
        try:
            conn = self.pool.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 as health_check")
            result = cursor.fetchone()
            cursor.close()
            
            if result[0] != 1:
                raise DatabaseError("Health check query failed")
                
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        if self._health_check_task and self._health_check_task.is_alive():
            return
        
        self._health_check_task = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_task.start()
    
    def _health_check_loop(self):
        """Background health check loop"""
        while not self._stop_health_check.wait(self.config.health_check_interval):
            try:
                asyncio.run(self._perform_health_check())
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
    
    async def _perform_health_check(self):
        """Perform database health check"""
        try:
            start_time = time.time()
            
            # Test basic connectivity
            is_healthy = await self.health_check()
            
            execution_time = time.time() - start_time
            
            if is_healthy:                self.logger.debug(
                    "Database health check passed",
                    meta={
                        "operation": "health_check",
                        "execution_time": execution_time,
                        "status": "healthy"
                    }
                )
            else:
                self.logger.warning("Database health check failed")
                
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        conn = None
        try:
            if not self.pool:
                raise DatabaseError("Database pool not initialized")
            
            conn = self.pool.getconn()
            with self.metrics_lock:
                self.metrics.active_connections += 1
                self.metrics.total_connections_created += 1
            
            yield conn
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            raise DatabaseError(f"Failed to get database connection: {str(e)}")
        finally:
            if conn:
                self.pool.putconn(conn)
                with self.metrics_lock:
                    self.metrics.active_connections -= 1
    
    @asynccontextmanager
    async def transaction(self, isolation_level: Optional[str] = None):
        """Async transaction context manager with automatic rollback"""
        with self.get_connection() as conn:
            original_autocommit = conn.autocommit
            cursor = None
            
            try:
                # Set isolation level if specified
                if isolation_level:
                    conn.set_isolation_level(getattr(psycopg2.extensions, f"ISOLATION_LEVEL_{isolation_level.upper()}"))
                
                conn.autocommit = False
                cursor = conn.cursor()
                
                self.logger.debug(
                    "Transaction started",
                    meta={
                        "operation": "transaction_start",
                        "isolation_level": isolation_level or "default"
                    }
                )
                
                yield cursor
                
                conn.commit()
                self.logger.debug(
                    "Transaction committed",
                    meta={
                        "operation": "transaction_commit"
                    }
                )
                
            except Exception as e:
                if conn:
                    conn.rollback()
                self.logger.error(
                    f"Transaction rolled back due to error: {e}",
                    meta={
                        "error_type": "transaction_rollback",
                        "error_details": str(e)
                    }
                )
                
                raise DatabaseError(f"Transaction failed: {str(e)}")
                
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.autocommit = original_autocommit
    
    def _get_query_hash(self, query: str, params: Optional[Tuple] = None) -> str:
        """Generate hash for query caching"""
        import hashlib
        query_str = f"{query}:{params or ''}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _get_cached_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result"""
        with self.cache_lock:
            entry = self.query_cache.get(query_hash)
            if entry and time.time() - entry.timestamp < self.config.cache_ttl:
                entry.hit_count += 1
                with self.metrics_lock:
                    self.metrics.cache_hits += 1
                return entry.result
            
            # Remove expired entries
            if entry:
                del self.query_cache[query_hash]
            
            with self.metrics_lock:
                self.metrics.cache_misses += 1
            
            return None
    
    def _cache_result(self, query_hash: str, result: Any, execution_time: float):
        """Cache query result"""
        if not self.config.enable_query_cache:
            return
        
        with self.cache_lock:
            # Limit cache size (simple LRU)
            if len(self.query_cache) > 1000:
                oldest_key = min(self.query_cache.keys(), key=lambda k: self.query_cache[k].timestamp)
                del self.query_cache[oldest_key]
            
            self.query_cache[query_hash] = QueryCacheEntry(
                result=result,
                timestamp=time.time(),
                query_hash=query_hash,
                execution_time=execution_time
            )
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch_mode: str = "all",  # "all", "one", "none"
        use_cache: bool = True
    ) -> Any:
        """Execute database query with caching and performance monitoring"""
        start_time = time.time()
        query_hash = self._get_query_hash(query, params) if use_cache else None
        
        try:
            # Check cache first
            if use_cache and query_hash:
                cached_result = self._get_cached_result(query_hash)
                if cached_result is not None:
                    return cached_result
            
            # Execute query
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
                
                cursor.execute(query, params)
                
                if fetch_mode == "all":
                    result = cursor.fetchall()
                elif fetch_mode == "one":
                    result = cursor.fetchone()
                else:  # "none"
                    result = cursor.rowcount
                
                cursor.close()
            
            execution_time = time.time() - start_time
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.successful_queries += 1
                self.metrics.add_query_time(execution_time)
            
            # Cache result
            if use_cache and query_hash and fetch_mode != "none":
                self._cache_result(query_hash, result, execution_time)
            
            self.logger.debug(                f"Query executed successfully",
                meta={
                    "operation": "database_query",
                    "execution_time": execution_time,
                    "rows_affected": len(result) if isinstance(result, list) else 1,
                    "cache_hit": False
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            with self.metrics_lock:
                self.metrics.failed_queries += 1
                self.metrics.add_query_time(execution_time)
            
            error_msg = f"Query execution failed: {str(e)}"
            self.logger.error(
                error_msg,
                meta={
                    "error_type": "query_execution_error",
                    "error_details": str(e),
                    "query": query[:100] + "..." if len(query) > 100 else query
                }
            )
            
            self.emit_error(DatabaseError(
                message=error_msg,
                error_code="QUERY_EXECUTION_ERROR",
                service_context="database_manager"
            ))
            
            raise DatabaseError(error_msg)
    
    async def execute_prepared_statement(
        self,
        statement_name: str,
        query: str,
        params: Optional[Tuple] = None
    ) -> Any:
        """Execute prepared statement for better performance"""
        if not self.config.enable_prepared_statements:
            return await self.execute_query(query, params)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare statement if not already prepared
                if statement_name not in self.prepared_statements:
                    cursor.execute(f"PREPARE {statement_name} AS {query}")
                    self.prepared_statements[statement_name] = query
                
                # Execute prepared statement
                if params:
                    cursor.execute(f"EXECUTE {statement_name} ({','.join(['%s'] * len(params))})", params)
                else:
                    cursor.execute(f"EXECUTE {statement_name}")
                
                result = cursor.fetchall()
                cursor.close()
                
                return result
                
        except Exception as e:
            error_msg = f"Prepared statement execution failed: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg)
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            result = await self.execute_query("SELECT 1 as health_check", use_cache=False)
            return len(result) > 0 and result[0]['health_check'] == 1
        except Exception:
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self.metrics_lock:
            return {
                "total_queries": self.metrics.total_queries,
                "successful_queries": self.metrics.successful_queries,
                "failed_queries": self.metrics.failed_queries,
                "success_rate": (self.metrics.successful_queries / max(self.metrics.total_queries, 1)) * 100,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "cache_hit_rate": (self.metrics.cache_hits / max(self.metrics.cache_hits + self.metrics.cache_misses, 1)) * 100,
                "avg_query_time": self.metrics.avg_query_time,
                "max_query_time": self.metrics.max_query_time,
                "min_query_time": self.metrics.min_query_time if self.metrics.min_query_time != float('inf') else 0,
                "active_connections": self.metrics.active_connections,
                "total_connections_created": self.metrics.total_connections_created,
                "query_cache_size": len(self.query_cache),
                "prepared_statements_count": len(self.prepared_statements)
            }
    
    def clear_cache(self):
        """Clear query cache"""
        with self.cache_lock:
            self.query_cache.clear()
            self.logger.info("Query cache cleared")
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            # Stop health monitoring
            self._stop_health_check.set()
            if self._health_check_task:
                self._health_check_task.join(timeout=5)
            
            # Close connection pool
            if self.pool:
                self.pool.closeall()
            
            self.is_connected = False
            self.logger.info(
                "Database manager shutdown completed",
                meta={
                    "operation": "database_shutdown",
                    "final_metrics": self.get_metrics()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global database manager instance
_db_manager: Optional[Platform3DatabaseManager] = None


def get_database_manager(config: Optional[DatabaseConfig] = None) -> Platform3DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = Platform3DatabaseManager(config)
    return _db_manager


# Decorator for database operations
def with_database(use_transaction: bool = False, isolation_level: Optional[str] = None):
    """Decorator for database operations with automatic transaction management"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            db = get_database_manager()
            
            if not db.is_connected:
                await db.initialize()
            
            if use_transaction:
                async with db.transaction(isolation_level=isolation_level) as cursor:
                    return await func(*args, cursor=cursor, **kwargs)
            else:
                return await func(*args, db=db, **kwargs)
        
        return wrapper
    return decorator


# Example usage functions
@with_database(use_transaction=True)
async def create_trade_record(trade_data: Dict[str, Any], cursor) -> bool:
    """Example: Create trade record with transaction"""
    try:
        query = """
        INSERT INTO trades (user_id, symbol, quantity, price, side, timestamp)
        VALUES (%(user_id)s, %(symbol)s, %(quantity)s, %(price)s, %(side)s, %(timestamp)s)
        RETURNING id
        """
        
        cursor.execute(query, trade_data)
        result = cursor.fetchone()
        
        return result is not None
        
    except Exception as e:
        raise DatabaseError(f"Failed to create trade record: {str(e)}")


@with_database()
async def get_user_positions(user_id: str, db) -> List[Dict[str, Any]]:
    """Example: Get user positions with caching"""
    query = """
    SELECT symbol, quantity, avg_price, unrealized_pnl
    FROM positions
    WHERE user_id = %s AND status = 'open'
    ORDER BY symbol
    """
    
    return await db.execute_query(query, (user_id,), fetch_mode="all", use_cache=True)


if __name__ == "__main__":
    # Example usage
    async def main():
        config = DatabaseConfig(
            host="localhost",
            database="platform3_test",
            max_connections=10,
            enable_query_cache=True
        )
        
        db = Platform3DatabaseManager(config)
        await db.initialize()
        
        # Test health check
        health = await db.health_check()
        print(f"Database health: {health}")
        
        # Get metrics
        metrics = db.get_metrics()
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        
        await db.shutdown()
    
    asyncio.run(main())
