#!/usr/bin/env python3
"""
Platform3 Agent State Persistence and Recovery Manager
Comprehensive state management for intelligent agents with PostgreSQL/Redis integration
"""

import os
import sys
import json
import pickle
import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from enum import Enum
import logging
import uuid

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from shared.platform3_logging.platform3_logger import Platform3Logger, get_logger
from shared.error_handling.platform3_error_system import BaseService, ServiceError, DatabaseError
from shared.database.platform3_database_manager import Platform3DatabaseManager, DatabaseConfig, with_database

try:
    import redis
    import redis.sentinel
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class StateType(Enum):
    """Types of agent state data"""
    CONFIGURATION = "configuration"
    LEARNING_MODEL = "learning_model"
    DECISION_HISTORY = "decision_history"
    PERFORMANCE_METRICS = "performance_metrics"
    COMMUNICATION_STATE = "communication_state"
    DEPENDENCY_STATE = "dependency_state"
    CHECKPOINT = "checkpoint"
    FULL_STATE = "full_state"


class StateFormat(Enum):
    """State serialization formats"""
    JSON = "json"
    PICKLE = "pickle"
    BINARY = "binary"
    COMPRESSED = "compressed"


@dataclass
class StateMetadata:
    """Metadata for stored state"""
    state_id: str
    agent_name: str
    state_type: StateType
    format: StateFormat
    timestamp: datetime
    version: str
    checksum: str
    size_bytes: int
    compression_ratio: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    expiry_date: Optional[datetime] = None


@dataclass
class StateSnapshot:
    """Complete state snapshot of an agent"""
    agent_name: str
    timestamp: datetime
    configuration: Dict[str, Any]
    learning_models: Dict[str, Any]
    decision_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    communication_state: Dict[str, Any]
    dependency_relationships: Dict[str, Any]
    version: str
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    

@dataclass
class RecoveryPlan:
    """Recovery plan for agent state restoration"""
    agent_name: str
    target_state: StateSnapshot
    recovery_steps: List[Dict[str, Any]]
    dependencies_to_restore: List[str]
    estimated_duration: float
    validation_checks: List[str]
    rollback_points: List[str]


class AgentStateManager(BaseService):
    """
    Comprehensive agent state persistence and recovery manager
    Integrates with PostgreSQL for metadata and Redis for fast access
    """
    
    def __init__(self, 
                 db_config: Optional[DatabaseConfig] = None,
                 redis_config: Optional[Dict[str, Any]] = None):
        super().__init__(service_name="agent_state_manager")
        
        self.logger = get_logger("platform3.agent_state")
        self.db_manager = Platform3DatabaseManager(db_config)
        
        # Redis configuration
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 1,  # Use db 1 for agent states
            'decode_responses': False,  # Keep binary for complex objects
            'socket_timeout': 5,
            'socket_connect_timeout': 5,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
        
        self.redis_client: Optional[redis.Redis] = None
        self.is_initialized = False
        
        # State management settings
        self.state_retention_days = 30
        self.snapshot_interval_hours = 24
        self.max_state_versions = 10
        self.compression_threshold_kb = 100
        
        # Performance tracking
        self.operation_metrics = {
            'saves': 0,
            'loads': 0,
            'failures': 0,
            'avg_save_time': 0.0,
            'avg_load_time': 0.0
        }
        
        # State caching
        self.state_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 100
    
    async def initialize(self) -> bool:
        """Initialize the state manager with database and Redis connections"""
        try:
            # Initialize database manager
            db_success = await self.db_manager.initialize()
            if not db_success:
                self.logger.error("Failed to initialize database manager")
                return False
            
            # Initialize Redis connection
            if HAS_REDIS:
                self.redis_client = redis.Redis(**self.redis_config)
                
                # Test Redis connection
                await self._test_redis_connection()
                self.logger.info("Redis connection established successfully")
            else:
                self.logger.warning("Redis not available, using database-only persistence")
            
            # Create required database tables
            await self._create_database_schema()
            
            # Start background tasks
            asyncio.create_task(self._background_maintenance())
            
            self.is_initialized = True
            self.logger.info("Agent State Manager initialized successfully")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize Agent State Manager: {str(e)}"
            self.logger.error(error_msg)
            self.emit_error(ServiceError(
                message=error_msg,
                error_code="STATE_INIT_ERROR",
                service_context="agent_state_manager"
            ))
            return False
    
    async def _test_redis_connection(self):
        """Test Redis connection"""
        if not self.redis_client:
            return
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
        except Exception as e:
            raise RuntimeError(f"Redis connection test failed: {e}")
    
    async def _create_database_schema(self):
        """Create required database tables for state management"""
        schema_sql = """
        -- Agent state metadata table
        CREATE TABLE IF NOT EXISTS agent_states (
            state_id VARCHAR(255) PRIMARY KEY,
            agent_name VARCHAR(100) NOT NULL,
            state_type VARCHAR(50) NOT NULL,
            format VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            version VARCHAR(50) NOT NULL,
            checksum VARCHAR(64) NOT NULL,
            size_bytes BIGINT NOT NULL,
            compression_ratio FLOAT,
            dependencies JSONB,
            tags JSONB,
            expiry_date TIMESTAMP,
            data_location VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Agent snapshots table
        CREATE TABLE IF NOT EXISTS agent_snapshots (
            snapshot_id VARCHAR(255) PRIMARY KEY,
            agent_name VARCHAR(100) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            version VARCHAR(50) NOT NULL,
            state_ids JSONB NOT NULL,
            snapshot_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Agent recovery logs table
        CREATE TABLE IF NOT EXISTS agent_recovery_logs (
            log_id SERIAL PRIMARY KEY,
            agent_name VARCHAR(100) NOT NULL,
            recovery_type VARCHAR(50) NOT NULL,
            source_snapshot_id VARCHAR(255),
            target_state VARCHAR(255),
            status VARCHAR(20) NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_ms BIGINT,
            steps_completed INTEGER,
            steps_total INTEGER,
            error_message TEXT,
            recovery_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_agent_states_agent_name ON agent_states(agent_name);
        CREATE INDEX IF NOT EXISTS idx_agent_states_timestamp ON agent_states(timestamp);
        CREATE INDEX IF NOT EXISTS idx_agent_states_type ON agent_states(state_type);
        CREATE INDEX IF NOT EXISTS idx_agent_snapshots_agent_name ON agent_snapshots(agent_name);
        CREATE INDEX IF NOT EXISTS idx_agent_snapshots_timestamp ON agent_snapshots(timestamp);
        CREATE INDEX IF NOT EXISTS idx_recovery_logs_agent_name ON agent_recovery_logs(agent_name);
        CREATE INDEX IF NOT EXISTS idx_recovery_logs_timestamp ON agent_recovery_logs(start_time);
        """
        
        try:
            await self.db_manager.execute_query(schema_sql, fetch_mode="none")
            self.logger.info("Database schema created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create database schema: {e}")
            raise
    
    def _generate_state_id(self, agent_name: str, state_type: StateType) -> str:
        """Generate unique state ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:8]
        return f"{agent_name}_{state_type.value}_{timestamp}_{unique_suffix}"
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum for data integrity"""
        return hashlib.sha256(data).hexdigest()
    
    def _serialize_state(self, data: Any, format: StateFormat) -> bytes:
        """Serialize state data in specified format"""
        if format == StateFormat.JSON:
            return json.dumps(data, default=str).encode('utf-8')
        elif format == StateFormat.PICKLE:
            return pickle.dumps(data)
        elif format == StateFormat.BINARY:
            if isinstance(data, bytes):
                return data
            else:
                return pickle.dumps(data)
        elif format == StateFormat.COMPRESSED:
            import gzip
            serialized = pickle.dumps(data)
            return gzip.compress(serialized)
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    
    def _deserialize_state(self, data: bytes, format: StateFormat) -> Any:
        """Deserialize state data from specified format"""
        if format == StateFormat.JSON:
            return json.loads(data.decode('utf-8'))
        elif format == StateFormat.PICKLE:
            return pickle.loads(data)
        elif format == StateFormat.BINARY:
            return pickle.loads(data)
        elif format == StateFormat.COMPRESSED:
            import gzip
            decompressed = gzip.decompress(data)
            return pickle.loads(decompressed)
        else:
            raise ValueError(f"Unsupported deserialization format: {format}")
    
    async def save_agent_state(self,
                              agent_name: str,
                              state_type: StateType,
                              state_data: Any,
                              version: str = "1.0",
                              tags: Optional[List[str]] = None,
                              expiry_hours: Optional[int] = None) -> str:
        """Save agent state data with metadata"""
        start_time = time.time()
        
        try:
            # Determine optimal serialization format
            format = StateFormat.JSON if isinstance(state_data, (dict, list, str, int, float, bool)) else StateFormat.PICKLE
            
            # Check if compression is beneficial
            serialized_data = self._serialize_state(state_data, format)
            if len(serialized_data) > self.compression_threshold_kb * 1024:
                format = StateFormat.COMPRESSED
                serialized_data = self._serialize_state(state_data, format)
            
            # Generate metadata
            state_id = self._generate_state_id(agent_name, state_type)
            checksum = self._calculate_checksum(serialized_data)
            size_bytes = len(serialized_data)
            timestamp = datetime.now()
            expiry_date = timestamp + timedelta(hours=expiry_hours) if expiry_hours else None
            
            metadata = StateMetadata(
                state_id=state_id,
                agent_name=agent_name,
                state_type=state_type,
                format=format,
                timestamp=timestamp,
                version=version,
                checksum=checksum,
                size_bytes=size_bytes,
                tags=tags or [],
                expiry_date=expiry_date
            )
            
            # Save to Redis for fast access (if available)
            redis_key = f"agent_state:{agent_name}:{state_type.value}:{state_id}"
            if self.redis_client:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.redis_client.setex,
                        redis_key,
                        self.cache_ttl,
                        serialized_data
                    )
                    data_location = f"redis:{redis_key}"
                except Exception as e:
                    self.logger.warning(f"Failed to save to Redis: {e}, falling back to database")
                    data_location = "database"
            else:
                data_location = "database"
            
            # Save metadata to PostgreSQL
            metadata_dict = asdict(metadata)
            metadata_dict['timestamp'] = metadata_dict['timestamp'].isoformat()
            if metadata_dict['expiry_date']:
                metadata_dict['expiry_date'] = metadata_dict['expiry_date'].isoformat()
            metadata_dict['state_type'] = metadata_dict['state_type'].value
            metadata_dict['format'] = metadata_dict['format'].value
            metadata_dict['data_location'] = data_location
            
            # If not in Redis, store in database as well
            if data_location == "database":
                # Store large data as base64 in JSONB
                import base64
                metadata_dict['data_blob'] = base64.b64encode(serialized_data).decode('ascii')
            
            query = """
            INSERT INTO agent_states (
                state_id, agent_name, state_type, format, timestamp, version,
                checksum, size_bytes, compression_ratio, dependencies, tags,
                expiry_date, data_location
            ) VALUES (
                %(state_id)s, %(agent_name)s, %(state_type)s, %(format)s, %(timestamp)s,
                %(version)s, %(checksum)s, %(size_bytes)s, %(compression_ratio)s,
                %(dependencies)s, %(tags)s, %(expiry_date)s, %(data_location)s
            )
            """
            
            await self.db_manager.execute_query(query, tuple(metadata_dict.values()), fetch_mode="none")
            
            # Update cache
            self.state_cache[f"{agent_name}:{state_type.value}"] = {
                'data': state_data,
                'metadata': metadata,
                'cached_at': time.time()
            }
            
            # Update metrics
            duration = time.time() - start_time
            self.operation_metrics['saves'] += 1
            self.operation_metrics['avg_save_time'] = (
                (self.operation_metrics['avg_save_time'] * (self.operation_metrics['saves'] - 1) + duration)
                / self.operation_metrics['saves']
            )
            
            self.logger.info(
                f"Successfully saved agent state",
                meta={
                    "agent_name": agent_name,
                    "state_type": state_type.value,
                    "state_id": state_id,
                    "size_bytes": size_bytes,
                    "duration_ms": duration * 1000,
                    "location": data_location
                }
            )
            
            return state_id
            
        except Exception as e:
            self.operation_metrics['failures'] += 1
            error_msg = f"Failed to save agent state: {str(e)}"
            self.logger.error(error_msg, meta={
                "agent_name": agent_name,
                "state_type": state_type.value,
                "error": str(e)
            })
            raise ServiceError(error_msg, "STATE_SAVE_ERROR", "agent_state_manager")
    
    async def load_agent_state(self,
                              agent_name: str,
                              state_type: StateType,
                              state_id: Optional[str] = None) -> Optional[Tuple[Any, StateMetadata]]:
        """Load agent state data, optionally by specific state ID"""
        start_time = time.time()
        cache_key = f"{agent_name}:{state_type.value}"
        
        try:
            # Check cache first
            if not state_id and cache_key in self.state_cache:
                cached = self.state_cache[cache_key]
                if time.time() - cached['cached_at'] < self.cache_ttl:
                    return cached['data'], cached['metadata']
            
            # Query database for metadata
            if state_id:
                query = """
                SELECT * FROM agent_states 
                WHERE state_id = %s AND agent_name = %s
                """
                params = (state_id, agent_name)
            else:
                # Get latest state
                query = """
                SELECT * FROM agent_states 
                WHERE agent_name = %s AND state_type = %s
                ORDER BY timestamp DESC
                LIMIT 1
                """
                params = (agent_name, state_type.value)
            
            results = await self.db_manager.execute_query(query, params, fetch_mode="all")
            
            if not results:
                return None
            
            metadata_row = results[0]
            
            # Reconstruct metadata
            metadata = StateMetadata(
                state_id=metadata_row['state_id'],
                agent_name=metadata_row['agent_name'],
                state_type=StateType(metadata_row['state_type']),
                format=StateFormat(metadata_row['format']),
                timestamp=metadata_row['timestamp'],
                version=metadata_row['version'],
                checksum=metadata_row['checksum'],
                size_bytes=metadata_row['size_bytes'],
                compression_ratio=metadata_row.get('compression_ratio'),
                dependencies=metadata_row.get('dependencies', []),
                tags=metadata_row.get('tags', []),
                expiry_date=metadata_row.get('expiry_date')
            )
            
            # Load data
            data_location = metadata_row['data_location']
            
            if data_location.startswith('redis:') and self.redis_client:
                # Load from Redis
                redis_key = data_location[6:]  # Remove 'redis:' prefix
                try:
                    serialized_data = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.get, redis_key
                    )
                    if serialized_data is None:
                        raise ValueError("Data not found in Redis")
                except Exception as e:
                    self.logger.warning(f"Failed to load from Redis: {e}, falling back to database")
                    # Fall back to database if available
                    if 'data_blob' in metadata_row:
                        import base64
                        serialized_data = base64.b64decode(metadata_row['data_blob'])
                    else:
                        raise ValueError("Data not available in Redis or database")
            else:
                # Load from database
                if 'data_blob' in metadata_row:
                    import base64
                    serialized_data = base64.b64decode(metadata_row['data_blob'])
                else:
                    raise ValueError("Data not found in database")
            
            # Verify data integrity
            calculated_checksum = self._calculate_checksum(serialized_data)
            if calculated_checksum != metadata.checksum:
                raise ValueError("Data integrity check failed - checksum mismatch")
            
            # Deserialize data
            state_data = self._deserialize_state(serialized_data, metadata.format)
            
            # Update cache
            self.state_cache[cache_key] = {
                'data': state_data,
                'metadata': metadata,
                'cached_at': time.time()
            }
            
            # Update metrics
            duration = time.time() - start_time
            self.operation_metrics['loads'] += 1
            self.operation_metrics['avg_load_time'] = (
                (self.operation_metrics['avg_load_time'] * (self.operation_metrics['loads'] - 1) + duration)
                / self.operation_metrics['loads']
            )
            
            self.logger.info(
                f"Successfully loaded agent state",
                meta={
                    "agent_name": agent_name,
                    "state_type": state_type.value,
                    "state_id": metadata.state_id,
                    "size_bytes": metadata.size_bytes,
                    "duration_ms": duration * 1000
                }
            )
            
            return state_data, metadata
            
        except Exception as e:
            self.operation_metrics['failures'] += 1
            error_msg = f"Failed to load agent state: {str(e)}"
            self.logger.error(error_msg, meta={
                "agent_name": agent_name,
                "state_type": state_type.value,
                "state_id": state_id,
                "error": str(e)
            })
            raise ServiceError(error_msg, "STATE_LOAD_ERROR", "agent_state_manager")
            
    async def create_agent_snapshot(self, agent_name: str) -> str:
        """Create a snapshot of all agent states"""
        try:
            snapshot_id = str(uuid.uuid4())
            timestamp = datetime.now()
            state_ids = []
            
            # Load all state types for this agent
            for state_type in StateType:
                try:
                    result = await self.load_agent_state(agent_name, state_type)
                    if result:
                        state_data, metadata = result
                        state_ids.append(metadata.state_id)
                except Exception:
                    continue
            
            if not state_ids:
                raise ValueError(f"No states found for agent {agent_name}")
            
            # Create snapshot record
            query = """
            INSERT INTO agent_snapshots (
                snapshot_id, agent_name, timestamp, version, state_ids, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING snapshot_id
            """
            
            params = (
                snapshot_id, 
                agent_name, 
                timestamp.isoformat(), 
                "1.0", 
                json.dumps(state_ids),
                datetime.now().isoformat()
            )
            
            await self.db_manager.execute_query(query, params, fetch_mode="one")
            
            self.logger.info(f"Created snapshot for agent {agent_name}: {snapshot_id}", meta={
                "agent_name": agent_name,
                "snapshot_id": snapshot_id,
                "states_included": len(state_ids)
            })
            
            return snapshot_id
            
        except Exception as e:
            error_msg = f"Failed to create agent snapshot: {str(e)}"
            self.logger.error(error_msg, meta={
                "agent_name": agent_name,
                "error": str(e)
            })
            raise ServiceError(error_msg, "SNAPSHOT_CREATE_ERROR", "agent_state_manager")
    
    async def restore_agent_from_snapshot(self, agent_name: str, snapshot_id: str) -> bool:
        """Restore agent state from a snapshot"""
        try:
            # Get snapshot record
            query = """
            SELECT * FROM agent_snapshots 
            WHERE snapshot_id = %s AND agent_name = %s
            """
            
            result = await self.db_manager.execute_query(query, (snapshot_id, agent_name), fetch_mode="one")
            
            if not result:
                raise ValueError(f"Snapshot {snapshot_id} not found for agent {agent_name}")
            
            state_ids = json.loads(result['state_ids'])
            
            # Create recovery log
            recovery_log_query = """
            INSERT INTO agent_recovery_logs (
                agent_name, recovery_type, source_snapshot_id, target_state,
                status, start_time, steps_total
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING log_id
            """
            
            recovery_params = (
                agent_name,
                "snapshot_restore",
                snapshot_id,
                "all",
                "in_progress",
                datetime.now().isoformat(),
                len(state_ids)
            )
            
            recovery_result = await self.db_manager.execute_query(recovery_log_query, recovery_params, fetch_mode="one")
            recovery_id = recovery_result['log_id'] if recovery_result else None
            
            # Restore each state
            steps_completed = 0
            for state_id in state_ids:
                try:
                    # Get state metadata
                    state_query = "SELECT * FROM agent_states WHERE state_id = %s"
                    state_result = await self.db_manager.execute_query(state_query, (state_id,), fetch_mode="one")
                    
                    if state_result:
                        state_type = StateType(state_result['state_type'])
                        
                        # Load the state data
                        state_result = await self.load_agent_state(agent_name, state_type, state_id)
                        if state_result:
                            state_data, _ = state_result
                            
                            # Save it as current state (creates a new version)
                            await self.save_agent_state(
                                agent_name=agent_name,
                                state_type=state_type,
                                state_data=state_data,
                                version=f"restored_from_{snapshot_id}",
                                tags=["restored", f"from_snapshot_{snapshot_id}"]
                            )
                            
                            steps_completed += 1
                            
                except Exception as e:
                    self.logger.warning(f"Error restoring state {state_id}: {e}")
            
            # Update recovery log
            if recovery_id:
                update_query = """
                UPDATE agent_recovery_logs SET
                    status = %s,
                    end_time = %s,
                    duration_ms = %s,
                    steps_completed = %s
                WHERE log_id = %s
                """
                
                end_time = datetime.now()
                duration_ms = int((end_time - datetime.fromisoformat(recovery_result['start_time'])).total_seconds() * 1000)
                status = "completed" if steps_completed == len(state_ids) else "partial"
                
                update_params = (
                    status,
                    end_time.isoformat(),
                    duration_ms,
                    steps_completed,
                    recovery_id
                )
                
                await self.db_manager.execute_query(update_query, update_params, fetch_mode="none")
            
            success = steps_completed > 0
            
            self.logger.info(f"Restored agent {agent_name} from snapshot {snapshot_id}", meta={
                "agent_name": agent_name,
                "snapshot_id": snapshot_id,
                "states_restored": steps_completed,
                "total_states": len(state_ids),
                "success": success
            })
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to restore from snapshot: {str(e)}"
            self.logger.error(error_msg, meta={
                "agent_name": agent_name,
                "snapshot_id": snapshot_id,
                "error": str(e)
            })
            return False
    
    async def get_agent_snapshots(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get snapshots for an agent"""
        try:
            query = """
            SELECT * FROM agent_snapshots 
            WHERE agent_name = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """
            
            results = await self.db_manager.execute_query(query, (agent_name, limit), fetch_mode="all")
            
            return results or []
            
        except Exception as e:
            self.logger.error(f"Failed to get agent snapshots: {e}")
            return []
    
    async def _background_maintenance(self):
        """Background task for maintenance operations"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run maintenance once per hour
                
                # Clean up expired states
                await self._cleanup_expired_states()
                
                # Enforce state version limits
                await self._enforce_version_limits()
                
                self.logger.debug("Completed background maintenance tasks")
                
            except Exception as e:
                self.logger.error(f"Error in background maintenance: {e}")
    
    async def _cleanup_expired_states(self):
        """Clean up expired states"""
        try:
            query = """
            DELETE FROM agent_states
            WHERE expiry_date IS NOT NULL AND expiry_date < %s
            RETURNING state_id
            """
            
            results = await self.db_manager.execute_query(query, (datetime.now().isoformat(),), fetch_mode="all")
            
            if results:
                deleted_count = len(results)
                self.logger.info(f"Cleaned up {deleted_count} expired states")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired states: {e}")
    
    async def _enforce_version_limits(self):
        """Enforce limits on the number of versions per agent and state type"""
        try:
            # Get all agent+state_type combinations
            query = """
            SELECT DISTINCT agent_name, state_type FROM agent_states
            """
            
            combinations = await self.db_manager.execute_query(query, fetch_mode="all")
            
            for combo in combinations:
                agent_name = combo['agent_name']
                state_type = combo['state_type']
                
                # Get excess versions to delete
                version_query = """
                SELECT state_id FROM agent_states
                WHERE agent_name = %s AND state_type = %s
                ORDER BY timestamp DESC
                OFFSET %s
                """
                
                excess_versions = await self.db_manager.execute_query(
                    version_query, 
                    (agent_name, state_type, self.max_state_versions),
                    fetch_mode="all"
                )
                
                if excess_versions:
                    # Delete excess versions
                    for version in excess_versions:
                        delete_query = "DELETE FROM agent_states WHERE state_id = %s"
                        await self.db_manager.execute_query(delete_query, (version['state_id'],), fetch_mode="none")
                    
                    self.logger.debug(f"Removed {len(excess_versions)} excess versions for {agent_name}/{state_type}")
                    
        except Exception as e:
            self.logger.error(f"Failed to enforce version limits: {e}")
            
    async def shutdown(self):
        """Clean up resources when shutting down"""
        try:
            # Close Redis connection if open
            if self.redis_client:
                self.redis_client.close()
            
            # Close database connection
            await self.db_manager.close()
            
            self.logger.info("Agent State Manager successfully shut down")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Factory function for easy instantiation
async def create_agent_state_manager(db_config: Optional[Dict[str, Any]] = None,
                                  redis_config: Optional[Dict[str, Any]] = None) -> AgentStateManager:
    """Factory function to create and initialize an AgentStateManager"""
    state_manager = AgentStateManager(db_config, redis_config)
    
    success = await state_manager.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Agent State Manager")
    
    return state_manager
    
    async def create_agent_snapshot(self, agent_name: str) -> str:
        """Create a complete snapshot of agent state"""
        try:
            snapshot_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Collect all state types for the agent
            state_types = [StateType.CONFIGURATION, StateType.LEARNING_MODEL, 
                          StateType.DECISION_HISTORY, StateType.PERFORMANCE_METRICS,
                          StateType.COMMUNICATION_STATE, StateType.DEPENDENCY_STATE]
            
            snapshot_data = {}
            state_ids = []
            
            for state_type in state_types:
                try:
                    result = await self.load_agent_state(agent_name, state_type)
                    if result:
                        state_data, metadata = result
                        snapshot_data[state_type.value] = state_data
                        state_ids.append(metadata.state_id)
                except Exception as e:
                    self.logger.warning(f"Could not load {state_type.value} for snapshot: {e}")
            
            # Create snapshot record
            snapshot = StateSnapshot(
                agent_name=agent_name,
                timestamp=timestamp,
                configuration=snapshot_data.get('configuration', {}),
                learning_models=snapshot_data.get('learning_model', {}),
                decision_history=snapshot_data.get('decision_history', []),
                performance_metrics=snapshot_data.get('performance_metrics', {}),
                communication_state=snapshot_data.get('communication_state', {}),
                dependency_relationships=snapshot_data.get('dependency_state', {}),
                version="1.0",
                snapshot_id=snapshot_id
            )
            
            # Save snapshot to database
            query = """
            INSERT INTO agent_snapshots (
                snapshot_id, agent_name, timestamp, version, state_ids, snapshot_data
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
            """
            
            params = (
                snapshot_id,
                agent_name,
                timestamp.isoformat(),
                "1.0",
                json.dumps(state_ids),
                json.dumps(asdict(snapshot), default=str)
            )
            
            await self.db_manager.execute_query(query, params, fetch_mode="none")
            
            self.logger.info(f"Created snapshot for agent {agent_name}", meta={
                "agent_name": agent_name,
                "snapshot_id": snapshot_id,
                "states_included": len(state_ids)
            })
            
            return snapshot_id
            
        except Exception as e:
            error_msg = f"Failed to create agent snapshot: {str(e)}"
            self.logger.error(error_msg, meta={"agent_name": agent_name, "error": str(e)})
            raise ServiceError(error_msg, "SNAPSHOT_CREATE_ERROR", "agent_state_manager")
    
    async def restore_agent_from_snapshot(self, agent_name: str, snapshot_id: str) -> bool:
        """Restore agent state from a snapshot"""
        try:
            # Load snapshot data
            query = """
            SELECT * FROM agent_snapshots 
            WHERE snapshot_id = %s AND agent_name = %s
            """
            
            results = await self.db_manager.execute_query(query, (snapshot_id, agent_name), fetch_mode="all")
            
            if not results:
                raise ValueError(f"Snapshot {snapshot_id} not found for agent {agent_name}")
            
            snapshot_row = results[0]
            snapshot_data = json.loads(snapshot_row['snapshot_data'])
            
            # Restore each state component
            restore_count = 0
            errors = []
            
            state_mappings = {
                'configuration': StateType.CONFIGURATION,
                'learning_models': StateType.LEARNING_MODEL,
                'decision_history': StateType.DECISION_HISTORY,
                'performance_metrics': StateType.PERFORMANCE_METRICS,
                'communication_state': StateType.COMMUNICATION_STATE,
                'dependency_relationships': StateType.DEPENDENCY_STATE
            }
            
            for data_key, state_type in state_mappings.items():
                if data_key in snapshot_data and snapshot_data[data_key]:
                    try:
                        await self.save_agent_state(
                            agent_name=agent_name,
                            state_type=state_type,
                            state_data=snapshot_data[data_key],
                            version="restored",
                            tags=["restored", f"from_snapshot_{snapshot_id}"]
                        )
                        restore_count += 1
                    except Exception as e:
                        errors.append(f"Failed to restore {state_type.value}: {e}")
                        self.logger.warning(f"Failed to restore {state_type.value}: {e}")
            
            # Log recovery
            await self._log_recovery(
                agent_name=agent_name,
                recovery_type="snapshot_restore",
                source_snapshot_id=snapshot_id,
                status="completed" if not errors else "partial",
                steps_completed=restore_count,
                steps_total=len(state_mappings),
                error_message="; ".join(errors) if errors else None
            )
            
            success = restore_count > 0
            
            self.logger.info(f"Restored agent from snapshot", meta={
                "agent_name": agent_name,
                "snapshot_id": snapshot_id,
                "states_restored": restore_count,
                "errors": len(errors),
                "success": success
            })
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to restore agent from snapshot: {str(e)}"
            self.logger.error(error_msg, meta={
                "agent_name": agent_name,
                "snapshot_id": snapshot_id,
                "error": str(e)
            })
            
            # Log failed recovery
            await self._log_recovery(
                agent_name=agent_name,
                recovery_type="snapshot_restore",
                source_snapshot_id=snapshot_id,
                status="failed",
                error_message=error_msg
            )
            
            raise ServiceError(error_msg, "SNAPSHOT_RESTORE_ERROR", "agent_state_manager")
    
    async def _log_recovery(self, 
                           agent_name: str,
                           recovery_type: str,
                           status: str,
                           source_snapshot_id: Optional[str] = None,
                           target_state: Optional[str] = None,
                           steps_completed: Optional[int] = None,
                           steps_total: Optional[int] = None,
                           error_message: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None):
        """Log recovery operation"""
        if start_time is None:
            start_time = datetime.now()
        if end_time is None:
            end_time = datetime.now()
        
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        query = """
        INSERT INTO agent_recovery_logs (
            agent_name, recovery_type, source_snapshot_id, target_state,
            status, start_time, end_time, duration_ms, steps_completed,
            steps_total, error_message, recovery_data
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        params = (
            agent_name, recovery_type, source_snapshot_id, target_state,
            status, start_time.isoformat(), end_time.isoformat(), duration_ms,
            steps_completed, steps_total, error_message, json.dumps({})
        )
        
        try:
            await self.db_manager.execute_query(query, params, fetch_mode="none")
        except Exception as e:
            self.logger.warning(f"Failed to log recovery operation: {e}")
    
    async def get_agent_snapshots(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of snapshots for an agent"""
        query = """
        SELECT snapshot_id, timestamp, version, state_ids
        FROM agent_snapshots 
        WHERE agent_name = %s
        ORDER BY timestamp DESC
        LIMIT %s
        """
        
        results = await self.db_manager.execute_query(query, (agent_name, limit), fetch_mode="all")
        
        snapshots = []
        for row in results:
            snapshots.append({
                'snapshot_id': row['snapshot_id'],
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'version': row['version'],
                'state_count': len(json.loads(row['state_ids']))
            })
        
        return snapshots
    
    async def get_recovery_history(self, agent_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recovery history for an agent"""
        query = """
        SELECT *
        FROM agent_recovery_logs 
        WHERE agent_name = %s
        ORDER BY start_time DESC
        LIMIT %s
        """
        
        results = await self.db_manager.execute_query(query, (agent_name, limit), fetch_mode="all")
        
        history = []
        for row in results:
            history.append({
                'log_id': row['log_id'],
                'recovery_type': row['recovery_type'],
                'status': row['status'],
                'start_time': row['start_time'].isoformat() if hasattr(row['start_time'], 'isoformat') else str(row['start_time']),
                'end_time': row['end_time'].isoformat() if hasattr(row['end_time'], 'isoformat') else str(row['end_time']) if row['end_time'] else None,
                'duration_ms': row['duration_ms'],
                'steps_completed': row['steps_completed'],
                'steps_total': row['steps_total'],
                'error_message': row['error_message']
            })
        
        return history
    
    async def cleanup_expired_states(self) -> int:
        """Clean up expired state data"""
        try:
            current_time = datetime.now()
            
            # Find expired states
            query = """
            SELECT state_id, data_location FROM agent_states 
            WHERE expiry_date IS NOT NULL AND expiry_date < %s
            """
            
            expired_states = await self.db_manager.execute_query(
                query, (current_time.isoformat(),), fetch_mode="all"
            )
            
            cleanup_count = 0
            
            for state in expired_states:
                try:
                    # Clean up Redis data if applicable
                    if state['data_location'].startswith('redis:') and self.redis_client:
                        redis_key = state['data_location'][6:]
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.redis_client.delete, redis_key
                        )
                    
                    # Remove from database
                    delete_query = "DELETE FROM agent_states WHERE state_id = %s"
                    await self.db_manager.execute_query(delete_query, (state['state_id'],), fetch_mode="none")
                    
                    cleanup_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup state {state['state_id']}: {e}")
            
            self.logger.info(f"Cleaned up {cleanup_count} expired states")
            return cleanup_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired states: {e}")
            return 0
    
    async def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored states"""
        try:
            # Get basic counts
            stats_query = """
            SELECT 
                COUNT(*) as total_states,
                COUNT(DISTINCT agent_name) as unique_agents,
                SUM(size_bytes) as total_size_bytes,
                AVG(size_bytes) as avg_size_bytes,
                MIN(timestamp) as oldest_state,
                MAX(timestamp) as newest_state
            FROM agent_states
            """
            
            stats_result = await self.db_manager.execute_query(stats_query, (), fetch_mode="one")
            
            # Get states by type
            type_query = """
            SELECT state_type, COUNT(*) as count, SUM(size_bytes) as total_size
            FROM agent_states
            GROUP BY state_type
            """
            
            type_results = await self.db_manager.execute_query(type_query, (), fetch_mode="all")
            
            # Get states by agent
            agent_query = """
            SELECT agent_name, COUNT(*) as count, SUM(size_bytes) as total_size
            FROM agent_states
            GROUP BY agent_name
            ORDER BY count DESC
            """
            
            agent_results = await self.db_manager.execute_query(agent_query, (), fetch_mode="all")
            
            statistics = {
                'total_states': stats_result['total_states'],
                'unique_agents': stats_result['unique_agents'],
                'total_size_bytes': stats_result['total_size_bytes'],
                'avg_size_bytes': float(stats_result['avg_size_bytes']) if stats_result['avg_size_bytes'] else 0,
                'oldest_state': stats_result['oldest_state'].isoformat() if stats_result['oldest_state'] else None,
                'newest_state': stats_result['newest_state'].isoformat() if stats_result['newest_state'] else None,
                'states_by_type': {row['state_type']: {'count': row['count'], 'total_size': row['total_size']} for row in type_results},
                'states_by_agent': {row['agent_name']: {'count': row['count'], 'total_size': row['total_size']} for row in agent_results},
                'operation_metrics': self.operation_metrics,
                'cache_size': len(self.state_cache)
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Failed to get state statistics: {e}")
            return {}
    
    async def validate_agent_state_consistency(self, agent_name: str) -> Dict[str, Any]:
        """Validate consistency of agent state data"""
        try:
            validation_results = {
                'agent_name': agent_name,
                'timestamp': datetime.now().isoformat(),
                'issues': [],
                'warnings': [],
                'state_integrity': True,
                'cross_state_consistency': True
            }
            
            # Load all state types for the agent
            state_data = {}
            for state_type in StateType:
                try:
                    result = await self.load_agent_state(agent_name, state_type)
                    if result:
                        data, metadata = result
                        state_data[state_type.value] = {
                            'data': data,
                            'metadata': metadata
                        }
                except Exception as e:
                    validation_results['issues'].append(f"Failed to load {state_type.value}: {e}")
                    validation_results['state_integrity'] = False
            
            # Check data consistency
            if 'dependency_state' in state_data and 'communication_state' in state_data:
                dep_data = state_data['dependency_state']['data']
                comm_data = state_data['communication_state']['data']
                
                # Check if dependency relationships match communication connections
                if isinstance(dep_data, dict) and isinstance(comm_data, dict):
                    dep_agents = set(dep_data.get('dependencies', []))
                    comm_agents = set(comm_data.get('connected_agents', []))
                    
                    if dep_agents != comm_agents:
                        validation_results['warnings'].append(
                            f"Dependency and communication agent lists don't match: deps={dep_agents}, comm={comm_agents}"
                        )
                        validation_results['cross_state_consistency'] = False
            
            # Check timestamp consistency
            timestamps = [state['metadata'].timestamp for state in state_data.values()]
            if timestamps:
                time_spread = max(timestamps) - min(timestamps)
                if time_spread.total_seconds() > 3600:  # More than 1 hour spread
                    validation_results['warnings'].append(
                        f"Large time spread between states: {time_spread}"
                    )
            
            # Overall validation status
            validation_results['overall_status'] = (
                validation_results['state_integrity'] and 
                validation_results['cross_state_consistency'] and 
                len(validation_results['issues']) == 0
            )
            
            return validation_results
            
        except Exception as e:
            return {
                'agent_name': agent_name,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'overall_status': False
            }
    
    async def _background_maintenance(self):
        """Background maintenance tasks"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup expired states
                await self.cleanup_expired_states()
                
                # Clean cache if too large
                if len(self.state_cache) > self.max_cache_size:
                    # Remove oldest cache entries
                    sorted_cache = sorted(
                        self.state_cache.items(),
                        key=lambda x: x[1]['cached_at']
                    )
                    
                    for key, _ in sorted_cache[:len(sorted_cache) - self.max_cache_size]:
                        del self.state_cache[key]
                
                self.logger.debug("Background maintenance completed")
                
            except Exception as e:
                self.logger.error(f"Background maintenance error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def shutdown(self):
        """Gracefully shutdown the state manager"""
        try:
            self.logger.info("Shutting down Agent State Manager")
            
            # Close Redis connection
            if self.redis_client:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.close
                )
            
            # Close database connections
            await self.db_manager.shutdown()
            
            self.is_initialized = False
            self.logger.info("Agent State Manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Factory function for easy instantiation
async def create_agent_state_manager(db_config: Optional[DatabaseConfig] = None,
                                   redis_config: Optional[Dict[str, Any]] = None) -> AgentStateManager:
    """Factory function to create and initialize an AgentStateManager"""
    manager = AgentStateManager(db_config, redis_config)
    
    success = await manager.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Agent State Manager")
    
    return manager


if __name__ == "__main__":
    # Example usage
    async def test_state_manager():
        try:
            # Create manager
            manager = await create_agent_state_manager()
            
            # Test state operations
            test_agent = "risk_genius"
            test_data = {
                "risk_tolerance": 0.15,
                "active_strategies": ["conservative", "moderate"],
                "last_analysis": "2025-06-04T12:00:00Z"
            }
            
            # Save state
            state_id = await manager.save_agent_state(
                agent_name=test_agent,
                state_type=StateType.CONFIGURATION,
                state_data=test_data,
                tags=["test", "configuration"]
            )
            
            print(f"Saved state with ID: {state_id}")
            
            # Load state
            result = await manager.load_agent_state(test_agent, StateType.CONFIGURATION)
            if result:
                loaded_data, metadata = result
                print(f"Loaded state: {loaded_data}")
                print(f"Metadata: {metadata}")
            
            # Create snapshot
            snapshot_id = await manager.create_agent_snapshot(test_agent)
            print(f"Created snapshot: {snapshot_id}")
            
            # Get statistics
            stats = await manager.get_state_statistics()
            print(f"Statistics: {stats}")
            
            # Shutdown
            await manager.shutdown()
            
        except Exception as e:
            print(f"Test failed: {e}")
            raise
    
    # Run test
    asyncio.run(test_state_manager())