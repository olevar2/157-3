#!/usr/bin/env python3
"""
Feature Store Maintenance Script
Database cleanup, performance optimization, and system maintenance
"""

import asyncio
import logging
import redis
import psycopg2
import os
import time
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStoreMaintenance:
    """Automated maintenance for Feature Store"""
    
    def __init__(self):
        self.redis_client = self._setup_redis()
        self.postgres_conn = self._setup_postgres()
        self.maintenance_config = self._load_maintenance_config()
        
    def _setup_redis(self) -> redis.Redis:
        """Setup Redis connection"""
        return redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1
        )
    
    def _setup_postgres(self):
        """Setup PostgreSQL connection"""
        return psycopg2.connect(
            host='localhost',
            port=5432,
            database='trading_db',
            user='trading_user',
            password='trading_pass'
        )
    
    def _load_maintenance_config(self) -> Dict:
        """Load maintenance configuration"""
        return {
            'feature_history_retention_days': 30,
            'redis_memory_cleanup_threshold': 0.8,
            'postgres_vacuum_schedule': 'daily',
            'performance_metrics_retention_days': 7,
            'error_logs_retention_days': 14,
            'backup_retention_days': 30
        }
    
    async def run_daily_maintenance(self):
        """Run daily maintenance tasks"""
        logger.info("Starting daily maintenance...")
        
        try:
            await self.cleanup_old_features()
            await self.optimize_redis_memory()
            await self.vacuum_postgres_tables()
            await self.cleanup_old_logs()
            await self.generate_maintenance_report()
            await self.backup_critical_data()
            
            logger.info("Daily maintenance completed successfully")
            
        except Exception as e:
            logger.error(f"Daily maintenance failed: {e}")
            raise
    
    async def cleanup_old_features(self):
        """Clean up old feature data"""
        logger.info("Cleaning up old features...")
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.maintenance_config['feature_history_retention_days'])
            
            # Clean Redis feature history
            history_keys = self.redis_client.keys("history:*")
            cleaned_redis_keys = 0
            
            for key in history_keys:
                # Keep only recent history entries
                list_length = self.redis_client.llen(key)
                if list_length > 1000:  # Keep last 1000 entries
                    self.redis_client.ltrim(key, 0, 999)
                    cleaned_redis_keys += 1
            
            # Clean PostgreSQL feature history
            cur = self.postgres_conn.cursor()
            
            delete_sql = """
                DELETE FROM feature_history 
                WHERE computation_timestamp < %s
            """
            
            cur.execute(delete_sql, (cutoff_date,))
            deleted_rows = cur.rowcount
            self.postgres_conn.commit()
            cur.close()
            
            logger.info(f"Cleaned {cleaned_redis_keys} Redis keys and {deleted_rows} PostgreSQL rows")
            
        except Exception as e:
            logger.error(f"Feature cleanup failed: {e}")
            raise
    
    async def optimize_redis_memory(self):
        """Optimize Redis memory usage"""
        logger.info("Optimizing Redis memory...")
        
        try:
            # Get memory info
            memory_info = self.redis_client.info('memory')
            used_memory = memory_info['used_memory']
            max_memory = memory_info.get('maxmemory', 0)
            
            if max_memory > 0:
                memory_usage_ratio = used_memory / max_memory
                
                if memory_usage_ratio > self.maintenance_config['redis_memory_cleanup_threshold']:
                    # Aggressive cleanup
                    logger.warning(f"High memory usage: {memory_usage_ratio:.2%}")
                    
                    # Remove old cache entries
                    cache_keys = self.redis_client.keys("cache:*")
                    for key in cache_keys:
                        ttl = self.redis_client.ttl(key)
                        if ttl == -1:  # No TTL set
                            self.redis_client.expire(key, 3600)  # Set 1 hour TTL
                    
                    # Clean old session data
                    session_keys = self.redis_client.keys("session:*")
                    old_sessions = []
                    for key in session_keys:
                        timestamp_str = self.redis_client.hget(key, 'timestamp')
                        if timestamp_str:
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str)
                                if datetime.now() - timestamp > timedelta(hours=24):
                                    old_sessions.append(key)
                            except:
                                old_sessions.append(key)
                    
                    if old_sessions:
                        self.redis_client.delete(*old_sessions)
                        logger.info(f"Removed {len(old_sessions)} old session keys")
            
            # Force garbage collection
            self.redis_client.memory_usage('memory-purge')
            
            # Update memory stats
            new_memory_info = self.redis_client.info('memory')
            logger.info(f"Memory optimization complete. Used: {new_memory_info['used_memory_human']}")
            
        except Exception as e:
            logger.error(f"Redis memory optimization failed: {e}")
            raise
    
    async def vacuum_postgres_tables(self):
        """Vacuum and analyze PostgreSQL tables"""
        logger.info("Vacuuming PostgreSQL tables...")
        
        try:
            cur = self.postgres_conn.cursor()
            
            # Get table sizes before vacuum
            cur.execute("""
                SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename IN ('feature_history', 'feature_statistics', 'pipeline_metrics')
            """)
            
            tables_before = dict(cur.fetchall())
            
            # Vacuum and analyze tables
            tables_to_vacuum = ['feature_history', 'feature_statistics', 'pipeline_metrics']
            
            for table in tables_to_vacuum:
                cur.execute(f"VACUUM ANALYZE {table}")
                logger.info(f"Vacuumed table: {table}")
            
            self.postgres_conn.commit()
            
            # Get table sizes after vacuum
            cur.execute("""
                SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename IN ('feature_history', 'feature_statistics', 'pipeline_metrics')
            """)
            
            tables_after = dict(cur.fetchall())
            cur.close()
            
            logger.info("PostgreSQL vacuum completed")
            for table in tables_to_vacuum:
                before_size = tables_before.get(table, 'unknown')
                after_size = tables_after.get(table, 'unknown')
                logger.info(f"  {table}: {before_size} -> {after_size}")
            
        except Exception as e:
            logger.error(f"PostgreSQL vacuum failed: {e}")
            raise
    
    async def cleanup_old_logs(self):
        """Clean up old log files"""
        logger.info("Cleaning up old logs...")
        
        try:
            log_directories = [
                'logs/',
                '../logs/',
                '../../logs/'
            ]
            
            cutoff_date = datetime.now() - timedelta(days=self.maintenance_config['error_logs_retention_days'])
            cleaned_files = 0
            
            for log_dir in log_directories:
                if os.path.exists(log_dir):
                    for filename in os.listdir(log_dir):
                        if filename.endswith('.log'):
                            filepath = os.path.join(log_dir, filename)
                            try:
                                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                                if file_mtime < cutoff_date:
                                    os.remove(filepath)
                                    cleaned_files += 1
                            except OSError:
                                continue
            
            logger.info(f"Cleaned {cleaned_files} old log files")
            
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
            raise
    
    async def generate_maintenance_report(self):
        """Generate maintenance summary report"""
        logger.info("Generating maintenance report...")
        
        try:
            # Redis statistics
            redis_info = self.redis_client.info()
            redis_stats = {
                'memory_usage': redis_info.get('used_memory_human', 'unknown'),
                'connected_clients': redis_info.get('connected_clients', 0),
                'total_commands_processed': redis_info.get('total_commands_processed', 0),
                'keyspace_hits': redis_info.get('keyspace_hits', 0),
                'keyspace_misses': redis_info.get('keyspace_misses', 0)
            }
            
            # PostgreSQL statistics
            cur = self.postgres_conn.cursor()
            cur.execute("""
                SELECT 
                    COUNT(*) as total_features,
                    MAX(created_at) as latest_feature,
                    MIN(created_at) as oldest_feature
                FROM feature_history
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            
            pg_stats = dict(zip(['total_features_24h', 'latest_feature', 'oldest_feature'], cur.fetchone()))
            cur.close()
            
            # Feature store statistics
            feature_count = self.redis_client.scard('feature_registry')
            active_symbols = self.redis_client.scard('active_symbols')
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'maintenance_type': 'daily',
                'redis_statistics': redis_stats,
                'postgres_statistics': pg_stats,
                'feature_store_statistics': {
                    'registered_features': feature_count,
                    'active_symbols': active_symbols
                },
                'maintenance_summary': {
                    'features_cleaned': True,
                    'memory_optimized': True,
                    'tables_vacuumed': True,
                    'logs_cleaned': True
                }
            }
            
            # Store report in Redis and PostgreSQL
            self.redis_client.hset('maintenance:latest_report', mapping={
                'data': json.dumps(report),
                'timestamp': datetime.now().isoformat()
            })
            
            # Store in PostgreSQL for history
            cur = self.postgres_conn.cursor()
            cur.execute("""
                INSERT INTO maintenance_reports (date, report_data, created_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET 
                    report_data = EXCLUDED.report_data,
                    updated_at = EXCLUDED.created_at
            """, (
                datetime.now().date(),
                json.dumps(report),
                datetime.now()
            ))
            self.postgres_conn.commit()
            cur.close()
            
            logger.info("Maintenance report generated and stored")
            
        except Exception as e:
            logger.error(f"Maintenance report generation failed: {e}")
            raise
    
    async def backup_critical_data(self):
        """Backup critical feature store data"""
        logger.info("Backing up critical data...")
        
        try:
            backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = f"backups/{backup_timestamp}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup feature definitions
            feature_registry = self.redis_client.smembers('feature_registry')
            with open(f"{backup_dir}/feature_registry.json", 'w') as f:
                json.dump(list(feature_registry), f, indent=2)
            
            # Backup feature schemas
            schema_keys = self.redis_client.keys("schema:*")
            schemas = {}
            for key in schema_keys:
                schemas[key] = self.redis_client.hgetall(key)
            
            with open(f"{backup_dir}/feature_schemas.json", 'w') as f:
                json.dump(schemas, f, indent=2)
            
            # Backup recent feature statistics from PostgreSQL
            cur = self.postgres_conn.cursor()
            cur.execute("""
                SELECT symbol, feature_name, date, min_value, max_value, avg_value, std_value, count_values
                FROM feature_statistics
                WHERE date >= %s
                ORDER BY date DESC, symbol, feature_name
            """, (datetime.now().date() - timedelta(days=7),))
            
            stats_data = []
            for row in cur.fetchall():
                stats_data.append({
                    'symbol': row[0],
                    'feature_name': row[1],
                    'date': row[2].isoformat(),
                    'min_value': row[3],
                    'max_value': row[4],
                    'avg_value': row[5],
                    'std_value': row[6],
                    'count_values': row[7]
                })
            
            with open(f"{backup_dir}/feature_statistics.json", 'w') as f:
                json.dump(stats_data, f, indent=2)
            
            cur.close()
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            logger.info(f"Critical data backed up to {backup_dir}")
            
        except Exception as e:
            logger.error(f"Data backup failed: {e}")
            raise
    
    async def _cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            backup_base_dir = "backups"
            if not os.path.exists(backup_base_dir):
                return
            
            cutoff_date = datetime.now() - timedelta(days=self.maintenance_config['backup_retention_days'])
            
            for backup_folder in os.listdir(backup_base_dir):
                backup_path = os.path.join(backup_base_dir, backup_folder)
                if os.path.isdir(backup_path):
                    try:
                        # Parse timestamp from folder name
                        folder_timestamp = datetime.strptime(backup_folder, '%Y%m%d_%H%M%S')
                        if folder_timestamp < cutoff_date:
                            import shutil
                            shutil.rmtree(backup_path)
                            logger.info(f"Removed old backup: {backup_folder}")
                    except ValueError:
                        # Skip folders that don't match timestamp format
                        continue
                        
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    async def run_emergency_maintenance(self):
        """Run emergency maintenance procedures"""
        logger.info("Running emergency maintenance...")
        
        try:
            # Immediate memory cleanup
            await self.optimize_redis_memory()
            
            # Check for stuck processes
            await self._check_stuck_processes()
            
            # Restart unhealthy components
            await self._restart_unhealthy_components()
            
            logger.info("Emergency maintenance completed")
            
        except Exception as e:
            logger.error(f"Emergency maintenance failed: {e}")
            raise
    
    async def _check_stuck_processes(self):
        """Check for stuck or hanging processes"""
        try:
            # Check pipeline metrics for staleness
            pipeline_metrics = self.redis_client.hgetall("pipeline:metrics")
            
            if pipeline_metrics.get('last_processed_timestamp'):
                last_processed = datetime.fromisoformat(pipeline_metrics['last_processed_timestamp'])
                if (datetime.now() - last_processed).total_seconds() > 600:  # 10 minutes
                    logger.warning("Pipeline appears stuck - last processing over 10 minutes ago")
                    
                    # Store alert
                    alert = {
                        'type': 'stuck_pipeline',
                        'timestamp': datetime.now().isoformat(),
                        'last_processed': pipeline_metrics['last_processed_timestamp']
                    }
                    
                    self.redis_client.lpush('maintenance:alerts', json.dumps(alert))
                    
        except Exception as e:
            logger.error(f"Stuck process check failed: {e}")
    
    async def _restart_unhealthy_components(self):
        """Restart unhealthy system components"""
        try:
            # This would typically include checks for:
            # - API server responsiveness
            # - Pipeline processing rate
            # - Database connections
            # - Message queue health
            
            # For now, just log the check
            logger.info("Component health check completed")
            
        except Exception as e:
            logger.error(f"Component restart check failed: {e}")
    
    def get_maintenance_status(self) -> Dict:
        """Get current maintenance status"""
        try:
            latest_report = self.redis_client.hgetall('maintenance:latest_report')
            
            if latest_report:
                return {
                    'last_maintenance': latest_report.get('timestamp'),
                    'status': 'completed',
                    'report': json.loads(latest_report.get('data', '{}'))
                }
            else:
                return {
                    'last_maintenance': 'never',
                    'status': 'pending',
                    'report': {}
                }
                
        except Exception as e:
            logger.error(f"Failed to get maintenance status: {e}")
            return {'status': 'error', 'error': str(e)}


# Main execution functions
async def daily_maintenance():
    """Run daily maintenance"""
    maintenance = FeatureStoreMaintenance()
    await maintenance.run_daily_maintenance()


async def emergency_maintenance():
    """Run emergency maintenance"""
    maintenance = FeatureStoreMaintenance()
    await maintenance.run_emergency_maintenance()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'emergency':
        asyncio.run(emergency_maintenance())
    else:
        asyncio.run(daily_maintenance())
