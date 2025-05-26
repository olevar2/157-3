#!/usr/bin/env python3
"""
Feature Store Monitoring Service
Real-time monitoring of feature quality, pipeline health, and performance metrics
"""

import asyncio
import logging
import redis
import psycopg2
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from kafka import KafkaProducer, KafkaConsumer
import yaml
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureMonitor:
    """Monitor feature store health and quality"""
    
    def __init__(self):
        self.redis_client = self._setup_redis()
        self.postgres_engine = self._setup_postgres()
        self.kafka_producer = self._setup_kafka_producer()
        self.alert_thresholds = self._load_alert_config()
        
        # Monitoring metrics
        self.quality_metrics = {}
        self.performance_metrics = {}
        self.error_counts = {}
        
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
    
    def _setup_kafka_producer(self) -> KafkaProducer:
        """Setup Kafka producer for alerts"""
        return KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
    
    def _load_alert_config(self) -> Dict:
        """Load alert thresholds from configuration"""
        return {
            'feature_freshness_max_age_seconds': 300,  # 5 minutes
            'feature_null_rate_threshold': 0.1,  # 10%
            'feature_outlier_rate_threshold': 0.05,  # 5%
            'pipeline_latency_max_ms': 1000,  # 1 second
            'api_response_time_max_ms': 100,  # 100ms
            'error_rate_threshold': 0.01  # 1%
        }
    
    async def start_monitoring(self):
        """Start the monitoring service"""
        logger.info("Starting Feature Store monitoring...")
        
        tasks = [
            asyncio.create_task(self._monitor_feature_quality()),
            asyncio.create_task(self._monitor_pipeline_health()),
            asyncio.create_task(self._monitor_api_performance()),
            asyncio.create_task(self._generate_daily_reports()),
            asyncio.create_task(self._check_system_resources())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            raise
    
    async def _monitor_feature_quality(self):
        """Monitor feature data quality"""
        while True:
            try:
                logger.info("Checking feature quality...")
                
                # Get all active features
                feature_keys = self.redis_client.keys("features:*")
                quality_issues = []
                
                for feature_key in feature_keys:
                    parts = feature_key.split(':')
                    if len(parts) >= 3:
                        symbol = parts[1]
                        feature_name = parts[2]
                        
                        # Check feature freshness
                        feature_data = self.redis_client.hgetall(feature_key)
                        if feature_data.get('timestamp'):
                            timestamp = datetime.fromisoformat(feature_data['timestamp'].replace('Z', '+00:00'))
                            age_seconds = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds()
                            
                            if age_seconds > self.alert_thresholds['feature_freshness_max_age_seconds']:
                                quality_issues.append({
                                    'type': 'stale_feature',
                                    'symbol': symbol,
                                    'feature': feature_name,
                                    'age_seconds': age_seconds,
                                    'severity': 'warning'
                                })
                        
                        # Check for null/invalid values
                        current_value = feature_data.get('current')
                        if current_value is None or current_value == 'null' or current_value == '':
                            quality_issues.append({
                                'type': 'null_feature',
                                'symbol': symbol,
                                'feature': feature_name,
                                'severity': 'error'
                            })
                        
                        # Check for outliers using historical data
                        await self._check_feature_outliers(symbol, feature_name, quality_issues)
                
                # Log and alert on quality issues
                if quality_issues:
                    await self._handle_quality_alerts(quality_issues)
                
                # Store quality metrics
                await self._store_quality_metrics(quality_issues)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Feature quality monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_feature_outliers(self, symbol: str, feature_name: str, quality_issues: List):
        """Check for feature outliers"""
        try:
            # Get historical values
            history_key = f"history:{symbol}:{feature_name}"
            history = self.redis_client.lrange(history_key, 0, 99)  # Last 100 values
            
            if len(history) >= 10:
                values = [float(v) for v in history if v and v != 'null']
                if len(values) >= 10:
                    # Calculate z-score for current value
                    current_key = f"features:{symbol}:{feature_name}"
                    current_data = self.redis_client.hgetall(current_key)
                    current_value = current_data.get('current')
                    
                    if current_value and current_value != 'null':
                        current_float = float(current_value)
                        mean = np.mean(values)
                        std = np.std(values)
                        
                        if std > 0:
                            z_score = abs((current_float - mean) / std)
                            if z_score > 3:  # More than 3 standard deviations
                                quality_issues.append({
                                    'type': 'outlier_feature',
                                    'symbol': symbol,
                                    'feature': feature_name,
                                    'current_value': current_float,
                                    'z_score': z_score,
                                    'severity': 'warning'
                                })
        
        except Exception as e:
            logger.error(f"Outlier check error for {symbol}:{feature_name}: {e}")
    
    async def _monitor_pipeline_health(self):
        """Monitor pipeline health and performance"""
        while True:
            try:
                logger.info("Checking pipeline health...")
                
                health_issues = []
                
                # Check Redis connectivity
                try:
                    self.redis_client.ping()
                except Exception as e:
                    health_issues.append({
                        'type': 'redis_connection',
                        'error': str(e),
                        'severity': 'critical'
                    })
                
                # Check PostgreSQL connectivity
                try:
                    conn = self.postgres_engine
                    cur = conn.cursor()
                    cur.execute("SELECT 1")
                    cur.close()
                except Exception as e:
                    health_issues.append({
                        'type': 'postgres_connection',
                        'error': str(e),
                        'severity': 'critical'
                    })
                
                # Check feature pipeline processing rate
                processing_metrics = self.redis_client.hgetall("pipeline:metrics")
                if processing_metrics:
                    last_processed = processing_metrics.get('last_processed_timestamp')
                    if last_processed:
                        last_time = datetime.fromisoformat(last_processed)
                        if (datetime.now() - last_time).total_seconds() > 300:  # 5 minutes
                            health_issues.append({
                                'type': 'pipeline_stalled',
                                'last_processed': last_processed,
                                'severity': 'critical'
                            })
                
                # Check memory usage
                redis_memory = self.redis_client.info('memory')
                memory_usage_ratio = redis_memory['used_memory'] / redis_memory['maxmemory'] if redis_memory['maxmemory'] > 0 else 0
                if memory_usage_ratio > 0.9:
                    health_issues.append({
                        'type': 'high_memory_usage',
                        'usage_ratio': memory_usage_ratio,
                        'severity': 'warning'
                    })
                
                # Handle health alerts
                if health_issues:
                    await self._handle_health_alerts(health_issues)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Pipeline health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_api_performance(self):
        """Monitor API performance metrics"""
        while True:
            try:
                logger.info("Checking API performance...")
                
                # Get API performance metrics from logs or Redis
                performance_data = self.redis_client.hgetall("api:performance")
                
                if performance_data:
                    avg_response_time = float(performance_data.get('avg_response_time_ms', 0))
                    error_rate = float(performance_data.get('error_rate', 0))
                    
                    alerts = []
                    
                    if avg_response_time > self.alert_thresholds['api_response_time_max_ms']:
                        alerts.append({
                            'type': 'high_api_latency',
                            'avg_response_time_ms': avg_response_time,
                            'severity': 'warning'
                        })
                    
                    if error_rate > self.alert_thresholds['error_rate_threshold']:
                        alerts.append({
                            'type': 'high_error_rate',
                            'error_rate': error_rate,
                            'severity': 'critical'
                        })
                    
                    if alerts:
                        await self._handle_performance_alerts(alerts)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"API performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _generate_daily_reports(self):
        """Generate daily monitoring reports"""
        while True:
            try:
                # Wait until midnight
                now = datetime.now()
                next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                sleep_seconds = (next_midnight - now).total_seconds()
                await asyncio.sleep(sleep_seconds)
                
                logger.info("Generating daily monitoring report...")
                
                # Generate comprehensive daily report
                report = await self._create_daily_report()
                
                # Store report in PostgreSQL
                await self._store_daily_report(report)
                
                # Send report via Kafka
                self.kafka_producer.send('monitoring-reports', report)
                
                logger.info("Daily report generated and stored")
                
            except Exception as e:
                logger.error(f"Daily report generation error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour if failed
    
    async def _create_daily_report(self) -> Dict:
        """Create comprehensive daily monitoring report"""
        try:
            report = {
                'date': datetime.now().date().isoformat(),
                'timestamp': datetime.now().isoformat(),
                'feature_statistics': {},
                'quality_summary': {},
                'performance_summary': {},
                'alerts_summary': {}
            }
            
            # Get feature statistics
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
            for symbol in symbols:
                feature_keys = self.redis_client.keys(f"features:{symbol}:*")
                report['feature_statistics'][symbol] = {
                    'total_features': len(feature_keys),
                    'active_features': 0,
                    'stale_features': 0
                }
                
                # Check feature freshness
                for key in feature_keys:
                    feature_data = self.redis_client.hgetall(key)
                    if feature_data.get('timestamp'):
                        timestamp = datetime.fromisoformat(feature_data['timestamp'].replace('Z', '+00:00'))
                        age_seconds = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds()
                        
                        if age_seconds < 300:  # Less than 5 minutes old
                            report['feature_statistics'][symbol]['active_features'] += 1
                        else:
                            report['feature_statistics'][symbol]['stale_features'] += 1
            
            return report
            
        except Exception as e:
            logger.error(f"Daily report creation error: {e}")
            return {'error': str(e)}
    
    async def _check_system_resources(self):
        """Monitor system resource usage"""
        while True:
            try:
                # Check Redis memory usage
                redis_info = self.redis_client.info()
                memory_usage = redis_info.get('used_memory_human', 'unknown')
                
                # Log resource statistics
                logger.info(f"System resources - Redis memory: {memory_usage}")
                
                # Store in Redis for API access
                self.redis_client.hset('system:resources', mapping={
                    'redis_memory_usage': memory_usage,
                    'timestamp': datetime.now().isoformat()
                })
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _handle_quality_alerts(self, quality_issues: List):
        """Handle feature quality alerts"""
        for issue in quality_issues:
            logger.warning(f"Quality alert: {issue}")
            
            # Send to Kafka alerts topic
            alert = {
                'type': 'feature_quality',
                'timestamp': datetime.now().isoformat(),
                'issue': issue
            }
            self.kafka_producer.send('feature-alerts', alert)
    
    async def _handle_health_alerts(self, health_issues: List):
        """Handle pipeline health alerts"""
        for issue in health_issues:
            logger.error(f"Health alert: {issue}")
            
            # Send to Kafka alerts topic
            alert = {
                'type': 'pipeline_health',
                'timestamp': datetime.now().isoformat(),
                'issue': issue
            }
            self.kafka_producer.send('feature-alerts', alert)
    
    async def _handle_performance_alerts(self, performance_issues: List):
        """Handle API performance alerts"""
        for issue in performance_issues:
            logger.warning(f"Performance alert: {issue}")
            
            # Send to Kafka alerts topic
            alert = {
                'type': 'api_performance',
                'timestamp': datetime.now().isoformat(),
                'issue': issue
            }
            self.kafka_producer.send('feature-alerts', alert)
    
    async def _store_quality_metrics(self, quality_issues: List):
        """Store quality metrics for trending"""
        timestamp = datetime.now()
        quality_score = max(0, 100 - len(quality_issues) * 10)  # Simple scoring
        
        self.redis_client.hset('quality:metrics', mapping={
            'score': quality_score,
            'issues_count': len(quality_issues),
            'timestamp': timestamp.isoformat()
        })
    
    async def _store_daily_report(self, report: Dict):
        """Store daily report in PostgreSQL"""
        try:
            conn = self.postgres_engine
            cur = conn.cursor()
            
            insert_sql = """
                INSERT INTO monitoring_reports (date, report_data, created_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET 
                    report_data = EXCLUDED.report_data,
                    updated_at = EXCLUDED.created_at
            """
            
            cur.execute(insert_sql, (
                report['date'],
                json.dumps(report),
                datetime.now()
            ))
            conn.commit()
            cur.close()
            
        except Exception as e:
            logger.error(f"Daily report storage error: {e}")


# Main execution
if __name__ == "__main__":
    monitor = FeatureMonitor()
    asyncio.run(monitor.start_monitoring())
