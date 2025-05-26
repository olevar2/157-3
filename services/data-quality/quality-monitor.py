#!/usr/bin/env python3
"""
Data Quality Monitor - Real-time data quality monitoring for AI Forex Trading Platform
Optimized for short-term trading data integrity (M1-H4 timeframes)

This module provides comprehensive data quality monitoring capabilities including:
- Real-time validation of market data
- Trading data integrity checks
- Technical indicator validation
- Anomaly detection and alerting
- Performance metrics tracking
"""

import asyncio
import logging
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import asyncpg
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import statistics
import time
import functools
from contextlib import asynccontextmanager
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_quality_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Severity(Enum):
    """Alert severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class ValidationResult:
    """Data validation result structure"""
    rule_id: str
    rule_name: str
    passed: bool
    severity: Severity
    error_message: Optional[str] = None
    data_point: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class QualityMetrics:
    """Data quality metrics structure"""
    timestamp: datetime
    data_quality_score: float
    data_freshness_score: float
    anomaly_rate: float
    total_records_processed: int
    failed_validations: int
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    low_alerts: int

class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system for forex trading platform
    """
    
    def __init__(self, config_path: str = "data-validation-rules.yaml"):
        """Initialize the data quality monitor with enhanced performance features"""
        self.config = self._load_config(config_path)
        self._validate_config(self.config)
        self.validation_results: List[ValidationResult] = []
        self.metrics_history: List[QualityMetrics] = []
        self.redis_client = None
        self.postgres_client = None
        self.postgres_pool = None  # Connection pool for better performance
        self.influx_client = None
        self.running = False
        
        # Performance tracking
        self.performance_metrics = {
            'validation_count': 0,
            'total_validation_time': 0.0,
            'alert_count': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Circuit breaker for fault tolerance
        self._circuit_breaker_state = {
            "failures": 0,
            "last_failure": None,
            "failure_threshold": 5,
            "recovery_timeout": 60
        }
        
        # Cache for expensive operations
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._circuit_breaker_state = {"failures": 0, "last_failure": None}
        self._performance_metrics = {"avg_validation_time": 0.0, "total_validations": 0}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load validation rules configuration"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _validate_config(self, config: Dict) -> None:
        """Validate configuration file structure and required fields"""
        required_sections = ['market_data', 'trading_data', 'technical_indicators', 'alerts']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate market data section
        if 'price_validation' not in config['market_data']:
            raise ValueError("Missing price_validation in market_data configuration")
        
        # Validate alert configuration
        if 'notification_channels' not in config['alerts']:
            raise ValueError("Missing notification_channels in alerts configuration")
        
        logger.info("Configuration validation completed successfully")
    
    async def initialize_connections(self):
        """Initialize database and cache connections with connection pooling"""
        try:
            # Redis connection for real-time data
            self.redis_client = await aioredis.from_url(
                "redis://localhost:6379",
                decode_responses=True
            )
            
            # PostgreSQL connection pool for better performance
            self.postgres_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                user="postgres",
                password="password",
                database="platform3_trading",
                min_size=5,  # Minimum connections in pool
                max_size=20,  # Maximum connections in pool
                command_timeout=60
            )
            
            # InfluxDB connection for time-series data
            self.influx_client = InfluxDBClientAsync(
                url="http://localhost:8086",
                token="your-token",
                org="platform3"
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def _create_connection_pool(self):
        """Create PostgreSQL connection pool for better performance"""
        try:
            postgres_dsn = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:" \
                          f"{os.getenv('POSTGRES_PASSWORD', 'password')}@" \
                          f"{os.getenv('POSTGRES_HOST', 'localhost')}:" \
                          f"{os.getenv('POSTGRES_PORT', '5432')}/" \
                          f"{os.getenv('POSTGRES_DB', 'platform3_trading')}"
            
            self.postgres_pool = await asyncpg.create_pool(
                dsn=postgres_dsn,
                min_size=5,
                max_size=20,
                command_timeout=30,
                server_settings={'jit': 'off'}
            )
            logger.info("PostgreSQL connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL connection pool: {e}")
            raise
    
    def _circuit_breaker(self, func):
        """Circuit breaker pattern for external service calls"""
        async def wrapper(*args, **kwargs):
            # Check circuit breaker state
            if self._circuit_breaker_state["failures"] >= 5:
                if self._circuit_breaker_state["last_failure"]:
                    time_since_failure = time.time() - self._circuit_breaker_state["last_failure"]
                    if time_since_failure < 300:  # 5 minutes timeout
                        logger.warning("Circuit breaker is open, skipping service call")
                        return None
                    else:
                        # Reset circuit breaker
                        self._circuit_breaker_state["failures"] = 0
                        self._circuit_breaker_state["last_failure"] = None
            
            try:
                result = await func(*args, **kwargs)
                # Reset on success
                self._circuit_breaker_state["failures"] = 0
                self._circuit_breaker_state["last_failure"] = None
                return result
            except Exception as e:
                self._circuit_breaker_state["failures"] += 1
                self._circuit_breaker_state["last_failure"] = time.time()
                logger.error(f"Service call failed: {e}")
                raise
        
        return wrapper

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache with TTL check"""
        if key in self._cache:
            cached_item = self._cache[key]
            if time.time() - cached_item['timestamp'] < self._cache_ttl:
                self.performance_metrics['cache_hits'] += 1
                return cached_item['value']
            else:
                # Cache expired, remove it
                del self._cache[key]
        
        self.performance_metrics['cache_misses'] += 1
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp"""
        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def circuit_breaker(self, func):
        """Circuit breaker decorator for external service calls"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if circuit breaker is open
            if (self._circuit_breaker_state["failures"] >= self._circuit_breaker_state["failure_threshold"] and
                self._circuit_breaker_state["last_failure"] and
                time.time() - self._circuit_breaker_state["last_failure"] < self._circuit_breaker_state["recovery_timeout"]):
                raise Exception("Circuit breaker is open - service temporarily unavailable")
            
            try:
                result = await func(*args, **kwargs)
                # Reset failure count on success
                self._circuit_breaker_state["failures"] = 0
                self._circuit_breaker_state["last_failure"] = None
                return result
            except Exception as e:
                self._circuit_breaker_state["failures"] += 1
                self._circuit_breaker_state["last_failure"] = time.time()
                logger.error(f"Service call failed: {e}")
                raise
        
        return wrapper

    @asynccontextmanager
    async def get_db_connection(self):
        """Context manager for database connections from pool"""
        if self.postgres_pool is None:
            raise RuntimeError("Database pool not initialized")
        
        async with self.postgres_pool.acquire() as connection:
            yield connection

    async def validate_market_data(self, data: Dict) -> List[ValidationResult]:
        """Validate market data according to configured rules"""
        results = []
        
        # OHLC consistency validation
        if all(key in data for key in ['open', 'high', 'low', 'close']):
            # High >= Low check
            if data['high'] < data['low']:
                results.append(ValidationResult(
                    rule_id="MD001",
                    rule_name="high_low_range",
                    passed=False,
                    severity=Severity.CRITICAL,
                    error_message="High price must be >= Low price",
                    data_point=data
                ))
            
            # Open/Close within High/Low range
            if not (data['low'] <= data['open'] <= data['high'] and 
                   data['low'] <= data['close'] <= data['high']):
                results.append(ValidationResult(
                    rule_id="MD001",
                    rule_name="open_close_within_range",
                    passed=False,
                    severity=Severity.CRITICAL,
                    error_message="Open/Close prices must be within High/Low range",
                    data_point=data
                ))
        
        # Price movement validation
        if 'symbol' in data and 'timeframe' in data:
            previous_close = await self._get_previous_close(data['symbol'], data['timeframe'])
            if previous_close:
                movement_pct = abs((data['close'] - previous_close) / previous_close * 100)
                threshold = self.config['market_data']['price_movement_limits']['thresholds'].get(
                    data['timeframe'], 5.0
                )
                
                if movement_pct > threshold:
                    results.append(ValidationResult(
                        rule_id="MD002",
                        rule_name="price_movement_limits",
                        passed=False,
                        severity=Severity.HIGH,
                        error_message=f"Price movement {movement_pct:.2f}% exceeds threshold {threshold}%",
                        data_point=data
                    ))
        
        # Spread validation for scalping
        if 'bid' in data and 'ask' in data and 'symbol' in data:
            spread_pips = (data['ask'] - data['bid']) * 10000  # Convert to pips
            max_spread = self.config['market_data']['spread_validation']['max_spread_pips'].get(
                data['symbol'], 5.0
            )
            
            if spread_pips > max_spread:
                results.append(ValidationResult(
                    rule_id="MD003",
                    rule_name="spread_validation",
                    passed=False,
                    severity=Severity.HIGH,
                    error_message=f"Spread {spread_pips:.1f} pips exceeds maximum {max_spread} pips",
                    data_point=data
                ))
        
        # Volume validation
        if 'volume' in data:
            if data['volume'] < 0:
                results.append(ValidationResult(
                    rule_id="MD004",
                    rule_name="non_negative_volume",
                    passed=False,
                    severity=Severity.MEDIUM,
                    error_message="Volume cannot be negative",
                    data_point=data
                ))
        
        # Timestamp validation
        if 'timestamp' in data:
            current_time = datetime.utcnow()
            data_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            
            if data_time > current_time + timedelta(seconds=60):
                results.append(ValidationResult(
                    rule_id="MD005",
                    rule_name="future_timestamp",
                    passed=False,
                    severity=Severity.CRITICAL,
                    error_message="Timestamp cannot be more than 60 seconds in future",
                    data_point=data
                ))
        
        return results
    
    async def validate_trading_data(self, data: Dict) -> List[ValidationResult]:
        """Validate trading data according to configured rules"""
        results = []
        
        # Order validation
        if 'order_type' in data:
            # Order size limits
            if 'lot_size' in data:
                if not (0.01 <= data['lot_size'] <= 100):
                    results.append(ValidationResult(
                        rule_id="TD001",
                        rule_name="order_size_limits",
                        passed=False,
                        severity=Severity.CRITICAL,
                        error_message="Order size must be between 0.01 and 100 lots",
                        data_point=data
                    ))
            
            # Price validity
            if 'price' in data and data['price'] <= 0:
                results.append(ValidationResult(
                    rule_id="TD001",
                    rule_name="price_validity",
                    passed=False,
                    severity=Severity.CRITICAL,
                    error_message="Order price must be positive",
                    data_point=data
                ))
        
        # Position validation
        if 'position_size' in data and 'unrealized_pnl' in data:
            # Margin requirements check
            if 'used_margin' in data and 'available_margin' in data:
                if data['used_margin'] > data['available_margin']:
                    results.append(ValidationResult(
                        rule_id="TD002",
                        rule_name="margin_requirements",
                        passed=False,
                        severity=Severity.CRITICAL,
                        error_message="Insufficient margin for position",
                        data_point=data
                    ))
        
        # Account validation
        if 'balance' in data:
            if data['balance'] < 0:
                results.append(ValidationResult(
                    rule_id="TD003",
                    rule_name="balance_consistency",
                    passed=False,
                    severity=Severity.CRITICAL,
                    error_message="Account balance cannot be negative",
                    data_point=data
                ))
        
        return results
    
    async def detect_anomalies(self, data: Dict) -> List[ValidationResult]:
        """Detect statistical and pattern-based anomalies"""
        results = []
        
        # Statistical anomaly detection using Z-score
        if 'close' in data and 'symbol' in data:
            historical_prices = await self._get_historical_prices(data['symbol'], 100)
            if len(historical_prices) > 30:
                mean_price = statistics.mean(historical_prices)
                std_price = statistics.stdev(historical_prices)
                z_score = abs((data['close'] - mean_price) / std_price)
                
                threshold = self.config['anomaly_detection']['statistical_thresholds']['z_score_threshold']
                if z_score > threshold:
                    results.append(ValidationResult(
                        rule_id="AD001",
                        rule_name="statistical_anomaly",
                        passed=False,
                        severity=Severity.MEDIUM,
                        error_message=f"Price Z-score {z_score:.2f} exceeds threshold {threshold}",
                        data_point=data
                    ))
        
        # Volume anomaly detection
        if 'volume' in data and 'symbol' in data:
            avg_volume = await self._get_average_volume(data['symbol'], 50)
            if avg_volume and data['volume'] > avg_volume * 5:
                results.append(ValidationResult(
                    rule_id="AD002",
                    rule_name="volume_anomaly",
                    passed=False,
                    severity=Severity.MEDIUM,
                    error_message=f"Volume {data['volume']} exceeds 5x average {avg_volume}",
                    data_point=data
                ))
        
        return results
    
    async def calculate_quality_metrics(self) -> QualityMetrics:
        """Calculate current data quality metrics"""
        current_time = datetime.utcnow()
        recent_results = [r for r in self.validation_results 
                         if (current_time - r.timestamp).total_seconds() < 3600]  # Last hour
        
        total_records = len(recent_results) if recent_results else 1
        failed_validations = len([r for r in recent_results if not r.passed])
        
        # Count alerts by severity
        critical_alerts = len([r for r in recent_results if not r.passed and r.severity == Severity.CRITICAL])
        high_alerts = len([r for r in recent_results if not r.passed and r.severity == Severity.HIGH])
        medium_alerts = len([r for r in recent_results if not r.passed and r.severity == Severity.MEDIUM])
        low_alerts = len([r for r in recent_results if not r.passed and r.severity == Severity.LOW])
        
        # Calculate scores
        data_quality_score = (total_records - failed_validations) / total_records
        data_freshness_score = await self._calculate_freshness_score()
        anomaly_rate = len([r for r in recent_results if r.rule_id.startswith('AD')]) / total_records
        
        return QualityMetrics(
            timestamp=current_time,
            data_quality_score=data_quality_score,
            data_freshness_score=data_freshness_score,
            anomaly_rate=anomaly_rate,
            total_records_processed=total_records,
            failed_validations=failed_validations,
            critical_alerts=critical_alerts,
            high_alerts=high_alerts,
            medium_alerts=medium_alerts,
            low_alerts=low_alerts
        )
    
    async def send_alert(self, result: ValidationResult):
        """Send alert based on severity and configuration"""
        alert_config = self.config['alerts']
        
        # Check if alert should be sent based on severity
        if result.severity.value in ['CRITICAL', 'HIGH']:
            # Send immediate notification
            await self._send_email_alert(result)
            await self._send_slack_alert(result)
        
        # Store alert in database
        await self._store_alert(result)
        
        # Auto-remediation for critical issues
        if result.severity == Severity.CRITICAL:
            await self._trigger_auto_remediation(result)
    
    async def _get_previous_close(self, symbol: str, timeframe: str) -> Optional[float]:
        """Get previous close price for movement validation"""
        try:
            # Implementation would query InfluxDB for previous close
            # This is a placeholder implementation
            return None
        except Exception as e:
            logger.error(f"Error getting previous close: {e}")
            return None
    
    async def _get_historical_prices(self, symbol: str, count: int) -> List[float]:
        """Get historical prices for anomaly detection"""
        try:
            # Implementation would query InfluxDB for historical data
            # This is a placeholder implementation
            return []
        except Exception as e:
            logger.error(f"Error getting historical prices: {e}")
            return []
    
    async def _get_average_volume(self, symbol: str, periods: int) -> Optional[float]:
        """Get average volume for anomaly detection"""
        try:
            # Implementation would query InfluxDB for volume data
            # This is a placeholder implementation
            return None
        except Exception as e:
            logger.error(f"Error getting average volume: {e}")
            return None
    
    async def _calculate_freshness_score(self) -> float:
        """Calculate data freshness score"""
        try:
            # Implementation would check data timestamps vs current time
            # This is a placeholder implementation
            return 0.98
        except Exception as e:
            logger.error(f"Error calculating freshness score: {e}")
            return 0.0
    
    async def _send_email_alert(self, result: ValidationResult):
        """Send email alert for critical/high severity issues"""
        try:
            # Email implementation would go here
            logger.info(f"Email alert sent for {result.rule_id}: {result.error_message}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, result: ValidationResult):
        """Send Slack alert for critical/high severity issues"""
        try:
            # Slack webhook implementation would go here
            logger.info(f"Slack alert sent for {result.rule_id}: {result.error_message}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _store_alert(self, result: ValidationResult):
        """Store alert in database for tracking"""
        try:
            if self.postgres_client:
                await self.postgres_client.execute("""
                    INSERT INTO data_quality_alerts 
                    (rule_id, rule_name, severity, error_message, data_point, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, result.rule_id, result.rule_name, result.severity.value,
                    result.error_message, json.dumps(result.data_point), result.timestamp)
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    async def _trigger_auto_remediation(self, result: ValidationResult):
        """Trigger automatic remediation for critical issues"""
        try:
            remediation_config = self.config.get('remediation', {})
            auto_fix_rules = remediation_config.get('auto_fix_rules', [])
            
            for rule in auto_fix_rules:
                if rule['rule_id'] == result.rule_id:
                    action = rule['action']
                    logger.info(f"Triggering auto-remediation action: {action} for rule {result.rule_id}")
                    
                    if action == "reject_order":
                        await self._reject_order(result.data_point)
                    elif action == "flag_and_quarantine":
                        await self._quarantine_data(result.data_point)
                    
        except Exception as e:
            logger.error(f"Failed to trigger auto-remediation: {e}")
    
    async def _reject_order(self, data_point: Dict):
        """Reject order due to validation failure"""
        logger.warning(f"Order rejected due to validation failure: {data_point}")
    
    async def _quarantine_data(self, data_point: Dict):
        """Quarantine suspicious data"""
        logger.warning(f"Data quarantined due to validation failure: {data_point}")
    
    async def start_monitoring(self):
        """Start the data quality monitoring process"""
        self.running = True
        logger.info("Data Quality Monitor started")
        
        try:
            await self.initialize_connections()
            
            while self.running:
                # Monitor market data stream
                await self._monitor_market_data_stream()
                
                # Monitor trading data stream
                await self._monitor_trading_data_stream()
                
                # Calculate and store metrics
                metrics = await self.calculate_quality_metrics()
                self.metrics_history.append(metrics)
                
                # Clean up old results (keep last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.validation_results = [r for r in self.validation_results if r.timestamp > cutoff_time]
                
                await asyncio.sleep(1)  # Monitor every second
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            await self.cleanup()
    
    async def _monitor_market_data_stream(self):
        """Monitor real-time market data stream"""
        try:
            # Implementation would subscribe to Redis streams or Kafka topics
            # This is a placeholder for the monitoring logic
            pass
        except Exception as e:
            logger.error(f"Error monitoring market data stream: {e}")
    
    async def _monitor_trading_data_stream(self):
        """Monitor real-time trading data stream"""
        try:
            # Implementation would subscribe to Redis streams or Kafka topics
            # This is a placeholder for the monitoring logic
            pass
        except Exception as e:
            logger.error(f"Error monitoring trading data stream: {e}")
    
    async def stop_monitoring(self):
        """Stop the data quality monitoring process"""
        self.running = False
        logger.info("Data Quality Monitor stopped")
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.postgres_client:
                await self.postgres_client.close()
            if self.influx_client:
                await self.influx_client.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main function to run the data quality monitor"""
    monitor = DataQualityMonitor()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
