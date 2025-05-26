"""
Execution Latency Testing & Optimization System
Rigorous testing and monitoring to ensure <10ms execution latency target

Features:
- Real-time latency monitoring
- End-to-end execution testing
- Performance bottleneck identification
- Latency optimization recommendations
- SLA compliance monitoring
- Performance regression detection
- Load testing capabilities
"""

import asyncio
import logging
import time
import statistics
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import redis
import json
import aiohttp
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LatencyType(Enum):
    ORDER_PLACEMENT = "order_placement"
    MARKET_DATA_FEED = "market_data_feed"
    SIGNAL_GENERATION = "signal_generation"
    RISK_CALCULATION = "risk_calculation"
    POSITION_UPDATE = "position_update"
    DATABASE_QUERY = "database_query"
    API_RESPONSE = "api_response"
    WEBSOCKET_MESSAGE = "websocket_message"

class PerformanceLevel(Enum):
    EXCELLENT = "excellent"     # <5ms
    GOOD = "good"              # 5-8ms
    ACCEPTABLE = "acceptable"   # 8-10ms
    POOR = "poor"              # 10-15ms
    CRITICAL = "critical"       # >15ms

@dataclass
class LatencyMeasurement:
    measurement_id: str
    latency_type: LatencyType
    service_name: str
    endpoint: str
    start_time: datetime
    end_time: datetime
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    payload_size: Optional[int] = None
    response_size: Optional[int] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None

@dataclass
class LatencyStats:
    latency_type: LatencyType
    service_name: str
    total_measurements: int
    avg_latency: float
    min_latency: float
    max_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    success_rate: float
    performance_level: PerformanceLevel
    sla_compliance: float  # % of requests meeting <10ms target
    trend: str  # 'improving', 'stable', 'degrading'
    last_updated: datetime

@dataclass
class PerformanceAlert:
    alert_id: str
    alert_type: str
    service_name: str
    severity: str
    message: str
    current_latency: float
    target_latency: float
    timestamp: datetime
    recommended_action: str

class LatencyTester:
    """
    Comprehensive latency testing and monitoring system
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Latency targets (in milliseconds)
        self.latency_targets = {
            LatencyType.ORDER_PLACEMENT: 10.0,
            LatencyType.MARKET_DATA_FEED: 5.0,
            LatencyType.SIGNAL_GENERATION: 8.0,
            LatencyType.RISK_CALCULATION: 5.0,
            LatencyType.POSITION_UPDATE: 3.0,
            LatencyType.DATABASE_QUERY: 2.0,
            LatencyType.API_RESPONSE: 10.0,
            LatencyType.WEBSOCKET_MESSAGE: 1.0
        }
        
        # Service endpoints for testing
        self.test_endpoints = {
            'trading-service': {
                'base_url': 'http://localhost:3001',
                'endpoints': {
                    LatencyType.ORDER_PLACEMENT: '/api/v1/orders',
                    LatencyType.POSITION_UPDATE: '/api/v1/positions',
                    LatencyType.API_RESPONSE: '/api/v1/health'
                }
            },
            'market-data-service': {
                'base_url': 'http://localhost:3002',
                'endpoints': {
                    LatencyType.MARKET_DATA_FEED: '/api/v1/prices/EURUSD',
                    LatencyType.API_RESPONSE: '/api/v1/health'
                }
            },
            'analytics-service': {
                'base_url': 'http://localhost:3003',
                'endpoints': {
                    LatencyType.SIGNAL_GENERATION: '/api/v1/analyze/EURUSD',
                    LatencyType.API_RESPONSE: '/api/v1/health'
                }
            },
            'risk-service': {
                'base_url': 'http://localhost:3004',
                'endpoints': {
                    LatencyType.RISK_CALCULATION: '/api/v1/risk/calculate',
                    LatencyType.API_RESPONSE: '/api/v1/health'
                }
            }
        }
        
        # Testing configuration
        self.test_config = {
            'continuous_test_interval': 30,    # seconds
            'load_test_duration': 300,         # seconds (5 minutes)
            'concurrent_requests': 10,         # concurrent requests for load testing
            'measurement_retention_days': 7,   # days to keep measurements
            'alert_threshold_violations': 5,   # consecutive violations before alert
            'performance_window': timedelta(minutes=15)  # window for performance calculation
        }
        
        # State tracking
        self.measurements = []
        self.performance_stats = {}
        self.alert_history = []
        self.running = False
        
        # Performance tracking
        self.system_stats = {
            'total_tests_executed': 0,
            'total_measurements': 0,
            'alerts_generated': 0,
            'sla_violations': 0,
            'average_latency_all_services': 0.0,
            'best_performing_service': None,
            'worst_performing_service': None,
            'last_test_run': None
        }
        
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        logger.info("LatencyTester initialized")

    async def start_monitoring(self):
        """Start continuous latency monitoring"""
        self.running = True
        logger.info("🚀 Starting latency monitoring and testing...")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._continuous_latency_testing()),
            asyncio.create_task(self._performance_analysis()),
            asyncio.create_task(self._system_resource_monitoring()),
            asyncio.create_task(self._cleanup_old_measurements())
        ]
        
        await asyncio.gather(*tasks)

    async def stop_monitoring(self):
        """Stop latency monitoring"""
        self.running = False
        logger.info("⏹️ Stopping latency monitoring...")

    async def measure_latency(self, latency_type: LatencyType, service_name: str, 
                            endpoint: str, payload: Optional[Dict] = None) -> LatencyMeasurement:
        """Measure latency for a specific operation"""
        measurement_id = f"LAT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            # Record system metrics before test
            cpu_before = psutil.cpu_percent()
            memory_before = psutil.virtual_memory().percent
            
            start_time = datetime.now()
            start_perf = time.perf_counter()
            
            # Execute the operation
            success, error_message, response_size = await self._execute_operation(
                service_name, endpoint, payload
            )
            
            end_perf = time.perf_counter()
            end_time = datetime.now()
            
            # Calculate latency
            latency_ms = (end_perf - start_perf) * 1000
            
            # Record system metrics after test
            cpu_after = psutil.cpu_percent()
            memory_after = psutil.virtual_memory().percent
            
            measurement = LatencyMeasurement(
                measurement_id=measurement_id,
                latency_type=latency_type,
                service_name=service_name,
                endpoint=endpoint,
                start_time=start_time,
                end_time=end_time,
                latency_ms=latency_ms,
                success=success,
                error_message=error_message,
                payload_size=len(json.dumps(payload)) if payload else 0,
                response_size=response_size,
                cpu_usage=(cpu_before + cpu_after) / 2,
                memory_usage=(memory_before + memory_after) / 2
            )
            
            # Store measurement
            await self._store_measurement(measurement)
            
            # Check for immediate alerts
            await self._check_latency_alert(measurement)
            
            self.system_stats['total_measurements'] += 1
            
            return measurement
            
        except Exception as e:
            logger.error(f"❌ Error measuring latency: {e}")
            return LatencyMeasurement(
                measurement_id=measurement_id,
                latency_type=latency_type,
                service_name=service_name,
                endpoint=endpoint,
                start_time=datetime.now(),
                end_time=datetime.now(),
                latency_ms=999.0,  # High latency to indicate error
                success=False,
                error_message=str(e)
            )

    async def _execute_operation(self, service_name: str, endpoint: str, 
                               payload: Optional[Dict] = None) -> Tuple[bool, Optional[str], int]:
        """Execute the actual operation and measure its performance"""
        try:
            service_config = self.test_endpoints.get(service_name)
            if not service_config:
                return False, f"Service {service_name} not configured", 0
            
            url = f"{service_config['base_url']}{endpoint}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                if payload:
                    async with session.post(url, json=payload) as response:
                        response_text = await response.text()
                        return response.status == 200, None, len(response_text)
                else:
                    async with session.get(url) as response:
                        response_text = await response.text()
                        return response.status == 200, None, len(response_text)
                        
        except asyncio.TimeoutError:
            return False, "Request timeout", 0
        except aiohttp.ClientError as e:
            return False, f"Client error: {str(e)}", 0
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", 0

    async def run_load_test(self, service_name: str, latency_type: LatencyType, 
                          concurrent_requests: int = 10, duration_seconds: int = 60) -> Dict:
        """Run load test to measure performance under stress"""
        logger.info(f"🔥 Starting load test: {service_name} - {latency_type.value}")
        
        start_time = datetime.now()
        measurements = []
        
        try:
            # Get endpoint for testing
            service_config = self.test_endpoints.get(service_name, {})
            endpoints = service_config.get('endpoints', {})
            endpoint = endpoints.get(latency_type)
            
            if not endpoint:
                logger.error(f"No endpoint configured for {service_name} - {latency_type.value}")
                return {'error': 'No endpoint configured'}
            
            # Run concurrent requests for specified duration
            end_time = start_time + timedelta(seconds=duration_seconds)
            
            async def worker():
                worker_measurements = []
                while datetime.now() < end_time:
                    measurement = await self.measure_latency(latency_type, service_name, endpoint)
                    worker_measurements.append(measurement)
                    await asyncio.sleep(0.1)  # Small delay between requests
                return worker_measurements
            
            # Start concurrent workers
            tasks = [asyncio.create_task(worker()) for _ in range(concurrent_requests)]
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            for worker_results in results:
                measurements.extend(worker_results)
            
            # Calculate load test statistics
            successful_measurements = [m for m in measurements if m.success]
            failed_measurements = [m for m in measurements if not m.success]
            
            if successful_measurements:
                latencies = [m.latency_ms for m in successful_measurements]
                
                stats = {
                    'test_duration': duration_seconds,
                    'concurrent_requests': concurrent_requests,
                    'total_requests': len(measurements),
                    'successful_requests': len(successful_measurements),
                    'failed_requests': len(failed_measurements),
                    'success_rate': len(successful_measurements) / len(measurements),
                    'avg_latency': statistics.mean(latencies),
                    'min_latency': min(latencies),
                    'max_latency': max(latencies),
                    'p50_latency': np.percentile(latencies, 50),
                    'p95_latency': np.percentile(latencies, 95),
                    'p99_latency': np.percentile(latencies, 99),
                    'requests_per_second': len(measurements) / duration_seconds,
                    'sla_compliance': len([l for l in latencies if l <= self.latency_targets[latency_type]]) / len(latencies)
                }
                
                logger.info(f"✅ Load test completed: {stats['avg_latency']:.2f}ms avg, {stats['success_rate']:.2%} success rate")
                return stats
            else:
                return {'error': 'No successful requests during load test'}
                
        except Exception as e:
            logger.error(f"❌ Load test failed: {e}")
            return {'error': str(e)}

    async def _continuous_latency_testing(self):
        """Continuous latency testing for all services"""
        while self.running:
            try:
                # Test all configured services and endpoints
                for service_name, service_config in self.test_endpoints.items():
                    for latency_type, endpoint in service_config['endpoints'].items():
                        await self.measure_latency(latency_type, service_name, endpoint)
                
                self.system_stats['total_tests_executed'] += 1
                self.system_stats['last_test_run'] = datetime.now()
                
                await asyncio.sleep(self.test_config['continuous_test_interval'])
                
            except Exception as e:
                logger.error(f"Error in continuous testing: {e}")
                await asyncio.sleep(self.test_config['continuous_test_interval'])

    async def _performance_analysis(self):
        """Analyze performance and generate statistics"""
        while self.running:
            try:
                # Calculate performance stats for each service/latency type combination
                for service_name in self.test_endpoints.keys():
                    for latency_type in LatencyType:
                        stats = await self._calculate_performance_stats(service_name, latency_type)
                        if stats:
                            key = f"{service_name}_{latency_type.value}"
                            self.performance_stats[key] = stats
                            
                            # Check for performance alerts
                            await self._check_performance_degradation(stats)
                
                # Update system-wide statistics
                await self._update_system_stats()
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
                await asyncio.sleep(300)

    async def _calculate_performance_stats(self, service_name: str, 
                                         latency_type: LatencyType) -> Optional[LatencyStats]:
        """Calculate performance statistics for a service/latency type"""
        try:
            # Get recent measurements
            recent_measurements = await self._get_recent_measurements(
                service_name, latency_type, self.test_config['performance_window']
            )
            
            if len(recent_measurements) < 5:  # Need minimum measurements
                return None
            
            successful_measurements = [m for m in recent_measurements if m.success]
            
            if not successful_measurements:
                return None
            
            latencies = [m.latency_ms for m in successful_measurements]
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            success_rate = len(successful_measurements) / len(recent_measurements)
            
            # SLA compliance (% meeting target)
            target = self.latency_targets[latency_type]
            sla_compliance = len([l for l in latencies if l <= target]) / len(latencies)
            
            # Performance level
            performance_level = self._determine_performance_level(avg_latency)
            
            # Trend analysis
            trend = await self._calculate_performance_trend(service_name, latency_type)
            
            return LatencyStats(
                latency_type=latency_type,
                service_name=service_name,
                total_measurements=len(recent_measurements),
                avg_latency=avg_latency,
                min_latency=min_latency,
                max_latency=max_latency,
                p50_latency=p50_latency,
                p95_latency=p95_latency,
                p99_latency=p99_latency,
                success_rate=success_rate,
                performance_level=performance_level,
                sla_compliance=sla_compliance,
                trend=trend,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return None

    def _determine_performance_level(self, avg_latency: float) -> PerformanceLevel:
        """Determine performance level based on average latency"""
        if avg_latency < 5:
            return PerformanceLevel.EXCELLENT
        elif avg_latency < 8:
            return PerformanceLevel.GOOD
        elif avg_latency <= 10:
            return PerformanceLevel.ACCEPTABLE
        elif avg_latency <= 15:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL

    async def _check_latency_alert(self, measurement: LatencyMeasurement):
        """Check if latency measurement triggers an alert"""
        try:
            target = self.latency_targets[measurement.latency_type]
            
            if measurement.latency_ms > target:
                await self._generate_latency_alert(
                    measurement,
                    "LATENCY_THRESHOLD_EXCEEDED",
                    "HIGH" if measurement.latency_ms > target * 1.5 else "MEDIUM",
                    f"Latency {measurement.latency_ms:.2f}ms exceeds target {target}ms",
                    "Investigate performance bottlenecks and optimize"
                )
                
                self.system_stats['sla_violations'] += 1
                
        except Exception as e:
            logger.error(f"Error checking latency alert: {e}")

    async def get_performance_report(self, timeframe: timedelta = timedelta(hours=1)) -> Dict:
        """Generate comprehensive performance report"""
        try:
            end_time = datetime.now()
            start_time = end_time - timeframe
            
            # Get all performance stats
            all_stats = list(self.performance_stats.values())
            
            if not all_stats:
                return {'error': 'No performance data available'}
            
            # Calculate overall metrics
            overall_avg_latency = statistics.mean([s.avg_latency for s in all_stats])
            overall_sla_compliance = statistics.mean([s.sla_compliance for s in all_stats])
            
            # Find best and worst performing services
            best_service = min(all_stats, key=lambda s: s.avg_latency)
            worst_service = max(all_stats, key=lambda s: s.avg_latency)
            
            return {
                'timeframe': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': timeframe.total_seconds() / 3600
                },
                'overall_performance': {
                    'average_latency': overall_avg_latency,
                    'sla_compliance': overall_sla_compliance,
                    'target_compliance': overall_sla_compliance >= 0.95,  # 95% SLA target
                    'best_performing_service': f"{best_service.service_name}_{best_service.latency_type.value}",
                    'worst_performing_service': f"{worst_service.service_name}_{worst_service.latency_type.value}"
                },
                'service_performance': [
                    {
                        'service_name': stats.service_name,
                        'latency_type': stats.latency_type.value,
                        'avg_latency': stats.avg_latency,
                        'p95_latency': stats.p95_latency,
                        'success_rate': stats.success_rate,
                        'sla_compliance': stats.sla_compliance,
                        'performance_level': stats.performance_level.value,
                        'trend': stats.trend
                    }
                    for stats in all_stats
                ],
                'latency_targets': {k.value: v for k, v in self.latency_targets.items()},
                'system_stats': self.system_stats,
                'recent_alerts': [
                    {
                        'alert_type': alert.alert_type,
                        'service_name': alert.service_name,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.alert_history[-10:]  # Last 10 alerts
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}

    # Additional helper methods would be implemented here for:
    # - _store_measurement()
    # - _get_recent_measurements()
    # - _calculate_performance_trend()
    # - _check_performance_degradation()
    # - _generate_latency_alert()
    # - _update_system_stats()
    # - _system_resource_monitoring()
    # - _cleanup_old_measurements()
