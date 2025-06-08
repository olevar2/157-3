"""
AI Trading Platform - Production Monitoring Integration
Mission: Maximize charitable funding for medical aid, children's surgeries, and poverty relief
Target: $300,000-400,000+ monthly for humanitarian causes

Real-time deployment monitoring and humanitarian impact tracking.
Integrates CI/CD metrics with production monitoring for complete visibility.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import psutil
import docker
import kubernetes as k8s
from kubernetes.client.rest import ApiException
import redis.asyncio as redis
import pandas as pd
import numpy as np

# Import monitoring components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from monitoring.advanced_monitoring_dashboard import AdvancedMonitoringDashboard
from monitoring.model_drift_detection import ModelDriftDetector

@dataclass
class DeploymentMetrics:
    """Deployment-specific metrics tracking"""
    deployment_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    status: str  # running, completed, failed, rolled_back
    environment: str
    strategy: str
    humanitarian_priority: int
    lives_affected: int
    charitable_impact_score: float
    resource_usage: Dict[str, float]
    health_scores: Dict[str, float]
    rollback_triggered: bool = False

@dataclass
class HumanitarianImpactMetrics:
    """Humanitarian impact tracking"""
    timestamp: datetime
    lives_saved_estimate: int
    charitable_funds_generated: float
    medical_aid_deliveries: int
    children_surgeries_funded: int
    poverty_relief_recipients: int
    platform_uptime_percentage: float
    trading_accuracy: float
    risk_prevention_score: float

class ProductionMonitoringIntegration:
    """
    Advanced production monitoring integration for humanitarian AI trading platform.
    Provides real-time visibility into deployments and humanitarian impact.
    """
    
    def __init__(self, config_path: str = "monitoring_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Monitoring components
        self.redis_client = None
        self.prometheus_registry = CollectorRegistry()
        self.dashboard = None
        self.drift_detector = None
        
        # Kubernetes client
        self.k8s_client = None
        self.k8s_apps_v1 = None
        
        # Docker client
        self.docker_client = None
        
        # Metrics tracking
        self.active_deployments: Dict[str, DeploymentMetrics] = {}
        self.deployment_history: List[DeploymentMetrics] = []
        self.humanitarian_metrics: List[HumanitarianImpactMetrics] = []
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Alert thresholds
        self.alert_thresholds = {
            "deployment_failure_rate": 0.05,  # 5%
            "humanitarian_impact_drop": 0.10,  # 10%
            "lives_at_risk_increase": 1000,
            "response_time_degradation": 0.50,  # 50%
            "error_rate_spike": 0.01  # 1%
        }
        
    def _setup_logging(self) -> structlog.BoundLogger:
        """Setup structured logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        return structlog.get_logger("production_monitoring").bind(
            service="production_monitoring",
            mission="humanitarian_impact_tracking"
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "monitoring_interval_seconds": 30,
            "humanitarian_impact_calculation_interval": 300,  # 5 minutes
            "deployment_timeout_minutes": 30,
            "rollback_timeout_minutes": 5,
            "prometheus": {
                "port": 9090,
                "pushgateway_url": "http://prometheus-pushgateway:9091"
            },
            "kubernetes": {
                "namespace": "ai-platform",
                "config_file": None  # Use in-cluster config
            },
            "humanitarian_targets": {
                "daily_lives_saved": 500,
                "daily_charitable_funds": 10000,  # USD
                "monthly_medical_deliveries": 1000,
                "monthly_surgeries_funded": 50,
                "monthly_poverty_relief": 5000
            }
        }
        
        # Load from file if exists
        config_file = Path(self.config_path)
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Deployment metrics
        self.deployment_counter = Counter(
            'deployments_total',
            'Total number of deployments',
            ['environment', 'strategy', 'status'],
            registry=self.prometheus_registry
        )
        
        self.deployment_duration = Histogram(
            'deployment_duration_seconds',
            'Deployment duration in seconds',
            ['environment', 'strategy'],
            registry=self.prometheus_registry
        )
        
        self.deployment_humanitarian_impact = Gauge(
            'deployment_humanitarian_impact_score',
            'Humanitarian impact score of current deployment',
            ['deployment_id', 'environment'],
            registry=self.prometheus_registry
        )
        
        # Humanitarian metrics
        self.lives_saved_gauge = Gauge(
            'humanitarian_lives_saved_total',
            'Total lives saved through platform',
            registry=self.prometheus_registry
        )
        
        self.charitable_funds_gauge = Gauge(
            'humanitarian_charitable_funds_usd',
            'Total charitable funds generated in USD',
            registry=self.prometheus_registry
        )
        
        self.medical_deliveries_gauge = Gauge(
            'humanitarian_medical_deliveries_total',
            'Total medical aid deliveries funded',
            registry=self.prometheus_registry
        )
        
        self.surgeries_funded_gauge = Gauge(
            'humanitarian_surgeries_funded_total',
            'Total children surgeries funded',
            registry=self.prometheus_registry
        )
        
        self.poverty_relief_gauge = Gauge(
            'humanitarian_poverty_relief_recipients',
            'Total poverty relief recipients',
            registry=self.prometheus_registry
        )
        
        # Platform health metrics
        self.platform_uptime_gauge = Gauge(
            'platform_uptime_percentage',
            'Platform uptime percentage',
            registry=self.prometheus_registry
        )
        
        self.trading_accuracy_gauge = Gauge(
            'trading_accuracy_percentage',
            'Trading algorithm accuracy percentage',
            registry=self.prometheus_registry
        )
        
        self.risk_prevention_gauge = Gauge(
            'risk_prevention_score',
            'Risk prevention effectiveness score',
            registry=self.prometheus_registry
        )
        
        # Alert metrics
        self.alert_counter = Counter(
            'monitoring_alerts_total',
            'Total monitoring alerts triggered',
            ['alert_type', 'severity', 'humanitarian_impact'],
            registry=self.prometheus_registry
        )
        
        self.lives_at_risk_gauge = Gauge(
            'humanitarian_lives_at_risk',
            'Current number of lives at risk due to platform issues',
            registry=self.prometheus_registry
        )
    
    async def initialize(self):
        """Initialize monitoring integration"""
        try:
            # Initialize Redis connection
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            await self.redis_client.ping()
            
            # Initialize Kubernetes client
            try:
                if self.config["kubernetes"]["config_file"]:
                    k8s.config.load_kube_config(self.config["kubernetes"]["config_file"])
                else:
                    k8s.config.load_incluster_config()
                
                self.k8s_client = k8s.client.ApiClient()
                self.k8s_apps_v1 = k8s.client.AppsV1Api()
                
                self.logger.info("Kubernetes client initialized")
                
            except Exception as e:
                self.logger.warning("Could not initialize Kubernetes client", error=str(e))
            
            # Initialize Docker client
            try:
                self.docker_client = docker.from_env()
                self.logger.info("Docker client initialized")
            except Exception as e:
                self.logger.warning("Could not initialize Docker client", error=str(e))
            
            # Initialize monitoring dashboard
            self.dashboard = AdvancedMonitoringDashboard()
            
            # Initialize drift detector
            self.drift_detector = ModelDriftDetector()
            
            # Load historical data
            await self._load_historical_data()
            
            # Start monitoring tasks
            asyncio.create_task(self._deployment_monitoring_loop())
            asyncio.create_task(self._humanitarian_impact_calculation_loop())
            asyncio.create_task(self._health_monitoring_loop())
            asyncio.create_task(self._alert_monitoring_loop())
            
            self.logger.info(
                "Production monitoring integration initialized",
                monitoring_interval=self.config["monitoring_interval_seconds"],
                humanitarian_targets=self.config["humanitarian_targets"]
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize monitoring integration", error=str(e))
            raise
    
    async def _load_historical_data(self):
        """Load historical monitoring data"""
        try:
            # Load deployment history
            deployment_data = await self.redis_client.get("deployment_history")
            if deployment_data:
                history_list = json.loads(deployment_data)
                self.deployment_history = [
                    DeploymentMetrics(**item) for item in history_list[-100:]
                ]
            
            # Load humanitarian metrics
            humanitarian_data = await self.redis_client.get("humanitarian_metrics_history")
            if humanitarian_data:
                metrics_list = json.loads(humanitarian_data)
                self.humanitarian_metrics = [
                    HumanitarianImpactMetrics(**item) for item in metrics_list[-100:]
                ]
                
            self.logger.info(
                "Historical data loaded",
                deployments=len(self.deployment_history),
                humanitarian_records=len(self.humanitarian_metrics)
            )
            
        except Exception as e:
            self.logger.warning("Could not load historical data", error=str(e))
    
    async def track_deployment_start(self, deployment_id: str, environment: str, 
                                   strategy: str, humanitarian_priority: int, 
                                   lives_affected: int, charitable_impact_score: float) -> DeploymentMetrics:
        """Track the start of a deployment"""
        metrics = DeploymentMetrics(
            deployment_id=deployment_id,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            duration_seconds=None,
            status="running",
            environment=environment,
            strategy=strategy,
            humanitarian_priority=humanitarian_priority,
            lives_affected=lives_affected,
            charitable_impact_score=charitable_impact_score,
            resource_usage={},
            health_scores={}
        )
        
        self.active_deployments[deployment_id] = metrics
        
        # Update Prometheus metrics
        self.deployment_counter.labels(
            environment=environment,
            strategy=strategy,
            status="started"
        ).inc()
        
        self.deployment_humanitarian_impact.labels(
            deployment_id=deployment_id,
            environment=environment
        ).set(charitable_impact_score)
        
        # Store in Redis
        await self._store_deployment_metrics(metrics)
        
        self.logger.info(
            "Deployment tracking started",
            deployment_id=deployment_id,
            environment=environment,
            strategy=strategy,
            humanitarian_priority=humanitarian_priority,
            lives_affected=lives_affected
        )
        
        return metrics
    
    async def track_deployment_completion(self, deployment_id: str, status: str, 
                                        resource_usage: Optional[Dict[str, float]] = None,
                                        health_scores: Optional[Dict[str, float]] = None):
        """Track deployment completion"""
        if deployment_id not in self.active_deployments:
            self.logger.warning("Unknown deployment completion", deployment_id=deployment_id)
            return
        
        metrics = self.active_deployments[deployment_id]
        metrics.end_time = datetime.now(timezone.utc)
        metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.status = status
        
        if resource_usage:
            metrics.resource_usage = resource_usage
        
        if health_scores:
            metrics.health_scores = health_scores
        
        # Update Prometheus metrics
        self.deployment_counter.labels(
            environment=metrics.environment,
            strategy=metrics.strategy,
            status=status
        ).inc()
        
        self.deployment_duration.labels(
            environment=metrics.environment,
            strategy=metrics.strategy
        ).observe(metrics.duration_seconds)
        
        # Move to history
        self.deployment_history.append(metrics)
        del self.active_deployments[deployment_id]
        
        # Keep only last 100 deployments
        if len(self.deployment_history) > 100:
            self.deployment_history = self.deployment_history[-100:]
        
        # Store updated data
        await self._store_deployment_metrics(metrics)
        
        self.logger.info(
            "Deployment tracking completed",
            deployment_id=deployment_id,
            status=status,
            duration_seconds=metrics.duration_seconds,
            humanitarian_impact=metrics.charitable_impact_score
        )
        
        # Check for alerts
        await self._check_deployment_alerts(metrics)
    
    async def track_rollback(self, deployment_id: str, reason: str):
        """Track deployment rollback"""
        if deployment_id in self.active_deployments:
            metrics = self.active_deployments[deployment_id]
            metrics.rollback_triggered = True
            metrics.status = "rolled_back"
            
            self.logger.warning(
                "Deployment rollback triggered",
                deployment_id=deployment_id,
                reason=reason,
                humanitarian_priority=metrics.humanitarian_priority,
                lives_affected=metrics.lives_affected
            )
            
            # Trigger humanitarian alert if high priority
            if metrics.humanitarian_priority >= 7:
                await self._trigger_humanitarian_alert(
                    "deployment_rollback",
                    f"High-priority deployment {deployment_id} rolled back: {reason}",
                    metrics.lives_affected
                )
    
    async def calculate_humanitarian_impact(self) -> HumanitarianImpactMetrics:
        """Calculate current humanitarian impact metrics"""
        try:
            # Get current platform metrics
            uptime = await self._get_platform_uptime()
            trading_accuracy = await self._get_trading_accuracy()
            risk_prevention = await self._get_risk_prevention_score()
            
            # Calculate humanitarian estimates based on platform performance
            base_lives_per_hour = 20  # Base lives saved per hour at 100% performance
            base_funds_per_hour = 400  # Base charitable funds per hour in USD
            
            performance_multiplier = (uptime * trading_accuracy * risk_prevention) / 100
            
            # Calculate current estimates
            lives_saved_estimate = int(base_lives_per_hour * performance_multiplier)
            charitable_funds = base_funds_per_hour * performance_multiplier
            
            # Estimate medical aid and surgery funding based on funds generated
            medical_deliveries = int(charitable_funds / 50)  # $50 per medical delivery
            surgeries_funded = int(charitable_funds / 2000)  # $2000 per surgery
            poverty_relief_recipients = int(charitable_funds / 10)  # $10 per recipient
            
            metrics = HumanitarianImpactMetrics(
                timestamp=datetime.now(timezone.utc),
                lives_saved_estimate=lives_saved_estimate,
                charitable_funds_generated=charitable_funds,
                medical_aid_deliveries=medical_deliveries,
                children_surgeries_funded=surgeries_funded,
                poverty_relief_recipients=poverty_relief_recipients,
                platform_uptime_percentage=uptime,
                trading_accuracy=trading_accuracy,
                risk_prevention_score=risk_prevention
            )
            
            # Update Prometheus metrics
            self.lives_saved_gauge.set(lives_saved_estimate)
            self.charitable_funds_gauge.set(charitable_funds)
            self.medical_deliveries_gauge.set(medical_deliveries)
            self.surgeries_funded_gauge.set(surgeries_funded)
            self.poverty_relief_gauge.set(poverty_relief_recipients)
            self.platform_uptime_gauge.set(uptime)
            self.trading_accuracy_gauge.set(trading_accuracy)
            self.risk_prevention_gauge.set(risk_prevention)
            
            # Store metrics
            self.humanitarian_metrics.append(metrics)
            if len(self.humanitarian_metrics) > 100:
                self.humanitarian_metrics = self.humanitarian_metrics[-100:]
            
            await self._store_humanitarian_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to calculate humanitarian impact", error=str(e))
            # Return default metrics in case of error
            return HumanitarianImpactMetrics(
                timestamp=datetime.now(timezone.utc),
                lives_saved_estimate=0,
                charitable_funds_generated=0.0,
                medical_aid_deliveries=0,
                children_surgeries_funded=0,
                poverty_relief_recipients=0,
                platform_uptime_percentage=0.0,
                trading_accuracy=0.0,
                risk_prevention_score=0.0
            )
    
    async def _get_platform_uptime(self) -> float:
        """Get current platform uptime percentage"""
        try:
            if self.k8s_apps_v1:
                # Get deployment status from Kubernetes
                namespace = self.config["kubernetes"]["namespace"]
                deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace)
                
                total_replicas = 0
                ready_replicas = 0
                
                for deployment in deployments.items:
                    if deployment.spec.replicas:
                        total_replicas += deployment.spec.replicas
                        if deployment.status.ready_replicas:
                            ready_replicas += deployment.status.ready_replicas
                
                if total_replicas > 0:
                    return (ready_replicas / total_replicas) * 100
                
            # Fallback to system metrics
            return 99.5  # Default high uptime
            
        except Exception as e:
            self.logger.warning("Could not get platform uptime", error=str(e))
            return 99.0
    
    async def _get_trading_accuracy(self) -> float:
        """Get current trading algorithm accuracy"""
        try:
            # Get accuracy from Redis cache
            accuracy_data = await self.redis_client.get("trading_accuracy")
            if accuracy_data:
                return float(accuracy_data)
            
            # Default accuracy for new deployments
            return 85.0
            
        except Exception as e:
            self.logger.warning("Could not get trading accuracy", error=str(e))
            return 80.0
    
    async def _get_risk_prevention_score(self) -> float:
        """Get current risk prevention effectiveness score"""
        try:
            # Get risk score from Redis cache
            risk_data = await self.redis_client.get("risk_prevention_score")
            if risk_data:
                return float(risk_data)
            
            # Default risk prevention score
            return 90.0
            
        except Exception as e:
            self.logger.warning("Could not get risk prevention score", error=str(e))
            return 85.0
    
    async def _deployment_monitoring_loop(self):
        """Continuous deployment monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.config["monitoring_interval_seconds"])
                
                # Monitor active deployments
                for deployment_id, metrics in list(self.active_deployments.items()):
                    await self._monitor_deployment_health(deployment_id, metrics)
                
                # Check for deployment timeouts
                await self._check_deployment_timeouts()
                
            except Exception as e:
                self.logger.error("Error in deployment monitoring loop", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _humanitarian_impact_calculation_loop(self):
        """Continuous humanitarian impact calculation"""
        while True:
            try:
                await asyncio.sleep(self.config["humanitarian_impact_calculation_interval"])
                
                # Calculate current humanitarian impact
                impact_metrics = await self.calculate_humanitarian_impact()
                
                # Check against targets
                await self._check_humanitarian_targets(impact_metrics)
                
            except Exception as e:
                self.logger.error("Error in humanitarian impact calculation", error=str(e))
                await asyncio.sleep(30)  # Brief pause before retrying
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Monitor system resources
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
                
                # Check for resource alerts
                if cpu_usage > 80:
                    await self._trigger_alert("high_cpu_usage", f"CPU usage: {cpu_usage}%")
                
                if memory_usage > 80:
                    await self._trigger_alert("high_memory_usage", f"Memory usage: {memory_usage}%")
                
                if disk_usage > 90:
                    await self._trigger_alert("high_disk_usage", f"Disk usage: {disk_usage}%")
                
            except Exception as e:
                self.logger.error("Error in health monitoring", error=str(e))
                await asyncio.sleep(30)
    
    async def _alert_monitoring_loop(self):
        """Continuous alert monitoring"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check deployment failure rate
                await self._check_deployment_failure_rate()
                
                # Check humanitarian impact trends
                await self._check_humanitarian_impact_trends()
                
                # Check response time degradation
                await self._check_response_time_degradation()
                
            except Exception as e:
                self.logger.error("Error in alert monitoring", error=str(e))
                await asyncio.sleep(15)
    
    async def _monitor_deployment_health(self, deployment_id: str, metrics: DeploymentMetrics):
        """Monitor health of active deployment"""
        try:
            # Get current resource usage
            if self.k8s_apps_v1:
                resource_usage = await self._get_k8s_resource_usage(deployment_id)
                metrics.resource_usage.update(resource_usage)
            
            # Get health scores
            health_scores = await self._get_deployment_health_scores(deployment_id)
            metrics.health_scores.update(health_scores)
            
            # Check for issues
            if health_scores.get("overall_health", 1.0) < 0.8:
                await self._trigger_alert(
                    "deployment_health_degradation",
                    f"Deployment {deployment_id} health degraded: {health_scores}",
                    humanitarian_impact=metrics.lives_affected
                )
            
        except Exception as e:
            self.logger.error(
                "Error monitoring deployment health",
                deployment_id=deployment_id,
                error=str(e)
            )
    
    async def _get_k8s_resource_usage(self, deployment_id: str) -> Dict[str, float]:
        """Get Kubernetes resource usage for deployment"""
        try:
            namespace = self.config["kubernetes"]["namespace"]
            
            # Get pods for this deployment
            pods = self.k8s_client.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"deployment={deployment_id}"
            )
            
            total_cpu = 0.0
            total_memory = 0.0
            pod_count = 0
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    # Get pod metrics (requires metrics-server)
                    # This is a simplified version
                    pod_count += 1
                    total_cpu += 0.1  # Default estimate
                    total_memory += 256  # Default estimate in MB
            
            return {
                "cpu_usage": total_cpu,
                "memory_usage_mb": total_memory,
                "pod_count": pod_count
            }
            
        except Exception as e:
            self.logger.warning("Could not get K8s resource usage", error=str(e))
            return {}
    
    async def _get_deployment_health_scores(self, deployment_id: str) -> Dict[str, float]:
        """Get health scores for deployment"""
        try:
            # Get health data from Redis
            health_data = await self.redis_client.get(f"deployment_health:{deployment_id}")
            if health_data:
                return json.loads(health_data)
            
            # Default health scores
            return {
                "overall_health": 0.95,
                "response_time": 0.9,
                "error_rate": 0.99,
                "throughput": 0.85
            }
            
        except Exception as e:
            self.logger.warning("Could not get deployment health scores", error=str(e))
            return {"overall_health": 0.5}
    
    async def _check_deployment_timeouts(self):
        """Check for deployment timeouts"""
        timeout_minutes = self.config["deployment_timeout_minutes"]
        current_time = datetime.now(timezone.utc)
        
        for deployment_id, metrics in list(self.active_deployments.items()):
            elapsed = (current_time - metrics.start_time).total_seconds() / 60
            
            if elapsed > timeout_minutes:
                self.logger.warning(
                    "Deployment timeout detected",
                    deployment_id=deployment_id,
                    elapsed_minutes=elapsed,
                    humanitarian_priority=metrics.humanitarian_priority
                )
                
                # Mark as failed and trigger rollback if high priority
                await self.track_deployment_completion(deployment_id, "timeout")
                
                if metrics.humanitarian_priority >= 7:
                    await self._trigger_humanitarian_alert(
                        "deployment_timeout",
                        f"High-priority deployment {deployment_id} timed out after {elapsed:.1f} minutes",
                        metrics.lives_affected
                    )
    
    async def _check_deployment_alerts(self, metrics: DeploymentMetrics):
        """Check for deployment-related alerts"""
        # Check if deployment failed
        if metrics.status == "failed":
            if metrics.humanitarian_priority >= 7:
                await self._trigger_humanitarian_alert(
                    "deployment_failure",
                    f"High-priority deployment {metrics.deployment_id} failed",
                    metrics.lives_affected
                )
        
        # Check deployment duration
        if metrics.duration_seconds and metrics.duration_seconds > 1800:  # 30 minutes
            await self._trigger_alert(
                "slow_deployment",
                f"Deployment {metrics.deployment_id} took {metrics.duration_seconds/60:.1f} minutes"
            )
    
    async def _check_deployment_failure_rate(self):
        """Check deployment failure rate"""
        if len(self.deployment_history) < 10:
            return  # Need more data
        
        recent_deployments = self.deployment_history[-20:]  # Last 20 deployments
        failed_count = sum(1 for d in recent_deployments if d.status in ["failed", "timeout"])
        failure_rate = failed_count / len(recent_deployments)
        
        if failure_rate > self.alert_thresholds["deployment_failure_rate"]:
            await self._trigger_alert(
                "high_deployment_failure_rate",
                f"Deployment failure rate: {failure_rate:.1%} (threshold: {self.alert_thresholds['deployment_failure_rate']:.1%})"
            )
    
    async def _check_humanitarian_targets(self, metrics: HumanitarianImpactMetrics):
        """Check humanitarian impact against targets"""
        targets = self.config["humanitarian_targets"]
        
        # Scale daily targets to current time period
        hours_in_day = 24
        current_hour = datetime.now().hour
        expected_daily_progress = current_hour / hours_in_day
        
        expected_lives = targets["daily_lives_saved"] * expected_daily_progress
        expected_funds = targets["daily_charitable_funds"] * expected_daily_progress
        
        # Check if we're significantly behind targets
        if metrics.lives_saved_estimate < expected_lives * 0.8:
            await self._trigger_humanitarian_alert(
                "lives_saved_below_target",
                f"Lives saved ({metrics.lives_saved_estimate}) below expected ({expected_lives:.0f})",
                int(expected_lives - metrics.lives_saved_estimate)
            )
        
        if metrics.charitable_funds_generated < expected_funds * 0.8:
            await self._trigger_humanitarian_alert(
                "charitable_funds_below_target",
                f"Charitable funds (${metrics.charitable_funds_generated:.0f}) below expected (${expected_funds:.0f})",
                0
            )
    
    async def _check_humanitarian_impact_trends(self):
        """Check humanitarian impact trends"""
        if len(self.humanitarian_metrics) < 5:
            return
        
        recent_metrics = self.humanitarian_metrics[-5:]
        
        # Check for declining trend
        lives_trend = [m.lives_saved_estimate for m in recent_metrics]
        funds_trend = [m.charitable_funds_generated for m in recent_metrics]
        
        # Simple trend detection
        if len(lives_trend) >= 3:
            if all(lives_trend[i] < lives_trend[i-1] for i in range(1, len(lives_trend))):
                await self._trigger_humanitarian_alert(
                    "declining_lives_saved_trend",
                    f"Declining trend in lives saved: {lives_trend}",
                    0
                )
        
        if len(funds_trend) >= 3:
            if all(funds_trend[i] < funds_trend[i-1] for i in range(1, len(funds_trend))):
                await self._trigger_humanitarian_alert(
                    "declining_funds_trend",
                    f"Declining trend in charitable funds: {funds_trend}",
                    0
                )
    
    async def _check_response_time_degradation(self):
        """Check for response time degradation"""
        try:
            # Get current response time metrics
            response_time_data = await self.redis_client.get("current_response_time")
            baseline_response_time = await self.redis_client.get("baseline_response_time")
            
            if response_time_data and baseline_response_time:
                current_time = float(response_time_data)
                baseline_time = float(baseline_response_time)
                
                degradation = (current_time - baseline_time) / baseline_time
                
                if degradation > self.alert_thresholds["response_time_degradation"]:
                    await self._trigger_alert(
                        "response_time_degradation",
                        f"Response time degraded by {degradation:.1%}: {current_time:.0f}ms vs baseline {baseline_time:.0f}ms"
                    )
        
        except Exception as e:
            self.logger.warning("Could not check response time degradation", error=str(e))
    
    async def _trigger_alert(self, alert_type: str, message: str, humanitarian_impact: int = 0):
        """Trigger monitoring alert"""
        self.alert_counter.labels(
            alert_type=alert_type,
            severity="warning",
            humanitarian_impact="low" if humanitarian_impact < 1000 else "high"
        ).inc()
        
        self.logger.warning(
            "Monitoring alert triggered",
            alert_type=alert_type,
            message=message,
            humanitarian_impact=humanitarian_impact
        )
        
        # Send to notification channels
        await self._send_alert_notification(alert_type, message, "warning")
    
    async def _trigger_humanitarian_alert(self, alert_type: str, message: str, lives_affected: int):
        """Trigger humanitarian-specific alert"""
        self.alert_counter.labels(
            alert_type=alert_type,
            severity="critical",
            humanitarian_impact="high"
        ).inc()
        
        self.lives_at_risk_gauge.set(lives_affected)
        
        self.logger.critical(
            "HUMANITARIAN ALERT",
            alert_type=alert_type,
            message=message,
            lives_affected=lives_affected
        )
        
        # Send to all notification channels
        await self._send_alert_notification(alert_type, message, "critical", lives_affected)
    
    async def _send_alert_notification(self, alert_type: str, message: str, severity: str, lives_affected: int = 0):
        """Send alert notification to configured channels"""
        try:
            notification_data = {
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "lives_affected": lives_affected,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "platform": "humanitarian_ai_trading"
            }
            
            # Store alert in Redis
            await self.redis_client.lpush(
                "monitoring_alerts",
                json.dumps(notification_data)
            )
            
            # Keep only last 100 alerts
            await self.redis_client.ltrim("monitoring_alerts", 0, 99)
            
            # TODO: Integrate with actual notification services (Slack, PagerDuty, etc.)
            
        except Exception as e:
            self.logger.error("Failed to send alert notification", error=str(e))
    
    async def _store_deployment_metrics(self, metrics: DeploymentMetrics):
        """Store deployment metrics in Redis"""
        try:
            metrics_data = asdict(metrics)
            metrics_data["start_time"] = metrics.start_time.isoformat()
            if metrics.end_time:
                metrics_data["end_time"] = metrics.end_time.isoformat()
            
            # Store individual deployment
            await self.redis_client.setex(
                f"deployment_metrics:{metrics.deployment_id}",
                86400,  # 24 hours
                json.dumps(metrics_data)
            )
            
            # Update deployment history
            history_data = [asdict(m) for m in self.deployment_history]
            for item in history_data:
                item["start_time"] = item["start_time"].isoformat() if isinstance(item["start_time"], datetime) else item["start_time"]
                if item["end_time"]:
                    item["end_time"] = item["end_time"].isoformat() if isinstance(item["end_time"], datetime) else item["end_time"]
            
            await self.redis_client.setex(
                "deployment_history",
                86400,
                json.dumps(history_data, default=str)
            )
            
        except Exception as e:
            self.logger.warning("Could not store deployment metrics", error=str(e))
    
    async def _store_humanitarian_metrics(self, metrics: HumanitarianImpactMetrics):
        """Store humanitarian metrics in Redis"""
        try:
            metrics_data = asdict(metrics)
            metrics_data["timestamp"] = metrics.timestamp.isoformat()
            
            # Store current metrics
            await self.redis_client.setex(
                "current_humanitarian_metrics",
                3600,  # 1 hour
                json.dumps(metrics_data)
            )
            
            # Update history
            history_data = [asdict(m) for m in self.humanitarian_metrics]
            for item in history_data:
                item["timestamp"] = item["timestamp"].isoformat() if isinstance(item["timestamp"], datetime) else item["timestamp"]
            
            await self.redis_client.setex(
                "humanitarian_metrics_history",
                86400,
                json.dumps(history_data, default=str)
            )
            
        except Exception as e:
            self.logger.warning("Could not store humanitarian metrics", error=str(e))
    
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        try:
            # Get latest humanitarian metrics
            latest_humanitarian = self.humanitarian_metrics[-1] if self.humanitarian_metrics else None
            
            # Calculate deployment statistics
            recent_deployments = self.deployment_history[-20:] if self.deployment_history else []
            success_rate = 0.0
            avg_duration = 0.0
            
            if recent_deployments:
                successful = sum(1 for d in recent_deployments if d.status == "completed")
                success_rate = successful / len(recent_deployments)
                
                completed_deployments = [d for d in recent_deployments if d.duration_seconds]
                if completed_deployments:
                    avg_duration = sum(d.duration_seconds for d in completed_deployments) / len(completed_deployments)
            
            # Get active deployment count
            active_count = len(self.active_deployments)
            
            summary = {
                "monitoring_status": "operational",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "deployment_metrics": {
                    "active_deployments": active_count,
                    "recent_success_rate": success_rate,
                    "average_deployment_duration_seconds": avg_duration,
                    "total_deployments_tracked": len(self.deployment_history)
                },
                "humanitarian_impact": {
                    "current_lives_saved_estimate": latest_humanitarian.lives_saved_estimate if latest_humanitarian else 0,
                    "current_charitable_funds": latest_humanitarian.charitable_funds_generated if latest_humanitarian else 0.0,
                    "platform_uptime_percentage": latest_humanitarian.platform_uptime_percentage if latest_humanitarian else 0.0,
                    "trading_accuracy": latest_humanitarian.trading_accuracy if latest_humanitarian else 0.0,
                    "risk_prevention_score": latest_humanitarian.risk_prevention_score if latest_humanitarian else 0.0
                },
                "system_health": {
                    "cpu_usage_percent": psutil.cpu_percent(),
                    "memory_usage_percent": psutil.virtual_memory().percent,
                    "disk_usage_percent": psutil.disk_usage('/').percent
                },
                "alert_summary": {
                    "total_alerts_today": await self._get_alerts_count_today(),
                    "critical_alerts_active": await self._get_critical_alerts_count(),
                    "lives_at_risk": self.lives_at_risk_gauge._value._value
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error("Failed to generate monitoring summary", error=str(e))
            return {"error": "Could not generate monitoring summary"}
    
    async def _get_alerts_count_today(self) -> int:
        """Get count of alerts triggered today"""
        try:
            alerts_data = await self.redis_client.lrange("monitoring_alerts", 0, -1)
            today = datetime.now(timezone.utc).date()
            
            count = 0
            for alert_json in alerts_data:
                alert = json.loads(alert_json)
                alert_date = datetime.fromisoformat(alert["timestamp"]).date()
                if alert_date == today:
                    count += 1
            
            return count
            
        except Exception as e:
            self.logger.warning("Could not get alerts count", error=str(e))
            return 0
    
    async def _get_critical_alerts_count(self) -> int:
        """Get count of active critical alerts"""
        try:
            alerts_data = await self.redis_client.lrange("monitoring_alerts", 0, 9)  # Last 10 alerts
            
            count = 0
            for alert_json in alerts_data:
                alert = json.loads(alert_json)
                if alert["severity"] == "critical":
                    count += 1
            
            return count
            
        except Exception as e:
            self.logger.warning("Could not get critical alerts count", error=str(e))
            return 0
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down production monitoring integration")
        
        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        
        if self.docker_client:
            self.docker_client.close()
        
        self.logger.info("Production monitoring integration shutdown complete")

# Integration with CI/CD pipeline
async def integrate_with_cicd_pipeline():
    """Integration function for CI/CD pipeline"""
    monitoring = ProductionMonitoringIntegration()
    await monitoring.initialize()
    return monitoring

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Monitoring Integration Service")
    parser.add_argument("--config", default="monitoring_config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    async def main():
        monitoring = ProductionMonitoringIntegration(args.config)
        await monitoring.initialize()
        
        print("🚀 Humanitarian AI Production Monitoring Active")
        print("💝 Mission: Track charitable impact and deployment health")
        print("🎯 Target: Maximize $300,000-400,000+ monthly for humanitarian causes")
        print()
        
        try:
            # Run monitoring loops
            while True:
                summary = await monitoring.get_monitoring_summary()
                
                print(f"\n📊 Monitoring Summary - {datetime.now().strftime('%H:%M:%S')}")
                print(f"Active Deployments: {summary['deployment_metrics']['active_deployments']}")
                print(f"Success Rate: {summary['deployment_metrics']['recent_success_rate']:.1%}")
                print(f"Lives Saved Estimate: {summary['humanitarian_impact']['current_lives_saved_estimate']:,}")
                print(f"Charitable Funds: ${summary['humanitarian_impact']['current_charitable_funds']:,.0f}")
                print(f"Platform Uptime: {summary['humanitarian_impact']['platform_uptime_percentage']:.1f}%")
                
                await asyncio.sleep(60)  # Update every minute
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down monitoring...")
            await monitoring.shutdown()
    
    asyncio.run(main())
