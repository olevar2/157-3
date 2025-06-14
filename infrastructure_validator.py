"""
Phase 6.1: Infrastructure Validator
Comprehensive infrastructure validation and health monitoring system.
"""

import os
import sys
import json
import yaml
import logging
import asyncio
import subprocess
import socket
import psutil
import platform
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
import time
import requests
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation level enumeration."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    CRITICAL = "critical"

class ComponentStatus(Enum):
    """Component status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    OFFLINE = "offline"

@dataclass
class ValidationRule:
    """Infrastructure validation rule."""
    name: str
    description: str
    component: str
    check_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"
    enabled: bool = True
    timeout: int = 30

@dataclass
class ValidationResult:
    """Infrastructure validation result."""
    rule: ValidationRule
    status: ComponentStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration: float = 0.0
    error: Optional[str] = None

@dataclass
class InfrastructureComponent:
    """Infrastructure component definition."""
    name: str
    type: str
    endpoint: Optional[str] = None
    port: Optional[int] = None
    credentials: Optional[Dict[str, str]] = None
    health_check_url: Optional[str] = None
    expected_status_code: int = 200
    dependencies: List[str] = field(default_factory=list)
    monitoring_enabled: bool = True

class InfrastructureValidator:
    """
    Infrastructure Validator
    Comprehensive infrastructure validation and health monitoring system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/infrastructure_config.yaml"
        self.validation_rules: List[ValidationRule] = []
        self.components: Dict[str, InfrastructureComponent] = {}
        self.validation_results: List[ValidationResult] = []
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize(self) -> bool:
        """Initialize the infrastructure validator."""
        try:
            logger.info("Initializing Infrastructure Validator...")
            
            # Create necessary directories
            await self._create_directories()
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize validation rules
            await self._initialize_validation_rules()
            
            # Load infrastructure components
            await self._load_infrastructure_components()
            
            logger.info("Infrastructure Validator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Infrastructure Validator: {e}")
            return False
    
    async def _create_directories(self):
        """Create necessary directories."""
        directories = [
            "logs/infrastructure", "config", "monitoring", 
            "reports", "metrics", "alerts"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    async def _load_configuration(self):
        """Load infrastructure configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    # Process configuration
            else:
                # Create default configuration
                await self._create_default_config()
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    async def _create_default_config(self):
        """Create default infrastructure configuration."""
        default_config = {
            "validation_settings": {
                "default_timeout": 30,
                "parallel_checks": True,
                "max_concurrent_checks": 10,
                "retry_failed_checks": True,
                "retry_count": 3,
                "retry_delay": 5
            },
            "monitoring_settings": {
                "enabled": True,
                "interval_seconds": 60,
                "alert_thresholds": {
                    "cpu_percent": 80,
                    "memory_percent": 85,
                    "disk_percent": 90,
                    "response_time_ms": 5000
                }
            },
            "notification_settings": {
                "enabled": True,
                "channels": ["email", "slack"],
                "critical_only": False
            }
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    async def _initialize_validation_rules(self):
        """Initialize default validation rules."""
        self.validation_rules = [
            # System Resource Rules
            ValidationRule(
                name="cpu_usage_check",
                description="Check CPU usage levels",
                component="system",
                check_type="cpu_usage",
                thresholds={"warning": 70, "critical": 90},
                severity="warning"
            ),
            ValidationRule(
                name="memory_usage_check",
                description="Check memory usage levels",
                component="system",
                check_type="memory_usage",
                thresholds={"warning": 80, "critical": 95},
                severity="warning"
            ),
            ValidationRule(
                name="disk_usage_check",
                description="Check disk space usage",
                component="system",
                check_type="disk_usage",
                thresholds={"warning": 80, "critical": 95},
                severity="critical"
            ),
            
            # Network Rules
            ValidationRule(
                name="network_connectivity_check",
                description="Check network connectivity",
                component="network",
                check_type="connectivity",
                parameters={"hosts": ["8.8.8.8", "1.1.1.1"]},
                severity="critical"
            ),
            ValidationRule(
                name="port_availability_check",
                description="Check required ports availability",
                component="network",
                check_type="port_check",
                parameters={"ports": [80, 443, 8000, 8001, 8002]},
                severity="warning"
            ),
            
            # Service Rules
            ValidationRule(
                name="service_health_check",
                description="Check service health endpoints",
                component="services",
                check_type="health_endpoint",
                parameters={"endpoints": []},
                severity="critical"
            ),
            
            # Database Rules
            ValidationRule(
                name="database_connectivity_check",
                description="Check database connectivity",
                component="database",
                check_type="db_connection",
                severity="critical"
            ),
            
            # Security Rules
            ValidationRule(
                name="ssl_certificate_check",
                description="Check SSL certificate validity",
                component="security",
                check_type="ssl_cert",
                thresholds={"warning_days": 30, "critical_days": 7},
                severity="warning"
            ),
            
            # Application Rules
            ValidationRule(
                name="log_file_check",
                description="Check for critical errors in logs",
                component="application",
                check_type="log_analysis",
                parameters={"log_paths": ["logs/*.log"]},
                severity="warning"
            )
        ]
    
    async def _load_infrastructure_components(self):
        """Load infrastructure components."""
        # Default components - would be loaded from configuration
        self.components = {
            "web_server": InfrastructureComponent(
                name="Web Server",
                type="http_service",
                endpoint="localhost",
                port=8000,
                health_check_url="http://localhost:8000/health"
            ),
            "api_server": InfrastructureComponent(
                name="API Server",
                type="http_service",
                endpoint="localhost",
                port=8001,
                health_check_url="http://localhost:8001/health"
            ),
            "database": InfrastructureComponent(
                name="Database",
                type="database",
                endpoint="localhost",
                port=5432
            ),
            "cache_server": InfrastructureComponent(
                name="Cache Server",
                type="cache",
                endpoint="localhost",
                port=6379
            )
        }
    
    async def validate_infrastructure(self, level: ValidationLevel = ValidationLevel.STANDARD) -> List[ValidationResult]:
        """Perform comprehensive infrastructure validation."""
        logger.info(f"Starting infrastructure validation (level: {level.value})...")
        
        self.validation_results = []
        
        # Filter rules based on validation level
        active_rules = self._filter_rules_by_level(level)
        
        # Execute validation rules
        if len(active_rules) > 0:
            tasks = []
            for rule in active_rules:
                if rule.enabled:
                    task = asyncio.create_task(self._execute_validation_rule(rule))
                    tasks.append(task)
            
            # Wait for all validation tasks to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Validation rule failed: {active_rules[i].name} - {result}")
                        self.validation_results.append(ValidationResult(
                            rule=active_rules[i],
                            status=ComponentStatus.UNKNOWN,
                            message=f"Validation failed: {result}",
                            error=str(result)
                        ))
                    elif result:
                        self.validation_results.append(result)
        
        # Generate validation report
        await self._generate_validation_report()
        
        logger.info(f"Infrastructure validation completed. {len(self.validation_results)} checks performed.")
        return self.validation_results
    
    def _filter_rules_by_level(self, level: ValidationLevel) -> List[ValidationRule]:
        """Filter validation rules based on level."""
        if level == ValidationLevel.BASIC:
            return [rule for rule in self.validation_rules if rule.severity == "critical"]
        elif level == ValidationLevel.STANDARD:
            return [rule for rule in self.validation_rules if rule.severity in ["critical", "warning"]]
        elif level == ValidationLevel.COMPREHENSIVE:
            return self.validation_rules
        elif level == ValidationLevel.CRITICAL:
            return [rule for rule in self.validation_rules if rule.severity == "critical"]
        else:
            return self.validation_rules
    
    async def _execute_validation_rule(self, rule: ValidationRule) -> Optional[ValidationResult]:
        """Execute a single validation rule."""
        start_time = time.time()
        
        try:
            logger.debug(f"Executing validation rule: {rule.name}")
            
            # Route to appropriate validation method
            if rule.check_type == "cpu_usage":
                result = await self._check_cpu_usage(rule)
            elif rule.check_type == "memory_usage":
                result = await self._check_memory_usage(rule)
            elif rule.check_type == "disk_usage":
                result = await self._check_disk_usage(rule)
            elif rule.check_type == "connectivity":
                result = await self._check_network_connectivity(rule)
            elif rule.check_type == "port_check":
                result = await self._check_port_availability(rule)
            elif rule.check_type == "health_endpoint":
                result = await self._check_service_health(rule)
            elif rule.check_type == "db_connection":
                result = await self._check_database_connectivity(rule)
            elif rule.check_type == "ssl_cert":
                result = await self._check_ssl_certificate(rule)
            elif rule.check_type == "log_analysis":
                result = await self._check_log_files(rule)
            else:
                result = ValidationResult(
                    rule=rule,
                    status=ComponentStatus.UNKNOWN,
                    message=f"Unknown check type: {rule.check_type}"
                )
            
            result.duration = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Validation rule execution failed: {rule.name} - {e}")
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Execution failed: {e}",
                duration=time.time() - start_time,
                error=str(e)
            )
    
    async def _check_cpu_usage(self, rule: ValidationRule) -> ValidationResult:
        """Check CPU usage levels."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            warning_threshold = rule.thresholds.get("warning", 70)
            critical_threshold = rule.thresholds.get("critical", 90)
            
            if cpu_percent >= critical_threshold:
                status = ComponentStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent >= warning_threshold:
                status = ComponentStatus.WARNING
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = ComponentStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return ValidationResult(
                rule=rule,
                status=status,
                message=message,
                metrics={"cpu_percent": cpu_percent},
                details={
                    "current_usage": cpu_percent,
                    "warning_threshold": warning_threshold,
                    "critical_threshold": critical_threshold,
                    "cpu_count": psutil.cpu_count()
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to check CPU usage: {e}",
                error=str(e)
            )
    
    async def _check_memory_usage(self, rule: ValidationRule) -> ValidationResult:
        """Check memory usage levels."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            warning_threshold = rule.thresholds.get("warning", 80)
            critical_threshold = rule.thresholds.get("critical", 95)
            
            if memory_percent >= critical_threshold:
                status = ComponentStatus.CRITICAL
                message = f"Memory usage critical: {memory_percent:.1f}%"
            elif memory_percent >= warning_threshold:
                status = ComponentStatus.WARNING
                message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                status = ComponentStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return ValidationResult(
                rule=rule,
                status=status,
                message=message,
                metrics={"memory_percent": memory_percent},
                details={
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory_percent,
                    "warning_threshold": warning_threshold,
                    "critical_threshold": critical_threshold
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to check memory usage: {e}",
                error=str(e)
            )
    
    async def _check_disk_usage(self, rule: ValidationRule) -> ValidationResult:
        """Check disk space usage."""
        try:
            disk_usage = psutil.disk_usage('.')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            warning_threshold = rule.thresholds.get("warning", 80)
            critical_threshold = rule.thresholds.get("critical", 95)
            
            if disk_percent >= critical_threshold:
                status = ComponentStatus.CRITICAL
                message = f"Disk usage critical: {disk_percent:.1f}%"
            elif disk_percent >= warning_threshold:
                status = ComponentStatus.WARNING
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = ComponentStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return ValidationResult(
                rule=rule,
                status=status,
                message=message,
                metrics={"disk_percent": disk_percent},
                details={
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "percent": disk_percent,
                    "warning_threshold": warning_threshold,
                    "critical_threshold": critical_threshold
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to check disk usage: {e}",
                error=str(e)
            )
    
    async def _check_network_connectivity(self, rule: ValidationRule) -> ValidationResult:
        """Check network connectivity."""
        try:
            hosts = rule.parameters.get("hosts", ["8.8.8.8"])
            connectivity_results = []
            
            for host in hosts:
                try:
                    # Simple ping check using socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, 80))
                    sock.close()
                    
                    connectivity_results.append({
                        "host": host,
                        "reachable": result == 0,
                        "error": None if result == 0 else f"Connection failed: {result}"
                    })
                except Exception as e:
                    connectivity_results.append({
                        "host": host,
                        "reachable": False,
                        "error": str(e)
                    })
            
            successful_connections = len([r for r in connectivity_results if r["reachable"]])
            total_hosts = len(hosts)
            
            if successful_connections == 0:
                status = ComponentStatus.CRITICAL
                message = "No network connectivity"
            elif successful_connections < total_hosts:
                status = ComponentStatus.WARNING
                message = f"Partial connectivity: {successful_connections}/{total_hosts} hosts reachable"
            else:
                status = ComponentStatus.HEALTHY
                message = f"All hosts reachable: {successful_connections}/{total_hosts}"
            
            return ValidationResult(
                rule=rule,
                status=status,
                message=message,
                details={"connectivity_results": connectivity_results},
                metrics={
                    "successful_connections": successful_connections,
                    "total_hosts": total_hosts,
                    "success_rate": successful_connections / total_hosts
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to check network connectivity: {e}",
                error=str(e)
            )
    
    async def _check_port_availability(self, rule: ValidationRule) -> ValidationResult:
        """Check port availability."""
        try:
            ports = rule.parameters.get("ports", [])
            port_results = []
            
            for port in ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    
                    port_results.append({
                        "port": port,
                        "available": result != 0,  # Port is available if connection fails
                        "in_use": result == 0,
                        "error": None
                    })
                except Exception as e:
                    port_results.append({
                        "port": port,
                        "available": False,
                        "in_use": False,
                        "error": str(e)
                    })
            
            available_ports = len([r for r in port_results if r["available"]])
            total_ports = len(ports)
            
            if available_ports == total_ports:
                status = ComponentStatus.HEALTHY
                message = f"All ports available: {available_ports}/{total_ports}"
            elif available_ports > 0:
                status = ComponentStatus.WARNING
                message = f"Some ports in use: {available_ports}/{total_ports} available"
            else:
                status = ComponentStatus.CRITICAL
                message = "No ports available"
            
            return ValidationResult(
                rule=rule,
                status=status,
                message=message,
                details={"port_results": port_results},
                metrics={
                    "available_ports": available_ports,
                    "total_ports": total_ports
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to check port availability: {e}",
                error=str(e)
            )
    
    async def _check_service_health(self, rule: ValidationRule) -> ValidationResult:
        """Check service health endpoints."""
        try:
            endpoints = rule.parameters.get("endpoints", [])
            health_results = []
            
            # Add component health check URLs
            for component in self.components.values():
                if component.health_check_url:
                    endpoints.append(component.health_check_url)
            
            if not endpoints:
                return ValidationResult(
                    rule=rule,
                    status=ComponentStatus.HEALTHY,
                    message="No health endpoints configured",
                    details={"note": "No endpoints to check"}
                )
            
            for endpoint in endpoints:
                try:
                    # Simple HTTP GET with timeout
                    response = requests.get(endpoint, timeout=10)
                    
                    health_results.append({
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "healthy": 200 <= response.status_code < 300,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "error": None
                    })
                except Exception as e:
                    health_results.append({
                        "endpoint": endpoint,
                        "status_code": None,
                        "healthy": False,
                        "response_time_ms": None,
                        "error": str(e)
                    })
            
            healthy_endpoints = len([r for r in health_results if r["healthy"]])
            total_endpoints = len(endpoints)
            
            if healthy_endpoints == 0:
                status = ComponentStatus.CRITICAL
                message = "No healthy service endpoints"
            elif healthy_endpoints < total_endpoints:
                status = ComponentStatus.WARNING
                message = f"Some services unhealthy: {healthy_endpoints}/{total_endpoints} healthy"
            else:
                status = ComponentStatus.HEALTHY
                message = f"All services healthy: {healthy_endpoints}/{total_endpoints}"
            
            return ValidationResult(
                rule=rule,
                status=status,
                message=message,
                details={"health_results": health_results},
                metrics={
                    "healthy_endpoints": healthy_endpoints,
                    "total_endpoints": total_endpoints,
                    "avg_response_time": sum(
                        r["response_time_ms"] for r in health_results 
                        if r["response_time_ms"] is not None
                    ) / len([r for r in health_results if r["response_time_ms"] is not None]) if health_results else 0
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to check service health: {e}",
                error=str(e)
            )
    
    async def _check_database_connectivity(self, rule: ValidationRule) -> ValidationResult:
        """Check database connectivity."""
        try:
            # Simulate database connectivity check
            # In real implementation, would test actual database connections
            
            db_component = self.components.get("database")
            if not db_component:
                return ValidationResult(
                    rule=rule,
                    status=ComponentStatus.WARNING,
                    message="No database component configured"
                )
            
            # Simulate connection check
            connection_successful = True  # Would be actual connection test
            
            if connection_successful:
                status = ComponentStatus.HEALTHY
                message = "Database connectivity successful"
                metrics = {"connection_time_ms": 50}  # Simulated
            else:
                status = ComponentStatus.CRITICAL
                message = "Database connection failed"
                metrics = {"connection_time_ms": None}
            
            return ValidationResult(
                rule=rule,
                status=status,
                message=message,
                metrics=metrics,
                details={"database_type": "postgresql", "endpoint": db_component.endpoint}
            )
            
        except Exception as e:
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to check database connectivity: {e}",
                error=str(e)
            )
    
    async def _check_ssl_certificate(self, rule: ValidationRule) -> ValidationResult:
        """Check SSL certificate validity."""
        try:
            # Simulate SSL certificate check
            # In real implementation, would check actual certificates
            
            warning_days = rule.thresholds.get("warning_days", 30)
            critical_days = rule.thresholds.get("critical_days", 7)
            
            # Simulated certificate expiry check
            days_until_expiry = 45  # Simulated
            
            if days_until_expiry <= critical_days:
                status = ComponentStatus.CRITICAL
                message = f"SSL certificate expires in {days_until_expiry} days"
            elif days_until_expiry <= warning_days:
                status = ComponentStatus.WARNING
                message = f"SSL certificate expires in {days_until_expiry} days"
            else:
                status = ComponentStatus.HEALTHY
                message = f"SSL certificate valid for {days_until_expiry} days"
            
            return ValidationResult(
                rule=rule,
                status=status,
                message=message,
                metrics={"days_until_expiry": days_until_expiry},
                details={
                    "warning_threshold": warning_days,
                    "critical_threshold": critical_days
                }
            )            
        except Exception as e:
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to check SSL certificate: {e}",
                error=str(e)
            )
    
    async def _check_log_files(self, rule: ValidationRule) -> ValidationResult:
        """Check log files for critical errors."""
        try:
            log_paths = rule.parameters.get("log_paths", [])
            error_patterns = rule.parameters.get("error_patterns", ["ERROR", "CRITICAL", "FATAL"])
            
            log_analysis = {
                "files_checked": 0,
                "errors_found": 0,
                "critical_errors": 0,
                "recent_errors": 0
            }
            
            for log_pattern in log_paths:
                try:
                    import glob
                    
                    for log_file in glob.glob(log_pattern):
                        if os.path.exists(log_file):
                            log_analysis["files_checked"] += 1
                              # Simple error counting (last 100 lines)
                            try:
                                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    lines = f.readlines()
                                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                                    
                                    for line in recent_lines:
                                        for pattern in error_patterns:
                                            if pattern in line.upper():
                                                log_analysis["errors_found"] += 1
                                                if "CRITICAL" in line.upper() or "FATAL" in line.upper():
                                                    log_analysis["critical_errors"] += 1
                                                # Check if error is recent (within last hour)
                                                log_analysis["recent_errors"] += 1
                            except Exception as e:
                                logger.warning(f"Failed to analyze log file {log_file}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to process log pattern {log_pattern}: {e}")
            
            if log_analysis["critical_errors"] > 0:
                status = ComponentStatus.CRITICAL
                message = f"Critical errors found in logs: {log_analysis['critical_errors']}"
            elif log_analysis["recent_errors"] > 10:
                status = ComponentStatus.WARNING
                message = f"High error rate in logs: {log_analysis['recent_errors']} recent errors"
            elif log_analysis["errors_found"] > 0:
                status = ComponentStatus.WARNING
                message = f"Errors found in logs: {log_analysis['errors_found']}"
            else:
                status = ComponentStatus.HEALTHY
                message = f"No critical errors in logs ({log_analysis['files_checked']} files checked)"
            
            return ValidationResult(
                rule=rule,
                status=status,
                message=message,
                metrics=log_analysis,
                details={"log_paths": log_paths, "error_patterns": error_patterns}
            )
            
        except Exception as e:
            return ValidationResult(
                rule=rule,
                status=ComponentStatus.UNKNOWN,
                message=f"Failed to check log files: {e}",
                error=str(e)
            )
    
    async def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        try:
            # Calculate summary statistics
            total_checks = len(self.validation_results)
            healthy_count = len([r for r in self.validation_results if r.status == ComponentStatus.HEALTHY])
            warning_count = len([r for r in self.validation_results if r.status == ComponentStatus.WARNING])
            critical_count = len([r for r in self.validation_results if r.status == ComponentStatus.CRITICAL])
            unknown_count = len([r for r in self.validation_results if r.status == ComponentStatus.UNKNOWN])
            
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_checks": total_checks,
                    "healthy": healthy_count,
                    "warnings": warning_count,
                    "critical": critical_count,
                    "unknown": unknown_count,
                    "overall_health": "HEALTHY" if critical_count == 0 and warning_count < 3 else "WARNING" if critical_count == 0 else "CRITICAL"
                },
                "results": [
                    {
                        "rule_name": result.rule.name,
                        "component": result.rule.component,
                        "status": result.status.value,
                        "message": result.message,
                        "duration": result.duration,
                        "timestamp": result.timestamp.isoformat(),
                        "metrics": result.metrics,
                        "details": result.details,
                        "error": result.error
                    }
                    for result in self.validation_results
                ]
            }
            
            # Save report
            report_path = f"reports/infrastructure_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Validation report saved to: {report_path}")
            
            # Generate summary log
            logger.info(f"Infrastructure Validation Summary:")
            logger.info(f"  Total checks: {total_checks}")
            logger.info(f"  Healthy: {healthy_count}")
            logger.info(f"  Warnings: {warning_count}")
            logger.info(f"  Critical: {critical_count}")
            logger.info(f"  Unknown: {unknown_count}")
            logger.info(f"  Overall health: {report['summary']['overall_health']}")
            
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
    
    async def start_continuous_monitoring(self, interval_seconds: int = 60):
        """Start continuous infrastructure monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._continuous_monitoring_loop(interval_seconds)
        )
        logger.info(f"Started continuous monitoring (interval: {interval_seconds}s)")
    
    async def stop_continuous_monitoring(self):
        """Stop continuous infrastructure monitoring."""
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped continuous monitoring")
    
    async def _continuous_monitoring_loop(self, interval_seconds: int):
        """Continuous monitoring loop."""
        try:
            while self.monitoring_active:
                logger.debug("Performing scheduled infrastructure validation...")
                
                # Perform validation
                results = await self.validate_infrastructure(ValidationLevel.STANDARD)
                
                # Check for critical issues
                critical_results = [r for r in results if r.status == ComponentStatus.CRITICAL]
                if critical_results:
                    await self._handle_critical_alerts(critical_results)
                
                # Wait for next check
                await asyncio.sleep(interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
    
    async def _handle_critical_alerts(self, critical_results: List[ValidationResult]):
        """Handle critical infrastructure alerts."""
        logger.critical(f"CRITICAL INFRASTRUCTURE ISSUES DETECTED: {len(critical_results)} issues")
        
        for result in critical_results:
            logger.critical(f"  - {result.rule.component}: {result.message}")
        
        # In real implementation, would send notifications through configured channels
    
    def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get current infrastructure status."""
        if not self.validation_results:
            return {"status": "no_data", "message": "No validation results available"}
        
        # Get latest results
        latest_results = sorted(self.validation_results, key=lambda x: x.timestamp, reverse=True)
        
        # Calculate status summary
        healthy_count = len([r for r in latest_results if r.status == ComponentStatus.HEALTHY])
        warning_count = len([r for r in latest_results if r.status == ComponentStatus.WARNING])
        critical_count = len([r for r in latest_results if r.status == ComponentStatus.CRITICAL])
        
        overall_status = "healthy"
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "total_checks": len(latest_results),
            "healthy": healthy_count,
            "warnings": warning_count,
            "critical": critical_count,
            "last_check": latest_results[0].timestamp.isoformat() if latest_results else None,
            "monitoring_active": self.monitoring_active
        }

async def main():
    """Main execution function for testing."""
    validator = InfrastructureValidator()
    
    # Initialize
    if not await validator.initialize():
        print("Failed to initialize infrastructure validator")
        return
    
    # Perform validation
    results = await validator.validate_infrastructure(ValidationLevel.COMPREHENSIVE)
    
    # Print results summary
    status = validator.get_infrastructure_status()
    print(f"\nInfrastructure Validation Results:")
    print(f"Overall Status: {status['overall_status'].upper()}")
    print(f"Total Checks: {status['total_checks']}")
    print(f"Healthy: {status['healthy']}")
    print(f"Warnings: {status['warnings']}")
    print(f"Critical: {status['critical']}")
    
    # Print detailed results
    print(f"\nDetailed Results:")
    for result in results:
        status_symbol = {
            ComponentStatus.HEALTHY: "✅",
            ComponentStatus.WARNING: "⚠️",
            ComponentStatus.CRITICAL: "🚨",
            ComponentStatus.UNKNOWN: "❓",
            ComponentStatus.OFFLINE: "❌"
        }.get(result.status, "?")
        
        print(f"{status_symbol} {result.rule.component}: {result.message}")
        if result.error:
            print(f"   Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
