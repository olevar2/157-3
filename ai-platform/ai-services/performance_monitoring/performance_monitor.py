"""
Platform3 AI Performance Monitoring Service
Real-time monitoring and optimization of AI model performance
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import psutil
import numpy as np
from collections import deque, defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "model-registry"))
from model_registry import get_registry

class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    CONFIDENCE = "confidence"

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    model_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None

@dataclass
class ModelPerformanceSnapshot:
    """Complete performance snapshot for a model"""
    model_id: str
    timestamp: datetime
    latency_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    accuracy: Optional[float] = None
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    error_rate: Optional[float] = None
    confidence: Optional[float] = None
    active_requests: int = 0

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class PerformanceAlert:
    """Performance alert"""
    model_id: str
    metric_type: MetricType
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime

class AIPerformanceMonitor:
    """
    Advanced performance monitoring for AI models
    Tracks latency, throughput, accuracy, resource usage, and more
    """
    
    def __init__(self, 
                 retention_hours: int = 24,
                 alert_check_interval: int = 60):
        self.registry = get_registry()
        self.retention_hours = retention_hours
        self.alert_check_interval = alert_check_interval
        
        # Metric storage (model_id -> metric_type -> deque of values)
        self.metrics: Dict[str, Dict[MetricType, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=10000)))
        
        # Real-time snapshots
        self.current_snapshots: Dict[str, ModelPerformanceSnapshot] = {}
        
        # Alert configuration
        self.alert_thresholds: Dict[str, Dict[MetricType, Dict[str, float]]] = {}
        self.active_alerts: List[PerformanceAlert] = []
        
        # Request tracking
        self.request_times: Dict[str, Dict[str, float]] = defaultdict(dict)  # model_id -> request_id -> start_time
        self.request_counters: Dict[str, int] = defaultdict(int)
        
        # System monitoring
        self.system_stats = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'timestamp': deque(maxlen=1000)
        }
        
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring threads
        self.monitoring_active = True
        self.system_monitor_thread = threading.Thread(target=self._system_monitor, daemon=True)
        self.alert_monitor_thread = threading.Thread(target=self._alert_monitor, daemon=True)
        
        self.system_monitor_thread.start()
        self.alert_monitor_thread.start()
        
        # Set default alert thresholds
        self._set_default_thresholds()
    
    def _set_default_thresholds(self):
        """Set default alert thresholds"""
        default_thresholds = {
            MetricType.LATENCY: {'warning': 1000.0, 'critical': 5000.0},  # ms
            MetricType.THROUGHPUT: {'warning': 10.0, 'critical': 5.0},    # requests/sec (lower is worse)
            MetricType.ERROR_RATE: {'warning': 0.05, 'critical': 0.1},    # 5% warning, 10% critical
            MetricType.MEMORY_USAGE: {'warning': 1024.0, 'critical': 2048.0},  # MB
            MetricType.CPU_USAGE: {'warning': 80.0, 'critical': 95.0},    # %
            MetricType.ACCURACY: {'warning': 0.8, 'critical': 0.7},       # accuracy (lower is worse)
        }
        
        # Apply to all models
        for model_info in self.registry.list_models():
            self.alert_thresholds[model_info.model_id] = default_thresholds.copy()
    
    def start_request(self, model_id: str, request_id: str):
        """Mark the start of a model request"""
        with self.lock:
            self.request_times[model_id][request_id] = time.time()
            self.request_counters[model_id] += 1
    
    def end_request(self, model_id: str, request_id: str, 
                   success: bool = True, 
                   accuracy: Optional[float] = None,
                   confidence: Optional[float] = None):
        """Mark the end of a model request and record metrics"""
        
        with self.lock:
            if model_id in self.request_times and request_id in self.request_times[model_id]:
                start_time = self.request_times[model_id][request_id]
                latency_ms = (time.time() - start_time) * 1000
                
                # Record latency
                self.record_metric(model_id, MetricType.LATENCY, latency_ms)
                
                # Record accuracy if provided
                if accuracy is not None:
                    self.record_metric(model_id, MetricType.ACCURACY, accuracy)
                
                # Record confidence if provided
                if confidence is not None:
                    self.record_metric(model_id, MetricType.CONFIDENCE, confidence)
                
                # Update error rate
                if not success:
                    self.record_metric(model_id, MetricType.ERROR_RATE, 1.0)
                else:
                    self.record_metric(model_id, MetricType.ERROR_RATE, 0.0)
                
                # Clean up
                del self.request_times[model_id][request_id]
                
                # Update current snapshot
                self._update_current_snapshot(model_id)
    
    def record_metric(self, model_id: str, metric_type: MetricType, value: float, 
                     context: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        
        metric = PerformanceMetric(
            model_id=model_id,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            context=context
        )
        
        with self.lock:
            self.metrics[model_id][metric_type].append(metric)
            
            # Update current snapshot
            self._update_current_snapshot(model_id)
        
        self.logger.debug(f"Recorded {metric_type.value} = {value} for model {model_id}")
    
    def _update_current_snapshot(self, model_id: str):
        """Update current performance snapshot for a model"""
        
        snapshot = ModelPerformanceSnapshot(
            model_id=model_id,
            timestamp=datetime.now()
        )
        
        # Calculate current metrics from recent data
        cutoff_time = datetime.now() - timedelta(minutes=5)  # Last 5 minutes
        
        for metric_type in MetricType:
            if metric_type in self.metrics[model_id]:
                recent_metrics = [m for m in self.metrics[model_id][metric_type] 
                               if m.timestamp >= cutoff_time]
                
                if recent_metrics:
                    values = [m.value for m in recent_metrics]
                    
                    if metric_type == MetricType.LATENCY:
                        snapshot.latency_ms = np.mean(values)
                    elif metric_type == MetricType.THROUGHPUT:
                        snapshot.throughput_rps = np.mean(values)
                    elif metric_type == MetricType.ACCURACY:
                        snapshot.accuracy = np.mean(values)
                    elif metric_type == MetricType.MEMORY_USAGE:
                        snapshot.memory_mb = np.mean(values)
                    elif metric_type == MetricType.CPU_USAGE:
                        snapshot.cpu_percent = np.mean(values)
                    elif metric_type == MetricType.ERROR_RATE:
                        snapshot.error_rate = np.mean(values)
                    elif metric_type == MetricType.CONFIDENCE:
                        snapshot.confidence = np.mean(values)
        
        # Active requests
        snapshot.active_requests = len(self.request_times.get(model_id, {}))
        
        self.current_snapshots[model_id] = snapshot
    
    def get_current_performance(self, model_id: str) -> Optional[ModelPerformanceSnapshot]:
        """Get current performance snapshot for a model"""
        return self.current_snapshots.get(model_id)
    
    def get_performance_history(self, model_id: str, 
                              metric_type: MetricType,
                              hours: int = 1) -> List[PerformanceMetric]:
        """Get performance history for a model and metric type"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            if model_id in self.metrics and metric_type in self.metrics[model_id]:
                return [m for m in self.metrics[model_id][metric_type] 
                       if m.timestamp >= cutoff_time]
        
        return []
    
    def get_performance_summary(self, model_id: str, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        summary = {
            'model_id': model_id,
            'time_window_hours': hours,
            'metrics': {}
        }
        
        with self.lock:
            if model_id in self.metrics:
                for metric_type in MetricType:
                    if metric_type in self.metrics[model_id]:
                        recent_metrics = [m for m in self.metrics[model_id][metric_type] 
                                        if m.timestamp >= cutoff_time]
                        
                        if recent_metrics:
                            values = [m.value for m in recent_metrics]
                            summary['metrics'][metric_type.value] = {
                                'count': len(values),
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values)),
                                'min': float(np.min(values)),
                                'max': float(np.max(values)),
                                'p50': float(np.percentile(values, 50)),
                                'p95': float(np.percentile(values, 95)),
                                'p99': float(np.percentile(values, 99))
                            }
        
        return summary
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance"""
        
        with self.lock:
            if self.system_stats['cpu_usage']:
                recent_cpu = list(self.system_stats['cpu_usage'])[-100:]  # Last 100 readings
                recent_memory = list(self.system_stats['memory_usage'])[-100:]
                
                return {
                    'cpu_usage': {
                        'current': recent_cpu[-1] if recent_cpu else 0,
                        'mean': float(np.mean(recent_cpu)) if recent_cpu else 0,
                        'max': float(np.max(recent_cpu)) if recent_cpu else 0
                    },
                    'memory_usage': {
                        'current': recent_memory[-1] if recent_memory else 0,
                        'mean': float(np.mean(recent_memory)) if recent_memory else 0,
                        'max': float(np.max(recent_memory)) if recent_memory else 0
                    },
                    'active_models': len(self.current_snapshots),
                    'total_active_requests': sum(len(requests) for requests in self.request_times.values())
                }
        
        return {}
    
    def _system_monitor(self):
        """Background system monitoring"""
        while self.monitoring_active:
            try:
                # Get system stats
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Store system stats
                with self.lock:
                    self.system_stats['cpu_usage'].append(cpu_percent)
                    self.system_stats['memory_usage'].append(memory_percent)
                    self.system_stats['timestamp'].append(datetime.now())
                
                # Calculate throughput for active models
                self._calculate_throughput()
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in system monitor: {e}")
                time.sleep(30)
    
    def _calculate_throughput(self):
        """Calculate throughput for models"""
        current_time = time.time()
        
        for model_id in self.request_counters:
            # Count recent requests
            recent_count = 0
            cutoff_time = current_time - 60  # Last minute
            
            if model_id in self.metrics and MetricType.LATENCY in self.metrics[model_id]:
                recent_requests = [m for m in self.metrics[model_id][MetricType.LATENCY]
                                 if (current_time - m.timestamp.timestamp()) <= 60]
                recent_count = len(recent_requests)
            
            # Calculate throughput (requests per second)
            throughput = recent_count / 60.0
            self.record_metric(model_id, MetricType.THROUGHPUT, throughput)
    
    def _alert_monitor(self):
        """Background alert monitoring"""
        while self.monitoring_active:
            try:
                self._check_alerts()
                time.sleep(self.alert_check_interval)
            except Exception as e:
                self.logger.error(f"Error in alert monitor: {e}")
                time.sleep(60)
    
    def _check_alerts(self):
        """Check for performance alerts"""
        current_time = datetime.now()
        
        for model_id, snapshot in self.current_snapshots.items():
            if model_id not in self.alert_thresholds:
                continue
            
            thresholds = self.alert_thresholds[model_id]
            
            # Check each metric
            for metric_type, metric_thresholds in thresholds.items():
                value = None
                
                if metric_type == MetricType.LATENCY and snapshot.latency_ms is not None:
                    value = snapshot.latency_ms
                elif metric_type == MetricType.THROUGHPUT and snapshot.throughput_rps is not None:
                    value = snapshot.throughput_rps
                elif metric_type == MetricType.ERROR_RATE and snapshot.error_rate is not None:
                    value = snapshot.error_rate
                elif metric_type == MetricType.MEMORY_USAGE and snapshot.memory_mb is not None:
                    value = snapshot.memory_mb
                elif metric_type == MetricType.CPU_USAGE and snapshot.cpu_percent is not None:
                    value = snapshot.cpu_percent
                elif metric_type == MetricType.ACCURACY and snapshot.accuracy is not None:
                    value = snapshot.accuracy
                
                if value is not None:
                    self._check_metric_thresholds(model_id, metric_type, value, metric_thresholds)
    
    def _check_metric_thresholds(self, model_id: str, metric_type: MetricType, 
                                value: float, thresholds: Dict[str, float]):
        """Check if a metric value triggers alerts"""
        
        alert_level = None
        threshold = None
        
        # Different logic for different metrics
        if metric_type in [MetricType.LATENCY, MetricType.ERROR_RATE, MetricType.MEMORY_USAGE, MetricType.CPU_USAGE]:
            # Higher values are worse
            if value >= thresholds.get('critical', float('inf')):
                alert_level = AlertLevel.CRITICAL
                threshold = thresholds['critical']
            elif value >= thresholds.get('warning', float('inf')):
                alert_level = AlertLevel.WARNING
                threshold = thresholds['warning']
        else:
            # Lower values are worse (throughput, accuracy)
            if value <= thresholds.get('critical', 0):
                alert_level = AlertLevel.CRITICAL
                threshold = thresholds['critical']
            elif value <= thresholds.get('warning', 0):
                alert_level = AlertLevel.WARNING
                threshold = thresholds['warning']
        
        if alert_level:
            alert = PerformanceAlert(
                model_id=model_id,
                metric_type=metric_type,
                level=alert_level,
                message=f"{metric_type.value} {alert_level.value}: {value:.2f} (threshold: {threshold})",
                value=value,
                threshold=threshold,
                timestamp=datetime.now()
            )
            
            self.active_alerts.append(alert)
            self.logger.warning(f"Performance alert: {alert.message}")
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts"""
        # Clean up old alerts (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.active_alerts = [alert for alert in self.active_alerts 
                             if alert.timestamp >= cutoff_time]
        
        return self.active_alerts.copy()
    
    def set_alert_threshold(self, model_id: str, metric_type: MetricType, 
                           warning: Optional[float] = None, 
                           critical: Optional[float] = None):
        """Set custom alert thresholds for a model"""
        
        if model_id not in self.alert_thresholds:
            self.alert_thresholds[model_id] = {}
        
        if metric_type not in self.alert_thresholds[model_id]:
            self.alert_thresholds[model_id][metric_type] = {}
        
        if warning is not None:
            self.alert_thresholds[model_id][metric_type]['warning'] = warning
        
        if critical is not None:
            self.alert_thresholds[model_id][metric_type]['critical'] = critical
    
    def export_metrics(self, file_path: str, hours: int = 24):
        """Export metrics to JSON file"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_window_hours': hours,
            'models': {}
        }
        
        with self.lock:
            for model_id in self.metrics:
                model_data = {'metrics': {}}
                
                for metric_type in MetricType:
                    if metric_type in self.metrics[model_id]:
                        recent_metrics = [m for m in self.metrics[model_id][metric_type] 
                                        if m.timestamp >= cutoff_time]
                        
                        model_data['metrics'][metric_type.value] = [
                            {
                                'value': m.value,
                                'timestamp': m.timestamp.isoformat(),
                                'context': m.context
                            }
                            for m in recent_metrics
                        ]
                
                export_data['models'][model_id] = model_data
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {file_path}")
    
    def shutdown(self):
        """Shutdown the performance monitor"""
        self.monitoring_active = False
        
        if self.system_monitor_thread.is_alive():
            self.system_monitor_thread.join(timeout=5)
        
        if self.alert_monitor_thread.is_alive():
            self.alert_monitor_thread.join(timeout=5)

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> AIPerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = AIPerformanceMonitor()
    return _performance_monitor

if __name__ == "__main__":
    # Test the performance monitor
    monitor = AIPerformanceMonitor()
    
    # Simulate some metrics
    monitor.record_metric("test_model", MetricType.LATENCY, 150.0)
    monitor.record_metric("test_model", MetricType.ACCURACY, 0.95)
    
    # Get summary
    summary = monitor.get_performance_summary("test_model")
    print(f"Performance Summary: {json.dumps(summary, indent=2)}")
