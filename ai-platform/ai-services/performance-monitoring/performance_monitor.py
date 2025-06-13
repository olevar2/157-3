"""
Performance Monitor Service - Humanitarian Trading Platform
Real-time monitoring of AI model performance for charitable mission

This service tracks the performance of all AI models generating profits for:
- Emergency medical aid for the poor
- Children's surgical procedures
- Global poverty alleviation
- Food security for struggling families

Author: Platform3 Humanitarian AI Team
Version: 1.0.0 - Monitoring for maximum charitable impact
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import psutil
import numpy as np
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    DRAWDOWN = "drawdown"
    CHARITABLE_IMPACT = "charitable_impact"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    model_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemPerformance:
    """System-wide performance metrics"""
    cpu_usage: Dict[str, float]
    memory_usage: Dict[str, float]
    disk_usage: Dict[str, float]
    network_io: Dict[str, float]
    active_models: int
    total_requests: int
    timestamp: datetime

@dataclass
class Alert:
    """Performance alert"""
    alert_id: str
    model_id: Optional[str]
    metric_type: MetricType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class ModelPerformanceStats:
    """Comprehensive model performance statistics"""
    model_id: str
    total_predictions: int
    avg_latency_ms: float
    error_rate: float
    accuracy: Optional[float]
    throughput_per_second: float
    last_prediction: datetime
    
    # Trading-specific metrics
    total_trades: int
    win_rate: Optional[float]
    profit_factor: Optional[float]
    max_drawdown: Optional[float]
    sharpe_ratio: Optional[float]
    
    # Humanitarian impact
    charitable_profit: float
    medical_procedures_funded: int
    families_helped: int
    
    # Resource utilization
    avg_cpu_usage: float
    avg_memory_usage_mb: float
    
    # Time windows
    last_hour_stats: Dict[str, float]
    last_day_stats: Dict[str, float]
    last_week_stats: Dict[str, float]

class PerformanceMonitor:
    """
    🏥 HUMANITARIAN AI PERFORMANCE MONITOR
    
    Real-time monitoring of all AI models serving our charitable mission.
    Tracks performance metrics and ensures optimal humanitarian impact.
    """
    
    def __init__(self, 
                 db_path: str = None,
                 alert_thresholds: Optional[Dict[MetricType, float]] = None):
        if db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            db_path = os.path.join(project_root, "data", "performance_monitor.db")
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Performance data storage
        self._metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._model_stats: Dict[str, ModelPerformanceStats] = {}
        self._active_alerts: Dict[str, Alert] = {}
        
        # Alert thresholds for humanitarian mission protection
        self.alert_thresholds = alert_thresholds or {
            MetricType.ERROR_RATE: 0.05,  # 5% error rate threshold
            MetricType.LATENCY: 1000.0,   # 1 second latency threshold
            MetricType.CPU_USAGE: 80.0,   # 80% CPU usage threshold
            MetricType.MEMORY_USAGE: 85.0, # 85% memory usage threshold
            MetricType.DRAWDOWN: 15.0,    # 15% drawdown threshold (humanitarian fund protection)
            MetricType.WIN_RATE: 0.40     # 40% minimum win rate for charitable sustainability
        }
        
        # Monitoring configuration
        self.monitoring_active = True
        self.collection_interval = 10  # seconds
        self.cleanup_interval = 3600   # 1 hour
        
        # Initialize database and start monitoring
        self._init_database()
        self._start_monitoring_thread()
        
        self.logger.info("🚀 Performance Monitor initialized for humanitarian trading mission")
    
    def _init_database(self):
        """Initialize SQLite database for performance data"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_db_connection() as conn:
            # Metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    metric_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            """)
            
            # System performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cpu_usage TEXT NOT NULL,
                    memory_usage TEXT NOT NULL,
                    disk_usage TEXT NOT NULL,
                    network_io TEXT NOT NULL,
                    active_models INTEGER NOT NULL,
                    total_requests INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_model_id ON metrics(model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._collect_system_metrics()
                    self._check_alert_conditions()
                    self._cleanup_old_data()
                    time.sleep(self.collection_interval)
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(self.collection_interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("📊 Performance monitoring thread started")
    
    def _collect_system_metrics(self):
        """Collect system-wide performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_per_core = psutil.cpu_percent(percpu=True)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_io = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            except:
                network_io = {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
            
            system_perf = SystemPerformance(
                cpu_usage={
                    'current': cpu_percent,
                    'per_core': cpu_per_core,
                    'average': sum(cpu_per_core) / len(cpu_per_core)
                },
                memory_usage={
                    'current': memory.percent,
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'total_gb': memory.total / (1024**3)
                },
                disk_usage={
                    'current': (disk.used / disk.total) * 100,
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'total_gb': disk.total / (1024**3)
                },
                network_io=network_io,
                active_models=len([stats for stats in self._model_stats.values() if stats.last_prediction > datetime.now() - timedelta(hours=1)]),
                total_requests=sum(stats.total_predictions for stats in self._model_stats.values()),
                timestamp=datetime.now()
            )
            
            # Save to database
            self._save_system_performance(system_perf)
            
            # Check for system-level alerts
            self._check_system_alerts(system_perf)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _save_system_performance(self, perf: SystemPerformance):
        """Save system performance data to database"""
        with self._get_db_connection() as conn:
            conn.execute("""
                INSERT INTO system_performance 
                (cpu_usage, memory_usage, disk_usage, network_io, active_models, total_requests, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                json.dumps(perf.cpu_usage),
                json.dumps(perf.memory_usage),
                json.dumps(perf.disk_usage),
                json.dumps(perf.network_io),
                perf.active_models,
                perf.total_requests,
                perf.timestamp.isoformat()
            ))
    
    def start_request(self, model_id: str, request_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Start tracking a prediction request"""
        start_time = time.perf_counter()
        
        # Store request start time
        if not hasattr(self, '_active_requests'):
            self._active_requests = {}
        
        self._active_requests[request_id] = {
            'model_id': model_id,
            'start_time': start_time,
            'metadata': metadata or {}
        }
    
    def end_request(self, model_id: str, request_id: str, success: bool = True, result: Any = None):
        """End tracking a prediction request and record metrics"""
        if not hasattr(self, '_active_requests') or request_id not in self._active_requests:
            self.logger.warning(f"Request {request_id} not found in active requests")
            return
        
        request_info = self._active_requests.pop(request_id)
        end_time = time.perf_counter()
        latency_ms = (end_time - request_info['start_time']) * 1000
        
        # Record metrics
        self.record_metric(model_id, MetricType.LATENCY, latency_ms)
        
        if not success:
            self.record_metric(model_id, MetricType.ERROR_RATE, 1.0)
        
        # Update model statistics
        self._update_model_stats(model_id, latency_ms, success, result)
        
        self.logger.debug(f"📊 Request {request_id} completed in {latency_ms:.2f}ms (success: {success})")
    
    def record_metric(self, 
                     model_id: str, 
                     metric_type: MetricType, 
                     value: float,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            model_id=model_id,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Add to buffer
        self._metrics_buffer[f"{model_id}_{metric_type.value}"].append(metric)
        
        # Save to database (async to avoid blocking)
        threading.Thread(target=self._save_metric_to_db, args=(metric,), daemon=True).start()
        
        # Check alert conditions
        self._check_metric_alert(metric)
    
    def _save_metric_to_db(self, metric: PerformanceMetric):
        """Save metric to database"""
        try:
            with self._get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO metrics (model_id, metric_type, value, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric.model_id,
                    metric.metric_type.value,
                    metric.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.metadata)
                ))
        except Exception as e:
            self.logger.error(f"Error saving metric to database: {e}")
    
    def _update_model_stats(self, model_id: str, latency_ms: float, success: bool, result: Any):
        """Update model performance statistics"""
        with self._lock:
            if model_id not in self._model_stats:
                self._model_stats[model_id] = ModelPerformanceStats(
                    model_id=model_id,
                    total_predictions=0,
                    avg_latency_ms=0.0,
                    error_rate=0.0,
                    accuracy=None,
                    throughput_per_second=0.0,
                    last_prediction=datetime.now(),
                    total_trades=0,
                    win_rate=None,
                    profit_factor=None,
                    max_drawdown=None,
                    sharpe_ratio=None,
                    charitable_profit=0.0,
                    medical_procedures_funded=0,
                    families_helped=0,
                    avg_cpu_usage=0.0,
                    avg_memory_usage_mb=0.0,
                    last_hour_stats={},
                    last_day_stats={},
                    last_week_stats={}
                )
            
            stats = self._model_stats[model_id]
            
            # Update basic stats
            stats.total_predictions += 1
            stats.last_prediction = datetime.now()
            
            # Update latency (exponential moving average)
            alpha = 0.1  # Smoothing factor
            stats.avg_latency_ms = alpha * latency_ms + (1 - alpha) * stats.avg_latency_ms
            
            # Update error rate
            if success:
                stats.error_rate = alpha * 0.0 + (1 - alpha) * stats.error_rate
            else:
                stats.error_rate = alpha * 1.0 + (1 - alpha) * stats.error_rate
            
            # Calculate throughput (requests per second over last minute)
            one_minute_ago = datetime.now() - timedelta(minutes=1)
            recent_metrics = [m for m in self._metrics_buffer[f"{model_id}_latency"] 
                            if m.timestamp > one_minute_ago]
            stats.throughput_per_second = len(recent_metrics) / 60.0
    
    def get_model_performance(self, model_id: str) -> Optional[ModelPerformanceStats]:
        """Get comprehensive performance statistics for a model"""
        return self._model_stats.get(model_id)
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage': {'current': cpu_percent},
                'memory_usage': {'current': memory.percent},
                'active_models': len([stats for stats in self._model_stats.values() 
                                    if stats.last_prediction > datetime.now() - timedelta(hours=1)]),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting system performance: {e}")
            return {'cpu_usage': {'current': 0}, 'memory_usage': {'current': 0}, 'active_models': 0}
    
    def _check_alert_conditions(self):
        """Check all alert conditions"""
        current_time = datetime.now()
        
        # Check model-specific alerts
        for model_id, stats in self._model_stats.items():
            # Check error rate
            if stats.error_rate > self.alert_thresholds[MetricType.ERROR_RATE]:
                self._create_alert(
                    model_id=model_id,
                    metric_type=MetricType.ERROR_RATE,
                    severity=AlertSeverity.WARNING,
                    message=f"High error rate: {stats.error_rate:.2%}",
                    value=stats.error_rate,
                    threshold=self.alert_thresholds[MetricType.ERROR_RATE]
                )
            
            # Check latency
            if stats.avg_latency_ms > self.alert_thresholds[MetricType.LATENCY]:
                self._create_alert(
                    model_id=model_id,
                    metric_type=MetricType.LATENCY,
                    severity=AlertSeverity.WARNING,
                    message=f"High latency: {stats.avg_latency_ms:.2f}ms",
                    value=stats.avg_latency_ms,
                    threshold=self.alert_thresholds[MetricType.LATENCY]
                )
            
            # Check win rate for humanitarian fund protection
            if stats.win_rate and stats.win_rate < self.alert_thresholds[MetricType.WIN_RATE]:
                self._create_alert(
                    model_id=model_id,
                    metric_type=MetricType.WIN_RATE,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Low win rate threatening charitable funding: {stats.win_rate:.2%}",
                    value=stats.win_rate,
                    threshold=self.alert_thresholds[MetricType.WIN_RATE]
                )
    
    def _check_system_alerts(self, perf: SystemPerformance):
        """Check system-level alert conditions"""
        # CPU usage alert
        if perf.cpu_usage['current'] > self.alert_thresholds[MetricType.CPU_USAGE]:
            self._create_alert(
                model_id=None,
                metric_type=MetricType.CPU_USAGE,
                severity=AlertSeverity.WARNING,
                message=f"High system CPU usage: {perf.cpu_usage['current']:.1f}%",
                value=perf.cpu_usage['current'],
                threshold=self.alert_thresholds[MetricType.CPU_USAGE]
            )
        
        # Memory usage alert
        if perf.memory_usage['current'] > self.alert_thresholds[MetricType.MEMORY_USAGE]:
            self._create_alert(
                model_id=None,
                metric_type=MetricType.MEMORY_USAGE,
                severity=AlertSeverity.CRITICAL,
                message=f"High system memory usage: {perf.memory_usage['current']:.1f}%",
                value=perf.memory_usage['current'],
                threshold=self.alert_thresholds[MetricType.MEMORY_USAGE]
            )
    
    def _check_metric_alert(self, metric: PerformanceMetric):
        """Check if a specific metric triggers an alert"""
        threshold = self.alert_thresholds.get(metric.metric_type)
        if not threshold:
            return
        
        # Determine if alert should be triggered based on metric type
        should_alert = False
        if metric.metric_type in [MetricType.ERROR_RATE, MetricType.LATENCY, MetricType.CPU_USAGE, MetricType.MEMORY_USAGE, MetricType.DRAWDOWN]:
            should_alert = metric.value > threshold
        elif metric.metric_type in [MetricType.WIN_RATE, MetricType.ACCURACY]:
            should_alert = metric.value < threshold
        
        if should_alert:
            severity = AlertSeverity.CRITICAL if metric.metric_type in [MetricType.DRAWDOWN, MetricType.WIN_RATE] else AlertSeverity.WARNING
            
            self._create_alert(
                model_id=metric.model_id,
                metric_type=metric.metric_type,
                severity=severity,
                message=f"{metric.metric_type.value} threshold exceeded: {metric.value}",
                value=metric.value,
                threshold=threshold
            )
    
    def _create_alert(self, 
                     model_id: Optional[str],
                     metric_type: MetricType,
                     severity: AlertSeverity,
                     message: str,
                     value: float,
                     threshold: float):
        """Create a new alert"""
        alert_id = f"{model_id or 'system'}_{metric_type.value}_{int(time.time())}"
        
        # Check if similar alert already exists and is not resolved
        existing_alert_key = f"{model_id}_{metric_type.value}"
        if existing_alert_key in self._active_alerts:
            return  # Don't create duplicate alerts
        
        alert = Alert(
            alert_id=alert_id,
            model_id=model_id,
            metric_type=metric_type,
            severity=severity,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        self._active_alerts[existing_alert_key] = alert
        
        # Save to database
        self._save_alert_to_db(alert)
        
        # Log alert
        severity_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "emergency": "🆘"}
        self.logger.warning(f"{severity_emoji.get(severity.value, '⚠️')} ALERT: {message}")
        
        # For critical alerts affecting humanitarian mission, log with higher priority
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            self.logger.error(f"🏥 HUMANITARIAN MISSION ALERT: {message} - This may impact our ability to fund medical aid for the poor")
    
    def _save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        try:
            with self._get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO alerts 
                    (alert_id, model_id, metric_type, severity, message, value, threshold, timestamp, resolved, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.model_id,
                    alert.metric_type.value,
                    alert.severity.value,
                    alert.message,
                    alert.value,
                    alert.threshold,
                    alert.timestamp.isoformat(),
                    int(alert.resolved),
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ))
        except Exception as e:
            self.logger.error(f"Error saving alert to database: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self._active_alerts.values() if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        for key, alert in self._active_alerts.items():
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                # Update in database
                try:
                    with self._get_db_connection() as conn:
                        conn.execute("""
                            UPDATE alerts SET resolved = 1, resolved_at = ? WHERE alert_id = ?
                        """, (alert.resolved_at.isoformat(), alert_id))
                except Exception as e:
                    self.logger.error(f"Error updating alert in database: {e}")
                
                self.logger.info(f"✅ Resolved alert: {alert.message}")
                return True
        
        return False
    
    def _cleanup_old_data(self):
        """Clean up old performance data to prevent database bloat"""
        try:
            cutoff_time = datetime.now() - timedelta(days=30)  # Keep 30 days of data
            
            with self._get_db_connection() as conn:
                # Clean old metrics
                conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time.isoformat(),))
                
                # Clean old system performance data
                conn.execute("DELETE FROM system_performance WHERE timestamp < ?", (cutoff_time.isoformat(),))
                
                # Clean resolved alerts older than 7 days
                alert_cutoff = datetime.now() - timedelta(days=7)
                conn.execute("DELETE FROM alerts WHERE resolved = 1 AND resolved_at < ?", (alert_cutoff.isoformat(),))
            
            # Clean in-memory buffers
            for key, buffer in self._metrics_buffer.items():
                # Keep only recent metrics in memory
                recent_metrics = [m for m in buffer if m.timestamp > cutoff_time]
                buffer.clear()
                buffer.extend(recent_metrics)
            
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {e}")
    
    def get_humanitarian_impact_report(self) -> Dict[str, Any]:
        """Generate humanitarian impact report from model performance"""
        total_charitable_profit = sum(stats.charitable_profit for stats in self._model_stats.values())
        total_procedures_funded = sum(stats.medical_procedures_funded for stats in self._model_stats.values())
        total_families_helped = sum(stats.families_helped for stats in self._model_stats.values())
        
        # Calculate performance metrics
        active_models = [stats for stats in self._model_stats.values() 
                        if stats.last_prediction > datetime.now() - timedelta(hours=24)]
        
        avg_accuracy = sum(stats.accuracy or 0 for stats in active_models) / len(active_models) if active_models else 0
        avg_win_rate = sum(stats.win_rate or 0 for stats in active_models) / len(active_models) if active_models else 0
        
        return {
            'humanitarian_impact': {
                'total_charitable_profit': total_charitable_profit,
                'medical_procedures_funded': total_procedures_funded,
                'families_helped': total_families_helped,
                'average_profit_per_day': total_charitable_profit / 30,  # Assuming 30-day period
                'estimated_lives_saved': total_procedures_funded * 0.85  # Conservative estimate
            },
            'model_performance': {
                'active_models': len(active_models),
                'average_accuracy': avg_accuracy,
                'average_win_rate': avg_win_rate,
                'total_predictions': sum(stats.total_predictions for stats in self._model_stats.values()),
                'average_latency_ms': sum(stats.avg_latency_ms for stats in active_models) / len(active_models) if active_models else 0
            },
            'system_health': {
                'active_alerts': len(self.get_active_alerts()),
                'critical_alerts': len([a for a in self.get_active_alerts() if a.severity == AlertSeverity.CRITICAL]),
                'overall_health_score': self._calculate_health_score()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        active_alerts = self.get_active_alerts()
        
        # Start with perfect score
        score = 100.0
        
        # Deduct points for alerts
        for alert in active_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                score -= 20
            elif alert.severity == AlertSeverity.WARNING:
                score -= 10
            elif alert.severity == AlertSeverity.EMERGENCY:
                score -= 50
        
        # Factor in model performance
        active_models = [stats for stats in self._model_stats.values() 
                        if stats.last_prediction > datetime.now() - timedelta(hours=1)]
        
        if active_models:
            avg_error_rate = sum(stats.error_rate for stats in active_models) / len(active_models)
            score -= avg_error_rate * 100  # Deduct based on error rate
        
        return max(0.0, score)
    
    def shutdown(self):
        """Shutdown the performance monitor"""
        self.monitoring_active = False
        self.logger.info("🛑 Performance Monitor shutdown completed")
