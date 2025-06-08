"""
🏥 HUMANITARIAN AI PLATFORM - PRODUCTION MONITORING & ALERTING SYSTEM
💝 Advanced monitoring for charitable trading mission

This service provides comprehensive monitoring and alerting for the humanitarian AI platform.
Ensures optimal performance for maximizing profits for medical aid, children's surgeries, and poverty relief.
"""

import asyncio
import time
import json
import logging
import psutil
import aiohttp
import smtplib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
import weakref
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"

class AlertType(Enum):
    """Types of alerts"""
    PERFORMANCE = "performance"
    SYSTEM = "system"
    TRADING = "trading"
    HUMANITARIAN = "humanitarian"
    SECURITY = "security"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metrics: Dict[str, Any] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class MetricThresholds:
    """Metric threshold configuration"""
    cpu_usage_warning: float = 70.0
    cpu_usage_critical: float = 85.0
    memory_usage_warning: float = 75.0
    memory_usage_critical: float = 90.0
    inference_latency_warning_ms: float = 5.0
    inference_latency_critical_ms: float = 10.0
    error_rate_warning: float = 1.0
    error_rate_critical: float = 5.0
    humanitarian_target_warning: float = 0.8  # 80% of target
    humanitarian_target_critical: float = 0.5  # 50% of target

@dataclass
class NotificationConfig:
    """Notification configuration"""
    email_enabled: bool = True
    slack_enabled: bool = False
    telegram_enabled: bool = False
    sms_enabled: bool = False
    webhook_enabled: bool = False
    
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    recipient_emails: List[str] = None
    
    # Slack settings
    slack_webhook_url: str = ""
    slack_channel: str = "#humanitarian-ai-alerts"
    
    # Telegram settings
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    def __post_init__(self):
        if self.recipient_emails is None:
            self.recipient_emails = []

class SystemMetrics:
    """System metrics collector"""
    
    def __init__(self):
        self.metrics_history = defaultdict(deque)
        self.max_history_size = 1000
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        # Network metrics
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        metrics = {
            "cpu_usage_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory_usage_percent": memory_percent,
            "memory_used_gb": memory_used_gb,
            "memory_total_gb": memory_total_gb,
            "disk_usage_percent": disk_percent,
            "disk_free_gb": disk_free_gb,
            "network_bytes_sent": network_bytes_sent,
            "network_bytes_recv": network_bytes_recv,
            "timestamp": time.time()
        }
        
        # Store in history
        for key, value in metrics.items():
            if key != "timestamp":
                self.metrics_history[key].append(value)
                if len(self.metrics_history[key]) > self.max_history_size:
                    self.metrics_history[key].popleft()
        
        return metrics
    
    def get_metric_trend(self, metric_name: str, window_minutes: int = 5) -> Dict[str, float]:
        """Get trend analysis for a metric"""
        if metric_name not in self.metrics_history:
            return {"trend": 0.0, "average": 0.0, "min": 0.0, "max": 0.0}
        
        values = list(self.metrics_history[metric_name])
        if len(values) < 2:
            return {"trend": 0.0, "average": values[0] if values else 0.0, "min": 0.0, "max": 0.0}
        
        # Calculate trend (simple linear regression slope)
        x = np.arange(len(values))
        y = np.array(values)
        trend = np.polyfit(x, y, 1)[0] if len(values) > 1 else 0.0
        
        return {
            "trend": float(trend),
            "average": float(np.mean(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "current": float(values[-1]),
            "samples": len(values)
        }

class TradingMetrics:
    """Trading performance metrics collector"""
    
    def __init__(self):
        self.trades_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.humanitarian_impact = {
            "total_profit": 0.0,
            "charitable_amount": 0.0,
            "medical_aids": 0,
            "surgeries_funded": 0,
            "families_fed": 0
        }
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a trade execution"""
        trade_data["timestamp"] = time.time()
        self.trades_history.append(trade_data)
        
        # Update humanitarian impact
        if "profit" in trade_data:
            profit = trade_data["profit"]
            charitable_amount = profit * 0.5  # 50% to charity
            
            self.humanitarian_impact["total_profit"] += profit
            self.humanitarian_impact["charitable_amount"] += charitable_amount
            
            # Calculate humanitarian impact
            if charitable_amount > 0:
                medical_aids = int(charitable_amount // 25)  # $25 per medical aid
                surgeries = int(charitable_amount // 500)    # $500 per surgery
                families = int(charitable_amount // 100)     # $100 per family/month
                
                self.humanitarian_impact["medical_aids"] += medical_aids
                self.humanitarian_impact["surgeries_funded"] += surgeries
                self.humanitarian_impact["families_fed"] += families
    
    def get_trading_metrics(self) -> Dict[str, Any]:
        """Get current trading metrics"""
        if not self.trades_history:
            return {
                "total_trades": 0,
                "profit_rate": 0.0,
                "average_profit": 0.0,
                "error_rate": 0.0,
                "humanitarian_impact": self.humanitarian_impact
            }
        
        trades = list(self.trades_history)
        profitable_trades = [t for t in trades if t.get("profit", 0) > 0]
        
        return {
            "total_trades": len(trades),
            "profitable_trades": len(profitable_trades),
            "profit_rate": len(profitable_trades) / len(trades) * 100,
            "average_profit": np.mean([t.get("profit", 0) for t in trades]),
            "total_profit": sum(t.get("profit", 0) for t in trades),
            "error_rate": len([t for t in trades if t.get("error", False)]) / len(trades) * 100,
            "humanitarian_impact": self.humanitarian_impact,
            "recent_trades": trades[-10:] if len(trades) >= 10 else trades
        }

class NotificationManager:
    """Manages alert notifications"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.notification_history = deque(maxlen=1000)
    
    async def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        try:
            # Email notification
            if self.config.email_enabled and self.config.recipient_emails:
                await self._send_email_alert(alert)
            
            # Slack notification
            if self.config.slack_enabled and self.config.slack_webhook_url:
                await self._send_slack_alert(alert)
            
            # Telegram notification
            if self.config.telegram_enabled and self.config.telegram_bot_token:
                await self._send_telegram_alert(alert)
            
            # Webhook notification
            if self.config.webhook_enabled:
                await self._send_webhook_alert(alert)
            
            # Record notification
            self.notification_history.append({
                "alert_id": alert.id,
                "timestamp": time.time(),
                "channels_sent": self._get_enabled_channels()
            })
            
            logger.info(f"✅ Alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send alert: {e}")
    
    def _get_enabled_channels(self) -> List[str]:
        """Get list of enabled notification channels"""
        channels = []
        if self.config.email_enabled:
            channels.append("email")
        if self.config.slack_enabled:
            channels.append("slack")
        if self.config.telegram_enabled:
            channels.append("telegram")
        return channels
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.config.recipient_emails:
            return
        
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.recipient_emails)
            msg['Subject'] = f"🏥 Humanitarian AI Alert: {alert.title}"
            
            # Email body
            body = f"""
🏥 HUMANITARIAN AI PLATFORM ALERT
💝 Charitable Trading Mission Alert

Alert Details:
- Type: {alert.type.value}
- Severity: {alert.severity.value.upper()}
- Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Source: {alert.source}

Message:
{alert.message}

Metrics:
{json.dumps(alert.metrics, indent=2) if alert.metrics else 'No metrics available'}

🎯 This alert affects our humanitarian mission to fund medical aid, children's surgeries, and poverty relief.
Please take immediate action to ensure optimal platform performance.

Best regards,
Humanitarian AI Platform Monitoring System
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info("✅ Email alert sent")
            
        except Exception as e:
            logger.error(f"❌ Email alert failed: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            # Slack message payload
            color = {"critical": "danger", "warning": "warning", "info": "good"}[alert.severity.value]
            
            payload = {
                "channel": self.config.slack_channel,
                "username": "Humanitarian AI Monitor",
                "icon_emoji": ":hospital:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"🏥 {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Type", "value": alert.type.value, "short": True},
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                        ],
                        "footer": "💝 Humanitarian AI Platform - Serving medical aid mission"
                    }
                ]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.slack_webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("✅ Slack alert sent")
                    else:
                        logger.error(f"❌ Slack alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ Slack alert failed: {e}")
    
    async def _send_telegram_alert(self, alert: Alert):
        """Send Telegram alert"""
        try:
            message = f"""
🏥 *HUMANITARIAN AI ALERT*
💝 Charitable Trading Mission Alert

*{alert.title}*

Type: {alert.type.value}
Severity: {alert.severity.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Source: {alert.source}

{alert.message}

🎯 This affects our mission to fund medical aid and children's surgeries.
            """
            
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.config.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("✅ Telegram alert sent")
                    else:
                        logger.error(f"❌ Telegram alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ Telegram alert failed: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        # Implementation for custom webhook alerts
        pass

class ProductionMonitoringSystem:
    """
    🏥 Production Monitoring & Alerting System for Humanitarian AI Platform
    
    Provides comprehensive monitoring for charitable trading mission:
    - Real-time system metrics monitoring
    - Trading performance tracking
    - Humanitarian impact monitoring
    - Intelligent alerting with multiple channels
    - Performance trend analysis
    - Automated alert escalation
    - Dashboard metrics for Grafana/Prometheus
    """
    
    def __init__(self, thresholds: MetricThresholds = None, notification_config: NotificationConfig = None):
        self.thresholds = thresholds or MetricThresholds()
        self.notification_config = notification_config or NotificationConfig()
        
        # Components
        self.system_metrics = SystemMetrics()
        self.trading_metrics = TradingMetrics()
        self.notification_manager = NotificationManager(self.notification_config)
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_counter = 0
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.monitoring_task = None
        
        # Health check status
        self.last_health_check = None
        self.health_status = "unknown"
        
        logger.info("🏥 Production Monitoring System initialized")
        logger.info("💝 Monitoring humanitarian AI platform for charitable mission")
        logger.info(f"🎯 Alert thresholds configured for optimal performance")
    
    async def start_monitoring(self):
        """Start monitoring services"""
        if self.monitoring_active:
            logger.warning("⚠️ Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("🚀 Production monitoring started")
        logger.info("💝 Monitoring for humanitarian mission success")
    
    async def stop_monitoring(self):
        """Stop monitoring services"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Production monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                system_metrics = self.system_metrics.collect_system_metrics()
                trading_metrics = self.trading_metrics.get_trading_metrics()
                
                # Check for alerts
                await self._check_system_alerts(system_metrics)
                await self._check_trading_alerts(trading_metrics)
                await self._check_humanitarian_alerts(trading_metrics)
                
                # Update health status
                self._update_health_status(system_metrics, trading_metrics)
                
                # Wait for next iteration
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_system_alerts(self, metrics: Dict[str, float]):
        """Check for system-related alerts"""
        # CPU usage alerts
        cpu_usage = metrics.get("cpu_usage_percent", 0)
        if cpu_usage >= self.thresholds.cpu_usage_critical:
            await self._create_alert(
                AlertType.SYSTEM,
                AlertSeverity.CRITICAL,
                "High CPU Usage",
                f"CPU usage at {cpu_usage:.1f}% (threshold: {self.thresholds.cpu_usage_critical}%)",
                "system_monitor",
                {"cpu_usage": cpu_usage}
            )
        elif cpu_usage >= self.thresholds.cpu_usage_warning:
            await self._create_alert(
                AlertType.SYSTEM,
                AlertSeverity.WARNING,
                "Elevated CPU Usage",
                f"CPU usage at {cpu_usage:.1f}% (threshold: {self.thresholds.cpu_usage_warning}%)",
                "system_monitor",
                {"cpu_usage": cpu_usage}
            )
        
        # Memory usage alerts
        memory_usage = metrics.get("memory_usage_percent", 0)
        if memory_usage >= self.thresholds.memory_usage_critical:
            await self._create_alert(
                AlertType.SYSTEM,
                AlertSeverity.CRITICAL,
                "High Memory Usage",
                f"Memory usage at {memory_usage:.1f}% (threshold: {self.thresholds.memory_usage_critical}%)",
                "system_monitor",
                {"memory_usage": memory_usage}
            )
        elif memory_usage >= self.thresholds.memory_usage_warning:
            await self._create_alert(
                AlertType.SYSTEM,
                AlertSeverity.WARNING,
                "Elevated Memory Usage",
                f"Memory usage at {memory_usage:.1f}% (threshold: {self.thresholds.memory_usage_warning}%)",
                "system_monitor",
                {"memory_usage": memory_usage}
            )
    
    async def _check_trading_alerts(self, metrics: Dict[str, Any]):
        """Check for trading-related alerts"""
        # Error rate alerts
        error_rate = metrics.get("error_rate", 0)
        if error_rate >= self.thresholds.error_rate_critical:
            await self._create_alert(
                AlertType.TRADING,
                AlertSeverity.CRITICAL,
                "High Trading Error Rate",
                f"Trading error rate at {error_rate:.1f}% (threshold: {self.thresholds.error_rate_critical}%)",
                "trading_monitor",
                {"error_rate": error_rate, "total_trades": metrics.get("total_trades", 0)}
            )
        elif error_rate >= self.thresholds.error_rate_warning:
            await self._create_alert(
                AlertType.TRADING,
                AlertSeverity.WARNING,
                "Elevated Trading Error Rate",
                f"Trading error rate at {error_rate:.1f}% (threshold: {self.thresholds.error_rate_warning}%)",
                "trading_monitor",
                {"error_rate": error_rate, "total_trades": metrics.get("total_trades", 0)}
            )
    
    async def _check_humanitarian_alerts(self, metrics: Dict[str, Any]):
        """Check for humanitarian mission alerts"""
        humanitarian_impact = metrics.get("humanitarian_impact", {})
        charitable_amount = humanitarian_impact.get("charitable_amount", 0)
        
        # Estimate monthly target progress (assuming 30-day month)
        daily_target = 50000 / 30  # $50K monthly target / 30 days
        current_daily = charitable_amount  # Current day's charitable amount
        
        if current_daily < daily_target * self.thresholds.humanitarian_target_critical:
            await self._create_alert(
                AlertType.HUMANITARIAN,
                AlertSeverity.CRITICAL,
                "Humanitarian Target at Risk",
                f"Daily charitable amount ${current_daily:.2f} is {(current_daily/daily_target)*100:.1f}% of target ${daily_target:.2f}",
                "humanitarian_monitor",
                {"daily_charitable": current_daily, "daily_target": daily_target, "humanitarian_impact": humanitarian_impact}
            )
        elif current_daily < daily_target * self.thresholds.humanitarian_target_warning:
            await self._create_alert(
                AlertType.HUMANITARIAN,
                AlertSeverity.WARNING,
                "Humanitarian Target Below Expected",
                f"Daily charitable amount ${current_daily:.2f} is {(current_daily/daily_target)*100:.1f}% of target ${daily_target:.2f}",
                "humanitarian_monitor",
                {"daily_charitable": current_daily, "daily_target": daily_target, "humanitarian_impact": humanitarian_impact}
            )
    
    async def _create_alert(self, alert_type: AlertType, severity: AlertSeverity, title: str, message: str, source: str, metrics: Dict[str, Any] = None):
        """Create and send alert"""
        # Generate unique alert ID
        self.alert_counter += 1
        alert_id = f"{alert_type.value}_{severity.value}_{self.alert_counter}_{int(time.time())}"
        
        # Check if similar alert already exists
        existing_alert_key = f"{alert_type.value}_{title}"
        if existing_alert_key in self.active_alerts:
            # Update existing alert instead of creating new one
            existing_alert = self.active_alerts[existing_alert_key]
            existing_alert.message = message
            existing_alert.timestamp = datetime.now()
            existing_alert.metrics = metrics
            return
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metrics=metrics
        )
        
        # Store alert
        self.active_alerts[existing_alert_key] = alert
        self.alert_history.append(alert)
        
        # Send notification
        await self.notification_manager.send_alert(alert)
        
        logger.warning(f"🚨 Alert created: {title} ({severity.value})")
    
    def _update_health_status(self, system_metrics: Dict[str, float], trading_metrics: Dict[str, Any]):
        """Update overall health status"""
        self.last_health_check = datetime.now()
        
        # Check critical thresholds
        cpu_ok = system_metrics.get("cpu_usage_percent", 0) < self.thresholds.cpu_usage_critical
        memory_ok = system_metrics.get("memory_usage_percent", 0) < self.thresholds.memory_usage_critical
        trading_ok = trading_metrics.get("error_rate", 0) < self.thresholds.error_rate_critical
        
        if cpu_ok and memory_ok and trading_ok:
            self.health_status = "healthy"
        elif trading_metrics.get("total_trades", 0) == 0:
            self.health_status = "initializing"
        else:
            self.health_status = "degraded"
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        # Get latest metrics
        system_metrics = self.system_metrics.collect_system_metrics()
        trading_metrics = self.trading_metrics.get_trading_metrics()
        
        # Get trends
        cpu_trend = self.system_metrics.get_metric_trend("cpu_usage_percent")
        memory_trend = self.system_metrics.get_metric_trend("memory_usage_percent")
        
        # Count active alerts by severity
        alert_counts = {"critical": 0, "warning": 0, "info": 0}
        for alert in self.active_alerts.values():
            if not alert.resolved:
                alert_counts[alert.severity.value] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health_status": self.health_status,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "monitoring_active": self.monitoring_active,
            "system_metrics": system_metrics,
            "trading_metrics": trading_metrics,
            "trends": {
                "cpu_usage": cpu_trend,
                "memory_usage": memory_trend
            },
            "alerts": {
                "active_count": len([a for a in self.active_alerts.values() if not a.resolved]),
                "total_count": len(self.alert_history),
                "by_severity": alert_counts,
                "recent_alerts": [asdict(alert) for alert in list(self.alert_history)[-5:]]
            },
            "humanitarian_mission": {
                "status": "active",
                "charitable_target_monthly": 50000,
                "current_impact": trading_metrics.get("humanitarian_impact", {}),
                "mission_health": "good" if self.health_status == "healthy" else "at_risk"
            }
        }
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        for key, alert in self.active_alerts.items():
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                del self.active_alerts[key]
                logger.info(f"✅ Alert resolved: {alert.title}")
                return True
        return False
    
    def record_trade_execution(self, trade_data: Dict[str, Any]):
        """Record a trade execution for monitoring"""
        self.trading_metrics.record_trade(trade_data)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            system_metrics = self.system_metrics.collect_system_metrics()
            trading_metrics = self.trading_metrics.get_trading_metrics()
            
            self._update_health_status(system_metrics, trading_metrics)
            
            return {
                "status": self.health_status,
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_active,
                "last_check": self.last_health_check.isoformat() if self.last_health_check else None,
                "system_ok": system_metrics.get("cpu_usage_percent", 0) < self.thresholds.cpu_usage_critical,
                "memory_ok": system_metrics.get("memory_usage_percent", 0) < self.thresholds.memory_usage_critical,
                "trading_ok": trading_metrics.get("error_rate", 0) < self.thresholds.error_rate_critical,
                "humanitarian_mission": "serving the poorest of the poor"
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

# Global monitoring system instance
monitoring_system = None

def get_monitoring_system() -> ProductionMonitoringSystem:
    """Get or create global monitoring system"""
    global monitoring_system
    
    if monitoring_system is None:
        monitoring_system = ProductionMonitoringSystem()
    
    return monitoring_system

# Example usage and testing
if __name__ == "__main__":
    async def test_monitoring_system():
        print("🏥 Testing Production Monitoring System")
        print("💝 Monitoring humanitarian AI platform")
        
        # Configure notifications (for testing, disable actual sending)
        notification_config = NotificationConfig(
            email_enabled=False,  # Disable for testing
            slack_enabled=False,
            telegram_enabled=False
        )
        
        # Initialize monitoring system
        monitoring = ProductionMonitoringSystem(notification_config=notification_config)
        
        # Start monitoring
        await monitoring.start_monitoring()
        
        # Simulate some metrics
        print("\n📊 Simulating platform activity...")
        
        # Simulate trades
        for i in range(10):
            trade_data = {
                "trade_id": f"trade_{i}",
                "profit": np.random.uniform(10, 100),
                "execution_time_ms": np.random.uniform(1, 5),
                "error": False
            }
            monitoring.record_trade_execution(trade_data)
        
        # Wait for monitoring cycle
        await asyncio.sleep(2)
        
        # Get dashboard data
        dashboard = monitoring.get_monitoring_dashboard()
        print(f"\n📋 Monitoring Dashboard:")
        print(f"   Health Status: {dashboard['health_status']}")
        print(f"   Active Alerts: {dashboard['alerts']['active_count']}")
        print(f"   Total Trades: {dashboard['trading_metrics']['total_trades']}")
        print(f"   Charitable Amount: ${dashboard['trading_metrics']['humanitarian_impact']['charitable_amount']:.2f}")
        print(f"   Medical Aids: {dashboard['trading_metrics']['humanitarian_impact']['medical_aids']}")
        print(f"   Families Fed: {dashboard['trading_metrics']['humanitarian_impact']['families_fed']}")
        
        # Test health check
        health = await monitoring.health_check()
        print(f"\n🏥 Health Check:")
        print(f"   Status: {health['status']}")
        print(f"   System OK: {health['system_ok']}")
        print(f"   Memory OK: {health['memory_ok']}")
        print(f"   Trading OK: {health['trading_ok']}")
        
        # Stop monitoring
        await monitoring.stop_monitoring()
        
        print("\n🎯 Monitoring System testing completed!")
        print("💝 Platform monitored for maximum charitable impact")
    
    # Run test
    asyncio.run(test_monitoring_system())
