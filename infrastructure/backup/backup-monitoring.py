#!/usr/bin/env python3
"""
Platform3 Forex Trading Platform - Backup Monitoring System
Real-time monitoring and alerting for backup operations and data integrity

This module provides comprehensive backup monitoring including:
- Backup job status monitoring
- Data integrity verification
- Recovery time estimation
- Automated alerting and notifications
- Performance metrics tracking
"""

import asyncio
import logging
import json
import os
import subprocess
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
import asyncpg
import aioredis
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import hashlib
import schedule
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/platform3/backups/logs/backup-monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupStatus(Enum):
    """Backup operation status"""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SCHEDULED = "SCHEDULED"
    CANCELLED = "CANCELLED"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

@dataclass
class BackupJob:
    """Backup job information"""
    job_id: str
    job_type: str
    component: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: BackupStatus = BackupStatus.SCHEDULED
    backup_size_mb: float = 0.0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    backup_path: Optional[str] = None

@dataclass
class BackupMetrics:
    """Backup performance metrics"""
    timestamp: datetime
    total_backup_size_gb: float
    backup_duration_minutes: float
    compression_ratio: float
    transfer_rate_mbps: float
    success_rate_24h: float
    last_successful_backup: datetime
    next_scheduled_backup: datetime
    storage_usage_percent: float

@dataclass
class DataIntegrityCheck:
    """Data integrity verification result"""
    component: str
    check_type: str
    timestamp: datetime
    passed: bool
    checksum_expected: str
    checksum_actual: str
    error_details: Optional[str] = None

class BackupMonitor:
    """
    Comprehensive backup monitoring system for Platform3 Forex Trading Platform
    """
    
    def __init__(self, config_path: str = "/opt/platform3/backups/config/monitor.json"):
        """Initialize the backup monitor"""
        self.config = self._load_config(config_path)
        self.backup_jobs: Dict[str, BackupJob] = {}
        self.metrics_history: List[BackupMetrics] = []
        self.integrity_checks: List[DataIntegrityCheck] = []
        self.running = False
        
        # Database connections
        self.postgres_client = None
        self.redis_client = None
        
        # Monitoring intervals
        self.check_interval = self.config.get('check_interval_seconds', 60)
        self.metrics_interval = self.config.get('metrics_interval_seconds', 300)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration"""
        default_config = {
            "backup_base_dir": "/opt/platform3/backups",
            "retention_days": 30,
            "check_interval_seconds": 60,
            "metrics_interval_seconds": 300,
            "alert_thresholds": {
                "backup_failure_count": 3,
                "storage_usage_percent": 85,
                "backup_duration_hours": 2,
                "data_integrity_failures": 1
            },
            "notifications": {
                "email": {
                    "enabled": True,
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "recipients": ["admin@platform3.com"]
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": ""
                }
            },
            "database": {
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "forex_trading",
                    "user": "forex_admin",
                    "password": "ForexSecure2025!"
                },
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "password": "RedisSecure2025!"
                }
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    return {**default_config, **config}
            else:
                # Create default config
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default configuration at {config_path}")
                return default_config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return default_config
    
    async def initialize_connections(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection
            db_config = self.config['database']['postgres']
            self.postgres_client = await asyncpg.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            )
            
            # Redis connection
            redis_config = self.config['database']['redis']
            self.redis_client = await aioredis.from_url(
                f"redis://:{redis_config['password']}@{redis_config['host']}:{redis_config['port']}",
                decode_responses=True
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def monitor_backup_jobs(self):
        """Monitor running and scheduled backup jobs"""
        try:
            backup_dir = Path(self.config['backup_base_dir'])
            
            # Check for running backup processes
            running_backups = self._get_running_backup_processes()
            
            for process in running_backups:
                job_id = f"backup_{process.pid}"
                if job_id not in self.backup_jobs:
                    self.backup_jobs[job_id] = BackupJob(
                        job_id=job_id,
                        job_type="full",
                        component="platform3",
                        start_time=datetime.fromtimestamp(process.create_time()),
                        status=BackupStatus.RUNNING
                    )
                
                # Update job status
                job = self.backup_jobs[job_id]
                if process.is_running():
                    job.status = BackupStatus.RUNNING
                    job.duration_seconds = time.time() - process.create_time()
                else:
                    job.status = BackupStatus.COMPLETED if process.returncode == 0 else BackupStatus.FAILED
                    job.end_time = datetime.now()
            
            # Check for completed backups
            await self._check_completed_backups()
            
            # Verify backup integrity
            await self._verify_backup_integrity()
            
        except Exception as e:
            logger.error(f"Error monitoring backup jobs: {e}")
    
    def _get_running_backup_processes(self) -> List[psutil.Process]:
        """Get list of running backup processes"""
        backup_processes = []
        
        for process in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = ' '.join(process.info['cmdline'] or [])
                if 'backup-strategy.sh' in cmdline or 'pg_dump' in cmdline or 'redis-cli' in cmdline:
                    backup_processes.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return backup_processes
    
    async def _check_completed_backups(self):
        """Check for newly completed backups"""
        backup_dir = Path(self.config['backup_base_dir'])
        
        # Look for backup directories created in the last hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        for backup_path in backup_dir.glob('20*'):
            if not backup_path.is_dir():
                continue
                
            try:
                backup_time = datetime.strptime(backup_path.name, '%Y%m%d-%H%M%S')
                if backup_time < cutoff_time:
                    continue
                
                manifest_file = backup_path / 'backup_manifest.json'
                if manifest_file.exists():
                    await self._process_backup_manifest(manifest_file)
                    
            except ValueError:
                # Not a backup directory
                continue
            except Exception as e:
                logger.error(f"Error processing backup {backup_path}: {e}")
    
    async def _process_backup_manifest(self, manifest_file: Path):
        """Process backup manifest and update job status"""
        try:
            async with aiofiles.open(manifest_file, 'r') as f:
                manifest = json.loads(await f.read())
            
            job_id = f"backup_{manifest['backup_timestamp']}"
            
            if job_id not in self.backup_jobs:
                self.backup_jobs[job_id] = BackupJob(
                    job_id=job_id,
                    job_type=manifest.get('backup_type', 'full'),
                    component='platform3',
                    start_time=datetime.fromisoformat(manifest['created_at']),
                    status=BackupStatus.COMPLETED,
                    backup_size_mb=manifest.get('backup_size_mb', 0),
                    backup_path=str(manifest_file.parent)
                )
            
            # Update job with manifest data
            job = self.backup_jobs[job_id]
            job.backup_size_mb = manifest.get('backup_size_mb', 0)
            job.backup_path = str(manifest_file.parent)
            
            logger.info(f"Processed backup manifest: {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing manifest {manifest_file}: {e}")
    
    async def _verify_backup_integrity(self):
        """Verify backup data integrity"""
        backup_dir = Path(self.config['backup_base_dir'])
        
        # Check recent backups (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for backup_path in backup_dir.glob('20*'):
            if not backup_path.is_dir():
                continue
                
            try:
                backup_time = datetime.strptime(backup_path.name, '%Y%m%d-%H%M%S')
                if backup_time < cutoff_time:
                    continue
                
                # Verify PostgreSQL backup
                await self._verify_postgresql_backup(backup_path)
                
                # Verify Redis backup
                await self._verify_redis_backup(backup_path)
                
                # Verify InfluxDB backup
                await self._verify_influxdb_backup(backup_path)
                
            except Exception as e:
                logger.error(f"Error verifying backup {backup_path}: {e}")
    
    async def _verify_postgresql_backup(self, backup_path: Path):
        """Verify PostgreSQL backup integrity"""
        pg_dir = backup_path / 'postgresql'
        if not pg_dir.exists():
            return
        
        for backup_file in pg_dir.glob('*.custom'):
            try:
                # Test backup file integrity using pg_restore --list
                result = subprocess.run([
                    'pg_restore', '--list', str(backup_file)
                ], capture_output=True, text=True, timeout=60)
                
                check = DataIntegrityCheck(
                    component='postgresql',
                    check_type='backup_integrity',
                    timestamp=datetime.now(),
                    passed=result.returncode == 0,
                    checksum_expected='',
                    checksum_actual='',
                    error_details=result.stderr if result.returncode != 0 else None
                )
                
                self.integrity_checks.append(check)
                
                if not check.passed:
                    await self._send_alert(
                        AlertSeverity.CRITICAL,
                        f"PostgreSQL backup integrity check failed: {backup_file}",
                        check.error_details
                    )
                
            except Exception as e:
                logger.error(f"Error verifying PostgreSQL backup {backup_file}: {e}")
    
    async def _verify_redis_backup(self, backup_path: Path):
        """Verify Redis backup integrity"""
        redis_dir = backup_path / 'redis'
        if not redis_dir.exists():
            return
        
        for backup_file in redis_dir.glob('*.rdb'):
            try:
                # Check RDB file integrity
                result = subprocess.run([
                    'redis-check-rdb', str(backup_file)
                ], capture_output=True, text=True, timeout=30)
                
                check = DataIntegrityCheck(
                    component='redis',
                    check_type='rdb_integrity',
                    timestamp=datetime.now(),
                    passed=result.returncode == 0,
                    checksum_expected='',
                    checksum_actual='',
                    error_details=result.stderr if result.returncode != 0 else None
                )
                
                self.integrity_checks.append(check)
                
                if not check.passed:
                    await self._send_alert(
                        AlertSeverity.CRITICAL,
                        f"Redis backup integrity check failed: {backup_file}",
                        check.error_details
                    )
                
            except Exception as e:
                logger.error(f"Error verifying Redis backup {backup_file}: {e}")
    
    async def _verify_influxdb_backup(self, backup_path: Path):
        """Verify InfluxDB backup integrity"""
        influx_dir = backup_path / 'influxdb'
        if not influx_dir.exists():
            return
        
        # Check for backup directory structure
        backup_dirs = list(influx_dir.glob('influx_backup_*'))
        
        for backup_dir in backup_dirs:
            try:
                # Check if backup directory contains expected files
                manifest_files = list(backup_dir.glob('*.manifest'))
                
                check = DataIntegrityCheck(
                    component='influxdb',
                    check_type='backup_structure',
                    timestamp=datetime.now(),
                    passed=len(manifest_files) > 0,
                    checksum_expected='',
                    checksum_actual='',
                    error_details=f"No manifest files found in {backup_dir}" if len(manifest_files) == 0 else None
                )
                
                self.integrity_checks.append(check)
                
                if not check.passed:
                    await self._send_alert(
                        AlertSeverity.WARNING,
                        f"InfluxDB backup structure check failed: {backup_dir}",
                        check.error_details
                    )
                
            except Exception as e:
                logger.error(f"Error verifying InfluxDB backup {backup_dir}: {e}")
    
    async def calculate_metrics(self) -> BackupMetrics:
        """Calculate backup performance metrics"""
        try:
            backup_dir = Path(self.config['backup_base_dir'])
            
            # Calculate total backup size
            total_size_bytes = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
            total_size_gb = total_size_bytes / (1024**3)
            
            # Calculate storage usage
            disk_usage = psutil.disk_usage(str(backup_dir))
            storage_usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Calculate success rate (last 24 hours)
            recent_jobs = [job for job in self.backup_jobs.values() 
                          if job.start_time > datetime.now() - timedelta(hours=24)]
            
            success_rate = 0.0
            if recent_jobs:
                successful_jobs = [job for job in recent_jobs if job.status == BackupStatus.COMPLETED]
                success_rate = len(successful_jobs) / len(recent_jobs) * 100
            
            # Find last successful backup
            successful_jobs = [job for job in self.backup_jobs.values() 
                             if job.status == BackupStatus.COMPLETED]
            last_successful = max(successful_jobs, key=lambda x: x.start_time).start_time if successful_jobs else datetime.min
            
            # Calculate average backup duration
            completed_jobs = [job for job in recent_jobs if job.status == BackupStatus.COMPLETED]
            avg_duration = sum(job.duration_seconds for job in completed_jobs) / len(completed_jobs) if completed_jobs else 0
            
            metrics = BackupMetrics(
                timestamp=datetime.now(),
                total_backup_size_gb=total_size_gb,
                backup_duration_minutes=avg_duration / 60,
                compression_ratio=0.0,  # TODO: Calculate compression ratio
                transfer_rate_mbps=0.0,  # TODO: Calculate transfer rate
                success_rate_24h=success_rate,
                last_successful_backup=last_successful,
                next_scheduled_backup=datetime.now() + timedelta(hours=24),  # TODO: Get from scheduler
                storage_usage_percent=storage_usage_percent
            )
            
            self.metrics_history.append(metrics)
            
            # Keep only last 7 days of metrics
            cutoff_time = datetime.now() - timedelta(days=7)
            self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return BackupMetrics(
                timestamp=datetime.now(),
                total_backup_size_gb=0.0,
                backup_duration_minutes=0.0,
                compression_ratio=0.0,
                transfer_rate_mbps=0.0,
                success_rate_24h=0.0,
                last_successful_backup=datetime.min,
                next_scheduled_backup=datetime.now(),
                storage_usage_percent=0.0
            )
    
    async def check_alert_conditions(self, metrics: BackupMetrics):
        """Check for alert conditions and send notifications"""
        thresholds = self.config['alert_thresholds']
        
        # Check storage usage
        if metrics.storage_usage_percent > thresholds['storage_usage_percent']:
            await self._send_alert(
                AlertSeverity.WARNING,
                f"Backup storage usage high: {metrics.storage_usage_percent:.1f}%",
                f"Storage usage exceeds threshold of {thresholds['storage_usage_percent']}%"
            )
        
        # Check backup duration
        if metrics.backup_duration_minutes > thresholds['backup_duration_hours'] * 60:
            await self._send_alert(
                AlertSeverity.WARNING,
                f"Backup duration excessive: {metrics.backup_duration_minutes:.1f} minutes",
                f"Duration exceeds threshold of {thresholds['backup_duration_hours']} hours"
            )
        
        # Check success rate
        if metrics.success_rate_24h < 90:
            await self._send_alert(
                AlertSeverity.CRITICAL,
                f"Backup success rate low: {metrics.success_rate_24h:.1f}%",
                "Multiple backup failures detected in the last 24 hours"
            )
        
        # Check last successful backup age
        hours_since_last_backup = (datetime.now() - metrics.last_successful_backup).total_seconds() / 3600
        if hours_since_last_backup > 25:  # More than 25 hours
            await self._send_alert(
                AlertSeverity.CRITICAL,
                f"No successful backup in {hours_since_last_backup:.1f} hours",
                f"Last successful backup: {metrics.last_successful_backup}"
            )
    
    async def _send_alert(self, severity: AlertSeverity, message: str, details: str = ""):
        """Send alert notification"""
        try:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'severity': severity.value,
                'message': message,
                'details': details,
                'component': 'backup-monitor'
            }
            
            # Log alert
            logger.warning(f"ALERT [{severity.value}]: {message}")
            
            # Send email notification
            if self.config['notifications']['email']['enabled']:
                await self._send_email_alert(alert_data)
            
            # Send Slack notification
            if self.config['notifications']['slack']['enabled']:
                await self._send_slack_alert(alert_data)
            
            # Store alert in database
            if self.postgres_client:
                await self.postgres_client.execute("""
                    INSERT INTO backup_alerts (timestamp, severity, message, details, component)
                    VALUES ($1, $2, $3, $4, $5)
                """, datetime.now(), severity.value, message, details, 'backup-monitor')
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    async def _send_email_alert(self, alert_data: Dict):
        """Send email alert notification"""
        try:
            email_config = self.config['notifications']['email']
            
            msg = MimeMultipart()
            msg['From'] = 'platform3-backup@localhost'
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"Platform3 Backup Alert - {alert_data['severity']}"
            
            body = f"""
            Platform3 Backup Monitoring Alert
            
            Severity: {alert_data['severity']}
            Timestamp: {alert_data['timestamp']}
            Message: {alert_data['message']}
            
            Details:
            {alert_data['details']}
            
            Component: {alert_data['component']}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert_data: Dict):
        """Send Slack alert notification"""
        try:
            slack_config = self.config['notifications']['slack']
            webhook_url = slack_config['webhook_url']
            
            if not webhook_url:
                return
            
            payload = {
                'text': f"Platform3 Backup Alert - {alert_data['severity']}",
                'attachments': [{
                    'color': 'danger' if alert_data['severity'] in ['CRITICAL', 'EMERGENCY'] else 'warning',
                    'fields': [
                        {'title': 'Message', 'value': alert_data['message'], 'short': False},
                        {'title': 'Details', 'value': alert_data['details'], 'short': False},
                        {'title': 'Timestamp', 'value': alert_data['timestamp'], 'short': True},
                        {'title': 'Component', 'value': alert_data['component'], 'short': True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def start_monitoring(self):
        """Start the backup monitoring process"""
        self.running = True
        logger.info("Backup monitoring started")
        
        try:
            await self.initialize_connections()
            
            while self.running:
                # Monitor backup jobs
                await self.monitor_backup_jobs()
                
                # Calculate metrics every 5 minutes
                if int(time.time()) % self.metrics_interval == 0:
                    metrics = await self.calculate_metrics()
                    await self.check_alert_conditions(metrics)
                
                await asyncio.sleep(self.check_interval)
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            await self.cleanup()
    
    async def stop_monitoring(self):
        """Stop the backup monitoring process"""
        self.running = False
        logger.info("Backup monitoring stopped")
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.postgres_client:
                await self.postgres_client.close()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main function to run the backup monitor"""
    monitor = BackupMonitor()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
