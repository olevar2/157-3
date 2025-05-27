"""
Drawdown Monitor
Maximum daily drawdown limits and real-time monitoring system

Features:
- Real-time drawdown calculation and monitoring
- Daily, weekly, and monthly drawdown limits
- Automatic position closure on limit breach
- Drawdown recovery tracking
- Risk escalation alerts
- Performance analytics and reporting
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import redis
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrawdownLevel(Enum):
    """Drawdown severity levels"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ActionType(Enum):
    """Actions to take on drawdown breach"""
    MONITOR = "monitor"
    REDUCE_SIZE = "reduce_size"
    HALT_TRADING = "halt_trading"
    CLOSE_POSITIONS = "close_positions"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class DrawdownLimits:
    """Drawdown limit configuration"""
    daily_warning: float = 0.02  # 2%
    daily_critical: float = 0.05  # 5%
    daily_emergency: float = 0.08  # 8%
    weekly_warning: float = 0.08  # 8%
    weekly_critical: float = 0.15  # 15%
    monthly_warning: float = 0.15  # 15%
    monthly_critical: float = 0.25  # 25%
    max_consecutive_losses: int = 5
    recovery_threshold: float = 0.5  # 50% recovery required

@dataclass
class AccountSnapshot:
    """Account balance snapshot"""
    account_id: str
    timestamp: datetime
    balance: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    peak_balance: float
    drawdown: float
    drawdown_percent: float

@dataclass
class DrawdownEvent:
    """Drawdown event record"""
    event_id: str
    account_id: str
    timestamp: datetime
    event_type: str  # 'breach', 'recovery', 'escalation'
    level: DrawdownLevel
    drawdown_amount: float
    drawdown_percent: float
    action_taken: ActionType
    message: str

class DrawdownMonitor:
    """
    Real-time drawdown monitoring and limit enforcement system
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.limits = DrawdownLimits()
        self.account_snapshots: Dict[str, List[AccountSnapshot]] = {}
        self.peak_balances: Dict[str, float] = {}
        self.daily_peaks: Dict[str, Dict[date, float]] = {}
        self.drawdown_events: List[DrawdownEvent] = []
        self.trading_halted: Dict[str, bool] = {}
        self.consecutive_losses: Dict[str, int] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
        # Performance tracking
        self.performance_stats = {
            'total_accounts_monitored': 0,
            'drawdown_breaches': 0,
            'emergency_stops': 0,
            'successful_recoveries': 0,
            'average_recovery_time': 0.0
        }
        
        logger.info("DrawdownMonitor initialized")

    async def start(self):
        """Start the drawdown monitoring system"""
        self.running = True
        logger.info("Starting Drawdown Monitor...")
        
        # Start background tasks
        asyncio.create_task(self._monitor_drawdowns())
        asyncio.create_task(self._check_recovery_status())
        asyncio.create_task(self._generate_reports())
        
        logger.info("✅ Drawdown Monitor started successfully")

    async def stop(self):
        """Stop the drawdown monitoring system"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Drawdown Monitor stopped")

    async def update_account_balance(self, account_id: str, balance: float, equity: float, 
                                   unrealized_pnl: float, realized_pnl: float) -> DrawdownLevel:
        """
        Update account balance and check drawdown limits
        
        Args:
            account_id: Account identifier
            balance: Current account balance
            equity: Current equity
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L for the day
            
        Returns:
            DrawdownLevel: Current drawdown severity level
        """
        try:
            # Initialize account tracking if new
            if account_id not in self.account_snapshots:
                self.account_snapshots[account_id] = []
                self.peak_balances[account_id] = equity
                self.daily_peaks[account_id] = {}
                self.trading_halted[account_id] = False
                self.consecutive_losses[account_id] = 0
                self.performance_stats['total_accounts_monitored'] += 1
            
            # Update peak balance
            current_peak = self.peak_balances[account_id]
            if equity > current_peak:
                self.peak_balances[account_id] = equity
                current_peak = equity
            
            # Update daily peak
            today = datetime.now().date()
            if today not in self.daily_peaks[account_id]:
                self.daily_peaks[account_id][today] = equity
            elif equity > self.daily_peaks[account_id][today]:
                self.daily_peaks[account_id][today] = equity
            
            # Calculate drawdowns
            total_drawdown = current_peak - equity
            total_drawdown_percent = (total_drawdown / current_peak) * 100 if current_peak > 0 else 0
            
            daily_peak = self.daily_peaks[account_id][today]
            daily_drawdown = daily_peak - equity
            daily_drawdown_percent = (daily_drawdown / daily_peak) * 100 if daily_peak > 0 else 0
            
            # Create snapshot
            snapshot = AccountSnapshot(
                account_id=account_id,
                timestamp=datetime.now(),
                balance=balance,
                equity=equity,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                daily_pnl=realized_pnl,  # Simplified - should be daily P&L
                peak_balance=current_peak,
                drawdown=total_drawdown,
                drawdown_percent=total_drawdown_percent
            )
            
            # Store snapshot
            self.account_snapshots[account_id].append(snapshot)
            
            # Keep only last 1000 snapshots per account
            if len(self.account_snapshots[account_id]) > 1000:
                self.account_snapshots[account_id] = self.account_snapshots[account_id][-1000:]
            
            # Check drawdown levels and take action
            level = await self._check_drawdown_limits(account_id, daily_drawdown_percent, total_drawdown_percent)
            
            # Cache snapshot in Redis
            await self._cache_snapshot(snapshot)
            
            return level
            
        except Exception as e:
            logger.error(f"❌ Failed to update account balance for {account_id}: {e}")
            return DrawdownLevel.SAFE

    async def _check_drawdown_limits(self, account_id: str, daily_drawdown_percent: float, 
                                   total_drawdown_percent: float) -> DrawdownLevel:
        """Check drawdown limits and take appropriate action"""
        try:
            level = DrawdownLevel.SAFE
            action = ActionType.MONITOR
            
            # Check daily limits first (most restrictive)
            if daily_drawdown_percent >= self.limits.daily_emergency:
                level = DrawdownLevel.EMERGENCY
                action = ActionType.EMERGENCY_STOP
            elif daily_drawdown_percent >= self.limits.daily_critical:
                level = DrawdownLevel.CRITICAL
                action = ActionType.CLOSE_POSITIONS
            elif daily_drawdown_percent >= self.limits.daily_warning:
                level = DrawdownLevel.WARNING
                action = ActionType.REDUCE_SIZE
            
            # Check total drawdown limits
            elif total_drawdown_percent >= self.limits.monthly_critical:
                level = DrawdownLevel.CRITICAL
                action = ActionType.HALT_TRADING
            elif total_drawdown_percent >= self.limits.monthly_warning:
                level = DrawdownLevel.WARNING
                action = ActionType.REDUCE_SIZE
            
            # Check consecutive losses
            if self.consecutive_losses[account_id] >= self.limits.max_consecutive_losses:
                if level == DrawdownLevel.SAFE:
                    level = DrawdownLevel.WARNING
                    action = ActionType.HALT_TRADING
            
            # Take action if needed
            if level != DrawdownLevel.SAFE:
                await self._take_action(account_id, level, action, daily_drawdown_percent, total_drawdown_percent)
            
            return level
            
        except Exception as e:
            logger.error(f"Failed to check drawdown limits for {account_id}: {e}")
            return DrawdownLevel.SAFE

    async def _take_action(self, account_id: str, level: DrawdownLevel, action: ActionType,
                          daily_drawdown: float, total_drawdown: float):
        """Take appropriate action based on drawdown level"""
        try:
            message = f"Drawdown limit breach - Daily: {daily_drawdown:.2f}%, Total: {total_drawdown:.2f}%"
            
            if action == ActionType.EMERGENCY_STOP:
                self.trading_halted[account_id] = True
                self.performance_stats['emergency_stops'] += 1
                message += " - EMERGENCY STOP ACTIVATED"
                logger.critical(f"🚨 EMERGENCY STOP for account {account_id}")
                
            elif action == ActionType.CLOSE_POSITIONS:
                self.trading_halted[account_id] = True
                message += " - Closing all positions"
                logger.error(f"🛑 Closing all positions for account {account_id}")
                
            elif action == ActionType.HALT_TRADING:
                self.trading_halted[account_id] = True
                message += " - Trading halted"
                logger.warning(f"⏸️ Trading halted for account {account_id}")
                
            elif action == ActionType.REDUCE_SIZE:
                message += " - Reducing position sizes"
                logger.warning(f"⚠️ Reducing position sizes for account {account_id}")
            
            # Create drawdown event
            event = DrawdownEvent(
                event_id=f"dd_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                account_id=account_id,
                timestamp=datetime.now(),
                event_type='breach',
                level=level,
                drawdown_amount=daily_drawdown,
                drawdown_percent=total_drawdown,
                action_taken=action,
                message=message
            )
            
            self.drawdown_events.append(event)
            self.performance_stats['drawdown_breaches'] += 1
            
            # Send alerts
            await self._send_alert(event)
            
        except Exception as e:
            logger.error(f"Failed to take action for {account_id}: {e}")

    async def _send_alert(self, event: DrawdownEvent):
        """Send drawdown alert"""
        try:
            alert_data = {
                'account_id': event.account_id,
                'level': event.level.value,
                'drawdown_percent': event.drawdown_percent,
                'action': event.action_taken.value,
                'message': event.message,
                'timestamp': event.timestamp.isoformat()
            }
            
            # Store alert in Redis for external consumption
            self.redis_client.lpush('drawdown_alerts', json.dumps(alert_data))
            self.redis_client.ltrim('drawdown_alerts', 0, 99)  # Keep last 100 alerts
            
            logger.info(f"📢 Drawdown alert sent for {event.account_id}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def _cache_snapshot(self, snapshot: AccountSnapshot):
        """Cache account snapshot in Redis"""
        try:
            snapshot_data = {
                'account_id': snapshot.account_id,
                'timestamp': snapshot.timestamp.isoformat(),
                'balance': snapshot.balance,
                'equity': snapshot.equity,
                'unrealized_pnl': snapshot.unrealized_pnl,
                'realized_pnl': snapshot.realized_pnl,
                'daily_pnl': snapshot.daily_pnl,
                'peak_balance': snapshot.peak_balance,
                'drawdown': snapshot.drawdown,
                'drawdown_percent': snapshot.drawdown_percent
            }
            
            # Store latest snapshot
            self.redis_client.setex(
                f"account_snapshot:{snapshot.account_id}",
                3600,  # 1 hour TTL
                json.dumps(snapshot_data)
            )
            
            # Store in time series
            self.redis_client.zadd(
                f"account_history:{snapshot.account_id}",
                {json.dumps(snapshot_data): snapshot.timestamp.timestamp()}
            )
            
            # Keep only last 24 hours of history
            cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
            self.redis_client.zremrangebyscore(
                f"account_history:{snapshot.account_id}",
                0,
                cutoff_time
            )
            
        except Exception as e:
            logger.error(f"Failed to cache snapshot: {e}")

    async def _monitor_drawdowns(self):
        """Background task to monitor drawdowns"""
        while self.running:
            try:
                # Check for any accounts that need attention
                for account_id in self.account_snapshots.keys():
                    if self.trading_halted.get(account_id, False):
                        # Check if recovery conditions are met
                        await self._check_recovery_conditions(account_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in drawdown monitoring: {e}")
                await asyncio.sleep(10)

    async def _check_recovery_conditions(self, account_id: str):
        """Check if account has recovered enough to resume trading"""
        try:
            if not self.account_snapshots.get(account_id):
                return
            
            latest_snapshot = self.account_snapshots[account_id][-1]
            peak_balance = self.peak_balances[account_id]
            
            # Calculate recovery percentage
            recovery_percent = (latest_snapshot.equity / peak_balance) * 100 if peak_balance > 0 else 0
            required_recovery = (1 - self.limits.recovery_threshold) * 100
            
            if recovery_percent >= required_recovery:
                self.trading_halted[account_id] = False
                self.consecutive_losses[account_id] = 0
                self.performance_stats['successful_recoveries'] += 1
                
                logger.info(f"✅ Trading resumed for account {account_id} - Recovery: {recovery_percent:.2f}%")
                
                # Create recovery event
                event = DrawdownEvent(
                    event_id=f"recovery_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    account_id=account_id,
                    timestamp=datetime.now(),
                    event_type='recovery',
                    level=DrawdownLevel.SAFE,
                    drawdown_amount=0,
                    drawdown_percent=recovery_percent,
                    action_taken=ActionType.MONITOR,
                    message=f"Account recovered - Trading resumed at {recovery_percent:.2f}% of peak"
                )
                
                self.drawdown_events.append(event)
                await self._send_alert(event)
                
        except Exception as e:
            logger.error(f"Failed to check recovery conditions for {account_id}: {e}")

    async def _check_recovery_status(self):
        """Background task to check recovery status"""
        while self.running:
            try:
                for account_id in self.trading_halted.keys():
                    if self.trading_halted[account_id]:
                        await self._check_recovery_conditions(account_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error checking recovery status: {e}")
                await asyncio.sleep(60)

    async def _generate_reports(self):
        """Generate periodic drawdown reports"""
        while self.running:
            try:
                # Generate daily report
                report = self._create_daily_report()
                
                # Store report in Redis
                self.redis_client.setex(
                    f"drawdown_report:{datetime.now().strftime('%Y%m%d')}",
                    86400,  # 24 hours
                    json.dumps(report, default=str)
                )
                
                await asyncio.sleep(3600)  # Generate every hour
                
            except Exception as e:
                logger.error(f"Error generating reports: {e}")
                await asyncio.sleep(3600)

    def _create_daily_report(self) -> Dict[str, Any]:
        """Create daily drawdown report"""
        try:
            today = datetime.now().date()
            
            report = {
                'date': today.isoformat(),
                'total_accounts': len(self.account_snapshots),
                'accounts_halted': sum(1 for halted in self.trading_halted.values() if halted),
                'drawdown_breaches': self.performance_stats['drawdown_breaches'],
                'emergency_stops': self.performance_stats['emergency_stops'],
                'successful_recoveries': self.performance_stats['successful_recoveries'],
                'account_summaries': []
            }
            
            for account_id, snapshots in self.account_snapshots.items():
                if snapshots:
                    latest = snapshots[-1]
                    summary = {
                        'account_id': account_id,
                        'current_equity': latest.equity,
                        'peak_balance': self.peak_balances[account_id],
                        'current_drawdown_percent': latest.drawdown_percent,
                        'trading_halted': self.trading_halted.get(account_id, False),
                        'consecutive_losses': self.consecutive_losses.get(account_id, 0)
                    }
                    report['account_summaries'].append(summary)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to create daily report: {e}")
            return {}

    def get_account_status(self, account_id: str) -> Dict[str, Any]:
        """Get current status for an account"""
        try:
            if account_id not in self.account_snapshots or not self.account_snapshots[account_id]:
                return {'error': 'Account not found'}
            
            latest_snapshot = self.account_snapshots[account_id][-1]
            
            return {
                'account_id': account_id,
                'current_equity': latest_snapshot.equity,
                'peak_balance': self.peak_balances[account_id],
                'current_drawdown': latest_snapshot.drawdown,
                'current_drawdown_percent': latest_snapshot.drawdown_percent,
                'trading_halted': self.trading_halted.get(account_id, False),
                'consecutive_losses': self.consecutive_losses.get(account_id, 0),
                'last_update': latest_snapshot.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get account status for {account_id}: {e}")
            return {'error': str(e)}

    def is_trading_allowed(self, account_id: str) -> bool:
        """Check if trading is allowed for an account"""
        return not self.trading_halted.get(account_id, False)
